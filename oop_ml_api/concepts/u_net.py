"""U-Net, a network that answers for every pixel.

Every endpoint here is registered under ``/concepts/u-net``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about shrinking a picture to work out what is in it, growing it
back to say where, and the connections that carry the fine detail across the
middle. The task is segmentation of the shared picture collection: every
picture carries its mask, and the network answers, for each of the 256 pixels,
whether it belongs to the shape.

Why the U is walked by hand
---------------------------
A :class:`~oop_ml.LayerStack` is a chain: layer ``k`` reads exactly what layer
``k - 1`` answered. A skip connection in a U-Net is an edge that bypasses part
of that chain, since the decoder layer at one depth reads the upsampled maps
from below *and* the encoder's maps from the same depth, laid side by side
along the channel axis. So the forward pass here calls each layer's
``respond_to`` in turn, keeps the encoder's answers, and concatenates. The
backward pass mirrors it: each layer's ``correction_for`` does its own
arithmetic, and where two blocks were concatenated the arriving blame is split
by channel, one part going down through the upsampling and the other added to
what reaches the encoder layer from the pooling above it. That addition is the
whole of the skip's backward pass, since the encoder's answer was used twice
and a value used twice is owed the slope from both uses. The test module
checks the walk against a central finite difference, and also checks that a
walk which forgets the addition is caught.

The arrangement
---------------
::

    (1, 16, 16)  convolve   (4, 16, 16)  kept for the skip at full size
                 pool       (4, 8, 8)
                 convolve   (8, 8, 8)    kept for the skip at half size
                 pool       (8, 4, 4)
                 convolve   (8, 4, 4)    the middle
                 repeat     (8, 8, 8)    nearest upsampling
                 join       (16, 8, 8)   the half-size skip laid alongside
                 convolve   (8, 8, 8)
                 repeat     (8, 16, 16)
                 join       (12, 16, 16) the full-size skip laid alongside
                 convolve   (4, 16, 16)
                 convolve   (1, 16, 16)  one by one, a score per pixel
                 flatten    (256,)       what the loss reads

Without the skips the two joins are absent and the two decoder convolutions
read 8 channels rather than 16 and 12, so the network without them has fewer
parameters, 1801 against 2521, and the page says so. Every convolution is three
by three with one pixel of padding except the last, which is one by one.

The loss is binary cross-entropy over the 256 flattened scores, which in this
library sums over every output and divides by the number of pictures. So its
slope on a shared kernel weight is a sum over 256 positions per picture, and
the step has to be correspondingly small. Measured before this module was
written, over weight seeds 0, 1 and 2 and 20 epochs: at a step of 0.01 both
arrangements fell into answering background everywhere from the first epoch,
at 0.005 the arrangement without skips did the same at seed 1, and at 0.003
nothing did. So the step is 0.003 for both.

What it found
-------------
With the skips the held-out mean intersection over union is about 0.99 at
every seed; without them it moved from about 0.16 to 0.62 across three seeds.
And a brightness threshold chosen separately for each picture, Otsu's, scores
better than either, because on these pictures a shape's pixels are brighter
than its ground by construction. The page reports that rather than hiding it.

Compute
-------
Each report trains one network, about three to five seconds from cold, and is
cached. Six fixed reports exist, two arrangements at three seeds, and the
three other fixed endpoints share those caches and the shared reference
classifier's; all of them are warmed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Query
from oop_ml import (
    BinaryCrossEntropy,
    Conv2d,
    Flatten,
    Identity,
    MaxPool2d,
    NearestUpsample2d,
    PassPurpose,
    RectifiedLinear,
)
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from pydantic import BaseModel
from scipy.ndimage import binary_fill_holes

from oop_ml_api.picture_collection import (
    DEFAULT_SEED,
    SIDE,
    PictureCollection,
    as_rows,
    drawn_collection,
)
from oop_ml_api.picture_network import reference_network, reference_split, scores_of

router = APIRouter(prefix="/concepts/u-net", tags=["u-net"])

SEEDS = (0, 1, 2)
"""The weight seeds each arrangement is trained under."""

EPOCHS = 20
"""Passes over the training half.

Twenty rather than the classifier's forty, because a picture here carries 256
targets rather than one, and measured, the arrangement with skips has its
held-out overlap within a hundredth of its final value by epoch 10.
"""

RATE = 0.003
"""The step size, chosen as the module docstring records."""

BATCH = 16
"""Pictures per step, the classifier's batch."""

FIRST, SECOND, MIDDLE = 4, 8, 8
"""Filters in the first and second encoder convolutions and in the middle one.

The first two are the shared classifier's, so the way down is the same shape as
the network on the convolutional networks page.
"""

UNSEEN_COUNT = 60
"""How many rings are drawn, as the convolutional networks page draws them."""

BLANK_BRIGHTNESS = 0.15
"""The brightness of the picture with nothing on it."""

GRADIENT_CHECK_STEP = 1e-6
"""The nudge in a central finite difference."""

CHECKED_PER_LAYER = 12
"""How many weights of each convolution the finite-difference check nudges."""

CUT_GRID = np.round(np.arange(0.20, 0.605, 0.01), 2)
"""The fixed brightness cuts tried on the training half."""

OTHER_SIDES = (8, 12, 16, 18, 20, 24, 30, 32)
"""Picture sides the joins are worked out for, the trained one among them."""

KIND_WORDS: dict[str, str] = {
    "cross": "a cross",
    "square": "a square outline",
    "disc": "a filled disc",
    "bar": "a diagonal bar",
    "ring": "a ring",
}
"""Each kind as the page names it."""

ORDER = (
    "down_first",
    "pool_first",
    "down_second",
    "pool_second",
    "middle",
    "up_second",
    "join_second",
    "up_first",
    "join_first",
    "answer",
    "flatten",
)
"""The layers in the order the forward pass visits them."""

LAYER_WORDS: dict[str, tuple[str, str]] = {
    "down_first": ("the way down", "convolution, 4 filters of 3 by 3"),
    "pool_first": ("the way down", "max pooling, 2 by 2"),
    "down_second": ("the way down", "convolution, 8 filters of 3 by 3"),
    "pool_second": ("the way down", "max pooling, 2 by 2"),
    "middle": ("the middle", "convolution, 8 filters of 3 by 3"),
    "up_second": ("the way up", "repeat each value over 2 by 2"),
    "join_second": ("the way up", "convolution, 8 filters of 3 by 3"),
    "up_first": ("the way up", "repeat each value over 2 by 2"),
    "join_first": ("the way up", "convolution, 4 filters of 3 by 3"),
    "answer": ("the answer", "convolution, 1 filter of 1 by 1"),
    "flatten": ("the answer", "flattening"),
}
"""What each layer is called on the page, and which part of the U it is in."""


# ---------------------------------------------------------------------------
# The network, and the walk through it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmallU:
    """The layers of one U, by name, and whether the skips are joined.

    A mapping rather than a stack, since a stack would refuse the joins: the
    decoder convolution at each depth reads more channels than the layer
    beneath it in the chain answers with.
    """

    layers: dict[str, Layer]
    skip: bool

    def stepped_by(
        self, gradients: dict[str, LayerGradient | None], learning_rate: float
    ) -> SmallU:
        """A new U with every layer moved against its own gradient."""
        return SmallU(
            layers={
                name: layer.stepped_by(gradients[name], learning_rate)
                for name, layer in self.layers.items()
            },
            skip=self.skip,
        )

    @property
    def n_parameters(self) -> int:
        return sum(_parameters_of(layer) for layer in self.layers.values())


def _parameters_of(layer: Layer) -> int:
    if isinstance(layer, Conv2d):
        return int(layer.kernels.size + layer.bias_vector.size)
    return 0


def _convolution(
    reads: tuple[int, int, int],
    n_filters: int,
    kernel_size: int,
    activation,
    random_seed: int,
) -> Conv2d:
    return Conv2d(
        reads=reads,
        n_filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=kernel_size // 2,
        random_seed=random_seed,
    )


def built_u(skip: bool, seed: int, side: int = SIDE) -> SmallU:
    """The arrangement in the module docstring, starting weights from ``seed``.

    Each convolution draws its kernels from its own seed, ``10 * seed`` plus
    its position among the six, so the encoder of the two arrangements starts
    from the same weights at the same seed and only the decoder differs.
    """
    half, quarter = side // 2, side // 4
    bend = RectifiedLinear()
    base = 10 * seed
    return SmallU(
        layers={
            "down_first": _convolution((1, side, side), FIRST, 3, bend, base),
            "pool_first": MaxPool2d(reads=(FIRST, side, side)),
            "down_second": _convolution((FIRST, half, half), SECOND, 3, bend, base + 1),
            "pool_second": MaxPool2d(reads=(SECOND, half, half)),
            "middle": _convolution(
                (SECOND, quarter, quarter), MIDDLE, 3, bend, base + 2
            ),
            "up_second": NearestUpsample2d(reads=(MIDDLE, quarter, quarter)),
            "join_second": _convolution(
                (MIDDLE + (SECOND if skip else 0), half, half),
                SECOND,
                3,
                bend,
                base + 3,
            ),
            "up_first": NearestUpsample2d(reads=(SECOND, half, half)),
            "join_first": _convolution(
                (SECOND + (FIRST if skip else 0), side, side), FIRST, 3, bend, base + 4
            ),
            "answer": _convolution((FIRST, side, side), 1, 1, Identity(), base + 5),
            "flatten": Flatten(reads=(1, side, side)),
        },
        skip=skip,
    )


def _resized(network: SmallU, side: int) -> SmallU:
    """The same trained weights, built to read pictures of another side.

    Every layer here is defined window by window, so nothing learned depends on
    the picture's size; only the extents each layer checks do.
    """
    fresh = built_u(network.skip, 0, side)
    layers: dict[str, Layer] = {}
    for name, layer in fresh.layers.items():
        trained = network.layers[name]
        if isinstance(layer, Conv2d) and isinstance(trained, Conv2d):
            layers[name] = layer.with_parameters(trained.kernels, trained.bias_vector)
        else:
            layers[name] = layer
    return SmallU(layers=layers, skip=network.skip)


def forward(
    network: SmallU,
    pictures: np.ndarray,
    purpose: PassPurpose = PassPurpose.PREDICTING,
) -> dict[str, LayerResponse]:
    """Every layer's response, walking the U by hand and joining the skips.

    The two encoder convolutions' answers are read twice: once by the pooling
    above them, and once more, laid alongside the upsampled maps, by the
    decoder convolution at the same depth.
    """
    layers = network.layers
    responses: dict[str, LayerResponse] = {}

    def through(name: str, block: np.ndarray) -> np.ndarray:
        responses[name] = layers[name].respond_to(block, purpose)
        return np.asarray(responses[name].outputs)

    first = through("down_first", pictures)
    second = through("down_second", through("pool_first", first))
    block = through("up_second", through("middle", through("pool_second", second)))
    if network.skip:
        block = np.concatenate([block, second], axis=1)
    block = through("up_first", through("join_second", block))
    if network.skip:
        block = np.concatenate([block, first], axis=1)
    through("flatten", through("answer", through("join_first", block)))
    return responses


@dataclass(frozen=True)
class BackwardWalk:
    """What one walk back through the U worked out.

    ``loss`` is binary cross-entropy summed over every pixel and divided by
    the pictures. ``gradients`` has one entry per layer. ``to_pictures`` is the
    slope of the loss at every input pixel. ``skip_blame`` and ``deep_blame``
    are, at each encoder convolution, the block of blame that crossed the skip
    and the block that came back down through the pooling above it; their sum
    is what that convolution's own correction starts from.
    """

    loss: float
    gradients: dict[str, LayerGradient | None]
    to_pictures: np.ndarray
    skip_blame: dict[str, np.ndarray]
    deep_blame: dict[str, np.ndarray]


def backward(
    network: SmallU,
    pictures: np.ndarray,
    targets: np.ndarray,
    add_skip_share: bool = True,
) -> BackwardWalk:
    """Forward, score, and walk the blame back down the U.

    Where two blocks were concatenated, the arriving blame is split by channel
    in the order they were laid side by side: the upsampled maps first, the
    encoder's maps after them. The first part continues down through the
    upsampling; the second is held and added to what reaches the encoder
    convolution through the pooling above it.

    ``add_skip_share`` exists only so the check can show that leaving the
    addition out is caught. Every caller that trains leaves it on.
    """
    layers = network.layers
    responses = forward(network, pictures, PassPurpose.TRAINING)
    measurement = BinaryCrossEntropy().measure(
        np.asarray(responses["flatten"].outputs), targets
    )
    gradients: dict[str, LayerGradient | None] = {}
    skip_blame: dict[str, np.ndarray] = {}
    deep_blame: dict[str, np.ndarray] = {}

    def back(name: str, arriving: np.ndarray) -> np.ndarray:
        correction = layers[name].correction_for(responses[name], arriving)
        gradients[name] = correction.gradient
        return np.asarray(correction.passed_down)

    arriving = back("join_first", back("answer", back("flatten", measurement.gradient)))
    if network.skip:
        skip_blame["down_first"] = arriving[:, SECOND:]
        arriving = arriving[:, :SECOND]
    arriving = back("join_second", back("up_first", arriving))
    if network.skip:
        skip_blame["down_second"] = arriving[:, MIDDLE:]
        arriving = arriving[:, :MIDDLE]
    arriving = back("pool_second", back("middle", back("up_second", arriving)))
    deep_blame["down_second"] = arriving
    if network.skip and add_skip_share:
        arriving = arriving + skip_blame["down_second"]
    arriving = back("pool_first", back("down_second", arriving))
    deep_blame["down_first"] = arriving
    if network.skip and add_skip_share:
        arriving = arriving + skip_blame["down_first"]
    to_pictures = back("down_first", arriving)
    return BackwardWalk(
        loss=measurement.value,
        gradients=gradients,
        to_pictures=to_pictures,
        skip_blame=skip_blame,
        deep_blame=deep_blame,
    )


def _flat_masks(collection: PictureCollection) -> np.ndarray:
    return collection.masks.reshape(len(collection), -1).astype(np.float64)


def scores_for(network: SmallU, pictures: np.ndarray) -> np.ndarray:
    """``(n, side, side)``, the raw score at every pixel before the squash."""
    side = pictures.shape[-1]
    outputs = np.asarray(forward(network, pictures)["flatten"].outputs)
    return outputs.reshape(len(pictures), side, side)


def _probabilities(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


# ---------------------------------------------------------------------------
# Scoring an answer for every pixel
# ---------------------------------------------------------------------------


def boundary_of(masks: np.ndarray) -> np.ndarray:
    """Pixels with a neighbour above, below, left or right of the other label.

    Both sides of the edge are counted, the shape's outermost pixels and the
    ground's innermost. The frame's own edge is not a boundary: a pixel on it
    is compared only with the neighbours that exist.
    """
    padded = np.pad(masks, ((0, 0), (1, 1), (1, 1)), mode="edge")
    centre = padded[:, 1:-1, 1:-1]
    different = np.zeros_like(masks, dtype=bool)
    for down, right in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = padded[
            :,
            1 + down : 1 + down + masks.shape[1],
            1 + right : 1 + right + masks.shape[2],
        ]
        different |= neighbour != centre
    return different


class Scores(BaseModel):
    """How well one set of per-pixel answers matches the masks.

    ``pixel_accuracy`` is the share of every pixel of every picture answered
    correctly. ``mean_overlap`` is the intersection over union of the called
    shape and the true one, worked out for each picture and averaged, so a
    small shape counts as much as a large one. ``boundary_accuracy`` is the
    share right among pixels on either side of a shape's edge, and
    ``interior_accuracy`` among the rest. ``missed`` counts shape pixels
    answered as ground and ``invented`` ground pixels answered as shape.
    """

    n_pictures: int
    pixel_accuracy: float
    mean_overlap: float
    boundary_accuracy: float
    interior_accuracy: float
    n_boundary: int
    n_interior: int
    boundary_wrong: int
    interior_wrong: int
    missed: int
    invented: int


def scored(called: np.ndarray, truth: np.ndarray) -> Scores:
    """Every figure in :class:`Scores`, for a block of called and true masks."""
    right = called == truth
    boundary = boundary_of(truth)
    intersection = (called & truth).sum(axis=(1, 2))
    union = (called | truth).sum(axis=(1, 2))
    return Scores(
        n_pictures=int(truth.shape[0]),
        pixel_accuracy=float(right.mean()),
        mean_overlap=float(np.mean(intersection / union)),
        boundary_accuracy=float(right[boundary].mean()),
        interior_accuracy=float(right[~boundary].mean()),
        n_boundary=int(boundary.sum()),
        n_interior=int((~boundary).sum()),
        boundary_wrong=int((~right & boundary).sum()),
        interior_wrong=int((~right & ~boundary).sum()),
        missed=int((truth & ~called).sum()),
        invented=int((called & ~truth).sum()),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EpochReading:
    epoch: int
    loss_per_pixel: float
    held_out_pixel_accuracy: float
    held_out_overlap: float


@dataclass(frozen=True)
class TrainedU:
    network: SmallU
    start: SmallU
    history: tuple[_EpochReading, ...]
    seconds: float


@lru_cache(maxsize=8)
def trained_u(skip: bool, seed: int) -> TrainedU:
    """One arrangement trained from one seed, the classifier's loop otherwise.

    The same training half, the same shuffling seed, shuffled batches of 16.
    The held-out reading after each epoch only reads, so it cannot change the
    fit.
    """
    training, held_out = reference_split()
    start = built_u(skip, seed)
    network = start
    targets = _flat_masks(training)
    generator = np.random.default_rng(0)
    history: list[_EpochReading] = []
    began = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = generator.permutation(len(training))
        total = 0.0
        for first in range(0, len(order), BATCH):
            chosen = order[first : first + BATCH]
            walk = backward(network, training.pictures[chosen], targets[chosen])
            total += walk.loss * len(chosen)
            network = network.stepped_by(walk.gradients, RATE)
        called = scores_for(network, held_out.pictures) > 0
        reading = scored(called, held_out.masks)
        history.append(
            _EpochReading(
                epoch=epoch,
                loss_per_pixel=total / len(order) / (SIDE * SIDE),
                held_out_pixel_accuracy=reading.pixel_accuracy,
                held_out_overlap=reading.mean_overlap,
            )
        )
    return TrainedU(
        network=network,
        start=start,
        history=tuple(history),
        seconds=time.perf_counter() - began,
    )


# ---------------------------------------------------------------------------
# The report on one arrangement
# ---------------------------------------------------------------------------


class LayerReading(BaseModel):
    """One layer of the U, counted.

    ``reads`` includes any channels the skip lays alongside, which is why the
    decoder convolutions read more than the layer before them answers.
    """

    part: str
    label: str
    reads: list[int]
    answers: list[int]
    parameters: int
    reads_skip: bool


class EpochReading(BaseModel):
    """One epoch: the mean loss per pixel over it, and the held-out scores after."""

    epoch: int
    loss_per_pixel: float
    held_out_pixel_accuracy: float
    held_out_overlap: float


class KindScore(BaseModel):
    """The held-out scores restricted to one kind of picture."""

    kind: str
    kind_words: str
    mean_overlap: float
    pixel_accuracy: float
    boundary_accuracy: float


class PictureAnswer(BaseModel):
    """One picture, its mask, and what the network answered at every pixel.

    ``truth`` and ``called`` are 0 or 1 per pixel, ``probability`` the squashed
    score. ``boundary`` marks the pixels on either side of the true edge.
    """

    kind: str
    kind_words: str
    picture: list[list[float]]
    truth: list[list[int]]
    boundary: list[list[int]]
    probability: list[list[float]]
    called: list[list[int]]
    pixels_right: int
    overlap: float
    boundary_right: int
    n_boundary: int


class RingErrors(BaseModel):
    """Where the network went wrong on the rings, sorted by where the pixel is.

    ``hole`` pixels are ground enclosed by the ring, ``outside`` pixels are
    ground not enclosed, and ``ring`` pixels are the ring itself.
    """

    n_hole: int
    hole_called_shape: int
    n_outside: int
    outside_called_shape: int
    n_ring: int
    ring_missed: int


class WorkedPixel(BaseModel):
    """One pixel of the held-out cross, taken from score to loss by hand.

    ``score`` is the one-by-one convolution's answer there, ``probability`` its
    squash, and ``loss_term`` that pixel's share of the loss, the negative
    logarithm of the probability given to the truth.
    """

    row: int
    column: int
    truth: int
    on_boundary: bool
    score: float
    probability: float
    loss_term: float


class WorkedOverlap(BaseModel):
    """The held-out cross's overlap, counted."""

    truth_pixels: int
    called_pixels: int
    intersection: int
    union: int
    overlap: float
    pixels_right: int
    n_boundary: int
    boundary_right: int


class UReport(BaseModel):
    """One arrangement, trained from one seed, and everything measured on it.

    ``seconds`` is how long the training took on whichever machine answered,
    which moves between runs; nothing else here does.
    """

    skip: bool
    seed: int
    label: str
    layers: list[LayerReading]
    n_parameters: int
    parameters_by_part: dict[str, int]
    n_training: int
    n_held_out: int
    epochs: int
    step_size: float
    history: list[EpochReading]
    held_out: Scores
    by_kind: list[KindScore]
    rings: Scores
    ring_errors: RingErrors
    examples: list[PictureAnswer]
    ring_examples: list[PictureAnswer]
    worked_pixel: WorkedPixel
    worked_overlap: WorkedOverlap
    seconds: float


def _layer_readings(network: SmallU) -> list[LayerReading]:
    readings = []
    for name in ORDER:
        layer = network.layers[name]
        part, label = LAYER_WORDS[name]
        readings.append(
            LayerReading(
                part=part,
                label=label,
                reads=list(layer.shape.reads),
                answers=list(layer.shape.answers),
                parameters=_parameters_of(layer),
                reads_skip=network.skip and name in ("join_second", "join_first"),
            )
        )
    return readings


def _grid(values: np.ndarray, places: int = 3) -> list[list[float]]:
    return [[round(float(value), places) for value in row] for row in values]


def _bits(values: np.ndarray) -> list[list[int]]:
    return [[int(value) for value in row] for row in values]


def _answers(
    network: SmallU, collection: PictureCollection, positions: list[int]
) -> list[PictureAnswer]:
    chosen = collection.subset(positions)
    probability = _probabilities(scores_for(network, chosen.pictures))
    called = probability > 0.5
    boundary = boundary_of(chosen.masks)
    answers = []
    for row, position in enumerate(positions):
        truth = chosen.masks[row]
        right = called[row] == truth
        kind = collection.kind_of(position)
        answers.append(
            PictureAnswer(
                kind=kind,
                kind_words=KIND_WORDS[kind],
                picture=as_rows(chosen.pictures[row]),
                truth=_bits(truth),
                boundary=_bits(boundary[row]),
                probability=_grid(probability[row]),
                called=_bits(called[row]),
                pixels_right=int(right.sum()),
                overlap=float(
                    (called[row] & truth).sum() / (called[row] | truth).sum()
                ),
                boundary_right=int(right[boundary[row]].sum()),
                n_boundary=int(boundary[row].sum()),
            )
        )
    return answers


def _first_of_each_kind(collection: PictureCollection) -> list[int]:
    return [
        int(np.flatnonzero(collection.labels == label)[0])
        for label in range(collection.n_kinds)
    ]


@lru_cache(maxsize=1)
def _rings() -> PictureCollection:
    return drawn_collection(UNSEEN_COUNT, seed=DEFAULT_SEED, kinds=("ring",))


def _ring_errors(called: np.ndarray, rings: PictureCollection) -> RingErrors:
    hole = np.stack([binary_fill_holes(mask) for mask in rings.masks]) & ~rings.masks
    outside = ~rings.masks & ~hole
    return RingErrors(
        n_hole=int(hole.sum()),
        hole_called_shape=int((called & hole).sum()),
        n_outside=int(outside.sum()),
        outside_called_shape=int((called & outside).sum()),
        n_ring=int(rings.masks.sum()),
        ring_missed=int((~called & rings.masks).sum()),
    )


def _worked(
    network: SmallU, held_out: PictureCollection
) -> tuple[WorkedPixel, WorkedOverlap]:
    """The cross, counted, and one of its boundary pixels taken by hand.

    The pixel is the first boundary pixel of the shape itself, reading row by
    row, so it is on the edge the skip connections are claimed to help with.
    """
    position = _first_of_each_kind(held_out)[0]
    picture = held_out.pictures[position : position + 1]
    truth = held_out.masks[position]
    score = scores_for(network, picture)[0]
    probability = _probabilities(score)
    called = probability > 0.5
    boundary = boundary_of(truth[np.newaxis])[0]
    rows, columns = np.nonzero(boundary & truth)
    row, column = int(rows[0]), int(columns[0])
    given = (
        probability[row, column] if truth[row, column] else 1 - probability[row, column]
    )
    right = called == truth
    pixel = WorkedPixel(
        row=row,
        column=column,
        truth=int(truth[row, column]),
        on_boundary=True,
        score=float(score[row, column]),
        probability=float(probability[row, column]),
        loss_term=float(-np.log(given)),
    )
    overlap = WorkedOverlap(
        truth_pixels=int(truth.sum()),
        called_pixels=int(called.sum()),
        intersection=int((called & truth).sum()),
        union=int((called | truth).sum()),
        overlap=float((called & truth).sum() / (called | truth).sum()),
        pixels_right=int(right.sum()),
        n_boundary=int(boundary.sum()),
        boundary_right=int(right[boundary].sum()),
    )
    return pixel, overlap


@lru_cache(maxsize=8)
def _report(skip: bool, seed: int) -> UReport:
    training, held_out = reference_split()
    trained = trained_u(skip, seed)
    network = trained.network
    layers = _layer_readings(network)
    called = scores_for(network, held_out.pictures) > 0
    by_kind = []
    for label, kind in enumerate(held_out.kind_names):
        chosen = held_out.labels == label
        reading = scored(called[chosen], held_out.masks[chosen])
        by_kind.append(
            KindScore(
                kind=kind,
                kind_words=KIND_WORDS[kind],
                mean_overlap=reading.mean_overlap,
                pixel_accuracy=reading.pixel_accuracy,
                boundary_accuracy=reading.boundary_accuracy,
            )
        )
    rings = _rings()
    ring_called = scores_for(network, rings.pictures) > 0
    parts: dict[str, int] = {}
    for reading in layers:
        parts[reading.part] = parts.get(reading.part, 0) + reading.parameters
    pixel, overlap = _worked(network, held_out)
    return UReport(
        skip=skip,
        seed=seed,
        label="with the skip connections" if skip else "without the skip connections",
        layers=layers,
        n_parameters=network.n_parameters,
        parameters_by_part=parts,
        n_training=len(training),
        n_held_out=len(held_out),
        epochs=EPOCHS,
        step_size=RATE,
        history=[
            EpochReading(
                epoch=reading.epoch,
                loss_per_pixel=reading.loss_per_pixel,
                held_out_pixel_accuracy=reading.held_out_pixel_accuracy,
                held_out_overlap=reading.held_out_overlap,
            )
            for reading in trained.history
        ],
        held_out=scored(called, held_out.masks),
        by_kind=by_kind,
        rings=scored(ring_called, rings.masks),
        ring_errors=_ring_errors(ring_called, rings),
        examples=_answers(network, held_out, _first_of_each_kind(held_out)),
        ring_examples=_answers(network, rings, [0, 1, 2, 3]),
        worked_pixel=pixel,
        worked_overlap=overlap,
        seconds=trained.seconds,
    )


@router.get("/network", response_model=UReport)
def read_network(
    skip: bool = Query(True, description="whether the skip connections are joined"),
    seed: int = Query(0, ge=0, le=2, description="the weight seed, 0, 1 or 2"),
) -> UReport:
    """One arrangement trained from one seed, counted and scored pixel by pixel."""
    return _report(skip, seed)


# ---------------------------------------------------------------------------
# What a classifier and a threshold manage
# ---------------------------------------------------------------------------


class ClassifierShape(BaseModel):
    """One layer of the shared classifier, and whether any arrangement is left."""

    label: str
    answers: list[int]
    spatial: bool


class Baseline(BaseModel):
    """One way of answering every pixel that is not a U-Net, scored."""

    name: str
    description: str
    scores: Scores


class BaselinesResponse(BaseModel):
    """The answers a U-Net has to beat, on the same held-out pictures.

    ``mask_share`` is the share of held-out pixels that belong to a shape, so
    answering ground everywhere gets one minus it right. ``best_cut`` is the
    fixed brightness above which a pixel is called shape, chosen as the one of
    ``cuts_tried`` that gets most training pixels right. ``classifier_accuracy``
    is the shared classifier's share of held-out kinds named correctly.
    """

    mask_share: float
    best_cut: float
    cuts_tried: int
    training_accuracy_at_cut: float
    classifier_accuracy: float
    classifier_shapes: list[ClassifierShape]
    baselines: list[Baseline]
    ring_cut: Scores
    ring_picture_threshold: Scores


def picture_threshold(picture: np.ndarray) -> float:
    """Otsu's threshold for one picture.

    Every cut between two sorted brightnesses is tried, and the one kept makes
    the two groups it separates as far apart in mean as their sizes allow: the
    largest ``n_low * n_high * (mean_low - mean_high) ** 2``, which is the same
    as the smallest spread within the two groups.
    """
    values = np.sort(picture.ravel())
    count = values.size
    below = np.arange(1, count)
    running = np.cumsum(values)[:-1]
    low_mean = running / below
    high_mean = (values.sum() - running) / (count - below)
    between = below * (count - below) * (low_mean - high_mean) ** 2
    best = int(np.argmax(between))
    return float((values[best] + values[best + 1]) / 2)


def _average_mask_answers(training: PictureCollection, kinds: np.ndarray) -> np.ndarray:
    """For each picture, the training masks of the named kind, averaged and cut at a half."""
    average = np.stack(
        [training.masks[training.labels == label].mean(axis=0) for label in range(4)]
    )
    return average[kinds] > 0.5


@lru_cache(maxsize=1)
def _baselines() -> BaselinesResponse:
    training, held_out = reference_split()
    classifier = reference_network().stack
    training_pixels = training.pictures[:, 0]
    held_pixels = held_out.pictures[:, 0]
    accuracies = [
        float(((training_pixels > cut) == training.masks).mean()) for cut in CUT_GRID
    ]
    best = int(np.argmax(accuracies))
    cut = float(CUT_GRID[best])
    otsu = np.stack([picture > picture_threshold(picture) for picture in held_pixels])
    called_kinds = scores_of(classifier, held_out.pictures).argmax(axis=1)
    rings = _rings()
    ring_pixels = rings.pictures[:, 0]
    shapes = []
    for layer in classifier:
        answers = list(layer.shape.answers)
        label = type(layer).__name__
        words = {
            "Conv2d": "convolution",
            "MaxPool2d": "max pooling",
            "Flatten": "flattening",
            "DenseLayer": "dense",
        }[label]
        shapes.append(
            ClassifierShape(label=words, answers=answers, spatial=len(answers) == 3)
        )
    return BaselinesResponse(
        mask_share=float(held_out.masks.mean()),
        best_cut=cut,
        cuts_tried=len(CUT_GRID),
        training_accuracy_at_cut=accuracies[best],
        classifier_accuracy=float((called_kinds == held_out.labels).mean()),
        classifier_shapes=shapes,
        baselines=[
            Baseline(
                name="the kind, from the classifier",
                description=(
                    "the average training mask of the kind the classifier calls "
                    "the picture, cut at a half"
                ),
                scores=scored(
                    _average_mask_answers(training, called_kinds), held_out.masks
                ),
            ),
            Baseline(
                name="the true kind",
                description=(
                    "the average training mask of the picture's own kind, cut at a half"
                ),
                scores=scored(
                    _average_mask_answers(training, held_out.labels), held_out.masks
                ),
            ),
            Baseline(
                name="one cut for every picture",
                description=(
                    f"a pixel brighter than {cut:.2f} is shape, the cut chosen on "
                    "the training half"
                ),
                scores=scored(held_pixels > cut, held_out.masks),
            ),
            Baseline(
                name="a cut for each picture",
                description=(
                    "Otsu's threshold, the cut that best splits that picture's own "
                    "brightnesses into two groups"
                ),
                scores=scored(otsu, held_out.masks),
            ),
        ],
        ring_cut=scored(ring_pixels > cut, rings.masks),
        ring_picture_threshold=scored(
            np.stack([picture > picture_threshold(picture) for picture in ring_pixels]),
            rings.masks,
        ),
    )


@router.get("/baselines", response_model=BaselinesResponse)
def read_baselines() -> BaselinesResponse:
    """A classifier's kind, a fixed cut and a per-picture cut, scored per pixel."""
    return _baselines()


# ---------------------------------------------------------------------------
# The walk checked against finite differences, and the blame that crosses
# ---------------------------------------------------------------------------


class LayerCheck(BaseModel):
    """One convolution's slopes from the walk beside a central finite difference.

    ``largest_slope`` is the largest slope among the entries checked, so the
    discrepancy can be read against it.
    """

    part: str
    label: str
    n_checked: int
    largest_slope: float
    largest_discrepancy: float
    largest_discrepancy_forgetting_skip: float


class BlameShare(BaseModel):
    """At one encoder convolution, how the blame reaching it divides.

    ``skip_share`` is the length of the block that crossed the skip over the
    length of the two blocks' sum, and ``deep_share`` the same for the block
    that came back down through the pooling above. They need not add to one,
    since the two blocks can point partly against each other.
    """

    label: str
    skip_length: float
    deep_length: float
    total_length: float
    skip_share: float
    deep_share: float


class GradientCheckResponse(BaseModel):
    """The hand-walked backward pass, checked, and the blame it routes.

    The check runs on the untrained arrangement with skips at seed 0, on the
    first held-out picture of each kind. ``pictures_*`` is the same comparison
    for twelve input pixels of those pictures. The blame shares are measured
    at the start of training and after it, on the same four pictures.
    """

    n_pictures: int
    step: float
    layers: list[LayerCheck]
    pictures_largest_slope: float
    pictures_largest_discrepancy: float
    blame_at_start: list[BlameShare]
    blame_after_training: list[BlameShare]


def _loss_of(network: SmallU, pictures: np.ndarray, targets: np.ndarray) -> float:
    outputs = np.asarray(forward(network, pictures)["flatten"].outputs)
    return BinaryCrossEntropy().measure(outputs, targets).value


def _with_kernel(network: SmallU, name: str, kernels: np.ndarray) -> SmallU:
    layer = network.layers[name]
    assert isinstance(layer, Conv2d)
    layers = dict(network.layers)
    layers[name] = layer.with_parameters(kernels, layer.bias_vector)
    return SmallU(layers=layers, skip=network.skip)


def _blame(walk: BackwardWalk) -> list[BlameShare]:
    shares = []
    for name, label in (
        ("down_first", "the first convolution, at full size"),
        ("down_second", "the second convolution, at half size"),
    ):
        skip = walk.skip_blame[name]
        deep = walk.deep_blame[name]
        total = float(np.linalg.norm(skip + deep))
        shares.append(
            BlameShare(
                label=label,
                skip_length=float(np.linalg.norm(skip)),
                deep_length=float(np.linalg.norm(deep)),
                total_length=total,
                skip_share=float(np.linalg.norm(skip)) / total,
                deep_share=float(np.linalg.norm(deep)) / total,
            )
        )
    return shares


@lru_cache(maxsize=1)
def _gradient_check() -> GradientCheckResponse:
    _, held_out = reference_split()
    chosen = held_out.subset(_first_of_each_kind(held_out))
    pictures = chosen.pictures
    targets = _flat_masks(chosen)
    network = built_u(True, 0)
    walk = backward(network, pictures, targets)
    forgetting = backward(network, pictures, targets, add_skip_share=False)
    generator = np.random.default_rng(0)
    step = GRADIENT_CHECK_STEP
    checks = []
    for name in ORDER:
        layer = network.layers[name]
        if not isinstance(layer, Conv2d):
            continue
        kernels = np.asarray(layer.kernels)
        gradient = walk.gradients[name]
        forgotten = forgetting.gradients[name]
        assert gradient is not None and forgotten is not None
        slopes = np.asarray(gradient.weights).reshape(kernels.shape)
        slopes_forgetting = np.asarray(forgotten.weights).reshape(kernels.shape)
        positions = generator.choice(
            kernels.size, size=min(CHECKED_PER_LAYER, kernels.size), replace=False
        )
        discrepancies, forgetting_discrepancies, magnitudes = [], [], []
        for flat in positions:
            index = np.unravel_index(int(flat), kernels.shape)
            raised = kernels.copy()
            raised[index] += step
            lowered = kernels.copy()
            lowered[index] -= step
            numeric = (
                _loss_of(_with_kernel(network, name, raised), pictures, targets)
                - _loss_of(_with_kernel(network, name, lowered), pictures, targets)
            ) / (2 * step)
            discrepancies.append(abs(numeric - slopes[index]))
            forgetting_discrepancies.append(abs(numeric - slopes_forgetting[index]))
            magnitudes.append(abs(numeric))
        part, label = LAYER_WORDS[name]
        checks.append(
            LayerCheck(
                part=part,
                label=label,
                n_checked=len(positions),
                largest_slope=float(max(magnitudes)),
                largest_discrepancy=float(max(discrepancies)),
                largest_discrepancy_forgetting_skip=float(
                    max(forgetting_discrepancies)
                ),
            )
        )

    pixel_positions = generator.choice(pictures.size, size=12, replace=False)
    pixel_discrepancies, pixel_magnitudes = [], []
    for flat in pixel_positions:
        index = np.unravel_index(int(flat), pictures.shape)
        raised = pictures.copy()
        raised[index] += step
        lowered = pictures.copy()
        lowered[index] -= step
        numeric = (
            _loss_of(network, raised, targets) - _loss_of(network, lowered, targets)
        ) / (2 * step)
        pixel_discrepancies.append(abs(numeric - walk.to_pictures[index]))
        pixel_magnitudes.append(abs(numeric))

    trained = trained_u(True, 0).network
    return GradientCheckResponse(
        n_pictures=len(chosen),
        step=step,
        layers=checks,
        pictures_largest_slope=float(max(pixel_magnitudes)),
        pictures_largest_discrepancy=float(max(pixel_discrepancies)),
        blame_at_start=_blame(walk),
        blame_after_training=_blame(backward(trained, pictures, targets)),
    )


@router.get("/gradient-check", response_model=GradientCheckResponse)
def read_gradient_check() -> GradientCheckResponse:
    """The walk through the joins against a finite difference, and the skip's share."""
    return _gradient_check()


# ---------------------------------------------------------------------------
# Other sizes, and a picture with nothing on it
# ---------------------------------------------------------------------------


class SideReading(BaseModel):
    """What two poolings and two repetitions make of a picture of one side.

    Each join lays the repeated maps beside the encoder's maps of the same
    depth, which is defined only when the two agree in extent.
    ``answer_side`` is what the arrangement without joins answers with, which
    covers the picture only when it equals ``side``.
    """

    side: int
    after_first_pooling: int
    after_second_pooling: int
    repeated_once: int
    repeated_twice: int
    half_size_join_defined: bool
    full_size_join_defined: bool
    answer_side: int


class LargerResponse(BaseModel):
    """The trained weights at another size, and on a picture with no shape.

    The four held-out pictures of the report's examples are laid out two by
    two as one picture of 32 by 32, and the network with skips at seed 0 is
    rebuilt at that size with the same weights. ``changed_pixels`` counts
    pixels whose call differs from calling the four pictures separately, and
    ``farthest_from_seam`` is how many pixels the farthest of those lies from
    the nearest seam between two of the four.
    """

    side: int
    picture: list[list[float]]
    truth: list[list[int]]
    probability: list[list[float]]
    scores: Scores
    separately: Scores
    changed_pixels: int
    farthest_from_seam: int
    blank_brightness: float
    blank_called_with_skip: int
    blank_largest_probability: float
    sides: list[SideReading]


@lru_cache(maxsize=1)
def _larger() -> LargerResponse:
    _, held_out = reference_split()
    network = trained_u(True, 0).network
    chosen = held_out.subset(_first_of_each_kind(held_out))
    tiles = chosen.pictures[:, 0]
    masks = chosen.masks
    big_side = 2 * SIDE
    picture = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
    truth = np.block([[masks[0], masks[1]], [masks[2], masks[3]]])
    larger = _resized(network, big_side)
    probability = _probabilities(scores_for(larger, picture[np.newaxis, np.newaxis])[0])
    called = probability > 0.5
    separate = scores_for(network, chosen.pictures) > 0
    separate_tiled = np.block([[separate[0], separate[1]], [separate[2], separate[3]]])
    changed = called != separate_tiled
    rows, columns = np.nonzero(changed)
    distance_to_seam = np.minimum(
        np.minimum(np.abs(rows - (SIDE - 0.5)), np.abs(columns - (SIDE - 0.5))), SIDE
    )
    blank = np.full((1, 1, SIDE, SIDE), BLANK_BRIGHTNESS)
    blank_probability = _probabilities(scores_for(network, blank)[0])
    sides = []
    for side in OTHER_SIDES:
        first, second = side // 2, side // 4
        sides.append(
            SideReading(
                side=side,
                after_first_pooling=first,
                after_second_pooling=second,
                repeated_once=2 * second,
                repeated_twice=2 * (2 * second),
                half_size_join_defined=2 * second == first,
                full_size_join_defined=2 * first == side,
                answer_side=4 * second,
            )
        )
    return LargerResponse(
        side=big_side,
        picture=_grid(picture),
        truth=_bits(truth),
        probability=_grid(probability),
        scores=scored(called[np.newaxis], truth[np.newaxis]),
        separately=scored(separate, masks),
        changed_pixels=int(changed.sum()),
        farthest_from_seam=int(np.ceil(distance_to_seam.max())) if rows.size else 0,
        blank_brightness=BLANK_BRIGHTNESS,
        blank_called_with_skip=int((blank_probability > 0.5).sum()),
        blank_largest_probability=float(blank_probability.max()),
        sides=sides,
    )


@router.get("/larger", response_model=LargerResponse)
def read_larger() -> LargerResponse:
    """The trained U on a picture twice the size, on a blank one, and at other sides."""
    return _larger()
