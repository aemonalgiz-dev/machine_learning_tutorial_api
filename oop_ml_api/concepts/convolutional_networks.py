"""Convolutional networks, assembled from the layers.

Every endpoint here is registered under ``/concepts/convolutional-networks``,
and the router is created and registered before any endpoint exists, so the
pages of this section can be built in parallel without many edits to one
shared file.

The page is about convolution, pooling and flattening stacked into a model that
reads a picture, and what each part contributes. The network is the shared one
in :mod:`oop_ml_api.picture_network`, trained on the shared picture collection,
so the reader meets the same network here that the pages on vectors for
pictures read their vectors off.

What each part contributes is measured by taking it away
----------------------------------------------------------
Four arrangements are trained under the same loop, the same pictures and the
same three weight seeds:

* **the network**, two rounds of convolve and pool, then a flattening, a dense
  layer of 16 and one score per kind;
* **pooling taken away**, the same two convolutions with no pooling between
  them, which leaves the flattening 2048 wide rather than 128 and so multiplies
  the dense layer's weights by sixteen. No arrangement removes pooling and
  changes nothing else, since the pooling is what decides that width, and the
  page says so;
* **one round**, the first convolve and pool only, so the flattening reads
  4 maps of 8 by 8, which is what the second round contributes;
* **no convolutions**, dense layers on the 256 raw pixels, 256 to 9 to 16 to
  4, which comes to 2541 parameters against the network's 2468 and is the
  comparison of comparable size.

Measured before this module was written, over weight seeds 0, 1 and 2: the
network scores 0.971, 0.921 and 0.967 on the held-out half; without pooling
0.875, 0.821 and 0.817; the dense network 0.679, 0.729 and 0.663, and it does
not even fit its own training half. Shifting a held-out picture by one pixel
changes the network's call on 5 to 12 per cent of pictures, without pooling
17 to 25 per cent, and for the dense network 31 to 38 per cent. So pooling
does reduce the sensitivity to a shift, and it does not remove it, which is
the textbook claim in the form that survived measurement.

Why the loop is repeated here rather than called
-------------------------------------------------
:func:`oop_ml_api.picture_network.trained_network` records the loss and the
training accuracy after every epoch and not the held-out accuracy, which is the
curve the page most needs. So the loop is written out once more here with that
one extra reading, and a test holds its reference fit to the shared one: the
same final loss and the same held-out accuracy, to the last bit, because the
arithmetic is the same arithmetic in the same order. The dense variants need
the loop anyway, since the shared module builds only convolutional designs.

Compute
-------
Each report trains one network, at most about seven seconds from cold for the
widest arrangement, and is cached, so every later visit is free. Twelve fixed
reports exist, four arrangements at three seeds, and all of them are warmed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from oop_ml import (
    Conv2d,
    DenseLayer,
    Flatten,
    HyperbolicTangent,
    Identity,
    LayerStack,
    MaxPool2d,
    Neuron,
    RectifiedLinear,
    SoftmaxCrossEntropy,
)
from oop_ml.core.computer_vision.edges import GradientOperator, weights_of
from pydantic import BaseModel, Field

from oop_ml_api.picture_collection import (
    DEFAULT_SEED,
    KINDS,
    SIDE,
    PictureCollection,
    as_rows,
    drawn_collection,
)
from oop_ml_api.picture_network import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_RATE,
    NetworkDesign,
    accuracy_of,
    built_network,
    reference_network,
    reference_split,
    scores_of,
)

router = APIRouter(
    prefix="/concepts/convolutional-networks", tags=["convolutional-networks"]
)

SEEDS = (0, 1, 2)
"""The weight seeds every arrangement is trained under.

Three, because the picture pages agreed to report a trained number with its
spread over a few starts, and the reference itself moves from 0.921 to 0.971
across these three.
"""

SHIFT_DISTANCES = (1, 2, 3)
"""How many pixels a held-out picture is moved by in the shift test."""

POKE = 0.5
"""How much one pixel is brightened when asking which pixels a unit can see.

Large against the noise on a pixel, 0.05, and a little under the lift that
separates a shape from its ground, so a poke is something the first filters
answer to without being a different picture.
"""

DEAD_BELOW = 0.05
"""A first-layer filter answering above zero on fewer positions than this is
counted as nearly silent. The bend passes no slope back from a position where
it answered zero, so such a filter barely trains."""

BASELINE_DRAWS = 2000
"""How many random three by three grids the edge-likeness baseline draws."""

BLANK_BRIGHTNESS = 0.15
"""The brightness of the picture with nothing drawn on it, the middle of the
range the collection draws its backgrounds from."""

UNSEEN_COUNT = 60
"""How many rings are drawn to ask what the network makes of a kind it never saw."""

OTHER_SIDES = (12, 15, 16, 17, 20, 32)
"""Picture sides the flattening width is worked out for, the trained one among them."""


class Variant(StrEnum):
    """Which arrangement a report is about."""

    REFERENCE = "reference"
    WITHOUT_POOLING = "without_pooling"
    ONE_ROUND = "one_round"
    DENSE = "dense"


VARIANT_LABELS: dict[Variant, str] = {
    Variant.REFERENCE: "the network, two rounds of convolve and pool",
    Variant.WITHOUT_POOLING: "pooling taken away",
    Variant.ONE_ROUND: "one round of convolve and pool",
    Variant.DENSE: "no convolutions, dense layers on the raw pixels",
}
"""What each arrangement is called in words, written once for page and widgets."""


class Depth(StrEnum):
    """A place inside the network where a unit can be asked what it sees."""

    FIRST_CONVOLUTION = "first_convolution"
    FIRST_POOLING = "first_pooling"
    SECOND_CONVOLUTION = "second_convolution"
    SECOND_POOLING = "second_pooling"


DEPTH_POSITIONS: dict[Depth, int] = {
    Depth.FIRST_CONVOLUTION: 0,
    Depth.FIRST_POOLING: 1,
    Depth.SECOND_CONVOLUTION: 2,
    Depth.SECOND_POOLING: 3,
}
"""Which layer of the reference stack each depth is."""

DEPTH_LABELS: dict[Depth, str] = {
    Depth.FIRST_CONVOLUTION: "after the first convolution",
    Depth.FIRST_POOLING: "after the first pooling",
    Depth.SECOND_CONVOLUTION: "after the second convolution",
    Depth.SECOND_POOLING: "after the second pooling",
}


class Kind(StrEnum):
    """One of the four kinds the network names."""

    CROSS = "cross"
    SQUARE = "square"
    DISC = "disc"
    BAR = "bar"


KIND_WORDS: dict[str, str] = {
    "cross": "a cross",
    "square": "a square outline",
    "disc": "a filled disc",
    "bar": "a diagonal bar",
    "ring": "a ring",
}
"""Each kind as the page names it."""

COMPASS = (
    "brighter to the right",
    "brighter to the lower right",
    "brighter below",
    "brighter to the lower left",
    "brighter to the left",
    "brighter to the upper left",
    "brighter above",
    "brighter to the upper right",
)
"""The eight directions a turned Sobel grid answers to, in turning order."""

RING = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
"""The outer cells of a three by three grid, clockwise from the top left.

Moving every outer weight one place along this ring turns a three by three
gradient grid by an eighth of a turn, which is how the eight compass versions
of the hand-designed grid are made from the one the edge page uses.
"""


# ---------------------------------------------------------------------------
# Building and training
# ---------------------------------------------------------------------------


def _dense(
    n_inputs: int, n_outputs: int, activation, generator: np.random.Generator
) -> DenseLayer:
    """A dense layer drawn at the scale the shared network draws its own."""
    spread = (2.0 / n_inputs) ** 0.5
    return DenseLayer(
        [
            Neuron(
                weights=generator.normal(0.0, spread, n_inputs),
                bias=0.0,
                activation=activation,
            )
            for _ in range(n_outputs)
        ]
    )


def _built(variant: Variant, seed: int, n_kinds: int) -> LayerStack:
    """The arrangement ``variant`` names, with starting weights from ``seed``."""
    if variant is Variant.REFERENCE:
        return built_network(NetworkDesign(weight_seed=seed), n_kinds)
    if variant is Variant.WITHOUT_POOLING:
        return built_network(
            NetworkDesign(weight_seed=seed, with_pooling=False), n_kinds
        )

    generator = np.random.default_rng(seed)
    if variant is Variant.ONE_ROUND:
        half = SIDE // 2
        return LayerStack(
            [
                Conv2d(
                    reads=(1, SIDE, SIDE),
                    n_filters=4,
                    kernel_size=3,
                    activation=RectifiedLinear(),
                    padding=1,
                    random_seed=seed,
                ),
                MaxPool2d(reads=(4, SIDE, SIDE)),
                Flatten(reads=(4, half, half)),
                _dense(4 * half * half, 16, HyperbolicTangent(), generator),
                _dense(16, n_kinds, Identity(), generator),
            ]
        )
    return LayerStack(
        [
            Flatten(reads=(1, SIDE, SIDE)),
            _dense(SIDE * SIDE, 9, RectifiedLinear(), generator),
            _dense(9, 16, HyperbolicTangent(), generator),
            _dense(16, n_kinds, Identity(), generator),
        ]
    )


@dataclass(frozen=True)
class _EpochReading:
    epoch: int
    loss: float
    training_accuracy: float
    held_out_accuracy: float


@dataclass(frozen=True)
class _Trained:
    stack: LayerStack
    start: LayerStack
    history: tuple[_EpochReading, ...]
    seconds: float


def _train(variant: Variant, seed: int) -> _Trained:
    """The shared loop, with the held-out accuracy read after every epoch.

    Everything the shared loop does, in the same order: the same shuffling
    seed, the same batches, the same rate and length. The held-out reading
    only reads, so it cannot change the fit.
    """
    training, held_out = reference_split()
    start = _built(variant, seed, training.n_kinds)
    stack = start
    loss = SoftmaxCrossEntropy()
    targets = training.one_hot()
    generator = np.random.default_rng(0)
    history: list[_EpochReading] = []
    began = time.perf_counter()
    for epoch in range(1, DEFAULT_EPOCHS + 1):
        order = generator.permutation(len(training))
        losses = []
        for first in range(0, len(order), DEFAULT_BATCH):
            chosen = order[first : first + DEFAULT_BATCH]
            backward = stack.backward_pass(
                training.pictures[chosen], targets[chosen], loss
            )
            losses.append(backward.loss * len(chosen))
            stack = stack.stepped_by(backward, DEFAULT_RATE)
        history.append(
            _EpochReading(
                epoch=epoch,
                loss=float(sum(losses) / len(order)),
                training_accuracy=accuracy_of(stack, training),
                held_out_accuracy=accuracy_of(stack, held_out),
            )
        )
    return _Trained(
        stack=stack,
        start=start,
        history=tuple(history),
        seconds=time.perf_counter() - began,
    )


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------


def _probabilities(stack: LayerStack, pictures: np.ndarray) -> np.ndarray:
    """The softmax of the scores, one row per picture."""
    scores = scores_of(stack, pictures)
    scores = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _shifted(picture: np.ndarray, down: int, right: int) -> np.ndarray:
    """One picture moved, with the strip it left behind filled from its edge.

    Filling from the edge rather than wrapping round, because a wrapped strip
    would carry the far side's pixels in, and filling with a constant would
    draw a border the collection never has.
    """
    moved = np.roll(picture, (down, right), axis=(0, 1))
    if down > 0:
        moved[:down, :] = moved[down, :]
    elif down < 0:
        moved[down:, :] = moved[down - 1, :]
    if right > 0:
        moved[:, :right] = moved[:, [right]]
    elif right < 0:
        moved[:, right:] = moved[:, [right - 1]]
    return moved


def _stays_inside(mask: np.ndarray, down: int, right: int) -> bool:
    """Whether the shape is still wholly inside the frame after the move."""
    rows, columns = np.nonzero(mask)
    return bool(
        rows.min() + down >= 0
        and rows.max() + down <= SIDE - 1
        and columns.min() + right >= 0
        and columns.max() + right <= SIDE - 1
    )


class ShiftReading(BaseModel):
    """What moving the held-out pictures by one distance did to the answers.

    Every picture whose shape stays inside the frame is moved that far right,
    left, down and up, so ``n_pictures`` counts moved pictures rather than
    distinct ones. ``calls_changed`` is the share whose highest-scoring kind
    changed, ``shifted_accuracy`` the share of moved pictures still named
    correctly, ``accuracy_before`` the same pictures unmoved, and
    ``probability_move`` the mean of half the summed absolute change in the
    four probabilities, which is zero for an answer that did not move and one
    for an answer that moved all its weight to other kinds.
    """

    distance: int
    n_pictures: int
    calls_changed: float
    shifted_accuracy: float
    accuracy_before: float
    probability_move: float


def _shift_readings(
    stack: LayerStack, held_out: PictureCollection
) -> list[ShiftReading]:
    """The shift test at every distance."""
    before = _probabilities(stack, held_out.pictures)
    readings = []
    for distance in SHIFT_DISTANCES:
        changed = 0
        correct_after = 0
        correct_before = 0
        moves: list[float] = []
        total = 0
        for down, right in (
            (0, distance),
            (0, -distance),
            (distance, 0),
            (-distance, 0),
        ):
            chosen = [
                index
                for index in range(len(held_out))
                if _stays_inside(held_out.masks[index], down, right)
            ]
            pictures = np.stack(
                [_shifted(held_out.pictures[index, 0], down, right) for index in chosen]
            )[:, np.newaxis]
            after = _probabilities(stack, pictures)
            earlier = before[chosen]
            labels = held_out.labels[chosen]
            changed += int((after.argmax(axis=1) != earlier.argmax(axis=1)).sum())
            correct_after += int((after.argmax(axis=1) == labels).sum())
            correct_before += int((earlier.argmax(axis=1) == labels).sum())
            moves.extend((0.5 * np.abs(after - earlier).sum(axis=1)).tolist())
            total += len(chosen)
        readings.append(
            ShiftReading(
                distance=distance,
                n_pictures=total,
                calls_changed=changed / total,
                shifted_accuracy=correct_after / total,
                accuracy_before=correct_before / total,
                probability_move=float(np.mean(moves)),
            )
        )
    return readings


class LayerReading(BaseModel):
    """One layer of an arrangement, counted.

    ``field`` is how many input pixels along one side a single unit of this
    layer's answer depends on, worked out from the windows and strides below
    it, and ``step`` is how many input pixels apart two neighbouring units of
    the answer sit. Once a layer reads everything, ``reads_whole_picture`` is
    true and ``field`` is the picture's side. ``multiply_adds`` counts the
    products one picture costs at this layer; pooling compares and flattening
    rearranges, and neither multiplies anything.
    """

    label: str
    reads: list[int]
    answers: list[int]
    parameters: int
    multiply_adds: int
    field: int
    step: int
    reads_whole_picture: bool


def _layer_readings(stack: LayerStack) -> list[LayerReading]:
    """Every layer of a stack with its counts and its receptive field.

    The field grows by the rule every window-and-stride stack obeys: a window
    of ``k`` placed over units ``step`` input pixels apart reaches
    ``(k - 1) * step`` further, and a stride of ``s`` spreads the next layer's
    units ``s`` times further apart.
    """
    field, step = 1, 1
    whole = False
    readings = []
    last = len(stack) - 1
    for position, layer in enumerate(stack):
        reads = list(layer.shape.reads)
        answers = list(layer.shape.answers)
        if isinstance(layer, Conv2d):
            field += (layer.kernel_size - 1) * step
            step *= layer.stride
            parameters = int(layer.kernels.size + layer.bias_vector.size)
            channels = reads[0]
            multiply_adds = int(
                np.prod(answers) * channels * layer.kernel_size * layer.kernel_size
            )
            label = (
                f"convolution, {layer.n_filters} filters of "
                f"{layer.kernel_size} by {layer.kernel_size}"
            )
        elif isinstance(layer, MaxPool2d):
            field += (layer.window - 1) * step
            step *= layer.stride
            parameters = 0
            multiply_adds = 0
            label = f"max pooling, {layer.window} by {layer.window}"
        elif isinstance(layer, Flatten):
            parameters = 0
            multiply_adds = 0
            label = "flattening"
        else:
            whole = True
            parameters = int(layer.weight_matrix.size + layer.bias_vector.size)
            multiply_adds = int(layer.weight_matrix.size)
            label = (
                "the answer, one score per kind"
                if position == last
                else f"dense, {answers[0]} units"
            )
        readings.append(
            LayerReading(
                label=label,
                reads=reads,
                answers=answers,
                parameters=parameters,
                multiply_adds=multiply_adds,
                field=SIDE if whole else min(field, SIDE),
                step=step,
                reads_whole_picture=whole,
            )
        )
    return readings


def _compass() -> list[np.ndarray]:
    """The hand-designed Sobel grid turned to each of eight directions."""
    base = np.asarray(weights_of(GradientOperator.SOBEL, vertical=False))
    values = [base[row, column] for row, column in RING]
    turned = []
    for turn in range(len(RING)):
        grid = np.zeros((3, 3))
        for place, (row, column) in enumerate(RING):
            grid[row, column] = values[(place - turn) % len(RING)]
        turned.append(grid)
    return turned


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        (first * second).sum() / (np.linalg.norm(first) * np.linalg.norm(second))
    )


def _best_match(kernel: np.ndarray) -> tuple[int, float]:
    """Which turned Sobel grid the kernel's shape is closest to, and how close.

    The kernel's mean is taken off first, because every Sobel grid sums to zero
    and a learned filter need not, so without it the comparison would reward a
    filter for how much it answers to plain brightness rather than for its
    shape.
    """
    centred = kernel - kernel.mean()
    scores = [_cosine(centred, grid) for grid in _compass()]
    best = int(np.argmax(scores))
    return best, scores[best]


class EdgeBaseline(BaseModel):
    """How Sobel-like a grid of nine random numbers already is.

    The best of eight cosines is large by chance, since the eight directions
    between them cover the shapes a three by three grid can take, so a learned
    filter's closeness means something only against this.
    """

    n_draws: int
    median: float
    ninetieth_percentile: float


@lru_cache(maxsize=1)
def _baseline() -> EdgeBaseline:
    generator = np.random.default_rng(0)
    best = [
        _best_match(generator.normal(size=(3, 3)))[1] for _ in range(BASELINE_DRAWS)
    ]
    return EdgeBaseline(
        n_draws=BASELINE_DRAWS,
        median=float(np.median(best)),
        ninetieth_percentile=float(np.quantile(best, 0.9)),
    )


class FilterReading(BaseModel):
    """One first-layer filter, before and after training.

    ``moved`` is how far training carried the nine weights, as a share of how
    long they were at the start. ``closest`` names the turned Sobel grid the
    trained shape is nearest, and ``closeness`` and ``closeness_at_start`` are
    that cosine after and before training, measured with the mean taken off.
    ``weight_total`` is what the nine weights sum to, which a Sobel grid's do
    not, and which says how much the filter answers to plain brightness.
    ``active_share`` is the share of positions over every held-out picture
    where the filter answered above zero.
    """

    index: int
    kernel: list[list[float]]
    start_kernel: list[list[float]]
    closest_grid: list[list[float]]
    bias: float
    moved: float
    closest: str
    closeness: float
    closeness_at_start: float
    weight_total: float
    active_share: float
    nearly_silent: bool


def _filter_readings(
    trained: _Trained, held_out: PictureCollection
) -> list[FilterReading]:
    first = trained.stack[0]
    if not isinstance(first, Conv2d):
        return []
    start = trained.start[0]
    assert isinstance(start, Conv2d)
    kernels = np.asarray(first.kernels)[:, 0]
    start_kernels = np.asarray(start.kernels)[:, 0]
    outputs = np.asarray(first.respond_to(held_out.pictures).outputs)
    compass = _compass()
    readings = []
    for index in range(kernels.shape[0]):
        kernel = kernels[index]
        best, closeness = _best_match(kernel)
        active = float((outputs[:, index] > 0).mean())
        readings.append(
            FilterReading(
                index=index,
                kernel=_rounded(kernel),
                start_kernel=_rounded(start_kernels[index]),
                closest_grid=_rounded(compass[best]),
                bias=float(first.bias_vector[index]),
                moved=float(
                    np.linalg.norm(kernel - start_kernels[index])
                    / np.linalg.norm(start_kernels[index])
                ),
                closest=COMPASS[best],
                closeness=closeness,
                closeness_at_start=_cosine(
                    start_kernels[index] - start_kernels[index].mean(), compass[best]
                ),
                weight_total=float(kernel.sum()),
                active_share=active,
                nearly_silent=active < DEAD_BELOW,
            )
        )
    return readings


def _rounded(values: np.ndarray, places: int = 4) -> list[list[float]]:
    return [[round(float(value), places) for value in row] for row in values]


class EpochReading(BaseModel):
    """One epoch of training: the mean loss over it, and both accuracies after it."""

    epoch: int
    loss: float
    training_accuracy: float
    held_out_accuracy: float


class VariantReport(BaseModel):
    """One arrangement, trained from one seed, and everything measured on it.

    ``seconds`` is how long the training took on whichever machine answered,
    which moves between runs; the counts beside it do not.
    """

    variant: Variant
    label: str
    seed: int
    layers: list[LayerReading]
    n_parameters: int
    n_multiply_adds: int
    n_training: int
    n_held_out: int
    training_accuracy: float
    held_out_accuracy: float
    best_held_out_accuracy: float
    best_epoch: int
    final_loss: float
    history: list[EpochReading]
    shifts: list[ShiftReading]
    filters: list[FilterReading]
    edge_baseline: EdgeBaseline
    n_nearly_silent: int
    seconds: float


@lru_cache(maxsize=16)
def _report(variant: Variant, seed: int) -> VariantReport:
    training, held_out = reference_split()
    trained = _train(variant, seed)
    layers = _layer_readings(trained.stack)
    filters = _filter_readings(trained, held_out)
    held = [reading.held_out_accuracy for reading in trained.history]
    best = int(np.argmax(held))
    return VariantReport(
        variant=variant,
        label=VARIANT_LABELS[variant],
        seed=seed,
        layers=layers,
        n_parameters=sum(layer.parameters for layer in layers),
        n_multiply_adds=sum(layer.multiply_adds for layer in layers),
        n_training=len(training),
        n_held_out=len(held_out),
        training_accuracy=trained.history[-1].training_accuracy,
        held_out_accuracy=trained.history[-1].held_out_accuracy,
        best_held_out_accuracy=held[best],
        best_epoch=best + 1,
        final_loss=trained.history[-1].loss,
        history=[
            EpochReading(
                epoch=reading.epoch,
                loss=reading.loss,
                training_accuracy=reading.training_accuracy,
                held_out_accuracy=reading.held_out_accuracy,
            )
            for reading in trained.history
        ],
        shifts=_shift_readings(trained.stack, held_out),
        filters=filters,
        edge_baseline=_baseline(),
        n_nearly_silent=sum(reading.nearly_silent for reading in filters),
        seconds=trained.seconds,
    )


@router.get("/network", response_model=VariantReport)
def read_network(
    variant: Variant = Variant.REFERENCE,
    seed: int = Query(0, ge=0, le=2, description="the weight seed, 0, 1 or 2"),
) -> VariantReport:
    """One arrangement trained from one seed, counted, scored and shifted."""
    return _report(variant, seed)


# ---------------------------------------------------------------------------
# One picture of each kind at every depth
# ---------------------------------------------------------------------------


class DepthMaps(BaseModel):
    """Every map one layer answers with, for one picture."""

    depth: Depth
    label: str
    side: int
    maps: list[list[list[float]]]
    largest: float


class PictureThroughNetwork(BaseModel):
    """One held-out picture and what the network made of it at every depth."""

    kind: str
    kind_words: str
    position: int
    picture: list[list[float]]
    depths: list[DepthMaps]
    vector: list[float]
    scores: list[float]
    probabilities: list[float]
    called: str


class WorkedCell(BaseModel):
    """One number of the first map worked through by hand, and its pooling.

    ``patch`` is the three by three the filter read at ``row``, ``column`` of
    the cross, with the border of zeros already in place where it reaches past
    the edge. ``products`` is that times the filter, entry by entry, whose sum
    plus ``bias`` is ``score``, and ``output`` is the score after the bend.
    ``pool_window`` is the two by two of first-layer answers that position
    shares a pooling window with, and ``pooled`` the largest of them.
    """

    filter_index: int
    row: int
    column: int
    patch: list[list[float]]
    kernel: list[list[float]]
    products: list[list[float]]
    product_total: float
    bias: float
    score: float
    output: float
    pool_top: int
    pool_left: int
    pool_window: list[list[float]]
    pooled: float


class MapsResponse(BaseModel):
    """One held-out picture of each kind, through the reference network."""

    kind_names: list[str]
    pictures: list[PictureThroughNetwork]
    worked: WorkedCell


def _first_of_each_kind(held_out: PictureCollection) -> list[int]:
    return [int(np.flatnonzero(held_out.labels == label)[0]) for label in range(4)]


@lru_cache(maxsize=1)
def _maps() -> MapsResponse:
    _, held_out = reference_split()
    stack = reference_network().stack
    positions = _first_of_each_kind(held_out)
    response = stack.respond_to(held_out.pictures[positions])
    probabilities = _probabilities(stack, held_out.pictures[positions])
    pictures = []
    for row, position in enumerate(positions):
        depths = []
        for depth, layer_position in DEPTH_POSITIONS.items():
            block = np.asarray(response[layer_position].outputs)[row]
            depths.append(
                DepthMaps(
                    depth=depth,
                    label=DEPTH_LABELS[depth],
                    side=int(block.shape[-1]),
                    maps=[_rounded(channel, 3) for channel in block],
                    largest=float(block.max()),
                )
            )
        scores = np.asarray(response[len(stack) - 1].outputs)[row]
        kind = held_out.kind_of(position)
        pictures.append(
            PictureThroughNetwork(
                kind=kind,
                kind_words=KIND_WORDS[kind],
                position=position,
                picture=as_rows(held_out.pictures[position]),
                depths=depths,
                vector=[
                    round(float(value), 4)
                    for value in np.asarray(response[len(stack) - 2].outputs)[row]
                ],
                scores=[round(float(value), 4) for value in scores],
                probabilities=[round(float(value), 4) for value in probabilities[row]],
                called=held_out.kind_names[int(np.argmax(scores))],
            )
        )

    first = stack[0]
    assert isinstance(first, Conv2d)
    first_maps = np.asarray(response[0].outputs)[0]
    filter_index, cell_row, cell_column = np.unravel_index(
        int(np.argmax(first_maps)), first_maps.shape
    )
    padded = np.pad(held_out.pictures[positions[0], 0], 1)
    patch = padded[cell_row : cell_row + 3, cell_column : cell_column + 3]
    kernel = np.asarray(first.kernels)[filter_index, 0]
    products = patch * kernel
    bias = float(first.bias_vector[filter_index])
    score = float(products.sum() + bias)
    top, left = (cell_row // 2) * 2, (cell_column // 2) * 2
    window = first_maps[filter_index, top : top + 2, left : left + 2]
    worked = WorkedCell(
        filter_index=int(filter_index),
        row=int(cell_row),
        column=int(cell_column),
        patch=_rounded(patch),
        kernel=_rounded(kernel),
        products=_rounded(products),
        product_total=float(products.sum()),
        bias=bias,
        score=score,
        output=max(score, 0.0),
        pool_top=int(top),
        pool_left=int(left),
        pool_window=_rounded(window),
        pooled=float(window.max()),
    )
    return MapsResponse(kind_names=list(KINDS), pictures=pictures, worked=worked)


@router.get("/maps", response_model=MapsResponse)
def read_maps() -> MapsResponse:
    """One picture of each kind through the network, every map at every depth."""
    return _maps()


# ---------------------------------------------------------------------------
# How far one unit can see
# ---------------------------------------------------------------------------


class ReachRequest(BaseModel):
    """One unit of the reference network, named by where it sits."""

    kind: Kind = Kind.CROSS
    depth: Depth = Depth.SECOND_POOLING
    row: int = Field(2, ge=0, le=SIDE - 1)
    column: int = Field(2, ge=0, le=SIDE - 1)

    model_config = {"extra": "forbid"}


class ReachResponse(BaseModel):
    """Which pixels one unit can see, worked out and then checked by poking.

    ``top``, ``left``, ``bottom`` and ``right`` bound the window the geometry
    says the unit reads, inclusive, and may run past the picture, where the
    padding's zeros are all the unit sees. ``influence`` is, for every pixel,
    how far the unit's answer moved, largest over its channels, when that pixel
    alone was brightened by ``poke``; it is taken before the bend at a
    convolution, so a unit whose bend answered zero still shows what it can
    see. ``n_moved_outside`` counts pixels outside the window that moved the
    unit, which the geometry says must be none.
    """

    kind: Kind
    depth: Depth
    label: str
    side: int
    row: int
    column: int
    field: int
    top: int
    left: int
    bottom: int
    right: int
    n_in_window: int
    n_moved: int
    n_moved_outside: int
    poke: float
    picture: list[list[float]]
    influence: list[list[float]]
    largest_influence: float


def _window_of(depth: Depth, row: int, column: int) -> tuple[int, int, int, int, int]:
    """The input window a unit at ``depth`` reads, from the layer geometry.

    Walks back down from the unit: a pooling window of two at stride two turns
    unit ``i`` into units ``2i`` to ``2i + 1`` below it, and a three by three
    convolution with one pixel of padding turns ``i`` into ``i - 1`` to
    ``i + 1``.
    """
    steps = {
        Depth.FIRST_CONVOLUTION: ("convolution",),
        Depth.FIRST_POOLING: ("pooling", "convolution"),
        Depth.SECOND_CONVOLUTION: ("convolution", "pooling", "convolution"),
        Depth.SECOND_POOLING: ("pooling", "convolution", "pooling", "convolution"),
    }[depth]
    top, bottom, left, right = row, row, column, column
    for step in steps:
        if step == "pooling":
            top, bottom, left, right = 2 * top, 2 * bottom + 1, 2 * left, 2 * right + 1
        else:
            top, bottom, left, right = top - 1, bottom + 1, left - 1, right + 1
    return top, left, bottom, right, bottom - top + 1


@router.post("/reach", response_model=ReachResponse)
def reach(request: ReachRequest) -> ReachResponse:
    """Which pixels one unit can see, by the geometry and by poking every pixel."""
    stack = reference_network().stack
    layer_position = DEPTH_POSITIONS[request.depth]
    side = int(stack[layer_position].shape.answers[-1])
    if request.row >= side or request.column >= side:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the maps {DEPTH_LABELS[request.depth]} are {side} by {side}, "
                f"so there is no unit at row {request.row}, column {request.column}"
            ),
        )

    _, held_out = reference_split()
    position = _first_of_each_kind(held_out)[KINDS.index(request.kind.value)]
    picture = held_out.pictures[position, 0]
    poked = np.repeat(picture[np.newaxis], SIDE * SIDE + 1, axis=0)
    for pixel in range(SIDE * SIDE):
        poked[pixel + 1, pixel // SIDE, pixel % SIDE] += POKE
    response = stack.respond_to(poked[:, np.newaxis])
    reading = response[layer_position]
    block = np.asarray(
        reading.scores if isinstance(stack[layer_position], Conv2d) else reading.outputs
    )
    at_unit = block[:, :, request.row, request.column]
    moved = np.abs(at_unit[1:] - at_unit[0]).max(axis=1).reshape(SIDE, SIDE)

    top, left, bottom, right, field = _window_of(
        request.depth, request.row, request.column
    )
    inside = np.zeros((SIDE, SIDE), dtype=bool)
    inside[
        max(top, 0) : min(bottom, SIDE - 1) + 1, max(left, 0) : min(right, SIDE - 1) + 1
    ] = True
    reached = moved > 1e-12
    return ReachResponse(
        kind=request.kind,
        depth=request.depth,
        label=DEPTH_LABELS[request.depth],
        side=side,
        row=request.row,
        column=request.column,
        field=field,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        n_in_window=int(inside.sum()),
        n_moved=int(reached.sum()),
        n_moved_outside=int((reached & ~inside).sum()),
        poke=POKE,
        picture=as_rows(picture),
        influence=_rounded(moved),
        largest_influence=float(moved.max()),
    )


# ---------------------------------------------------------------------------
# Where the arrangement stops being defined
# ---------------------------------------------------------------------------


class UnseenPicture(BaseModel):
    """One ring and what the network called it."""

    picture: list[list[float]]
    probabilities: list[float]
    called: str


class SideReading(BaseModel):
    """What two poolings of two leave of a picture of one side.

    ``flattened`` is how many numbers the flattening would hand the dense
    layer, which was built to read ``trained_width`` and nothing else.
    ``first_drops`` is how many rows (and as many columns) of first-layer maps
    the first pooling leaves out of every window when the side is odd, and
    ``second_drops`` the same for the second pooling.
    """

    side: int
    after_first_pooling: int
    after_second_pooling: int
    flattened: int
    trained_width: int
    readable: bool
    first_drops: int
    second_drops: int


class UnseenResponse(BaseModel):
    """What the network answers when the question is outside what it was built for.

    ``ring_calls`` counts how many of the rings each kind was called, in the
    order of ``kind_names``. The confidence figures are the probability of the
    called kind, averaged, and the share of pictures on which that probability
    passed 0.9, for the rings and for the held-out pictures of the four trained
    kinds.
    """

    kind_names: list[str]
    n_rings: int
    ring_calls: list[int]
    ring_mean_confidence: float
    ring_share_above_ninety: float
    held_out_mean_confidence: float
    held_out_share_above_ninety: float
    examples: list[UnseenPicture]
    blank_brightness: float
    blank_probabilities: list[float]
    blank_called: str
    sides: list[SideReading]


@lru_cache(maxsize=1)
def _unseen() -> UnseenResponse:
    _, held_out = reference_split()
    stack = reference_network().stack
    rings = drawn_collection(UNSEEN_COUNT, seed=DEFAULT_SEED, kinds=("ring",))
    ring_probabilities = _probabilities(stack, rings.pictures)
    held_probabilities = _probabilities(stack, held_out.pictures)
    ring_top = ring_probabilities.max(axis=1)
    held_top = held_probabilities.max(axis=1)
    blank = np.full((1, 1, SIDE, SIDE), BLANK_BRIGHTNESS)
    blank_probabilities = _probabilities(stack, blank)[0]
    trained_width = int(stack[4].shape.answers[0])
    n_filters = int(stack[4].shape.reads[0])
    sides = []
    for side in OTHER_SIDES:
        first = side // 2
        second = first // 2
        flattened = n_filters * second * second
        sides.append(
            SideReading(
                side=side,
                after_first_pooling=first,
                after_second_pooling=second,
                flattened=flattened,
                trained_width=trained_width,
                readable=flattened == trained_width,
                first_drops=side % 2,
                second_drops=first % 2,
            )
        )
    return UnseenResponse(
        kind_names=list(KINDS),
        n_rings=len(rings),
        ring_calls=np.bincount(ring_probabilities.argmax(axis=1), minlength=4).tolist(),
        ring_mean_confidence=float(ring_top.mean()),
        ring_share_above_ninety=float((ring_top > 0.9).mean()),
        held_out_mean_confidence=float(held_top.mean()),
        held_out_share_above_ninety=float((held_top > 0.9).mean()),
        examples=[
            UnseenPicture(
                picture=as_rows(rings.pictures[index]),
                probabilities=[
                    round(float(value), 4) for value in ring_probabilities[index]
                ],
                called=KINDS[int(ring_probabilities[index].argmax())],
            )
            for index in range(4)
        ],
        blank_brightness=BLANK_BRIGHTNESS,
        blank_probabilities=[round(float(value), 4) for value in blank_probabilities],
        blank_called=KINDS[int(blank_probabilities.argmax())],
        sides=sides,
    )


@router.get("/unseen", response_model=UnseenResponse)
def read_unseen() -> UnseenResponse:
    """A kind never shown, a blank picture, and pictures of other sizes."""
    return _unseen()
