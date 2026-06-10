"""Learning the metric itself, from pairs rather than labels.

Every endpoint here is registered under ``/concepts/learning-the-metric-itself``.

What the page is about
----------------------
A network whose answer is a position rather than a score per kind, trained on
pairs of pictures. Two pictures of one kind are pulled together; two of
different kinds are pushed apart until they are a margin away, and past the
margin they are left alone. Both pictures of a pair go through the same
weights, which is all the old name for the arrangement, a Siamese network,
describes.

The pair loss is computed here, outside the layers
--------------------------------------------------
The library's losses each score one row against a target, and a pair loss
scores two rows against each other, so there is no loss object to hand to
``LayerStack.backward_pass``. The loss and its slope at the two positions are
computed in :func:`pair_loss`, and the backward walk is done by hand: each
layer's ``correction_for`` down the stack, which is the four-line walk
``backward_pass`` does itself, once for each picture of a pair. The two
gradients are added layer by layer, because the two towers are one set of
weights and a weight used twice has a slope that is the sum of its two uses.
:func:`two_tower_gradients` does exactly that, and ``/shared-weights`` checks
the sum against a finite difference of the pair loss.

``LayerStack.stepped_by`` rebuilds every position of a stack independently, so
one layer object placed at two positions would come apart after a step. That
is not what happens here. One stack is run twice and stepped once, with the
summed gradient, so there is only ever one copy of every weight.

Training takes every pair inside a batch
----------------------------------------
Walking two towers for every pair costs two forward and two backward passes per
pair. A batch of twenty-four pictures, eight of each kind, holds 276 pairs, and
because the gradient of a sum of pair losses is a sum over rows, the same
answer comes from one pass over the twenty-four pictures with each picture's
arriving slope the sum of its slopes from every pair it belongs to. That is
:func:`_trained_on_pairs`: the two towers' sum, computed once per picture
rather than once per pair. Measured while designing this module, forty epochs
took 2.5 s that way, where one partner per picture through two literal towers
took 5.2 s and placed the seen kinds less well.

The tower
---------
The reference arrangement from :mod:`oop_ml_api.picture_network` with its last
layer replaced: eight numbers with no bend, rather than four scores. It is
``built_network(REFERENCE_DESIGN, 8)``, so every starting weight below the last
layer is the reference's own.

Three readings of whether a kind keeps together
-----------------------------------------------
Each kind is read three ways among the 300 evaluation pictures, and the page
reports all three, because they disagree and choosing one would choose the
answer. The single nearest neighbour is what a search returns. The ten nearest
are a neighbourhood. The mean distance inside a kind over the mean distance to
everything else is a whole-cloud reading, and it has a confound worth naming:
pushing the other kinds apart from one another enlarges every kind's distance
to everything else, so the ratio can fall for a kind that did not gather at
all.

What was measured before the page was written
---------------------------------------------
The published claim for this method is that a metric learned from sameness
carries to kinds the training never showed. On these pictures, trained on
three kinds with the fourth held back, the answer depended on the reading.
Whichever kind was held back, a held-out picture of it found a nearest
neighbour of its own kind less often under the pair-trained positions than
under the hidden layer of an ordinary classifier trained on the same three
kinds. On the whole-cloud ratio the held-back kind and the ring usually came
out tighter under the pair-trained positions. The endpoints below measure all
of that; the page reports it as found.

Read among the 180 held-out pictures of the three training kinds alone, the
pair tower is close to the classifier on two seeds of three, so it is not a
weak tower. What goes wrong is where it puts what it was never shown: held-back
bars land among the crosses, and a cross's nearest picture among all 300 is
then often a bar. A tower answering with sixteen numbers rather than eight does
no better on the bar, which ``/wider-positions`` measures, so the narrower
answer is not the reason.

The margin is a unit of length for the loss and not for the training. Scaling
every position by ``c`` scales the loss by ``c ** 2``, so the loss surface at
margin ``m`` is the surface at margin one, stretched. But the rate and the
starting weights do not scale with it, and the slope reaching the lower layers
grows with the margin, so margin 2 at the rate that trains margin 1 froze: every
position landed on one point, where a different-kind pair's push has no
direction, and the loss stayed at exactly the share of different-kind pairs in
a batch times ``m ** 2 / 2``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from oop_ml import (
    BackwardPass,
    Conv2d,
    DenseLayer,
    Feature,
    LayerGradient,
    LayerStack,
    PassPurpose,
    PrincipalComponentAnalysis,
    StackResponse,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.picture_collection import (
    KINDS,
    UNSEEN_KIND,
    PictureCollection,
    drawn_collection,
)
from oop_ml_api.picture_network import (
    NetworkDesign,
    accuracy_of,
    built_network,
    reference_network,
    reference_split,
    trained_network,
    vectors_of,
)

router = APIRouter(
    prefix="/concepts/learning-the-metric-itself", tags=["learning-the-metric-itself"]
)

POSITION_WIDTH = 8
"""How many numbers the tower answers with, in place of the reference's four scores."""

MARGIN = 1.0
"""How far apart a different-kind pair is pushed before it is left alone.

One, because at the reference starting weights the mean distance between two
pictures' positions is about 0.73, so one is a distance the untrained tower
already nearly reaches and not a scale it has to be dragged to. Chosen before
any margin was compared.
"""

RATE = 1.0
"""The step size at the default margin.

Ten times the reference classifier's, because a pair loss averaged over 276
pairs produces slopes about a tenth as large; at 0.1 the loss was still
falling steeply after forty epochs.
"""

EPOCHS = 40
"""The same length the reference classifier trains for."""

SPREAD_EPOCHS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40)
"""The epochs after which the spread of the positions is measured.

Every epoch early, where the collapse without different-kind pairs is fast,
and every fifth after. Measuring after all forty cost about a second per run,
which the cold budget could not spare.
"""

PER_KIND_IN_BATCH = 8
"""Pictures of each kind in one batch, so a batch of three kinds holds 24."""

HELD_BACK = "bar"
"""The kind the pair tower and the three-kind classifier never see."""

RING_COUNT = 60
"""Rings in the evaluation set, as many as each kind has in the held-out half."""

RING_SEED = 11
"""The draw for the rings, any seed other than the one the reference collection uses."""

NEIGHBOURHOOD = 10
"""How many nearest pictures the neighbourhood reading counts."""

NEIGHBOURS_SHOWN = 5
"""How many nearest pictures the sphere offers for each picture."""

GREY_LEVELS = 100
"""The sphere's pictures are sent as whole numbers from 0 to this, to keep the answer small."""

NUDGE = 1e-5
"""The step of the central finite difference that checks the summed gradient."""

MARGINS = (0.5, 1.0, 2.0, 4.0)
"""The margins the margin sweep trains at."""

MARGIN_SEEDS = (0, 1, 2)
"""The weight seeds the margin sweep may be run from."""

MAX_COORDINATE = 100.0
"""The largest coordinate the pair-loss calculator accepts, in absolute value."""

MAX_WIDTH = 16
"""The most coordinates a position handed to the calculator may have."""

SpaceName = Literal["pixels", "classifier_all_four", "classifier_three", "pairs"]

SPACE_TITLES: dict[str, str] = {
    "pixels": "the pixels themselves",
    "classifier_all_four": "a classifier’s layer, trained to name all four kinds",
    "classifier_three": "a classifier’s layer, trained to name the same three kinds",
    "pairs": "positions trained on pairs of the three kinds",
}
"""What each space is, in the page's own words, since a widget prints these."""


# --------------------------------------------------------------------------
# The collections.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    """The held-out half of all four kinds, then sixty rings.

    ``labels`` index ``names``, which is the four kinds and the ring, so the
    ring is label four.
    """

    pictures: np.ndarray
    labels: np.ndarray
    names: tuple[str, ...]


@lru_cache(maxsize=1)
def _evaluation() -> Evaluation:
    _, held = reference_split()
    rings = drawn_collection(RING_COUNT, seed=RING_SEED, kinds=(UNSEEN_KIND,))
    return Evaluation(
        pictures=np.concatenate([held.pictures, rings.pictures]),
        labels=np.concatenate([held.labels, np.full(RING_COUNT, len(KINDS))]),
        names=(*KINDS, UNSEEN_KIND),
    )


def _without(collection: PictureCollection, held_back: str) -> PictureCollection:
    """``collection`` with one kind removed and the rest numbered from zero."""
    kept = [label for label, name in enumerate(KINDS) if name != held_back]
    rows = np.isin(collection.labels, kept)
    renumbered = np.full(len(KINDS), -1, dtype=np.int64)
    renumbered[kept] = np.arange(len(kept))
    return PictureCollection(
        pictures=collection.pictures[rows],
        labels=renumbered[collection.labels[rows]],
        kind_names=tuple(KINDS[label] for label in kept),
        masks=collection.masks[rows],
    )


@lru_cache(maxsize=4)
def _training_without(held_back: str) -> PictureCollection:
    training, _ = reference_split()
    return _without(training, held_back)


@lru_cache(maxsize=4)
def _held_out_without(held_back: str) -> PictureCollection:
    _, held = reference_split()
    return _without(held, held_back)


def _training_kinds(held_back: str) -> tuple[str, ...]:
    return tuple(name for name in KINDS if name != held_back)


# --------------------------------------------------------------------------
# The pair loss, and the walk down two towers.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairLoss:
    """Every pair's loss and distance, and each pair's slope at its two positions.

    ``slope_first[i]`` is the slope of pair ``i``'s own loss at its first
    position, not yet divided by the number of pairs; ``slope_second`` is its
    negative, because the loss reads only the difference between the two.
    """

    losses: np.ndarray
    distances: np.ndarray
    slope_first: np.ndarray
    slope_second: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.losses.mean())


def pair_loss(
    first: np.ndarray, second: np.ndarray, same: np.ndarray, margin: float
) -> PairLoss:
    """The contrastive loss of Hadsell, Chopra and LeCun, pair by pair.

    For a same-kind pair at distance ``d`` the loss is ``d ** 2 / 2`` and its
    slope at the first position is the gap itself, pulling the two together.
    For a different-kind pair it is ``max(0, margin - d) ** 2 / 2``, whose
    slope is ``-(margin - d) / d`` times the gap, pushing them apart until
    they are ``margin`` away and then not at all.

    At ``d = 0`` a different-kind pair's push has a size, ``margin``, and no
    direction, since the gap it would point along is empty. The slope is
    returned as zero there, which is one choice among many and the reason a
    tower that has put everything on one point stays there.
    """
    gap = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    distances = np.sqrt((gap**2).sum(axis=1))
    short = np.maximum(margin - distances, 0.0)
    losses = np.where(same, 0.5 * distances**2, 0.5 * short**2)
    reach = np.where(distances == 0.0, 1.0, distances)
    pull = np.where(same, 1.0, -short / reach)
    slope_first = pull[:, np.newaxis] * gap
    return PairLoss(
        losses=losses,
        distances=distances,
        slope_first=slope_first,
        slope_second=-slope_first,
    )


def walked(
    stack: LayerStack, response: StackResponse, arriving: np.ndarray
) -> list[LayerGradient | None]:
    """Each layer's ``correction_for`` down the stack, the walk ``backward_pass`` does.

    ``arriving`` is the slope of whatever is being minimised at the stack's
    final outputs; it leaves each layer as the slope at that layer's inputs.
    The gradients come out top to bottom and are returned in layer order.
    """
    gradients: list[LayerGradient | None] = []
    for layer, layer_response in zip(reversed(stack), reversed(response), strict=True):
        correction = layer.correction_for(layer_response, arriving)
        gradients.append(correction.gradient)
        arriving = np.asarray(correction.passed_down)
    gradients.reverse()
    return gradients


def summed(
    first: list[LayerGradient | None], second: list[LayerGradient | None]
) -> list[LayerGradient | None]:
    """Two walks' gradients added layer by layer, for one set of weights used twice."""
    total: list[LayerGradient | None] = []
    for one, other in zip(first, second, strict=True):
        if one is None or other is None:
            total.append(None)
            continue
        total.append(
            LayerGradient(
                weights=np.asarray(one.weights) + np.asarray(other.weights),
                biases=np.asarray(one.biases) + np.asarray(other.biases),
            )
        )
    return total


@dataclass(frozen=True)
class TwoTowerPass:
    """Both towers' gradients for one block of pairs, and their sum."""

    loss: PairLoss
    first_tower: list[LayerGradient | None]
    second_tower: list[LayerGradient | None]
    both: list[LayerGradient | None]


def two_tower_gradients(
    stack: LayerStack,
    first_pictures: np.ndarray,
    second_pictures: np.ndarray,
    same: np.ndarray,
    margin: float,
) -> TwoTowerPass:
    """Run each picture of every pair through ``stack``, walk both down, add."""
    first = stack.respond_to(first_pictures, PassPurpose.TRAINING)
    second = stack.respond_to(second_pictures, PassPurpose.TRAINING)
    loss = pair_loss(
        np.asarray(first.outputs), np.asarray(second.outputs), same, margin
    )
    count = len(loss.losses)
    first_tower = walked(stack, first, loss.slope_first / count)
    second_tower = walked(stack, second, loss.slope_second / count)
    return TwoTowerPass(
        loss=loss,
        first_tower=first_tower,
        second_tower=second_tower,
        both=summed(first_tower, second_tower),
    )


# --------------------------------------------------------------------------
# Distances, readings and training.
# --------------------------------------------------------------------------


def _positions(stack: LayerStack, pictures: np.ndarray) -> np.ndarray:
    return np.asarray(stack.respond_to(pictures).outputs, dtype=np.float64)


def _distances(vectors: np.ndarray) -> np.ndarray:
    """Every Euclidean distance, by the matrix product, centred first.

    Centring changes no distance and keeps the three-term expansion exact to
    the last bits where raw magnitudes would cancel badly.
    """
    centred = vectors - vectors.mean(axis=0)
    squared = (centred**2).sum(axis=1)
    inner = centred @ centred.T
    distances = np.sqrt(
        np.maximum(squared[:, None] - 2.0 * inner + squared[None, :], 0.0)
    )
    np.fill_diagonal(distances, 0.0)
    return distances


def _mean_spread(vectors: np.ndarray) -> float:
    """The mean distance between every two positions."""
    distances = _distances(vectors)
    count = len(vectors)
    return float(distances.sum() / (count * (count - 1)))


@dataclass(frozen=True)
class KindReading:
    """How well one kind's pictures keep company with their own kind.

    ``nearest_same_kind`` is the share of the kind's pictures whose single
    nearest other picture, among all 300, is the same kind, counting only
    pictures whose nearest is not tied; ``None`` when every one is tied.
    ``ten_nearest_same_kind`` is the mean share of each picture's ten nearest
    that are its own kind. ``within_over_across`` is the mean distance inside
    the kind over the mean distance from it to everything else, ``None`` when
    the second is zero.
    """

    name: str
    nearest_same_kind: float | None
    ten_nearest_same_kind: float
    within_over_across: float | None
    n_pictures: int
    n_counted: int
    nearest_count: int


def _readings(vectors: np.ndarray, evaluation: Evaluation) -> list[KindReading]:
    distances = _distances(vectors)
    np.fill_diagonal(distances, np.inf)
    nearest_distance = distances.min(axis=1)
    tied = (distances == nearest_distance[:, None]).sum(axis=1) > 1
    nearest = distances.argmin(axis=1)
    agrees = evaluation.labels[nearest] == evaluation.labels
    neighbourhood = np.argsort(distances, axis=1, kind="stable")[:, :NEIGHBOURHOOD]
    neighbourhood_shares = (
        evaluation.labels[neighbourhood] == evaluation.labels[:, None]
    ).mean(axis=1)
    np.fill_diagonal(distances, np.nan)

    readings = []
    for label, name in enumerate(evaluation.names):
        rows = evaluation.labels == label
        counted = rows & ~tied
        within = float(np.nanmean(distances[np.ix_(rows, rows)]))
        across = float(np.nanmean(distances[np.ix_(rows, ~rows)]))
        readings.append(
            KindReading(
                name=name,
                nearest_same_kind=(
                    float(agrees[counted].mean()) if counted.any() else None
                ),
                ten_nearest_same_kind=float(neighbourhood_shares[rows].mean()),
                within_over_across=within / across if across > 0.0 else None,
                n_pictures=int(rows.sum()),
                n_counted=int(counted.sum()),
                nearest_count=int(agrees[counted].sum()),
            )
        )
    return readings


@dataclass(frozen=True)
class KindCountReading:
    """How many of one kind's pictures have a nearest picture of their own kind."""

    name: str
    nearest_count: int
    n_pictures: int


def _nearest_among(
    vectors: np.ndarray, evaluation: Evaluation, kinds: tuple[str, ...]
) -> list[KindCountReading]:
    """The nearest-picture count, searching only among pictures of ``kinds``.

    This is the reading that asks how well a space keeps apart the kinds it
    was trained on, with the kinds it never saw taken out of the search, so a
    picture of an unseen kind cannot be anyone's nearest.
    """
    wanted = [evaluation.names.index(name) for name in kinds]
    chosen = np.flatnonzero(np.isin(evaluation.labels, wanted))
    labels = evaluation.labels[chosen]
    distances = _distances(vectors[chosen])
    np.fill_diagonal(distances, np.inf)
    agrees = labels[distances.argmin(axis=1)] == labels
    return [
        KindCountReading(
            name=name,
            nearest_count=int(agrees[labels == label].sum()),
            n_pictures=int((labels == label).sum()),
        )
        for name, label in zip(kinds, wanted, strict=True)
    ]


@dataclass(frozen=True)
class AgainstMargin:
    """Distances inside and across kinds, read against the margin."""

    mean_within: float
    mean_across: float
    beyond_margin: float


def _against_margin(
    vectors: np.ndarray, labels: np.ndarray, margin: float
) -> AgainstMargin:
    distances = _distances(vectors)
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(vectors), dtype=bool)
    across = distances[~same]
    return AgainstMargin(
        mean_within=float(distances[same & off].mean()),
        mean_across=float(across.mean()),
        beyond_margin=float((across >= margin).mean()),
    )


@dataclass(frozen=True)
class PairRun:
    """One tower trained on pairs, and what the training did along the way.

    ``spreads[i]`` is the mean distance between every two evaluation positions
    after epoch ``spread_epochs[i]``.
    """

    stack: LayerStack
    losses: tuple[float, ...]
    spread_epochs: tuple[int, ...]
    spreads: tuple[float, ...]
    start_spread: float
    training_seconds: float
    pairs_per_step: int
    steps_per_epoch: int
    distinct_pairs_met: int
    pair_evaluations: int
    diverged: bool


def _trained_on_pairs(
    collection: PictureCollection,
    margin: float,
    rate: float,
    seed: int,
    with_different_kinds: bool = True,
    epochs: int = EPOCHS,
    width: int = POSITION_WIDTH,
) -> PairRun:
    """Train a fresh tower on every pair inside balanced batches.

    Each step takes eight pictures of each kind, runs the twenty-four through
    the tower once, scores every pair among them (or every same-kind pair only,
    when ``with_different_kinds`` is false), adds each pair's two slopes onto
    its two pictures' rows, walks that block down once and steps. ``seed``
    draws both the starting weights and the batches.

    The spread is measured on the evaluation pictures after each epoch in
    :data:`SPREAD_EPOCHS`, which is how the fall toward one point is read
    rather than assumed. The training time excludes that measuring.
    """
    stack = built_network(NetworkDesign(weight_seed=seed), width)
    generator = np.random.default_rng(seed)
    by_kind = [
        np.flatnonzero(collection.labels == kind)
        for kind in np.unique(collection.labels)
    ]
    batch_size = PER_KIND_IN_BATCH * len(by_kind)
    steps_per_epoch = min(len(rows) for rows in by_kind) // PER_KIND_IN_BATCH
    first_index, second_index = np.triu_indices(batch_size, k=1)
    evaluation = _evaluation().pictures

    start_spread = _mean_spread(_positions(stack, evaluation))
    losses: list[float] = []
    spread_epochs: list[int] = []
    spreads: list[float] = []
    met: list[np.ndarray] = []
    pairs_per_step = 0
    training_seconds = 0.0
    diverged = False

    for epoch in range(1, epochs + 1):
        shuffled = [generator.permutation(rows) for rows in by_kind]
        epoch_losses = []
        began = time.perf_counter()
        for step in range(steps_per_epoch):
            chosen = np.concatenate(
                [
                    rows[step * PER_KIND_IN_BATCH : (step + 1) * PER_KIND_IN_BATCH]
                    for rows in shuffled
                ]
            )
            labels = collection.labels[chosen]
            same = labels[first_index] == labels[second_index]
            if with_different_kinds:
                first, second, pair_same = first_index, second_index, same
            else:
                first, second = first_index[same], second_index[same]
                pair_same = np.ones(len(first), dtype=bool)
            pairs_per_step = len(first)

            try:
                response = stack.respond_to(
                    collection.pictures[chosen], PassPurpose.TRAINING
                )
                placed = np.asarray(response.outputs, dtype=np.float64)
                loss = pair_loss(placed[first], placed[second], pair_same, margin)
                arriving = np.zeros_like(placed)
                np.add.at(arriving, first, loss.slope_first / len(first))
                np.add.at(arriving, second, loss.slope_second / len(first))
                stack = stack.stepped_by(
                    BackwardPass(
                        loss=loss.mean, gradients=walked(stack, response, arriving)
                    ),
                    rate,
                )
            except MLLibError:
                # A step that sent a weight to infinity leaves the next layer a
                # non-finite block, which it refuses. The run stops there and
                # says so rather than answering with nonsense.
                diverged = True
                break
            if not np.isfinite(loss.mean):
                diverged = True
                break
            epoch_losses.append(loss.mean)
            low = np.minimum(chosen[first], chosen[second])
            high = np.maximum(chosen[first], chosen[second])
            met.append(low * len(collection) + high)
        training_seconds += time.perf_counter() - began
        if diverged:
            break
        losses.append(float(np.mean(epoch_losses)))
        if epoch in SPREAD_EPOCHS:
            spread_epochs.append(epoch)
            spreads.append(_mean_spread(_positions(stack, evaluation)))

    codes = np.concatenate(met) if met else np.zeros(0, dtype=np.int64)
    return PairRun(
        stack=stack,
        losses=tuple(losses),
        spread_epochs=tuple(spread_epochs),
        spreads=tuple(spreads),
        start_spread=start_spread,
        training_seconds=training_seconds,
        pairs_per_step=pairs_per_step,
        steps_per_epoch=steps_per_epoch,
        distinct_pairs_met=int(np.unique(codes).size),
        pair_evaluations=int(codes.size),
        diverged=diverged,
    )


@dataclass(frozen=True)
class Study:
    """The pair tower and the classifier it is compared with, on one held-back kind."""

    held_back: str
    seed: int
    classifier_vectors: np.ndarray
    classifier_accuracy: float
    classifier_seconds: float
    pair_run: PairRun
    pair_positions: np.ndarray


@lru_cache(maxsize=8)
def _study(held_back: str, seed: int) -> Study:
    """Two networks on the three kinds left: a classifier, and a tower trained on pairs.

    The classifier is the reference loop at weight seed ``seed``, with the
    reference's shuffling seed; the tower draws its weights and its batches
    from ``seed``.
    """
    collection = _training_without(held_back)
    began = time.perf_counter()
    classifier = trained_network(collection, design=NetworkDesign(weight_seed=seed))
    classifier_seconds = time.perf_counter() - began
    evaluation = _evaluation()
    run = _trained_on_pairs(collection, margin=MARGIN, rate=RATE, seed=seed)
    return Study(
        held_back=held_back,
        seed=seed,
        classifier_vectors=vectors_of(classifier, evaluation.pictures),
        classifier_accuracy=accuracy_of(classifier.stack, _held_out_without(held_back)),
        classifier_seconds=classifier_seconds,
        pair_run=run,
        pair_positions=_positions(run.stack, evaluation.pictures),
    )


@lru_cache(maxsize=1)
def _untrained_readings() -> tuple[list[KindReading], list[KindReading]]:
    """The pixels, and the reference classifier's layer, which need no pairs."""
    evaluation = _evaluation()
    pixels = evaluation.pictures.reshape(len(evaluation.pictures), -1)
    reference = vectors_of(reference_network(), evaluation.pictures)
    return _readings(pixels, evaluation), _readings(reference, evaluation)


# --------------------------------------------------------------------------
# Response models shared by several endpoints.
# --------------------------------------------------------------------------


class KindFigure(BaseModel):
    """One kind's three readings in one space.

    ``in_training`` says whether the network behind the space was shown this
    kind; for the pixels, nothing was trained, and it is false throughout.
    """

    kind: str
    in_training: bool
    nearest_same_kind: float | None
    ten_nearest_same_kind: float
    within_over_across: float | None
    n_pictures: int
    n_counted: int
    nearest_count: int


def _figures(
    readings: list[KindReading], trained_on: tuple[str, ...]
) -> list[KindFigure]:
    return [
        KindFigure(
            kind=reading.name,
            in_training=reading.name in trained_on,
            nearest_same_kind=reading.nearest_same_kind,
            ten_nearest_same_kind=reading.ten_nearest_same_kind,
            within_over_across=reading.within_over_across,
            n_pictures=reading.n_pictures,
            n_counted=reading.n_counted,
            nearest_count=reading.nearest_count,
        )
        for reading in readings
    ]


class KindCount(BaseModel):
    """How many of one kind's pictures found a nearest picture of their own kind."""

    kind: str
    nearest_count: int
    n_pictures: int


class SeenKindsReading(BaseModel):
    """One space searched only among the pictures of the three training kinds."""

    space: SpaceName
    title: str
    kinds: list[KindCount]


def _counts(readings: list[KindCountReading]) -> list[KindCount]:
    return [
        KindCount(
            kind=reading.name,
            nearest_count=reading.nearest_count,
            n_pictures=reading.n_pictures,
        )
        for reading in readings
    ]


# --------------------------------------------------------------------------
# The comparison: does a metric learned from sameness carry?
# --------------------------------------------------------------------------


class SpaceReading(BaseModel):
    """Every kind's readings in one space."""

    space: SpaceName
    title: str
    width: int
    kinds: list[KindFigure]


class ComparisonResponse(BaseModel):
    """Four spaces, five kinds, one held-back kind and one weight seed.

    The evaluation set is the held-out half of the reference collection, sixty
    of each of the four kinds, and sixty rings, 300 pictures in all, and each
    kind's reading is taken among all 300 of them. ``classifier_accuracy`` is
    the three-kind classifier's share of held-out pictures of its own three
    kinds named correctly, which is how it is known to be a working
    classifier rather than a weak one.
    """

    held_back: str
    seed: int
    training_kinds: list[str]
    n_training_pictures: int
    n_evaluation_pictures: int
    margin: float
    spaces: list[SpaceReading]
    seen_kinds_only: list[SeenKindsReading]
    classifier_accuracy: float
    pair_loss_first_epoch: float
    pair_loss_last_epoch: float


@lru_cache(maxsize=8)
def _comparison(held_back: str, seed: int) -> ComparisonResponse:
    study = _study(held_back, seed)
    evaluation = _evaluation()
    pixels, reference = _untrained_readings()
    training_kinds = _training_kinds(held_back)
    flat = evaluation.pictures.reshape(len(evaluation.pictures), -1)
    reference_vectors = vectors_of(reference_network(), evaluation.pictures)
    seen: tuple[tuple[SpaceName, np.ndarray], ...] = (
        ("pixels", flat),
        ("classifier_all_four", reference_vectors),
        ("classifier_three", study.classifier_vectors),
        ("pairs", study.pair_positions),
    )
    return ComparisonResponse(
        seen_kinds_only=[
            SeenKindsReading(
                space=space,
                title=SPACE_TITLES[space],
                kinds=_counts(_nearest_among(vectors, evaluation, training_kinds)),
            )
            for space, vectors in seen
        ],
        held_back=held_back,
        seed=seed,
        training_kinds=list(training_kinds),
        n_training_pictures=len(_training_without(held_back)),
        n_evaluation_pictures=len(evaluation.labels),
        margin=MARGIN,
        spaces=[
            SpaceReading(
                space="pixels",
                title=SPACE_TITLES["pixels"],
                width=int(evaluation.pictures[0].size),
                kinds=_figures(pixels, ()),
            ),
            SpaceReading(
                space="classifier_all_four",
                title=SPACE_TITLES["classifier_all_four"],
                width=int(study.classifier_vectors.shape[1]),
                kinds=_figures(reference, KINDS),
            ),
            SpaceReading(
                space="classifier_three",
                title=SPACE_TITLES["classifier_three"],
                width=int(study.classifier_vectors.shape[1]),
                kinds=_figures(
                    _readings(study.classifier_vectors, evaluation), training_kinds
                ),
            ),
            SpaceReading(
                space="pairs",
                title=SPACE_TITLES["pairs"],
                width=POSITION_WIDTH,
                kinds=_figures(
                    _readings(study.pair_positions, evaluation), training_kinds
                ),
            ),
        ],
        classifier_accuracy=study.classifier_accuracy,
        pair_loss_first_epoch=study.pair_run.losses[0],
        pair_loss_last_epoch=study.pair_run.losses[-1],
    )


@router.get("/comparison", response_model=ComparisonResponse)
def comparison() -> ComparisonResponse:
    """The page's main study: trained on cross, square and disc, weight seed 0."""
    return _comparison(HELD_BACK, 0)


@router.get("/comparison/seed-1", response_model=ComparisonResponse)
def comparison_seed_one() -> ComparisonResponse:
    """The same study from weight seed 1."""
    return _comparison(HELD_BACK, 1)


@router.get("/comparison/seed-2", response_model=ComparisonResponse)
def comparison_seed_two() -> ComparisonResponse:
    """The same study from weight seed 2."""
    return _comparison(HELD_BACK, 2)


@router.get("/comparison/without-cross", response_model=ComparisonResponse)
def comparison_without_cross() -> ComparisonResponse:
    """The cross held back instead of the bar, weight seed 0."""
    return _comparison("cross", 0)


@router.get("/comparison/without-disc", response_model=ComparisonResponse)
def comparison_without_disc() -> ComparisonResponse:
    """The disc held back instead of the bar, weight seed 0."""
    return _comparison("disc", 0)


@router.get("/comparison/without-square", response_model=ComparisonResponse)
def comparison_without_square() -> ComparisonResponse:
    """The square held back instead of the bar, weight seed 0."""
    return _comparison("square", 0)


# --------------------------------------------------------------------------
# The sphere: the held-out pictures, placed.
# --------------------------------------------------------------------------


class SpherePoint(BaseModel):
    """One picture on the drawable sphere, and its nearest pictures in the full space."""

    kind: str
    in_training: bool
    x: float
    y: float
    z: float
    neighbours: list[int]


class SphereSpace(BaseModel):
    """One space's pictures on the sphere.

    ``kept_share`` is how much of the spread of the directions the three drawn
    ones hold. The neighbours are by distance in the full space, not on the
    drawing, so they are the answer the space itself gives.
    """

    space: SpaceName
    title: str
    width: int
    kept_share: float
    points: list[SpherePoint]


class SphereResponse(BaseModel):
    """The 300 evaluation pictures in two spaces, and the pictures themselves.

    Every picture is sent once as whole-number grey levels from 0, the
    darkest pixel of the whole set, to :data:`GREY_LEVELS`, the brightest,
    so a widget can show a picture beside the ones nearest it without the
    answer running to half a megabyte.
    """

    held_back: str
    seed: int
    kinds: list[str]
    grey_levels: int
    levels: list[list[list[int]]]
    darkest: float
    brightest: float
    spaces: list[SphereSpace]


def _on_sphere(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre, unit-normalise, project to three directions, re-normalise.

    Centred first because a pair loss reads only differences between
    positions, so where the origin sits is an accident of the starting
    weights, and a direction measured from an accident means nothing. The
    directions are then measured from the middle of the collection.
    """
    centred = vectors - vectors.mean(axis=0)
    lengths = np.linalg.norm(centred, axis=1, keepdims=True)
    directions = centred / np.where(lengths == 0.0, 1.0, lengths)
    columns = [
        Feature(f"coordinate {position + 1}", directions[:, position])
        for position in range(directions.shape[1])
    ]
    analysis = PrincipalComponentAnalysis(n_components=3).fit(columns)
    placed = np.column_stack(
        [
            np.asarray(component.values, dtype=np.float64)
            for component in analysis.transform(columns)
        ]
    )
    placed_lengths = np.linalg.norm(placed, axis=1, keepdims=True)
    placed = placed / np.where(placed_lengths == 0.0, 1.0, placed_lengths)
    return placed, float(sum(analysis.components.variance_shares))


def _nearest(vectors: np.ndarray, count: int) -> np.ndarray:
    distances = _distances(vectors)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :count]


@lru_cache(maxsize=1)
def _sphere() -> SphereResponse:
    study = _study(HELD_BACK, 0)
    evaluation = _evaluation()
    training_kinds = _training_kinds(HELD_BACK)
    spaces = []
    placements: tuple[tuple[SpaceName, np.ndarray], ...] = (
        ("pairs", study.pair_positions),
        ("classifier_three", study.classifier_vectors),
    )
    for space, vectors in placements:
        placed, kept = _on_sphere(vectors)
        nearest = _nearest(vectors, NEIGHBOURS_SHOWN)
        spaces.append(
            SphereSpace(
                space=space,
                title=SPACE_TITLES[space],
                width=int(vectors.shape[1]),
                kept_share=kept,
                points=[
                    SpherePoint(
                        kind=evaluation.names[int(label)],
                        in_training=evaluation.names[int(label)] in training_kinds,
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                        neighbours=[int(index) for index in near],
                    )
                    for label, point, near in zip(
                        evaluation.labels, placed, nearest, strict=True
                    )
                ],
            )
        )
    flat = evaluation.pictures.reshape(
        len(evaluation.pictures), *evaluation.pictures.shape[2:]
    )
    darkest = float(flat.min())
    brightest = float(flat.max())
    levels = np.rint((flat - darkest) / (brightest - darkest) * GREY_LEVELS).astype(
        np.int64
    )
    return SphereResponse(
        held_back=HELD_BACK,
        seed=0,
        kinds=list(evaluation.names),
        grey_levels=GREY_LEVELS,
        levels=levels.tolist(),
        darkest=darkest,
        brightest=brightest,
        spaces=spaces,
    )


@router.get("/sphere", response_model=SphereResponse)
def sphere() -> SphereResponse:
    """The held-out pictures and the rings, placed by the pair tower and by the classifier."""
    return _sphere()


# --------------------------------------------------------------------------
# Two real pairs, small enough to check.
# --------------------------------------------------------------------------


class GridPicture(BaseModel):
    """One picture as rows of brightness, with the two ends of its grey scale."""

    kind: str
    rows: list[list[float]]
    darkest: float
    brightest: float


class WorkedPair(BaseModel):
    """Two held-out pictures, their positions, and what the pair loss makes of them.

    ``loss`` is the loss under the pair's true label; ``loss_if_same`` and
    ``loss_if_different`` are both readings, so a reader can see what the
    label changes. ``slope_on_first`` is the slope of this pair's loss alone
    at the first position, which is what that picture would be moved against.
    """

    title: str
    same_kind: bool
    first: GridPicture
    second: GridPicture
    first_position: list[float]
    second_position: list[float]
    gap: list[float]
    distance: float
    loss: float
    loss_if_same: float
    loss_if_different: float
    slope_on_first: list[float]
    pixel_distance: float


class WorkedPairResponse(BaseModel):
    margin: float
    width: int
    pairs: list[WorkedPair]


def _grid(picture: np.ndarray, kind: str) -> GridPicture:
    flat = np.asarray(picture, dtype=np.float64).reshape(
        picture.shape[-2], picture.shape[-1]
    )
    return GridPicture(
        kind=kind,
        rows=[[round(float(value), 3) for value in row] for row in flat],
        darkest=float(flat.min()),
        brightest=float(flat.max()),
    )


@lru_cache(maxsize=1)
def _worked_pairs() -> WorkedPairResponse:
    study = _study(HELD_BACK, 0)
    evaluation = _evaluation()
    crosses = np.flatnonzero(evaluation.labels == KINDS.index("cross"))
    squares = np.flatnonzero(evaluation.labels == KINDS.index("square"))
    chosen = (
        ("two crosses", int(crosses[0]), int(crosses[1])),
        ("a cross and a square", int(crosses[0]), int(squares[0])),
    )
    pairs = []
    for title, first, second in chosen:
        same = bool(evaluation.labels[first] == evaluation.labels[second])
        first_position = study.pair_positions[first]
        second_position = study.pair_positions[second]
        as_same = pair_loss(
            first_position[None, :], second_position[None, :], np.array([True]), MARGIN
        )
        as_different = pair_loss(
            first_position[None, :],
            second_position[None, :],
            np.array([False]),
            MARGIN,
        )
        actual = as_same if same else as_different
        pairs.append(
            WorkedPair(
                title=title,
                same_kind=same,
                first=_grid(
                    evaluation.pictures[first],
                    evaluation.names[int(evaluation.labels[first])],
                ),
                second=_grid(
                    evaluation.pictures[second],
                    evaluation.names[int(evaluation.labels[second])],
                ),
                first_position=[float(value) for value in first_position],
                second_position=[float(value) for value in second_position],
                gap=[float(value) for value in first_position - second_position],
                distance=float(actual.distances[0]),
                loss=float(actual.losses[0]),
                loss_if_same=float(as_same.losses[0]),
                loss_if_different=float(as_different.losses[0]),
                slope_on_first=[float(value) for value in actual.slope_first[0]],
                pixel_distance=float(
                    np.linalg.norm(
                        evaluation.pictures[first] - evaluation.pictures[second]
                    )
                ),
            )
        )
    return WorkedPairResponse(margin=MARGIN, width=POSITION_WIDTH, pairs=pairs)


@router.get("/worked-pair", response_model=WorkedPairResponse)
def worked_pair() -> WorkedPairResponse:
    """Two crosses, and a cross with a square, placed by the trained tower."""
    return _worked_pairs()


# --------------------------------------------------------------------------
# The pair loss on any two positions.
# --------------------------------------------------------------------------

Coordinate = Annotated[float, Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)]


class PairLossRequest(BaseModel):
    """Two positions, whether they are one kind, and the margin."""

    model_config = {"extra": "forbid"}

    first: list[Coordinate] = Field(min_length=1, max_length=MAX_WIDTH)
    second: list[Coordinate] = Field(min_length=1, max_length=MAX_WIDTH)
    same_kind: bool
    margin: float = Field(default=MARGIN, ge=0.0, le=10.0)


class PairLossResponse(BaseModel):
    """The pair loss and its slopes, for one pair.

    ``direction_defined`` is false only for a different-kind pair at distance
    zero, where the push has a size and nothing to point along; the slopes
    are returned as zero there.
    """

    distance: float
    loss: float
    loss_if_same: float
    loss_if_different: float
    beyond_margin: bool
    slope_on_first: list[float]
    slope_on_second: list[float]
    direction_defined: bool


@router.post("/pair-loss", response_model=PairLossResponse)
def measure_pair_loss(request: PairLossRequest) -> PairLossResponse:
    """The same function the training uses, on two positions a reader chose.

    Two positions of different widths are refused here rather than by the
    request model, so the refusal reads in the page's own words.
    """
    if len(request.first) != len(request.second):
        raise HTTPException(
            status_code=422,
            detail=(
                "two positions can only be compared when they have the same "
                f"number of coordinates, and these have {len(request.first)} "
                f"and {len(request.second)}"
            ),
        )
    first = np.asarray([request.first], dtype=np.float64)
    second = np.asarray([request.second], dtype=np.float64)
    actual = pair_loss(first, second, np.array([request.same_kind]), request.margin)
    as_same = pair_loss(first, second, np.array([True]), request.margin)
    as_different = pair_loss(first, second, np.array([False]), request.margin)
    distance = float(actual.distances[0])
    return PairLossResponse(
        distance=distance,
        loss=float(actual.losses[0]),
        loss_if_same=float(as_same.losses[0]),
        loss_if_different=float(as_different.losses[0]),
        beyond_margin=distance >= request.margin,
        slope_on_first=[float(value) for value in actual.slope_first[0]],
        slope_on_second=[float(value) for value in actual.slope_second[0]],
        direction_defined=not (
            distance == 0.0 and not request.same_kind and request.margin > 0.0
        ),
    )


# --------------------------------------------------------------------------
# One set of weights, two towers: the sum against a finite difference.
# --------------------------------------------------------------------------


class CheckedPair(BaseModel):
    same_kind: bool
    first_kind: str
    second_kind: str
    distance: float
    loss: float


class SharedWeightRow(BaseModel):
    """One parameter's slope from each tower, their sum, and the measured slope.

    ``finite_difference`` nudges the parameter both ways by ``nudge`` and
    measures the pair loss again, which knows nothing about towers.
    """

    layer: str
    parameter: str
    first_tower: float
    second_tower: float
    both: float
    finite_difference: float
    gap_to_both: float
    gap_to_first_tower: float


class SharedWeightsResponse(BaseModel):
    margin: float
    nudge: float
    pairs: list[CheckedPair]
    mean_loss: float
    rows: list[SharedWeightRow]
    largest_gap_to_both: float
    position_biases_cancel_exactly: bool
    largest_position_bias_slope_one_tower: float


def _layer_names(stack: LayerStack) -> dict[int, str]:
    names: dict[int, str] = {}
    ordinals = ("first", "second", "third", "fourth")
    convolutions = [
        position for position, layer in enumerate(stack) if isinstance(layer, Conv2d)
    ]
    for order, position in enumerate(convolutions):
        names[position] = f"{ordinals[min(order, 3)]} convolution"
    for position, layer in enumerate(stack):
        if isinstance(layer, DenseLayer):
            names[position] = (
                "position layer" if position == len(stack) - 1 else "hidden layer"
            )
    return names


def _parameter_label(
    stack: LayerStack, position: int, index: tuple[int, int] | None, bias: int | None
) -> str:
    if bias is not None:
        return f"unit {bias + 1}, its bias"
    if index is None:
        return "no parameter"
    row, column = index
    if isinstance(stack[position], Conv2d):
        return f"filter {row + 1}, weight {column + 1}"
    return f"unit {row + 1}, input {column + 1}"


def _nudged(
    stack: LayerStack,
    template: list[LayerGradient | None],
    position: int,
    index: tuple[int, int] | None,
    bias: int | None,
    amount: float,
) -> LayerStack:
    """``stack`` with one parameter moved by ``amount``, through the ordinary step.

    A gradient that is zero everywhere except a one at the chosen parameter,
    stepped by a rate of ``-amount``, adds exactly ``amount`` to that one
    parameter, since a step subtracts the rate times the gradient.
    """
    gradients: list[LayerGradient | None] = []
    for layer_position, gradient in enumerate(template):
        if gradient is None:
            gradients.append(None)
            continue
        weights = np.zeros_like(np.asarray(gradient.weights))
        biases = np.zeros_like(np.asarray(gradient.biases))
        if layer_position == position:
            if index is not None:
                weights[index] = 1.0
            elif bias is not None:
                biases[bias] = 1.0
        gradients.append(LayerGradient(weights=weights, biases=biases))
    return stack.stepped_by(BackwardPass(loss=0.0, gradients=gradients), -amount)


def _mean_pair_loss(
    stack: LayerStack,
    first_pictures: np.ndarray,
    second_pictures: np.ndarray,
    same: np.ndarray,
) -> float:
    return pair_loss(
        _positions(stack, first_pictures),
        _positions(stack, second_pictures),
        same,
        MARGIN,
    ).mean


def _entry(
    gradients: list[LayerGradient | None],
    position: int,
    index: tuple[int, int] | None,
    bias: int | None,
) -> float:
    gradient = gradients[position]
    if gradient is None:
        return 0.0
    if index is not None:
        return float(np.asarray(gradient.weights)[index])
    return float(np.asarray(gradient.biases)[bias])


@lru_cache(maxsize=1)
def _shared_weights() -> SharedWeightsResponse:
    collection = _training_without(HELD_BACK)
    rows_of = [
        np.flatnonzero(collection.labels == kind) for kind in range(collection.n_kinds)
    ]
    first_rows = np.array([rows_of[0][0], rows_of[1][0], rows_of[0][1], rows_of[2][0]])
    second_rows = np.array([rows_of[0][2], rows_of[1][3], rows_of[2][1], rows_of[1][1]])
    same = collection.labels[first_rows] == collection.labels[second_rows]
    first_pictures = collection.pictures[first_rows]
    second_pictures = collection.pictures[second_rows]

    stack = built_network(NetworkDesign(weight_seed=0), POSITION_WIDTH)
    towers = two_tower_gradients(stack, first_pictures, second_pictures, same, MARGIN)
    names = _layer_names(stack)
    last = len(stack) - 1

    # The largest summed slope in every layer that learns, so that every row of
    # the table tests a weight that actually moves, and then the position
    # layer's largest single-tower bias slope, which the sum cancels.
    checks: list[tuple[int, tuple[int, int] | None, int | None]] = []
    for position, gradient in enumerate(towers.both):
        if gradient is None:
            continue
        magnitudes = np.abs(np.asarray(gradient.weights))
        row, column = np.unravel_index(int(np.argmax(magnitudes)), magnitudes.shape)
        checks.append((position, (int(row), int(column)), None))
    position_bias = towers.first_tower[last]
    largest_bias_slope = 0.0
    if position_bias is not None:
        bias_magnitudes = np.abs(np.asarray(position_bias.biases))
        largest_bias_slope = float(bias_magnitudes.max())
        checks.append((last, None, int(np.argmax(bias_magnitudes))))

    rows = []
    for position, index, bias in checks:
        up = _mean_pair_loss(
            _nudged(stack, towers.both, position, index, bias, NUDGE),
            first_pictures,
            second_pictures,
            same,
        )
        down = _mean_pair_loss(
            _nudged(stack, towers.both, position, index, bias, -NUDGE),
            first_pictures,
            second_pictures,
            same,
        )
        difference = (up - down) / (2.0 * NUDGE)
        both = _entry(towers.both, position, index, bias)
        first_tower = _entry(towers.first_tower, position, index, bias)
        rows.append(
            SharedWeightRow(
                layer=names[position],
                parameter=_parameter_label(stack, position, index, bias),
                first_tower=first_tower,
                second_tower=_entry(towers.second_tower, position, index, bias),
                both=both,
                finite_difference=difference,
                gap_to_both=abs(both - difference),
                gap_to_first_tower=abs(first_tower - difference),
            )
        )

    summed_biases = towers.both[last]
    return SharedWeightsResponse(
        margin=MARGIN,
        nudge=NUDGE,
        pairs=[
            CheckedPair(
                same_kind=bool(same[position]),
                first_kind=collection.kind_names[
                    int(collection.labels[first_rows[position]])
                ],
                second_kind=collection.kind_names[
                    int(collection.labels[second_rows[position]])
                ],
                distance=float(towers.loss.distances[position]),
                loss=float(towers.loss.losses[position]),
            )
            for position in range(len(first_rows))
        ],
        mean_loss=towers.loss.mean,
        rows=rows,
        largest_gap_to_both=max(row.gap_to_both for row in rows),
        position_biases_cancel_exactly=bool(
            summed_biases is not None
            and np.all(np.asarray(summed_biases.biases) == 0.0)
        ),
        largest_position_bias_slope_one_tower=largest_bias_slope,
    )


@router.get("/shared-weights", response_model=SharedWeightsResponse)
def shared_weights() -> SharedWeightsResponse:
    """Four pairs through the untrained tower: each tower's slope, the sum, the check."""
    return _shared_weights()


# --------------------------------------------------------------------------
# The margin.
# --------------------------------------------------------------------------

RateRule = Literal["scaled", "fixed"]


class MarginResponse(BaseModel):
    """One tower trained at one margin, and where its pictures ended up.

    ``rate_rule`` is ``"fixed"`` for the default rate at every margin and
    ``"scaled"`` for the default rate divided by the margin squared, which
    keeps the step the lower layers take the same size as at margin one. The
    distances are on held-out pictures of the three training kinds.
    """

    margin: float
    rate: float
    rate_rule: RateRule
    seed: int
    losses: list[float]
    spread_epochs: list[int]
    spreads: list[float]
    start_spread: float
    final_loss: float | None
    diverged: bool
    mean_within: float
    mean_across: float
    beyond_margin: float
    kinds: list[KindFigure]


@lru_cache(maxsize=24)
def _margin_run(
    margin: float, rate: float, rate_rule: RateRule, seed: int
) -> MarginResponse:
    collection = _training_without(HELD_BACK)
    if margin == MARGIN:
        # The default margin at the default rate is the main study's own tower,
        # already trained, whichever rule was asked for.
        run = _study(HELD_BACK, seed).pair_run
    else:
        run = _trained_on_pairs(collection, margin=margin, rate=rate, seed=seed)
    evaluation = _evaluation()
    held = _held_out_without(HELD_BACK)
    against = _against_margin(_positions(run.stack, held.pictures), held.labels, margin)
    return MarginResponse(
        margin=margin,
        rate=rate,
        rate_rule=rate_rule,
        seed=seed,
        losses=list(run.losses),
        spread_epochs=list(run.spread_epochs),
        spreads=list(run.spreads),
        start_spread=run.start_spread,
        final_loss=run.losses[-1] if run.losses else None,
        diverged=run.diverged,
        mean_within=against.mean_within,
        mean_across=against.mean_across,
        beyond_margin=against.beyond_margin,
        kinds=_figures(
            _readings(_positions(run.stack, evaluation.pictures), evaluation),
            _training_kinds(HELD_BACK),
        ),
    )


@router.get("/margin", response_model=MarginResponse)
def margin_run(
    margin: Annotated[float, Query()] = MARGIN,
    rate_rule: Annotated[str, Query()] = "scaled",
    seed: Annotated[int, Query()] = 0,
) -> MarginResponse:
    """Train at one of the swept margins, with the rate held or scaled.

    The three values are checked here rather than by the request model, so a
    refusal reads in the page's own words and names no parameter.
    """
    if margin not in MARGINS:
        raise HTTPException(
            status_code=422,
            detail="the margins measured here are 0.5, 1, 2 and 4",
        )
    if rate_rule not in ("scaled", "fixed"):
        raise HTTPException(
            status_code=422,
            detail=(
                "the rate is either held where it is or divided by the square of "
                "the margin"
            ),
        )
    if seed not in MARGIN_SEEDS:
        raise HTTPException(
            status_code=422,
            detail="the sweep is run from weight seeds 0, 1 and 2",
        )
    rule: RateRule = "scaled" if rate_rule == "scaled" else "fixed"
    if margin == MARGIN:
        # One run, whichever rule was asked for, since the two agree here.
        rule = "scaled"
    rate = RATE / margin**2 if rule == "scaled" else RATE
    return _margin_run(float(margin), float(rate), rule, seed)


# --------------------------------------------------------------------------
# Without different-kind pairs.
# --------------------------------------------------------------------------


class CollapseResponse(BaseModel):
    """The same tower trained with and without the push, from the same start.

    The spreads are the mean distance between every two of the 300 evaluation
    positions after each epoch in ``spread_epochs``; ``start_spread`` is before
    the first step, which the two runs share because they share their
    starting weights.
    """

    margin: float
    start_spread: float
    spread_epochs: list[int]
    spreads_without: list[float]
    spreads_with: list[float]
    losses_without: list[float]
    losses_with: list[float]
    kinds_without: list[KindFigure]
    kinds_with: list[KindFigure]


@lru_cache(maxsize=1)
def _collapse() -> CollapseResponse:
    study = _study(HELD_BACK, 0)
    collection = _training_without(HELD_BACK)
    pulled = _trained_on_pairs(
        collection, margin=MARGIN, rate=RATE, seed=0, with_different_kinds=False
    )
    evaluation = _evaluation()
    training_kinds = _training_kinds(HELD_BACK)
    return CollapseResponse(
        margin=MARGIN,
        start_spread=pulled.start_spread,
        spread_epochs=list(pulled.spread_epochs),
        spreads_without=list(pulled.spreads),
        spreads_with=list(study.pair_run.spreads),
        losses_without=list(pulled.losses),
        losses_with=list(study.pair_run.losses),
        kinds_without=_figures(
            _readings(_positions(pulled.stack, evaluation.pictures), evaluation),
            training_kinds,
        ),
        kinds_with=_figures(
            _readings(study.pair_positions, evaluation), training_kinds
        ),
    )


@router.get("/without-different-kinds", response_model=CollapseResponse)
def without_different_kinds() -> CollapseResponse:
    """Same-kind pairs only, against the ordinary run, both from weight seed 0."""
    return _collapse()


# --------------------------------------------------------------------------
# The cost.
# --------------------------------------------------------------------------


class GrowthRow(BaseModel):
    """How many pictures, and how many pairs they make."""

    pictures: int
    pairs: int


class CostResponse(BaseModel):
    """What the pairs cost and how few of them were still teaching at the end.

    The two times are wall-clock seconds on whatever machine answered, so they
    differ from visit to visit; the counts do not.
    """

    n_training_pictures: int
    all_pairs: int
    same_kind_pairs: int
    different_kind_pairs: int
    batch_pictures: int
    pairs_per_batch: int
    same_kind_per_batch: int
    steps_per_epoch: int
    epochs: int
    pair_evaluations: int
    distinct_pairs_met: int
    share_of_pairs_met: float
    margin: float
    beyond_margin_training: float
    beyond_margin_held_out: float
    pair_seconds: float
    classifier_seconds: float
    growth: list[GrowthRow]


@lru_cache(maxsize=1)
def _cost() -> CostResponse:
    study = _study(HELD_BACK, 0)
    collection = _training_without(HELD_BACK)
    held = _held_out_without(HELD_BACK)
    count = len(collection)
    sizes = np.bincount(collection.labels)
    same_kind = int(sum(int(size) * (int(size) - 1) // 2 for size in sizes))
    all_pairs = count * (count - 1) // 2
    batch = PER_KIND_IN_BATCH * collection.n_kinds
    run = study.pair_run
    return CostResponse(
        n_training_pictures=count,
        all_pairs=all_pairs,
        same_kind_pairs=same_kind,
        different_kind_pairs=all_pairs - same_kind,
        batch_pictures=batch,
        pairs_per_batch=batch * (batch - 1) // 2,
        same_kind_per_batch=collection.n_kinds
        * PER_KIND_IN_BATCH
        * (PER_KIND_IN_BATCH - 1)
        // 2,
        steps_per_epoch=run.steps_per_epoch,
        epochs=EPOCHS,
        pair_evaluations=run.pair_evaluations,
        distinct_pairs_met=run.distinct_pairs_met,
        share_of_pairs_met=run.distinct_pairs_met / all_pairs,
        margin=MARGIN,
        beyond_margin_training=_against_margin(
            _positions(run.stack, collection.pictures), collection.labels, MARGIN
        ).beyond_margin,
        beyond_margin_held_out=_against_margin(
            _positions(run.stack, held.pictures), held.labels, MARGIN
        ).beyond_margin,
        pair_seconds=run.training_seconds,
        classifier_seconds=study.classifier_seconds,
        growth=[
            GrowthRow(pictures=pictures, pairs=pictures * (pictures - 1) // 2)
            for pictures in (count, 1_000, 10_000, 100_000, 1_000_000)
        ],
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """How the pairs grow, how many training met, and how many still taught."""
    return _cost()


# --------------------------------------------------------------------------
# Sixteen numbers rather than eight.
# --------------------------------------------------------------------------

WIDER = 16
"""The width of the classifier's vector, which the wider tower is given."""


class WiderResponse(BaseModel):
    """The main study's tower retrained answering with sixteen numbers.

    Same starting seed, same batches, same margin and rate; only the width of
    the answer changes, so any difference from the eight-wide tower is the
    width's. ``narrow`` is the main study's own tower for comparison.
    """

    held_back: str
    seed: int
    narrow_width: int
    wide_width: int
    narrow: list[KindFigure]
    wide: list[KindFigure]
    wide_final_loss: float


@lru_cache(maxsize=1)
def _wider() -> WiderResponse:
    study = _study(HELD_BACK, 0)
    evaluation = _evaluation()
    training_kinds = _training_kinds(HELD_BACK)
    run = _trained_on_pairs(
        _training_without(HELD_BACK), margin=MARGIN, rate=RATE, seed=0, width=WIDER
    )
    return WiderResponse(
        held_back=HELD_BACK,
        seed=0,
        narrow_width=POSITION_WIDTH,
        wide_width=WIDER,
        narrow=_figures(_readings(study.pair_positions, evaluation), training_kinds),
        wide=_figures(
            _readings(_positions(run.stack, evaluation.pictures), evaluation),
            training_kinds,
        ),
        wide_final_loss=run.losses[-1],
    )


@router.get("/wider-positions", response_model=WiderResponse)
def wider_positions() -> WiderResponse:
    """The main study's tower answering with sixteen numbers instead of eight."""
    return _wider()
