"""Normalisation layers, for the page that standardises inside a model.

Four endpoints, and the first is the playground's. It takes one block of
numbers, rows down and features across, pushes it through one of the
library's three normalising layers, ``BatchNormalization``,
``LayerNormalization`` or ``RMSNormalization``, chosen by a closed enum, and
returns what the layer answered together with the statistics it divided by,
one per feature for the batch layer and one per row for the other two. The
page needs the statistics beside the answers because the whole lesson is which
axis they were read along, and a widget that only showed the normalised
numbers would hide the one thing that differs between the three layers. The
request may also carry a scale and a shift per feature, which is how the page
shows that the learned pair can undo the standardisation exactly.

For the batch layer the first endpoint also returns the same rows passed while
predicting. A batch layer standardises by the batch in front of it while
training and by running figures while predicting, and the page wants to show
that the two answers differ and say why. The running figures start at a mean
of zero and a variance of one and move toward the batch only when the layer is
stepped, so the endpoint takes exactly one step. The cheapest honest step is a
backward pass with a slope of zero arriving, which the layer turns into a
``BatchStatistics`` whose scale and shift slopes are zero and whose batch mean
and variance are the ones the forward pass measured. Stepping by that moves
neither the scale nor the shift, since there is nothing to multiply, and folds
the batch a tenth of the way into the running figures under the library's
default momentum of 0.9. The predicting block is the stepped layer's answer to
the same rows.

The second endpoint runs the backward pass. A normalising layer's mean and
deviation are functions of every value in the band, so one input reaches the
loss by three routes rather than one, and the tempting answer, the arriving
slope over the deviation, has the right shape and trains on a gradient that is
not the gradient of the loss. The endpoint returns the library's passed-down
block, that naive block computed here for comparison, and a central finite
difference of the objective ``sum(arriving * outputs)`` taken by nudging every
input through the library's own forward pass, which is the oracle the
library's spec uses. It also sums the passed-down block along each band,
because shifting a whole band by a constant cannot change what a centring
layer answers, so those sums are zero exactly for the true gradient and are
not for the naive one.

The third endpoint follows the running figures over many steps of the same
batch, reporting after each step what they are and how far the predicting
answer still is from the training one, and it also passes the first row of
the block through on its own, which is the case the batch layer cannot
standardise while training: one row has no spread, so it comes back as zeros.

The fourth is an experiment on the classification pages' tangled crowd, the
same twenty-five people and the same sixteen-unit rectified network the
dropout page trains. It records how the block the output layer reads drifts
as the hidden layer learns, with and without a batch layer between them, then
trains the network under five arrangements, no normalisation, a batch layer,
a layer-normalising layer, an RMS layer and a weight-normalised hidden layer,
held out fold by fold across several seeds, and returns training against
held-out accuracy for every combination and the loss curve of one seed for
each. It closes with the reparameterisation check on a small fixture: a
weight-normalised layer answers exactly as the dense layer holding its
effective weights, its gradient in the new coordinates agrees with a finite
difference, the form that forgets the projection does not, and one step at
the same rate lands the two layers in different places. The library records
no history of its own, so this module owns the loop, as the training page's
module does, and nothing here computes a gradient or a step.

Every layer keeps its epsilon inside the square root, so a deviation of 3
comes back as the root of 9.00001, and the page quotes that figure rather than
rounding it away. The variance is the biased one, dividing by the number of
rows, which is why a four-row block can have a whole-number deviation and a
three-row block with any spread in it never can.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BatchNormalization,
    BinaryCrossEntropy,
    DenseLayer,
    HyperbolicTangent,
    Identity,
    LayerNormalization,
    LayerStack,
    Neuron,
    PassPurpose,
    RectifiedLinear,
    RMSNormalization,
)
from oop_ml.core.network.activation import Activation
from oop_ml.core.network.layer import Layer
from oop_ml.core.network.weight_normalisation import WeightNormalization
from pydantic import BaseModel, Field, model_validator

from oop_ml_api.concepts.dropout import TANGLED_CROWD
from oop_ml_api.concepts.two_feature_classification import LabelledPoint
from oop_ml_api.limits import MAX_MAGNITUDE, MIN_POINTS

router = APIRouter(tags=["normalisation-layers"])

MIN_ROWS = MIN_POINTS
"""The fewest rows a batch can be. One row has no spread to divide by."""

MAX_ROWS = 20
"""The most rows the widget's grid may hold."""

MIN_FEATURES = 2
"""The fewest features a row can hold. One value has no spread within the row."""

MAX_FEATURES = 6
"""The most features the widget's grid may hold."""

ZERO_LEARNING_RATE = 0.0
"""The rate the one step is taken at.

The arriving slope is zero, so the scale and shift slopes are zero and the
rate has nothing to multiply. It is passed as zero to say so, and the step
still updates the running figures, which do not read it.
"""

MAX_RUNNING_STEPS = 300
"""The most steps the running figures may be followed for.

At the default momentum the predicting answer is within a thousandth of the
training one by the eightieth step, so three hundred is well past anything the
page needs and keeps a crafted request small.
"""

FINITE_DIFFERENCE_STEP = 1e-5
"""How far each input is nudged when the backward pass is checked.

Measured on the page's worked block, the library's passed-down block agrees
with the central difference to about 2e-10 at this step, the best of the
three sizes tried, which is where the rounding of a difference of two nearby
totals and the curvature of the forward pass are about equal.
"""

HIDDEN_WIDTH = 16
"""How many rectified units the crowd network's hidden layer holds, the
dropout page's width, so the two pages train the same network."""

EXPERIMENT_EPOCHS = 400
"""How many full-batch steps each training run takes, the dropout page's."""

LEARNING_RATE = 1.0
"""The stride, the dropout page's, held the same under every arrangement so
that the arrangements are compared and not the strides."""

EXPERIMENT_SEEDS = (1, 2, 3, 4, 5)
"""The seeds the sweep runs under. Each seeds the weights and the fold deal
together, so one seed is one repeatable experiment."""

N_FOLDS = 5
"""Five folds of the twenty-five people, so every person is held out once
and the held-out accuracy is counted over all twenty-five."""

CURVE_SEED = 1
"""The seed whose whole-crowd loss curves and drift are recorded."""

DRIFT_EVERY = 4
"""Record the drifting statistics every this many epochs, so a curve holds a
hundred points and the summaries are still taken over every epoch."""

LOSS_THRESHOLD = 0.3
"""The loss a curve is timed against, the epoch it first falls below this."""

REPARAMETERISATION_RATE = 0.1
"""The stride both layers take in the reparameterisation check."""

REPARAMETERISATION_DIRECTIONS: tuple[tuple[float, float], ...] = (
    (0.6, -0.8),
    (1.5, 2.0),
    (-0.3, 0.4),
)
"""Three neurons' stored directions, two inputs each, whose lengths are 1,
2.5 and 0.5, so the default magnitudes come out whole."""

REPARAMETERISATION_BIASES: tuple[float, ...] = (0.1, -0.2, 0.3)

REPARAMETERISATION_ARRIVING: tuple[tuple[float, float, float], ...] = (
    (1.0, 2.0, 3.0),
    (4.0, 5.0, 6.0),
    (7.0, 8.0, 9.0),
    (1.0, 0.0, -1.0),
    (-2.0, 1.0, 0.5),
)
"""The slope arriving at each of the five rows' three outputs, graded so no
two rows pull alike."""

REPARAMETERISATION_PEOPLE = 5
"""How many of the crowd the reparameterisation fixture reads, the first
five, in standard units of the whole crowd."""


class NormalisationLayer(StrEnum):
    """The three layers, named by which axis their statistics run along."""

    BATCH = "batch"
    LAYER = "layer"
    RMS = "rms"


class Arrangement(StrEnum):
    """What sits between the crowd network's hidden layer and its output layer.

    ``WEIGHT`` is the odd one out, since weight normalisation is not a layer
    between the two but a different way of storing the hidden layer's own
    weights, so under that arrangement the hidden layer is replaced and
    nothing is inserted.
    """

    NONE = "none"
    BATCH = "batch"
    LAYER = "layer"
    RMS = "rms"
    WEIGHT = "weight"


BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]

BlockRow = Annotated[
    list[BoundedValue], Field(min_length=MIN_FEATURES, max_length=MAX_FEATURES)
]

PerFeature = Annotated[
    list[BoundedValue], Field(min_length=MIN_FEATURES, max_length=MAX_FEATURES)
]


def _check_rectangle(rows: list[list[float]]) -> int:
    """The block's width, refusing a ragged block in words a learner can read."""
    width = len(rows[0])
    for position, row in enumerate(rows):
        if len(row) != width:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"row {position + 1} holds {len(row)} values where the first "
                    f"row holds {width}, and a batch has to be a rectangle before "
                    "any statistic can run along it"
                ),
            )
    return width


def _check_per_feature(values: list[float] | None, width: int, role: str) -> None:
    if values is not None and len(values) != width:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the {role} is one value per feature, so {width} values for this "
                f"block, got {len(values)}"
            ),
        )


class NormaliseRequest(BaseModel):
    """The block, rows down and features across, and which layer to push it through.

    ``scale`` and ``shift`` are the learned pair, one of each per feature.
    Left out, the layer starts at a scale of one and a shift of zero, so its
    answer is the standardised block exactly. Supplied, the answer is
    ``scale * standardised + shift``, which is how the page shows that setting
    the scale to a band's deviation and the shift to its mean recovers the raw
    numbers.
    """

    rows: list[BlockRow] = Field(min_length=MIN_ROWS, max_length=MAX_ROWS)
    layer: NormalisationLayer
    scale: PerFeature | None = None
    shift: PerFeature | None = None

    model_config = {"extra": "forbid"}


class ReducedStatistic(BaseModel):
    """What one band of the block was standardised by.

    Under the batch layer there is one of these per feature, read down the
    rows. Under the layer and RMS layers there is one per row, read across the
    features. ``mean`` is what was subtracted, and is ``None`` for the RMS
    layer, which subtracts nothing. ``deviation`` is the figure the layer
    divided by, the root of the spread plus epsilon, so for the RMS layer it
    is the row's root mean square.
    """

    mean: float | None
    deviation: float


class NormaliseResponse(BaseModel):
    """The block after the layer, and the statistics the layer used.

    ``standardised`` is the block after the centring and dividing and before
    the scale and shift. ``normalised`` is the layer's answer, which at a
    scale of one and a shift of zero is the same block. ``largest_gap_to_raw``
    is the widest distance between the answer and the raw block, which is
    what the page reads when the scale and shift are set to undo the
    standardisation. ``as_predicting`` and ``predicting_statistics`` are
    present for the batch layer only, the same rows answered by the running
    figures after one training pass and one step, and are ``None`` for the two
    layers that have no such distinction.
    """

    layer: NormalisationLayer
    standardised: list[list[float]]
    normalised: list[list[float]]
    largest_gap_to_raw: float
    statistics: list[ReducedStatistic]
    as_predicting: list[list[float]] | None
    predicting_statistics: list[ReducedStatistic] | None


@router.post(
    "/concepts/normalisation-layers/normalise", response_model=NormaliseResponse
)
def normalise(request: NormaliseRequest) -> NormaliseResponse:
    """Push the block through the chosen layer and report what it divided by."""
    width = _check_rectangle(request.rows)
    _check_per_feature(request.scale, width, "scale")
    _check_per_feature(request.shift, width, "shift")

    block = np.asarray(request.rows, dtype=np.float64)
    scale = None if request.scale is None else np.asarray(request.scale)
    shift = None if request.shift is None else np.asarray(request.shift)

    if request.layer == NormalisationLayer.BATCH:
        return _batch_normalised(block, scale, shift)
    if request.layer == NormalisationLayer.LAYER:
        layer = LayerNormalization(n_features=width, scale=scale, shift=shift)
        return _row_normalised(block, layer)
    layer = RMSNormalization(n_features=width, scale=scale, shift=shift)
    return _row_normalised(block, layer)


def _batch_normalised(
    block: np.ndarray, scale: np.ndarray | None, shift: np.ndarray | None
) -> NormaliseResponse:
    """The batch layer's answer while training, and again after one step."""
    layer = BatchNormalization(n_features=block.shape[1], scale=scale, shift=shift)
    training = layer.respond_to(block, purpose=PassPurpose.TRAINING)

    # The one step. A zero slope arriving means the scale and shift slopes are
    # zero and only the running figures move, a tenth of the way toward the
    # batch the forward pass measured.
    correction = layer.correction_for(training, np.zeros_like(training.outputs))
    stepped = layer.stepped_by(correction.gradient, learning_rate=ZERO_LEARNING_RATE)
    predicting = stepped.respond_to(block, purpose=PassPurpose.PREDICTING)
    running_deviation = np.sqrt(stepped.running_variance + stepped.epsilon)

    return NormaliseResponse(
        layer=NormalisationLayer.BATCH,
        standardised=_as_lists(training.scores),
        normalised=_as_lists(training.outputs),
        largest_gap_to_raw=float(np.max(np.abs(training.outputs - block))),
        statistics=_statistics(training.batch_mean, training.deviation),
        as_predicting=_as_lists(predicting.outputs),
        predicting_statistics=_statistics(stepped.running_mean, running_deviation),
    )


def _row_deviations(
    block: np.ndarray, layer: LayerNormalization | RMSNormalization
) -> tuple[np.ndarray | None, np.ndarray]:
    """Each row's centre and the deviation the layer divided it by.

    The row layers carry no statistics on their response, since nothing needs
    them afterwards, so the figures are read off the block here by the same
    definition the layer uses, the biased spread about the row's centre plus
    epsilon, under the root. The test module checks that dividing by them
    reproduces the layer's answer to the last bit.
    """
    if layer.centres:
        row_means = block.mean(axis=1)
        centred = block - row_means[:, None]
    else:
        row_means = None
        centred = block
    return row_means, np.sqrt((centred**2).mean(axis=1) + layer.epsilon)


def _row_normalised(
    block: np.ndarray, layer: LayerNormalization | RMSNormalization
) -> NormaliseResponse:
    """A within-row layer's answer, with each row's own centre and deviation."""
    response = layer.respond_to(block)
    row_means, row_deviations = _row_deviations(block, layer)

    return NormaliseResponse(
        layer=(NormalisationLayer.LAYER if layer.centres else NormalisationLayer.RMS),
        standardised=_as_lists(response.scores),
        normalised=_as_lists(response.outputs),
        largest_gap_to_raw=float(np.max(np.abs(response.outputs - block))),
        statistics=_statistics(row_means, row_deviations),
        as_predicting=None,
        predicting_statistics=None,
    )


def _statistics(
    means: np.ndarray | None, deviations: np.ndarray
) -> list[ReducedStatistic]:
    """One statistic per band, the means absent when the layer did not centre."""
    if means is None:
        return [
            ReducedStatistic(mean=None, deviation=float(value)) for value in deviations
        ]
    return [
        ReducedStatistic(mean=float(mean), deviation=float(deviation))
        for mean, deviation in zip(means, deviations, strict=True)
    ]


def _as_lists(block: np.ndarray) -> list[list[float]]:
    """A two-dimensional array as plain floats, rows down and features across."""
    return [[float(value) for value in row] for row in block]


def _as_floats(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


class BackwardRequest(BaseModel):
    """The block, the layer, and the slope arriving at every output.

    ``arriving`` is the slope of some loss at the layer's outputs, arranged
    exactly as the block is. Left out, it counts up from one across the block,
    row by row, so that no two entries pull alike; a block of ones would make
    the true passed-down block vanish under a centring layer, since the mean
    route cancels the direct one exactly, and there would be nothing to see.
    """

    rows: list[BlockRow] = Field(min_length=MIN_ROWS, max_length=MAX_ROWS)
    layer: NormalisationLayer
    arriving: list[list[BoundedValue]] | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _arriving_matches_the_block(self) -> BackwardRequest:
        if self.arriving is not None and (
            len(self.arriving) != len(self.rows)
            or any(
                len(slope_row) != len(row)
                for slope_row, row in zip(self.arriving, self.rows, strict=True)
            )
        ):
            raise ValueError(
                "the arriving block must be arranged exactly as the rows are"
            )
        return self


class BackwardResponse(BaseModel):
    """The backward pass through one normalising layer, checked three ways.

    ``passed_down`` is the library's answer, the slope of the objective
    ``sum(arriving * outputs)`` at every input. ``naive`` treats the band's
    centre and deviation as constants and answers the arriving slope over the
    deviation, computed here for comparison. ``finite_difference`` nudges every
    input through the library's forward pass and differences the objective.
    ``largest_gap`` and ``largest_naive_gap`` are how far each of the first two
    is from the third, and ``largest_true_slope`` is the biggest entry of the
    third, so the gaps can be read against the size of what they are gaps in.

    ``band_sums`` sums the passed-down block along each band, down each
    column for the batch layer and along each row for the other two, with
    ``naive_band_sums`` the same on the naive block; ``largest_band_sum`` and
    ``largest_naive_band_sum`` are their largest magnitudes. For a centring
    layer the true sums are zero exactly. ``scale_slope`` and ``shift_slope``
    are the two parameter slopes, which do not depend on which passed-down
    form is used. ``deviations`` is what each band was divided by.
    """

    layer: NormalisationLayer
    arriving: list[list[float]]
    passed_down: list[list[float]]
    naive: list[list[float]]
    finite_difference: list[list[float]]
    largest_gap: float
    largest_naive_gap: float
    largest_true_slope: float
    band_sums: list[float]
    naive_band_sums: list[float]
    largest_band_sum: float
    largest_naive_band_sum: float
    scale_slope: list[float]
    shift_slope: list[float]
    deviations: list[float]


def _counting_block(n_rows: int, width: int) -> np.ndarray:
    return np.arange(1, n_rows * width + 1, dtype=np.float64).reshape(n_rows, width)


def _normalising_layer(
    which: NormalisationLayer, width: int
) -> BatchNormalization | LayerNormalization | RMSNormalization:
    if which == NormalisationLayer.BATCH:
        return BatchNormalization(n_features=width)
    if which == NormalisationLayer.LAYER:
        return LayerNormalization(n_features=width)
    return RMSNormalization(n_features=width)


def _objective(layer: Layer, block: np.ndarray, arriving: np.ndarray) -> float:
    """``sum(arriving * outputs)`` through a training pass of the layer."""
    outputs = layer.respond_to(block, purpose=PassPurpose.TRAINING).outputs
    return float(np.sum(arriving * outputs))


@router.post("/concepts/normalisation-layers/backward", response_model=BackwardResponse)
def backward(request: BackwardRequest) -> BackwardResponse:
    """Run the backward pass through the chosen layer and check it against a finite difference."""
    width = _check_rectangle(request.rows)
    block = np.asarray(request.rows, dtype=np.float64)
    n_rows = block.shape[0]
    arriving = (
        _counting_block(n_rows, width)
        if request.arriving is None
        else np.asarray(request.arriving, dtype=np.float64)
    )

    layer = _normalising_layer(request.layer, width)
    response = layer.respond_to(block, purpose=PassPurpose.TRAINING)
    correction = layer.correction_for(response, arriving)
    passed_down = np.asarray(correction.passed_down, dtype=np.float64)
    if correction.gradient is None:
        raise TypeError("a normalising layer has a scale and a shift to report")

    # The naive block, the arriving slope over the deviation as though the
    # centre and the deviation were constants. Under the batch layer the
    # deviation is one per feature; under the row layers one per row.
    if isinstance(layer, BatchNormalization):
        deviations = np.asarray(response.deviation, dtype=np.float64)
        naive = arriving * layer.scale / deviations
        band_axis = 0
    else:
        _, deviations = _row_deviations(block, layer)
        naive = arriving * layer.scale / deviations[:, None]
        band_axis = 1

    # The oracle. Nudge one input each way, rerun the forward pass, and
    # difference the objective.
    finite_difference = np.empty_like(block)
    for row_index in range(n_rows):
        for column_index in range(width):
            up = block.copy()
            up[row_index, column_index] += FINITE_DIFFERENCE_STEP
            down = block.copy()
            down[row_index, column_index] -= FINITE_DIFFERENCE_STEP
            finite_difference[row_index, column_index] = (
                _objective(layer, up, arriving) - _objective(layer, down, arriving)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)

    band_sums = passed_down.sum(axis=band_axis)
    naive_band_sums = naive.sum(axis=band_axis)

    return BackwardResponse(
        layer=request.layer,
        arriving=_as_lists(arriving),
        passed_down=_as_lists(passed_down),
        naive=_as_lists(naive),
        finite_difference=_as_lists(finite_difference),
        largest_gap=float(np.max(np.abs(finite_difference - passed_down))),
        largest_naive_gap=float(np.max(np.abs(finite_difference - naive))),
        largest_true_slope=float(np.max(np.abs(finite_difference))),
        band_sums=_as_floats(band_sums),
        naive_band_sums=_as_floats(naive_band_sums),
        largest_band_sum=float(np.max(np.abs(band_sums))),
        largest_naive_band_sum=float(np.max(np.abs(naive_band_sums))),
        scale_slope=_as_floats(correction.gradient.weights[:, 0]),
        shift_slope=_as_floats(correction.gradient.biases),
        deviations=_as_floats(deviations),
    )


class RunningFiguresRequest(BaseModel):
    """The block, and how many steps of the same batch to follow the figures for."""

    rows: list[BlockRow] = Field(min_length=MIN_ROWS, max_length=MAX_ROWS)
    n_steps: int = Field(default=100, ge=1, le=MAX_RUNNING_STEPS)

    model_config = {"extra": "forbid"}


class RunningStep(BaseModel):
    """The running figures after one more step of the same batch.

    ``largest_gap`` is the widest distance, over every cell, between the
    block answered by those running figures and the block answered by the
    batch's own statistics, which is the distance between predicting and
    training that step.
    """

    step: int
    running_mean: list[float]
    running_deviation: list[float]
    largest_gap: float


class LoneRow(BaseModel):
    """The first row of the block, passed through on its own.

    ``training_answer`` is the batch layer's answer while training, where the
    one row is its own batch and has no spread, so every entry is zero.
    ``predicting_answer_fresh`` is the same row answered by a fresh layer's
    running figures, a mean of zero and a variance of one, and
    ``predicting_answer_final`` by the running figures after the last step.
    ``layer_answer`` and ``rms_answer`` are the two row layers, which answer
    one row exactly as they answer it among any others.
    """

    row: list[float]
    training_answer: list[float]
    predicting_answer_fresh: list[float]
    predicting_answer_final: list[float]
    layer_answer: list[float]
    rms_answer: list[float]


class RunningFiguresResponse(BaseModel):
    """The running figures followed step by step, and the row that has no batch.

    ``batch_mean`` and ``batch_deviation`` are where the running figures are
    heading. ``first_step_within_hundredth`` and
    ``first_step_within_thousandth`` are the first steps at which the largest
    gap falls below 0.01 and 0.001, or ``None`` if it never does within the
    steps asked for.
    """

    momentum: float
    batch_mean: list[float]
    batch_deviation: list[float]
    training_answer: list[list[float]]
    steps: list[RunningStep]
    first_step_within_hundredth: int | None
    first_step_within_thousandth: int | None
    lone_row: LoneRow


def _first_step_below(steps: list[RunningStep], threshold: float) -> int | None:
    return next((each.step for each in steps if each.largest_gap < threshold), None)


@router.post(
    "/concepts/normalisation-layers/running-figures",
    response_model=RunningFiguresResponse,
)
def running_figures(request: RunningFiguresRequest) -> RunningFiguresResponse:
    """Step the batch layer on the same block repeatedly and watch the figures move."""
    width = _check_rectangle(request.rows)
    block = np.asarray(request.rows, dtype=np.float64)

    layer = BatchNormalization(n_features=width)
    fresh = layer
    training = layer.respond_to(block, purpose=PassPurpose.TRAINING)
    training_outputs = np.asarray(training.outputs, dtype=np.float64)

    steps: list[RunningStep] = []
    for step in range(1, request.n_steps + 1):
        response = layer.respond_to(block, purpose=PassPurpose.TRAINING)
        correction = layer.correction_for(response, np.zeros_like(block))
        layer = layer.stepped_by(correction.gradient, learning_rate=ZERO_LEARNING_RATE)
        predicting = np.asarray(
            layer.respond_to(block, purpose=PassPurpose.PREDICTING).outputs,
            dtype=np.float64,
        )
        steps.append(
            RunningStep(
                step=step,
                running_mean=_as_floats(layer.running_mean),
                running_deviation=_as_floats(
                    np.sqrt(layer.running_variance + layer.epsilon)
                ),
                largest_gap=float(np.max(np.abs(predicting - training_outputs))),
            )
        )

    lone = block[:1]
    lone_row = LoneRow(
        row=_as_floats(lone[0]),
        training_answer=_as_floats(
            fresh.respond_to(lone, purpose=PassPurpose.TRAINING).outputs[0]
        ),
        predicting_answer_fresh=_as_floats(fresh.respond_to(lone).outputs[0]),
        predicting_answer_final=_as_floats(layer.respond_to(lone).outputs[0]),
        layer_answer=_as_floats(
            LayerNormalization(n_features=width).respond_to(lone).outputs[0]
        ),
        rms_answer=_as_floats(
            RMSNormalization(n_features=width).respond_to(lone).outputs[0]
        ),
    )

    return RunningFiguresResponse(
        momentum=float(fresh.momentum),
        batch_mean=_as_floats(training.batch_mean),
        batch_deviation=_as_floats(training.deviation),
        training_answer=_as_lists(training_outputs),
        steps=steps,
        first_step_within_hundredth=_first_step_below(steps, 0.01),
        first_step_within_thousandth=_first_step_below(steps, 0.001),
        lone_row=lone_row,
    )


class UnitDrift(BaseModel):
    """One hidden unit's mean and deviation over the batch, epoch by epoch,
    as read by the layer above it."""

    means: list[float]
    deviations: list[float]


class DriftReading(BaseModel):
    """How the block the output layer reads moved as the hidden layer learned.

    ``units`` holds one entry per hidden unit, each with the mean and the
    biased deviation of that unit over the whole crowd at every recorded
    epoch, beginning with the untrained network. The summaries are taken over
    every epoch, not only the recorded ones. ``largest_mean_shift`` is the
    widest distance any unit's mean travelled between the first epoch and the
    last, ``largest_deviation_change`` the same for the deviation, and the two
    ``single_epoch`` figures the widest move any unit made in one step.
    ``mean_matches_shift_gap`` is present under the batch arrangement only,
    the widest distance between what the output layer read as a unit's mean
    and the batch layer's shift for that unit, across every epoch.
    """

    arrangement: Arrangement
    recorded_epochs: list[int]
    units: list[UnitDrift]
    largest_mean_shift: float
    largest_deviation_change: float
    largest_single_epoch_mean_move: float
    largest_single_epoch_deviation_move: float
    mean_matches_shift_gap: float | None


class SeedReading(BaseModel):
    """One seed's result under one arrangement: mean training accuracy over the
    folds, and the held-out accuracy pooled over every person held out once."""

    seed: int
    training_accuracy: float
    held_out_accuracy: float
    training_loss: float


class ArrangementReading(BaseModel):
    """One arrangement across every seed, with the means the page compares.

    ``training_loss_mean`` is the loss the backward pass measured on the
    training rows at the end of each run, averaged over the folds and seeds.
    """

    arrangement: Arrangement
    training_accuracy_mean: float
    held_out_accuracy_mean: float
    held_out_accuracy_low: float
    held_out_accuracy_high: float
    training_loss_mean: float
    per_seed: list[SeedReading]


class CurveReading(BaseModel):
    """What the loop wrote down after one epoch on the whole crowd.

    ``loss`` is the number the backward pass measured on the way forward, so
    it belongs to the stack as it stood before this epoch's step, and under
    the batch arrangement it is the loss by the batch's own statistics.
    ``predicting_accuracy`` is the stepped stack asked properly and
    ``training_accuracy`` the same stack asked while still training, which
    differ only under the batch arrangement, where the running figures lag.
    """

    epoch: int
    loss: float
    predicting_accuracy: float
    training_accuracy: float


class ArrangementCurve(BaseModel):
    """One arrangement's whole-crowd run under the curve seed.

    ``first_epoch_below_threshold`` is the first epoch whose measured loss
    fell below ``LOSS_THRESHOLD``, or ``None`` if it never did.
    """

    arrangement: Arrangement
    readings: list[CurveReading]
    first_epoch_below_threshold: int | None
    final_loss: float
    final_accuracy: float


class Reparameterisation(BaseModel):
    """A weight-normalised layer against the dense layer holding its weights.

    ``output_gap`` is the widest difference between the two layers' answers
    to the same five people, which is zero because the forward pass is the
    same arithmetic. ``direction_slopes`` and ``magnitude_slopes`` are the
    library's gradient in the new coordinates, and ``direction_gap`` and
    ``magnitude_gap`` how far each is from a central finite difference of
    the objective taken through the layer's own forward pass.
    ``unprojected_gap`` is the same distance for the form that forgets to
    project the radial component out, and ``largest_direction_slope`` the
    biggest true slope, so the two gaps can be read against it.
    ``radial_component`` is the largest, over the neurons, of the direction
    slope dotted with the stored direction, which is zero for the true
    gradient because scaling a direction changes nothing the layer answers;
    ``unprojected_radial_component`` is the same for the unprojected form.

    The step figures take one step at ``rate`` on both layers from the same
    slope. ``destination_gap`` is the widest difference between the
    weight-normalised layer's effective weights afterwards and the dense
    layer's, and the three length lists say what each neuron's length was and
    became under each.
    """

    people: list[list[float]]
    directions: list[list[float]]
    magnitudes: list[float]
    output_gap: float
    direction_slopes: list[list[float]]
    magnitude_slopes: list[float]
    direction_gap: float
    magnitude_gap: float
    unprojected_gap: float
    largest_direction_slope: float
    radial_component: float
    unprojected_radial_component: float
    rate: float
    effective_weights_after: list[list[float]]
    dense_weights_after: list[list[float]]
    destination_gap: float
    lengths_before: list[float]
    lengths_after: list[float]
    dense_lengths_after: list[float]


class CrowdResponse(BaseModel):
    """The whole experiment, computed once and cached."""

    crowd: list[LabelledPoint]
    hidden_width: int
    epochs: int
    learning_rate: float
    n_folds: int
    seeds: list[int]
    loss_threshold: float
    drift: list[DriftReading]
    sweep: list[ArrangementReading]
    curves: list[ArrangementCurve]
    reparameterisation: Reparameterisation


def _crowd_arrays() -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([[height, weight] for height, weight, _ in TANGLED_CROWD])
    labels = np.asarray([float(label) for _, _, label in TANGLED_CROWD])
    return rows, labels


def _standardised(fitted_on: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rows in standard units of the rows the network is fitted on.

    Fitted on the training rows only, so a held-out person's coordinates never
    reach the fit, which is the pipelines page's rule.
    """
    return (rows - fitted_on.mean(axis=0)) / fitted_on.std(axis=0)


def _fresh_stack(arrangement: Arrangement, seed: int) -> LayerStack:
    """The crowd network under one arrangement, from one seed.

    The hidden weights are drawn with a spread of the square root of two over
    the fan-in, the initialisation the convolution layer uses, and the
    weight-normalised hidden layer draws its directions from the same stream
    in the same order, so the two start as the identical network and differ
    only in how their weights are stored.
    """
    generator = np.random.default_rng(seed)

    def dense(n_inputs: int, n_outputs: int, activation: Activation) -> DenseLayer:
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

    layers: list[Layer]
    if arrangement == Arrangement.WEIGHT:
        spread = (2.0 / 2) ** 0.5
        layers = [
            WeightNormalization(
                directions=generator.normal(0.0, spread, (HIDDEN_WIDTH, 2)),
                activation=RectifiedLinear(),
            )
        ]
    else:
        layers = [dense(2, HIDDEN_WIDTH, RectifiedLinear())]

    if arrangement == Arrangement.BATCH:
        layers.append(BatchNormalization(n_features=HIDDEN_WIDTH))
    elif arrangement == Arrangement.LAYER:
        layers.append(LayerNormalization(n_features=HIDDEN_WIDTH))
    elif arrangement == Arrangement.RMS:
        layers.append(RMSNormalization(n_features=HIDDEN_WIDTH))

    layers.append(dense(HIDDEN_WIDTH, 1, Identity()))
    return LayerStack(layers)


def _scores(stack: LayerStack, rows: np.ndarray, purpose: PassPurpose) -> np.ndarray:
    return np.asarray(stack.respond_to(rows, purpose).outputs, dtype=np.float64).ravel()


def _squashed(scores: np.ndarray) -> np.ndarray:
    """The loss's own squash, applied here because the loss hands back only scores."""
    return np.where(
        scores >= 0.0,
        1.0 / (1.0 + np.exp(-scores)),
        np.exp(scores) / (1.0 + np.exp(scores)),
    )


def _called(scores: np.ndarray) -> np.ndarray:
    return (_squashed(scores) >= 0.5).astype(np.float64)


def _accuracy(
    stack: LayerStack,
    rows: np.ndarray,
    labels: np.ndarray,
    purpose: PassPurpose = PassPurpose.PREDICTING,
) -> float:
    return float((_called(_scores(stack, rows, purpose)) == labels).mean())


def _folds(seed: int) -> list[np.ndarray]:
    """Five folds dealt from the seed, each a set of held-out row indices."""
    order = np.random.default_rng(seed).permutation(len(TANGLED_CROWD))
    return [order[fold::N_FOLDS] for fold in range(N_FOLDS)]


def _read_by_output_layer(stack: LayerStack, rows: np.ndarray) -> np.ndarray:
    """The block the last layer reads, through every layer beneath it while training."""
    block = rows
    for position in range(len(stack) - 1):
        block = np.asarray(
            stack[position].respond_to(block, purpose=PassPurpose.TRAINING).outputs,
            dtype=np.float64,
        )
    return block


def _drift(
    arrangement: Arrangement, rows: np.ndarray, targets: np.ndarray
) -> DriftReading:
    """Train on the whole crowd and watch what the output layer reads."""
    loss = BinaryCrossEntropy()
    stack = _fresh_stack(arrangement, CURVE_SEED)

    means: list[np.ndarray] = []
    deviations: list[np.ndarray] = []
    shift_gaps: list[float] = []

    def record(current: LayerStack) -> None:
        read = _read_by_output_layer(current, rows)
        means.append(read.mean(axis=0))
        deviations.append(read.std(axis=0))
        if arrangement == Arrangement.BATCH:
            normaliser = current[1]
            if not isinstance(normaliser, BatchNormalization):
                raise TypeError("the batch arrangement holds its batch layer second")
            shift_gaps.append(float(np.max(np.abs(means[-1] - normaliser.shift))))

    record(stack)
    for _ in range(EXPERIMENT_EPOCHS):
        stack = stack.stepped_by(
            stack.backward_pass(rows, targets, loss), LEARNING_RATE
        )
        record(stack)

    mean_curve = np.stack(means)
    deviation_curve = np.stack(deviations)
    recorded = list(range(0, EXPERIMENT_EPOCHS + 1, DRIFT_EVERY))

    return DriftReading(
        arrangement=arrangement,
        recorded_epochs=recorded,
        units=[
            UnitDrift(
                means=_as_floats(mean_curve[recorded, unit]),
                deviations=_as_floats(deviation_curve[recorded, unit]),
            )
            for unit in range(HIDDEN_WIDTH)
        ],
        largest_mean_shift=float(np.max(np.abs(mean_curve[-1] - mean_curve[0]))),
        largest_deviation_change=float(
            np.max(np.abs(deviation_curve[-1] - deviation_curve[0]))
        ),
        largest_single_epoch_mean_move=float(
            np.max(np.abs(np.diff(mean_curve, axis=0)))
        ),
        largest_single_epoch_deviation_move=float(
            np.max(np.abs(np.diff(deviation_curve, axis=0)))
        ),
        mean_matches_shift_gap=(max(shift_gaps) if shift_gaps else None),
    )


def _sweep(rows: np.ndarray, labels: np.ndarray) -> list[ArrangementReading]:
    n_people = len(TANGLED_CROWD)
    loss = BinaryCrossEntropy()
    sweep: list[ArrangementReading] = []
    for arrangement in Arrangement:
        per_seed: list[SeedReading] = []
        for seed in EXPERIMENT_SEEDS:
            held_correct = 0
            training_total = 0.0
            loss_total = 0.0
            for held_out in _folds(seed):
                in_fold = np.ones(n_people, dtype=bool)
                in_fold[held_out] = False
                training_rows = _standardised(rows[in_fold], rows[in_fold])
                held_rows = _standardised(rows[in_fold], rows[held_out])
                training_labels = labels[in_fold]
                targets = training_labels.reshape(-1, 1)

                stack = _fresh_stack(arrangement, seed)
                for _ in range(EXPERIMENT_EPOCHS):
                    backward = stack.backward_pass(training_rows, targets, loss)
                    stack = stack.stepped_by(backward, LEARNING_RATE)

                training_total += _accuracy(stack, training_rows, training_labels)
                loss_total += stack.backward_pass(training_rows, targets, loss).loss
                held_scores = _scores(stack, held_rows, PassPurpose.PREDICTING)
                held_correct += int((_called(held_scores) == labels[held_out]).sum())
            per_seed.append(
                SeedReading(
                    seed=seed,
                    training_accuracy=training_total / N_FOLDS,
                    held_out_accuracy=held_correct / n_people,
                    training_loss=loss_total / N_FOLDS,
                )
            )
        held = [reading.held_out_accuracy for reading in per_seed]
        sweep.append(
            ArrangementReading(
                arrangement=arrangement,
                training_accuracy_mean=float(
                    np.mean([reading.training_accuracy for reading in per_seed])
                ),
                held_out_accuracy_mean=float(np.mean(held)),
                held_out_accuracy_low=float(min(held)),
                held_out_accuracy_high=float(max(held)),
                training_loss_mean=float(
                    np.mean([reading.training_loss for reading in per_seed])
                ),
                per_seed=per_seed,
            )
        )
    return sweep


def _curve(
    arrangement: Arrangement, rows: np.ndarray, labels: np.ndarray, targets: np.ndarray
) -> ArrangementCurve:
    loss = BinaryCrossEntropy()
    stack = _fresh_stack(arrangement, CURVE_SEED)
    readings: list[CurveReading] = []
    for epoch in range(1, EXPERIMENT_EPOCHS + 1):
        backward = stack.backward_pass(rows, targets, loss)
        stack = stack.stepped_by(backward, LEARNING_RATE)
        readings.append(
            CurveReading(
                epoch=epoch,
                loss=backward.loss,
                predicting_accuracy=_accuracy(stack, rows, labels),
                training_accuracy=_accuracy(stack, rows, labels, PassPurpose.TRAINING),
            )
        )
    return ArrangementCurve(
        arrangement=arrangement,
        readings=readings,
        first_epoch_below_threshold=next(
            (reading.epoch for reading in readings if reading.loss < LOSS_THRESHOLD),
            None,
        ),
        final_loss=stack.backward_pass(rows, targets, loss).loss,
        final_accuracy=_accuracy(stack, rows, labels),
    )


def _reparameterisation(standard_rows: np.ndarray) -> Reparameterisation:
    """The weight-normalised layer against the dense layer holding its weights."""
    people = standard_rows[:REPARAMETERISATION_PEOPLE]
    directions = np.asarray(REPARAMETERISATION_DIRECTIONS, dtype=np.float64)
    biases = np.asarray(REPARAMETERISATION_BIASES, dtype=np.float64)
    arriving = np.asarray(REPARAMETERISATION_ARRIVING, dtype=np.float64)
    bend = HyperbolicTangent()

    normalised = WeightNormalization(
        directions=directions, biases=biases, activation=bend
    )
    magnitudes = np.asarray(normalised.magnitudes, dtype=np.float64)
    dense = DenseLayer(
        [
            Neuron(
                weights=np.asarray(normalised.effective_weights)[neuron],
                bias=float(biases[neuron]),
                activation=bend,
            )
            for neuron in range(directions.shape[0])
        ]
    )

    normalised_response = normalised.respond_to(people)
    dense_response = dense.respond_to(people)
    output_gap = float(
        np.max(np.abs(normalised_response.outputs - dense_response.outputs))
    )

    correction = normalised.correction_for(normalised_response, arriving)
    gradient = correction.gradient
    if gradient is None or not hasattr(gradient, "magnitudes"):
        raise TypeError("a weight-normalised layer reports slopes for its lengths")
    direction_slopes = np.asarray(gradient.weights, dtype=np.float64)
    magnitude_slopes = np.asarray(gradient.magnitudes, dtype=np.float64)

    dense_correction = dense.correction_for(dense_response, arriving)
    if dense_correction.gradient is None:
        raise TypeError("a dense layer reports slopes for its weights")
    weight_slope = np.asarray(dense_correction.gradient.weights, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    unprojected = (magnitudes / norms)[:, None] * weight_slope

    def objective(
        candidate_directions: np.ndarray, candidate_magnitudes: np.ndarray
    ) -> float:
        layer = WeightNormalization(
            directions=candidate_directions,
            magnitudes=candidate_magnitudes,
            biases=biases,
            activation=bend,
        )
        return float(np.sum(arriving * layer.respond_to(people).outputs))

    step = FINITE_DIFFERENCE_STEP
    direction_difference = np.empty_like(directions)
    for neuron in range(directions.shape[0]):
        for input_index in range(directions.shape[1]):
            up = directions.copy()
            up[neuron, input_index] += step
            down = directions.copy()
            down[neuron, input_index] -= step
            direction_difference[neuron, input_index] = (
                objective(up, magnitudes) - objective(down, magnitudes)
            ) / (2.0 * step)
    magnitude_difference = np.empty_like(magnitudes)
    for neuron in range(directions.shape[0]):
        up = magnitudes.copy()
        up[neuron] += step
        down = magnitudes.copy()
        down[neuron] -= step
        magnitude_difference[neuron] = (
            objective(directions, up) - objective(directions, down)
        ) / (2.0 * step)

    stepped = normalised.stepped_by(gradient, REPARAMETERISATION_RATE)
    dense_stepped = dense.stepped_by(dense_correction.gradient, REPARAMETERISATION_RATE)
    effective_after = np.asarray(stepped.effective_weights, dtype=np.float64)
    dense_after = np.asarray(dense_stepped.weight_matrix, dtype=np.float64)

    return Reparameterisation(
        people=_as_lists(people),
        directions=_as_lists(directions),
        magnitudes=_as_floats(magnitudes),
        output_gap=output_gap,
        direction_slopes=_as_lists(direction_slopes),
        magnitude_slopes=_as_floats(magnitude_slopes),
        direction_gap=float(np.max(np.abs(direction_difference - direction_slopes))),
        magnitude_gap=float(np.max(np.abs(magnitude_difference - magnitude_slopes))),
        unprojected_gap=float(np.max(np.abs(direction_difference - unprojected))),
        largest_direction_slope=float(np.max(np.abs(direction_difference))),
        radial_component=float(
            np.max(np.abs((direction_slopes * directions).sum(axis=1)))
        ),
        unprojected_radial_component=float(
            np.max(np.abs((unprojected * directions).sum(axis=1)))
        ),
        rate=REPARAMETERISATION_RATE,
        effective_weights_after=_as_lists(effective_after),
        dense_weights_after=_as_lists(dense_after),
        destination_gap=float(np.max(np.abs(effective_after - dense_after))),
        lengths_before=_as_floats(magnitudes),
        lengths_after=_as_floats(np.asarray(stepped.magnitudes)),
        dense_lengths_after=_as_floats(np.linalg.norm(dense_after, axis=1)),
    )


@cache
def _crowd() -> CrowdResponse:
    rows, labels = _crowd_arrays()
    standard_rows = _standardised(rows, rows)
    targets = labels.reshape(-1, 1)

    return CrowdResponse(
        crowd=[
            LabelledPoint(x=float(height), y=float(weight), label=label)
            for height, weight, label in TANGLED_CROWD
        ],
        hidden_width=HIDDEN_WIDTH,
        epochs=EXPERIMENT_EPOCHS,
        learning_rate=LEARNING_RATE,
        n_folds=N_FOLDS,
        seeds=list(EXPERIMENT_SEEDS),
        loss_threshold=LOSS_THRESHOLD,
        drift=[
            _drift(Arrangement.NONE, standard_rows, targets),
            _drift(Arrangement.BATCH, standard_rows, targets),
        ],
        sweep=_sweep(rows, labels),
        curves=[
            _curve(arrangement, standard_rows, labels, targets)
            for arrangement in Arrangement
        ],
        reparameterisation=_reparameterisation(standard_rows),
    )


@router.get("/concepts/normalisation-layers/crowd", response_model=CrowdResponse)
def crowd() -> CrowdResponse:
    """The tangled crowd under five arrangements, held out fold by fold."""
    return _crowd()
