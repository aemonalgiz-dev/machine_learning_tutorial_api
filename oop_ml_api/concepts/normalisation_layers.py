"""Normalisation layers, for the page that standardises inside a model.

One endpoint, one block of numbers, rows down and features across. It pushes
the block through one of the library's three normalising layers,
``BatchNormalization``, ``LayerNormalization`` or ``RMSNormalization``, chosen
by a closed enum, and returns what the layer answered together with the
statistics it divided by, one per feature for the batch layer and one per row
for the other two. The page needs the statistics beside the answers because
the whole lesson is which axis they were read along, and a widget that only
showed the normalised numbers would hide the one thing that differs between
the three layers.

For the batch layer the endpoint also returns the same rows passed while
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

Every layer keeps its epsilon inside the square root, so a deviation of 3
comes back as the root of 9.00001, and the page quotes that figure rather than
rounding it away. The variance is the biased one, dividing by the number of
rows, which is why a four-row block can have a whole-number deviation and a
three-row block with any spread in it never can.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BatchNormalization,
    LayerNormalization,
    PassPurpose,
    RMSNormalization,
)
from pydantic import BaseModel, Field

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


class NormalisationLayer(StrEnum):
    """The three layers, named by which axis their statistics run along."""

    BATCH = "batch"
    LAYER = "layer"
    RMS = "rms"


BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]

BlockRow = Annotated[
    list[BoundedValue], Field(min_length=MIN_FEATURES, max_length=MAX_FEATURES)
]


class NormaliseRequest(BaseModel):
    """The block, rows down and features across, and which layer to push it through."""

    rows: list[BlockRow] = Field(min_length=MIN_ROWS, max_length=MAX_ROWS)
    layer: NormalisationLayer

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

    ``normalised`` is the layer's answer in the input arrangement. The scale
    starts at one and the shift at zero, so it is the standardised block
    exactly. ``as_predicting`` and ``predicting_statistics`` are present for
    the batch layer only, the same rows answered by the running figures after
    one training pass and one step, and are ``None`` for the two layers that
    have no such distinction.
    """

    layer: NormalisationLayer
    normalised: list[list[float]]
    statistics: list[ReducedStatistic]
    as_predicting: list[list[float]] | None
    predicting_statistics: list[ReducedStatistic] | None


@router.post(
    "/concepts/normalisation-layers/normalise", response_model=NormaliseResponse
)
def normalise(request: NormaliseRequest) -> NormaliseResponse:
    """Push the block through the chosen layer and report what it divided by."""
    width = len(request.rows[0])
    for position, row in enumerate(request.rows):
        if len(row) != width:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"row {position + 1} holds {len(row)} values where the first "
                    f"row holds {width}, and a batch has to be a rectangle before "
                    "any statistic can run along it"
                ),
            )

    block = np.asarray(request.rows, dtype=np.float64)

    if request.layer == NormalisationLayer.BATCH:
        return _batch_normalised(block)
    if request.layer == NormalisationLayer.LAYER:
        return _row_normalised(block, LayerNormalization(n_features=width))
    return _row_normalised(block, RMSNormalization(n_features=width))


def _batch_normalised(block: np.ndarray) -> NormaliseResponse:
    """The batch layer's answer while training, and again after one step."""
    layer = BatchNormalization(n_features=block.shape[1])
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
        normalised=_as_lists(training.outputs),
        statistics=_statistics(training.batch_mean, training.deviation),
        as_predicting=_as_lists(predicting.outputs),
        predicting_statistics=_statistics(stepped.running_mean, running_deviation),
    )


def _row_normalised(
    block: np.ndarray, layer: LayerNormalization | RMSNormalization
) -> NormaliseResponse:
    """A within-row layer's answer, with each row's own centre and deviation.

    The row layers carry no statistics on their response, since nothing needs
    them afterwards, so the figures are read off the block here by the same
    definition the layer uses, the biased spread about the row's centre plus
    epsilon, under the root. The test module checks that dividing by them
    reproduces the layer's answer to the last bit.
    """
    response = layer.respond_to(block)

    if layer.centres:
        row_means = block.mean(axis=1)
        centred = block - row_means[:, None]
    else:
        row_means = None
        centred = block
    row_deviations = np.sqrt((centred**2).mean(axis=1) + layer.epsilon)

    return NormaliseResponse(
        layer=(NormalisationLayer.LAYER if layer.centres else NormalisationLayer.RMS),
        normalised=_as_lists(response.outputs),
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
