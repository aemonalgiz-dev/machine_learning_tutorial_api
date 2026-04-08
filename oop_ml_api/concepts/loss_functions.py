"""The five losses, for the loss-functions page.

A loss takes the last layer's raw output and the truth and answers two things,
what the miss costs and which way to move the output to shrink it. The page
puts one slider on that raw output and asks all five of the library's losses
at once, so the reader watches five costs and five slopes move together and
sees where they agree and where they part.

Two endpoints. ``curve`` sweeps the raw output across a range and returns each
loss's value and gradient at every sample, which draws the five curves.
``measure`` evaluates all five at one raw output, which is what the slider's
readout shows and what the page's worked example quotes.

Two truths travel in every request, because the losses ask two different
questions. The three regression losses compare the raw output to a number,
``target``. The two classification losses compare it to a yes or a no,
``label``, and apply their own squash first, which is why the request carries
both and why the response reports the squashed probability only for those two.

Every loss here is measured one row at a time. The library divides by the row
count so batches of different sizes stay comparable, and with one row that
division is by one, which keeps the page's hand arithmetic honest.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    SoftmaxCrossEntropy,
    SquaredError,
)
from pydantic import BaseModel, Field

router = APIRouter(tags=["loss-functions"])

RAW_OUTPUT_LOW = -5.0
RAW_OUTPUT_HIGH = 5.0
"""The range the slider walks. Wide enough to show every loss's far behaviour."""

MAX_TARGET = 5.0
"""The furthest a regression truth may sit, matching the raw output's range."""

MAX_THRESHOLD = 10.0
"""The largest Huber knee the page may ask for."""

MIN_SAMPLES = 21
MAX_SAMPLES = 201
"""How finely the curve endpoint may sample the raw output."""

FIRST_CLASS = 0
"""The softmax column that carries the raw output. The other column's logit is
held at zero, which is what makes a two-class softmax the sigmoid."""


class LossSetting(BaseModel):
    """The two truths, and the one loss that takes a parameter."""

    target: float = Field(ge=-MAX_TARGET, le=MAX_TARGET)
    label: int = Field(ge=0, le=1)
    huber_threshold: float = Field(default=1.0, gt=0.0, le=MAX_THRESHOLD)

    model_config = {"extra": "forbid"}


class MeasureRequest(LossSetting):
    """The setting, plus the one raw output to measure at."""

    raw_output: float = Field(ge=RAW_OUTPUT_LOW, le=RAW_OUTPUT_HIGH)


class CurveRequest(LossSetting):
    """The setting, plus how finely to sample the sweep."""

    n_samples: int = Field(default=101, ge=MIN_SAMPLES, le=MAX_SAMPLES)


class LossReading(BaseModel):
    """What one loss answered at one raw output.

    ``probability`` is the squash the loss applied on its way to the value,
    and only the two classification losses apply one.
    """

    value: float
    gradient: float
    probability: float | None = None


class FiveReadings(BaseModel):
    """All five losses at one raw output."""

    squared_error: LossReading
    absolute_error: LossReading
    huber_error: LossReading
    binary_cross_entropy: LossReading
    softmax_cross_entropy: LossReading


class MeasureResponse(BaseModel):
    """The raw output measured, and the five answers."""

    raw_output: float
    readings: FiveReadings


class CurveSample(BaseModel):
    """One loss at one sample of the sweep."""

    raw_output: float
    value: float
    gradient: float
    probability: float | None = None


class LossCurves(BaseModel):
    """Five curves over the same sweep of raw outputs."""

    squared_error: list[CurveSample]
    absolute_error: list[CurveSample]
    huber_error: list[CurveSample]
    binary_cross_entropy: list[CurveSample]
    softmax_cross_entropy: list[CurveSample]


def _readings_at(
    raw_output: float, target: float, label: int, huber_threshold: float
) -> FiveReadings:
    """Ask all five losses about one raw output, one row each."""
    outputs = np.array([[raw_output]])
    number_truth = np.array([[target]])
    yes_or_no_truth = np.array([[float(label)]])

    squared = SquaredError().measure(outputs, number_truth)
    absolute = AbsoluteError().measure(outputs, number_truth)
    huber = HuberError(threshold=huber_threshold).measure(outputs, number_truth)
    binary = BinaryCrossEntropy().measure(outputs, yes_or_no_truth)

    # Two logits, the raw output in the first column and zero in the second,
    # so the first class's probability is the sigmoid of the raw output and
    # the page can show the two classification losses agreeing exactly.
    two_logits = np.array([[raw_output, 0.0]])
    true_column = FIRST_CLASS if label == 1 else 1 - FIRST_CLASS
    one_hot = SoftmaxCrossEntropy.one_hot([true_column], 2)
    softmax = SoftmaxCrossEntropy().measure(two_logits, one_hot)

    # The probabilities the two classification losses applied on the way to
    # their values. The library does not hand them back, so they are recomputed
    # here from the same definition, the sigmoid of the raw output, which is
    # arithmetic on the request and not modelling.
    yes_probability = float(1.0 / (1.0 + np.exp(-raw_output)))

    return FiveReadings(
        squared_error=LossReading(
            value=squared.value, gradient=float(squared.gradient[0, 0])
        ),
        absolute_error=LossReading(
            value=absolute.value, gradient=float(absolute.gradient[0, 0])
        ),
        huber_error=LossReading(
            value=huber.value, gradient=float(huber.gradient[0, 0])
        ),
        binary_cross_entropy=LossReading(
            value=binary.value,
            gradient=float(binary.gradient[0, 0]),
            probability=yes_probability,
        ),
        softmax_cross_entropy=LossReading(
            value=softmax.value,
            gradient=float(softmax.gradient[0, FIRST_CLASS]),
            probability=yes_probability,
        ),
    )


@router.post("/concepts/loss-functions/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> MeasureResponse:
    """All five losses at one raw output."""
    return MeasureResponse(
        raw_output=request.raw_output,
        readings=_readings_at(
            request.raw_output,
            request.target,
            request.label,
            request.huber_threshold,
        ),
    )


@router.post("/concepts/loss-functions/curve", response_model=LossCurves)
def curve(request: CurveRequest) -> LossCurves:
    """All five losses across a sweep of the raw output."""
    sweep = np.linspace(RAW_OUTPUT_LOW, RAW_OUTPUT_HIGH, request.n_samples)
    readings = [
        _readings_at(
            float(raw_output), request.target, request.label, request.huber_threshold
        )
        for raw_output in sweep
    ]

    def samples_of(name: str) -> list[CurveSample]:
        return [
            CurveSample(
                raw_output=float(raw_output),
                value=getattr(reading, name).value,
                gradient=getattr(reading, name).gradient,
                probability=getattr(reading, name).probability,
            )
            for raw_output, reading in zip(sweep, readings, strict=True)
        ]

    return LossCurves(
        squared_error=samples_of("squared_error"),
        absolute_error=samples_of("absolute_error"),
        huber_error=samples_of("huber_error"),
        binary_cross_entropy=samples_of("binary_cross_entropy"),
        softmax_cross_entropy=samples_of("softmax_cross_entropy"),
    )
