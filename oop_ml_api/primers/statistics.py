"""Means, spreads, correlation and sampling, for the statistics primer.

Not part of the ``oop_ml`` library. This exists only to draw the statistics
primer's pictures. One endpoint summarises a small cloud of (x, y) points the
way the primer's prose does by hand, means, spreads, covariance and
correlation, all with the population divisor n so the arithmetic a reader
checks on paper is the arithmetic returned. The other draws people from a fixed
imagined population of heights, normal with mean 170 cm and standard deviation
7 cm, and bins them into a histogram, so the primer can show noisy draws
settling around the value underneath them.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS

router = APIRouter(prefix="/primers/statistics", tags=["statistics-primer"])

TRUE_MEAN = 170.0
"""The mean height of the imagined population the sampler draws from."""

TRUE_STANDARD_DEVIATION = 7.0
"""Its standard deviation."""

BIN_START = 146.0
"""Where the histogram's first bin begins, in cm."""

BIN_WIDTH = 2.0
"""How wide each histogram bin is, in cm."""

BIN_COUNT = 24
"""How many bins the histogram holds, spanning 146 to 194."""

MAX_SAMPLES = 500
"""The most drawn people a request may accumulate."""

MAX_DRAW = 100
"""The most people one press may draw."""


BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]

SampleValue = Annotated[float, Field(ge=140.0, le=200.0)]


class Point(BaseModel):
    """One (x, y) observation."""

    x: BoundedValue
    y: BoundedValue


class SummarizeRequest(BaseModel):
    """A cloud of points to summarise."""

    points: list[Point] = Field(min_length=1, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class SummarizeResponse(BaseModel):
    """Everything the scatter widget reads off the cloud.

    All divisors are the population n, matching the primer's hand arithmetic.
    ``correlation`` is ``None`` when either spread is zero, because dividing by
    a zero spread asks a question the data cannot answer.
    """

    mean_x: float
    mean_y: float
    variance_x: float
    variance_y: float
    standard_deviation_x: float
    standard_deviation_y: float
    covariance: float
    correlation: float | None


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Summarise the cloud the way the primer's prose does by hand."""
    x_values = np.array([point.x for point in request.points], dtype=np.float64)
    y_values = np.array([point.y for point in request.points], dtype=np.float64)

    mean_x = float(np.mean(x_values))
    mean_y = float(np.mean(y_values))

    x_deviations = x_values - mean_x
    y_deviations = y_values - mean_y

    variance_x = float(np.mean(x_deviations**2))
    variance_y = float(np.mean(y_deviations**2))
    standard_deviation_x = float(np.sqrt(variance_x))
    standard_deviation_y = float(np.sqrt(variance_y))
    covariance = float(np.mean(x_deviations * y_deviations))

    correlation: float | None
    if standard_deviation_x == 0.0 or standard_deviation_y == 0.0:
        correlation = None
    else:
        correlation = covariance / (standard_deviation_x * standard_deviation_y)

    return SummarizeResponse(
        mean_x=mean_x,
        mean_y=mean_y,
        variance_x=variance_x,
        variance_y=variance_y,
        standard_deviation_x=standard_deviation_x,
        standard_deviation_y=standard_deviation_y,
        covariance=covariance,
        correlation=correlation,
    )


class SampleRequest(BaseModel):
    """The draws so far, and how many more to make.

    The widget keeps the accumulated values and sends them back each press, so
    the endpoint stays stateless. The seed makes a press reproducible.
    """

    values: list[SampleValue] = Field(default_factory=list, max_length=MAX_SAMPLES)
    draw: int = Field(ge=1, le=MAX_DRAW)
    seed: int = Field(ge=0, le=2_000_000_000)

    model_config = {"extra": "forbid"}


class HistogramBin(BaseModel):
    """One bar of the histogram."""

    start: float
    end: float
    count: int


BELL_SAMPLES = 97
"""How many points the bell curve is sampled at across the histogram."""


class BellPoint(BaseModel):
    """One point of the true bell, scaled to the people drawn so far.

    ``expected_count`` is the density times the sample size times the bin
    width, so the curve sits on the same axis as the histogram's counts and
    shows the shape the bars are settling toward.
    """

    x: float
    expected_count: float


class SampleResponse(BaseModel):
    """The extended draws, their summary, and the histogram to draw."""

    values: list[float]
    count: int
    mean: float
    standard_deviation: float
    bins: list[HistogramBin]
    bell: list[BellPoint]
    true_mean: float
    true_standard_deviation: float


@router.post("/sample", response_model=SampleResponse)
def sample(request: SampleRequest) -> SampleResponse:
    """Draw more people from the fixed population and re-bin the histogram."""
    generator = np.random.default_rng(request.seed)

    room = MAX_SAMPLES - len(request.values)
    draw = min(request.draw, room)

    drawn = generator.normal(TRUE_MEAN, TRUE_STANDARD_DEVIATION, size=draw)
    drawn = np.clip(drawn, BIN_START, BIN_START + BIN_WIDTH * BIN_COUNT - 0.1)
    drawn = np.round(drawn, 1)

    values = np.array([*request.values, *drawn], dtype=np.float64)

    bin_indices = np.floor((values - BIN_START) / BIN_WIDTH).astype(int)
    bin_indices = np.clip(bin_indices, 0, BIN_COUNT - 1)
    counts = np.bincount(bin_indices, minlength=BIN_COUNT)

    bins = [
        HistogramBin(
            start=BIN_START + index * BIN_WIDTH,
            end=BIN_START + (index + 1) * BIN_WIDTH,
            count=int(counts[index]),
        )
        for index in range(BIN_COUNT)
    ]

    grid = np.linspace(BIN_START, BIN_START + BIN_WIDTH * BIN_COUNT, BELL_SAMPLES)
    standardized = (grid - TRUE_MEAN) / TRUE_STANDARD_DEVIATION
    density = np.exp(-0.5 * standardized**2) / (
        TRUE_STANDARD_DEVIATION * np.sqrt(2.0 * np.pi)
    )
    expected_counts = density * values.size * BIN_WIDTH

    return SampleResponse(
        values=[float(value) for value in values],
        count=int(values.size),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values)),
        bins=bins,
        bell=[
            BellPoint(x=float(x), expected_count=float(expected))
            for x, expected in zip(grid, expected_counts, strict=True)
        ],
        true_mean=TRUE_MEAN,
        true_standard_deviation=TRUE_STANDARD_DEVIATION,
    )


class DescribeValuesRequest(BaseModel):
    """A handful of single-feature values to take apart around their mean."""

    values: list[BoundedValue] = Field(min_length=2, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DescribeValuesResponse(BaseModel):
    """The mean and everything the primer builds from it, in order.

    ``deviation_total`` is returned rather than asserted so the widget can
    show the cancellation the prose describes, the below balancing the above
    to zero, before the squares repair it.
    """

    mean: float
    deviations: list[float]
    deviation_total: float
    squared_deviations: list[float]
    variance: float
    standard_deviation: float


@router.post("/describe-values", response_model=DescribeValuesResponse)
def describe_values(request: DescribeValuesRequest) -> DescribeValuesResponse:
    """Walk one list of values from mean to standard deviation.

    Population divisor n throughout, like the summarize endpoint, so the
    numbers match what a reader checks on paper.
    """
    values = np.asarray(request.values, dtype=np.float64)
    mean = float(np.mean(values))
    deviations = values - mean
    squared = deviations**2
    variance = float(np.mean(squared))

    return DescribeValuesResponse(
        mean=mean,
        deviations=[float(value) for value in deviations],
        deviation_total=float(np.sum(deviations)),
        squared_deviations=[float(value) for value in squared],
        variance=variance,
        standard_deviation=float(np.sqrt(variance)),
    )
