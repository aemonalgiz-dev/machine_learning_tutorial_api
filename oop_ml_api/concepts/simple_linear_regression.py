"""Fit a line to a handful of points, and hand back everything a widget draws.

The first concept, and the template for the rest. The endpoint is a plain
``def`` -- not ``async def`` -- on purpose: FastAPI runs a synchronous path
operation in its worker threadpool, off the event loop, which is exactly the
architecture a CPU-bound numpy library wants behind a web server. An
``async def`` here would run the fit *on* the event loop and block every other
request for its duration. So the rule for every compute endpoint in this API is:
plain ``def``, and let the framework thread it.

The response carries more than the fit. A teaching widget needs to draw the
line, mark the residual for each point, and show the numbers the technical
section explains -- so the endpoint returns the slope and intercept, the two
endpoints of the fitted line across the data's x-range, the per-point residuals,
and R^2. Computing those here rather than in the browser keeps the one source of
truth -- the library -- authoritative, and keeps the frontend a view.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from oop_ml import Feature, SimpleLinearRegression
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(
    prefix="/concepts/simple-linear-regression",
    tags=["simple-linear-regression"],
)


class Point(BaseModel):
    """One (x, y) observation, bounded to the teaching range."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class FitRequest(BaseModel):
    """The points to fit a line through.

    The length bounds are the API's, enforced before the library sees anything:
    a request with too few or too many points is a 422 naming the limit, not a
    slow compute or an underlying ``TooFewValuesError`` a learner cannot parse.
    """

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)


class FittedLine(BaseModel):
    """The fitted line as two endpoints, so the frontend draws without maths."""

    x_start: float
    y_start: float
    x_end: float
    y_end: float


class FitResponse(BaseModel):
    """What the fit found, plus everything the widget renders from it."""

    slope: float
    intercept: float
    r_squared: float
    line: FittedLine
    residuals: list[float]


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Fit ``y = slope * x + intercept`` to the points and describe the result.

    A synchronous endpoint by design -- see the module docstring. The body is a
    straight translation: points become two ``Feature`` columns, the library
    fits, and the fitted line is drawn across the observed x-range so the widget
    has endpoints to connect. A degenerate input -- every x identical, so no
    line is determined -- raises a typed library error that the shared handler
    turns into a 422; nothing here has to anticipate it.
    """
    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]

    model = SimpleLinearRegression().fit(Feature("x", x_values), Feature("y", y_values))

    predictions = [float(value) for value in model.predict(x_values)]
    residuals = [
        actual - predicted
        for actual, predicted in zip(y_values, predictions, strict=True)
    ]

    x_start, x_end = min(x_values), max(x_values)

    return FitResponse(
        slope=model.slope,
        intercept=model.intercept,
        r_squared=model.score(Feature("x", x_values), Feature("y", y_values)),
        line=FittedLine(
            x_start=x_start,
            y_start=model.slope * x_start + model.intercept,
            x_end=x_end,
            y_end=model.slope * x_end + model.intercept,
        ),
        residuals=residuals,
    )


SURFACE_STEPS = 21
"""The loss surface is sampled on a lattice this many settings per side."""


class LossSurfaceRequest(BaseModel):
    """The points whose residual sum of squares is to be drawn as a surface."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class LossMinimum(BaseModel):
    """Where the library's fit says the bowl bottoms out."""

    slope: float
    intercept: float
    rss: float


class LossSurfaceResponse(BaseModel):
    """The bowl the derivation talks about, sampled around its floor.

    ``rss`` is row-major over intercepts then slopes, so ``rss[i][j]`` is the
    residual sum of squares at ``intercepts[i]`` and ``slopes[j]``.
    """

    slopes: list[float]
    intercepts: list[float]
    rss: list[list[float]]
    minimum: LossMinimum


@router.post("/loss-surface", response_model=LossSurfaceResponse)
def loss_surface(request: LossSurfaceRequest) -> LossSurfaceResponse:
    """Sample RSS over a window of slopes and intercepts around the best fit.

    The best pair comes from the library's own fit, and the window is centred
    on it, wide enough in the intercept that the bowl's diagonal valley, a
    larger slope partly repaid by a smaller intercept, shows itself.
    """
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)

    model = SimpleLinearRegression().fit(Feature("x", x_values), Feature("y", y_values))
    best_slope = float(model.slope)
    best_intercept = float(model.intercept)

    slope_span = max(0.5, abs(best_slope))
    intercept_span = max(10.0, slope_span * float(np.mean(np.abs(x_values))))

    slopes = np.linspace(
        best_slope - slope_span, best_slope + slope_span, SURFACE_STEPS
    )
    intercepts = np.linspace(
        best_intercept - intercept_span,
        best_intercept + intercept_span,
        SURFACE_STEPS,
    )

    predictions = (
        slopes[None, :, None] * x_values[None, None, :] + intercepts[:, None, None]
    )
    rss = np.sum((y_values[None, None, :] - predictions) ** 2, axis=2)

    best_rss = float(np.sum((y_values - (best_slope * x_values + best_intercept)) ** 2))

    return LossSurfaceResponse(
        slopes=[float(value) for value in slopes],
        intercepts=[float(value) for value in intercepts],
        rss=[[float(value) for value in row] for row in rss],
        minimum=LossMinimum(slope=best_slope, intercept=best_intercept, rss=best_rss),
    )
