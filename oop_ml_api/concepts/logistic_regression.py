"""Fit a pass-or-fail probability curve, for the logistic regression page.

One input, hours studied, and a yes-or-no outcome. The library's
``LogisticRegression`` fits by gradient ascent, which matters for a widget,
since a learner will drag the two classes fully apart and a Newton solver
would refuse at that point while the ascent simply steepens toward a step and
stops at its epoch cap. The response carries the fitted slope and intercept,
the boundary where the probability crosses one half, the accuracy, and a dense
sampling of the probability curve for the widget to draw.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from oop_ml import Feature, LogisticRegression
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/logistic-regression", tags=["logistic-regression"])

CURVE_SAMPLES = 81
"""How many points the probability curve is sampled at across the data."""

FLAT_SLOPE = 1e-9
"""Below this the curve is level and no boundary crossing exists to report."""


class Outcome(BaseModel):
    """One person, how long they studied, and whether they passed."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=1)


class LogisticFitRequest(BaseModel):
    """The observed outcomes to fit the curve through."""

    points: list[Outcome] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class LogisticFitResponse(BaseModel):
    """The fitted curve and everything the widget reads off it.

    ``boundary`` is where the probability crosses one half, ``None`` when the
    fitted curve is level and never crosses.
    """

    slope: float
    intercept: float
    boundary: float | None
    accuracy: float
    curve: list[CurvePoint]


@router.post("/fit", response_model=LogisticFitResponse)
def fit(request: LogisticFitRequest) -> LogisticFitResponse:
    """Fit the probability curve and sample it for drawing.

    A single-class target is refused by the library with its own typed error,
    which the shared handler turns into a 422; nothing here anticipates it.
    """
    hours = Feature("hours", [point.x for point in request.points])
    passed = Feature("passed", [float(point.label) for point in request.points])

    model = LogisticRegression().fit([hours], passed)
    slope = float(next(iter(model.coefficients)).value)
    intercept = float(model.intercept)

    boundary: float | None
    if abs(slope) < FLAT_SLOPE:
        boundary = None
    else:
        boundary = -intercept / slope

    x_values = [point.x for point in request.points]
    grid = np.linspace(min(x_values), max(x_values), CURVE_SAMPLES)
    probabilities = np.asarray(
        model.predict_probability([Feature("hours", grid)]), dtype=np.float64
    )

    return LogisticFitResponse(
        slope=slope,
        intercept=intercept,
        boundary=boundary,
        accuracy=float(model.score([hours], passed)),
        curve=[
            CurvePoint(x=float(x), y=float(p))
            for x, p in zip(grid, probabilities, strict=True)
        ],
    )
