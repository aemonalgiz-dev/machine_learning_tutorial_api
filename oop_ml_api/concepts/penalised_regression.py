"""Ridge and lasso on a deliberately overgrown polynomial, for the penalty page.

The page's story is that a degree-9 curve fits noise, and a penalty on the
coefficients is the cure, so this endpoint fits exactly that. The power columns
are manufactured by the library's ``PolynomialFeatures``, standardized by its
``Standardizer`` so the penalty treats every power alike, a power of nine
otherwise dwarfing a power of one, and then fitted by ``RidgeRegression`` or
``LassoRegression`` at the requested penalty. The response carries the
standardized-scale coefficients, which is what the widget's bar chart draws,
along with how many of them survived at exactly nonzero, which is the lasso's
half of the story.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Feature,
    LassoRegression,
    PolynomialFeatures,
    RidgeRegression,
    Standardizer,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import (
    CurvePoint,
    NamedCoefficient,
    Point,
)
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/ridge-lasso", tags=["ridge-lasso"])

DEGREE = 9
"""The deliberately overgrown polynomial every fit here bends."""

MAX_PENALTY = 1000.0
"""The largest penalty the slider may ask for."""

CURVE_SAMPLES = 81
"""How many points the fitted curve is sampled at across the data's x-range."""

ZERO_TOLERANCE = 1e-9
"""A coefficient within this of zero counts as switched off."""


class PenalisedFitRequest(BaseModel):
    """The points, which penalty family, and how hard to pull."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    model: Literal["ridge", "lasso"]
    penalty: float = Field(ge=0.0, le=MAX_PENALTY)

    model_config = {"extra": "forbid"}


class PenalisedFitResponse(BaseModel):
    """The tamed curve, its score, and the coefficients the penalty left."""

    coefficients: list[NamedCoefficient]
    intercept: float
    r_squared: float
    nonzero_count: int
    curve: list[CurvePoint]


@router.post("/fit", response_model=PenalisedFitResponse)
def fit(request: PenalisedFitRequest) -> PenalisedFitResponse:
    """Fit the degree-9 polynomial under the requested penalty."""
    if len(request.points) < DEGREE + 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the degree {DEGREE} curve here has {DEGREE + 1} numbers to "
                "pin down, so it needs at least that many points"
            ),
        )

    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]

    input_feature = Feature("t", x_values)
    target = Feature("y", y_values)

    polynomial = PolynomialFeatures(degree=DEGREE).fit([input_feature])
    terms = polynomial.transform([input_feature])
    standardizer = Standardizer().fit(terms)
    standardized = standardizer.transform(terms)

    if request.model == "ridge":
        model = RidgeRegression(penalty=request.penalty).fit(standardized, target)
    else:
        model = LassoRegression(penalty=request.penalty).fit(standardized, target)

    grid = np.linspace(min(x_values), max(x_values), CURVE_SAMPLES)
    grid_terms = standardizer.transform(polynomial.transform([Feature("t", grid)]))
    curve_heights = np.asarray(model.predict(grid_terms), dtype=np.float64)

    coefficients = [
        NamedCoefficient(name=coefficient.name, value=float(coefficient.value))
        for coefficient in model.coefficients
    ]

    return PenalisedFitResponse(
        coefficients=coefficients,
        intercept=float(model.intercept),
        r_squared=float(model.score(standardized, target)),
        nonzero_count=sum(
            1 for coefficient in coefficients if abs(coefficient.value) > ZERO_TOLERANCE
        ),
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
    )
