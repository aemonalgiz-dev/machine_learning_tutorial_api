"""Fit a polynomial through draggable points, for the curved-regression page.

The second concept endpoint, and the first to lean on the library's
preprocessing. The points arrive as one input column, the library's
``PolynomialFeatures`` manufactures the power columns from it, and
``MultipleLinearRegression`` fits them exactly as it would fit any other set of
features, which is the page's whole teaching point. The response carries the
named coefficients, so a degree-2 fit of a thrown ball hands back the physics,
and a dense sampling of the fitted curve so the widget can draw it, including
whatever wild swings a high degree produces between the points.

A plain ``def`` like every compute endpoint here, so FastAPI runs it in the
worker threadpool off the event loop.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import Feature, MultipleLinearRegression, PolynomialFeatures
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(
    prefix="/concepts/multiple-polynomial-regression",
    tags=["multiple-polynomial-regression"],
)

MAX_DEGREE = 9
"""The highest power the widget may ask for."""

CURVE_SAMPLES = 81
"""How many points the fitted curve is sampled at across the data's x-range."""


class Point(BaseModel):
    """One (x, y) observation, bounded to the teaching range."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class PolynomialFitRequest(BaseModel):
    """The points and the degree of the curve to bend through them."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}


class NamedCoefficient(BaseModel):
    """One fitted weight, bound to the name of the column it multiplies."""

    name: str
    value: float


class CurvePoint(BaseModel):
    """A sample of the fitted curve."""

    x: float
    y: float


class PolynomialFitResponse(BaseModel):
    """The fitted polynomial, named, scored and sampled for drawing."""

    coefficients: list[NamedCoefficient]
    intercept: float
    r_squared: float
    curve: list[CurvePoint]


@router.post("/fit", response_model=PolynomialFitResponse)
def fit(request: PolynomialFitRequest) -> PolynomialFitResponse:
    """Manufacture the power columns and fit them as ordinary features.

    A degree-d curve has d + 1 numbers to pin down, so it needs at least that
    many points; the refusal happens here, in words a learner can read, rather
    than surfacing as a collinearity message from the solver.
    """
    if len(request.points) < request.degree + 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a degree {request.degree} curve has {request.degree + 1} "
                "numbers to pin down, so it needs at least that many points"
            ),
        )

    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]

    input_feature = Feature("t", x_values)
    target = Feature("y", y_values)

    polynomial = PolynomialFeatures(degree=request.degree).fit([input_feature])
    terms = polynomial.transform([input_feature])

    model = MultipleLinearRegression().fit(terms, target)

    grid = np.linspace(min(x_values), max(x_values), CURVE_SAMPLES)
    grid_terms = polynomial.transform([Feature("t", grid)])
    curve_heights = np.asarray(model.predict(grid_terms), dtype=np.float64)

    return PolynomialFitResponse(
        coefficients=[
            NamedCoefficient(name=coefficient.name, value=float(coefficient.value))
            for coefficient in model.coefficients
        ],
        intercept=float(model.intercept),
        r_squared=float(model.score(terms, target)),
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
    )
