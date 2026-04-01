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


PLANE_STEPS = 7
"""The fitted plane is sampled on a lattice this many values per side."""


class Person3d(BaseModel):
    """One person with two inputs and the weight the plane predicts from them."""

    height: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    age: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    weight: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class PlaneFitRequest(BaseModel):
    """The people to fit the plane through."""

    people: list[Person3d] = Field(min_length=3, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class PlaneFitResponse(BaseModel):
    """The fitted plane, its numbers, and a lattice of its surface to draw.

    ``surface`` is row-major over ages then heights, each entry the plane's
    predicted weight at that lattice corner.
    """

    height_coefficient: float
    age_coefficient: float
    intercept: float
    r_squared: float
    heights_axis: list[float]
    ages_axis: list[float]
    surface: list[list[float]]


@router.post("/fit-plane", response_model=PlaneFitResponse)
def fit_plane(request: PlaneFitRequest) -> PlaneFitResponse:
    """Fit weight from height and age together, and sample the plane."""
    heights = Feature("height", [person.height for person in request.people])
    ages = Feature("age", [person.age for person in request.people])
    weights = Feature("weight", [person.weight for person in request.people])

    model = MultipleLinearRegression().fit([heights, ages], weights)
    named = {
        coefficient.name: float(coefficient.value) for coefficient in model.coefficients
    }

    heights_axis = np.linspace(min(heights.values), max(heights.values), PLANE_STEPS)
    ages_axis = np.linspace(min(ages.values), max(ages.values), PLANE_STEPS)
    lattice_h, lattice_a = np.meshgrid(heights_axis, ages_axis)
    surface = np.asarray(
        model.predict(
            [
                Feature("height", lattice_h.ravel()),
                Feature("age", lattice_a.ravel()),
            ]
        ),
        dtype=np.float64,
    ).reshape(PLANE_STEPS, PLANE_STEPS)

    return PlaneFitResponse(
        height_coefficient=named["height"],
        age_coefficient=named["age"],
        intercept=float(model.intercept),
        r_squared=float(model.score([heights, ages], weights)),
        heights_axis=[float(value) for value in heights_axis],
        ages_axis=[float(value) for value in ages_axis],
        surface=[[float(value) for value in row] for row in surface],
    )


class DegreeSweepRequest(BaseModel):
    """The points, and the highest degree the sweep should climb to."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_degree: int = Field(ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}


class DegreeScore(BaseModel):
    """One degree of the sweep and the R² the fit reached there."""

    degree: int
    r_squared: float


class DegreeSweepResponse(BaseModel):
    """The whole staircase of fits, one score per degree."""

    scores: list[DegreeScore]


@router.post("/degree-sweep", response_model=DegreeSweepResponse)
def degree_sweep(request: DegreeSweepRequest) -> DegreeSweepResponse:
    """Fit every degree from one upward and report each R².

    The sweep stops early where the points run out, since a degree needs one
    more point than its power to pin down, so the response may hold fewer
    scores than the request asked for rather than a refusal.
    """
    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]

    input_feature = Feature("t", x_values)
    target = Feature("y", y_values)

    highest = min(request.max_degree, len(request.points) - 1)
    scores: list[DegreeScore] = []
    for degree in range(1, highest + 1):
        polynomial = PolynomialFeatures(degree=degree).fit([input_feature])
        terms = polynomial.transform([input_feature])
        model = MultipleLinearRegression().fit(terms, target)
        scores.append(
            DegreeScore(degree=degree, r_squared=float(model.score(terms, target)))
        )

    return DegreeSweepResponse(scores=scores)
