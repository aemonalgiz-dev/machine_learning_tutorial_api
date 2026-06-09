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
from oop_ml_api.concepts.split import held_out_indices
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


class TermContribution(BaseModel):
    """One term's share of the fitted curve, sampled where the curve is.

    The coefficient multiplies a standardized column, so the contribution at
    a time is the coefficient times that column's standardized value there.
    Summed over the terms and the intercept, the contributions are the curve,
    which is how a moderate prediction can be seen to come out of large
    opposing pieces.
    """

    name: str
    values: list[float]


class PenalisedFitResponse(BaseModel):
    """The tamed curve, its score, the coefficients the penalty left, and the
    two halves of the objective the fit actually minimised.

    ``penalty_cost`` is the penalty times its sum, so ``objective`` is
    ``residual_sum_of_squares + penalty_cost``, the number the fit lowered.
    """

    coefficients: list[NamedCoefficient]
    intercept: float
    r_squared: float
    residual_sum_of_squares: float
    penalty_cost: float
    objective: float
    nonzero_count: int
    curve: list[CurvePoint]
    term_contributions: list[TermContribution]


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
    standardized_columns = {term.name: np.asarray(term.values) for term in grid_terms}

    evaluation = model.evaluate(standardized, target)
    weights = np.asarray([coefficient.value for coefficient in coefficients])
    residual_sum_of_squares = float(evaluation.residual_sum_of_squares)
    penalty_cost = request.penalty * _penalty_sum(request.model, weights)

    return PenalisedFitResponse(
        coefficients=coefficients,
        intercept=float(model.intercept),
        r_squared=float(evaluation.r2_score),
        residual_sum_of_squares=residual_sum_of_squares,
        penalty_cost=penalty_cost,
        objective=residual_sum_of_squares + penalty_cost,
        nonzero_count=sum(
            1 for coefficient in coefficients if abs(coefficient.value) > ZERO_TOLERANCE
        ),
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
        term_contributions=[
            TermContribution(
                name=coefficient.name,
                values=[
                    float(value)
                    for value in coefficient.value
                    * standardized_columns[coefficient.name]
                ],
            )
            for coefficient in coefficients
        ],
    )


def _penalty_sum(model: Literal["ridge", "lasso"], weights: np.ndarray) -> float:
    """What the penalty multiplies: squared for ridge, absolute for lasso."""
    if model == "ridge":
        return float(np.sum(weights**2))
    return float(np.sum(np.abs(weights)))


def _penalised_model(
    model: Literal["ridge", "lasso"], penalty: float
) -> RidgeRegression | LassoRegression:
    if model == "ridge":
        return RidgeRegression(penalty=penalty)
    return LassoRegression(penalty=penalty)


PATH_PENALTIES = [float(value) for value in np.logspace(-6, 2, 33)]
"""The penalties the path is traced at, log-spaced from a millionth to a hundred.

It starts that low because on the split fit the held-out score is still
climbing between a millionth and a thousandth, and a path that began at a
thousandth would show only the fall.
"""


class RegularisationPathRequest(BaseModel):
    """The points and the family whose whole path is to be traced."""

    points: list[Point] = Field(min_length=DEGREE + 1, max_length=MAX_POINTS)
    model: Literal["ridge", "lasso"]

    model_config = {"extra": "forbid"}


class PathStep(BaseModel):
    """Everything the page reads off one penalty on the path.

    The objective, its two halves and the coefficients come from a fit on
    every point. The two scores come from a second fit on the training share
    of the evaluation page's seeded split, scored on each share, so the path
    can show where the held-out score turns while the training score only
    ever falls.
    """

    penalty: float
    coefficients: list[float]
    nonzero_count: int
    residual_sum_of_squares: float
    penalty_cost: float
    objective: float
    train_r_squared: float
    held_out_r_squared: float


class RegularisationPathResponse(BaseModel):
    """The path, one step per penalty, with the term names once."""

    names: list[str]
    steps: list[PathStep]


@router.post("/regularisation-path", response_model=RegularisationPathResponse)
def regularisation_path(
    request: RegularisationPathRequest,
) -> RegularisationPathResponse:
    """Refit the degree-9 polynomial at every penalty on the path, twice."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    held = held_out_indices(x_values.size)
    train_mask = np.ones(x_values.size, dtype=bool)
    train_mask[held] = False
    n_training = int(np.sum(train_mask))
    if n_training <= DEGREE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the degree {DEGREE} curve needs more than {DEGREE} training "
                f"points and the training share holds only {n_training}"
            ),
        )

    every_input = Feature("t", x_values)
    every_target = Feature("y", y_values)
    polynomial = PolynomialFeatures(degree=DEGREE).fit([every_input])
    every_terms = polynomial.transform([every_input])
    every_standardized = Standardizer().fit(every_terms).transform(every_terms)

    train_input = Feature("t", x_values[train_mask])
    train_target = Feature("y", y_values[train_mask])
    train_polynomial = PolynomialFeatures(degree=DEGREE).fit([train_input])
    train_terms = train_polynomial.transform([train_input])
    train_standardizer = Standardizer().fit(train_terms)
    train_standardized = train_standardizer.transform(train_terms)
    held_standardized = train_standardizer.transform(
        train_polynomial.transform([Feature("t", x_values[held])])
    )
    held_target = Feature("y", y_values[held])

    names = [term.name for term in every_terms]
    steps: list[PathStep] = []
    for penalty in PATH_PENALTIES:
        model = _penalised_model(request.model, penalty).fit(
            every_standardized, every_target
        )
        weights = np.asarray([coefficient.value for coefficient in model.coefficients])
        residual_sum_of_squares = float(
            model.evaluate(every_standardized, every_target).residual_sum_of_squares
        )
        penalty_cost = penalty * _penalty_sum(request.model, weights)
        split_model = _penalised_model(request.model, penalty).fit(
            train_standardized, train_target
        )
        steps.append(
            PathStep(
                penalty=penalty,
                coefficients=[float(value) for value in weights],
                nonzero_count=int(np.sum(np.abs(weights) > ZERO_TOLERANCE)),
                residual_sum_of_squares=residual_sum_of_squares,
                penalty_cost=penalty_cost,
                objective=residual_sum_of_squares + penalty_cost,
                train_r_squared=float(
                    split_model.score(train_standardized, train_target)
                ),
                held_out_r_squared=float(
                    split_model.score(held_standardized, held_target)
                ),
            )
        )

    return RegularisationPathResponse(names=names, steps=steps)


GEOMETRY_ROWS = 24
"""How many rows the two-coefficient teaching data holds."""

GEOMETRY_SEED = 11
"""The two correlated columns and their target are drawn once, seeded."""

GEOMETRY_GRID = 41
"""The residual sum of squares is sampled on a lattice this many per side."""

GEOMETRY_PENALTIES = [float(value) for value in np.logspace(-1, 3, 17)]
"""The penalties the ridge and lasso solutions are traced at, 0.1 to 1000."""

GEOMETRY_NUDGE_SEED = 12
"""The perturbed copy of the target is drawn from this seed."""

GEOMETRY_NUDGE = 0.35
"""How much the target is jostled to make the perturbed sample."""


def _correlated_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two centred columns that track one another, and a centred target.

    The second column is close to twice the first, so the least-squares
    solution lives in a long shallow valley and a small change to the target
    slides it a long way along the floor.
    """
    generator = np.random.default_rng(GEOMETRY_SEED)
    first = generator.normal(0.0, 1.0, GEOMETRY_ROWS)
    second = 2.0 * first + generator.normal(0.0, 0.15, GEOMETRY_ROWS)
    target = 3.0 * first + 1.0 * second + generator.normal(0.0, 1.0, GEOMETRY_ROWS)
    return first - first.mean(), second - second.mean(), target - target.mean()


class CoefficientPair(BaseModel):
    """A point in the plane of the two coefficients."""

    first: float
    second: float


class GeometrySolutions(BaseModel):
    """The least-squares point and the two penalised paths for one sample."""

    least_squares: CoefficientPair
    ridge: list[CoefficientPair]
    lasso: list[CoefficientPair]


class PenaltyGeometryResponse(BaseModel):
    """The RSS lattice over two correlated coefficients, and where each fit lands.

    ``rss`` is row-major over the second coefficient then the first. The
    ``perturbed`` solutions come from the same columns with the target
    jostled a little, which is what shows the least-squares point sliding
    along the valley while the penalised points stay put.
    """

    first_axis: list[float]
    second_axis: list[float]
    rss: list[list[float]]
    penalties: list[float]
    original: GeometrySolutions
    perturbed: GeometrySolutions


def _pair_of(model: RidgeRegression | LassoRegression) -> CoefficientPair:
    named = {each.name: float(each.value) for each in model.coefficients}
    return CoefficientPair(first=named["first"], second=named["second"])


def _geometry_solutions(
    first: np.ndarray, second: np.ndarray, target: np.ndarray
) -> GeometrySolutions:
    features = [Feature("first", first), Feature("second", second)]
    target_feature = Feature("target", target)
    return GeometrySolutions(
        least_squares=_pair_of(
            RidgeRegression(penalty=0.0, fit_intercept=False).fit(
                features, target_feature
            )
        ),
        ridge=[
            _pair_of(
                RidgeRegression(penalty=penalty, fit_intercept=False).fit(
                    features, target_feature
                )
            )
            for penalty in GEOMETRY_PENALTIES
        ],
        lasso=[
            _pair_of(
                LassoRegression(
                    penalty=penalty,
                    fit_intercept=False,
                    max_iterations=LASSO_PATH_ITERATIONS,
                    tolerance=LASSO_PATH_TOLERANCE,
                ).fit(features, target_feature)
            )
            for penalty in GEOMETRY_PENALTIES
        ],
    )


@router.get("/penalty-geometry", response_model=PenaltyGeometryResponse)
def penalty_geometry() -> PenaltyGeometryResponse:
    """Sample RSS over two correlated coefficients and place every solution on it."""
    first, second, target = _correlated_pair()
    original = _geometry_solutions(first, second, target)

    nudged = target + np.random.default_rng(GEOMETRY_NUDGE_SEED).normal(
        0.0, GEOMETRY_NUDGE, GEOMETRY_ROWS
    )
    perturbed = _geometry_solutions(first, second, nudged - nudged.mean())

    reach = 1.6 * max(
        abs(original.least_squares.first),
        abs(original.least_squares.second),
        abs(perturbed.least_squares.first),
        abs(perturbed.least_squares.second),
        1.0,
    )
    first_axis = np.linspace(-reach, reach, GEOMETRY_GRID)
    second_axis = np.linspace(-reach, reach, GEOMETRY_GRID)
    lattice_first, lattice_second = np.meshgrid(first_axis, second_axis)
    predictions = (
        lattice_first.ravel()[:, None] * first[None, :]
        + lattice_second.ravel()[:, None] * second[None, :]
    )
    rss = np.sum((target[None, :] - predictions) ** 2, axis=1).reshape(
        GEOMETRY_GRID, GEOMETRY_GRID
    )

    return PenaltyGeometryResponse(
        first_axis=[float(value) for value in first_axis],
        second_axis=[float(value) for value in second_axis],
        rss=[[float(value) for value in row] for row in rss],
        penalties=GEOMETRY_PENALTIES,
        original=original,
        perturbed=perturbed,
    )


PATH_SAMPLES = 51
"""How many penalties the shrinkage path is traced at."""

PATH_MAX_PENALTY = 500.0
"""Where the traced path ends, far enough that lasso has reached zero."""

LASSO_PATH_ITERATIONS = 100_000
"""Sweep budget for each lasso fit on the path.

The library's default budget is sized for the standardized degree-9 fit
above; on the raw single-feature people the coordinate steps are tiny and
the default stops early, reporting a slope well short of where the descent
was heading. A deep budget and a tight tolerance let every fit actually
arrive, which is what makes the path's exact zero at the soft-threshold
penalty visible instead of smeared.
"""

LASSO_PATH_TOLERANCE = 1e-12
"""Convergence tolerance for the path's lasso fits."""


class ShrinkagePathRequest(BaseModel):
    """The single-feature points whose two shrinkage paths are to be traced."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class ShrinkagePathResponse(BaseModel):
    """One slope per penalty for each family, over the same penalties.

    Ridge divides, so its slope shrinks toward zero and never lands on it.
    Lasso subtracts, so its slope hits exactly zero at a finite penalty and
    stays there.
    """

    penalties: list[float]
    ridge_slopes: list[float]
    lasso_slopes: list[float]


@router.post("/shrinkage-path", response_model=ShrinkagePathResponse)
def shrinkage_path(request: ShrinkagePathRequest) -> ShrinkagePathResponse:
    """Fit ridge and lasso at each penalty and trace both slopes.

    The feature is centred first. Centring changes only the intercept, never
    the slope, and it uncouples the two so the lasso's coordinate sweeps
    converge in a handful of passes instead of crawling; uncentred, the whole
    path took five seconds, centred it takes milliseconds, and every reported
    slope is identical.
    """
    raw_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    centred_values = raw_values - float(np.mean(raw_values))
    input_feature = Feature("height", centred_values)
    target = Feature("weight", [point.y for point in request.points])

    penalties = [
        float(value) for value in np.linspace(0.0, PATH_MAX_PENALTY, PATH_SAMPLES)
    ]

    ridge_slopes: list[float] = []
    lasso_slopes: list[float] = []
    for penalty in penalties:
        ridge = RidgeRegression(penalty=penalty).fit([input_feature], target)
        ridge_slopes.append(float(next(iter(ridge.coefficients)).value))

        lasso = LassoRegression(
            penalty=penalty,
            max_iterations=LASSO_PATH_ITERATIONS,
            tolerance=LASSO_PATH_TOLERANCE,
        ).fit([input_feature], target)
        lasso_slopes.append(float(next(iter(lasso.coefficients)).value))

    return ShrinkagePathResponse(
        penalties=penalties,
        ridge_slopes=ridge_slopes,
        lasso_slopes=lasso_slopes,
    )
