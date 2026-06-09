"""Kernel ridge regression on draggable points, for the kernel-ridge page.

The page's claim is that ridge regression, rearranged so its rows appear only
inside inner products, can have those inner products swapped for a kernel and
then draws a curve through the data without one manufactured column. The fit
endpoint runs that claim on the library's ``KernelRidgeRegression`` under
whichever of its four kernels the reader picks, and fits the plain
``RidgeRegression`` on the same points beside it, because under the linear
kernel the two must coincide. That coincidence is the control the page
teaches, and the response carries the largest gap between the two sampled
curves so the widget can read it out rather than ask the eye to judge.

The fit also hands back what the kernel model actually learned, one dual
weight per training row, and beside each weight the row's residual divided by
the penalty, because the optimality condition of the dual says those two
columns are the same column, which is the page's reading of what a dual
weight is. For the linear kernel it adds the slope those weights imply, which
is the weight the dual form never forms and the number the worked example
checks against ridge's slope by hand. The Gram matrix's extreme eigenvalues
and the condition number of the penalised system ride along too, since the
penalty's job in the dual is visible only through them.

The gram endpoint hands back the Gram matrix itself for a small set of rows,
which is the one table the model ever reads, and its eigenvalues, which is
how a function that is not a kernel gives itself away before the solve does.
The penalty sweep refits one kernel at a ladder of penalties and reports how
the fit, the weights and the conditioning move together. The cost endpoint
times the library's own fits as the rows and the features grow, because the
page's claim that the dual costs rows rather than features is a claim about
time and should be measured as one.

The centring endpoint exists for one section of the page. The library
centres both the features and the target before its solve, and the reason is
recorded in its own docstring, that centring only the target gave an answer
close enough to ridge's to pass for rounding and wrong for a reason. The
library refuses to make that mistake, so the mistake has to be made here to be
shown. The endpoint solves the dual system once by hand on uncentred rows,
using the library's kernel for the Gram matrix and nothing else, and reports
the line that variant implies beside the two correct ones. It is the only
place in this API that solves a linear system itself, and it does so to draw
a wrong line, on purpose, next to two right ones.

Every endpoint is a plain ``def`` so FastAPI runs it in the worker threadpool
off the event loop.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Feature,
    Kernel,
    KernelRidgeRegression,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    RidgeRegression,
    RowBlock,
    SigmoidKernel,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint, Point
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/kernel-ridge", tags=["kernel-ridge"])

MAX_PENALTY = 1000.0
"""The largest penalty the slider may ask for."""

MIN_PENALTY = 1e-9
"""The smallest penalty the sweep may ask for.

Small enough that the radial kernel interpolates every training row and the
condition number climbs past a billion, which is the picture the penalty
section needs, and large enough that the solve still completes.
"""

MAX_DEGREE = 6
"""The highest power the polynomial kernel may raise to."""

MAX_GAMMA = 1000.0
"""How sharply the radial and sigmoid kernels may be tuned."""

MIN_GAMMA = 1e-6
"""How gently they may be tuned.

Heights in centimetres are tens of centimetres apart, so a gamma that reads
the crowd as one connected shape is around a thousandth, and the slider has
to reach it.
"""

MAX_CONSTANT = 10.0
"""How far the sigmoid kernel's constant may be shifted either way.

Negative values are allowed on purpose. They are where the sigmoid stops
being a kernel, and the library's refusal when that happens is part of what
the page shows.
"""

CURVE_SAMPLES = 81
"""How many points each fitted curve is sampled at."""

CURVE_PAD_SHARE = 0.1
"""The curves are sampled this much past the data's x-range on either side.

A little extrapolation is the honest picture. The radial kernel's curve drifts
back toward the target mean once it leaves the data, and a line does not,
and neither fact is visible if the sampling stops at the last point.
"""

MAX_SWEEP_PENALTIES = 40
"""How many penalties one sweep request may ask for."""

MAX_GRAM_ROWS = 20
"""How many rows the gram endpoint will lay out as a table.

The matrix is meant to be read, and a table wider than this is not.
"""

INTERPOLATION_TOLERANCE = 1e-3
"""A training row whose residual is within this counts as passed through."""

FEATURE_NAME = "x"
"""The one input column every fit here reads."""

TARGET_NAME = "y"
"""The column every fit here predicts."""


class LinearKernelChoice(BaseModel):
    """The plain dot product, which makes the fit ordinary ridge regression."""

    name: Literal["linear"]

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return LinearKernel()


class PolynomialKernelChoice(BaseModel):
    """The dot product plus one, raised to a power.

    The library's default constant of one is kept, so every lower power rides
    along with the highest, and its default gamma of one leaves the dot
    product unscaled.
    """

    name: Literal["polynomial"]
    degree: int = Field(ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return PolynomialKernel(degree=self.degree)


class RadialKernelChoice(BaseModel):
    """A bump that falls with distance, with gamma setting how fast."""

    name: Literal["rbf"]
    gamma: float = Field(ge=MIN_GAMMA, le=MAX_GAMMA)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return RadialBasisKernel(gamma=self.gamma)


class SigmoidKernelChoice(BaseModel):
    """A hyperbolic tangent of the scaled dot product, not always a kernel."""

    name: Literal["sigmoid"]
    gamma: float = Field(ge=MIN_GAMMA, le=MAX_GAMMA)
    constant: float = Field(ge=-MAX_CONSTANT, le=MAX_CONSTANT)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return SigmoidKernel(gamma=self.gamma, constant=self.constant)


KernelChoice = Annotated[
    LinearKernelChoice
    | PolynomialKernelChoice
    | RadialKernelChoice
    | SigmoidKernelChoice,
    Field(discriminator="name"),
]
"""Which reading of similarity to hand the fit, with exactly its own parameters."""


class KernelRidgeFitRequest(BaseModel):
    """The points, the kernel, how hard to pull, and one x to be asked about.

    ``curve_from`` and ``curve_to`` widen the sampled range past the data,
    which is how the extrapolation section watches the radial curve return
    to the mean. Left unset, the curves run a tenth past the data either side.
    """

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice
    penalty: float = Field(ge=MIN_PENALTY, le=MAX_PENALTY)
    query_x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    curve_from: float | None = Field(default=None, ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    curve_to: float | None = Field(default=None, ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)

    model_config = {"extra": "forbid"}


class KernelRidgeFitResponse(BaseModel):
    """The kernel curve, the ridge line beside it, and what the fit learned.

    ``dual_weights`` holds one number per training row, in the order the
    points arrived, which is the whole of what the kernel model learned.
    ``residuals`` are the training rows' misses under the kernel fit and
    ``residuals_over_penalty`` those misses divided by the penalty, which the
    dual's optimality condition says are the dual weights; ``dual_residual_gap``
    is the largest difference between the two columns. ``implied_slope`` is
    the weight the dual weights add up to under the linear kernel, and
    ``None`` under any other, where no slope exists to report.
    ``largest_gap`` is the widest vertical distance between the two sampled
    curves, the control readout, zero to within rounding under the linear
    kernel. ``gram_smallest_eigenvalue`` and ``gram_largest_eigenvalue``
    describe the Gram matrix before the penalty is added and
    ``condition_number`` the system after, which is where the penalty's
    conditioning job shows.
    """

    curve: list[CurvePoint]
    ridge_curve: list[CurvePoint]
    dual_weights: list[float]
    fitted: list[float]
    residuals: list[float]
    residuals_over_penalty: list[float]
    dual_residual_gap: float
    max_abs_dual_weight: float
    feature_mean: float
    target_mean: float
    implied_slope: float | None
    ridge_slope: float
    ridge_intercept: float
    r_squared: float
    ridge_r_squared: float
    largest_gap: float
    prediction_at_query: float
    ridge_at_query: float
    n_training_rows: int
    gram_smallest_eigenvalue: float
    gram_largest_eigenvalue: float
    condition_number: float


@router.post("/fit", response_model=KernelRidgeFitResponse)
def fit(request: KernelRidgeFitRequest) -> KernelRidgeFitResponse:
    """Fit kernel ridge under the chosen kernel, and plain ridge beside it."""
    x_values = [point.x for point in request.points]
    input_feature = Feature(FEATURE_NAME, x_values)
    target = Feature(TARGET_NAME, [point.y for point in request.points])

    kernel_model = KernelRidgeRegression(
        kernel=request.kernel.kernel(), penalty=request.penalty
    ).fit([input_feature], target)
    ridge_model = RidgeRegression(penalty=request.penalty).fit([input_feature], target)

    grid = _sampling_grid(x_values, request.curve_from, request.curve_to)
    grid_feature = Feature(FEATURE_NAME, grid)
    kernel_heights = np.asarray(kernel_model.predict([grid_feature]), dtype=np.float64)
    ridge_heights = np.asarray(ridge_model.predict([grid_feature]), dtype=np.float64)

    query_feature = Feature(FEATURE_NAME, [request.query_x])
    kernel_at_query = np.asarray(
        kernel_model.predict([query_feature]), dtype=np.float64
    )
    ridge_at_query = np.asarray(ridge_model.predict([query_feature]), dtype=np.float64)

    fitted = np.asarray(kernel_model.predict([input_feature]), dtype=np.float64)
    residuals = np.asarray(target.values, dtype=np.float64) - fitted
    dual_weights = np.asarray(kernel_model.dual_weights, dtype=np.float64)
    over_penalty = residuals / request.penalty
    implied_slope = (
        _implied_slope(kernel_model)
        if isinstance(request.kernel, LinearKernelChoice)
        else None
    )
    eigenvalues = _gram_eigenvalues(kernel_model)

    return KernelRidgeFitResponse(
        curve=_curve_of(grid, kernel_heights),
        ridge_curve=_curve_of(grid, ridge_heights),
        dual_weights=[float(weight) for weight in dual_weights],
        fitted=[float(value) for value in fitted],
        residuals=[float(value) for value in residuals],
        residuals_over_penalty=[float(value) for value in over_penalty],
        dual_residual_gap=float(np.max(np.abs(dual_weights - over_penalty))),
        max_abs_dual_weight=float(np.max(np.abs(dual_weights))),
        feature_mean=float(np.mean(x_values)),
        target_mean=float(kernel_model.target_mean),
        implied_slope=implied_slope,
        ridge_slope=float(ridge_model.coefficients[FEATURE_NAME]),
        ridge_intercept=float(ridge_model.intercept),
        r_squared=float(kernel_model.score([input_feature], target)),
        ridge_r_squared=float(ridge_model.score([input_feature], target)),
        largest_gap=float(np.max(np.abs(kernel_heights - ridge_heights))),
        prediction_at_query=float(kernel_at_query[0]),
        ridge_at_query=float(ridge_at_query[0]),
        n_training_rows=kernel_model.n_training_rows,
        gram_smallest_eigenvalue=float(eigenvalues[0]),
        gram_largest_eigenvalue=float(eigenvalues[-1]),
        condition_number=_condition_number(eigenvalues, request.penalty),
    )


class GramRequest(BaseModel):
    """A few points and a kernel, to lay the Gram matrix out as a table."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_GRAM_ROWS)
    kernel: KernelChoice

    model_config = {"extra": "forbid"}


class GramResponse(BaseModel):
    """The one table a kernel model reads, on the centred rows.

    ``centred_x`` is each row's input with the mean removed, which is what
    the library hands its kernel. ``matrix[i][j]`` is the kernel between rows
    ``i`` and ``j``. ``eigenvalues`` are in ascending order, so the first is
    the smallest, and a negative one means the function that built the table
    is not a kernel on these rows. ``zero_eigenvalues`` counts how many are
    zero to within rounding, which is how many directions the unpenalised
    system has no answer along.
    """

    centred_x: list[float]
    matrix: list[list[float]]
    eigenvalues: list[float]
    zero_eigenvalues: int
    symmetric: bool


@router.post("/gram", response_model=GramResponse)
def gram(request: GramRequest) -> GramResponse:
    """Build the Gram matrix the fit would read, and report its eigenvalues."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    centred = x_values - np.mean(x_values)
    rows = RowBlock(centred.reshape(-1, 1), [FEATURE_NAME])
    matrix = request.kernel.kernel().between(rows, rows).values
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)

    return GramResponse(
        centred_x=[float(value) for value in centred],
        matrix=[[float(value) for value in row] for row in matrix],
        eigenvalues=[float(value) for value in eigenvalues],
        zero_eigenvalues=int(np.sum(np.abs(eigenvalues) < 1e-9 * scale)),
        symmetric=bool(np.allclose(matrix, matrix.T)),
    )


class PenaltySweepRequest(BaseModel):
    """The points, one kernel, and the ladder of penalties to refit at."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice
    penalties: list[float] = Field(min_length=1, max_length=MAX_SWEEP_PENALTIES)

    model_config = {"extra": "forbid"}


class PenaltyStep(BaseModel):
    """One refit at one penalty, and what moved.

    ``rows_passed_through`` counts training rows the curve misses by less
    than a thousandth, which is the interpolation the penalty is there to
    stop. ``farthest_from_mean`` is how far the sampled curve strays from the
    target mean across the data, which falls to zero as the penalty grows.
    """

    penalty: float
    r_squared: float
    max_abs_dual_weight: float
    condition_number: float
    rows_passed_through: int
    farthest_from_mean: float
    curve: list[CurvePoint]


class PenaltySweepResponse(BaseModel):
    """The refits in the order the penalties were asked for."""

    target_mean: float
    steps: list[PenaltyStep]


@router.post("/penalty-sweep", response_model=PenaltySweepResponse)
def penalty_sweep(request: PenaltySweepRequest) -> PenaltySweepResponse:
    """Refit one kernel at every penalty asked for and report each fit."""
    x_values = [point.x for point in request.points]
    input_feature = Feature(FEATURE_NAME, x_values)
    target = Feature(TARGET_NAME, [point.y for point in request.points])
    target_values = np.asarray(target.values, dtype=np.float64)
    grid = _sampling_grid(x_values, None, None)
    grid_feature = Feature(FEATURE_NAME, grid)
    kernel = request.kernel.kernel()

    steps: list[PenaltyStep] = []
    for penalty in request.penalties:
        if not MIN_PENALTY <= penalty <= MAX_PENALTY:
            raise HTTPException(
                status_code=422,
                detail=f"every penalty must lie between {MIN_PENALTY} and {MAX_PENALTY}",
            )
        model = KernelRidgeRegression(kernel=kernel, penalty=penalty).fit(
            [input_feature], target
        )
        heights = np.asarray(model.predict([grid_feature]), dtype=np.float64)
        fitted = np.asarray(model.predict([input_feature]), dtype=np.float64)
        residuals = target_values - fitted
        eigenvalues = _gram_eigenvalues(model)
        steps.append(
            PenaltyStep(
                penalty=penalty,
                r_squared=float(model.score([input_feature], target)),
                max_abs_dual_weight=float(np.max(np.abs(model.dual_weights))),
                condition_number=_condition_number(eigenvalues, penalty),
                rows_passed_through=int(
                    np.sum(np.abs(residuals) < INTERPOLATION_TOLERANCE)
                ),
                farthest_from_mean=float(np.max(np.abs(heights - model.target_mean))),
                curve=_curve_of(grid, heights),
            )
        )

    return PenaltySweepResponse(target_mean=float(np.mean(target_values)), steps=steps)


class CentringRequest(BaseModel):
    """The points and the penalty for the linear-kernel control."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    penalty: float = Field(ge=MIN_PENALTY, le=MAX_PENALTY)

    model_config = {"extra": "forbid"}


class ControlLine(BaseModel):
    """One straight answer, where it stands over the mean x, and its samples."""

    slope: float
    intercept: float
    at_mean: float
    curve: list[CurvePoint]


class CentringResponse(BaseModel):
    """Three lines through one set of points, two of them the same line.

    ``ridge`` is the library's ridge regression with its intercept unpenalised.
    ``kernel_ridge`` is the library's kernel ridge under the linear kernel,
    which centres both the features and the target, and it lands on ``ridge``
    exactly. ``target_only`` centres the target and leaves the features where
    they were, the mistake the library's docstring records, and its line
    misses the mean point every least squares line passes through.
    """

    mean: Point
    ridge: ControlLine
    kernel_ridge: ControlLine
    target_only: ControlLine


@router.post("/centring", response_model=CentringResponse)
def centring(request: CentringRequest) -> CentringResponse:
    """Fit the control two right ways and one wrong way, and report each line."""
    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]
    input_feature = Feature(FEATURE_NAME, x_values)
    target = Feature(TARGET_NAME, y_values)
    feature_mean = float(np.mean(x_values))
    target_mean = float(np.mean(y_values))

    ridge_model = RidgeRegression(penalty=request.penalty).fit([input_feature], target)
    kernel_model = KernelRidgeRegression(
        kernel=LinearKernel(), penalty=request.penalty
    ).fit([input_feature], target)

    probe = Feature(FEATURE_NAME, [0.0, feature_mean])
    ridge_probe = np.asarray(ridge_model.predict([probe]), dtype=np.float64)
    kernel_probe = np.asarray(kernel_model.predict([probe]), dtype=np.float64)

    grid = _sampling_grid(x_values, None, None)
    grid_feature = Feature(FEATURE_NAME, grid)
    ridge_heights = np.asarray(ridge_model.predict([grid_feature]), dtype=np.float64)
    kernel_heights = np.asarray(kernel_model.predict([grid_feature]), dtype=np.float64)

    return CentringResponse(
        mean=Point(x=feature_mean, y=target_mean),
        ridge=ControlLine(
            slope=float(ridge_model.coefficients[FEATURE_NAME]),
            intercept=float(ridge_probe[0]),
            at_mean=float(ridge_probe[1]),
            curve=_curve_of(grid, ridge_heights),
        ),
        kernel_ridge=ControlLine(
            slope=_implied_slope(kernel_model),
            intercept=float(kernel_probe[0]),
            at_mean=float(kernel_probe[1]),
            curve=_curve_of(grid, kernel_heights),
        ),
        target_only=_target_only_centred_line(
            x_values, y_values, request.penalty, grid
        ),
    )


COST_SEED = 11
"""The timing data is drawn once, from this seed."""

COST_ROWS = (50, 100, 200, 400, 800, 1600)
"""How many rows each fit in the row sweep sees, at two features."""

COST_FEATURES = (2, 4, 8, 16, 32, 64)
"""How many features each fit in the feature sweep sees, at two hundred rows."""

COST_FIXED_FEATURES = 2
COST_FIXED_ROWS = 200
COST_REPEATS = 5
"""Each timing is the median of this many runs."""

COST_QUERIES = 100
"""How many rows each timed prediction answers."""

COST_GAMMA = 0.1
"""The radial kernel every timed kernel fit uses."""


class CostPoint(BaseModel):
    """One size, and how long each model took to fit and to predict.

    Times are in milliseconds, the median of ``COST_REPEATS`` runs, and they
    include the library's own boundary checks, since those are part of what
    a fit costs. ``kernel_model_numbers`` is how many numbers the fitted
    kernel model keeps, its rows times its features plus one dual weight per
    row, and ``ridge_model_numbers`` the coefficients plus the intercept.
    """

    rows: int
    features: int
    kernel_fit_ms: float
    ridge_fit_ms: float
    kernel_predict_ms: float
    ridge_predict_ms: float
    kernel_model_numbers: int
    ridge_model_numbers: int


class CostResponse(BaseModel):
    """The two sweeps, rows growing at fixed features and the reverse."""

    repeats: int
    queries: int
    by_rows: list[CostPoint]
    by_features: list[CostPoint]


def _timed_ms(action: Callable[[], object]) -> float:
    """The median wall-clock time of the action over the repeats, in ms."""
    samples: list[float] = []
    for _ in range(COST_REPEATS):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000.0)

    return float(statistics.median(samples))


def _cost_point(generator: np.random.Generator, rows: int, features: int) -> CostPoint:
    """Draw one dataset of the given size and time both models on it."""
    matrix = generator.normal(size=(rows, features))
    weights = generator.normal(size=features)
    target_values = matrix @ weights + generator.normal(scale=0.1, size=rows)
    inputs = [
        Feature(f"feature_{index}", matrix[:, index]) for index in range(features)
    ]
    target = Feature(TARGET_NAME, target_values)
    queries = generator.normal(size=(COST_QUERIES, features))
    query_inputs = [
        Feature(f"feature_{index}", queries[:, index]) for index in range(features)
    ]

    kernel_model = KernelRidgeRegression(
        kernel=RadialBasisKernel(gamma=COST_GAMMA), penalty=1.0
    )
    ridge_model = RidgeRegression(penalty=1.0)
    kernel_fit_ms = _timed_ms(lambda: kernel_model.fit(inputs, target))
    ridge_fit_ms = _timed_ms(lambda: ridge_model.fit(inputs, target))
    kernel_predict_ms = _timed_ms(lambda: kernel_model.predict(query_inputs))
    ridge_predict_ms = _timed_ms(lambda: ridge_model.predict(query_inputs))

    return CostPoint(
        rows=rows,
        features=features,
        kernel_fit_ms=kernel_fit_ms,
        ridge_fit_ms=ridge_fit_ms,
        kernel_predict_ms=kernel_predict_ms,
        ridge_predict_ms=ridge_predict_ms,
        kernel_model_numbers=kernel_model.n_training_rows * (features + 1),
        ridge_model_numbers=len(ridge_model.coefficients) + 1,
    )


@cache
def _cost() -> CostResponse:
    generator = np.random.default_rng(COST_SEED)
    by_rows = [_cost_point(generator, rows, COST_FIXED_FEATURES) for rows in COST_ROWS]
    by_features = [
        _cost_point(generator, COST_FIXED_ROWS, features) for features in COST_FEATURES
    ]

    return CostResponse(
        repeats=COST_REPEATS,
        queries=COST_QUERIES,
        by_rows=by_rows,
        by_features=by_features,
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """Time both fits as the rows grow and as the features grow."""
    return _cost()


def _sampling_grid(
    x_values: list[float], curve_from: float | None, curve_to: float | None
) -> np.ndarray:
    """Evenly spaced x positions across the data, padded a little either side.

    An explicit range replaces the padding, which is how a curve is watched
    far past the last measurement.
    """
    lowest, highest = min(x_values), max(x_values)
    pad = CURVE_PAD_SHARE * ((highest - lowest) or 1.0)
    start = lowest - pad if curve_from is None else curve_from
    stop = highest + pad if curve_to is None else curve_to
    if stop <= start:
        raise HTTPException(status_code=422, detail="the curve range must run upward")

    return np.linspace(start, stop, CURVE_SAMPLES)


def _curve_of(grid: np.ndarray, heights: np.ndarray) -> list[CurvePoint]:
    """Pair each sampled x with the height a model answered there."""
    return [
        CurvePoint(x=float(x_value), y=float(height))
        for x_value, height in zip(grid, heights, strict=True)
    ]


def _implied_slope(kernel_model: KernelRidgeRegression) -> float:
    """The weight the dual weights add up to, under the linear kernel.

    The dual form never forms ``w = X'a``, which is the point of it, but on
    one centred feature that product is a single number, and it is the slope
    the fitted curve has. Reading it is what lets the page check the kernel
    fit against ridge's slope by hand.
    """
    rows = kernel_model.training_rows.values
    weights = kernel_model.dual_weights

    return float((rows.T @ weights)[0])


def _gram_eigenvalues(kernel_model: KernelRidgeRegression) -> np.ndarray:
    """The Gram matrix's eigenvalues, ascending, on the rows the fit kept."""
    rows = kernel_model.training_rows
    matrix = kernel_model.kernel.between(rows, rows).values

    return np.linalg.eigvalsh(matrix)


def _condition_number(eigenvalues: np.ndarray, penalty: float) -> float:
    """How ill-conditioned ``K + penalty I`` is, largest over smallest."""
    return float((eigenvalues[-1] + penalty) / (eigenvalues[0] + penalty))


def _target_only_centred_line(
    x_values: list[float],
    y_values: list[float],
    penalty: float,
    grid: np.ndarray,
) -> ControlLine:
    """The dual solve on uncentred features, which is the recorded mistake.

    The Gram matrix comes from the library's own linear kernel on the rows as
    they arrived. The target is centred, the features are not, and the solve
    is the same ``(K + penalty I) a = y`` the library performs. What comes out
    is a line whose slope is the weight the rows add up to and whose
    intercept is the target mean, because a prediction is ``k(x, X) a`` plus
    that mean, and at ``x = 0`` the kernel values are all zero.
    """
    raw_rows = RowBlock(
        np.asarray(x_values, dtype=np.float64).reshape(-1, 1), [FEATURE_NAME]
    )
    gram_matrix = LinearKernel().between(raw_rows, raw_rows).values
    target_mean = float(np.mean(y_values))
    centred_target = np.asarray(y_values, dtype=np.float64) - target_mean

    dual_weights = np.linalg.solve(
        gram_matrix + penalty * np.eye(len(x_values)), centred_target
    )
    slope = float((raw_rows.values.T @ dual_weights)[0])
    heights = slope * grid + target_mean

    return ControlLine(
        slope=slope,
        intercept=target_mean,
        at_mean=slope * float(np.mean(x_values)) + target_mean,
        curve=_curve_of(grid, heights),
    )
