"""Kernel ridge regression on draggable points, for the kernel-ridge page.

The page's claim is that ridge regression, rearranged so its rows appear only
inside dot products, can have those dot products swapped for a kernel and
then draws a curve through the data without one manufactured column. The fit
endpoint runs that claim on the library's ``KernelRidgeRegression`` under
whichever of its four kernels the reader picks, and fits the plain
``RidgeRegression`` on the same points beside it, because under the linear
kernel the two must coincide. That coincidence is the control the page
teaches, and the response carries the largest gap between the two sampled
curves so the widget can read it out rather than ask the eye to judge.

The fit also hands back what the kernel model actually learned, one dual
weight per training row, and for the linear kernel the slope those weights
imply, which is the weight the dual form never forms and the number the
worked example checks against ridge's slope by hand.

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

from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
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

MAX_DEGREE = 6
"""The highest power the polynomial kernel may raise to."""

MAX_GAMMA = 1000.0
"""How sharply the radial and sigmoid kernels may be tuned."""

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
    gamma: float = Field(gt=0.0, le=MAX_GAMMA)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return RadialBasisKernel(gamma=self.gamma)


class SigmoidKernelChoice(BaseModel):
    """A hyperbolic tangent of the scaled dot product, not always a kernel."""

    name: Literal["sigmoid"]
    gamma: float = Field(gt=0.0, le=MAX_GAMMA)
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
    """The points, the kernel, how hard to pull, and one x to be asked about."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice
    penalty: float = Field(gt=0.0, le=MAX_PENALTY)
    query_x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)

    model_config = {"extra": "forbid"}


class KernelRidgeFitResponse(BaseModel):
    """The kernel curve, the ridge line beside it, and what the fit learned.

    ``dual_weights`` holds one number per training row, in the order the
    points arrived, which is the whole of what the kernel model learned.
    ``implied_slope`` is the weight those dual weights add up to under the
    linear kernel, and ``None`` under any other, where no slope exists to
    report. ``largest_gap`` is the widest vertical distance between the two
    sampled curves, which is the control readout, zero to within rounding
    under the linear kernel.
    """

    curve: list[CurvePoint]
    ridge_curve: list[CurvePoint]
    dual_weights: list[float]
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

    grid = _sampling_grid(x_values)
    grid_feature = Feature(FEATURE_NAME, grid)
    kernel_heights = np.asarray(kernel_model.predict([grid_feature]), dtype=np.float64)
    ridge_heights = np.asarray(ridge_model.predict([grid_feature]), dtype=np.float64)

    query_feature = Feature(FEATURE_NAME, [request.query_x])
    kernel_at_query = np.asarray(
        kernel_model.predict([query_feature]), dtype=np.float64
    )
    ridge_at_query = np.asarray(ridge_model.predict([query_feature]), dtype=np.float64)

    dual_weights = np.asarray(kernel_model.dual_weights, dtype=np.float64)
    implied_slope = (
        _implied_slope(kernel_model)
        if isinstance(request.kernel, LinearKernelChoice)
        else None
    )

    return KernelRidgeFitResponse(
        curve=_curve_of(grid, kernel_heights),
        ridge_curve=_curve_of(grid, ridge_heights),
        dual_weights=[float(weight) for weight in dual_weights],
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
    )


class CentringRequest(BaseModel):
    """The points and the penalty for the linear-kernel control."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    penalty: float = Field(gt=0.0, le=MAX_PENALTY)

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

    grid = _sampling_grid(x_values)
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


def _sampling_grid(x_values: list[float]) -> np.ndarray:
    """Evenly spaced x positions across the data, padded a little either side."""
    lowest, highest = min(x_values), max(x_values)
    pad = CURVE_PAD_SHARE * ((highest - lowest) or 1.0)

    return np.linspace(lowest - pad, highest + pad, CURVE_SAMPLES)


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
    gram = LinearKernel().between(raw_rows, raw_rows).values
    target_mean = float(np.mean(y_values))
    centred_target = np.asarray(y_values, dtype=np.float64) - target_mean

    dual_weights = np.linalg.solve(
        gram + penalty * np.eye(len(x_values)), centred_target
    )
    slope = float((raw_rows.values.T @ dual_weights)[0])
    heights = slope * grid + target_mean

    return ControlLine(
        slope=slope,
        intercept=target_mean,
        at_mean=slope * float(np.mean(x_values)) + target_mean,
        curve=_curve_of(grid, heights),
    )
