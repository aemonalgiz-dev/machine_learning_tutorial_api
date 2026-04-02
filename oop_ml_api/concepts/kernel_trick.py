"""A kernelised classifier on the clinic's vitals, for the kernel trick page.

The library's ``SupportVectorClassifier`` reads its data only through inner
products, which is the structural fact the page teaches, so swapping its
kernel changes the space the straight boundary is drawn in without changing
the machinery. The endpoint standardizes the two vitals first, since a kernel
built on distances and products inherits the units of its data, the same trap
the k-nearest neighbours page worked, and returns the accuracy alongside the
fitted model's answer across the plane, where the linear kernel's straight cut
and the swapped kernels' closed curves become visible.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Feature,
    LinearKernel,
    LogisticRegression,
    PolynomialKernel,
    RadialBasisKernel,
    RowBlock,
    Standardizer,
    SupportVectorClassifier,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    GRID_CELLS,
    LabelledPoint,
    RegionGrid,
)
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/kernel-trick", tags=["kernel-trick"])

POLYNOMIAL_DEGREE = 2
"""The widget's polynomial kernel squares, matching the page's worked identity."""


class KernelRequest(BaseModel):
    """The patients, and which reading of similarity to hand the machine."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: Literal["linear", "polynomial", "rbf"]

    model_config = {"extra": "forbid"}


class KernelResponse(BaseModel):
    """The fitted boundary's accuracy and its answer across the plane."""

    accuracy: float
    regions: RegionGrid


def _kernel_for(name: str):
    if name == "linear":
        return LinearKernel()
    if name == "polynomial":
        return PolynomialKernel(degree=POLYNOMIAL_DEGREE)
    return RadialBasisKernel()


@router.post("/classify", response_model=KernelResponse)
def classify(request: KernelRequest) -> KernelResponse:
    """Standardize the vitals, fit under the chosen kernel, shade the plane."""
    temperature = Feature("temperature", [point.x for point in request.points])
    heart_rate = Feature("heart_rate", [point.y for point in request.points])
    labels = Feature("healthy", [float(point.label) for point in request.points])

    standardizer = Standardizer().fit([temperature, heart_rate])
    standardized = standardizer.transform([temperature, heart_rate])

    model = SupportVectorClassifier(kernel=_kernel_for(request.kernel)).fit(
        standardized, labels
    )

    return KernelResponse(
        accuracy=float(model.score(standardized, labels)),
        regions=_regions(model, standardizer, request.points),
    )


class LiftedPoint(BaseModel):
    """One patient carried through the worked mapping, with their label."""

    u: float
    v: float
    w: float
    label: int


class SeparatingPlane(BaseModel):
    """The flat boundary found in the lifted space, a·u + b·v + c·w = d."""

    a: float
    b: float
    c: float
    d: float


class LiftResponse(BaseModel):
    """The clinic lifted into three dimensions, and the plane that cuts it."""

    points: list[LiftedPoint]
    plane: SeparatingPlane
    accuracy: float


@router.post("/lift", response_model=LiftResponse)
def lift(request: KernelRequest) -> LiftResponse:
    """Carry the patients through the worked φ and find the flat boundary.

    The mapping is the page's worked one, squared and crossed terms of the
    standardized vitals, and the boundary comes from fitting the library's
    logistic regression on the three lifted columns, whose coefficients and
    intercept are exactly a plane. The plane is returned with its normal
    scaled to unit length so it draws the same at any fit steepness. The
    ``kernel`` field of the request is ignored here, since the lift exists to
    show what the squared kernel implies.
    """
    temperature = Feature("temperature", [point.x for point in request.points])
    heart_rate = Feature("heart_rate", [point.y for point in request.points])

    standardizer = Standardizer().fit([temperature, heart_rate])
    standardized = standardizer.transform([temperature, heart_rate])
    first = np.asarray(standardized[0].values, dtype=np.float64)
    second = np.asarray(standardized[1].values, dtype=np.float64)

    u_values = first**2
    v_values = np.sqrt(2.0) * first * second
    w_values = second**2

    lifted = [
        Feature("u", u_values),
        Feature("v", v_values),
        Feature("w", w_values),
    ]
    labels = Feature("healthy", [float(point.label) for point in request.points])
    boundary = LogisticRegression().fit(lifted, labels)

    named = {
        coefficient.name: float(coefficient.value)
        for coefficient in boundary.coefficients
    }
    normal = np.array([named["u"], named["v"], named["w"]], dtype=np.float64)
    length = float(np.linalg.norm(normal)) or 1.0

    return LiftResponse(
        points=[
            LiftedPoint(u=float(u), v=float(v), w=float(w), label=point.label)
            for u, v, w, point in zip(
                u_values, v_values, w_values, request.points, strict=True
            )
        ],
        plane=SeparatingPlane(
            a=float(normal[0] / length),
            b=float(normal[1] / length),
            c=float(normal[2] / length),
            d=float(-boundary.intercept / length),
        ),
        accuracy=float(boundary.score(lifted, labels)),
    )


def _regions(
    model: SupportVectorClassifier,
    standardizer: Standardizer,
    points: list[LabelledPoint],
) -> RegionGrid:
    """The fitted boundary's class at every cell, through the same scaling."""
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
    y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad

    lattice_x, lattice_y = np.meshgrid(
        np.linspace(x_min, x_max, GRID_CELLS),
        np.linspace(y_min, y_max, GRID_CELLS),
    )
    lattice = standardizer.transform(
        [
            Feature("temperature", lattice_x.ravel()),
            Feature("heart_rate", lattice_y.ravel()),
        ]
    )
    predictions = np.asarray(model.predict(lattice), dtype=np.float64).reshape(
        GRID_CELLS, GRID_CELLS
    )

    return RegionGrid(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        cells=GRID_CELLS,
        labels=[[int(value) for value in row] for row in predictions],
    )


SIMILARITY_SAMPLES = 161
"""How many positions the similarity curves are sampled at."""

SIMILARITY_REFERENCE = 1.5
"""Where the reference point stands on the one-dimensional axis."""

SIMILARITY_SPAN = 4.0
"""The sweep runs from minus this to this."""

SIMILARITY_KERNELS = (
    LinearKernel(),
    PolynomialKernel(degree=POLYNOMIAL_DEGREE),
    RadialBasisKernel(gamma=0.5),
    RadialBasisKernel(gamma=2.0),
)
"""The kernels the chart compares, two widths of radial so gamma is visible."""


class SimilarityCurve(BaseModel):
    """One kernel's answer at every swept position, labelled by its own formula."""

    label: str
    values: list[float]


class SimilarityCurveResponse(BaseModel):
    """Each kernel's similarity to one fixed reference point, swept along a line.

    The labels are the library kernels' own descriptions, so the chart cannot
    drift from what the code actually computes.
    """

    positions: list[float]
    reference: float
    curves: list[SimilarityCurve]


@router.post("/similarity-curve", response_model=SimilarityCurveResponse)
def similarity_curve() -> SimilarityCurveResponse:
    """Sweep a point along a line and ask each kernel how alike the pair is.

    No request body, since the sweep is a fixed illustration rather than a
    fit, the same standing as the ensemble pages' formula curves.
    """
    positions = np.linspace(-SIMILARITY_SPAN, SIMILARITY_SPAN, SIMILARITY_SAMPLES)
    swept = RowBlock(positions.reshape(-1, 1), ["x"])
    reference = RowBlock(np.asarray([[SIMILARITY_REFERENCE]], dtype=np.float64), ["x"])

    curves = [
        SimilarityCurve(
            label=kernel.description,
            values=[
                float(value)
                for value in np.asarray(
                    kernel.between(swept, reference), dtype=np.float64
                ).ravel()
            ],
        )
        for kernel in SIMILARITY_KERNELS
    ]

    return SimilarityCurveResponse(
        positions=[float(value) for value in positions],
        reference=SIMILARITY_REFERENCE,
        curves=curves,
    )
