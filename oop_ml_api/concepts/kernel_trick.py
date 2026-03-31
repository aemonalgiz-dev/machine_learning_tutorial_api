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
    PolynomialKernel,
    RadialBasisKernel,
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
