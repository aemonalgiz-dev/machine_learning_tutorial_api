"""Kernel principal components, for the kernel-pca page.

One endpoint. It takes a cloud of points and a kernel the reader chooses, fits
the library's ``KernelPrincipalComponentAnalysis`` keeping two directions, and
returns where every point lands along them, the two directions themselves and
the two scales their eigenvalues are read at. Beside that answer it fits
ordinary ``PrincipalComponentAnalysis`` on the same points, which is the
contrast the page draws, a rotation of the plane against a rearrangement of it.

A kernel direction is not a weight per feature. It lives in a space whose
coordinates have no names, so what the library hands back is one coefficient
per training row, and the endpoint passes those through as they are. The raw
Gram eigenvalue is the one figure the library reports only after scaling, since
its ``_solve`` divides by ``n - 1`` before calling the result a variance. The
endpoint multiplies that factor back so the page can show both scales side by
side, the raw eigenvalue the eigenvector is normalised by and the variance the
shares are built from. The page's test checks the raw figure a second way, from
the coefficients alone, because the two routes agreeing is the whole lesson.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Feature,
    Kernel,
    KernelPrincipalComponentAnalysis,
    LinearKernel,
    PolynomialKernel,
    PrincipalComponentAnalysis,
    RadialBasisKernel,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["kernel-pca"])

DIRECTIONS_KEPT = 2
"""The widget draws a plane, so two directions are kept and no more."""

POLYNOMIAL_DEGREE = 2
"""The squared kernel, the degree the kernel trick page works by hand."""

MAX_GAMMA = 100.0
"""The sharpest the radial and squared kernels may be set.

At a hundred every point on the page's clouds is similar only to itself, which
is already the failure the page teaches, and anything sharper only underflows.
"""

KernelName = Literal["linear", "polynomial", "rbf"]


class KernelPcaRequest(BaseModel):
    """The cloud, the reading of similarity to decompose it under, and how sharp."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelName
    gamma: float = Field(default=1.0, gt=0.0, le=MAX_GAMMA)

    model_config = {"extra": "forbid"}


class KernelComponentDocument(BaseModel):
    """One direction in the implied space, with its eigenvalue at both scales.

    ``raw_eigenvalue`` is what the solver returned for the centred Gram matrix
    and is ``n - 1`` times ``variance``. ``share`` is the variance over the
    total across every direction, kept and discarded. ``row_coefficients``
    holds one weight per training row, which is the only description the
    direction has.
    """

    name: str
    raw_eigenvalue: float
    variance: float
    share: float
    row_coefficients: list[float]


class PlaneCoordinates(BaseModel):
    """Where one point lands along the first two directions."""

    first: float
    second: float


class OrdinaryComponentDocument(BaseModel):
    """One of ordinary PCA's directions on the same cloud, for contrast."""

    variance: float
    share: float


class OrdinaryDecomposition(BaseModel):
    """Ordinary PCA's answer on the same points, in the same order."""

    components: list[OrdinaryComponentDocument]
    coordinates: list[PlaneCoordinates]


class KernelPcaResponse(BaseModel):
    """The kernel decomposition, and ordinary PCA's answer beside it.

    ``coordinates`` is in the order the points were sent.
    ``first_row_kernel_values`` is the first row of the centred kernel matrix,
    one value per training row, which is what the first point's coordinate is
    summed from. A sign on any direction is the solver's choice rather than a
    fact about the data, so a whole column may come back negated between two
    fits that differ by one dragged point.
    """

    n_rows: int
    total_variance: float
    components: list[KernelComponentDocument]
    coordinates: list[PlaneCoordinates]
    first_row_kernel_values: list[float]
    ordinary: OrdinaryDecomposition


def _kernel_for(name: KernelName, gamma: float) -> Kernel:
    """The library kernel the request names, sharpened by its gamma."""
    if name == "linear":
        return LinearKernel()
    if name == "polynomial":
        return PolynomialKernel(degree=POLYNOMIAL_DEGREE, gamma=gamma)
    return RadialBasisKernel(gamma=gamma)


@router.post("/concepts/kernel-pca/analyze", response_model=KernelPcaResponse)
def analyze(request: KernelPcaRequest) -> KernelPcaResponse:
    """Decompose the cloud through the chosen kernel, and through no kernel at all."""
    features = [
        Feature("x", [point.x for point in request.points]),
        Feature("y", [point.y for point in request.points]),
    ]
    n_rows = len(request.points)

    kernel_model = KernelPrincipalComponentAnalysis(
        kernel=_kernel_for(request.kernel, request.gamma),
        n_components=DIRECTIONS_KEPT,
    ).fit(features)

    if kernel_model.components.n_components < DIRECTIONS_KEPT:
        raise HTTPException(
            status_code=422,
            detail=(
                "under this kernel the cloud spreads along only one direction, "
                "and the picture needs two"
            ),
        )

    components = [
        KernelComponentDocument(
            name=component.name,
            raw_eigenvalue=float(component.variance * (n_rows - 1)),
            variance=float(component.variance),
            share=float(share),
            row_coefficients=[float(value) for value in component.row_coefficients],
        )
        for component, share in zip(
            kernel_model.components,
            kernel_model.components.variance_shares,
            strict=True,
        )
    ]

    centred_kernel_values = np.asarray(
        kernel_model.query_matrix(features).values, dtype=np.float64
    )

    ordinary_model = PrincipalComponentAnalysis().fit(features)

    return KernelPcaResponse(
        n_rows=n_rows,
        total_variance=float(kernel_model.components.total_variance),
        components=components,
        coordinates=_plane_coordinates(kernel_model.transform(features)),
        first_row_kernel_values=[float(value) for value in centred_kernel_values[0]],
        ordinary=OrdinaryDecomposition(
            components=[
                OrdinaryComponentDocument(
                    variance=float(component.variance), share=float(share)
                )
                for component, share in zip(
                    ordinary_model.components,
                    ordinary_model.components.variance_shares,
                    strict=True,
                )
            ],
            coordinates=_plane_coordinates(ordinary_model.transform(features)),
        ),
    )


def _plane_coordinates(transformed: list[Feature]) -> list[PlaneCoordinates]:
    """Pair the first two transformed columns up, one entry per point."""
    first_column = np.asarray(transformed[0].values, dtype=np.float64)
    second_column = np.asarray(transformed[1].values, dtype=np.float64)

    return [
        PlaneCoordinates(first=float(first), second=float(second))
        for first, second in zip(first_column, second_column, strict=True)
    ]
