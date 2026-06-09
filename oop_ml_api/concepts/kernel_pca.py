"""Kernel principal components, for the kernel-pca page.

Four endpoints and one probe. ``analyze`` is the playground's: a cloud
decomposed through a kernel the reader chooses, keeping two directions, with
ordinary ``PrincipalComponentAnalysis`` on the same points beside it for the
contrast the page draws, a rotation of the plane against a rearrangement of
it. ``gram-route`` is the page's second Part, PCA written with nothing but
inner products: the uncentred and centred Gram matrices under the linear
kernel, their eigenvalues against the scatter matrix's, and the coordinates
the two routes reach, which agree to rounding. ``lifted`` is the full
anatomy of one kernel fit, every intermediate the later Parts quote: the
kernel table before and after centring, every eigenvalue including the most
negative one before it was clamped, every kept direction's row
coefficients, each person's lifted deviation and how much of it the kept
directions carry, and the two facts about the coefficient vectors the page
measures rather than asserts. ``sweep`` runs the radial kernel's reach across
a list of gammas and reports what the first axis does at each. ``edges``
runs the failure contracts live and returns what came back.

A kernel direction is not a weight per feature. It lives in a space whose
coordinates have no names, so what the library hands back is one coefficient
per training row, and the endpoints pass those through as they are. The raw
Gram eigenvalue is the one figure the library reports only after scaling,
since its ``_solve`` divides by ``n - 1`` before calling the result a
variance. The endpoints multiply that factor back so the page can show both
scales side by side, the raw eigenvalue the eigenvector is normalised by and
the variance the shares are built from. The tests check the raw figure a
second way, from the coefficients alone, because the two routes agreeing is
the whole lesson.

Two of the page's claims turned out to be measurements against the textbook
rather than for it, and the endpoints exist partly so the page can say so
with numbers. The coefficient vectors of distinct components are
perpendicular in the ordinary sense, to rounding, since they are eigenvectors
of one symmetric table; what they are not is unit length. And on the page's
arc the radial kernel does not unroll the bend into a line, it folds it,
which ``order_reversals_along_x`` counts.
"""

from __future__ import annotations

from itertools import combinations
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
    SigmoidKernel,
)
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["kernel-pca"])

DIRECTIONS_KEPT = 2
"""The playground draws a plane, so two directions are kept and no more."""

POLYNOMIAL_DEGREE = 2
"""The squared kernel, the degree the kernel trick page works by hand."""

MAX_GAMMA = 100.0
"""The sharpest the radial and squared kernels may be set.

At a hundred every point on the page's clouds is similar only to itself, which
is already the failure the page teaches, and anything sharper only underflows.
"""

MAX_SWEEP_LENGTH = 60
"""How many gammas one sweep request may ask for."""

FEATURE_NAMES = ("height", "weight")
"""The page's people are measured two ways, and a kernel pairs rows over both."""

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
    total across every direction, kept and discarded, and
    ``cumulative_share`` the running total through this one.
    ``row_coefficients`` holds one weight per training row, which is the only
    description the direction has.
    """

    name: str
    raw_eigenvalue: float
    variance: float
    share: float
    cumulative_share: float
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


def _features(points: list[PlainPoint]) -> list[Feature]:
    return [
        Feature(FEATURE_NAMES[0], [point.x for point in points]),
        Feature(FEATURE_NAMES[1], [point.y for point in points]),
    ]


def _block(points: list[PlainPoint]) -> RowBlock:
    return rows_of(
        np.asarray([[point.x, point.y] for point in points], dtype=np.float64),
        FEATURE_NAMES,
    )


def _matrix(values: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _component_documents(
    fitted: KernelPrincipalComponentAnalysis, n_rows: int
) -> list[KernelComponentDocument]:
    components = fitted.components
    return [
        KernelComponentDocument(
            name=component.name,
            raw_eigenvalue=float(component.variance * (n_rows - 1)),
            variance=float(component.variance),
            share=float(share),
            cumulative_share=float(cumulative),
            row_coefficients=[float(value) for value in component.row_coefficients],
        )
        for component, share, cumulative in zip(
            components,
            components.variance_shares,
            components.cumulative_shares,
            strict=True,
        )
    ]


def _transformed(
    fitted: KernelPrincipalComponentAnalysis | PrincipalComponentAnalysis,
    features: list[Feature],
) -> np.ndarray:
    """Every kept coordinate of every row, rows down and components across."""
    return np.column_stack(
        [
            np.asarray(feature.values, dtype=np.float64)
            for feature in fitted.transform(features)
        ]
    )


def _plane_coordinates(transformed: np.ndarray) -> list[PlaneCoordinates]:
    """Pair the first two transformed columns up, one entry per point."""
    return [
        PlaneCoordinates(first=float(row[0]), second=float(row[1]))
        for row in transformed
    ]


def _order_reversals_along_x(points: list[PlainPoint], coordinate: np.ndarray) -> int:
    """How often the coordinate changes direction when the people are read by height.

    Zero means the axis orders the people the way height does, in one
    direction or the other. On the page's arc, ordinary PCA's first axis
    scores zero and the radial kernel's first axis scores five, which is the
    fold the page reports instead of the unrolling the textbook promises.
    """
    order = np.argsort([point.x for point in points], kind="stable")
    steps = np.diff(coordinate[order])
    return int(min(np.sum(steps < 0), np.sum(steps > 0)))


def _first_axis_gap(coordinate: np.ndarray, inner_count: int) -> float:
    """How far apart the two groups are along the axis, negative if they overlap.

    Positive means every inner person sits to one side of every outer person
    and the number is the empty stretch between the nearest pair; negative
    means the groups interleave and the number is how far they overlap.
    """
    inner = coordinate[:inner_count]
    outer = coordinate[inner_count:]
    return float(max(inner.min() - outer.max(), outer.min() - inner.max()))


def _check_inner_count(inner_count: int | None, n_rows: int) -> None:
    if inner_count is not None and inner_count >= n_rows:
        raise HTTPException(
            status_code=422,
            detail=(
                f"inner_count must leave at least one person outside the group; "
                f"got {inner_count} of {n_rows}"
            ),
        )


@router.post("/concepts/kernel-pca/analyze", response_model=KernelPcaResponse)
def analyze(request: KernelPcaRequest) -> KernelPcaResponse:
    """Decompose the cloud through the chosen kernel, and through no kernel at all."""
    features = _features(request.points)
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

    centred_kernel_values = np.asarray(
        kernel_model.query_matrix(features).values, dtype=np.float64
    )

    ordinary_model = PrincipalComponentAnalysis().fit(features)

    return KernelPcaResponse(
        n_rows=n_rows,
        total_variance=float(kernel_model.components.total_variance),
        components=_component_documents(kernel_model, n_rows),
        coordinates=_plane_coordinates(_transformed(kernel_model, features)),
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
            coordinates=_plane_coordinates(_transformed(ordinary_model, features)),
        ),
    )


class PointsRequest(BaseModel):
    """The cloud, and nothing else."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class GramRouteResponse(BaseModel):
    """PCA by two routes on one cloud, under the linear kernel.

    ``uncentred_gram`` is every pair's dot product of the raw rows and
    ``centred_gram`` the same table after the centring identity, which is the
    table the fit decomposes. ``uncentred_top_share`` is how much of the
    uncentred table's trace its largest eigenvalue takes, and
    ``uncentred_top_eigenvector`` is the direction that eigenvalue names, one
    weight per person, nearly uniform because it points at the mean, and
    ``uncentred_eigenvalues`` is that table's whole spectrum, largest first,
    so the ones left under the mean direction can be read beside it.
    ``scatter_eigenvalues`` come from the two-by-two scatter matrix the PCA
    page builds and ``gram_eigenvalues`` from the n-by-n centred Gram
    matrix, clamped at zero; the non-zero ones agree. ``coordinates`` is the
    Gram route's placement of every person and ``ordinary_coordinates`` the
    covariance route's, and ``largest_coordinate_gap`` is the biggest
    difference between them once each column's sign is dropped.
    ``coefficient_dot_products`` is the coefficient matrix against its own
    transpose, ``kernel_metric_products`` the same with the centred Gram
    matrix in the middle, and ``first_row_centred_against_training`` the
    first person's kernel row centred the way a query is, which reproduces
    the first row of ``centred_gram``.
    """

    n_rows: int
    divisor: int
    uncentred_gram: list[list[float]]
    centred_gram: list[list[float]]
    uncentred_top_eigenvalue: float
    uncentred_top_share: float
    uncentred_top_eigenvector: list[float]
    uncentred_eigenvalues: list[float]
    scatter_matrix: list[list[float]]
    scatter_eigenvalues: list[float]
    gram_eigenvalues: list[float]
    total_variance: float
    components: list[KernelComponentDocument]
    coordinates: list[PlaneCoordinates]
    ordinary_variances: list[float]
    ordinary_shares: list[float]
    ordinary_coordinates: list[PlaneCoordinates]
    largest_coordinate_gap: float
    coefficient_squared_lengths: list[float]
    coefficient_dot_products: list[list[float]]
    kernel_metric_products: list[list[float]]
    first_row_centred_against_training: list[float]


@router.post("/concepts/kernel-pca/gram-route", response_model=GramRouteResponse)
def gram_route(request: PointsRequest) -> GramRouteResponse:
    """Decompose the cloud by the Gram matrix and by the covariance matrix, and compare."""
    features = _features(request.points)
    n_rows = len(request.points)
    block = _block(request.points)
    kernel = LinearKernel()

    uncentred = kernel.between(block, block)
    centred = uncentred.centred()
    uncentred_values = uncentred.values
    centred_values = centred.values

    uncentred_eigenvalues, uncentred_eigenvectors = np.linalg.eigh(uncentred_values)
    gram_eigenvalues = np.maximum(np.linalg.eigvalsh(centred_values)[::-1], 0.0)

    raw = np.asarray([[point.x, point.y] for point in request.points], dtype=np.float64)
    deviations = raw - raw.mean(axis=0)
    scatter = deviations.T @ deviations
    scatter_eigenvalues = np.linalg.eigvalsh(scatter)[::-1]

    kernel_model = KernelPrincipalComponentAnalysis(kernel=kernel).fit(features)
    ordinary_model = PrincipalComponentAnalysis().fit(features)

    kernel_coordinates = _transformed(kernel_model, features)
    ordinary_coordinates = _transformed(ordinary_model, features)

    if kernel_coordinates.shape[1] < DIRECTIONS_KEPT:
        raise HTTPException(
            status_code=422,
            detail=(
                "the cloud spreads along only one direction, and the comparison "
                "needs two"
            ),
        )

    shared = min(kernel_coordinates.shape[1], ordinary_coordinates.shape[1])
    largest_gap = float(
        np.max(
            np.abs(
                np.abs(kernel_coordinates[:, :shared])
                - np.abs(ordinary_coordinates[:, :shared])
            )
        )
    )

    coefficients = kernel_model.components.coefficients
    first_row_query = kernel.between(_block(request.points[:1]), block).centred_against(
        uncentred
    )

    return GramRouteResponse(
        n_rows=n_rows,
        divisor=n_rows - 1,
        uncentred_gram=_matrix(uncentred_values),
        centred_gram=_matrix(centred_values),
        uncentred_top_eigenvalue=float(uncentred_eigenvalues[-1]),
        uncentred_top_share=float(
            uncentred_eigenvalues[-1] / np.trace(uncentred_values)
        ),
        uncentred_top_eigenvector=[
            float(value) for value in uncentred_eigenvectors[:, -1]
        ],
        uncentred_eigenvalues=[float(value) for value in uncentred_eigenvalues[::-1]],
        scatter_matrix=_matrix(scatter),
        scatter_eigenvalues=[float(value) for value in scatter_eigenvalues],
        gram_eigenvalues=[float(value) for value in gram_eigenvalues],
        total_variance=float(kernel_model.components.total_variance),
        components=_component_documents(kernel_model, n_rows),
        coordinates=_plane_coordinates(kernel_coordinates),
        ordinary_variances=[
            float(component.variance) for component in ordinary_model.components
        ],
        ordinary_shares=[
            float(share) for share in ordinary_model.components.variance_shares
        ],
        ordinary_coordinates=_plane_coordinates(ordinary_coordinates),
        largest_coordinate_gap=largest_gap,
        coefficient_squared_lengths=[
            float(value) for value in np.sum(coefficients**2, axis=1)
        ],
        coefficient_dot_products=_matrix(coefficients @ coefficients.T),
        kernel_metric_products=_matrix(coefficients @ centred_values @ coefficients.T),
        first_row_centred_against_training=[
            float(value) for value in first_row_query.values[0]
        ],
    )


class LiftedRequest(BaseModel):
    """One kernel fit, taken apart.

    ``n_components`` is how many directions to keep, at most the row count,
    which the library refuses past. ``inner_count`` names how many of the
    first points form a group, so the response can say whether the first
    axis separates that group from the rest; leave it out for a cloud with
    no groups.
    """

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelName
    gamma: float = Field(default=1.0, gt=0.0, le=MAX_GAMMA)
    n_components: int = Field(default=DIRECTIONS_KEPT, ge=1, le=MAX_POINTS)
    inner_count: int | None = Field(default=None, ge=1)

    model_config = {"extra": "forbid"}


class LiftedResponse(BaseModel):
    """Every intermediate of one kernel fit.

    ``uncentred_kernel`` is the kernel table of the training rows and
    ``centred_kernel`` the same after the centring identity.
    ``raw_eigenvalues`` are the centred table's, largest first, clamped at
    zero the way the fit clamps them; ``most_negative_eigenvalue`` is the
    smallest one before clamping and ``negative_count`` how many were below
    zero by more than rounding, which is the witness that a kernel was not
    one. ``rank`` counts the eigenvalues large enough to be directions.
    ``coordinates`` holds every kept coordinate of every person, rows in the
    order sent. ``lifted_squared_deviations`` is each person's squared
    distance from the lifted mean, the diagonal of the centred table, and
    ``kept_squared`` how much of it the kept directions carry;
    ``total_kept_share`` is the kept eigenvalues over all of them.
    ``largest_coefficient_dot_product`` is the biggest dot product between
    the coefficient vectors of two different kept directions, and
    ``largest_kernel_metric_gap`` how far the coefficient matrix with the
    centred table in the middle sits from the identity. ``reach`` is one
    over the square root of gamma, the distance at which the radial kernel
    has fallen to about a third. ``median_squared_distance`` is over every
    pair of people. ``order_reversals_along_x`` counts how often the first
    axis turns back when the people are read by height, and
    ``first_axis_gap`` is the empty stretch between the two groups along it,
    negative where they overlap, present only when ``inner_count`` was sent.
    ``ordinary_coordinates`` and ``ordinary_order_reversals_along_x`` are
    ordinary PCA's placement of the same people and the same count on its
    first axis, so a widget can show the two side by side.
    """

    n_rows: int
    divisor: int
    reach: float
    uncentred_kernel: list[list[float]]
    centred_kernel: list[list[float]]
    raw_eigenvalues: list[float]
    most_negative_eigenvalue: float
    negative_count: int
    rank: int
    total_variance: float
    components: list[KernelComponentDocument]
    coordinates: list[list[float]]
    lifted_squared_deviations: list[float]
    kept_squared: list[float]
    total_kept_share: float
    largest_coefficient_dot_product: float
    largest_kernel_metric_gap: float
    median_squared_distance: float
    order_reversals_along_x: int
    first_axis_gap: float | None
    ordinary_coordinates: list[PlaneCoordinates]
    ordinary_order_reversals_along_x: int


EIGENVALUE_NOISE = 1e-10
"""Below this an eigenvalue is rounding, the library's own threshold."""


@router.post("/concepts/kernel-pca/lifted", response_model=LiftedResponse)
def lifted(request: LiftedRequest) -> LiftedResponse:
    """Fit through the chosen kernel and return every intermediate."""
    features = _features(request.points)
    n_rows = len(request.points)
    _check_inner_count(request.inner_count, n_rows)
    block = _block(request.points)
    kernel = _kernel_for(request.kernel, request.gamma)

    uncentred = kernel.between(block, block)
    centred = uncentred.centred()
    centred_values = centred.values
    unclamped = np.linalg.eigvalsh(centred_values)[::-1]
    raw_eigenvalues = np.maximum(unclamped, 0.0)

    fitted = KernelPrincipalComponentAnalysis(
        kernel=kernel, n_components=request.n_components
    ).fit(features)
    coordinates = _transformed(fitted, features)
    coefficients = fitted.components.coefficients

    dot_products = coefficients @ coefficients.T
    off_diagonal = dot_products - np.diag(np.diag(dot_products))
    metric_products = coefficients @ centred_values @ coefficients.T
    identity = np.eye(coefficients.shape[0])

    raw = np.asarray([[point.x, point.y] for point in request.points], dtype=np.float64)
    squared_distances = [
        float(np.sum((raw[left] - raw[right]) ** 2))
        for left, right in combinations(range(n_rows), 2)
    ]

    lifted_deviations = np.diag(centred_values)
    kept = np.sum(coordinates**2, axis=1)
    total_raw = float(np.sum(raw_eigenvalues))

    ordinary_coordinates = _transformed(
        PrincipalComponentAnalysis().fit(features), features
    )

    return LiftedResponse(
        n_rows=n_rows,
        divisor=n_rows - 1,
        reach=float(1.0 / np.sqrt(request.gamma)),
        uncentred_kernel=_matrix(uncentred.values),
        centred_kernel=_matrix(centred_values),
        raw_eigenvalues=[float(value) for value in raw_eigenvalues],
        most_negative_eigenvalue=float(unclamped.min()),
        negative_count=int(np.sum(unclamped < -EIGENVALUE_NOISE)),
        rank=int(np.sum(raw_eigenvalues >= EIGENVALUE_NOISE)),
        total_variance=float(fitted.components.total_variance),
        components=_component_documents(fitted, n_rows),
        coordinates=_matrix(coordinates),
        lifted_squared_deviations=[float(value) for value in lifted_deviations],
        kept_squared=[float(value) for value in kept],
        total_kept_share=float(np.sum(coordinates**2) / total_raw)
        if total_raw > 0
        else 0.0,
        largest_coefficient_dot_product=float(np.max(np.abs(off_diagonal)))
        if coefficients.shape[0] > 1
        else 0.0,
        largest_kernel_metric_gap=float(np.max(np.abs(metric_products - identity))),
        median_squared_distance=float(np.median(squared_distances)),
        order_reversals_along_x=_order_reversals_along_x(
            request.points, coordinates[:, 0]
        ),
        first_axis_gap=(
            _first_axis_gap(coordinates[:, 0], request.inner_count)
            if request.inner_count is not None
            else None
        ),
        ordinary_coordinates=_plane_coordinates(ordinary_coordinates),
        ordinary_order_reversals_along_x=_order_reversals_along_x(
            request.points, ordinary_coordinates[:, 0]
        ),
    )


class SweepRequest(BaseModel):
    """The cloud and the list of gammas to run the radial kernel across."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    gammas: list[float] = Field(min_length=1, max_length=MAX_SWEEP_LENGTH)
    inner_count: int | None = Field(default=None, ge=1)

    model_config = {"extra": "forbid"}


class SweepEntry(BaseModel):
    """What one gamma did to the first two directions."""

    gamma: float
    reach: float
    first_share: float
    second_share: float
    cumulative_two: float
    total_variance: float
    rank: int
    order_reversals_along_x: int
    first_axis_gap: float | None


class SweepResponse(BaseModel):
    """One entry per gamma, in the order asked.

    ``median_squared_distance`` is over every pair of people, so a widget can
    mark where the usual rule of thumb, one over the median, falls on the
    sweep.
    """

    entries: list[SweepEntry]
    median_squared_distance: float


@router.post("/concepts/kernel-pca/sweep", response_model=SweepResponse)
def sweep(request: SweepRequest) -> SweepResponse:
    """Refit under the radial kernel at every gamma and report the first axis."""
    features = _features(request.points)
    n_rows = len(request.points)
    _check_inner_count(request.inner_count, n_rows)
    block = _block(request.points)
    raw = np.asarray([[point.x, point.y] for point in request.points], dtype=np.float64)
    squared_distances = [
        float(np.sum((raw[left] - raw[right]) ** 2))
        for left, right in combinations(range(n_rows), 2)
    ]

    entries: list[SweepEntry] = []
    for gamma in request.gammas:
        if not 0.0 < gamma <= MAX_GAMMA:
            raise HTTPException(
                status_code=422,
                detail=f"every gamma must lie in (0, {MAX_GAMMA}]; got {gamma}",
            )
        kernel = RadialBasisKernel(gamma=gamma)
        fitted = KernelPrincipalComponentAnalysis(kernel=kernel).fit(features)
        shares = fitted.components.variance_shares
        coordinates = _transformed(fitted, features)
        centred_values = kernel.between(block, block).centred().values
        raw_eigenvalues = np.maximum(np.linalg.eigvalsh(centred_values), 0.0)
        entries.append(
            SweepEntry(
                gamma=gamma,
                reach=float(1.0 / np.sqrt(gamma)),
                first_share=float(shares[0]),
                second_share=float(shares[1]) if len(shares) > 1 else 0.0,
                cumulative_two=float(sum(shares[:2])),
                total_variance=float(fitted.components.total_variance),
                rank=int(np.sum(raw_eigenvalues >= EIGENVALUE_NOISE)),
                order_reversals_along_x=_order_reversals_along_x(
                    request.points, coordinates[:, 0]
                ),
                first_axis_gap=(
                    _first_axis_gap(coordinates[:, 0], request.inner_count)
                    if request.inner_count is not None
                    else None
                ),
            )
        )

    return SweepResponse(
        entries=entries, median_squared_distance=float(np.median(squared_distances))
    )


class EdgeOutcome(BaseModel):
    """One edge run against the library, and what came back.

    ``accepted`` is whether the library computed an answer; ``detail`` is the
    refusal's own message or a description of what was accepted;
    ``figure`` is the one number the page quotes from the case, when there
    is one.
    """

    edge: str
    accepted: bool
    detail: str
    figure: float | None


class EdgesResponse(BaseModel):
    """Every edge the page's contracts table quotes, probed live."""

    outcomes: list[EdgeOutcome]


ARC = [
    PlainPoint(x=120, y=80),
    PlainPoint(x=128, y=62),
    PlainPoint(x=136, y=49),
    PlainPoint(x=144, y=40),
    PlainPoint(x=152, y=35),
    PlainPoint(x=160, y=33),
    PlainPoint(x=168, y=35),
    PlainPoint(x=176, y=40),
    PlainPoint(x=184, y=49),
    PlainPoint(x=192, y=62),
    PlainPoint(x=200, y=80),
    PlainPoint(x=156, y=33),
]
"""The PCA page's twelve people along a bend, which the edges are probed on."""

MEASURED_FOUR = [
    PlainPoint(x=180, y=78),
    PlainPoint(x=160, y=58),
    PlainPoint(x=175, y=63),
    PlainPoint(x=165, y=73),
]
"""The PCA page's measured four, whose centred Gram matrix is two blocks."""

EDGE_GAMMA = 0.002
"""The radial reach the edges use on the arc, a reach of about 22 units."""

STRANGERS = [PlainPoint(x=400, y=400), PlainPoint(x=1000, y=-1000)]
"""Two people far from everyone, who the radial kernel scores at zero throughout."""

SIGMOID_GAMMA = 0.0001
"""A sigmoid setting on the arc whose table is not positive semi-definite."""


def _outcome(edge: str, attempt) -> EdgeOutcome:
    """Run one case and record what the library did with it."""
    try:
        detail, figure = attempt()
    except MLLibError as error:
        return EdgeOutcome(edge=edge, accepted=False, detail=str(error), figure=None)
    except ValidationError as error:
        message = error.errors()[0]["msg"]
        return EdgeOutcome(edge=edge, accepted=False, detail=message, figure=None)
    return EdgeOutcome(edge=edge, accepted=True, detail=detail, figure=figure)


def _edge_cases() -> list[EdgeOutcome]:
    arc = _features(ARC)
    radial = RadialBasisKernel(gamma=EDGE_GAMMA)

    def one_row():
        KernelPrincipalComponentAnalysis().fit(_features(MEASURED_FOUR[:1]))
        return "fitted", None

    def identical_rows():
        KernelPrincipalComponentAnalysis(kernel=radial).fit(
            _features([PlainPoint(x=170, y=68)] * 3)
        )
        return "fitted", None

    def two_rows():
        fitted = KernelPrincipalComponentAnalysis(kernel=radial).fit(
            _features(MEASURED_FOUR[:2])
        )
        return "fitted, with one direction", float(fitted.components.n_components)

    def constant_column():
        fitted = KernelPrincipalComponentAnalysis(kernel=radial).fit(
            [
                Feature("height", [160.0, 170.0, 180.0, 175.0]),
                Feature("weight", [60.0, 60.0, 60.0, 60.0]),
            ]
        )
        return (
            f"fitted, with {fitted.components.n_components} directions from four rows",
            float(fitted.components.n_components),
        )

    def non_finite():
        KernelPrincipalComponentAnalysis().fit(
            [
                Feature("height", [160.0, float("nan"), 180.0]),
                Feature("weight", [60.0, 61.0, 62.0]),
            ]
        )
        return "fitted", None

    def past_the_rows():
        KernelPrincipalComponentAnalysis(n_components=5).fit(_features(MEASURED_FOUR))
        return "fitted", None

    def past_the_rank():
        fitted = KernelPrincipalComponentAnalysis(n_components=4).fit(
            _features(MEASURED_FOUR)
        )
        return (
            f"fitted, and answered {fitted.components.n_components} directions for the four asked",
            float(fitted.components.n_components),
        )

    def all_rows_radial():
        fitted = KernelPrincipalComponentAnalysis(kernel=radial, n_components=12).fit(
            arc
        )
        return (
            f"fitted, and answered {fitted.components.n_components} directions for the twelve asked",
            float(fitted.components.n_components),
        )

    def zero_components():
        KernelPrincipalComponentAnalysis(n_components=0)
        return "constructed", None

    def zero_gamma():
        RadialBasisKernel(gamma=0.0)
        return "constructed", None

    def negative_constant():
        PolynomialKernel(degree=2, constant=-1.0)
        return "constructed", None

    def unfitted():
        KernelPrincipalComponentAnalysis().transform(arc)
        return "transformed", None

    def wrong_names():
        fitted = KernelPrincipalComponentAnalysis().fit(_features(MEASURED_FOUR))
        fitted.transform([Feature("height", [170.0]), Feature("age", [30.0])])
        return "transformed", None

    def missing_feature():
        fitted = KernelPrincipalComponentAnalysis().fit(_features(MEASURED_FOUR))
        fitted.transform([Feature("height", [170.0])])
        return "transformed", None

    def reordered():
        fitted = KernelPrincipalComponentAnalysis().fit(_features(MEASURED_FOUR))
        forward = _transformed(fitted, _features(MEASURED_FOUR))
        backward = _transformed(fitted, list(reversed(_features(MEASURED_FOUR))))
        return "transformed, matched by name", float(np.max(np.abs(forward - backward)))

    def overflow():
        KernelPrincipalComponentAnalysis(
            kernel=PolynomialKernel(degree=200, gamma=1.0)
        ).fit(arc)
        return "fitted", None

    def indefinite():
        kernel = SigmoidKernel(gamma=SIGMOID_GAMMA, constant=0.0)
        block = _block(ARC)
        unclamped = np.linalg.eigvalsh(kernel.between(block, block).centred().values)
        fitted = KernelPrincipalComponentAnalysis(kernel=kernel).fit(arc)
        negatives = int(np.sum(unclamped < -EIGENVALUE_NOISE))
        first_share = fitted.components.variance_shares[0]
        detail = (
            f"fitted, with {negatives} negative eigenvalues clamped to zero; the most "
            f"negative was {unclamped.min():.4f} against a largest of "
            f"{unclamped.max():.4f}, and the first share reads {first_share:.4f}"
        )
        return detail, float(unclamped.min())

    def strangers():
        fitted = KernelPrincipalComponentAnalysis(kernel=radial, n_components=2).fit(
            arc
        )
        placed = _transformed(fitted, _features(STRANGERS))
        gap = float(np.max(np.abs(placed[0] - placed[1])))
        detail = (
            f"transformed; both strangers land at ({placed[0][0]:.4f}, "
            f"{placed[0][1]:.4f}), {gap:.1e} apart"
        )
        return detail, float(placed[0][0])

    def self_centred_query():
        query = radial.between(_block(STRANGERS[:1]), _block(ARC))
        query.centred()
        return "centred", None

    def metres():
        in_metres = [PlainPoint(x=point.x / 100, y=point.y) for point in ARC]
        fitted = KernelPrincipalComponentAnalysis(kernel=radial).fit(
            _features(in_metres)
        )
        in_centimetres = KernelPrincipalComponentAnalysis(kernel=radial).fit(arc)
        metre_share = fitted.components.variance_shares[0]
        centimetre_share = in_centimetres.components.variance_shares[0]
        detail = (
            f"fitted, with a first share of {metre_share:.4f} where centimetres "
            f"gave {centimetre_share:.4f}"
        )
        return detail, float(metre_share)

    return [
        _outcome("one row", one_row),
        _outcome("three identical rows", identical_rows),
        _outcome("two rows", two_rows),
        _outcome("a constant column", constant_column),
        _outcome("a non-finite value", non_finite),
        _outcome("five directions from four rows", past_the_rows),
        _outcome("four directions from four rows on a plane", past_the_rank),
        _outcome("twelve directions from the twelve on the arc", all_rows_radial),
        _outcome("zero directions", zero_components),
        _outcome("a gamma of zero", zero_gamma),
        _outcome("a negative polynomial constant", negative_constant),
        _outcome("transform before fit", unfitted),
        _outcome("transform with a feature under another name", wrong_names),
        _outcome("transform with a feature missing", missing_feature),
        _outcome("transform with the features reordered", reordered),
        _outcome("a polynomial kernel of degree 200 on raw units", overflow),
        _outcome("a sigmoid kernel that is not a kernel", indefinite),
        _outcome("two strangers far from everyone", strangers),
        _outcome("centring a query row against itself", self_centred_query),
        _outcome("height in metres at the same gamma", metres),
    ]


@router.get("/concepts/kernel-pca/edges", response_model=EdgesResponse)
def edges() -> EdgesResponse:
    """Run every edge the contracts table quotes and report what the library did."""
    return EdgesResponse(outcomes=_edge_cases())
