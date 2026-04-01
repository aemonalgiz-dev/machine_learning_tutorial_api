"""k-means and principal component analysis, for the unsupervised pages.

Both endpoints take unlabelled people, which is the bucket's defining fact.
The k-means endpoint fits the library's ``KMeans`` under a fixed seed and
returns each person's cluster, the resting centres, the inertia and the
nearest-centre regions, so the page can check by hand that a resting centre is
nothing more than the mean of its group. The PCA endpoint fits the library's
``PrincipalComponentAnalysis`` twice, once whole for the directions and their
variance shares, and once truncated to a single component whose transform and
inverse transform produce the reconstructions the projection toggle draws.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import Feature, KMeans, PrincipalComponentAnalysis
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    GRID_CELLS,
    RegionGrid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS

router = APIRouter(tags=["unsupervised"])

MAX_CLUSTERS = 6
"""The most groups the widget may ask for."""

FIXED_SEED = 3
"""The seeding is fixed so the pages can quote what the widget shows."""


class PlainPoint(BaseModel):
    """One unlabelled person."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class ClusterRequest(BaseModel):
    """The people, and how many groups to sort them into."""

    points: list[PlainPoint] = Field(min_length=1, max_length=MAX_POINTS)
    k: int = Field(ge=1, le=MAX_CLUSTERS)

    model_config = {"extra": "forbid"}


class ClusterResponse(BaseModel):
    """The grouping, the resting centres, and the total it minimised."""

    labels: list[int]
    centres: list[PlainPoint]
    inertia: float
    iterations_run: int
    regions: RegionGrid


@router.post("/concepts/k-means/cluster", response_model=ClusterResponse)
def cluster(request: ClusterRequest) -> ClusterResponse:
    """Sort the people into k groups and report where the centres settled."""
    if request.k > len(request.points):
        raise HTTPException(
            status_code=422,
            detail=(
                f"k is {request.k} but only {len(request.points)} people are "
                "here, and a group cannot be conjured for no one"
            ),
        )

    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    model = KMeans(n_clusters=request.k, random_seed=FIXED_SEED).fit([heights, weights])

    labels = np.asarray(model.predict([heights, weights]), dtype=np.float64)
    centres = [
        PlainPoint(
            x=float(centroid.coordinate_for("height")),
            y=float(centroid.coordinate_for("weight")),
        )
        for centroid in model.centroids
    ]

    return ClusterResponse(
        labels=[int(value) for value in labels],
        centres=centres,
        inertia=float(model.inertia),
        iterations_run=int(model.iterations_run),
        regions=_cluster_regions(model, request.points),
    )


def _cluster_regions(model: KMeans, points: list[PlainPoint]) -> RegionGrid:
    """The fitted model's group at every cell of a lattice over the plane."""
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
    y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad

    x_centres = np.linspace(x_min, x_max, GRID_CELLS)
    y_centres = np.linspace(y_min, y_max, GRID_CELLS)
    lattice_x, lattice_y = np.meshgrid(x_centres, y_centres)

    predictions = np.asarray(
        model.predict(
            [
                Feature("height", lattice_x.ravel()),
                Feature("weight", lattice_y.ravel()),
            ]
        ),
        dtype=np.float64,
    ).reshape(GRID_CELLS, GRID_CELLS)

    return RegionGrid(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        cells=GRID_CELLS,
        labels=[[int(value) for value in row] for row in predictions],
    )


class AnalyzeRequest(BaseModel):
    """The people whose cloud is to be described by its directions."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class ComponentDocument(BaseModel):
    """One direction of the cloud, its variance, and its share of the total."""

    dx: float
    dy: float
    variance: float
    share: float


class AnalyzeResponse(BaseModel):
    """The cloud's mean, its two directions, and the one-component shadows.

    ``reconstructions`` holds each person rebuilt from their first-component
    value alone, which is where they land when the cloud is flattened onto its
    longest direction.
    """

    mean: PlainPoint
    components: list[ComponentDocument]
    reconstructions: list[PlainPoint]


@router.post("/concepts/pca/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Find the cloud's directions and its one-component reconstructions."""
    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    whole = PrincipalComponentAnalysis().fit([heights, weights])
    shares = np.asarray(whole.components.variance_shares, dtype=np.float64)

    components = [
        ComponentDocument(
            dx=float(component.direction[0]),
            dy=float(component.direction[1]),
            variance=float(component.variance),
            share=float(share),
        )
        for component, share in zip(whole.components, shares, strict=True)
    ]

    flattened = PrincipalComponentAnalysis(n_components=1).fit([heights, weights])
    rebuilt = flattened.inverse_transform(flattened.transform([heights, weights]))
    rebuilt_x = np.asarray(rebuilt[0].values, dtype=np.float64)
    rebuilt_y = np.asarray(rebuilt[1].values, dtype=np.float64)

    return AnalyzeResponse(
        mean=PlainPoint(
            x=float(np.mean([point.x for point in request.points])),
            y=float(np.mean([point.y for point in request.points])),
        ),
        components=components,
        reconstructions=[
            PlainPoint(x=float(x), y=float(y))
            for x, y in zip(rebuilt_x, rebuilt_y, strict=True)
        ],
    )


MAX_ITERATION_SWEEP = 12
"""How many loop passes the inertia staircase traces."""


class InertiaCurveRequest(BaseModel):
    """The people, and how many groups the walk is looking for."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    n_clusters: int = Field(ge=1, le=MAX_CLUSTERS)

    model_config = {"extra": "forbid"}


class InertiaCurveResponse(BaseModel):
    """The total the loop tracks, after each pass it was allowed.

    Each entry refits the same seeded walk with the pass budget raised by
    one, so the sequence is the single walk watched at every stage, and it
    can only fall or hold still.
    """

    iterations: list[int]
    inertias: list[float]


@router.post("/concepts/k-means/inertia-curve", response_model=InertiaCurveResponse)
def inertia_curve(request: InertiaCurveRequest) -> InertiaCurveResponse:
    """Trace the inertia of the seeded walk, one pass budget at a time."""
    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    iterations = list(range(1, MAX_ITERATION_SWEEP + 1))
    inertias: list[float] = []
    for budget in iterations:
        model = KMeans(
            n_clusters=request.n_clusters,
            n_initialisations=1,
            random_seed=FIXED_SEED,
            max_iterations=budget,
        ).fit([heights, weights])
        inertias.append(float(model.inertia))

    return InertiaCurveResponse(iterations=iterations, inertias=inertias)


SWEEP_DEGREES = 91
"""The variance sweep samples the half-turn every two degrees."""


class VarianceSweepRequest(BaseModel):
    """The people whose cloud is to be measured in every direction."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class SweepComponent(BaseModel):
    """Where one of the cloud's own directions lands on the sweep."""

    angle_degrees: float
    variance: float


class VarianceSweepResponse(BaseModel):
    """The variance of the cloud along each direction of a half-turn.

    The curve is built from the fitted components themselves: the variance
    along a direction u is the sum over components of the component's
    variance times the squared alignment of u with it, which is the
    derivation's u'Cu with C written in its own eigenpairs. The peaks
    therefore land exactly on the components the analyze endpoint reports.
    """

    angles_degrees: list[float]
    variances: list[float]
    components: list[SweepComponent]


@router.post("/concepts/pca/variance-sweep", response_model=VarianceSweepResponse)
def variance_sweep(request: VarianceSweepRequest) -> VarianceSweepResponse:
    """Sweep a direction through half a turn and measure the variance along it."""
    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    fitted = PrincipalComponentAnalysis().fit([heights, weights])

    directions = np.asarray(
        [list(component.direction) for component in fitted.components],
        dtype=np.float64,
    )
    variances = np.asarray(
        [float(component.variance) for component in fitted.components],
        dtype=np.float64,
    )

    angles = np.linspace(0.0, 180.0, SWEEP_DEGREES)
    radians = np.deg2rad(angles)
    sweep_directions = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    alignments = sweep_directions @ directions.T
    swept = (alignments**2) @ variances

    components = []
    for direction, variance in zip(directions, variances, strict=True):
        angle = float(np.degrees(np.arctan2(direction[1], direction[0]))) % 180.0
        components.append(SweepComponent(angle_degrees=angle, variance=float(variance)))

    return VarianceSweepResponse(
        angles_degrees=[float(value) for value in angles],
        variances=[float(value) for value in swept],
        components=components,
    )
