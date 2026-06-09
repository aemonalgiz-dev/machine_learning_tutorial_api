"""k-means and principal component analysis, for the unsupervised pages.

Both endpoints take unlabelled people, which is the bucket's defining fact.
The k-means endpoint fits the library's ``KMeans`` under a fixed seed and
returns each person's cluster, the resting centres, the inertia and the
nearest-centre regions, so the page can check by hand that a resting centre is
nothing more than the mean of its group. The PCA endpoint fits the library's
``PrincipalComponentAnalysis`` twice, once whole for the directions and their
variance shares, and once truncated to a single component whose transform and
inverse transform produce the reconstructions the projection toggle draws.

The further k-means endpoints at the foot of the module report the road to
that fit rather than the fit alone: one seeded walk stepped pass by pass with
the inertia read after each half of the loop, the same data fitted from
several seeds so the valleys can be compared, the inertia against every
``k``, the same people under three other definitions of a group, the same
people with height in three units, and the arithmetic behind the update step,
which is why the centre is a mean and not a median.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DBSCAN,
    AgglomerativeClustering,
    Centroid,
    Centroids,
    Feature,
    GaussianMixture,
    KMeans,
    Linkage,
    PrincipalComponentAnalysis,
    Standardizer,
)
from oop_ml.core.data.row_block import RowBlock, rows_of
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
    """The grouping, the resting centres, and the total it minimised.

    ``sizes`` counts the people in each group, in group order, so a group
    nobody chose shows up as a zero rather than vanishing.
    """

    labels: list[int]
    centres: list[PlainPoint]
    inertia: float
    iterations_run: int
    sizes: list[int]
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
        sizes=[int(size) for size in model.clustering.sizes],
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


class Scores(BaseModel):
    """One person's coordinates along the two components."""

    first: float
    second: float


class StandardizedReport(BaseModel):
    """The same cloud fitted after standardising both features.

    ``correlation`` is the population correlation of the two features, which
    is the off-diagonal entry of the covariance matrix the standardised fit
    actually decomposes; ``scale_x`` and ``scale_y`` are the population
    standard deviations each feature was divided by.
    """

    components: list[ComponentDocument]
    correlation: float
    scale_x: float
    scale_y: float


class AnalyzeResponse(BaseModel):
    """The cloud's mean, its two directions, and the one-component shadows.

    ``reconstructions`` holds each person rebuilt from their first-component
    value alone, which is where they land when the cloud is flattened onto its
    longest direction. ``scores`` are the library's transform, one pair per
    person. ``scatter`` is the matrix of deviation sums before any divisor,
    ``covariance`` divides it by ``n`` and ``sample_covariance`` by
    ``n - 1``; ``scatter_eigenvalues`` are the scatter matrix's, so they are
    the component variances times whichever divisor the library used.
    ``reconstruction_error_one`` is the total squared distance between each
    person and their one-component shadow.
    """

    mean: PlainPoint
    components: list[ComponentDocument]
    reconstructions: list[PlainPoint]
    deviations: list[PlainPoint]
    scores: list[Scores]
    scatter: list[list[float]]
    covariance: list[list[float]]
    sample_covariance: list[list[float]]
    scatter_eigenvalues: list[float]
    cumulative_shares: list[float]
    reconstruction_error_one: float
    standardized: StandardizedReport


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

    matrix = np.asarray(
        [[point.x, point.y] for point in request.points], dtype=np.float64
    )
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    scatter = centred.T @ centred
    count = len(request.points)
    directions = np.asarray(
        [list(component.direction) for component in whole.components],
        dtype=np.float64,
    )
    scatter_eigenvalues = [
        float(direction @ scatter @ direction) for direction in directions
    ]

    transformed = whole.transform([heights, weights])
    first_scores = np.asarray(transformed[0].values, dtype=np.float64)
    second_scores = np.asarray(transformed[1].values, dtype=np.float64)

    reconstruction_error = float(
        np.sum((matrix[:, 0] - rebuilt_x) ** 2 + (matrix[:, 1] - rebuilt_y) ** 2)
    )

    standardised = PrincipalComponentAnalysis(standardize=True).fit([heights, weights])
    standardised_shares = np.asarray(
        standardised.components.variance_shares, dtype=np.float64
    )
    scale_x, scale_y = float(centred[:, 0].std()), float(centred[:, 1].std())
    correlation = (
        float(scatter[0, 1] / (count * scale_x * scale_y))
        if scale_x > 0 and scale_y > 0
        else 0.0
    )

    return AnalyzeResponse(
        mean=PlainPoint(x=float(mean[0]), y=float(mean[1])),
        components=components,
        reconstructions=[
            PlainPoint(x=float(x), y=float(y))
            for x, y in zip(rebuilt_x, rebuilt_y, strict=True)
        ],
        deviations=[PlainPoint(x=float(row[0]), y=float(row[1])) for row in centred],
        scores=[
            Scores(first=float(first), second=float(second))
            for first, second in zip(first_scores, second_scores, strict=True)
        ],
        scatter=[[float(value) for value in row] for row in scatter],
        covariance=[[float(value) for value in row] for row in scatter / count],
        sample_covariance=[
            [float(value) for value in row] for row in scatter / max(count - 1, 1)
        ],
        scatter_eigenvalues=scatter_eigenvalues,
        cumulative_shares=[
            float(value) for value in whole.components.cumulative_shares
        ],
        reconstruction_error_one=reconstruction_error,
        standardized=StandardizedReport(
            components=[
                ComponentDocument(
                    dx=float(component.direction[0]),
                    dy=float(component.direction[1]),
                    variance=float(component.variance),
                    share=float(share),
                )
                for component, share in zip(
                    standardised.components, standardised_shares, strict=True
                )
            ],
            correlation=correlation,
            scale_x=scale_x,
            scale_y=scale_y,
        ),
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


# --- The k-means page's further endpoints ----------------------------------

FEATURE_NAMES = ("height", "weight")
"""The two features every k-means endpoint here clusters on, in order."""

MAX_SEEDS = 20
"""The most single-start fits the restart gallery may ask for."""

MAX_ELBOW_K = 8
"""The largest group count the elbow sweep reaches."""


def _features(points: list[PlainPoint]) -> list[Feature]:
    return [
        Feature("height", [point.x for point in points]),
        Feature("weight", [point.y for point in points]),
    ]


def _rows(points: list[PlainPoint]) -> RowBlock:
    return rows_of(
        np.asarray([[point.x, point.y] for point in points], dtype=np.float64),
        FEATURE_NAMES,
    )


def _centroids_at(positions: np.ndarray) -> Centroids:
    """The library's centres, built at the given positions."""
    return Centroids(
        [
            Centroid(KMeans.name_for(position), row, FEATURE_NAMES)
            for position, row in enumerate(positions)
        ]
    )


def _assignment(centroids: Centroids, rows: RowBlock) -> np.ndarray:
    """Each row's nearest centre, by the library's own distance table."""
    return np.argmin(centroids.squared_distances_to(rows), axis=1)


def _inertia_under(centroids: Centroids, rows: RowBlock, labels: np.ndarray) -> float:
    """The objective's definition: each row's squared distance to its own centre."""
    distances = centroids.squared_distances_to(rows)
    return float(np.sum(distances[np.arange(labels.size), labels]))


def _points_of(centroids: Centroids) -> list[PlainPoint]:
    return [
        PlainPoint(
            x=float(centroid.coordinate_for("height")),
            y=float(centroid.coordinate_for("weight")),
        )
        for centroid in centroids
    ]


def _same_partition(first: list[int], second: list[int]) -> bool:
    """Whether two labellings put the same people together, whatever the numbers."""
    return all(
        (first[i] == first[j]) == (second[i] == second[j])
        for i in range(len(first))
        for j in range(i + 1, len(first))
    )


def _too_many_groups(k: int, points: list[PlainPoint]) -> None:
    if k > len(points):
        raise HTTPException(
            status_code=422,
            detail=(
                f"k is {k} but only {len(points)} people are here, and a group "
                "cannot be conjured for no one"
            ),
        )


def _lattice(points: list[PlainPoint]) -> tuple[RowBlock, dict[str, float]]:
    """The same lattice ``_cluster_regions`` shades, as rows the centres can read."""
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    bounds = {
        "x_min": min(x_values) - x_pad,
        "x_max": max(x_values) + x_pad,
        "y_min": min(y_values) - y_pad,
        "y_max": max(y_values) + y_pad,
    }
    lattice_x, lattice_y = np.meshgrid(
        np.linspace(bounds["x_min"], bounds["x_max"], GRID_CELLS),
        np.linspace(bounds["y_min"], bounds["y_max"], GRID_CELLS),
    )
    return (
        rows_of(np.column_stack([lattice_x.ravel(), lattice_y.ravel()]), FEATURE_NAMES),
        bounds,
    )


def _regions_of(centroids: Centroids, points: list[PlainPoint]) -> RegionGrid:
    """Which centre is nearest at every cell of the lattice over the plane."""
    lattice, bounds = _lattice(points)
    labels = _assignment(centroids, lattice).reshape(GRID_CELLS, GRID_CELLS)
    return RegionGrid(
        x_min=float(bounds["x_min"]),
        x_max=float(bounds["x_max"]),
        y_min=float(bounds["y_min"]),
        y_max=float(bounds["y_max"]),
        cells=GRID_CELLS,
        labels=[[int(value) for value in row] for row in labels],
    )


class WalkRequest(BaseModel):
    """The people, how many groups, and which seeding to walk from."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    k: int = Field(ge=1, le=MAX_CLUSTERS)
    seed: int = Field(default=FIXED_SEED, ge=0, le=1000)

    model_config = {"extra": "forbid"}


class WalkStage(BaseModel):
    """The loop's state at one moment: where the centres are and who chose them."""

    centres: list[PlainPoint]
    labels: list[int]
    inertia: float
    regions: RegionGrid


class WalkPass(BaseModel):
    """One pass of the loop, with the inertia read after each of its two halves.

    ``labels_used`` is the assignment the update step averaged, which is the
    previous stage's labels. ``inertia_after_update`` keeps those labels and
    moves the centres; ``inertia_after_assign`` keeps the moved centres and
    reassigns. ``largest_shift`` is the furthest any centre moved, squared,
    which is what the library compares against its tolerance.
    """

    pass_number: int
    labels_used: list[int]
    centres: list[PlainPoint]
    inertia_after_update: float
    labels_after_assign: list[int]
    inertia_after_assign: float
    largest_shift: float
    regions: RegionGrid


class WalkResponse(BaseModel):
    """One seeded start followed until it settles.

    The library keeps no record of its walk, so each pass is the same seeded
    fit rerun with its pass budget raised by one, which is the one walk
    watched at every stage; the seeded start is read from the generator that
    fit's first restart draws from. ``settled_pass`` is the pass on which no
    centre moved further than the tolerance, and ``best_of_ten_inertia`` is
    what the same seed reaches when the library is allowed its usual ten
    starts.
    """

    start: WalkStage
    passes: list[WalkPass]
    settled_pass: int
    resting_inertia: float
    best_of_ten_inertia: float
    rests_in_the_best_valley: bool


@router.post("/concepts/k-means/walk", response_model=WalkResponse)
def walk(request: WalkRequest) -> WalkResponse:
    """Follow one seeded start through the loop, pass by pass."""
    _too_many_groups(request.k, request.points)

    features = _features(request.points)
    rows = _rows(request.points)

    unfitted = KMeans(
        n_clusters=request.k, n_initialisations=1, random_seed=request.seed
    )
    seeded = _centroids_at(
        unfitted._seeded_positions(rows, np.random.default_rng(request.seed))
    )
    seeded_labels = _assignment(seeded, rows)
    start = WalkStage(
        centres=_points_of(seeded),
        labels=[int(value) for value in seeded_labels],
        inertia=_inertia_under(seeded, rows, seeded_labels),
        regions=_regions_of(seeded, request.points),
    )

    passes: list[WalkPass] = []
    previous_centroids = seeded
    previous_labels = seeded_labels
    settled_pass = MAX_ITERATION_SWEEP
    for budget in range(1, MAX_ITERATION_SWEEP + 1):
        model = KMeans(
            n_clusters=request.k,
            n_initialisations=1,
            random_seed=request.seed,
            max_iterations=budget,
        ).fit(features)
        moved = model.centroids
        reassigned = np.asarray(model.predict(features)).astype(np.intp)
        shift = float(
            np.max(
                np.sum((moved.positions - previous_centroids.positions) ** 2, axis=1)
            )
        )
        passes.append(
            WalkPass(
                pass_number=budget,
                labels_used=[int(value) for value in previous_labels],
                centres=_points_of(moved),
                inertia_after_update=_inertia_under(moved, rows, previous_labels),
                labels_after_assign=[int(value) for value in reassigned],
                inertia_after_assign=float(model.inertia),
                largest_shift=shift,
                regions=_cluster_regions(model, request.points),
            )
        )
        previous_centroids = moved
        previous_labels = reassigned
        if shift <= unfitted.tolerance:
            settled_pass = budget
            break

    best = KMeans(n_clusters=request.k, random_seed=request.seed).fit(features)
    resting = passes[-1].inertia_after_assign
    return WalkResponse(
        start=start,
        passes=passes,
        settled_pass=settled_pass,
        resting_inertia=resting,
        best_of_ten_inertia=float(best.inertia),
        rests_in_the_best_valley=bool(abs(resting - best.inertia) <= 1e-9),
    )


class RestartsRequest(BaseModel):
    """The people, how many groups, and how many single-start fits to compare."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    k: int = Field(ge=1, le=MAX_CLUSTERS)
    n_seeds: int = Field(default=10, ge=1, le=MAX_SEEDS)

    model_config = {"extra": "forbid"}


class SeededFit(BaseModel):
    """What one start settled on."""

    seed: int
    labels: list[int]
    centres: list[PlainPoint]
    inertia: float
    iterations_run: int
    sizes: list[int]
    regions: RegionGrid
    same_grouping_as_best: bool


class RestartsResponse(BaseModel):
    """Every single-start fit, and the one the library keeps when given them all.

    ``best_seed`` is the lowest inertia among the starts, ties to the earlier
    seed, which is the library's own rule for choosing among restarts.
    ``with_restarts_inertia`` is one fit from seed zero allowed ``n_seeds``
    starts, which the library seeds as exactly the starts listed, so the two
    agree. ``distinct_inertias`` counts how many different resting values the
    starts reached, so one means every start found the same valley.
    """

    fits: list[SeededFit]
    best_seed: int
    best_inertia: float
    with_restarts_inertia: float
    distinct_inertias: int


@router.post("/concepts/k-means/restarts", response_model=RestartsResponse)
def restarts(request: RestartsRequest) -> RestartsResponse:
    """Fit the same people once from each seed and line the valleys up."""
    _too_many_groups(request.k, request.points)

    features = _features(request.points)
    models = [
        KMeans(n_clusters=request.k, n_initialisations=1, random_seed=seed).fit(
            features
        )
        for seed in range(request.n_seeds)
    ]
    best_seed = 0
    for seed, model in enumerate(models):
        if model.inertia < models[best_seed].inertia:
            best_seed = seed
    best_labels = [
        int(value) for value in np.asarray(models[best_seed].predict(features))
    ]

    fits = []
    for seed, model in enumerate(models):
        labels = [int(value) for value in np.asarray(model.predict(features))]
        fits.append(
            SeededFit(
                seed=seed,
                labels=labels,
                centres=_points_of(model.centroids),
                inertia=float(model.inertia),
                iterations_run=int(model.iterations_run),
                sizes=[int(size) for size in model.clustering.sizes],
                regions=_cluster_regions(model, request.points),
                same_grouping_as_best=_same_partition(labels, best_labels),
            )
        )

    with_restarts = KMeans(
        n_clusters=request.k, n_initialisations=request.n_seeds, random_seed=0
    ).fit(features)
    return RestartsResponse(
        fits=fits,
        best_seed=best_seed,
        best_inertia=float(models[best_seed].inertia),
        with_restarts_inertia=float(with_restarts.inertia),
        distinct_inertias=len({round(model.inertia, 6) for model in models}),
    )


class ElbowRequest(BaseModel):
    """The people, and the largest group count to sweep up to."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    max_k: int = Field(default=6, ge=1, le=MAX_ELBOW_K)

    model_config = {"extra": "forbid"}


class ElbowPoint(BaseModel):
    """The best-of-ten fit at one group count."""

    k: int
    inertia: float
    sizes: list[int]
    drop_from_previous: float | None
    share_of_total: float


class ElbowResponse(BaseModel):
    """Inertia against ``k``, each point the library's usual ten-start fit.

    ``total_scatter`` is the sum of every person's squared distance from the
    grand mean, which is what one centre serving everyone must pay, so the
    ``k = 1`` inertia equals it and every share is measured against it. The
    sweep stops at the number of people if that is smaller than ``max_k``.
    """

    points: list[ElbowPoint]
    total_scatter: float


@router.post("/concepts/k-means/elbow", response_model=ElbowResponse)
def elbow(request: ElbowRequest) -> ElbowResponse:
    """Sweep the group count and report what inertia does."""
    reach = min(request.max_k, len(request.points))
    features = _features(request.points)
    matrix = np.asarray([[point.x, point.y] for point in request.points])
    total = float(np.sum((matrix - matrix.mean(axis=0)) ** 2))

    sweep: list[ElbowPoint] = []
    previous: float | None = None
    for k in range(1, reach + 1):
        model = KMeans(n_clusters=k, random_seed=FIXED_SEED).fit(features)
        inertia = float(model.inertia)
        sweep.append(
            ElbowPoint(
                k=k,
                inertia=inertia,
                sizes=[int(size) for size in model.clustering.sizes],
                drop_from_previous=None if previous is None else previous - inertia,
                share_of_total=inertia / total if total > 0 else 0.0,
            )
        )
        previous = inertia
    return ElbowResponse(points=sweep, total_scatter=total)


class OtherDefinitionsRequest(BaseModel):
    """The people, the group count, and the density clusterer's two numbers."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    k: int = Field(ge=1, le=MAX_CLUSTERS)
    radius: float = Field(default=10.0, gt=0.0, le=MAX_MAGNITUDE)
    min_neighbourhood_size: int = Field(default=2, ge=1, le=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DensityGrouping(BaseModel):
    """The density clusterer's answer; a label of ``-1`` is a person in no group."""

    labels: list[int]
    n_clusters: int
    n_noise: int


class LabelsOnly(BaseModel):
    """A grouping with no centres to report."""

    labels: list[int]


class OtherDefinitionsResponse(BaseModel):
    """The same people grouped four ways.

    ``k_means`` is the nearest-centre grouping; ``density`` asks whether a
    person sits in a crowd joined to another crowd, and needs no ``k``;
    ``single_linkage`` merges whichever two groups have the closest pair of
    members until ``k`` remain; ``mixture`` fits ``k`` Gaussians by
    expectation-maximisation, starting from a k-means grouping.
    """

    k_means: ClusterResponse
    density: DensityGrouping
    single_linkage: LabelsOnly
    mixture: LabelsOnly


@router.post(
    "/concepts/k-means/other-definitions", response_model=OtherDefinitionsResponse
)
def other_definitions(request: OtherDefinitionsRequest) -> OtherDefinitionsResponse:
    """Group the same people under four definitions of what a group is."""
    k_means = cluster(ClusterRequest(points=request.points, k=request.k))
    features = _features(request.points)

    density = DBSCAN(
        radius=request.radius, min_neighbourhood_size=request.min_neighbourhood_size
    ).fit(features)
    single = AgglomerativeClustering(n_clusters=request.k, linkage=Linkage.SINGLE).fit(
        features
    )
    mixture = GaussianMixture(n_components=request.k, random_seed=FIXED_SEED).fit(
        features
    )

    return OtherDefinitionsResponse(
        k_means=k_means,
        density=DensityGrouping(
            labels=[int(value) for value in np.asarray(density.labels)],
            n_clusters=int(density.n_clusters),
            n_noise=int(density.n_noise),
        ),
        single_linkage=LabelsOnly(
            labels=[int(value) for value in np.asarray(single.labels)]
        ),
        mixture=LabelsOnly(
            labels=[int(value) for value in np.asarray(mixture.predict(features))]
        ),
    )


HEIGHT_UNITS = (("metres", 0.01), ("centimetres", 1.0), ("millimetres", 10.0))
"""Height rewritten in three units, each a factor on the same numbers."""


class UnitsRequest(BaseModel):
    """The people in centimetres and kilograms, and how many groups."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    k: int = Field(ge=1, le=MAX_CLUSTERS)

    model_config = {"extra": "forbid"}


class UnitGrouping(BaseModel):
    """The fit with height in one unit; the centres are in that unit."""

    unit: str
    factor: float
    labels: list[int]
    centres: list[PlainPoint]
    inertia: float
    sizes: list[int]
    same_grouping_as_centimetres: bool


class UnitsResponse(BaseModel):
    """The same people grouped with height in three units, and standardised.

    ``standardised`` divides each feature's deviations by its own standard
    deviation before the fit, so neither feature outweighs the other by its
    unit; its centres are reported back in centimetres and kilograms and its
    inertia is in standardised units.
    """

    variants: list[UnitGrouping]
    standardised: UnitGrouping


@router.post("/concepts/k-means/units", response_model=UnitsResponse)
def units(request: UnitsRequest) -> UnitsResponse:
    """Refit the people with height in each unit and report what moved."""
    _too_many_groups(request.k, request.points)

    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    weights = np.asarray([point.y for point in request.points], dtype=np.float64)

    def grouping(unit: str, factor: float, features: list[Feature]) -> UnitGrouping:
        model = KMeans(n_clusters=request.k, random_seed=FIXED_SEED).fit(features)
        labels = [int(value) for value in np.asarray(model.predict(features))]
        return UnitGrouping(
            unit=unit,
            factor=factor,
            labels=labels,
            centres=_points_of(model.centroids),
            inertia=float(model.inertia),
            sizes=[int(size) for size in model.clustering.sizes],
            same_grouping_as_centimetres=False,
        )

    variants = [
        grouping(
            unit,
            factor,
            [Feature("height", heights * factor), Feature("weight", weights)],
        )
        for unit, factor in HEIGHT_UNITS
    ]
    reference = next(variant for variant in variants if variant.factor == 1.0)
    for variant in variants:
        variant.same_grouping_as_centimetres = _same_partition(
            variant.labels, reference.labels
        )

    raw = [Feature("height", heights), Feature("weight", weights)]
    scaler = Standardizer().fit(raw)
    standardised = grouping("standardised", 1.0, scaler.transform(raw))
    standardised.same_grouping_as_centimetres = _same_partition(
        standardised.labels, reference.labels
    )
    scaled = np.asarray(
        [[centre.x, centre.y] for centre in standardised.centres], dtype=np.float64
    )
    restored_heights = scaler.scalings["height"].restore(scaled[:, 0])
    restored_weights = scaler.scalings["weight"].restore(scaled[:, 1])
    standardised.centres = [
        PlainPoint(x=float(x), y=float(y))
        for x, y in zip(restored_heights, restored_weights, strict=True)
    ]
    return UnitsResponse(variants=variants, standardised=standardised)


class CentreChoiceRequest(BaseModel):
    """One group of people, whose best centre is to be found under two distances."""

    points: list[PlainPoint] = Field(min_length=1, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class CentreChoiceResponse(BaseModel):
    """The mean and the coordinate-wise median, each scored under both distances.

    ``squared_at_mean`` and ``squared_at_median`` total the squared Euclidean
    distance from every person to that centre; the ``absolute`` pair total the
    Manhattan distance instead. The library has no median-centred clusterer,
    so this is the objective's own arithmetic rather than a fit.
    """

    mean: PlainPoint
    median: PlainPoint
    squared_at_mean: float
    squared_at_median: float
    absolute_at_mean: float
    absolute_at_median: float


@router.post("/concepts/k-means/centre-choice", response_model=CentreChoiceResponse)
def centre_choice(request: CentreChoiceRequest) -> CentreChoiceResponse:
    """Score the mean and the median as a group's centre under both distances."""
    matrix = np.asarray([[point.x, point.y] for point in request.points])
    mean = matrix.mean(axis=0)
    median = np.median(matrix, axis=0)

    def squared(centre: np.ndarray) -> float:
        return float(np.sum((matrix - centre) ** 2))

    def absolute(centre: np.ndarray) -> float:
        return float(np.sum(np.abs(matrix - centre)))

    return CentreChoiceResponse(
        mean=PlainPoint(x=float(mean[0]), y=float(mean[1])),
        median=PlainPoint(x=float(median[0]), y=float(median[1])),
        squared_at_mean=squared(mean),
        squared_at_median=squared(median),
        absolute_at_mean=absolute(mean),
        absolute_at_median=absolute(median),
    )
