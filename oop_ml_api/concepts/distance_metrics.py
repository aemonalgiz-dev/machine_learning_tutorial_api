"""What near means, for the distance-metrics page.

Seven endpoints, each thin over ``DistanceMetric``, the library's closed enum
pairing six names with the objects that compute them.

The unit-circle endpoint hands a metric a lattice of cells and one centre, asks
how far every cell sits from that centre, and thresholds the library's answer
at one. The shape of "one unit away" is therefore read off a library answer
rather than drawn from a formula this module knows. The centre defaults to the
origin and may be moved, because three of the six do not shift with their
centre the way the p-norms do, and the origin is the one place where cosine,
Hamming and Canberra say almost nothing.

The neighbours endpoint fits the k-nearest page's classifier under a chosen
metric and returns the library's observed ``neighbour_search`` route, so the
same crowd and the same query can be watched consulting different people, and
under cosine reaching a different answer. The between endpoint measures one
pair under all six at once and shows the working a hand calculation passes
through, together with the p-norm of the same gap at a sweep of orders, which
is the derivation's picture of Manhattan sliding into Chebyshev, and the three
routes to one Euclidean answer, of which only the library's survives a pair
far from the origin.

The axioms endpoint takes three people and reports, under every metric, the
four checks a distance has to pass, so the page can show cosine failing the
triangle inequality on three people of one height rather than asserting it.
The units endpoint refits the crowd with weight in grams and again
standardised, which is the measurement behind the advice to scale first. The
models endpoint fits the density clusterer under every metric and works the
mean-against-median arithmetic that explains why k-means takes none. The
dimensions endpoint draws uniform rows in a rising number of features and
measures how much further the farthest row is than the nearest, which is the
curse of dimensionality as a number rather than a warning.

The dimensions endpoint is the only seeded thing here, and its seed is a
request field with a fixed default, so the page can quote what it shows.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    DBSCAN,
    AgglomerativeClustering,
    DistanceMetric,
    Feature,
    KNearestNeighboursClassifier,
    Linkage,
    MinkowskiDistance,
    RowBlock,
    Standardizer,
)
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.k_nearest import MAX_K, ChosenNeighbour
from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS

router = APIRouter(prefix="/concepts/distance-metrics", tags=["distance-metrics"])

LATTICE_EXTENT = 2
"""The lattice runs this far from the origin along each axis, both ways."""

LATTICE_STEPS_PER_UNIT = 10
"""Cells per unit of distance, so every cell centre lands on an exact tenth.

The coordinates are integers divided by ten rather than a ``linspace``, so
that zero and one are exact lattice values. Hamming reads exact equality and
Canberra divides by the size of the values, so a centre at (1, 1) has to
coincide with a cell for their pictures to be right.
"""

LATTICE_HALF_CELLS = LATTICE_EXTENT * LATTICE_STEPS_PER_UNIT
"""Cells from the origin to one edge, twenty at the settings above."""

LATTICE_CELLS = 2 * LATTICE_HALF_CELLS + 1
"""Cells along each side of the lattice, forty-one at the settings above."""

UNIT_TOLERANCE = 1e-9
"""How far from one a distance may fall and still count as one unit away.

The p-norm boundaries pass through lattice cells exactly, 0.3 + 0.7 under
Manhattan for one, and a sum of tenths in floating point can land an ulp
either side of 1.0. Without the allowance a boundary cell is sometimes in and
sometimes out, and the diamond grows notches that are facts about rounding
rather than about the metric.
"""

TRIANGLE_TOLERANCE = 1e-12
"""How far below the direct distance a two-leg route may fall and still pass.

Three people in one line make the two legs sum to the direct distance exactly
under every p-norm, and a sum of two floats can land an ulp under the third.
Without the allowance a genuine equality reads as a violation.
"""

LATTICE_FEATURE_NAMES = ("horizontal", "vertical")
"""What the lattice's two columns are called. The shapes are abstract."""

PAIR_FEATURE_NAMES = ("height", "weight")
"""What a measured pair's two columns are called, the k-nearest page's names."""

MINKOWSKI_ORDERS: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0, 10.0, np.inf)
"""The orders the between endpoint sweeps, Manhattan through to Chebyshev."""

MAX_ORDER = 100.0
"""The largest p-norm order a request may ask for by itself."""

MAX_WEIGHT_FACTOR = 10_000.0
"""How far a request may rescale the weight column. A thousand is grams."""

MAX_DIMENSION_POINTS = 300
"""The most uniform rows the dimensions endpoint will draw per dimension."""

DIMENSIONS: tuple[int, ...] = (1, 2, 5, 10, 50, 200)
"""The feature counts the dimensions endpoint measures, the library's own."""

MAX_RADIUS = 1_000_000.0
"""The largest density radius a request may ask for."""


class LatticeCentre(BaseModel):
    """Where the unit shape is drawn around, kept inside the lattice."""

    x: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    y: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)

    model_config = {"extra": "forbid"}


class UnitCircleRequest(BaseModel):
    """Which metric to ask, and where to put its centre."""

    metric: DistanceMetric
    centre: LatticeCentre = LatticeCentre(x=0.0, y=0.0)

    model_config = {"extra": "forbid"}


class UnitCircleResponse(BaseModel):
    """The metric's distance at every lattice cell, and which cells lie within one.

    ``distances``, ``inside`` and ``on_boundary`` are row-major from the
    bottom-left, the convention the region grids use. ``inside_count`` is how
    many cells the shape covers and ``boundary_count`` how many sit at
    distance one exactly, which are the two numbers the page compares across
    metrics.
    """

    metric: DistanceMetric
    centre: LatticeCentre
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cells: int
    distances: list[list[float]]
    inside: list[list[bool]]
    on_boundary: list[list[bool]]
    inside_count: int
    boundary_count: int


def _lattice_coordinates() -> np.ndarray:
    """The cell centres along one axis, exact tenths from -2 to 2."""
    return np.arange(-LATTICE_HALF_CELLS, LATTICE_HALF_CELLS + 1) / float(
        LATTICE_STEPS_PER_UNIT
    )


@router.post("/unit-circle", response_model=UnitCircleResponse)
def unit_circle(request: UnitCircleRequest) -> UnitCircleResponse:
    """Ask the metric how far every lattice cell is from the centre."""
    coordinates = _lattice_coordinates()
    lattice_x, lattice_y = np.meshgrid(coordinates, coordinates)
    lattice_rows = RowBlock(
        np.column_stack([lattice_x.ravel(), lattice_y.ravel()]),
        LATTICE_FEATURE_NAMES,
    )
    centre_row = RowBlock(
        np.array([[request.centre.x, request.centre.y]], dtype=np.float64),
        LATTICE_FEATURE_NAMES,
    )

    distances = request.metric.between(lattice_rows, centre_row).reshape(
        LATTICE_CELLS, LATTICE_CELLS
    )
    inside = distances <= 1.0 + UNIT_TOLERANCE
    on_boundary = np.abs(distances - 1.0) <= UNIT_TOLERANCE

    return UnitCircleResponse(
        metric=request.metric,
        centre=request.centre,
        x_min=float(coordinates[0]),
        x_max=float(coordinates[-1]),
        y_min=float(coordinates[0]),
        y_max=float(coordinates[-1]),
        cells=LATTICE_CELLS,
        distances=[[float(value) for value in row] for row in distances],
        inside=[[bool(value) for value in row] for row in inside],
        on_boundary=[[bool(value) for value in row] for row in on_boundary],
        inside_count=int(inside.sum()),
        boundary_count=int(on_boundary.sum()),
    )


class MetricNeighboursRequest(BaseModel):
    """The stored people, the query, which metric decides near, and k."""

    points: list[LabelledPoint] = Field(min_length=1, max_length=MAX_POINTS)
    query: PlainPoint
    metric: DistanceMetric
    k: int = Field(ge=1, le=MAX_K)

    model_config = {"extra": "forbid"}


class RankedPerson(BaseModel):
    """One stored person, where they came in the ranking, and whether they voted."""

    index: int
    distance: float
    label: int
    chosen: bool


class MetricNeighboursResponse(BaseModel):
    """The vote under this metric, its voters, the whole ranking, and the regions.

    ``neighbours`` holds the k who voted, nearest first. ``ranked`` holds every
    stored person in the same order with the voters flagged, because the
    observed route keeps every distance and the page is about what the other
    distances would have said.
    """

    metric: DistanceMetric
    prediction: int
    neighbours: list[ChosenNeighbour]
    ranked: list[RankedPerson]
    votes_for_zero: int
    votes_for_one: int
    regions: RegionGrid


def _ranking(
    model: KNearestNeighboursClassifier,
    query_features: list[Feature],
    labels: list[int],
) -> tuple[list[ChosenNeighbour], list[RankedPerson]]:
    """The k who voted and the whole crowd ranked, from the observed route."""
    chosen: list[ChosenNeighbour] = []
    ranked: list[RankedPerson] = []
    for query in model.neighbour_search(query_features):
        chosen_positions = {int(index) for index in query.chosen_indices}
        for index, distance in zip(
            query.chosen_indices, query.chosen_distances, strict=True
        ):
            chosen.append(
                ChosenNeighbour(
                    index=int(index),
                    distance=float(distance),
                    label=labels[int(index)],
                )
            )
        for index in np.argsort(query.distances, kind="stable"):
            ranked.append(
                RankedPerson(
                    index=int(index),
                    distance=float(query.distances[index]),
                    label=labels[int(index)],
                    chosen=int(index) in chosen_positions,
                )
            )

    return chosen, ranked


@router.post("/neighbours", response_model=MetricNeighboursResponse)
def neighbours(request: MetricNeighboursRequest) -> MetricNeighboursResponse:
    """Fit on the stored people under the chosen metric and consult the k nearest.

    Asking for more neighbours than there are people is refused by the library
    itself, in its own words, and arrives as a 422 through the error handler.
    """
    heights, weights, labels = features_of(request.points)
    model = KNearestNeighboursClassifier(
        n_neighbours=request.k, metric=request.metric
    ).fit([heights, weights], labels)

    query_features = [
        Feature("height", [request.query.x]),
        Feature("weight", [request.query.y]),
    ]

    prediction = int(np.asarray(model.predict(query_features), dtype=np.float64)[0])
    chosen, ranked = _ranking(
        model, query_features, [point.label for point in request.points]
    )
    votes_for_one = sum(1 for neighbour in chosen if neighbour.label == 1)

    return MetricNeighboursResponse(
        metric=request.metric,
        prediction=prediction,
        neighbours=chosen,
        ranked=ranked,
        votes_for_zero=len(chosen) - votes_for_one,
        votes_for_one=votes_for_one,
        regions=region_grid(model, request.points),
    )


class PairRequest(BaseModel):
    """Two people, to be measured under every metric at once.

    ``order`` asks for one more p-norm beside the fixed sweep. It is bounded
    below by zero and not by one, on purpose: an order below one is accepted
    by the library and is not a metric, and the page shows that rather than
    hiding it.
    """

    first: PlainPoint
    second: PlainPoint
    order: float | None = Field(default=None, gt=0.0, le=MAX_ORDER)

    model_config = {"extra": "forbid"}


class SixDistances(BaseModel):
    """One pair's distance under each of the six metrics, by name."""

    euclidean: float
    manhattan: float
    chebyshev: float
    cosine: float
    hamming: float
    canberra: float


class PairWorking(BaseModel):
    """The intermediate numbers a hand calculation of the six passes through.

    The gap feeds the three p-norms. The dot product and the two lengths feed
    cosine, whose similarity is reported as zero when either row has no length,
    matching the library's convention that a zero row is at distance one from
    everything. The Canberra terms are each feature's gap over the size of its
    values, with 0/0 read as zero as the library reads it, and the count of
    differing features over the feature count is Hamming.
    """

    gap_x: float
    gap_y: float
    dot_product: float
    first_length: float
    second_length: float
    cosine_similarity: float
    canberra_terms: list[float]
    differing_features: int
    feature_count: int


class OrderedNorm(BaseModel):
    """The p-norm of the pair's gap at one order. ``None`` is infinity.

    JSON has no spelling for infinity, so the Chebyshev end of the sweep is
    carried as a missing order and the page writes the symbol itself.
    """

    order: float | None
    distance: float


class EuclideanRoutes(BaseModel):
    """Three ways to the one Euclidean number, and where they part.

    ``direct`` squares the gaps, sums and roots, which is the definition.
    ``expanded_raw`` opens the square into ``|a|^2 - 2 a.b + |b|^2`` on the
    coordinates as given, which is the matrix-multiply form the library uses
    for speed, and it loses the answer when both people sit far from the
    origin, because it recovers a small number as the difference of two large
    ones. ``library`` is the same expansion after both rows are shifted by the
    remembered rows' mean, which is what the library actually computes. The
    first two are definitions worked here; the third is the library's.
    """

    direct: float
    expanded_raw: float
    library: float


class PairResponse(BaseModel):
    """The six distances between the pair, the working, and the p-norm sweep."""

    distances: SixDistances
    working: PairWorking
    orders: list[OrderedNorm]
    at_order: float | None
    euclidean_routes: EuclideanRoutes


def _blocks(first: PlainPoint, second: PlainPoint) -> tuple[RowBlock, RowBlock]:
    """Two one-row blocks under the pair's feature names."""
    first_row = np.array([[first.x, first.y]], dtype=np.float64)
    second_row = np.array([[second.x, second.y]], dtype=np.float64)
    return (
        RowBlock(first_row, PAIR_FEATURE_NAMES),
        RowBlock(second_row, PAIR_FEATURE_NAMES),
    )


def _six(first: RowBlock, second: RowBlock) -> SixDistances:
    """The first row against the second under every named metric."""
    return SixDistances(
        **{
            metric.value: float(metric.between(first, second)[0, 0])
            for metric in DistanceMetric
        }
    )


@router.post("/between", response_model=PairResponse)
def between(request: PairRequest) -> PairResponse:
    """Measure one pair under all six metrics, with the working shown."""
    first_block, second_block = _blocks(request.first, request.second)
    distances = _six(first_block, second_block)

    orders = [
        OrderedNorm(
            order=None if np.isinf(order) else float(order),
            distance=float(
                MinkowskiDistance(order).between(first_block, second_block)[0, 0]
            ),
        )
        for order in MINKOWSKI_ORDERS
    ]
    at_order = (
        None
        if request.order is None
        else float(
            MinkowskiDistance(request.order).between(first_block, second_block)[0, 0]
        )
    )

    first_values = first_block.values[0]
    second_values = second_block.values[0]
    gap = first_values - second_values
    first_length = float(np.sqrt(first_values @ first_values))
    second_length = float(np.sqrt(second_values @ second_values))
    dot_product = float(first_values @ second_values)
    scales = np.abs(first_values) + np.abs(second_values)
    canberra_terms = np.divide(
        np.abs(gap), scales, out=np.zeros_like(gap), where=scales > 0.0
    )

    working = PairWorking(
        gap_x=float(gap[0]),
        gap_y=float(gap[1]),
        dot_product=dot_product,
        first_length=first_length,
        second_length=second_length,
        cosine_similarity=(
            dot_product / (first_length * second_length)
            if first_length > 0.0 and second_length > 0.0
            else 0.0
        ),
        canberra_terms=[float(term) for term in canberra_terms],
        differing_features=int(np.sum(first_values != second_values)),
        feature_count=int(first_values.size),
    )

    expanded_raw = float(
        np.sqrt(max(0.0, first_length**2 - 2.0 * dot_product + second_length**2))
    )
    euclidean_routes = EuclideanRoutes(
        direct=float(np.sqrt(gap @ gap)),
        expanded_raw=expanded_raw,
        library=distances.euclidean,
    )

    return PairResponse(
        distances=distances,
        working=working,
        orders=orders,
        at_order=at_order,
        euclidean_routes=euclidean_routes,
    )


class TripleRequest(BaseModel):
    """Three people, for the four checks a distance has to pass.

    The direct route runs from ``first`` to ``second``; the two-leg route goes
    by way of ``third``.
    """

    first: PlainPoint
    second: PlainPoint
    third: PlainPoint

    model_config = {"extra": "forbid"}


class MetricAxioms(BaseModel):
    """One metric's answers to the four checks on the three people.

    ``self_distance`` is the first person against themselves, which a
    distance must put at zero. ``forward`` and ``backward`` are the direct
    route measured each way, which must agree. ``via`` is the two legs by way
    of the third person summed, which may never fall below ``forward``;
    ``triangle_holds`` is that comparison, with an ulp of tolerance.
    ``origin_self_distance`` is a row of zeros against itself, which is the
    one place cosine has no direction and answers one.
    """

    metric: DistanceMetric
    self_distance: float
    forward: float
    backward: float
    leg_one: float
    leg_two: float
    via: float
    triangle_holds: bool
    origin_self_distance: float


class TripleResponse(BaseModel):
    """The four checks under every metric, in the order the library declares."""

    metrics: list[MetricAxioms]


@router.post("/axioms", response_model=TripleResponse)
def axioms(request: TripleRequest) -> TripleResponse:
    """Put three people through the checks a distance has to pass, six times."""
    first, second = _blocks(request.first, request.second)
    third, _ = _blocks(request.third, request.third)
    origin = RowBlock(np.zeros((1, 2), dtype=np.float64), PAIR_FEATURE_NAMES)

    metrics: list[MetricAxioms] = []
    for metric in DistanceMetric:
        forward = float(metric.between(first, second)[0, 0])
        leg_one = float(metric.between(first, third)[0, 0])
        leg_two = float(metric.between(third, second)[0, 0])
        via = leg_one + leg_two
        metrics.append(
            MetricAxioms(
                metric=metric,
                self_distance=float(metric.between(first, first)[0, 0]),
                forward=forward,
                backward=float(metric.between(second, first)[0, 0]),
                leg_one=leg_one,
                leg_two=leg_two,
                via=via,
                triangle_holds=bool(via >= forward - TRIANGLE_TOLERANCE),
                origin_self_distance=float(metric.between(origin, origin)[0, 0]),
            )
        )

    return TripleResponse(metrics=metrics)


class UnitsRequest(BaseModel):
    """The crowd, the query, the metric, k, and how far to rescale weight.

    ``weight_factor`` multiplies every weight, the query's included, so a
    thousand turns kilograms into grams without changing a single person.
    """

    points: list[LabelledPoint] = Field(min_length=1, max_length=MAX_POINTS)
    query: PlainPoint
    metric: DistanceMetric
    k: int = Field(ge=1, le=MAX_K)
    weight_factor: float = Field(default=1000.0, gt=0.0, le=MAX_WEIGHT_FACTOR)

    model_config = {"extra": "forbid"}


class Treatment(BaseModel):
    """The crowd refitted after one treatment of its columns.

    ``height_share_of_squared_gaps`` is a definition rather than a fit: over
    every stored person, the squared height gap to the query summed, over the
    squared height and weight gaps summed together. It is the share of the
    Euclidean arithmetic one column is doing, whatever metric the ranking
    used, and it is the number that says which column the distance is
    actually reading.
    """

    name: str
    query: PlainPoint
    prediction: int
    neighbours: list[ChosenNeighbour]
    ranked: list[RankedPerson]
    height_share_of_squared_gaps: float


class UnitsResponse(BaseModel):
    """The same crowd fitted raw, with weight rescaled, and standardised."""

    metric: DistanceMetric
    weight_factor: float
    treatments: list[Treatment]


def _treatment(
    name: str,
    heights: Feature,
    weights: Feature,
    labels: Feature,
    query_features: list[Feature],
    point_labels: list[int],
    metric: DistanceMetric,
    k: int,
) -> Treatment:
    """Fit the classifier on one treatment of the columns and rank the crowd."""
    model = KNearestNeighboursClassifier(n_neighbours=k, metric=metric).fit(
        [heights, weights], labels
    )
    prediction = int(np.asarray(model.predict(query_features), dtype=np.float64)[0])
    chosen, ranked = _ranking(model, query_features, point_labels)

    query_x = float(query_features[0].values[0])
    query_y = float(query_features[1].values[0])
    height_gaps = heights.values - query_x
    weight_gaps = weights.values - query_y
    squared_height = float(np.sum(height_gaps**2))
    squared_total = squared_height + float(np.sum(weight_gaps**2))

    return Treatment(
        name=name,
        query=PlainPoint(x=query_x, y=query_y),
        prediction=prediction,
        neighbours=chosen,
        ranked=ranked,
        height_share_of_squared_gaps=(
            squared_height / squared_total if squared_total > 0.0 else 0.0
        ),
    )


@router.post("/units", response_model=UnitsResponse)
def units(request: UnitsRequest) -> UnitsResponse:
    """Refit the crowd raw, with weight rescaled, and standardised.

    The standardised treatment fits the library's standardizer on the crowd
    and passes the query through the same scalings, which is the order the
    pipelines page insists on. A constant column is refused by that fit, in
    the library's words.
    """
    heights, weights, labels = features_of(request.points)
    point_labels = [point.label for point in request.points]
    query = [
        Feature("height", [request.query.x]),
        Feature("weight", [request.query.y]),
    ]

    raw = _treatment(
        "raw", heights, weights, labels, query, point_labels, request.metric, request.k
    )

    scaled_weights = Feature("weight", weights.values * request.weight_factor)
    scaled_query = [
        query[0],
        Feature("weight", [request.query.y * request.weight_factor]),
    ]
    rescaled = _treatment(
        "weight_rescaled",
        heights,
        scaled_weights,
        labels,
        scaled_query,
        point_labels,
        request.metric,
        request.k,
    )

    standardizer = Standardizer().fit([heights, weights])
    standard_heights, standard_weights = standardizer.transform([heights, weights])
    standard_query = standardizer.transform(query)
    standardised = _treatment(
        "standardised",
        standard_heights,
        standard_weights,
        labels,
        standard_query,
        point_labels,
        request.metric,
        request.k,
    )

    return UnitsResponse(
        metric=request.metric,
        weight_factor=request.weight_factor,
        treatments=[raw, rescaled, standardised],
    )


class ModelsRequest(BaseModel):
    """The crowd, unlabelled, and the density clusterer's two settings."""

    points: list[PlainPoint] = Field(min_length=1, max_length=MAX_POINTS)
    radius: float = Field(gt=0.0, le=MAX_RADIUS)
    min_neighbourhood_size: int = Field(default=2, ge=1, le=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DensityUnderMetric(BaseModel):
    """What the density clusterer found on the crowd under one metric.

    ``labels`` carries -1 for a person in no group, the library's own answer
    for noise.
    """

    metric: DistanceMetric
    n_clusters: int
    n_noise: int
    labels: list[int]


class LinkageUnderMetric(BaseModel):
    """Single linkage on the crowd under one metric, cut at two groups.

    Reported because it is the second model that takes the metric as a real
    choice, and because its merge heights are in the metric's own units. Two
    people cut at two groups make no merge at all, and the heights are then
    ``None`` rather than a number invented for the occasion.
    """

    metric: DistanceMetric
    labels: list[int]
    first_merge: float | None
    last_merge: float | None


class CentreArgument(BaseModel):
    """Why k-means takes no metric, worked on the crowd's weight column.

    The total squared gap from a centre is smallest at the mean, and the total
    absolute gap is smallest at the median. Each pair of numbers shows the
    other centre losing, which is the whole of the argument: the update step
    of k-means is a mean, and a mean is the right centre for squared
    Euclidean distance and for nothing else. Definitions computed here.
    """

    mean: float
    median: float
    squared_at_mean: float
    squared_at_median: float
    absolute_at_mean: float
    absolute_at_median: float


class WardRefusal(BaseModel):
    """What happens when ward linkage is paired with a metric other than Euclidean."""

    refused: bool
    message: str


class ModelsResponse(BaseModel):
    """The crowd under every metric on the two models that let it choose."""

    density: list[DensityUnderMetric]
    single_linkage: list[LinkageUnderMetric]
    centre_argument: CentreArgument
    ward_with_manhattan: WardRefusal


def _plain_features(points: list[PlainPoint]) -> list[Feature]:
    """The two columns an unlabelled crowd supplies."""
    return [
        Feature("height", [point.x for point in points]),
        Feature("weight", [point.y for point in points]),
    ]


@router.post("/models", response_model=ModelsResponse)
def models(request: ModelsRequest) -> ModelsResponse:
    """Fit the density and linkage clusterers under every metric, and refuse ward.

    Too few people for the linkage cut is refused by the library and arrives
    as a 422.
    """
    features = _plain_features(request.points)

    density: list[DensityUnderMetric] = []
    single_linkage: list[LinkageUnderMetric] = []
    for metric in DistanceMetric:
        fitted = DBSCAN(
            radius=request.radius,
            min_neighbourhood_size=request.min_neighbourhood_size,
            metric=metric,
        ).fit(features)
        density.append(
            DensityUnderMetric(
                metric=metric,
                n_clusters=fitted.n_clusters,
                n_noise=fitted.n_noise,
                labels=[int(value) for value in fitted.labels],
            )
        )

        linked = AgglomerativeClustering(
            n_clusters=2, linkage=Linkage.SINGLE, metric=metric
        ).fit(features)
        heights_of_merges = np.asarray(linked.merge_distances, dtype=np.float64)
        single_linkage.append(
            LinkageUnderMetric(
                metric=metric,
                labels=[int(value) for value in linked.labels],
                first_merge=(
                    float(heights_of_merges[0]) if heights_of_merges.size else None
                ),
                last_merge=(
                    float(heights_of_merges[-1]) if heights_of_merges.size else None
                ),
            )
        )

    weights = np.array([point.y for point in request.points], dtype=np.float64)
    mean = float(weights.mean())
    median = float(np.median(weights))
    centre_argument = CentreArgument(
        mean=mean,
        median=median,
        squared_at_mean=float(np.sum((weights - mean) ** 2)),
        squared_at_median=float(np.sum((weights - median) ** 2)),
        absolute_at_mean=float(np.sum(np.abs(weights - mean))),
        absolute_at_median=float(np.sum(np.abs(weights - median))),
    )

    try:
        AgglomerativeClustering(
            n_clusters=2, linkage=Linkage.WARD, metric=DistanceMetric.MANHATTAN
        )
        ward = WardRefusal(refused=False, message="")
    except ValidationError as error:
        ward = WardRefusal(refused=True, message=error.errors()[0]["msg"])

    return ModelsResponse(
        density=density,
        single_linkage=single_linkage,
        centre_argument=centre_argument,
        ward_with_manhattan=ward,
    )


class DimensionsRequest(BaseModel):
    """How many uniform rows to draw per feature count, and the seed."""

    n_points: int = Field(default=200, ge=10, le=MAX_DIMENSION_POINTS)
    seed: int = Field(default=0, ge=0, le=1_000_000)

    model_config = {"extra": "forbid"}


class ContrastAtDimension(BaseModel):
    """How much further the farthest row is than the nearest, at one width.

    Every row is a query against every other row. ``nearest`` and
    ``farthest`` are the means over rows of each row's nearest and farthest
    distance, and ``relative_contrast`` is the mean over rows of
    ``(farthest - nearest) / nearest``, which is the library's own measure.
    ``median_relative_contrast`` is the median of the same ratios, kept
    because one row whose nearest neighbour is almost on top of it makes the
    mean enormous without saying anything about the other rows, which is what
    cosine does in two features. Near zero, either way, the nearest row is
    barely nearer than the farthest.
    """

    dimensions: int
    nearest: float
    farthest: float
    relative_contrast: float
    median_relative_contrast: float


class MetricContrast(BaseModel):
    """One metric's contrast at every feature count measured."""

    metric: DistanceMetric
    contrasts: list[ContrastAtDimension]


class DimensionsResponse(BaseModel):
    """Every metric's contrast curve, on rows drawn once and reused."""

    n_points: int
    seed: int
    dimensions: list[int]
    metrics: list[MetricContrast]


def _contrast(metric: DistanceMetric, rows: RowBlock) -> ContrastAtDimension:
    """Each row against every other, and how far the farthest is beyond the nearest."""
    distances = metric.between(rows, rows)
    np.fill_diagonal(distances, np.nan)
    nearest = np.nanmin(distances, axis=1)
    farthest = np.nanmax(distances, axis=1)
    ratio = np.divide(
        farthest - nearest,
        nearest,
        out=np.zeros_like(nearest),
        where=nearest > 0.0,
    )

    return ContrastAtDimension(
        dimensions=rows.n_features,
        nearest=float(nearest.mean()),
        farthest=float(farthest.mean()),
        relative_contrast=float(ratio.mean()),
        median_relative_contrast=float(np.median(ratio)),
    )


@cache
def _dimensions(n_points: int, seed: int) -> DimensionsResponse:
    """Draw the rows once per setting; the page asks for the same one repeatedly."""
    generator = np.random.default_rng(seed)
    widest = generator.random((n_points, max(DIMENSIONS)))

    metrics: list[MetricContrast] = []
    for metric in DistanceMetric:
        contrasts: list[ContrastAtDimension] = []
        for dimensions in DIMENSIONS:
            rows = RowBlock(
                np.ascontiguousarray(widest[:, :dimensions]),
                tuple(f"feature_{position}" for position in range(dimensions)),
            )
            contrasts.append(_contrast(metric, rows))
        metrics.append(MetricContrast(metric=metric, contrasts=contrasts))

    return DimensionsResponse(
        n_points=n_points, seed=seed, dimensions=list(DIMENSIONS), metrics=metrics
    )


@router.post("/dimensions", response_model=DimensionsResponse)
def dimensions(request: DimensionsRequest) -> DimensionsResponse:
    """Measure how distance stops discriminating as the feature count grows.

    The rows in one feature are the first column of the rows in two hundred,
    so every width measures the same draw truncated rather than a fresh one.
    """
    return _dimensions(request.n_points, request.seed)
