"""What near means, for the distance-metrics page.

Three endpoints, each thin over ``DistanceMetric``, the library's closed enum
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
is the derivation's picture of Manhattan sliding into Chebyshev.

There is no seed anywhere in this module. Nothing on the page is random.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    DistanceMetric,
    Feature,
    KNearestNeighboursClassifier,
    MinkowskiDistance,
    RowBlock,
)
from pydantic import BaseModel, Field

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

LATTICE_FEATURE_NAMES = ("horizontal", "vertical")
"""What the lattice's two columns are called. The shapes are abstract."""

PAIR_FEATURE_NAMES = ("height", "weight")
"""What a measured pair's two columns are called, the k-nearest page's names."""

MINKOWSKI_ORDERS: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0, 10.0, np.inf)
"""The orders the between endpoint sweeps, Manhattan through to Chebyshev."""


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
                    label=request.points[int(index)].label,
                )
            )
        for index in np.argsort(query.distances, kind="stable"):
            ranked.append(
                RankedPerson(
                    index=int(index),
                    distance=float(query.distances[index]),
                    label=request.points[int(index)].label,
                    chosen=int(index) in chosen_positions,
                )
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
    """Two people, to be measured under every metric at once."""

    first: PlainPoint
    second: PlainPoint

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


class PairResponse(BaseModel):
    """The six distances between the pair, the working, and the p-norm sweep."""

    distances: SixDistances
    working: PairWorking
    orders: list[OrderedNorm]


@router.post("/between", response_model=PairResponse)
def between(request: PairRequest) -> PairResponse:
    """Measure one pair under all six metrics, with the working shown."""
    first_row = np.array([[request.first.x, request.first.y]], dtype=np.float64)
    second_row = np.array([[request.second.x, request.second.y]], dtype=np.float64)
    first_block = RowBlock(first_row, PAIR_FEATURE_NAMES)
    second_block = RowBlock(second_row, PAIR_FEATURE_NAMES)

    distances = SixDistances(
        **{
            metric.value: float(metric.between(first_block, second_block)[0, 0])
            for metric in DistanceMetric
        }
    )

    orders = [
        OrderedNorm(
            order=None if np.isinf(order) else float(order),
            distance=float(
                MinkowskiDistance(order).between(first_block, second_block)[0, 0]
            ),
        )
        for order in MINKOWSKI_ORDERS
    ]

    first_values = first_row[0]
    second_values = second_row[0]
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

    return PairResponse(distances=distances, working=working, orders=orders)
