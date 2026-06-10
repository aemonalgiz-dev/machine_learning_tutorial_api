"""Searching a collection of pictures by their vectors.

Every endpoint here is registered under
``/concepts/searching-a-collection-of-pictures``, and the router is created and
registered before any endpoint exists, so the pages of this section can be
built in parallel without many edits to one shared file.

The page is about finding the pictures nearest a query, what the exact answer
costs, and what an index gives up to avoid touching every picture.

The collection and the queries
------------------------------
Every picture's position is the reference network's hidden layer of 16, the
vector the page on a vector for a picture explains. A forward pass is cheap,
so the collection is far larger than anything the network was trained on:
1000 pictures of each of the four kinds, drawn under a seed of this page's own
(:data:`COLLECTION_SEED`), so none of them is a picture the network saw. The
queries are the reference split's held-out half, 60 of each kind, which the
network was scored on and never trained on, and which were drawn under a
different seed from the collection, so no query is also in the collection.

What is the library's and what is not
-------------------------------------
Every distance is the library's Euclidean or cosine calculation, the exact
k nearest come from its k-nearest-neighbours search, the cells of the inverted
file are its k-means clustering, and the projections for drawing are its
principal components. Random hyperplane hashing has no class in the library,
so it is written here from its definition, a sign per hyperplane, which is
three lines and is checked against the probability it is meant to have.

The inverted file ranks a probed cell's pictures by the same distances the
exact search computed. A distance between two vectors does not depend on which
other pictures are being searched, so reading it from the one matrix rather
than recomputing it over a subset changes nothing but the last bits of the
arithmetic, and it keeps the exact answer and the index's answer comparable
without a tie being broken two different ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Path
from oop_ml import (
    CosineDistance,
    EuclideanDistance,
    Feature,
    KMeans,
    PrincipalComponentAnalysis,
    RowBlock,
)
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from pydantic import BaseModel, Field

from oop_ml_api.picture_collection import (
    KINDS,
    PictureCollection,
    as_rows,
    drawn_collection,
)
from oop_ml_api.picture_network import (
    NetworkDesign,
    TrainedNetwork,
    accuracy_of,
    reference_network,
    reference_split,
    trained_network,
    vectors_of,
)

router = APIRouter(
    prefix="/concepts/searching-a-collection-of-pictures",
    tags=["searching-a-collection-of-pictures"],
)

COLLECTION_SEED = 11
"""The draw of the searched collection, chosen once and stated on the page."""

COLLECTION_PER_KIND = 1000
"""Pictures of each kind in the collection, so 4000 in all."""

MEASURED_K = 10
"""The neighbour count every summary figure on the page is quoted at."""

PRECISION_KS = (1, 5, 10, 20)
"""The neighbour counts the precision table reports."""

CELL_COUNTS = (16, 32, 64)
"""How many cells the inverted file may cut the collection into."""

KMEANS_RESTARTS = 3
"""Seedings per k-means fit, the best of which is kept."""

KMEANS_SEED = 0
"""The seed of the k-means seedings."""

PROBE_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
"""The numbers of cells probed in the trade-off sweep, where they exist."""

HASH_BITS = (8, 12, 16)
"""How many hyperplanes a hashing table draws, one bit each."""

HASH_RADII = (0, 1, 2, 3, 4)
"""How far in bits from the query's own bucket a hashing search looks."""

HASH_SEED = 0
"""The seed the hyperplanes are drawn under."""

WORKED_BITS = 12
"""The code length of the worked hashing example."""

CHECK_PLANES = 2000
"""Hyperplanes drawn to check the agreement probability, one pair at a time."""

CHECK_PAIRS = 200
"""Pairs of vectors in that check: the first query against 200 pictures."""

CONCENTRATION_SEED = 5
"""The seed of the random vectors the concentration comparison draws."""

RANDOM_DIMENSIONS = (2, 16, 64, 256, 1024)
"""Dimensions of the uniform random vectors set beside the pictures."""

INDEX_ON_RANDOM_DIMENSIONS = (16, 64)
"""Dimensions at which the inverted file is also tried on random vectors."""

INDEX_ON_RANDOM_CELLS = 32
"""Cells of the inverted file in that comparison."""

WORKED_QUERY = 0
"""The query every worked example on the page takes, the first held-out cross."""

# --------------------------------------------------------------------------
# The shared, cached space
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """The collection, the queries, their vectors and every exact distance."""

    network: TrainedNetwork
    collection: PictureCollection
    queries: PictureCollection
    vectors: np.ndarray
    query_vectors: np.ndarray
    distances: np.ndarray
    exact_order: np.ndarray


def _names(width: int) -> list[str]:
    return [f"unit_{position + 1}" for position in range(width)]


def _features(matrix: np.ndarray) -> list[Feature]:
    names = _names(matrix.shape[1])
    return [Feature(name, matrix[:, position]) for position, name in enumerate(names)]


def _block(matrix: np.ndarray) -> RowBlock:
    return RowBlock(
        np.ascontiguousarray(matrix, dtype=np.float64), _names(matrix.shape[1])
    )


@lru_cache(maxsize=1)
def searched_collection() -> PictureCollection:
    """The 4000 pictures every search on the page runs over, drawn once."""
    return drawn_collection(COLLECTION_PER_KIND, seed=COLLECTION_SEED)


@lru_cache(maxsize=1)
def search_space() -> SearchSpace:
    """Draw the collection, give every picture its vector, measure every distance."""
    network = reference_network()
    _, held_out = reference_split()
    collection = searched_collection()
    vectors = vectors_of(network, collection.pictures)
    query_vectors = vectors_of(network, held_out.pictures)
    distances = EuclideanDistance().between(_block(query_vectors), _block(vectors))
    exact_order = np.argsort(distances, axis=1, kind="stable")
    return SearchSpace(
        network=network,
        collection=collection,
        queries=held_out,
        vectors=vectors,
        query_vectors=query_vectors,
        distances=distances,
        exact_order=exact_order,
    )


def _exact_nearest(space: SearchSpace, k: int, positions: np.ndarray) -> np.ndarray:
    """The k nearest pictures to each chosen query, by the neighbour search."""
    search = KNearestNeighboursClassifier(n_neighbours=k).fit(
        _features(space.vectors),
        Feature("kind", space.collection.labels.astype(np.float64)),
    )
    return np.asarray(
        search.neighbour_search(_features(space.query_vectors[positions])).result
    )


@dataclass(frozen=True)
class Cells:
    """An inverted file: the collection cut into cells by k-means."""

    n_cells: int
    labels: np.ndarray
    centres: np.ndarray
    sizes: np.ndarray
    probe_order: np.ndarray
    centre_distances: np.ndarray
    iterations: int
    inertia: float


def _cells_of(vectors: np.ndarray, query_vectors: np.ndarray, n_cells: int) -> Cells:
    fitted = KMeans(
        n_clusters=n_cells, n_initialisations=KMEANS_RESTARTS, random_seed=KMEANS_SEED
    ).fit(_features(vectors))
    labels = np.asarray(fitted.clustering.labels, dtype=np.int64)
    squared = fitted.centroids.squared_distances_to(_block(query_vectors))
    return Cells(
        n_cells=n_cells,
        labels=labels,
        centres=fitted.centroids.positions,
        sizes=np.bincount(labels, minlength=n_cells),
        probe_order=np.argsort(squared, axis=1, kind="stable"),
        centre_distances=np.sqrt(squared),
        iterations=fitted.iterations_run,
        inertia=float(fitted.inertia),
    )


@lru_cache(maxsize=len(CELL_COUNTS))
def cells(n_cells: int) -> Cells:
    """The collection's inverted file at one cell count, fitted once."""
    space = search_space()
    return _cells_of(space.vectors, space.query_vectors, n_cells)


@dataclass(frozen=True)
class Probe:
    """What searching the nearest few cells found for one query."""

    members: np.ndarray
    found: np.ndarray
    recall: float
    distances_computed: int


def _probe(
    cells_used: Cells,
    distances: np.ndarray,
    exact_order: np.ndarray,
    query: int,
    probes: int,
    k: int,
) -> Probe:
    probed = cells_used.probe_order[query, :probes]
    members = np.flatnonzero(np.isin(cells_used.labels, probed))
    ranked = members[np.argsort(distances[query, members], kind="stable")]
    found = ranked[:k]
    truth = set(exact_order[query, :k].tolist())
    return Probe(
        members=members,
        found=found,
        recall=len(truth.intersection(found.tolist())) / k,
        distances_computed=int(cells_used.n_cells + members.size),
    )


def _grid(picture: np.ndarray) -> PictureGrid:
    rows = as_rows(picture)
    flat = [value for row in rows for value in row]
    return PictureGrid(rows=rows, darkest=min(flat), brightest=max(flat))


# --------------------------------------------------------------------------
# Response models shared by several endpoints
# --------------------------------------------------------------------------


class PictureGrid(BaseModel):
    """One 16 by 16 picture as rows of brightness, with its own two ends."""

    rows: list[list[float]]
    darkest: float
    brightest: float


class FoundPicture(BaseModel):
    """One picture a search returned, and where it stands."""

    position: int
    kind: str
    distance: float
    rank: int
    cell: int
    in_exact_answer: bool
    picture: PictureGrid


# --------------------------------------------------------------------------
# The overview: the collection, the exact search, and one distance by hand
# --------------------------------------------------------------------------


class PrecisionRow(BaseModel):
    """At one neighbour count, how often a neighbour shares the query's kind."""

    k: int
    precision: float
    queries_all_same_kind: int


class KindPrecision(BaseModel):
    """Precision at the measured k for the queries of one kind."""

    kind: str
    precision: float


class CoordinatePair(BaseModel):
    """One of the sixteen coordinates of the worked pair, and their gap."""

    query: float
    neighbour: float
    gap: float
    squared_gap: float


class WorkedDistance(BaseModel):
    """The worked query, its nearest picture, and the distance done by hand.

    The coordinates are rounded to three places so a reader can redo the sum,
    and ``distance_from_rounded`` is that sum's root; ``distance`` is the
    exact figure the search used.
    """

    query: int
    query_kind: str
    neighbour: int
    neighbour_kind: str
    coordinates: list[CoordinatePair]
    sum_of_squares_from_rounded: float
    distance_from_rounded: float
    distance: float
    second_distance: float
    farthest_distance: float


class Stranger(BaseModel):
    """A query whose single nearest picture is of another kind."""

    query: int
    query_kind: str
    neighbour: int
    neighbour_kind: str
    distance: float
    same_kind_in_ten: int


class OverviewResponse(BaseModel):
    """The collection, what an exact search returns, and what it costs."""

    collection_seed: int
    collection_size: int
    per_kind: int
    n_queries: int
    dimension: int
    kinds: list[str]
    network_held_out_accuracy: float
    measured_k: int
    precision: list[PrecisionRow]
    by_kind: list[KindPrecision]
    vote_accuracy: float
    chance_precision: float
    worked: WorkedDistance
    strangers: list[Stranger]
    distances_per_query: int
    multiply_adds_per_query: int
    distances_for_every_query: int
    search_agrees_with_distance_ranking: bool


@lru_cache(maxsize=1)
def _overview() -> OverviewResponse:
    space = search_space()
    n_queries = space.query_vectors.shape[0]
    everyone = np.arange(n_queries)
    largest = max(PRECISION_KS)
    nearest = _exact_nearest(space, largest, everyone)
    shared = space.collection.labels[nearest] == space.queries.labels[:, None]

    precision = [
        PrecisionRow(
            k=k,
            precision=float(shared[:, :k].mean()),
            queries_all_same_kind=int(shared[:, :k].all(axis=1).sum()),
        )
        for k in PRECISION_KS
    ]
    by_kind = [
        KindPrecision(
            kind=kind,
            precision=float(shared[space.queries.labels == label, :MEASURED_K].mean()),
        )
        for label, kind in enumerate(KINDS)
    ]
    neighbour_labels = space.collection.labels[nearest[:, :MEASURED_K]]
    votes = np.array(
        [np.bincount(row, minlength=len(KINDS)).argmax() for row in neighbour_labels]
    )

    query = WORKED_QUERY
    neighbour = int(nearest[query, 0])
    query_rounded = np.round(space.query_vectors[query], 3)
    neighbour_rounded = np.round(space.vectors[neighbour], 3)
    gaps = query_rounded - neighbour_rounded
    worked = WorkedDistance(
        query=query,
        query_kind=space.queries.kind_of(query),
        neighbour=neighbour,
        neighbour_kind=space.collection.kind_of(neighbour),
        coordinates=[
            CoordinatePair(
                query=float(first),
                neighbour=float(second),
                gap=round(float(gap), 3),
                squared_gap=round(float(gap) ** 2, 6),
            )
            for first, second, gap in zip(
                query_rounded, neighbour_rounded, gaps, strict=True
            )
        ],
        sum_of_squares_from_rounded=float(np.sum(gaps**2)),
        distance_from_rounded=float(np.sqrt(np.sum(gaps**2))),
        distance=float(space.distances[query, neighbour]),
        second_distance=float(space.distances[query, nearest[query, 1]]),
        farthest_distance=float(space.distances[query].max()),
    )

    strangers = [
        Stranger(
            query=int(position),
            query_kind=space.queries.kind_of(int(position)),
            neighbour=int(nearest[position, 0]),
            neighbour_kind=space.collection.kind_of(int(nearest[position, 0])),
            distance=float(space.distances[position, nearest[position, 0]]),
            same_kind_in_ten=int(shared[position, :MEASURED_K].sum()),
        )
        for position in np.flatnonzero(~shared[:, 0])
    ]

    size, dimension = space.vectors.shape
    agrees = bool(
        np.array_equal(
            np.sort(nearest[:, :MEASURED_K], axis=1),
            np.sort(space.exact_order[:, :MEASURED_K], axis=1),
        )
    )
    return OverviewResponse(
        collection_seed=COLLECTION_SEED,
        collection_size=size,
        per_kind=COLLECTION_PER_KIND,
        n_queries=n_queries,
        dimension=dimension,
        kinds=list(KINDS),
        network_held_out_accuracy=accuracy_of(space.network.stack, space.queries),
        measured_k=MEASURED_K,
        precision=precision,
        by_kind=by_kind,
        vote_accuracy=float((votes == space.queries.labels).mean()),
        chance_precision=float(
            np.mean(
                [
                    np.mean(space.collection.labels == label)
                    for label in space.queries.labels
                ]
            )
        ),
        worked=worked,
        strangers=strangers,
        distances_per_query=size,
        multiply_adds_per_query=size * dimension,
        distances_for_every_query=size * n_queries,
        search_agrees_with_distance_ranking=agrees,
    )


@router.get("/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    """The collection, the exact k nearest for every query, and their cost."""
    return _overview()


# --------------------------------------------------------------------------
# One query, searched exactly and through the inverted file
# --------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Which query, how many neighbours, and how the inverted file is set.

    ``query`` indexes the 240 held-out pictures. ``probes`` may not exceed
    ``n_cells``, and a request asking for more is refused in plain words.
    """

    query: int = Field(default=WORKED_QUERY, ge=0, le=239)
    k: int = Field(default=MEASURED_K, ge=1, le=50)
    n_cells: Literal[16, 32, 64] = 32
    probes: int = Field(default=2, ge=1, le=64)

    model_config = {"extra": "forbid"}


class ProbedCell(BaseModel):
    """One cell the index searched, in the order it was chosen."""

    cell: int
    size: int
    centre_distance: float
    holds_true_neighbours: int


class SearchResponse(BaseModel):
    """One query answered twice, exactly and by the nearest few cells."""

    query: int
    query_kind: str
    query_picture: PictureGrid
    query_cell: int
    k: int
    n_cells: int
    probes: int
    exact: list[FoundPicture]
    indexed: list[FoundPicture]
    exact_precision: float
    recall: float
    exact_distances: int
    index_distances: int
    touched_share: float
    probed: list[ProbedCell]
    kth_distance: float
    next_distance: float


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Search one query exactly and through the inverted file, side by side."""
    if request.probes > request.n_cells:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the collection is cut into {request.n_cells} cells at this "
                f"setting, so {request.probes} of them cannot be searched"
            ),
        )
    space = search_space()
    cells_used = cells(request.n_cells)
    query = request.query
    k = request.k
    exact_positions = _exact_nearest(space, k, np.array([query]))[0]
    truth = set(exact_positions.tolist())
    probe = _probe(
        cells_used, space.distances, space.exact_order, query, request.probes, k
    )

    def found(positions: np.ndarray) -> list[FoundPicture]:
        return [
            FoundPicture(
                position=int(position),
                kind=space.collection.kind_of(int(position)),
                distance=float(space.distances[query, position]),
                rank=rank + 1,
                cell=int(cells_used.labels[position]),
                in_exact_answer=int(position) in truth,
                picture=_grid(space.collection.pictures[position]),
            )
            for rank, position in enumerate(positions)
        ]

    query_label = space.queries.labels[query]
    ordered = np.sort(space.distances[query])
    probed_cells = cells_used.probe_order[query, : request.probes]
    return SearchResponse(
        query=query,
        query_kind=space.queries.kind_of(query),
        query_picture=_grid(space.queries.pictures[query]),
        query_cell=int(cells_used.probe_order[query, 0]),
        k=k,
        n_cells=request.n_cells,
        probes=request.probes,
        exact=found(exact_positions),
        indexed=found(probe.found),
        exact_precision=float(
            np.mean(space.collection.labels[exact_positions] == query_label)
        ),
        recall=probe.recall,
        exact_distances=int(space.vectors.shape[0]),
        index_distances=probe.distances_computed,
        touched_share=float(probe.members.size / space.vectors.shape[0]),
        probed=[
            ProbedCell(
                cell=int(cell),
                size=int(cells_used.sizes[cell]),
                centre_distance=float(cells_used.centre_distances[query, cell]),
                holds_true_neighbours=int(
                    np.sum(cells_used.labels[exact_positions] == cell)
                ),
            )
            for cell in probed_cells
        ],
        kth_distance=float(ordered[k - 1]),
        next_distance=float(ordered[k]),
    )


# --------------------------------------------------------------------------
# The map: the collection projected to two directions, with its cells
# --------------------------------------------------------------------------


class MapPoint(BaseModel):
    """One picture of the collection, placed for drawing."""

    x: float
    y: float
    kind: int
    cells: list[int]


class MapCentre(BaseModel):
    """One cell's centre, placed for drawing."""

    x: float
    y: float
    size: int


class MapCellCount(BaseModel):
    """The centres of the inverted file at one cell count."""

    n_cells: int
    centres: list[MapCentre]


class MapResponse(BaseModel):
    """The collection and the queries on the two directions of greatest spread.

    ``kept_share`` is the share of the sixteen-dimensional spread those two
    directions hold, which is how much of what the search measures the
    drawing can show. ``cell_counts`` follows the order of each point's
    ``cells``.
    """

    points: list[MapPoint]
    queries: list[MapPoint]
    cell_counts: list[MapCellCount]
    kept_share: float


@lru_cache(maxsize=1)
def _map() -> MapResponse:
    space = search_space()
    analysis = PrincipalComponentAnalysis(n_components=2).fit(_features(space.vectors))

    def placed(matrix: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(component.values, dtype=np.float64)
                for component in analysis.transform(_features(matrix))
            ]
        )

    collection_at = placed(space.vectors)
    queries_at = placed(space.query_vectors)
    fits = [cells(count) for count in CELL_COUNTS]
    return MapResponse(
        points=[
            MapPoint(
                x=round(float(point[0]), 4),
                y=round(float(point[1]), 4),
                kind=int(space.collection.labels[position]),
                cells=[int(fit.labels[position]) for fit in fits],
            )
            for position, point in enumerate(collection_at)
        ],
        queries=[
            MapPoint(
                x=round(float(point[0]), 4),
                y=round(float(point[1]), 4),
                kind=int(space.queries.labels[position]),
                cells=[int(fit.probe_order[position, 0]) for fit in fits],
            )
            for position, point in enumerate(queries_at)
        ],
        cell_counts=[
            MapCellCount(
                n_cells=fit.n_cells,
                centres=[
                    MapCentre(x=float(at[0]), y=float(at[1]), size=int(size))
                    for at, size in zip(placed(fit.centres), fit.sizes, strict=True)
                ],
            )
            for fit in fits
        ],
        kept_share=float(sum(analysis.components.variance_shares)),
    )


@router.get("/map", response_model=MapResponse)
def collection_map() -> MapResponse:
    """Every picture and every cell centre on the two widest directions."""
    return _map()


# --------------------------------------------------------------------------
# The trade-off: recall against the share touched, for both routes
# --------------------------------------------------------------------------


class ProbeRow(BaseModel):
    """One number of cells probed, averaged over the 240 queries."""

    probes: int
    recall: float
    worst_recall: float
    queries_with_every_neighbour: int
    worst_query: int
    touched_share: float
    distances_computed: float


class CellCountReport(BaseModel):
    """The inverted file at one cell count.

    ``size_weighted_share`` is the sum of each cell's squared share of the
    collection, which is the share a question drawn the way the collection was
    drawn should expect its first cell to hold. It exceeds one over the cell
    count whenever the cells are uneven.
    """

    n_cells: int
    smallest_cell: int
    largest_cell: int
    empty_cells: int
    size_weighted_share: float
    iterations: int
    rows: list[ProbeRow]


class RadiusRow(BaseModel):
    """One hashing search radius, averaged over the 240 queries."""

    radius: int
    buckets_probed: int
    recall: float
    touched_share: float


class HashReport(BaseModel):
    """One hashing table at one code length."""

    bits: int
    possible_buckets: int
    occupied_buckets: int
    largest_bucket: int
    rows: list[RadiusRow]


class TradeOffResponse(BaseModel):
    """Both routes measured at the page's k against their own exact answers.

    The inverted file is measured against the exact Euclidean k nearest and
    the hashing against the exact cosine k nearest, since each approximates
    its own distance. ``cosine_euclidean_overlap`` is how much those two
    exact answers share.
    """

    k: int
    collection_size: int
    cell_counts: list[CellCountReport]
    hashing: list[HashReport]
    cosine_euclidean_overlap: float


def _cosine_order(space: SearchSpace) -> tuple[np.ndarray, np.ndarray]:
    cosine = CosineDistance().between(
        _block(space.query_vectors), _block(space.vectors)
    )
    return cosine, np.argsort(cosine, axis=1, kind="stable")


@lru_cache(maxsize=1)
def _cosine() -> tuple[np.ndarray, np.ndarray]:
    return _cosine_order(search_space())


def _codes(vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """One bit per hyperplane: which side of it each vector lies on."""
    return (vectors @ planes.T) > 0.0


def _planes(bits: int, dimension: int) -> np.ndarray:
    return np.random.default_rng(HASH_SEED).normal(size=(bits, dimension))


def _index_rows(
    cells_used: Cells, distances: np.ndarray, exact_order: np.ndarray, k: int
) -> list[ProbeRow]:
    n_queries = distances.shape[0]
    size = distances.shape[1]
    rows = []
    for probes in PROBE_COUNTS:
        if probes > cells_used.n_cells:
            continue
        found = [
            _probe(cells_used, distances, exact_order, query, probes, k)
            for query in range(n_queries)
        ]
        recalls = np.array([one.recall for one in found])
        rows.append(
            ProbeRow(
                probes=probes,
                recall=float(recalls.mean()),
                worst_recall=float(recalls.min()),
                queries_with_every_neighbour=int(np.sum(recalls == 1.0)),
                worst_query=int(np.argmin(recalls)),
                touched_share=float(
                    np.mean([one.members.size for one in found]) / size
                ),
                distances_computed=float(
                    np.mean([one.distances_computed for one in found])
                ),
            )
        )
    return rows


@lru_cache(maxsize=1)
def _trade_off() -> TradeOffResponse:
    space = search_space()
    k = MEASURED_K
    reports = []
    for count in CELL_COUNTS:
        fit = cells(count)
        reports.append(
            CellCountReport(
                n_cells=count,
                smallest_cell=int(fit.sizes.min()),
                largest_cell=int(fit.sizes.max()),
                empty_cells=int(np.sum(fit.sizes == 0)),
                size_weighted_share=float(np.sum((fit.sizes / fit.sizes.sum()) ** 2)),
                iterations=fit.iterations,
                rows=_index_rows(fit, space.distances, space.exact_order, k),
            )
        )

    cosine, cosine_order = _cosine()
    truth = [set(row[:k].tolist()) for row in cosine_order]
    hashing = []
    for bits in HASH_BITS:
        planes = _planes(bits, space.vectors.shape[1])
        collection_codes = _codes(space.vectors, planes)
        query_codes = _codes(space.query_vectors, planes)
        apart = (query_codes[:, None, :] != collection_codes[None, :, :]).sum(axis=2)
        _, bucket_sizes = np.unique(collection_codes, axis=0, return_counts=True)
        rows = []
        for radius in HASH_RADII:
            recalls = []
            touched = []
            for query in range(apart.shape[0]):
                members = np.flatnonzero(apart[query] <= radius)
                ranked = members[np.argsort(cosine[query, members], kind="stable")]
                recalls.append(len(truth[query].intersection(ranked[:k].tolist())) / k)
                touched.append(members.size)
            rows.append(
                RadiusRow(
                    radius=radius,
                    buckets_probed=sum(comb(bits, step) for step in range(radius + 1)),
                    recall=float(np.mean(recalls)),
                    touched_share=float(np.mean(touched) / space.vectors.shape[0]),
                )
            )
        hashing.append(
            HashReport(
                bits=bits,
                possible_buckets=2**bits,
                occupied_buckets=int(bucket_sizes.size),
                largest_bucket=int(bucket_sizes.max()),
                rows=rows,
            )
        )

    overlap = np.mean(
        [
            len(set(first[:k].tolist()).intersection(second[:k].tolist())) / k
            for first, second in zip(cosine_order, space.exact_order, strict=True)
        ]
    )
    return TradeOffResponse(
        k=k,
        collection_size=int(space.vectors.shape[0]),
        cell_counts=reports,
        hashing=hashing,
        cosine_euclidean_overlap=float(overlap),
    )


@router.get("/trade-off", response_model=TradeOffResponse)
def trade_off() -> TradeOffResponse:
    """Recall of the true k nearest against the share of the collection touched."""
    return _trade_off()


# --------------------------------------------------------------------------
# Hashing by random hyperplanes, checked and worked
# --------------------------------------------------------------------------


class CodedPicture(BaseModel):
    """One picture's code, and how it compares with the query's."""

    position: int
    kind: str
    code: str
    angle_degrees: float
    bits_agreeing: int
    expected_agreeing: float
    picture: PictureGrid


class HashingResponse(BaseModel):
    """Random hyperplane hashing, worked on one query and checked in bulk.

    ``largest_gap`` and ``mean_gap`` compare, over ``check_pairs`` pairs, the
    share of ``check_planes`` hyperplanes that put the two vectors on the
    same side against one minus their angle over pi. ``standard_error`` is
    the largest spread that share can have from the finite count of planes.
    """

    bits: int
    query: int
    query_kind: str
    query_code: str
    query_picture: PictureGrid
    others: list[CodedPicture]
    check_planes: int
    check_pairs: int
    largest_gap: float
    mean_gap: float
    standard_error: float
    all_positive_coordinates: int
    mean_vector_length: float


@lru_cache(maxsize=1)
def _hashing() -> HashingResponse:
    space = search_space()
    cosine, cosine_order = _cosine()
    query = WORKED_QUERY
    query_vector = space.query_vectors[query]
    planes = _planes(WORKED_BITS, space.vectors.shape[1])
    query_code = _codes(query_vector[None, :], planes)[0]

    nearest = int(cosine_order[query, 0])
    query_label = space.queries.labels[query]
    other_kinds = np.flatnonzero(space.collection.labels != query_label)
    farthest_other = int(other_kinds[np.argmax(cosine[query, other_kinds])])
    median_position = int(cosine_order[query, cosine_order.shape[1] // 2])

    def angle(position: int) -> float:
        other = space.vectors[position]
        cosine_value = float(
            query_vector @ other / np.linalg.norm(query_vector) / np.linalg.norm(other)
        )
        return float(np.degrees(np.arccos(np.clip(cosine_value, -1.0, 1.0))))

    def coded(position: int) -> CodedPicture:
        code = _codes(space.vectors[position][None, :], planes)[0]
        degrees = angle(position)
        return CodedPicture(
            position=position,
            kind=space.collection.kind_of(position),
            code="".join("1" if bit else "0" for bit in code),
            angle_degrees=degrees,
            bits_agreeing=int(np.sum(code == query_code)),
            expected_agreeing=WORKED_BITS * (1.0 - degrees / 180.0),
            picture=_grid(space.collection.pictures[position]),
        )

    check_planes = np.random.default_rng(HASH_SEED + 1).normal(
        size=(CHECK_PLANES, space.vectors.shape[1])
    )
    pairs = space.vectors[:CHECK_PAIRS]
    cosines = (pairs @ query_vector) / (
        np.linalg.norm(pairs, axis=1) * np.linalg.norm(query_vector)
    )
    theta = np.arccos(np.clip(cosines, -1.0, 1.0))
    agreeing = (
        _codes(pairs, check_planes) == _codes(query_vector[None, :], check_planes)
    ).mean(axis=1)
    gaps = np.abs(agreeing - (1.0 - theta / np.pi))

    return HashingResponse(
        bits=WORKED_BITS,
        query=query,
        query_kind=space.queries.kind_of(query),
        query_code="".join("1" if bit else "0" for bit in query_code),
        query_picture=_grid(space.queries.pictures[query]),
        others=[coded(nearest), coded(median_position), coded(farthest_other)],
        check_planes=CHECK_PLANES,
        check_pairs=CHECK_PAIRS,
        largest_gap=float(gaps.max()),
        mean_gap=float(gaps.mean()),
        standard_error=float(np.sqrt(0.25 / CHECK_PLANES)),
        all_positive_coordinates=int(np.sum(space.vectors.mean(axis=0) > 0.0)),
        mean_vector_length=float(np.linalg.norm(space.vectors, axis=1).mean()),
    )


@router.get("/hashing", response_model=HashingResponse)
def hashing() -> HashingResponse:
    """Twelve hyperplanes on the worked query, and the probability they keep."""
    return _hashing()


# --------------------------------------------------------------------------
# Concentration: nearest against farthest, pictures against random vectors
# --------------------------------------------------------------------------


class RatioRow(BaseModel):
    """Nearest over farthest distance, per query, for one kind of vector."""

    label: str
    dimension: int
    mean_ratio: float
    median_ratio: float
    smallest_ratio: float
    largest_ratio: float


class RandomIndexRow(BaseModel):
    """The inverted file on one kind of vector, at a few probe counts."""

    label: str
    dimension: int
    rows: list[ProbeRow]


class ConcentrationResponse(BaseModel):
    """How far apart nearest and farthest are, and what that does to an index.

    ``spread_shares`` are the picture vectors' principal component shares,
    largest first, and ``components_for_ninety`` how many of them it takes to
    hold nine tenths of the spread; ``uniform_spread_shares`` is the same for
    uniform random vectors of the same dimension.
    """

    collection_size: int
    n_queries: int
    ratios: list[RatioRow]
    index_rows: list[RandomIndexRow]
    spread_shares: list[float]
    components_for_ninety: int
    uniform_spread_shares: list[float]
    uniform_components_for_ninety: int


def _ratio_row(label: str, dimension: int, distances: np.ndarray) -> RatioRow:
    ratio = distances.min(axis=1) / distances.max(axis=1)
    return RatioRow(
        label=label,
        dimension=dimension,
        mean_ratio=float(ratio.mean()),
        median_ratio=float(np.median(ratio)),
        smallest_ratio=float(ratio.min()),
        largest_ratio=float(ratio.max()),
    )


def _shares(matrix: np.ndarray) -> list[float]:
    analysis = PrincipalComponentAnalysis().fit(_features(matrix))
    return [float(share) for share in analysis.components.variance_shares]


def _needed_for_ninety(shares: list[float]) -> int:
    return int(np.searchsorted(np.cumsum(shares), 0.9) + 1)


@lru_cache(maxsize=1)
def _concentration() -> ConcentrationResponse:
    space = search_space()
    size, n_queries = space.vectors.shape[0], space.query_vectors.shape[0]
    generator = np.random.default_rng(CONCENTRATION_SEED)

    ratios = [_ratio_row("picture vectors", space.vectors.shape[1], space.distances)]

    pixels = space.collection.pictures.reshape(size, -1)
    query_pixels = space.queries.pictures.reshape(n_queries, -1)
    ratios.append(
        _ratio_row(
            "raw pixels",
            pixels.shape[1],
            EuclideanDistance().between(_block(query_pixels), _block(pixels)),
        )
    )

    random_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dimension in RANDOM_DIMENSIONS:
        remembered = generator.uniform(-1.0, 1.0, size=(size, dimension))
        asked = generator.uniform(-1.0, 1.0, size=(n_queries, dimension))
        random_sets[dimension] = (remembered, asked)
        ratios.append(
            _ratio_row(
                "uniform random",
                dimension,
                EuclideanDistance().between(_block(asked), _block(remembered)),
            )
        )

    index_rows = [
        RandomIndexRow(
            label="picture vectors",
            dimension=space.vectors.shape[1],
            rows=_index_rows(
                cells(INDEX_ON_RANDOM_CELLS),
                space.distances,
                space.exact_order,
                MEASURED_K,
            ),
        )
    ]
    for dimension in INDEX_ON_RANDOM_DIMENSIONS:
        remembered, asked = random_sets[dimension]
        distances = EuclideanDistance().between(_block(asked), _block(remembered))
        order = np.argsort(distances, axis=1, kind="stable")
        fit = _cells_of(remembered, asked, INDEX_ON_RANDOM_CELLS)
        index_rows.append(
            RandomIndexRow(
                label="uniform random",
                dimension=dimension,
                rows=_index_rows(fit, distances, order, MEASURED_K),
            )
        )

    shares = _shares(space.vectors)
    uniform_shares = _shares(random_sets[space.vectors.shape[1]][0])
    return ConcentrationResponse(
        collection_size=size,
        n_queries=n_queries,
        ratios=ratios,
        index_rows=index_rows,
        spread_shares=shares,
        components_for_ninety=_needed_for_ninety(shares),
        uniform_spread_shares=uniform_shares,
        uniform_components_for_ninety=_needed_for_ninety(uniform_shares),
    )


@router.get("/concentration", response_model=ConcentrationResponse)
def concentration() -> ConcentrationResponse:
    """Nearest against farthest on the pictures and on random vectors."""
    return _concentration()


# --------------------------------------------------------------------------
# Seeds: the figures that depend on training, under other starting weights
# --------------------------------------------------------------------------


class SeedResponse(BaseModel):
    """The search's headline figures under one network's starting weights."""

    weight_seed: int
    network_held_out_accuracy: float
    precision_at_k: float
    precision_at_one: float
    index_recall: float
    index_touched_share: float
    k: int
    n_cells: int
    probes: int


SEED_PROBES = 2
"""Cells probed in the seed comparison, at :data:`INDEX_ON_RANDOM_CELLS` cells."""


@lru_cache(maxsize=3)
def _seed(weight_seed: int) -> SeedResponse:
    training, held_out = reference_split()
    network = (
        reference_network()
        if weight_seed == 0
        else trained_network(training, NetworkDesign(weight_seed=weight_seed))
    )
    collection = searched_collection()
    vectors = vectors_of(network, collection.pictures)
    query_vectors = vectors_of(network, held_out.pictures)
    distances = EuclideanDistance().between(_block(query_vectors), _block(vectors))
    order = np.argsort(distances, axis=1, kind="stable")
    shared = collection.labels[order[:, :MEASURED_K]] == held_out.labels[:, None]
    fit = _cells_of(vectors, query_vectors, INDEX_ON_RANDOM_CELLS)
    row = next(
        one
        for one in _index_rows(fit, distances, order, MEASURED_K)
        if one.probes == SEED_PROBES
    )
    return SeedResponse(
        weight_seed=weight_seed,
        network_held_out_accuracy=accuracy_of(network.stack, held_out),
        precision_at_k=float(shared.mean()),
        precision_at_one=float(shared[:, 0].mean()),
        index_recall=row.recall,
        index_touched_share=row.touched_share,
        k=MEASURED_K,
        n_cells=INDEX_ON_RANDOM_CELLS,
        probes=SEED_PROBES,
    )


@router.get("/seeds/{weight_seed}", response_model=SeedResponse)
def seed(weight_seed: int = Path(ge=0, le=2)) -> SeedResponse:
    """Precision and index recall with the network trained from one weight seed.

    One seed per request, because each is a network trained from scratch and
    two in one request would not answer inside the page's time budget.
    """
    return _seed(weight_seed)
