"""A vector for a picture, read off a network trained to name it.

Every endpoint here is registered under ``/concepts/a-vector-for-a-picture``.
The page is about the layer before a classifier's answer as a position for a
picture, and whether pictures of one kind land near each other there.

What the page measures
----------------------
The shared reference network (``picture_network``) is trained to name four
kinds of picture and nothing else. Its hidden layer of sixteen is read off as
each picture's vector, and every question is asked of the held-out half of the
reference collection, 60 pictures of each kind the network never learned from.

* ``/walk`` takes the network apart layer by layer and puts the first four
  held-out pictures, one of each kind, through it, so a reader can check a
  handful of cosines by hand.
* ``/sphere`` places every held-out picture, and 60 rings, as a direction on a
  sphere, for the trained network and for an untrained one of the same design.
  The projection is fitted on the held-out four kinds only and the rings are
  then placed with it, because a projection chosen with the rings in view
  would already know something about them.
* ``/neighbours`` answers one picture's nearest pictures twice, by angle
  between vectors and by straight-line distance between pixels.
* ``/seed/0``, ``/seed/1`` and ``/seed/2`` repeat every measurement under one
  weight seed each. Seed 0 is the reference network; each of the other two
  trains one more network, which keeps every endpoint inside the compute
  budget, and the page reads all three so no claim rests on one seed.
* ``/edges`` holds the cases where the vector stops meaning what the page
  says it means: a picture with nothing in it, a brightness training never
  showed, and what a vector's length carries.

Every nearest-neighbour count is leave-one-out inside the held-out half, a
picture's nearest other picture, and "by angle" means the cosine between the
two vectors, which is the measure the word page argues for and the one
``/concepts/distance-and-similarity`` defines.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Conv2d,
    DenseLayer,
    Feature,
    Flatten,
    LayerStack,
    MaxPool2d,
    PrincipalComponentAnalysis,
)
from pydantic import BaseModel, Field

from oop_ml_api.picture_collection import (
    DEFAULT_SEED,
    KINDS,
    UNSEEN_KIND,
    PictureCollection,
    as_rows,
    drawn_collection,
)
from oop_ml_api.picture_network import (
    DEFAULT_EPOCHS,
    NetworkDesign,
    TrainedNetwork,
    accuracy_of,
    built_network,
    reference_network,
    reference_split,
    trained_network,
)

router = APIRouter(
    prefix="/concepts/a-vector-for-a-picture", tags=["a-vector-for-a-picture"]
)

N_RINGS = 60
"""Rings drawn for the unseen kind, as many as each held-out kind holds."""

SEEDS = (0, 1, 2)
"""The weight seeds the page reports over. Seed 0 is the reference network."""

FLATTENED, HIDDEN, SCORES = 4, 5, 6
"""Positions in the stack of the three layers a vector could be read from."""

BRIGHTNESS_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
"""Whole-picture multipliers for the brightness sweep, 1.0 being the picture."""

SATURATED = 0.95
"""A coordinate beyond this in size is counted as pressed against its bound."""

KindName = Literal["cross", "square", "disc", "bar", "ring"]
ALL_KINDS: tuple[str, ...] = (*KINDS, UNSEEN_KIND)


# ---------------------------------------------------------------------------
# The shared arithmetic
# ---------------------------------------------------------------------------


def _held_out() -> PictureCollection:
    return reference_split()[1]


@lru_cache(maxsize=1)
def _rings() -> PictureCollection:
    return drawn_collection(N_RINGS, seed=DEFAULT_SEED, kinds=(UNSEEN_KIND,))


@lru_cache(maxsize=len(SEEDS))
def _network(seed: int) -> TrainedNetwork:
    """The reference network for seed 0, otherwise the same design retrained."""
    if seed == 0:
        return reference_network()
    training, _ = reference_split()
    return trained_network(training, NetworkDesign(weight_seed=seed))


@lru_cache(maxsize=len(SEEDS))
def _untrained(seed: int) -> LayerStack:
    """The same design with its starting weights, never shown a picture."""
    return built_network(NetworkDesign(weight_seed=seed), len(KINDS))


def _layer_rows(stack: LayerStack, pictures: np.ndarray) -> list[np.ndarray]:
    """Every layer's answer for every picture, each flattened to one row."""
    response = stack.respond_to(pictures)
    return [
        np.asarray(response[position].outputs, dtype=np.float64).reshape(
            len(pictures), -1
        )
        for position in range(len(stack))
    ]


def _pixels(collection: PictureCollection) -> np.ndarray:
    return collection.pictures.reshape(len(collection), -1)


def _unit(rows: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(lengths == 0.0, 1.0, lengths)


def _cosines(rows: np.ndarray) -> np.ndarray:
    directions = _unit(rows)
    return directions @ directions.T


def _squared_gaps(rows: np.ndarray) -> np.ndarray:
    squares = (rows**2).sum(axis=1)
    return np.maximum(squares[:, None] - 2.0 * rows @ rows.T + squares[None, :], 0.0)


def _nearest_by_angle(rows: np.ndarray) -> np.ndarray:
    """Each row's nearest other row by cosine; ties go to the earlier row."""
    similarity = _cosines(rows)
    np.fill_diagonal(similarity, -np.inf)
    return similarity.argmax(axis=1)


def _nearest_by_gap(rows: np.ndarray) -> np.ndarray:
    """Each row's nearest other row by straight-line distance."""
    gaps = _squared_gaps(rows)
    np.fill_diagonal(gaps, np.inf)
    return gaps.argmin(axis=1)


def _own_kind_share(nearest: np.ndarray, labels: np.ndarray) -> float:
    return float((labels[nearest] == labels).mean())


def _within_and_across(rows: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean cosine between two different pictures of one kind, and of two."""
    similarity = _cosines(rows)
    same = labels[:, None] == labels[None, :]
    different_picture = ~np.eye(len(labels), dtype=bool)
    return (
        float(similarity[same & different_picture].mean()),
        float(similarity[~same].mean()),
    )


def _probabilities(scores: np.ndarray) -> np.ndarray:
    lifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return lifted / lifted.sum(axis=-1, keepdims=True)


def _parameter_count(layer: object) -> int:
    if isinstance(layer, Conv2d):
        return int(layer.kernels.size + layer.bias_vector.size)
    if isinstance(layer, DenseLayer):
        return int(layer.weight_matrix.size + layer.bias_vector.size)
    return 0


def _layer_name(layer: object, position: int, stack: LayerStack) -> str:
    if isinstance(layer, Conv2d):
        return f"convolve with {layer.n_filters} filters"
    if isinstance(layer, MaxPool2d):
        return "keep the largest of each two by two"
    if isinstance(layer, Flatten):
        return "lay the channels out in one row"
    if position == len(stack) - 2:
        return "dense layer of 16, the vector"
    return "dense layer of 4, one score per kind"


# ---------------------------------------------------------------------------
# /walk
# ---------------------------------------------------------------------------


class LayerRow(BaseModel):
    """One layer of the network, what it hands on and what it learned."""

    position: int
    name: str
    answer_shape: list[int]
    n_numbers: int
    n_parameters: int


class WalkedPicture(BaseModel):
    """One picture taken through the network and read at the vector."""

    position: int
    kind: str
    rows: list[list[float]]
    vector: list[float]
    length: float
    scores: list[float]
    probabilities: list[float]
    called: str


class WorkedPair(BaseModel):
    """One cosine written out, so it can be checked from the two vectors."""

    first: str
    second: str
    dot: float
    first_length: float
    second_length: float
    cosine: float


class WalkResponse(BaseModel):
    """The network, and the first four held-out pictures read through it.

    ``cosines`` and ``pixel_gaps`` are four by four in the order of
    ``pictures``, the first by angle between vectors and the second by
    straight-line distance between the 256 pixels.
    """

    layers: list[LayerRow]
    n_parameters: int
    epochs: int
    n_training: int
    n_held_out: int
    per_kind_held_out: int
    held_out_accuracy: float
    training_accuracy: float
    kinds: list[str]
    pictures: list[WalkedPicture]
    cosines: list[list[float]]
    pixel_gaps: list[list[float]]
    worked: WorkedPair


def _walked(
    stack: LayerStack, collection: PictureCollection, position: int
) -> WalkedPicture:
    picture = collection.pictures[position : position + 1]
    rows = _layer_rows(stack, picture)
    vector = rows[HIDDEN][0]
    scores = rows[SCORES][0]
    return WalkedPicture(
        position=position,
        kind=collection.kind_of(position),
        rows=as_rows(picture[0]),
        vector=[float(value) for value in vector],
        length=float(np.linalg.norm(vector)),
        scores=[float(value) for value in scores],
        probabilities=[float(value) for value in _probabilities(scores)],
        called=KINDS[int(scores.argmax())],
    )


@router.get("/walk", response_model=WalkResponse)
@lru_cache(maxsize=1)
def walk() -> WalkResponse:
    """The reference network layer by layer, and four pictures through it."""
    network = reference_network()
    training, held_out = reference_split()
    stack = network.stack
    layers = [
        LayerRow(
            position=position,
            name=_layer_name(layer, position, stack),
            answer_shape=list(layer.shape.answers),
            n_numbers=int(layer.shape.n_outputs),
            n_parameters=_parameter_count(layer),
        )
        for position, layer in enumerate(stack)
    ]
    first_of_each = [
        int(np.flatnonzero(held_out.labels == label)[0]) for label in range(len(KINDS))
    ]
    pictures = [_walked(stack, held_out, position) for position in first_of_each]
    vectors = np.asarray([picture.vector for picture in pictures])
    pixels = _pixels(held_out)[first_of_each]
    cross, square = vectors[0], vectors[1]
    return WalkResponse(
        layers=layers,
        n_parameters=sum(row.n_parameters for row in layers),
        epochs=DEFAULT_EPOCHS,
        n_training=len(training),
        n_held_out=len(held_out),
        per_kind_held_out=len(held_out) // len(KINDS),
        held_out_accuracy=accuracy_of(stack, held_out),
        training_accuracy=network.history[-1].training_accuracy,
        kinds=list(KINDS),
        pictures=pictures,
        cosines=_cosines(vectors).tolist(),
        pixel_gaps=np.sqrt(_squared_gaps(pixels)).tolist(),
        worked=WorkedPair(
            first=pictures[0].kind,
            second=pictures[1].kind,
            dot=float(cross @ square),
            first_length=float(np.linalg.norm(cross)),
            second_length=float(np.linalg.norm(square)),
            cosine=float(
                cross @ square / (np.linalg.norm(cross) * np.linalg.norm(square))
            ),
        ),
    )


# ---------------------------------------------------------------------------
# /sphere
# ---------------------------------------------------------------------------


class SpherePoint(BaseModel):
    """One picture as a direction, pushed back onto the sphere after projecting."""

    x: float
    y: float
    z: float
    kind: str
    position: int


class SphereView(BaseModel):
    """One network's arrangement of the pictures.

    ``kept_share`` is the spread of the four kinds' directions the three drawn
    axes hold. ``within`` and ``across`` are mean cosines measured in the full
    sixteen numbers rather than in the drawing, and ``nearest_own_kind`` is the
    share of held-out pictures whose nearest other picture is their own kind.
    """

    network: str
    points: list[SpherePoint]
    kept_share: float
    within: float
    across: float
    nearest_own_kind: float


class SphereResponse(BaseModel):
    """The trained reference network and an untrained one of the same design."""

    dimension: int
    n_pictures: int
    n_rings: int
    views: list[SphereView]


def _sphere_view(name: str, stack: LayerStack) -> SphereView:
    held_out = _held_out()
    rings = _rings()
    held_rows = _unit(_layer_rows(stack, held_out.pictures)[HIDDEN])
    ring_rows = _unit(_layer_rows(stack, rings.pictures)[HIDDEN])
    names = [f"coordinate {position + 1}" for position in range(held_rows.shape[1])]

    def columns(rows: np.ndarray) -> list[Feature]:
        return [Feature(names[at], rows[:, at]) for at in range(rows.shape[1])]

    analysis = PrincipalComponentAnalysis(n_components=3).fit(columns(held_rows))

    def placed(rows: np.ndarray) -> np.ndarray:
        drawn = np.column_stack(
            [
                np.asarray(component.values, dtype=np.float64)
                for component in analysis.transform(columns(rows))
            ]
        )
        return _unit(drawn)

    points = [
        SpherePoint(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            kind=held_out.kind_of(position),
            position=position,
        )
        for position, point in enumerate(placed(held_rows))
    ] + [
        SpherePoint(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            kind=UNSEEN_KIND,
            position=position,
        )
        for position, point in enumerate(placed(ring_rows))
    ]
    within, across = _within_and_across(held_rows, held_out.labels)
    return SphereView(
        network=name,
        points=points,
        kept_share=float(sum(analysis.components.variance_shares)),
        within=within,
        across=across,
        nearest_own_kind=_own_kind_share(_nearest_by_angle(held_rows), held_out.labels),
    )


@router.get("/sphere", response_model=SphereResponse)
@lru_cache(maxsize=1)
def sphere() -> SphereResponse:
    """Every held-out picture and every ring as a direction on a sphere."""
    views = [
        _sphere_view("trained", reference_network().stack),
        _sphere_view("untrained", _untrained(0)),
    ]
    return SphereResponse(
        dimension=reference_network().design.hidden_width,
        n_pictures=len(_held_out()),
        n_rings=len(_rings()),
        views=views,
    )


# ---------------------------------------------------------------------------
# /neighbours
# ---------------------------------------------------------------------------


class NeighboursRequest(BaseModel):
    """Which picture to stand at, and which network reads it."""

    model_config = {"extra": "forbid"}

    collection: Literal["held_out", "rings"] = "held_out"
    position: int = Field(0, ge=0, le=239)
    network: Literal["trained", "untrained"] = "trained"
    with_rings: bool = False
    n_results: int = Field(4, ge=1, le=8)


class Neighbour(BaseModel):
    """One picture found near the query, and how near."""

    collection: str
    position: int
    kind: str
    rows: list[list[float]]
    cosine: float
    gap: float


class NeighboursResponse(BaseModel):
    """One picture's nearest pictures by vector and by pixels.

    ``pixel_misses`` lists every held-out picture whose nearest other picture
    by pixels is a different kind, and ``vector_misses`` the same by the
    chosen network's vectors, both counted over the held-out half alone, so a
    widget can offer a picture the pixels get wrong.
    """

    collection: str
    position: int
    kind: str
    rows: list[list[float]]
    called: str
    probabilities: list[float]
    length: float
    pool_size: int
    by_vector: list[Neighbour]
    by_pixels: list[Neighbour]
    pixel_misses: list[int]
    vector_misses: list[int]


@lru_cache(maxsize=2)
def _misses(network: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    held_out = _held_out()
    stack = reference_network().stack if network == "trained" else _untrained(0)
    vectors = _layer_rows(stack, held_out.pictures)[HIDDEN]
    labels = held_out.labels
    by_pixels = _nearest_by_gap(_pixels(held_out))
    by_vector = _nearest_by_angle(vectors)
    return (
        tuple(int(at) for at in np.flatnonzero(labels[by_pixels] != labels)),
        tuple(int(at) for at in np.flatnonzero(labels[by_vector] != labels)),
    )


@router.post("/neighbours", response_model=NeighboursResponse)
def neighbours(request: NeighboursRequest) -> NeighboursResponse:
    """Stand at one picture and list its nearest pictures two ways."""
    held_out = _held_out()
    rings = _rings()
    if request.collection == "rings" and request.position >= len(rings):
        raise HTTPException(
            status_code=422,
            detail=(
                f"there are {len(rings)} rings, numbered 0 to {len(rings) - 1}, "
                f"so there is no ring {request.position}"
            ),
        )
    stack = reference_network().stack if request.network == "trained" else _untrained(0)
    include_rings = request.with_rings or request.collection == "rings"
    pool_pictures = held_out.pictures
    pool_labels = [held_out.kind_of(at) for at in range(len(held_out))]
    pool_origin = [("held_out", at) for at in range(len(held_out))]
    if include_rings:
        pool_pictures = np.concatenate([pool_pictures, rings.pictures])
        pool_labels += [UNSEEN_KIND] * len(rings)
        pool_origin += [("rings", at) for at in range(len(rings))]
    query_index = (
        request.position
        if request.collection == "held_out"
        else len(held_out) + request.position
    )

    layer_rows = _layer_rows(stack, pool_pictures)
    vectors = layer_rows[HIDDEN]
    pixels = pool_pictures.reshape(len(pool_pictures), -1)
    directions = _unit(vectors)
    cosines = directions @ directions[query_index]
    gaps = np.sqrt(((pixels - pixels[query_index]) ** 2).sum(axis=1))
    others = np.array([at != query_index for at in range(len(pool_pictures))])

    def listed(order: np.ndarray) -> list[Neighbour]:
        chosen = [int(at) for at in order if others[at]][: request.n_results]
        return [
            Neighbour(
                collection=pool_origin[at][0],
                position=pool_origin[at][1],
                kind=pool_labels[at],
                rows=as_rows(pool_pictures[at]),
                cosine=float(cosines[at]),
                gap=float(gaps[at]),
            )
            for at in chosen
        ]

    scores = layer_rows[SCORES][query_index]
    pixel_misses, vector_misses = _misses(request.network)
    return NeighboursResponse(
        collection=request.collection,
        position=request.position,
        kind=pool_labels[query_index],
        rows=as_rows(pool_pictures[query_index]),
        called=KINDS[int(scores.argmax())],
        probabilities=[float(value) for value in _probabilities(scores)],
        length=float(np.linalg.norm(vectors[query_index])),
        pool_size=len(pool_pictures),
        by_vector=listed(np.argsort(-cosines, kind="stable")),
        by_pixels=listed(np.argsort(gaps, kind="stable")),
        pixel_misses=list(pixel_misses),
        vector_misses=list(vector_misses),
    )


# ---------------------------------------------------------------------------
# /seed/0, /seed/1, /seed/2
# ---------------------------------------------------------------------------


class Representation(BaseModel):
    """One way of turning a picture into numbers, and how well kinds gather.

    ``measure`` says whether nearness was read as an angle or as a
    straight-line distance. ``trained`` is ``None`` for the pixels, which no
    network produced.
    """

    name: str
    layer: Literal["pixels", "pixels less their mean", "flattened", "hidden", "scores"]
    trained: bool | None
    measure: Literal["angle", "distance"]
    width: int
    nearest_own_kind: float
    within: float
    across: float


class RingStudy(BaseModel):
    """Where the sixty rings land under one seed.

    ``called`` counts what the network names each ring, one count per trained
    kind. ``nearest_*`` count, over the five kinds, which kind each ring's
    nearest other picture is when the rings are pooled with the held-out half.
    ``gathered`` is how many rings have another ring as their nearest picture,
    per way of reading a picture.
    """

    called: list[int]
    nearest_by_vector: list[int]
    nearest_by_pixels: list[int]
    nearest_by_untrained: list[int]
    ring_to_ring: float
    ring_to_kind: list[float]
    gathered: dict[str, int]


class AgainstReference(BaseModel):
    """How this seed's vectors compare with the reference network's.

    ``same_picture_cosine`` is the mean cosine between one picture's vector
    here and the same picture's vector under seed 0, which a coordinate that
    meant something fixed would hold near one. ``same_nearest_picture`` is the
    share of pictures whose nearest other picture is the same picture under
    both seeds, and ``same_nearest_kind`` the share whose nearest picture is
    the same kind under both.
    """

    same_picture_cosine: float
    matching_coordinate_correlation: float
    same_nearest_picture: float
    same_nearest_kind: float


class SeedResponse(BaseModel):
    """Every measurement the page repeats across weight seeds, for one seed."""

    weight_seed: int
    held_out_accuracy: float
    untrained_accuracy: float
    kinds: list[str]
    representations: list[Representation]
    vector_confusion: list[list[int]]
    pixel_confusion: list[list[int]]
    rings: RingStudy
    against_reference: AgainstReference | None


def _confusion(nearest: np.ndarray, labels: np.ndarray, size: int) -> list[list[int]]:
    table = np.zeros((size, size), dtype=np.int64)
    for truth, found in zip(labels, labels[nearest], strict=True):
        table[truth, found] += 1
    return table.tolist()


def _ring_counts(held_rows: np.ndarray, ring_rows: np.ndarray, by_angle: bool):
    held_out = _held_out()
    pooled = np.vstack([held_rows, ring_rows])
    labels = np.concatenate(
        [held_out.labels, np.full(len(ring_rows), len(KINDS), dtype=np.int64)]
    )
    nearest = _nearest_by_angle(pooled) if by_angle else _nearest_by_gap(pooled)
    found = labels[nearest[len(held_rows) :]]
    return np.bincount(found, minlength=len(ALL_KINDS)).tolist()


def _seed_study(seed: int) -> SeedResponse:
    held_out = _held_out()
    rings = _rings()
    labels = held_out.labels
    network = _network(seed)
    untrained = _untrained(seed)
    trained_rows = _layer_rows(network.stack, held_out.pictures)
    untrained_rows = _layer_rows(untrained, held_out.pictures)
    pixels = _pixels(held_out)
    centred = pixels - pixels.mean(axis=1, keepdims=True)

    def entry(
        name: str,
        layer: Literal[
            "pixels", "pixels less their mean", "flattened", "hidden", "scores"
        ],
        trained: bool | None,
        measure: Literal["angle", "distance"],
        rows: np.ndarray,
    ) -> Representation:
        nearest = (
            _nearest_by_angle(rows) if measure == "angle" else _nearest_by_gap(rows)
        )
        within, across = _within_and_across(rows, labels)
        return Representation(
            name=name,
            layer=layer,
            trained=trained,
            measure=measure,
            width=int(rows.shape[1]),
            nearest_own_kind=_own_kind_share(nearest, labels),
            within=within,
            across=across,
        )

    representations = [
        entry("pixels, by distance", "pixels", None, "distance", pixels),
        entry("pixels, by angle", "pixels", None, "angle", pixels),
        entry(
            "pixels less their own mean, by angle",
            "pixels less their mean",
            None,
            "angle",
            centred,
        ),
    ]
    for trained, rows in ((True, trained_rows), (False, untrained_rows)):
        state = "trained" if trained else "untrained"
        representations += [
            entry(
                f"flattened 128, {state}",
                "flattened",
                trained,
                "angle",
                rows[FLATTENED],
            ),
            entry(f"hidden 16, {state}", "hidden", trained, "angle", rows[HIDDEN]),
            entry(f"scores 4, {state}", "scores", trained, "angle", rows[SCORES]),
        ]
    representations.append(
        entry(
            "hidden 16, trained, by distance",
            "hidden",
            True,
            "distance",
            trained_rows[HIDDEN],
        )
    )

    ring_trained = _layer_rows(network.stack, rings.pictures)
    ring_untrained = _layer_rows(untrained, rings.pictures)
    ring_pixels = _pixels(rings)
    ring_directions = _unit(ring_trained[HIDDEN])
    held_directions = _unit(trained_rows[HIDDEN])
    not_self = ~np.eye(len(rings), dtype=bool)
    ring_centred = ring_pixels - ring_pixels.mean(axis=1, keepdims=True)

    def gathered(held_rows, ring_rows, by_angle=True) -> int:
        return _ring_counts(held_rows, ring_rows, by_angle)[len(KINDS)]

    ring_study = RingStudy(
        called=np.bincount(
            ring_trained[SCORES].argmax(axis=1), minlength=len(KINDS)
        ).tolist(),
        nearest_by_vector=_ring_counts(
            trained_rows[HIDDEN], ring_trained[HIDDEN], True
        ),
        nearest_by_pixels=_ring_counts(pixels, ring_pixels, False),
        nearest_by_untrained=_ring_counts(
            untrained_rows[HIDDEN], ring_untrained[HIDDEN], True
        ),
        ring_to_ring=float((ring_directions @ ring_directions.T)[not_self].mean()),
        ring_to_kind=[
            float((ring_directions @ held_directions[labels == label].T).mean())
            for label in range(len(KINDS))
        ],
        gathered={
            "pixels, by distance": gathered(pixels, ring_pixels, by_angle=False),
            "pixels less their own mean, by angle": gathered(centred, ring_centred),
            "flattened 128, trained": gathered(
                trained_rows[FLATTENED], ring_trained[FLATTENED]
            ),
            "hidden 16, trained": gathered(trained_rows[HIDDEN], ring_trained[HIDDEN]),
            "scores 4, trained": gathered(trained_rows[SCORES], ring_trained[SCORES]),
            "hidden 16, untrained": gathered(
                untrained_rows[HIDDEN], ring_untrained[HIDDEN]
            ),
        },
    )

    against = None
    if seed != 0:
        reference_rows = _layer_rows(reference_network().stack, held_out.pictures)[
            HIDDEN
        ]
        here = trained_rows[HIDDEN]
        correlations = np.corrcoef(reference_rows.T, here.T)[
            : here.shape[1], here.shape[1] :
        ]
        reference_nearest = _nearest_by_angle(reference_rows)
        here_nearest = _nearest_by_angle(here)
        against = AgainstReference(
            same_picture_cosine=float(
                (_unit(reference_rows) * _unit(here)).sum(1).mean()
            ),
            matching_coordinate_correlation=float(np.abs(np.diag(correlations)).mean()),
            same_nearest_picture=float((reference_nearest == here_nearest).mean()),
            same_nearest_kind=float(
                (labels[reference_nearest] == labels[here_nearest]).mean()
            ),
        )

    return SeedResponse(
        weight_seed=seed,
        held_out_accuracy=accuracy_of(network.stack, held_out),
        untrained_accuracy=accuracy_of(untrained, held_out),
        kinds=list(KINDS),
        representations=representations,
        vector_confusion=_confusion(
            _nearest_by_angle(trained_rows[HIDDEN]), labels, len(KINDS)
        ),
        pixel_confusion=_confusion(_nearest_by_gap(pixels), labels, len(KINDS)),
        rings=ring_study,
        against_reference=against,
    )


@router.get("/seed/0", response_model=SeedResponse)
@lru_cache(maxsize=1)
def seed_zero() -> SeedResponse:
    """Every repeated measurement on the reference network, weight seed 0."""
    return _seed_study(0)


@router.get("/seed/1", response_model=SeedResponse)
@lru_cache(maxsize=1)
def seed_one() -> SeedResponse:
    """The same measurements on the same design retrained from weight seed 1."""
    return _seed_study(1)


@router.get("/seed/2", response_model=SeedResponse)
@lru_cache(maxsize=1)
def seed_two() -> SeedResponse:
    """The same measurements on the same design retrained from weight seed 2."""
    return _seed_study(2)


# ---------------------------------------------------------------------------
# /edges
# ---------------------------------------------------------------------------


class LengthStudy(BaseModel):
    """What the length of a picture's vector is, and what it tracks.

    Every coordinate of the vector is a hyperbolic tangent and so lies strictly
    between minus one and one, which makes ``bound`` the square root of the
    width. ``confidence_correlation`` is the correlation, over the held-out
    half, between a vector's length and the probability the network gives its
    top kind.
    """

    shortest: float
    longest: float
    mean: float
    bound: float
    saturated_share: float
    confidence_correlation: float
    nearest_own_kind_by_angle: float
    nearest_own_kind_by_distance: float


class BlankPicture(BaseModel):
    """A picture holding no shape at all, and where the network puts it.

    ``lit_after_first_convolution`` counts the first convolution's 1024
    numbers that are not zero, and ``lit_after_convolutions`` the flattened
    128. When the first count is zero the picture has left no trace at all,
    and every such picture, whatever its brightness, is given the one vector
    the network's biases produce, which ``cosine_to_all_black`` shows by
    comparing it with the all-black picture's vector.
    """

    name: str
    level: float
    length: float
    called: str
    probabilities: list[float]
    cosine_to_kind: list[float]
    lit_after_first_convolution: int
    lit_after_convolutions: int
    cosine_to_all_black: float


class BrightnessStep(BaseModel):
    factor: float
    cosine_to_original: float
    length: float
    called: str


class BrightnessSweep(BaseModel):
    """One picture multiplied through, background and shape alike."""

    position: int
    kind: str
    steps: list[BrightnessStep]


class EdgesResponse(BaseModel):
    """The cases where a picture's vector stops meaning what the page says."""

    kinds: list[str]
    lengths: LengthStudy
    blanks: list[BlankPicture]
    brightness: list[BrightnessSweep]
    training_lift: list[float]
    training_background: list[float]


BLANK_LEVELS = (
    ("all black", 0.0),
    ("flat at 0.15", 0.15),
    ("flat at 0.3", 0.3),
    ("flat at 0.5", 0.5),
)
"""Pictures with nothing drawn in them, at four flat brightnesses.

0.3 is the brightest background the collection ever draws, so the first three
are backgrounds training could have shown and the last is not.
"""


@router.get("/edges", response_model=EdgesResponse)
@lru_cache(maxsize=1)
def edges() -> EdgesResponse:
    """Blank pictures, brightness outside training, and what length carries."""
    network = reference_network()
    stack = network.stack
    held_out = _held_out()
    labels = held_out.labels
    rows = _layer_rows(stack, held_out.pictures)
    vectors = rows[HIDDEN]
    lengths = np.linalg.norm(vectors, axis=1)
    top = _probabilities(rows[SCORES]).max(axis=1)
    kind_directions = [_unit(vectors[labels == label]) for label in range(len(KINDS))]
    side = held_out.pictures.shape[-1]
    all_black = _unit(_layer_rows(stack, np.zeros((1, 1, side, side)))[HIDDEN])[0]

    blanks = []
    for name, level in BLANK_LEVELS:
        picture = np.full((1, 1, side, side), level)
        blank_rows = _layer_rows(stack, picture)
        vector = blank_rows[HIDDEN][0]
        direction = vector / np.linalg.norm(vector)
        blanks.append(
            BlankPicture(
                name=name,
                level=level,
                length=float(np.linalg.norm(vector)),
                called=KINDS[int(blank_rows[SCORES][0].argmax())],
                probabilities=[
                    float(value) for value in _probabilities(blank_rows[SCORES][0])
                ],
                cosine_to_kind=[
                    float((kind @ direction).mean()) for kind in kind_directions
                ],
                lit_after_first_convolution=int(np.count_nonzero(blank_rows[0])),
                lit_after_convolutions=int(np.count_nonzero(blank_rows[FLATTENED])),
                cosine_to_all_black=float(direction @ all_black),
            )
        )

    sweeps = []
    for label in range(len(KINDS)):
        position = int(np.flatnonzero(labels == label)[0])
        picture = held_out.pictures[position : position + 1]
        original = _unit(_layer_rows(stack, picture)[HIDDEN])[0]
        steps = []
        for factor in BRIGHTNESS_FACTORS:
            scaled = _layer_rows(stack, picture * factor)
            vector = scaled[HIDDEN][0]
            steps.append(
                BrightnessStep(
                    factor=factor,
                    cosine_to_original=float(
                        original @ vector / np.linalg.norm(vector)
                    ),
                    length=float(np.linalg.norm(vector)),
                    called=KINDS[int(scaled[SCORES][0].argmax())],
                )
            )
        sweeps.append(
            BrightnessSweep(position=position, kind=KINDS[label], steps=steps)
        )

    width = vectors.shape[1]
    return EdgesResponse(
        kinds=list(KINDS),
        lengths=LengthStudy(
            shortest=float(lengths.min()),
            longest=float(lengths.max()),
            mean=float(lengths.mean()),
            bound=float(np.sqrt(width)),
            saturated_share=float((np.abs(vectors) > SATURATED).mean()),
            confidence_correlation=float(np.corrcoef(lengths, top)[0, 1]),
            nearest_own_kind_by_angle=_own_kind_share(
                _nearest_by_angle(vectors), labels
            ),
            nearest_own_kind_by_distance=_own_kind_share(
                _nearest_by_gap(vectors), labels
            ),
        ),
        blanks=blanks,
        brightness=sweeps,
        training_lift=[0.4, 0.7],
        training_background=[0.0, 0.3],
    )
