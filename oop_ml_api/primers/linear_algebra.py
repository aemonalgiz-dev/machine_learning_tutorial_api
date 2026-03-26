"""Vectors, matrices and eigen directions, for the linear algebra primer.

Not part of the ``oop_ml`` library. This exists only to draw the linear algebra
primer's pictures. One endpoint measures a pair of vectors, their lengths, the
distance between their tips and their dot product, for the widget that builds
the vocabulary. The other applies a 2x2 matrix to the plane, handing back the
images of a small grid, the unit square and the two basis arrows, together with
the real eigen directions when the matrix has them, for the widget that shows a
matrix as a transformation.

Everything is exact small-matrix arithmetic, so the numbers a reader works by
hand in the prose are the numbers these endpoints return.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/primers/linear-algebra", tags=["linear-algebra-primer"])

MAX_COORDINATE = 10.0
"""How large any vector coordinate or matrix entry may be, in absolute value."""

GRID_EXTENT = 3
"""The grid runs from minus this to this in both directions before transforming."""

PARALLEL_TOLERANCE = 1e-9
"""Two eigen directions closer than this in cross product are one direction."""


class PlanePoint(BaseModel):
    """A point or arrow tip in the plane."""

    x: float
    y: float


class BoundedPoint(BaseModel):
    """A point a request supplies, held to the teaching range.

    Separate from :class:`PlanePoint` because responses legitimately leave the
    range, a stretch pushes grid corners well past it, while inputs must not.
    """

    x: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    y: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)


class VectorPairRequest(BaseModel):
    """Two vectors drawn from the origin."""

    first: BoundedPoint
    second: BoundedPoint

    model_config = {"extra": "forbid"}


class VectorPairResponse(BaseModel):
    """Everything the vector widget reads off a pair of arrows.

    ``cosine`` is the dot product with both lengths divided out, the pure
    measure of alignment, and is ``None`` when either vector has no length to
    divide by.
    """

    length_first: float
    length_second: float
    distance: float
    dot: float
    cosine: float | None


@router.post("/vectors", response_model=VectorPairResponse)
def vectors(request: VectorPairRequest) -> VectorPairResponse:
    """Measure a pair of vectors the way the primer's prose does by hand."""
    first = np.array([request.first.x, request.first.y], dtype=np.float64)
    second = np.array([request.second.x, request.second.y], dtype=np.float64)

    length_first = float(np.hypot(first[0], first[1]))
    length_second = float(np.hypot(second[0], second[1]))
    difference = second - first
    distance = float(np.hypot(difference[0], difference[1]))
    dot = float(first @ second)

    cosine: float | None
    if length_first == 0.0 or length_second == 0.0:
        cosine = None
    else:
        cosine = dot / (length_first * length_second)

    return VectorPairResponse(
        length_first=length_first,
        length_second=length_second,
        distance=distance,
        dot=dot,
        cosine=cosine,
    )


class MatrixRequest(BaseModel):
    """A 2x2 matrix, written [[a, b], [c, d]], applied as (x, y) -> (ax+by, cx+dy)."""

    a: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    b: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    c: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    d: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)

    model_config = {"extra": "forbid"}


class Segment(BaseModel):
    """A straight piece of the transformed grid, given by its two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float


class EigenDirection(BaseModel):
    """One direction the matrix only stretches, unit length, and its factor."""

    value: float
    x: float
    y: float


class TransformResponse(BaseModel):
    """The transformed plane, ready to draw.

    A linear map sends straight lines to straight lines, so each grid line is
    fully described by where its two endpoints land. The basis images are the
    matrix's columns, which is the fact the primer teaches. ``eigen`` is empty
    when the matrix turns every direction, as a rotation does.
    """

    grid_lines: list[Segment]
    unit_square: list[PlanePoint]
    basis_first: PlanePoint
    basis_second: PlanePoint
    eigen: list[EigenDirection]
    has_real_eigen: bool


@router.post("/transform", response_model=TransformResponse)
def transform(request: MatrixRequest) -> TransformResponse:
    """Apply the matrix to the grid, the unit square, the basis and itself."""
    matrix = np.array(
        [[request.a, request.b], [request.c, request.d]], dtype=np.float64
    )

    grid_lines = [
        _transformed_segment(matrix, start, end) for start, end in _grid_segments()
    ]

    square_corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    unit_square = [_to_point(matrix @ corner) for corner in square_corners]

    eigen = _real_eigen_directions(matrix)

    return TransformResponse(
        grid_lines=grid_lines,
        unit_square=unit_square,
        basis_first=_to_point(matrix @ np.array([1.0, 0.0])),
        basis_second=_to_point(matrix @ np.array([0.0, 1.0])),
        eigen=eigen,
        has_real_eigen=bool(eigen),
    )


class ApplyRequest(MatrixRequest):
    """A matrix and one arrow to run through it."""

    vector: BoundedPoint


class ApplyResponse(BaseModel):
    """One arrow in, one arrow out, and how the two compare.

    ``aligned`` is whether the image points along the same line as the input,
    which is what being an eigen direction means, and ``factor`` is then the
    stretch, negative when the image points the opposite way. ``angle_degrees``
    is the turn between input and image, ``None`` when either has no length to
    measure a turn against. The eigen directions ride along so the widget can
    draw them without a second request.
    """

    image: PlanePoint
    aligned: bool
    factor: float | None
    angle_degrees: float | None
    eigen: list[EigenDirection]
    has_real_eigen: bool


@router.post("/apply", response_model=ApplyResponse)
def apply(request: ApplyRequest) -> ApplyResponse:
    """Send one arrow through the matrix and report whether it only stretched."""
    matrix = np.array(
        [[request.a, request.b], [request.c, request.d]], dtype=np.float64
    )
    vector = np.array([request.vector.x, request.vector.y], dtype=np.float64)
    image = matrix @ vector

    vector_length = float(np.hypot(vector[0], vector[1]))
    image_length = float(np.hypot(image[0], image[1]))

    aligned = False
    factor: float | None = None
    angle_degrees: float | None = None

    if vector_length > 0.0 and image_length == 0.0:
        # The matrix flattened this arrow to nothing, which is the stretch
        # factor zero, an eigen direction of a singular matrix.
        aligned = True
        factor = 0.0
    elif vector_length > 0.0:
        cross = float(vector[0] * image[1] - vector[1] * image[0])
        dot = float(vector @ image)
        cosine = np.clip(dot / (vector_length * image_length), -1.0, 1.0)
        angle_degrees = float(np.degrees(np.arccos(cosine)))
        if abs(cross) <= 1e-9 * max(1.0, vector_length * image_length):
            aligned = True
            factor = dot / (vector_length * vector_length)

    eigen = _real_eigen_directions(matrix)

    return ApplyResponse(
        image=_to_point(image),
        aligned=aligned,
        factor=factor,
        angle_degrees=angle_degrees,
        eigen=eigen,
        has_real_eigen=bool(eigen),
    )


def _grid_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """The untransformed grid, one segment per grid line."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(-GRID_EXTENT, GRID_EXTENT + 1):
        segments.append(
            (
                np.array([float(k), float(-GRID_EXTENT)]),
                np.array([float(k), float(GRID_EXTENT)]),
            )
        )
        segments.append(
            (
                np.array([float(-GRID_EXTENT), float(k)]),
                np.array([float(GRID_EXTENT), float(k)]),
            )
        )
    return segments


def _transformed_segment(
    matrix: np.ndarray, start: np.ndarray, end: np.ndarray
) -> Segment:
    moved_start = matrix @ start
    moved_end = matrix @ end
    return Segment(
        x1=float(moved_start[0]),
        y1=float(moved_start[1]),
        x2=float(moved_end[0]),
        y2=float(moved_end[1]),
    )


def _to_point(values: np.ndarray) -> PlanePoint:
    return PlanePoint(x=float(values[0]), y=float(values[1]))


def _real_eigen_directions(matrix: np.ndarray) -> list[EigenDirection]:
    """The real eigen directions, unit length, largest factor first.

    A rotation has none, and the empty list says so. A shear's two reported
    directions are the same line twice, so parallel duplicates collapse to one.
    Signs are made canonical, first nonzero component positive, so the same
    direction always reads the same way.
    """
    values, directions = np.linalg.eig(matrix)

    if np.any(np.abs(values.imag) > 1e-9):
        return []

    real_values = values.real
    real_directions = directions.real

    kept: list[EigenDirection] = []
    for index in np.argsort(-np.abs(real_values)):
        direction = real_directions[:, index]
        direction = direction / np.hypot(direction[0], direction[1])
        if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
            direction = -direction

        cross_with_kept = (
            abs(direction[0] * kept_direction.y - direction[1] * kept_direction.x)
            for kept_direction in kept
        )
        if any(cross < PARALLEL_TOLERANCE for cross in cross_with_kept):
            continue

        kept.append(
            EigenDirection(
                value=float(real_values[index]),
                x=float(direction[0]),
                y=float(direction[1]),
            )
        )

    return kept
