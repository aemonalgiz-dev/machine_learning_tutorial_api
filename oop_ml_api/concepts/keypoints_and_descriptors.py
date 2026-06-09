"""Keypoints and Descriptors.

Every endpoint here is registered under ``/concepts/keypoints-and-descriptors``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about which few places in a picture are worth describing, and what surrounds each.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, so that a reader meeting the workbench scene on one page
recognises it on the next.

Two things are computed here rather than in the library, and both are
deliberate. The first is the *naive* border order, which drops candidates
before the suppression runs instead of after: it is the mistake the library
records having made, and a page that only shows the repaired answer cannot
show what the mistake looked like. The second is describing **every** position
of a picture and comparing all of them against all of them, which is the
alternative this method replaces; the library has no reason to describe a
position nobody chose, so the dense route is written here with numpy and a
test pins that it reproduces the library's own description at a keypoint.
Everything else -- every gradient, every tensor, every score, every keypoint,
every descriptor and every match -- is the library's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.computer_vision.edges import GradientField
from oop_ml.core.computer_vision.keypoints import (
    CornerMeasure,
    CornerResponse,
    DescriptorMatch,
    Descriptors,
    HarrisMeasure,
    Keypoint,
    PatchDescriptor,
    SmallestEigenvalueMeasure,
    StructureTensor,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

from oop_ml_api.vision_scenes import (
    MOTIF_AT,
    MOTIF_SIDE,
    as_rows,
    relit,
    turned,
    workbench,
)

router = APIRouter(
    prefix="/concepts/keypoints-and-descriptors", tags=["keypoints-and-descriptors"]
)

PATCH_SIDE = 5
"""How many pixels on a side of the square a keypoint is described by.

Odd, so the patch has a centre pixel to sit on, and small enough that all
twenty-five of its numbers can be printed for a reader to check.
"""

SQUARE_FRAME = 21
SQUARE_TOP = 6
SQUARE_LEFT = 6
SQUARE_SIDE = 9
"""The bright square on dark ground the page works by hand.

The same fixture the library's own spec uses, chosen because every number it
produces comes out round: the smaller eigenvalue is exactly 4.0 at a corner and
exactly 0.0 along an edge, so a reader can check the arithmetic rather than
take it.
"""

TIGHT_FRAME = 17
TIGHT_TOP = 2
TIGHT_LEFT = 2
"""The same square in a smaller frame, so three of its corners sit two pixels
from the edge. That is what makes the border order visible."""

FLOOR_HEIGHT = 25
FLOOR_WIDTH = 33
FLOOR_GROUND = 0.6
FLOOR_BLOCK = 1.0
FLOOR_SHADOW = 0.2
FLOOR_TOP = 7
FLOOR_SIDE = 11
FLOOR_BLOCK_LEFT = 4
FLOOR_SHADOW_LEFT = 19
"""A lit floor carrying a pale block and, beside it, a dark patch of shadow.

The two rectangles are the same size and sit the same distance from the
ground's brightness, one above it and one below, which is the arrangement that
makes the point exactly rather than approximately: a corner measure reads the
size of the brightness step and nothing else, so the shadow's corners and the
block's corners come out at the same number.
"""

HORIZON_FRAME = 21
HORIZON_ROW = 10
"""A picture that is dark above and bright below, and has no corner anywhere."""

TEXTURE_AMPLITUDE = 0.05
"""How large the ripple laid over the twin squares is.

Small beside the squares' own contrast. Its only job is to break an exact tie
between two identical patches, so that a coincidental match reports a ratio it
computed rather than the convention a zero over a zero falls back on.
"""

RELIGHTING_SCALE = 1.6
"""What the shared scene's relit twin multiplies every pixel by.

Written down here because the page quotes the square of it: a gradient scales
with the lighting, and the structure tensor is built from products of two
gradients, so every score scales by 2.56.
"""

DEFAULT_SEPARATION = 3
DEFAULT_BORDER = 2
DEFAULT_STRENGTH = 0.05
"""The playground's opening settings on the shared scene."""

SceneName = Literal["workbench", "relit", "turned", "square", "floor", "horizon"]
MeasureName = Literal["smallest_eigenvalue", "harris"]


def _square(
    frame: int = SQUARE_FRAME,
    top: int = SQUARE_TOP,
    left: int = SQUARE_LEFT,
    side: int = SQUARE_SIDE,
) -> Picture:
    """A bright square of ``side`` pixels on dark ground."""
    values = np.zeros((frame, frame), dtype=np.float64)
    values[top : top + side, left : left + side] = 1.0
    return Picture(values)


def _twin_squares(frame: int = 31) -> Picture:
    """Two identical squares, which is the repeated structure that defeats this."""
    values = np.zeros((frame, frame), dtype=np.float64)
    values[5:12, 3:10] = 1.0
    values[5:12, 18:25] = 1.0
    return Picture(values)


def _faint_texture(shape: tuple[int, int]) -> FloatArray:
    """A deterministic ripple, small beside the squares' own contrast."""
    rows = np.arange(shape[0])[:, None]
    columns = np.arange(shape[1])[None, :]
    return TEXTURE_AMPLITUDE * np.sin(rows * 1.7) * np.cos(columns * 2.3)


def _floor() -> Picture:
    """The lit floor with a block on it and a shadow beside it."""
    values = np.full((FLOOR_HEIGHT, FLOOR_WIDTH), FLOOR_GROUND, dtype=np.float64)
    values[
        FLOOR_TOP : FLOOR_TOP + FLOOR_SIDE,
        FLOOR_BLOCK_LEFT : FLOOR_BLOCK_LEFT + FLOOR_SIDE,
    ] = FLOOR_BLOCK
    values[
        FLOOR_TOP : FLOOR_TOP + FLOOR_SIDE,
        FLOOR_SHADOW_LEFT : FLOOR_SHADOW_LEFT + FLOOR_SIDE,
    ] = FLOOR_SHADOW
    return Picture(values)


def _horizon() -> Picture:
    """One long straight edge and nothing else."""
    values = np.zeros((HORIZON_FRAME, HORIZON_FRAME), dtype=np.float64)
    values[HORIZON_ROW:, :] = 1.0
    return Picture(values)


SCENES: dict[SceneName, Callable[[], Picture]] = {
    "workbench": workbench,
    "relit": relit,
    "turned": turned,
    "square": _square,
    "floor": _floor,
    "horizon": _horizon,
}
"""Every picture the page can be pointed at, by the name the request uses."""


def _scene(name: SceneName) -> Picture:
    return SCENES[name]()


def _measure(name: MeasureName, sensitivity: float) -> CornerMeasure:
    """The corner measure a request asked for."""
    if name == "harris":
        return HarrisMeasure(sensitivity=sensitivity)
    return SmallestEigenvalueMeasure()


class KeypointReport(BaseModel):
    """One position worth describing, and how strongly it was preferred."""

    row: int
    column: int
    strength: float


def _reported(keypoints: Iterable[Keypoint]) -> list[KeypointReport]:
    """A run of keypoints in the shape the browser draws from."""
    return [
        KeypointReport(
            row=point.row, column=point.column, strength=round(point.strength, 6)
        )
        for point in keypoints
    ]


class DetectRequest(BaseModel):
    """Which picture to look at, how to score it, and what to keep.

    Every field is a choice the method leaves open, which is why they are all
    here rather than fixed: the window decides how large a neighbourhood has to
    disagree with itself, the threshold decides how much disagreement counts,
    the separation decides how far apart two answers must be, and the border
    decides how close to the frame a patch may be cut.
    """

    scene: SceneName = "workbench"
    measure: MeasureName = "smallest_eigenvalue"
    sensitivity: float = Field(default=0.04, gt=0.0, lt=0.25)
    window_side: int = Field(default=3, ge=1, le=11)
    minimum_strength: float = Field(default=DEFAULT_STRENGTH, ge=0.0, le=100.0)
    separation: int = Field(default=DEFAULT_SEPARATION, ge=1, le=20)
    border: int = Field(default=DEFAULT_BORDER, ge=0, le=10)
    limit: int | None = Field(default=None, ge=1, le=500)

    model_config = {"extra": "forbid"}


class DetectResponse(BaseModel):
    """A picture, its corner scores, and the peaks kept from them.

    ``picture`` and ``scores`` are the same shape, so a browser can draw them
    beside each other pixel for pixel. ``n_candidates`` is how many pixels
    scored above the threshold before anything was suppressed, which is the
    number that says how much of the work the suppression is doing.
    """

    height: int
    width: int
    picture: list[list[float]]
    picture_low: float
    picture_high: float
    scores: list[list[float]]
    score_low: float
    score_high: float
    keypoints: list[KeypointReport]
    n_candidates: int
    n_keypoints: int


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    """Score every pixel of one picture and keep the peaks."""
    picture = _scene(request.scene)
    response = CornerResponse.of(
        picture,
        _measure(request.measure, request.sensitivity),
        window_side=request.window_side,
    )
    keypoints = response.peaks(
        minimum_strength=request.minimum_strength,
        separation=request.separation,
        limit=request.limit,
        border=request.border,
    )
    scores = np.asarray(response.scores)
    return DetectResponse(
        height=picture.height,
        width=picture.width,
        picture=as_rows(picture),
        picture_low=round(picture.darkest, 4),
        picture_high=round(picture.brightest, 4),
        scores=as_rows(response.scores),
        score_low=round(response.scores.darkest, 6),
        score_high=round(response.scores.brightest, 6),
        keypoints=_reported(keypoints),
        n_candidates=int(np.sum(scores > request.minimum_strength)),
        n_keypoints=len(keypoints),
    )


class GradientVector(BaseModel):
    """How fast brightness rises to the right, and how fast it rises downward."""

    across: float
    down: float


class ProbeReport(BaseModel):
    """Everything one position of the square has to say about itself.

    ``gradients`` is the three by three window's own gradient vectors, which is
    what the three averaged products are built from, so a reader can see the
    directions before they see the numbers that summarise them.
    """

    name: str
    row: int
    column: int
    gradients: list[list[GradientVector]]
    horizontal_squared: float
    vertical_squared: float
    cross: float
    trace: float
    determinant: float
    smallest_eigenvalue: float
    harris: float


class WindowStep(BaseModel):
    """What one width of averaging window makes of the same square."""

    window_side: int
    corner_score: float
    brightest: float
    n_keypoints: int


class WhyACornerResponse(BaseModel):
    """The square, three positions on it, and what the window width does.

    ``probes`` are a corner, the middle of an edge and flat ground, and the
    order they come in is the order the page reads them.
    ``harris_at_high_sensitivity`` is the same corner scored with the
    subtracted term turned almost to the ceiling above which no picture can
    score positively at all.
    """

    height: int
    width: int
    picture: list[list[float]]
    probes: list[ProbeReport]
    trace_ratio: float
    window_sweep: list[WindowStep]
    harris_sensitivity: float
    harris_at_high_sensitivity: float
    harris_keypoints_at_high_sensitivity: int


HIGH_SENSITIVITY = 0.24
"""Just below the quarter at which the subtracted term swallows everything."""


def _probe(
    tensor: StructureTensor,
    field: GradientField,
    name: str,
    at: tuple[int, int],
) -> ProbeReport:
    """Read one position of the square every way the page needs it read."""
    row, column = at
    smallest = SmallestEigenvalueMeasure()
    harris = HarrisMeasure()
    gradients = [
        [
            GradientVector(
                across=float(field.horizontal.values[row + down, column + across]),
                down=float(field.vertical.values[row + down, column + across]),
            )
            for across in (-1, 0, 1)
        ]
        for down in (-1, 0, 1)
    ]
    return ProbeReport(
        name=name,
        row=row,
        column=column,
        gradients=gradients,
        horizontal_squared=float(tensor.horizontal_squared.values[row, column]),
        vertical_squared=float(tensor.vertical_squared.values[row, column]),
        cross=float(tensor.cross.values[row, column]),
        trace=float(tensor.trace.values[row, column]),
        determinant=float(tensor.determinant.values[row, column]),
        smallest_eigenvalue=float(smallest.score(tensor).values[row, column]),
        harris=float(harris.score(tensor).values[row, column]),
    )


@cache
def _why_a_corner() -> WhyACornerResponse:
    picture = _square()
    tensor = StructureTensor.of(picture)
    field = GradientField.of(picture)
    probes = [
        _probe(tensor, field, "a corner", (SQUARE_TOP, SQUARE_LEFT)),
        _probe(tensor, field, "the middle of an edge", (SQUARE_TOP + 4, SQUARE_LEFT)),
        _probe(tensor, field, "flat ground", (2, 2)),
    ]
    sweep: list[WindowStep] = []
    for window_side in (1, 3, 5, 7, 9):
        response = CornerResponse.of(picture, window_side=window_side)
        sweep.append(
            WindowStep(
                window_side=window_side,
                corner_score=round(response.at(SQUARE_TOP, SQUARE_LEFT), 6),
                brightest=round(response.scores.brightest, 6),
                n_keypoints=len(
                    response.peaks(minimum_strength=0.5, separation=DEFAULT_SEPARATION)
                ),
            )
        )
    strict = CornerResponse.of(picture, HarrisMeasure(sensitivity=HIGH_SENSITIVITY))
    return WhyACornerResponse(
        height=picture.height,
        width=picture.width,
        picture=as_rows(picture),
        probes=probes,
        trace_ratio=round(probes[0].trace / probes[1].trace, 6),
        window_sweep=sweep,
        harris_sensitivity=HIGH_SENSITIVITY,
        harris_at_high_sensitivity=round(strict.at(SQUARE_TOP, SQUARE_LEFT), 6),
        harris_keypoints_at_high_sensitivity=len(strict.peaks()),
    )


@router.get("/why-a-corner", response_model=WhyACornerResponse)
def why_a_corner() -> WhyACornerResponse:
    """A corner, an edge and flat ground, read every way the page needs."""
    return _why_a_corner()


class SlidStep(BaseModel):
    """One position along the square's left edge, and its patch."""

    row: int
    patch: list[list[float]]
    distance_from_edge: float
    distance_from_corner: float


class ApertureResponse(BaseModel):
    """A window slid along one edge, and how far each stop is from two others.

    The two references are the patch centred on the middle of the left edge and
    the patch centred on the top-left corner. ``edge_matches`` and
    ``corner_matches`` count the stops sitting at a distance of exactly zero
    from each, which is the aperture problem as a pair of integers.
    """

    height: int
    width: int
    picture: list[list[float]]
    edge_row: int
    corner_row: int
    column: int
    edge_patch: list[list[float]]
    corner_patch: list[list[float]]
    steps: list[SlidStep]
    edge_matches: int
    corner_matches: int


APERTURE_ROWS = range(3, 18)
"""Every position the window is slid to, from above the square to below it."""


@cache
def _aperture() -> ApertureResponse:
    picture = _square()
    column = SQUARE_LEFT
    edge_row = SQUARE_TOP + 4
    edge_reference = PatchDescriptor.of(
        picture, Keypoint(edge_row, column, 0.0), PATCH_SIDE
    )
    corner_reference = PatchDescriptor.of(
        picture, Keypoint(SQUARE_TOP, column, 0.0), PATCH_SIDE
    )
    half = PATCH_SIDE // 2
    steps: list[SlidStep] = []
    for row in APERTURE_ROWS:
        here = PatchDescriptor.of(picture, Keypoint(row, column, 0.0), PATCH_SIDE)
        steps.append(
            SlidStep(
                row=row,
                patch=as_rows(
                    picture.patch_at(row - half, column - half, PATCH_SIDE, PATCH_SIDE)
                ),
                distance_from_edge=round(edge_reference.distance_to(here), 6),
                distance_from_corner=round(corner_reference.distance_to(here), 6),
            )
        )
    return ApertureResponse(
        height=picture.height,
        width=picture.width,
        picture=as_rows(picture),
        edge_row=edge_row,
        corner_row=SQUARE_TOP,
        column=column,
        edge_patch=as_rows(
            picture.patch_at(edge_row - half, column - half, PATCH_SIDE, PATCH_SIDE)
        ),
        corner_patch=as_rows(
            picture.patch_at(SQUARE_TOP - half, column - half, PATCH_SIDE, PATCH_SIDE)
        ),
        steps=steps,
        edge_matches=sum(
            1
            for row in APERTURE_ROWS
            if edge_reference.distance_to(
                PatchDescriptor.of(picture, Keypoint(row, column, 0.0), PATCH_SIDE)
            )
            == 0.0
        ),
        corner_matches=sum(
            1
            for row in APERTURE_ROWS
            if corner_reference.distance_to(
                PatchDescriptor.of(picture, Keypoint(row, column, 0.0), PATCH_SIDE)
            )
            == 0.0
        ),
    )


@router.get("/aperture", response_model=ApertureResponse)
def aperture() -> ApertureResponse:
    """A five-wide window slid along one straight edge, stop by stop."""
    return _aperture()


def _naive_border_peaks(
    response: CornerResponse, minimum_strength: float, separation: int, border: int
) -> list[KeypointReport]:
    """The border applied to the candidates instead of to what is reported.

    Written here rather than taken from the library, because the library does
    it the other way round on purpose and this is what the other way round
    answers. Dropping a candidate before the suppression runs means the corner
    inside the border never suppresses anything, so the pixel on its shoulder
    survives and is reported in its place: the right number of answers, plausible
    strengths, and every position wrong by a pixel.
    """
    scores = np.asarray(response.scores)
    height, width = scores.shape
    rows, columns = np.nonzero(scores > minimum_strength)
    inside = (
        (rows >= border)
        & (rows < height - border)
        & (columns >= border)
        & (columns < width - border)
    )
    rows, columns = rows[inside], columns[inside]
    if rows.size == 0:
        return []
    strengths = scores[rows, columns]
    order = np.lexsort((columns, rows, -strengths))
    taken: list[Keypoint] = []
    for position in order:
        candidate = Keypoint(
            int(rows[position]), int(columns[position]), float(strengths[position])
        )
        if all(candidate.is_further_than(separation, kept) for kept in taken):
            taken.append(candidate)
    return _reported(taken)


class SeparationStep(BaseModel):
    """How many keypoints survive one suppression radius."""

    separation: int
    n_keypoints: int


class SuppressionResponse(BaseModel):
    """What the suppression and the border each do to the same raw response.

    ``square_sweep`` and ``scene_sweep`` are the same question asked of the
    hand-worked square and of the shared scene. ``in_order`` and ``out_of_order``
    are the two ways of applying the border to the tight square, which differ in
    every position and in nothing else a caller could notice.
    """

    height: int
    width: int
    picture: list[list[float]]
    scores: list[list[float]]
    score_high: float
    square_candidates: int
    square_sweep: list[SeparationStep]
    scene_candidates: int
    scene_sweep: list[SeparationStep]
    border: int
    without_border: list[KeypointReport]
    in_order: list[KeypointReport]
    out_of_order: list[KeypointReport]
    shoulder_score: float
    corner_score: float


BORDER_FOR_THE_TIGHT_SQUARE = 3
"""Wide enough to leave out three of the four corners of the tight square."""


@cache
def _suppression() -> SuppressionResponse:
    square = _square()
    square_response = CornerResponse.of(square)
    scene_response = CornerResponse.of(workbench())
    tight = _square(frame=TIGHT_FRAME, top=TIGHT_TOP, left=TIGHT_LEFT)
    tight_response = CornerResponse.of(tight)
    return SuppressionResponse(
        height=tight.height,
        width=tight.width,
        picture=as_rows(tight),
        scores=as_rows(tight_response.scores),
        score_high=round(tight_response.scores.brightest, 6),
        square_candidates=int(np.sum(np.asarray(square_response.scores) > 0.5)),
        square_sweep=[
            SeparationStep(
                separation=separation,
                n_keypoints=len(
                    square_response.peaks(minimum_strength=0.5, separation=separation)
                ),
            )
            for separation in (1, 2, 3, 5, 7)
        ],
        scene_candidates=int(
            np.sum(np.asarray(scene_response.scores) > DEFAULT_STRENGTH)
        ),
        scene_sweep=[
            SeparationStep(
                separation=separation,
                n_keypoints=len(
                    scene_response.peaks(
                        minimum_strength=DEFAULT_STRENGTH,
                        separation=separation,
                        border=DEFAULT_BORDER,
                    )
                ),
            )
            for separation in (1, 2, 3, 5, 7)
        ],
        border=BORDER_FOR_THE_TIGHT_SQUARE,
        without_border=_reported(tight_response.peaks(minimum_strength=0.5)),
        in_order=_reported(
            tight_response.peaks(
                minimum_strength=0.5, border=BORDER_FOR_THE_TIGHT_SQUARE
            )
        ),
        out_of_order=_naive_border_peaks(
            tight_response, 0.5, DEFAULT_SEPARATION, BORDER_FOR_THE_TIGHT_SQUARE
        ),
        shoulder_score=round(tight_response.at(3, 3), 6),
        corner_score=round(tight_response.at(TIGHT_TOP, TIGHT_LEFT), 6),
    )


@router.get("/suppression", response_model=SuppressionResponse)
def suppression() -> SuppressionResponse:
    """How many answers survive each radius, and what the border order costs."""
    return _suppression()


class NamedDistance(BaseModel):
    """One comparison the page quotes, with the number it came out at."""

    name: str
    distance: float


class DescriptorResponse(BaseModel):
    """One corner's patch taken all the way to a description, and compared.

    ``raw`` is the twenty-five brightnesses as they were read, ``centred`` the
    same with the patch's own mean taken out, and ``unit`` that divided by its
    own length. ``comparisons`` are the distances the page quotes, each on a
    scale running from zero for an identical description to two for its exact
    negative.
    """

    height: int
    width: int
    picture: list[list[float]]
    row: int
    column: int
    side: int
    raw: list[list[float]]
    centred: list[list[float]]
    unit: list[list[float]]
    patch_mean: float
    centred_length: float
    comparisons: list[NamedDistance]
    shift_rows: int
    shift_columns: int
    shifted_strengths_agree: bool
    shifted_positions_agree: bool
    turned_gaps: list[float]


@cache
def _descriptor() -> DescriptorResponse:
    picture = _square()
    corner = Keypoint(SQUARE_TOP, SQUARE_LEFT, 4.0)
    half = PATCH_SIDE // 2
    patch = np.asarray(
        picture.patch_at(
            corner.row - half, corner.column - half, PATCH_SIDE, PATCH_SIDE
        )
    )
    centred = patch - patch.mean()
    length = float(np.linalg.norm(centred))
    here = PatchDescriptor.of(picture, corner, PATCH_SIDE)

    brighter = PatchDescriptor.of(Picture(picture.values + 10.0), corner, PATCH_SIDE)
    harder = PatchDescriptor.of(Picture(picture.values * 3.0), corner, PATCH_SIDE)
    negative = PatchDescriptor.of(Picture(-picture.values), corner, PATCH_SIDE)
    top_right = PatchDescriptor.of(
        picture, Keypoint(SQUARE_TOP, SQUARE_LEFT + SQUARE_SIDE - 1, 0.0), PATCH_SIDE
    )
    bottom_right = PatchDescriptor.of(
        picture,
        Keypoint(SQUARE_TOP + SQUARE_SIDE - 1, SQUARE_LEFT + SQUARE_SIDE - 1, 0.0),
        PATCH_SIDE,
    )
    outside = PatchDescriptor.of(picture, Keypoint(2, 2, 0.0), PATCH_SIDE)
    inside = PatchDescriptor.of(picture, Keypoint(10, 10, 0.0), PATCH_SIDE)

    shift_rows, shift_columns = 2, 3
    shifted = _square(top=SQUARE_TOP + shift_rows, left=SQUARE_LEFT + shift_columns)
    found = CornerResponse.of(picture).peaks(minimum_strength=0.5)
    found_shifted = CornerResponse.of(shifted).peaks(minimum_strength=0.5)
    shifted_descriptor = PatchDescriptor.of(
        shifted,
        Keypoint(corner.row + shift_rows, corner.column + shift_columns, 4.0),
        PATCH_SIDE,
    )

    spun = Picture(np.rot90(picture.values))
    turned_gaps = [
        round(
            PatchDescriptor.of(picture, point, PATCH_SIDE).distance_to(
                PatchDescriptor.of(
                    spun,
                    Keypoint(
                        picture.height - 1 - point.column, point.row, point.strength
                    ),
                    PATCH_SIDE,
                )
            ),
            6,
        )
        for point in found
    ]

    comparisons = [
        NamedDistance(
            name="the same corner, ten added to every pixel",
            distance=here.distance_to(brighter),
        ),
        NamedDistance(
            name="the same corner, every pixel multiplied by three",
            distance=here.distance_to(harder),
        ),
        NamedDistance(
            name="the same corner, the picture shifted down two and right three",
            distance=here.distance_to(shifted_descriptor),
        ),
        NamedDistance(
            name="the top-right corner of the same square",
            distance=round(here.distance_to(top_right), 6),
        ),
        NamedDistance(
            name="the bottom-right corner of the same square",
            distance=round(here.distance_to(bottom_right), 6),
        ),
        NamedDistance(
            name="flat ground outside the square, against flat ground inside it",
            distance=outside.distance_to(inside),
        ),
        NamedDistance(
            name="the same corner in the picture with its brightness reversed",
            distance=round(here.distance_to(negative), 6),
        ),
    ]
    return DescriptorResponse(
        height=picture.height,
        width=picture.width,
        picture=as_rows(picture),
        row=corner.row,
        column=corner.column,
        side=PATCH_SIDE,
        raw=[[round(float(value), 4) for value in row] for row in patch],
        centred=[[round(float(value), 4) for value in row] for row in centred],
        unit=[
            [round(float(value), 6) for value in row]
            for row in np.asarray(here).reshape(PATCH_SIDE, PATCH_SIDE)
        ],
        patch_mean=round(float(patch.mean()), 6),
        centred_length=round(length, 6),
        comparisons=comparisons,
        shift_rows=shift_rows,
        shift_columns=shift_columns,
        shifted_strengths_agree=[point.strength for point in found]
        == [point.strength for point in found_shifted],
        shifted_positions_agree=[
            (point.row + shift_rows, point.column + shift_columns) for point in found
        ]
        == [(point.row, point.column) for point in found_shifted],
        turned_gaps=turned_gaps,
    )


@router.get("/descriptor", response_model=DescriptorResponse)
def descriptor() -> DescriptorResponse:
    """One corner's patch, centred, normalised and compared against six others."""
    return _descriptor()


class MatchReport(BaseModel):
    """One description's nearest neighbour, and what it beat."""

    from_row: int
    from_column: int
    to_row: int
    to_column: int
    distance: float
    runner_up_distance: float
    ratio: float
    kept: bool


class MatchScene(BaseModel):
    """Two pictures, their keypoints, and every match between them."""

    name: str
    label: str
    height: int
    width: int
    first_picture: list[list[float]]
    second_picture: list[list[float]]
    first_keypoints: list[KeypointReport]
    second_keypoints: list[KeypointReport]
    matches: list[MatchReport]
    n_kept: int


class MatchingResponse(BaseModel):
    """The three arrangements the ratio test is asked about.

    A genuine repeat of one scene, the same scene holding two identical
    squares, and that second scene with a faint ripple laid over it so the two
    equally good answers stop being exactly equal.
    """

    maximum_ratio: float
    scenes: list[MatchScene]


LOWE_RATIO = 0.8
"""The threshold Lowe published, and the one the page holds every match to."""


def _matched(
    name: str, label: str, first: Picture, second: Picture, border: int = 3
) -> MatchScene:
    """Detect, describe and match one pair of pictures."""
    first_keypoints = CornerResponse.of(first).peaks(
        minimum_strength=0.5, border=border
    )
    second_keypoints = CornerResponse.of(second).peaks(
        minimum_strength=0.5, border=border
    )
    asked = Descriptors.of(first, first_keypoints, PATCH_SIDE)
    searched = Descriptors.of(second, second_keypoints, PATCH_SIDE)
    everything = asked.matched_to(searched, maximum_ratio=1.0)
    matches = [
        MatchReport(
            from_row=match.first.keypoint.row,
            from_column=match.first.keypoint.column,
            to_row=match.second.keypoint.row,
            to_column=match.second.keypoint.column,
            distance=round(match.distance, 6),
            runner_up_distance=round(match.runner_up_distance, 6),
            ratio=round(match.ratio, 6),
            kept=match.ratio <= LOWE_RATIO,
        )
        for match in everything
    ]
    return MatchScene(
        name=name,
        label=label,
        height=first.height,
        width=first.width,
        first_picture=as_rows(first),
        second_picture=as_rows(second),
        first_keypoints=_reported(first_keypoints),
        second_keypoints=_reported(second_keypoints),
        matches=matches,
        n_kept=len(asked.matched_to(searched, maximum_ratio=LOWE_RATIO)),
    )


@cache
def _matching() -> MatchingResponse:
    moved_rows, moved_columns = 4, 5
    twins = _twin_squares()
    textured = Picture(twins.values + _faint_texture(twins.shape))
    return MatchingResponse(
        maximum_ratio=LOWE_RATIO,
        scenes=[
            _matched(
                "one square, photographed twice",
                f"the same square, moved down {moved_rows} and right {moved_columns}",
                _square(frame=31),
                _square(
                    frame=31,
                    top=SQUARE_TOP + moved_rows,
                    left=SQUARE_LEFT + moved_columns,
                ),
            ),
            _matched(
                "two identical squares",
                "one square searched for in a picture holding two of it",
                _square(frame=31, top=5, left=3, side=7),
                twins,
            ),
            _matched(
                "two identical squares, with a ripple",
                "the same, with a faint texture laid over the search picture",
                _square(frame=31, top=5, left=3, side=7),
                textured,
            ),
        ],
    )


@router.get("/matching", response_model=MatchingResponse)
def matching() -> MatchingResponse:
    """A genuine match, a coincidence, and the same coincidence with the tie broken."""
    return _matching()


def _on_a_motif(keypoint: Keypoint) -> bool:
    """Whether a place sits on one of the scene's three identical crosses.

    The crosses are the shared scene's repeated structure, so they are what a
    ratio test ought to refuse, and counting the refusals that land on them is
    how the page checks that claim rather than asserting it.
    """
    return any(
        row <= keypoint.row < row + MOTIF_SIDE
        and column <= keypoint.column < column + MOTIF_SIDE
        for row, column in MOTIF_AT
    )


def _dense_descriptions(picture: Picture) -> FloatArray:
    """Every position of ``picture`` described, as one array of unit rows.

    The alternative this method replaces, written with numpy because the
    library has no reason to describe a position nobody chose. The arithmetic
    is the same one a description is defined by, and a test pins that a row of
    this agrees with the library's own description at a keypoint to the last
    bit.
    """
    patches = np.lib.stride_tricks.sliding_window_view(
        picture.values, (PATCH_SIDE, PATCH_SIDE)
    ).reshape(-1, PATCH_SIDE * PATCH_SIDE)
    centred = patches - patches.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centred, axis=1, keepdims=True)
    return np.divide(centred, lengths, out=np.zeros_like(centred), where=lengths > 0.0)


class StrayMatch(BaseModel):
    """A match the ratio test kept and should not have.

    The distances are carried at full precision rather than rounded, because
    the whole of what went wrong is that both of them are rounding.
    """

    from_row: int
    from_column: int
    to_row: int
    to_column: int
    distance: float
    runner_up_distance: float
    ratio: float


class SceneReportResponse(BaseModel):
    """What the shared scene does under relighting, under a quarter turn, and
    what describing all of it would have cost.

    ``score_ratio`` is what every score is multiplied by when the lighting is
    turned up, ``lighting_scale`` squared. ``turned_positions_agree`` is
    whether every keypoint of the turned picture is where turning the original
    picture's keypoints says it should be. ``distinct_descriptions`` counts how
    many of the positions describable at all have a description no other
    position shares.
    """

    n_keypoints: int
    minimum_strength: float
    lighting_scale: float
    score_ratio: float
    relit_score_low: float
    relit_score_high: float
    relit_descriptor_gap: float
    turned_positions_agree: bool
    n_turned_keypoints: int
    turned_gap_low: float
    turned_gap_high: float
    n_matches: int
    n_kept: int
    n_refused: int
    n_refused_on_the_repeated_motif: int
    n_on_the_same_position: int
    strays: list[StrayMatch]
    refused_ratio_low: float
    n_positions: int
    distinct_descriptions: int
    n_sharing_a_description: int
    largest_identical_group: int
    dense_numbers: int
    sparse_numbers: int
    dense_pairs: int
    sparse_pairs: int
    dense_seconds: float
    sparse_seconds: float


@cache
def _scene_report() -> SceneReportResponse:
    scene = workbench()
    lit = relit()
    spun = turned()
    response = CornerResponse.of(scene)
    lit_response = CornerResponse.of(lit)
    scaled = RELIGHTING_SCALE * RELIGHTING_SCALE

    keypoints = response.peaks(
        minimum_strength=DEFAULT_STRENGTH,
        separation=DEFAULT_SEPARATION,
        border=DEFAULT_BORDER,
    )
    lit_keypoints = lit_response.peaks(
        minimum_strength=DEFAULT_STRENGTH * scaled,
        separation=DEFAULT_SEPARATION,
        border=DEFAULT_BORDER,
    )
    spun_keypoints = CornerResponse.of(spun).peaks(
        minimum_strength=DEFAULT_STRENGTH,
        separation=DEFAULT_SEPARATION,
        border=DEFAULT_BORDER,
    )

    first = next(iter(keypoints))
    score_ratio = lit_response.at(first.row, first.column) / response.at(
        first.row, first.column
    )
    relit_gap = max(
        PatchDescriptor.of(scene, point, PATCH_SIDE).distance_to(
            PatchDescriptor.of(lit, point, PATCH_SIDE)
        )
        for point in keypoints
    )
    turned_gaps = [
        PatchDescriptor.of(scene, point, PATCH_SIDE).distance_to(
            PatchDescriptor.of(
                spun,
                Keypoint(scene.height - 1 - point.column, point.row, point.strength),
                PATCH_SIDE,
            )
        )
        for point in keypoints
    ]

    asked = Descriptors.of(scene, keypoints, PATCH_SIDE)
    searched = Descriptors.of(lit, lit_keypoints, PATCH_SIDE)
    everything = asked.matched_to(searched, maximum_ratio=1.0)
    kept = [match for match in everything if match.ratio <= LOWE_RATIO]
    refused = [match for match in everything if match.ratio > LOWE_RATIO]

    def landed_where_it_started(match: DescriptorMatch) -> bool:
        return (match.second.keypoint.row, match.second.keypoint.column) == (
            match.first.keypoint.row,
            match.first.keypoint.column,
        )

    on_the_same_position = [match for match in kept if landed_where_it_started(match)]
    strays = [
        StrayMatch(
            from_row=match.first.keypoint.row,
            from_column=match.first.keypoint.column,
            to_row=match.second.keypoint.row,
            to_column=match.second.keypoint.column,
            distance=match.distance,
            runner_up_distance=match.runner_up_distance,
            ratio=match.ratio,
        )
        for match in kept
        if not landed_where_it_started(match)
    ]

    dense = _dense_descriptions(scene)
    dense_lit = _dense_descriptions(lit)
    _, counts = np.unique(dense, axis=0, return_counts=True)
    n_positions = int(dense.shape[0])

    dense_start = time.perf_counter()
    np.linalg.norm(dense[:, None, :] - dense_lit[None, :, :], axis=2)
    dense_seconds = time.perf_counter() - dense_start

    sparse_start = time.perf_counter()
    Descriptors.of(scene, keypoints, PATCH_SIDE).matched_to(
        Descriptors.of(lit, lit_keypoints, PATCH_SIDE)
    )
    sparse_seconds = time.perf_counter() - sparse_start

    return SceneReportResponse(
        n_keypoints=len(keypoints),
        minimum_strength=DEFAULT_STRENGTH,
        lighting_scale=RELIGHTING_SCALE,
        score_ratio=round(score_ratio, 6),
        relit_score_low=round(lit_response.scores.darkest, 6),
        relit_score_high=round(lit_response.scores.brightest, 6),
        relit_descriptor_gap=relit_gap,
        turned_positions_agree=sorted(
            (scene.height - 1 - point.column, point.row) for point in keypoints
        )
        == sorted((point.row, point.column) for point in spun_keypoints),
        n_turned_keypoints=len(spun_keypoints),
        turned_gap_low=round(min(turned_gaps), 6),
        turned_gap_high=round(max(turned_gaps), 6),
        n_matches=len(everything),
        n_kept=len(kept),
        n_refused=len(refused),
        n_refused_on_the_repeated_motif=sum(
            1 for match in refused if _on_a_motif(match.first.keypoint)
        ),
        n_on_the_same_position=len(on_the_same_position),
        strays=strays,
        refused_ratio_low=round(min(match.ratio for match in refused), 6),
        n_positions=n_positions,
        distinct_descriptions=int(counts.size),
        n_sharing_a_description=int(counts[counts > 1].sum()),
        largest_identical_group=int(counts.max()),
        dense_numbers=n_positions * PATCH_SIDE * PATCH_SIDE,
        sparse_numbers=len(keypoints) * PATCH_SIDE * PATCH_SIDE,
        dense_pairs=n_positions * n_positions,
        sparse_pairs=len(keypoints) * len(lit_keypoints),
        dense_seconds=round(dense_seconds, 4),
        sparse_seconds=round(sparse_seconds, 5),
    )


@router.get("/scene-report", response_model=SceneReportResponse)
def scene_report() -> SceneReportResponse:
    """The shared scene relit, turned, matched against itself, and costed."""
    return _scene_report()


class WithoutCornersResponse(BaseModel):
    """The one picture this method has nothing to say about, and what does.

    Two photographs of a horizon, the second taken with the camera three rows
    lower. There is no corner anywhere in either, so the whole method answers
    nothing; describing every position instead still recovers the move exactly,
    for every position whose window straddles the edge.
    """

    height: int
    width: int
    first_picture: list[list[float]]
    second_picture: list[list[float]]
    shift_rows: int
    score_high: float
    n_keypoints: int
    n_positions: int
    n_flat_positions: int
    n_informative_positions: int
    n_recovering_the_shift: int
    furthest_true_partner: float


HORIZON_SHIFT = 3
"""How far the camera moved between the two photographs of the horizon."""


@cache
def _without_corners() -> WithoutCornersResponse:
    first = _horizon()
    second_values = np.zeros((HORIZON_FRAME, HORIZON_FRAME), dtype=np.float64)
    second_values[HORIZON_ROW + HORIZON_SHIFT :, :] = 1.0
    second = Picture(second_values)

    response = CornerResponse.of(first)
    keypoints = response.peaks(minimum_strength=0.0)

    here = _dense_descriptions(first)
    there = _dense_descriptions(second)
    across = first.width - PATCH_SIDE + 1
    distances = np.linalg.norm(here[:, None, :] - there[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    moved = nearest // across - np.arange(here.shape[0]) // across
    informative = np.any(here, axis=1)

    return WithoutCornersResponse(
        height=first.height,
        width=first.width,
        first_picture=as_rows(first),
        second_picture=as_rows(second),
        shift_rows=HORIZON_SHIFT,
        score_high=round(response.scores.brightest, 6),
        n_keypoints=len(keypoints),
        n_positions=int(here.shape[0]),
        n_flat_positions=int(np.sum(~informative)),
        n_informative_positions=int(np.sum(informative)),
        n_recovering_the_shift=int(np.sum(moved[informative] == HORIZON_SHIFT)),
        furthest_true_partner=float(distances[informative].min(axis=1).max()),
    )


@router.get("/without-corners", response_model=WithoutCornersResponse)
def without_corners() -> WithoutCornersResponse:
    """A horizon photographed twice, where this method has nothing to offer."""
    return _without_corners()
