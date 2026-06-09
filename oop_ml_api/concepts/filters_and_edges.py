"""Filters and Edges.

Every endpoint here is registered under ``/concepts/filters-and-edges``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about the sweep, the gradient operators, and why an edge survives a change in lighting.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, so that a reader meeting the workbench scene on one page
recognises it on the next.

What each endpoint is for
-------------------------
``/scene`` is the page's opening claim, measured. It hands back the workbench,
the same scene without its brightness ramp, and the same scene under the shared
relighting, and it reports what a single brightness threshold does to each. A
threshold anywhere between 0.42 and 0.8119 labels all 2304 pixels of the
workbench correctly; the same threshold on the relit scene disagrees with itself
about 877 of them. Beside that it reports what the relighting did to the
gradient, which is to multiply every magnitude by the same 1.6 and to leave
every direction where it was.

``/field`` sweeps one operator across one of those scenes under one border rule
and answers the horizontal rate, the vertical rate, the magnitude and the
direction, which is everything the playground draws.

``/hand-sweep`` is the worked instance. The shared six by six picture is swept
with the reader's choice of operator and border rule, and every answer cell
comes back with the window that produced it, the values that window read and the
products that were summed, so the page can lay the arithmetic out without doing
any of it. It also takes a flag that reverses the weights, which is the whole
difference between correlation and convolution and shows up as a sign.

``/borders`` runs all four border rules over that same picture, with a flat
average and with the usual operator, and says which rules answer a sharp change
where the picture holds none. Extending and treating the outside as black both
answer 0 at the dark left border; reading from the opposite side answers a third
there, and treating the outside as black answers a full step at the *bright*
right border, so neither of the two is innocent in general.

``/operators`` is the comparison. Each operator's weights, the sum of its
positive weights, and what it answers on a clean step, which is that same sum
because every positive weight meets a one and every negative weight meets a
zero. Then the diagonal bar from the shared scene, where the four agree on the
angle to within 0.2839 degrees while their magnitudes span a factor of 26. Then
a straight edge drawn at each angle in turn and blurred across a couple of
pixels, which is what a lens does to a real edge and is the case Scharr's
weights were chosen for: the worst reported angle is out by 4.13 degrees for the
bare central difference, 3.10 for Prewitt, 1.16 for Sobel and 0.28 for Scharr.
The same sweep on a hard one-pixel step is run beside it, where every operator
is out by nine degrees or more and Scharr is the worst of the three smoothed
ones, because a one-pixel step is not something a derivative estimate can be
accurate on. That is why the hand-drawn workbench cannot show the difference the
operators were designed around, and the page says so.

``/thresholds`` takes the crudest possible edge decision and shows what it
cannot do. A stated amount of sensor noise is added to the workbench under a
fixed draw, and every threshold is scored against two populations known from the
ramp-free scene, the 450 pixels on a real boundary and the 1854 pixels on flat
ground. No threshold keeps all of the first and none of the second.

The noise is the one thing on this page that is not in the shared scene. The
workbench is drawn by arithmetic and so is perfectly clean, and a threshold on a
perfectly clean picture is a solved problem, which would make the closing
section a claim rather than a measurement.
"""

from __future__ import annotations

import time
from enum import StrEnum
from functools import lru_cache

import numpy as np
from fastapi import APIRouter
from oop_ml.core.computer_vision.edges import (
    HORIZONTAL_WEIGHTS,
    GradientField,
    GradientOperator,
    weights_of,
)
from oop_ml.core.computer_vision.filtering import EdgeRule, swept
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.types import FloatArray
from pydantic import BaseModel

from oop_ml_api.vision_scenes import (
    BAR_FROM,
    BAR_THICKNESS,
    RAMP_ACROSS,
    SCENE_SIDE,
    as_rows,
    hand_sized,
    relit,
    workbench,
)

router = APIRouter(prefix="/concepts/filters-and-edges", tags=["filters-and-edges"])

SHAPE_THRESHOLD = 0.7
"""What separates a shape from the ground in the workbench, before relighting.

The scene is drawn with its ground at 0.12 and its shapes at 0.78, and the ramp
adds at most 0.30 to both, so anything at or above 0.7 belongs to a shape and
nothing else does. Used only to *label* the pixels, so that the endpoint can
then ask how a threshold chosen by a reader does against that labelling.
"""

BRIGHTNESS_THRESHOLD = 0.6
"""The threshold the page quotes, chosen inside the window that works.

Every value between the brightest ground pixel and the darkest shape pixel
labels the workbench perfectly, and 0.6 sits in the middle of that window. It is
quoted rather than derived so that a reader can check it against the two numbers
the endpoint reports beside it.
"""

NOISE_LEVEL = 0.08
"""How much noise is added before the thresholds are scored.

The spread of a normal draw, in the same units as the brightness, against a
scene whose shapes stand 0.66 clear of its ground. Twelve per cent of the
contrast is heavy for a bright photograph and ordinary for a dim one. It is
fixed rather than offered as a dial because the page quotes counts from it and
a reader changing it would read different counts beside the same sentences.
"""

NOISE_SEED = 0
"""The draw the noise comes from, fixed so every request answers the same thing."""

FAINT_BELOW = 1.0
"""What counts as the faint half of a real edge.

A boundary pixel whose magnitude in the clean scene falls below this. Those are
the obliquely crossed pixels, on the disc's rim and along the diagonal bar,
where a three-wide grid straddles the change rather than meeting it square.
"""

EDGE_ABOVE = 0.2
"""What counts as a real boundary pixel in the scene without its ramp.

Read off the ramp-free scene rather than the ramped one, so that the ramp's own
gentle change does not count as a boundary. Without the ramp a flat region
answers exactly zero, so this only has to sit clear of zero.
"""

ANGLE_STEP_DEGREES = 5
"""How finely the straight-edge sweep turns."""

BLUR_WIDTH = 1.0
"""How far a straight edge is spread across, in pixels, before it is swept.

A photographed edge is never one pixel wide; the lens and the sensor spread it
over two or three. The sweep is run at this width and again at zero width, and
the two answer very different things, which is the point of running both.
"""

TIMING_SIDES = (48, 256, 1024)
"""The picture sides the cost is measured at.

The scene itself, and two larger ones, because at 48 by 48 most of the time is
the fixed cost of setting a sweep up rather than the arithmetic of doing it, and
a per-pixel figure read off the scene alone would be twice what it should be.
"""

TIMING_REPEATS = {48: 50, 256: 20, 1024: 3}
"""How many sweeps go into one timing sample, per side."""

TIMING_SAMPLES = 7
"""How many samples the median is taken over."""

MULTIPLIES_PER_PIXEL = 9
"""One product per weight, for a three by three grid."""

ADDITIONS_PER_PIXEL = 8
"""One fewer addition than there are products, to sum nine of them."""


class SceneName(StrEnum):
    """Which of the shared pictures an endpoint is asked about.

    A closed set rather than a free string, so that a name nothing draws is a
    refusal at the door rather than an answer computed from a picture nobody
    chose.
    """

    WORKBENCH = "workbench"
    WITHOUT_RAMP = "without_ramp"
    RELIT = "relit"
    HAND = "hand"


SCENE_LABELS: dict[SceneName, str] = {
    SceneName.WORKBENCH: "the scene",
    SceneName.WITHOUT_RAMP: "the scene without its ramp",
    SceneName.RELIT: "the scene under a brighter lamp",
    SceneName.HAND: "the six by six picture",
}
"""What each name is called in words, written once here so the page and the
widgets cannot drift apart about it."""

EDGE_RULE_LABELS: dict[EdgeRule, str] = {
    EdgeRule.EXTEND: "Repeat the edge pixel outward",
    EdgeRule.PAD_WITH_ZERO: "Treat everything outside as black",
    EdgeRule.WRAP: "Read from the opposite side",
    EdgeRule.KEEP_VALID: "Answer only where the weights fit",
}
"""Each border rule in words a reader can act on."""

OPERATOR_LABELS: dict[GradientOperator, str] = {
    GradientOperator.CENTRAL_DIFFERENCE: "Central difference",
    GradientOperator.PREWITT: "Prewitt",
    GradientOperator.SOBEL: "Sobel",
    GradientOperator.SCHARR: "Scharr",
}
"""Each operator under the name it is published as."""

OPERATOR_NOTES: dict[GradientOperator, str] = {
    GradientOperator.CENTRAL_DIFFERENCE: (
        "The bare difference of the two neighbours, halved. One row, so nothing "
        "is averaged and a single stray pixel moves the answer as much as a "
        "real edge does."
    ),
    GradientOperator.PREWITT: (
        "Three rows, weighted equally. The difference is taken across the edge "
        "and the three answers are averaged along it, which is what makes a "
        "stray pixel count for a third of what it did."
    ),
    GradientOperator.SOBEL: (
        "The same three rows with the middle one counted twice, so the row the "
        "answer is reported at has more say than the rows above and below it."
    ),
    GradientOperator.SCHARR: (
        "Three and ten rather than one and two, chosen so that the angle the "
        "operator reports is as close to independent of how the edge is turned "
        "as three columns of weights allow."
    ),
}
"""One sentence per operator, saying what its weights buy."""


def _picture(name: SceneName) -> Picture:
    """The shared picture one name asks for."""
    if name is SceneName.WORKBENCH:
        return workbench()
    if name is SceneName.WITHOUT_RAMP:
        return workbench(with_ramp=False)
    if name is SceneName.RELIT:
        return relit()
    return hand_sized()


def _rounded(values: FloatArray, places: int = 4) -> list[list[float]]:
    """A bare array as nested lists, rounded for sending."""
    return [[round(float(value), places) for value in row] for row in values]


def _straight_edge(degrees: float, side: int, blur: float) -> Picture:
    """A step across a square picture, with its bright side in ``degrees``.

    At a blur of zero the step is one pixel wide and the picture is the hard
    edge a hand-drawn scene contains. Above zero the step is spread across that
    many pixels, which is what a lens and a sensor do to the boundary of a real
    object, and it is the case the smoothed operators were designed for.
    """
    angle = np.radians(degrees)
    rows, columns = np.mgrid[:side, :side]
    across = (columns - side // 2) * np.cos(angle) + (rows - side // 2) * np.sin(angle)
    if blur <= 0.0:
        return Picture((across >= 0.0).astype(np.float64))
    return Picture(0.5 + 0.5 * np.tanh(across / blur))


def _angle_error(reported: float, wanted: float) -> float:
    """How far apart two angles are in degrees, the short way round."""
    return float(abs(((reported - wanted + 180.0) % 360.0) - 180.0))


class ThresholdReading(BaseModel):
    """What one brightness threshold makes of one picture.

    ``kept`` is how many pixels sit at or above it, and ``wrongly_labelled`` is
    how many of those disagree with which pixels actually belong to a shape.
    """

    threshold: float
    kept: int
    wrongly_labelled: int


class LightingReading(BaseModel):
    """What a change of lighting did to the pixels and to the gradient.

    ``brightest_ground`` and ``darkest_shape`` bracket every threshold that
    labels that picture perfectly, so the two pairs together say whether one
    number can serve both pictures. ``magnitude_ratio_low`` and
    ``magnitude_ratio_high`` bracket what the relighting did to the length of
    the gradient, and ``largest_direction_move`` is the largest angle in radians
    that any pixel's gradient turned through.
    """

    scale: float
    shift: float
    n_pixels: int
    n_shape_pixels: int
    mean_brightness_move: float
    largest_brightness_move: float
    brightest_ground: float
    darkest_shape: float
    relit_brightest_ground: float
    relit_darkest_shape: float
    threshold: float
    scene_reading: ThresholdReading
    relit_reading: ThresholdReading
    n_disagreeing: int
    share_disagreeing: float
    magnitude_ratio_low: float
    magnitude_ratio_high: float
    largest_direction_move: float
    n_sharp_pixels: int
    largest_direction_move_where_sharp: float


class SceneResponse(BaseModel):
    """The three pictures the page opens on, and what the lamp did to them."""

    height: int
    width: int
    scene: list[list[float]]
    without_ramp: list[list[float]]
    relit: list[list[float]]
    darkest: float
    brightest: float
    ramp_across: float
    ramp_per_column: float
    lighting: LightingReading


class FieldRequest(BaseModel):
    """One operator swept across one picture under one border rule."""

    scene: SceneName = SceneName.WORKBENCH
    operator: GradientOperator = GradientOperator.SOBEL
    edge_rule: EdgeRule = EdgeRule.EXTEND

    model_config = {"extra": "forbid"}


class FieldResponse(BaseModel):
    """The two rates of change, and the length and angle they make together.

    ``direction`` is in radians, measured from pointing right and turning
    towards pointing down. ``edge_direction`` is that turned by a right angle,
    which is the way the edge itself runs. Where the picture is flat the angle
    is reported as zero, which is a convention and not a measurement, so
    ``magnitude`` is what says whether an angle means anything.
    """

    scene: SceneName
    scene_label: str
    operator: GradientOperator
    operator_label: str
    edge_rule: EdgeRule
    edge_rule_label: str
    picture: list[list[float]]
    height: int
    width: int
    answer_height: int
    answer_width: int
    horizontal: list[list[float]]
    vertical: list[list[float]]
    magnitude: list[list[float]]
    direction: list[list[float]]
    edge_direction: list[list[float]]
    largest_magnitude: float
    largest_rate: float


class SweepCell(BaseModel):
    """The window behind one answer of the hand-worked sweep.

    ``top`` and ``left`` are where the window's first pixel sits, negative
    where the window begins outside the picture. ``patch`` is what the window
    read, whatever the border rule invented included, and ``products`` is that
    patch times the weights entry by entry, whose sum is ``answer``.
    """

    row: int
    column: int
    top: int
    left: int
    patch: list[list[float]]
    products: list[list[float]]
    answer: float


class HandSweepRequest(BaseModel):
    """The worked sweep, with the weights optionally reversed."""

    operator: GradientOperator = GradientOperator.CENTRAL_DIFFERENCE
    edge_rule: EdgeRule = EdgeRule.EXTEND
    vertical: bool = False
    reversed_weights: bool = False

    model_config = {"extra": "forbid"}


class HandSweepResponse(BaseModel):
    """The six by six picture, the weights, the answer, and every window.

    ``padded`` is the picture with whatever the border rule invented already
    added around it, so the page can show the reader what the weights were
    actually laid over. Under the rule that answers only where the weights fit,
    nothing is added and the answer is smaller than the picture, which is what
    ``answer_height`` and ``answer_width`` report.
    """

    operator: GradientOperator
    operator_label: str
    edge_rule: EdgeRule
    edge_rule_label: str
    reversed_weights: bool
    weights: list[list[float]]
    positive_weight_total: float
    weight_total: float
    picture: list[list[float]]
    padded: list[list[float]]
    pad_rows: int
    pad_columns: int
    answer: list[list[float]]
    answer_height: int
    answer_width: int
    cells: list[SweepCell]


class BorderReading(BaseModel):
    """What one border rule answers on the hand-worked picture.

    ``averaged`` is a flat three by three average, which shows what the rule
    invented as a brightness. ``magnitude`` is the usual operator's answer,
    which shows whether the invention reads as an edge. ``invented_at`` names
    the columns of the middle row where a sharp change is reported and the
    picture holds none, which is the whole argument for choosing between these
    deliberately.
    """

    edge_rule: EdgeRule
    label: str
    answer_height: int
    answer_width: int
    averaged: list[list[float]]
    magnitude: list[list[float]]
    averaged_middle_row: list[float]
    magnitude_middle_row: list[float]
    invented_at: list[int]
    note: str


class BordersResponse(BaseModel):
    """All four rules over the same picture, with its one real edge named."""

    picture: list[list[float]]
    height: int
    width: int
    real_edge_between: list[int]
    readings: list[BorderReading]


class AngleReading(BaseModel):
    """One operator's answer at one drawn angle."""

    degrees: int
    reported: float
    error: float


class OperatorSweep(BaseModel):
    """One operator across every drawn angle, blurred and hard.

    ``blurred`` is the case a photographed edge is in, and ``hard`` is the case
    a hand-drawn picture is in. The two orderings are not the same, which is
    reported rather than smoothed over.
    """

    blurred: list[AngleReading]
    blurred_mean_error: float
    blurred_worst_error: float
    hard_mean_error: float
    hard_worst_error: float


class OperatorReading(BaseModel):
    """One operator, all the way through.

    ``clean_step_answer`` is what it reports at a boundary between a column of
    zeros and a column of ones, which comes out equal to
    ``positive_weight_total`` for every one of them. ``bar_angle`` is the angle
    it reports on the shared scene's diagonal bar, in degrees, against a bar
    whose edge runs at exactly minus 135.
    """

    operator: GradientOperator
    label: str
    note: str
    weights: list[list[float]]
    n_rows: int
    n_columns: int
    positive_weight_total: float
    weight_total: float
    clean_step_answer: float
    clean_step_direction: float
    clean_step_edge_direction: float
    bar_angle: float
    bar_angle_error: float
    bar_magnitude: float
    sweep: OperatorSweep


class CostReading(BaseModel):
    """What one sweep of one square picture took, and what it had to do.

    The seconds are a measurement of whichever machine served the request and
    move a little between runs, so what the page leans on is the count of
    operations and the way the per-pixel figure falls as the picture grows.
    """

    side: int
    n_pixels: int
    seconds_per_sweep: float
    nanoseconds_per_pixel: float
    n_multiplies: int
    n_additions: int


class OperatorsResponse(BaseModel):
    """The four operators compared, and what a sweep costs.

    ``bar_angle_spread`` is how far apart the four reported angles are on the
    diagonal bar, and ``bar_magnitude_ratio`` is the largest of the four
    magnitudes over the smallest, so the two together say that the operators
    part company over how big the answer is rather than over which way it
    points.
    """

    angles: list[int]
    blur_width: float
    readings: list[OperatorReading]
    bar_angle_spread: float
    bar_magnitude_ratio: float
    bar_true_angle: float
    costs: list[CostReading]
    n_operations_per_pixel: int
    n_operations_for_scene: int


class ThresholdRow(BaseModel):
    """One threshold on the magnitude, scored against what is really there.

    ``flat_kept`` is how many pixels on flat ground survive it and ``edge_lost``
    is how many pixels on a real boundary do not, so a threshold that did the
    job would answer zero to both.
    """

    threshold: float
    kept: int
    flat_kept: int
    edge_kept: int
    edge_lost: int
    faint_kept: int
    faint_lost: int
    n_wrong: int


class ThresholdsResponse(BaseModel):
    """What a single threshold on the magnitude can and cannot do.

    ``best_threshold`` is whichever row leaves the fewest pixels labelled
    wrongly, and ``best_wrong`` is how many that still is, which is the number
    the page's closing claim rests on.
    """

    noise_level: float
    height: int
    width: int
    magnitude: list[list[float]]
    clean_magnitude: list[list[float]]
    largest_magnitude: float
    n_pixels: int
    n_edge_pixels: int
    n_flat_pixels: int
    n_faint_pixels: int
    faint_below: float
    clean_flat_low: float
    clean_flat_high: float
    noisy_flat_low: float
    noisy_flat_high: float
    rows: list[ThresholdRow]
    best_threshold: float
    best_wrong: int
    lowest_clean_threshold: float
    highest_clean_threshold: float


def _threshold_reading(
    values: FloatArray, is_shape: FloatArray, threshold: float
) -> ThresholdReading:
    """Score one brightness threshold against which pixels are really shapes."""
    kept = values >= threshold
    return ThresholdReading(
        threshold=threshold,
        kept=int(kept.sum()),
        wrongly_labelled=int((kept != is_shape).sum()),
    )


@lru_cache(maxsize=1)
def _lighting() -> LightingReading:
    """Measure what the shared relighting did to the pixels and the gradient."""
    scene = workbench()
    lit = relit()
    here = np.asarray(scene)
    there = np.asarray(lit)
    is_shape = here >= SHAPE_THRESHOLD

    moved = np.abs(there - here)
    here_field = GradientField.of(scene)
    there_field = GradientField.of(lit)
    here_magnitude = np.asarray(here_field.magnitude)
    there_magnitude = np.asarray(there_field.magnitude)
    turning = np.abs(
        (np.asarray(there_field.direction) - np.asarray(here_field.direction) + np.pi)
        % (2 * np.pi)
        - np.pi
    )
    lively = here_magnitude > 1e-9
    ratio = there_magnitude[lively] / here_magnitude[lively]
    sharp = here_magnitude > 0.5

    scene_reading = _threshold_reading(here, is_shape, BRIGHTNESS_THRESHOLD)
    relit_reading = _threshold_reading(there, is_shape, BRIGHTNESS_THRESHOLD)
    disagreeing = int(
        ((here >= BRIGHTNESS_THRESHOLD) != (there >= BRIGHTNESS_THRESHOLD)).sum()
    )

    return LightingReading(
        scale=1.6,
        shift=0.15,
        n_pixels=int(here.size),
        n_shape_pixels=int(is_shape.sum()),
        mean_brightness_move=float(moved.mean()),
        largest_brightness_move=float(moved.max()),
        brightest_ground=float(here[~is_shape].max()),
        darkest_shape=float(here[is_shape].min()),
        relit_brightest_ground=float(there[~is_shape].max()),
        relit_darkest_shape=float(there[is_shape].min()),
        threshold=BRIGHTNESS_THRESHOLD,
        scene_reading=scene_reading,
        relit_reading=relit_reading,
        n_disagreeing=disagreeing,
        share_disagreeing=float(disagreeing / here.size),
        magnitude_ratio_low=float(ratio.min()),
        magnitude_ratio_high=float(ratio.max()),
        largest_direction_move=float(turning.max()),
        n_sharp_pixels=int(sharp.sum()),
        largest_direction_move_where_sharp=float(turning[sharp].max()),
    )


@router.get("/scene", response_model=SceneResponse)
def read_scene() -> SceneResponse:
    """The workbench, the same scene without its ramp, and the same relit."""
    scene = workbench()
    return SceneResponse(
        height=scene.height,
        width=scene.width,
        scene=as_rows(scene),
        without_ramp=as_rows(workbench(with_ramp=False)),
        relit=as_rows(relit()),
        darkest=scene.darkest,
        brightest=scene.brightest,
        ramp_across=RAMP_ACROSS,
        ramp_per_column=RAMP_ACROSS / (SCENE_SIDE - 1),
        lighting=_lighting(),
    )


@router.post("/field", response_model=FieldResponse)
def sweep_field(request: FieldRequest) -> FieldResponse:
    """Sweep one operator across one picture and answer all four maps."""
    picture = _picture(request.scene)
    field = GradientField.of(picture, request.operator, request.edge_rule)
    horizontal = np.asarray(field.horizontal)
    vertical = np.asarray(field.vertical)
    magnitude = np.asarray(field.magnitude)
    answer_height, answer_width = field.shape
    return FieldResponse(
        scene=request.scene,
        scene_label=SCENE_LABELS[request.scene],
        operator=request.operator,
        operator_label=OPERATOR_LABELS[request.operator],
        edge_rule=request.edge_rule,
        edge_rule_label=EDGE_RULE_LABELS[request.edge_rule],
        picture=as_rows(picture),
        height=picture.height,
        width=picture.width,
        answer_height=answer_height,
        answer_width=answer_width,
        horizontal=_rounded(horizontal),
        vertical=_rounded(vertical),
        magnitude=_rounded(magnitude),
        direction=_rounded(np.asarray(field.direction)),
        edge_direction=_rounded(np.asarray(field.edge_direction)),
        largest_magnitude=float(magnitude.max()),
        largest_rate=float(max(np.abs(horizontal).max(), np.abs(vertical).max())),
    )


@router.post("/hand-sweep", response_model=HandSweepResponse)
def sweep_by_hand(request: HandSweepRequest) -> HandSweepResponse:
    """The six by six picture swept, with the arithmetic behind every answer."""
    picture = hand_sized()
    grid = np.asarray(
        weights_of(request.operator, vertical=request.vertical), dtype=np.float64
    )
    if request.reversed_weights:
        grid = grid[::-1, ::-1]

    answer = swept(picture, grid, request.edge_rule)
    height, width = grid.shape
    pad_rows, pad_columns = height // 2, width // 2
    if request.edge_rule is EdgeRule.KEEP_VALID:
        padded = np.asarray(picture)
        pad_rows = pad_columns = 0
    else:
        padded = np.pad(
            np.asarray(picture),
            ((pad_rows, pad_rows), (pad_columns, pad_columns)),
            mode={
                EdgeRule.EXTEND: "edge",
                EdgeRule.WRAP: "wrap",
                EdgeRule.PAD_WITH_ZERO: "constant",
            }[request.edge_rule],
        )

    answer_height, answer_width = answer.shape
    cells: list[SweepCell] = []
    for row in range(answer_height):
        for column in range(answer_width):
            patch = padded[row : row + height, column : column + width]
            products = patch * grid
            cells.append(
                SweepCell(
                    row=row,
                    column=column,
                    top=row - pad_rows,
                    left=column - pad_columns,
                    patch=_rounded(patch),
                    products=_rounded(products),
                    answer=round(float(products.sum()), 6),
                )
            )

    return HandSweepResponse(
        operator=request.operator,
        operator_label=OPERATOR_LABELS[request.operator],
        edge_rule=request.edge_rule,
        edge_rule_label=EDGE_RULE_LABELS[request.edge_rule],
        reversed_weights=request.reversed_weights,
        weights=_rounded(grid),
        positive_weight_total=float(grid[grid > 0].sum()),
        weight_total=float(grid.sum()),
        picture=as_rows(picture),
        padded=_rounded(padded),
        pad_rows=pad_rows,
        pad_columns=pad_columns,
        answer=as_rows(answer),
        answer_height=answer_height,
        answer_width=answer_width,
        cells=cells,
    )


BORDER_NOTES: dict[EdgeRule, str] = {
    EdgeRule.EXTEND: (
        "The picture is assumed to carry on as it was, so a flat border stays "
        "flat and neither side reports a change that is not there. It is the "
        "smallest thing any of these rules invents, which is why it is the one "
        "to reach for by default."
    ),
    EdgeRule.PAD_WITH_ZERO: (
        "The world outside the frame is assumed to be black. That happens to "
        "be true at the dark left border of this picture and false at the "
        "bright right one, where it reports a change as sharp as the real edge."
    ),
    EdgeRule.WRAP: (
        "The picture is assumed to tile, so the left border reads the bright "
        "right-hand side and the right border reads the dark left-hand side. "
        "Both ends then report an edge, and the scene contains neither."
    ),
    EdgeRule.KEEP_VALID: (
        "Nothing is invented and nothing is answered where the weights do not "
        "fit, so the answer comes back smaller than the question by one less "
        "than each side of the grid."
    ),
}
"""What each rule assumes about the world outside the frame."""


@lru_cache(maxsize=1)
def _borders() -> BordersResponse:
    """Run all four border rules over the hand-worked picture."""
    picture = hand_sized()
    average = [[1 / 9] * 3] * 3
    middle = picture.height // 2

    readings: list[BorderReading] = []
    for rule in EdgeRule:
        averaged = np.asarray(swept(picture, average, rule))
        magnitude = np.asarray(
            GradientField.of(picture, GradientOperator.SOBEL, rule).magnitude
        )
        answer_height, answer_width = magnitude.shape
        row = magnitude[min(middle, answer_height - 1)]
        real = {2, 3} if rule is not EdgeRule.KEEP_VALID else {1, 2}
        invented = [
            int(column)
            for column in range(answer_width)
            if row[column] > 0.5 and column not in real
        ]
        readings.append(
            BorderReading(
                edge_rule=rule,
                label=EDGE_RULE_LABELS[rule],
                answer_height=answer_height,
                answer_width=answer_width,
                averaged=_rounded(averaged),
                magnitude=_rounded(magnitude),
                averaged_middle_row=_rounded(averaged)[min(middle, answer_height - 1)],
                magnitude_middle_row=_rounded(magnitude)[
                    min(middle, answer_height - 1)
                ],
                invented_at=invented,
                note=BORDER_NOTES[rule],
            )
        )

    return BordersResponse(
        picture=as_rows(picture),
        height=picture.height,
        width=picture.width,
        real_edge_between=[2, 3],
        readings=readings,
    )


@router.get("/borders", response_model=BordersResponse)
def read_borders() -> BordersResponse:
    """Every border rule over one picture with one edge in a known place."""
    return _borders()


def _timed_sweep(side: int) -> CostReading:
    """Time one sweep of a square picture of this side."""
    picture = Picture(np.random.default_rng(1).random((side, side)))
    grid = weights_of(GradientOperator.SOBEL, vertical=False)
    repeats = TIMING_REPEATS[side]
    for _ in range(2):
        swept(picture, grid)
    samples = []
    for _ in range(TIMING_SAMPLES):
        start = time.perf_counter()
        for _ in range(repeats):
            swept(picture, grid)
        samples.append((time.perf_counter() - start) / repeats)
    seconds = float(np.median(samples))
    return CostReading(
        side=side,
        n_pixels=side * side,
        seconds_per_sweep=seconds,
        nanoseconds_per_pixel=seconds / (side * side) * 1e9,
        n_multiplies=MULTIPLIES_PER_PIXEL * side * side,
        n_additions=ADDITIONS_PER_PIXEL * side * side,
    )


@lru_cache(maxsize=1)
def _operators() -> OperatorsResponse:
    """Compare the four operators on a step, on the bar and across the angles."""
    scene = workbench()
    step = Picture([[0.0, 0.0, 1.0, 1.0]] * 5)
    bar_row, bar_column = BAR_FROM
    bar_at = (bar_row - 2, bar_column + 2 + BAR_THICKNESS)
    angles = list(range(0, 90, ANGLE_STEP_DEGREES))
    side = 21

    readings: list[OperatorReading] = []
    for operator in GradientOperator:
        grid = np.asarray(HORIZONTAL_WEIGHTS[operator], dtype=np.float64)
        step_field = GradientField.of(step, operator)
        bar_field = GradientField.of(scene, operator)
        bar_angle = float(np.degrees(np.asarray(bar_field.direction)[bar_at]))

        blurred: list[AngleReading] = []
        hard_errors: list[float] = []
        for degrees in angles:
            drawn = _straight_edge(float(degrees), side, BLUR_WIDTH)
            reported = float(
                np.degrees(
                    np.asarray(GradientField.of(drawn, operator).direction)[
                        side // 2, side // 2
                    ]
                )
            )
            blurred.append(
                AngleReading(
                    degrees=degrees,
                    reported=reported,
                    error=_angle_error(reported, float(degrees)),
                )
            )
            sharp = _straight_edge(float(degrees), side, 0.0)
            hard_reported = float(
                np.degrees(
                    np.asarray(GradientField.of(sharp, operator).direction)[
                        side // 2, side // 2
                    ]
                )
            )
            hard_errors.append(_angle_error(hard_reported, float(degrees)))

        blurred_errors = [reading.error for reading in blurred]
        readings.append(
            OperatorReading(
                operator=operator,
                label=OPERATOR_LABELS[operator],
                note=OPERATOR_NOTES[operator],
                weights=_rounded(grid),
                n_rows=int(grid.shape[0]),
                n_columns=int(grid.shape[1]),
                positive_weight_total=float(grid[grid > 0].sum()),
                weight_total=float(grid.sum()),
                clean_step_answer=float(np.asarray(step_field.magnitude)[2, 1]),
                clean_step_direction=float(np.asarray(step_field.direction)[2, 1]),
                clean_step_edge_direction=float(
                    np.asarray(step_field.edge_direction)[2, 1]
                ),
                bar_angle=bar_angle,
                bar_angle_error=_angle_error(bar_angle, -135.0),
                bar_magnitude=float(np.asarray(bar_field.magnitude)[bar_at]),
                sweep=OperatorSweep(
                    blurred=blurred,
                    blurred_mean_error=float(np.mean(blurred_errors)),
                    blurred_worst_error=float(np.max(blurred_errors)),
                    hard_mean_error=float(np.mean(hard_errors)),
                    hard_worst_error=float(np.max(hard_errors)),
                ),
            )
        )

    bar_angles = [reading.bar_angle for reading in readings]
    bar_magnitudes = [reading.bar_magnitude for reading in readings]
    return OperatorsResponse(
        angles=angles,
        blur_width=BLUR_WIDTH,
        readings=readings,
        bar_angle_spread=float(max(bar_angles) - min(bar_angles)),
        bar_magnitude_ratio=float(max(bar_magnitudes) / min(bar_magnitudes)),
        bar_true_angle=-135.0,
        costs=[_timed_sweep(side) for side in TIMING_SIDES],
        n_operations_per_pixel=2 * (MULTIPLIES_PER_PIXEL + ADDITIONS_PER_PIXEL),
        n_operations_for_scene=2
        * (MULTIPLIES_PER_PIXEL + ADDITIONS_PER_PIXEL)
        * SCENE_SIDE
        * SCENE_SIDE,
    )


@router.get("/operators", response_model=OperatorsResponse)
def read_operators() -> OperatorsResponse:
    """The four operators compared, and what one sweep costs."""
    return _operators()


THRESHOLD_STEPS = [round(0.1 * step, 2) for step in range(1, 31)]
"""The thresholds scored, from 0.1 to 3.0 in tenths."""


@lru_cache(maxsize=1)
def _thresholds() -> ThresholdsResponse:
    """Score every threshold on the magnitude against what is really there."""
    scene = workbench()
    values = np.asarray(scene)
    clean_magnitude = np.asarray(GradientField.of(scene).magnitude)
    plain_magnitude = np.asarray(GradientField.of(workbench(with_ramp=False)).magnitude)
    on_edge = plain_magnitude > EDGE_ABOVE
    on_flat = plain_magnitude == 0.0
    faint = on_edge & (clean_magnitude < FAINT_BELOW)

    generator = np.random.default_rng(NOISE_SEED)
    noisy = Picture(values + generator.normal(0.0, NOISE_LEVEL, values.shape))
    magnitude = np.asarray(GradientField.of(noisy).magnitude)

    rows: list[ThresholdRow] = []
    for threshold in THRESHOLD_STEPS:
        kept = magnitude >= threshold
        rows.append(
            ThresholdRow(
                threshold=threshold,
                kept=int(kept.sum()),
                flat_kept=int((kept & on_flat).sum()),
                edge_kept=int((kept & on_edge).sum()),
                edge_lost=int((~kept & on_edge).sum()),
                faint_kept=int((kept & faint).sum()),
                faint_lost=int((~kept & faint).sum()),
                n_wrong=int((kept & on_flat).sum()) + int((~kept & on_edge).sum()),
            )
        )

    best = min(rows, key=lambda row: (row.n_wrong, row.threshold))
    clean_flat = clean_magnitude[on_flat]
    clean_edge = clean_magnitude[on_edge]
    return ThresholdsResponse(
        noise_level=NOISE_LEVEL,
        height=scene.height,
        width=scene.width,
        magnitude=_rounded(magnitude),
        clean_magnitude=_rounded(clean_magnitude),
        largest_magnitude=float(magnitude.max()),
        n_pixels=scene.n_pixels,
        n_edge_pixels=int(on_edge.sum()),
        n_flat_pixels=int(on_flat.sum()),
        n_faint_pixels=int(faint.sum()),
        faint_below=FAINT_BELOW,
        clean_flat_low=float(clean_flat.min()),
        clean_flat_high=float(clean_flat.max()),
        noisy_flat_low=float(magnitude[on_flat].min()),
        noisy_flat_high=float(magnitude[on_flat].max()),
        rows=rows,
        best_threshold=best.threshold,
        best_wrong=best.n_wrong,
        lowest_clean_threshold=float(clean_flat.max()),
        highest_clean_threshold=float(clean_edge.min()),
    )


@router.get("/thresholds", response_model=ThresholdsResponse)
def read_thresholds() -> ThresholdsResponse:
    """Every threshold on the magnitude, scored against what is really there."""
    return _thresholds()
