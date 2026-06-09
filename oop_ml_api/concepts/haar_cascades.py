"""Haar Cascades.

Every endpoint here is registered under ``/concepts/haar-cascades``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about the integral image, rectangle features, and stages ordered so the cheap ones run on everything.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, so that a reader meeting the workbench scene on one page
recognises it on the next.

Why this page needs pictures of its own as well
-----------------------------------------------
The workbench holds one copy each of a square, a disc, a bar and three small
crosses, which is a fine thing to *measure* and useless as a thing to *learn
from*: this is the only method in the section that is fitted rather than
designed, and a fit needs a few thousand examples of what the target is not.
So the detector here is trained on a fixture built in this module, and it is
the same fixture the library's own spec for the cascade uses, so that every
number the page quotes is pinned by a test somebody can go and read.

The fixture is twelve by twelve windows. A target is a four-row bright bar
across the middle of a cluttered ground, jittered one row either way; a
background is the cluttered ground alone. The clutter is blocky rather than
per-pixel, which is what makes the problem worth measuring, since structure at
the scale the features read is structure the features can be fooled by. Fifty
targets and two thousand backgrounds, which is the ratio the method exists for.

The workbench is still used, for the integral image, because that idea is about
a real picture and not about a fitted anything.

Every fit, scan and timing run is cached for the life of the process, since all
of them are deterministic and the fit costs about six tenths of a second.
"""

from __future__ import annotations

import re
import timeit
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.computer_vision.cascade import (
    CELL_SIGNS,
    LOOKUPS_PER_RECTANGLE,
    CascadeScan,
    CascadeStage,
    FeatureArrangement,
    FeatureBank,
    HaarCascade,
    IntegralImage,
    Rectangle,
    RectangleFeature,
    RuleDirection,
    ThresholdRule,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

from oop_ml_api import vision_scenes

router = APIRouter(prefix="/concepts/haar-cascades", tags=["haar-cascades"])


WINDOW_SIDE = 12
"""How many pixels on each side of the window the detector reads."""

CLUTTER_BLOCK = 3
"""How coarse the random ground is, in pixels.

Blocky rather than per-pixel on purpose. Per-pixel noise averages away inside
the first rectangle sum, which would leave the two classes trivially apart and
make every measurement on this page flattering.
"""

BAR_ROWS = 4
"""How many rows the bright bar covers."""

N_TARGETS = 50
N_BACKGROUNDS = 2000
"""Fifty of the thing and two thousand of everything else."""

TRAINING_SEED = 7
HELD_OUT_SEED = 99
POSE_SEED = 5
SCENE_SEED = 3
EXAMPLE_SEED = 41

SCENE_SIDE = 66
"""How many pixels on each side of the scene the detector is swept over."""

PLANTED_AT = ((4, 5), (30, 40), (44, 12))
"""Where the three targets sit in the swept scene, as ``(row, column)``."""

NEAR_ENOUGH = 2
"""How many pixels away from a planted target still counts as finding it."""

TIMING_REPETITIONS = 2000
"""How many times each read is repeated before the clock is divided.

Large enough that the clock's own resolution is not what is being measured,
small enough that the whole comparison finishes in under half a second.
"""

BUILD_REPETITIONS = 50
"""How many times the table over the large picture is built before averaging."""

LARGE_SIDE = 600
"""The side of the picture the constant-cost claim is measured on.

Six hundred squared is 360,000 pixels, so the whole-picture box holds 360,000
times the pixels of the one-pixel box and the two are read at the same cost.
"""

ARRANGEMENT_LABELS: dict[FeatureArrangement, str] = {
    FeatureArrangement.TWO_HORIZONTAL: "Two cells side by side",
    FeatureArrangement.TWO_VERTICAL: "Two cells stacked",
    FeatureArrangement.THREE_HORIZONTAL: "Three cells in a row",
    FeatureArrangement.THREE_VERTICAL: "Three cells in a column",
    FeatureArrangement.FOUR_CHEQUER: "Four cells in a chequer",
}
"""What each layout is called on the page.

The names are written here rather than passed through from the vocabulary the
code uses, because a reader has no reason to meet an identifier and every
string in this module is page text.
"""

ARRANGEMENT_READS: dict[FeatureArrangement, str] = {
    FeatureArrangement.TWO_HORIZONTAL: "an edge running down the window",
    FeatureArrangement.TWO_VERTICAL: "an edge running across the window",
    FeatureArrangement.THREE_HORIZONTAL: "a vertical strip against its two flanks",
    FeatureArrangement.THREE_VERTICAL: "a horizontal band against what is above "
    "and below it",
    FeatureArrangement.FOUR_CHEQUER: "one diagonal pair against the other",
}

DIRECTION_WORDS: dict[RuleDirection, str] = {
    RuleDirection.ABOVE: "above",
    RuleDirection.BELOW: "at or below",
}

COUNTED_HEIGHT = 6
COUNTED_WIDTH = 7
"""The hand-sized picture, whose pixel at row ``r`` and column ``c`` is
``10r + c`` so that no two boxes can accidentally sum to the same number and
every total can be worked out on paper."""

WINDOW_SIDES_SHOWN = (2, 4, 6, 8, 12, 16, 24)
"""The window sizes the feature count is reported for."""

CONSTRUCTED_OBJECT = re.compile(r"\b[A-Z][A-Za-z]*\([^()]*\)")
"""A refusal that names the thing it refused by spelling out how it was built.

The refusals here are written to be read and most of them already are, but a
box that will not fit reports itself as the call that made it, and a reader of
this page has no reason to meet that. ``_in_plain_words`` substitutes it and a
test asserts that nothing of the sort survives, so a message that grows one
later fails there rather than reaching the page.
"""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with anything spelled as code taken out of it."""
    return CONSTRUCTED_OBJECT.sub("that box", str(refusal))


# --- the fixture -----------------------------------------------------------


def _clutter(generator: np.random.Generator) -> FloatArray:
    """A blocky random ground, structured at the scale the features read."""
    coarse = generator.uniform(
        0.15, 0.85, (WINDOW_SIDE // CLUTTER_BLOCK, WINDOW_SIDE // CLUTTER_BLOCK)
    )
    return np.asarray(
        np.kron(coarse, np.ones((CLUTTER_BLOCK, CLUTTER_BLOCK))), dtype=np.float64
    )


def _speckled(generator: np.random.Generator, values: FloatArray) -> Picture:
    return Picture(
        np.clip(values + generator.normal(0.0, 0.05, values.shape), 0.0, 1.0)
    )


def _target(
    generator: np.random.Generator,
    lit_from_the_other_side: bool = False,
    turned_a_quarter: bool = False,
) -> Picture:
    """A bright bar across the middle of a cluttered ground."""
    values = np.full((WINDOW_SIDE, WINDOW_SIDE), 0.25)
    top = 4 + int(generator.integers(-1, 2))
    bar = slice(top, top + BAR_ROWS)
    if turned_a_quarter:
        values[:, bar] = 0.85
    else:
        values[bar, :] = 0.85
    if lit_from_the_other_side:
        values = 1.10 - values
    return _speckled(generator, 0.55 * values + 0.45 * _clutter(generator))


def _background(generator: np.random.Generator) -> Picture:
    """The cluttered ground with no bar in it."""
    return _speckled(generator, _clutter(generator))


def _training_set(seed: int) -> tuple[list[Picture], Feature]:
    generator = np.random.default_rng(seed)
    pictures = [_target(generator) for _ in range(N_TARGETS)]
    pictures.extend(_background(generator) for _ in range(N_BACKGROUNDS))
    return pictures, Feature("target", [1.0] * N_TARGETS + [0.0] * N_BACKGROUNDS)


def _planted_scene() -> Picture:
    """The scene the detector is swept over, with three targets hidden in it."""
    generator = np.random.default_rng(SCENE_SEED)
    blocks = SCENE_SIDE // CLUTTER_BLOCK
    values = np.kron(
        generator.uniform(0.15, 0.85, (blocks, blocks)),
        np.ones((CLUTTER_BLOCK, CLUTTER_BLOCK)),
    )
    for top, left in PLANTED_AT:
        values[top : top + WINDOW_SIDE, left : left + WINDOW_SIDE] = np.asarray(
            _target(generator).values
        )
    return Picture(values)


def _crowded_scene() -> Picture:
    """The same ground tiled edge to edge with targets.

    The case the ordering is no help in. A cascade is quick because most
    windows die at the first stage, so a scene in which most windows are the
    thing being looked for is a scene in which almost nothing dies early.
    """
    generator = np.random.default_rng(SCENE_SEED)
    blocks = SCENE_SIDE // CLUTTER_BLOCK
    values = np.kron(
        generator.uniform(0.15, 0.85, (blocks, blocks)),
        np.ones((CLUTTER_BLOCK, CLUTTER_BLOCK)),
    )
    for top in range(0, SCENE_SIDE - WINDOW_SIDE + 1, WINDOW_SIDE):
        for left in range(0, SCENE_SIDE - WINDOW_SIDE + 1, WINDOW_SIDE):
            values[top : top + WINDOW_SIDE, left : left + WINDOW_SIDE] = np.asarray(
                _target(generator).values
            )
    return Picture(values)


def _clutter_only_scene() -> Picture:
    """The same ground with nothing hidden in it at all."""
    generator = np.random.default_rng(SCENE_SEED)
    blocks = SCENE_SIDE // CLUTTER_BLOCK
    return Picture(
        np.kron(
            generator.uniform(0.15, 0.85, (blocks, blocks)),
            np.ones((CLUTTER_BLOCK, CLUTTER_BLOCK)),
        )
    )


def _counted_picture() -> Picture:
    """The hand-sized picture, whose pixel at ``(row, column)`` is ``10 row + column``."""
    return Picture(
        [
            [10.0 * row + column for column in range(COUNTED_WIDTH)]
            for row in range(COUNTED_HEIGHT)
        ]
    )


@cache
def _fitted() -> HaarCascade:
    """The one fit the whole page is measured on. About six tenths of a second."""
    pictures, labels = _training_set(TRAINING_SEED)
    return HaarCascade(
        n_stages=5, rules_per_stage=2, n_candidate_features=600, random_seed=0
    ).fit(pictures, labels)


@cache
def _whole_window() -> Rectangle:
    return Rectangle(0, 0, WINDOW_SIDE, WINDOW_SIDE)


# --- shared shapes ---------------------------------------------------------


class PictureGrid(BaseModel):
    """A picture as rows of brightness, ready for a browser to draw."""

    height: int
    width: int
    rows: list[list[float]]


class Position(BaseModel):
    """A pixel, as the row and column it sits at."""

    top: int
    left: int


class Box(BaseModel):
    """A rectangle, by its top-left pixel and its size."""

    top: int
    left: int
    height: int
    width: int
    n_pixels: int


class SignedBox(BaseModel):
    """One cell of a feature, and whether it is added or subtracted."""

    top: int
    left: int
    height: int
    width: int
    sign: int


def _as_grid(picture: Picture) -> PictureGrid:
    return PictureGrid(
        height=picture.height,
        width=picture.width,
        rows=vision_scenes.as_rows(picture),
    )


def _as_box(rectangle: Rectangle) -> Box:
    return Box(
        top=rectangle.top,
        left=rectangle.left,
        height=rectangle.height,
        width=rectangle.width,
        n_pixels=rectangle.n_pixels,
    )


# --- the integral image ----------------------------------------------------


class IntegralRequest(BaseModel):
    """Which picture to accumulate, and which box to read off the table."""

    picture: Literal["counted", "workbench"] = "counted"
    top: int = Field(default=1, ge=0, le=200)
    left: int = Field(default=2, ge=0, le=200)
    height: int = Field(default=3, ge=1, le=200)
    width: int = Field(default=3, ge=1, le=200)

    model_config = {"extra": "forbid"}


class CornerReading(BaseModel):
    """One entry of the table, and the sign it enters the total with."""

    row: int
    column: int
    sign: int
    value: float


class Agreement(BaseModel):
    """Whether the four-lookup total is the sum of the pixels, on every box."""

    picture_height: int
    picture_width: int
    n_rectangles: int
    n_agreeing: int
    largest_gap: float


class IntegralResponse(BaseModel):
    """A picture, its table of running totals, and one box read off it.

    ``from_the_table`` is the four corners added with their signs and
    ``from_the_pixels`` is every pixel inside the box added up, so the two
    numbers are the claim and its check.
    """

    picture: PictureGrid
    table: list[list[float]]
    rectangle: Box
    corners: list[CornerReading]
    from_the_table: float
    from_the_pixels: float
    n_lookups: int
    agreement: Agreement


@cache
def _agreement() -> Agreement:
    """Every box of the hand-sized picture, both ways."""
    picture = _counted_picture()
    integral_image = IntegralImage.of(picture)
    n_rectangles = 0
    n_agreeing = 0
    largest_gap = 0.0
    for top in range(picture.height):
        for left in range(picture.width):
            for height in range(1, picture.height - top + 1):
                for width in range(1, picture.width - left + 1):
                    rectangle = Rectangle(top, left, height, width)
                    from_the_table = integral_image.sum_over(rectangle)
                    from_the_pixels = float(
                        picture.values[
                            rectangle.top : rectangle.bottom,
                            rectangle.left : rectangle.right,
                        ].sum()
                    )
                    gap = abs(from_the_table - from_the_pixels)
                    n_rectangles += 1
                    n_agreeing += int(gap == 0.0)
                    largest_gap = max(largest_gap, gap)
    return Agreement(
        picture_height=picture.height,
        picture_width=picture.width,
        n_rectangles=n_rectangles,
        n_agreeing=n_agreeing,
        largest_gap=largest_gap,
    )


@router.post("/integral-image", response_model=IntegralResponse)
def read_a_box(request: IntegralRequest) -> IntegralResponse:
    """Accumulate a picture and read one rectangle's total off the table.

    A box running off the edge of the picture comes back as a refusal written
    in the page's own words rather than in the words the code refused it with.
    """
    picture = (
        _counted_picture()
        if request.picture == "counted"
        else vision_scenes.workbench()
    )
    integral_image = IntegralImage.of(picture)
    try:
        rectangle = Rectangle(request.top, request.left, request.height, request.width)
        integral_image.check_contains(rectangle)
    except MLLibError as refusal:
        raise HTTPException(status_code=422, detail=_in_plain_words(refusal)) from None
    corners = [
        CornerReading(
            row=lookup.row,
            column=lookup.column,
            sign=lookup.sign,
            value=round(float(integral_image.table[lookup.row, lookup.column]), 4),
        )
        for lookup in integral_image.corners_for(rectangle)
    ]
    from_the_pixels = float(
        picture.values[
            rectangle.top : rectangle.bottom, rectangle.left : rectangle.right
        ].sum()
    )
    return IntegralResponse(
        picture=_as_grid(picture),
        table=[
            [round(float(value), 4) for value in row] for row in integral_image.table
        ],
        rectangle=_as_box(rectangle),
        corners=corners,
        from_the_table=round(integral_image.sum_over(rectangle), 4),
        from_the_pixels=round(from_the_pixels, 4),
        n_lookups=LOOKUPS_PER_RECTANGLE,
        agreement=_agreement(),
    )


# --- what a read costs -----------------------------------------------------


class BoxCost(BaseModel):
    """How many pixels a box holds, against how many table entries it reads."""

    height: int
    width: int
    n_pixels: int
    n_lookups: int


class ReadingCostResponse(BaseModel):
    """The constant cost, demonstrated by counting and then by the clock.

    ``times_larger`` is how many times more pixels the whole-picture box holds
    than the one-pixel box. ``table_ratio`` is how much longer the table takes
    to answer the large box than the small one, and ``pixel_ratio`` the same
    for adding the pixels up. ``reads_to_repay_the_table`` is how many
    whole-picture reads it takes before building the table has paid for itself.
    """

    boxes: list[BoxCost]
    side: int
    repetitions: int
    times_larger: int
    table_ratio: float
    pixel_ratio: float
    table_small_microseconds: float
    table_large_microseconds: float
    pixel_small_microseconds: float
    pixel_large_microseconds: float
    build_microseconds: float
    reads_to_repay_the_table: float


@cache
def _reading_cost() -> ReadingCostResponse:
    picture = Picture(np.random.default_rng(0).random((LARGE_SIDE, LARGE_SIDE)))
    integral_image = IntegralImage.of(picture)
    values = picture.values
    one_pixel = Rectangle(0, 0, 1, 1)
    whole_picture = Rectangle(0, 0, LARGE_SIDE, LARGE_SIDE)

    table_small = timeit.timeit(
        lambda: integral_image.sum_over(one_pixel), number=TIMING_REPETITIONS
    )
    table_large = timeit.timeit(
        lambda: integral_image.sum_over(whole_picture), number=TIMING_REPETITIONS
    )
    pixel_small = timeit.timeit(
        lambda: values[0:1, 0:1].sum(), number=TIMING_REPETITIONS
    )
    pixel_large = timeit.timeit(
        lambda: values[0:LARGE_SIDE, 0:LARGE_SIDE].sum(), number=TIMING_REPETITIONS
    )
    build = (
        timeit.timeit(lambda: IntegralImage.of(picture), number=BUILD_REPETITIONS)
        / BUILD_REPETITIONS
    )

    per_read_saved = (pixel_large - table_large) / TIMING_REPETITIONS
    boxes = [
        BoxCost(
            height=height,
            width=width,
            n_pixels=Rectangle(0, 0, height, width).n_pixels,
            n_lookups=len(integral_image.corners_for(Rectangle(0, 0, height, width))),
        )
        for height, width in ((1, 1), (17, 23), (LARGE_SIDE, LARGE_SIDE))
    ]
    return ReadingCostResponse(
        boxes=boxes,
        side=LARGE_SIDE,
        repetitions=TIMING_REPETITIONS,
        times_larger=whole_picture.n_pixels // one_pixel.n_pixels,
        table_ratio=round(table_large / table_small, 3),
        pixel_ratio=round(pixel_large / pixel_small, 1),
        table_small_microseconds=round(table_small / TIMING_REPETITIONS * 1e6, 3),
        table_large_microseconds=round(table_large / TIMING_REPETITIONS * 1e6, 3),
        pixel_small_microseconds=round(pixel_small / TIMING_REPETITIONS * 1e6, 3),
        pixel_large_microseconds=round(pixel_large / TIMING_REPETITIONS * 1e6, 3),
        build_microseconds=round(build * 1e6, 1),
        reads_to_repay_the_table=round(build / per_read_saved, 1),
    )


@router.get("/reading-cost", response_model=ReadingCostResponse)
def reading_cost() -> ReadingCostResponse:
    """Count the lookups, then time them, on a picture large enough to matter."""
    return _reading_cost()


# --- the arrangements ------------------------------------------------------


class ArrangementShape(BaseModel):
    """One layout of cells, with the sign each cell carries."""

    label: str
    reads: str
    rows_of_cells: int
    columns_of_cells: int
    signs: list[list[int]]
    sign_total: int
    balanced: bool


class WorkedReading(BaseModel):
    """One arrangement on one hand-drawn picture, and what it answers."""

    label: str
    arrangement: str
    picture: PictureGrid
    cells: list[SignedBox]
    value: float


class LevelReading(BaseModel):
    """What an arrangement answers on a picture with no contrast in it."""

    arrangement: str
    on_dim: float
    on_bright: float
    balanced: bool


class WindowCount(BaseModel):
    """How many features a window of one size admits."""

    side: int
    n_pixels: int
    n_features: int
    features_per_pixel: float


class RoundingFinding(BaseModel):
    """Two ways of adding a feature's cells up, and where they part.

    ``compensated`` is the total taken with the rounding compensated, which
    Python's own summing has done since version 3.12; ``plain`` adds the cells
    one at a time, which is what reading many windows at once does. The values
    are sent as text because the difference is in the last bit and a browser
    would round it away.
    """

    n_values: int
    n_disagreeing: int
    largest_gap: float
    plain: str
    compensated: str
    threshold: str
    plain_votes_yes: bool
    compensated_votes_yes: bool


class ArrangementsResponse(BaseModel):
    """Everything the page says about what a detector reads."""

    arrangements: list[ArrangementShape]
    worked: list[WorkedReading]
    level: list[LevelReading]
    counts: list[WindowCount]
    doubling_factor: float
    rounding: RoundingFinding


BRIGHT_OVER_DARK = [[1.0] * 4, [1.0] * 4, [0.0] * 4, [0.0] * 4]
ONE_BRIGHT_ROW = [[0.0] * 2, [1.0] * 2, [0.0] * 2, [0.0] * 2]
TWO_BRIGHT_ROWS = [[0.0] * 2, [1.0] * 2, [1.0] * 2, [0.0] * 2]
CHEQUER = [[1.0, 0.0], [0.0, 1.0]]
DIM = [[0.25] * 4] * 4
BRIGHT = [[0.9] * 4] * 4
"""Hand-drawn pictures, small enough that every reading below can be checked
by counting the pixels."""


def _worked(
    label: str,
    arrangement: FeatureArrangement,
    rows: list[list[float]],
    cell_height: int,
    cell_width: int,
) -> WorkedReading:
    picture = Picture(rows)
    window = Rectangle(0, 0, picture.height, picture.width)
    feature = RectangleFeature(arrangement, 0, 0, cell_height, cell_width)
    integral_image = IntegralImage.of(picture)
    return WorkedReading(
        label=label,
        arrangement=ARRANGEMENT_LABELS[arrangement],
        picture=_as_grid(picture),
        cells=[
            SignedBox(
                top=cell.rectangle.top,
                left=cell.rectangle.left,
                height=cell.rectangle.height,
                width=cell.rectangle.width,
                sign=cell.sign,
            )
            for cell in feature.cells_in(window)
        ],
        value=round(feature.value_in(integral_image, window), 6),
    )


@cache
def _rounding_finding() -> RoundingFinding:
    """Where the two ways of totalling a feature's cells part company.

    Both totals are built from the same rectangle sums; the only difference is
    whether they are added with the rounding compensated or one at a time.
    """
    generator = np.random.default_rng(11)
    pictures = [Picture(generator.random((8, 8))) for _ in range(5)]
    integral_images = [IntegralImage.of(picture) for picture in pictures]
    window = Rectangle(0, 0, 8, 8)
    bank = FeatureBank.exhaustive(8, 8).sample(40, np.random.default_rng(2))

    plain = np.array(
        [
            [feature.value_in(integral_image, window) for feature in bank]
            for integral_image in integral_images
        ]
    )
    compensated = np.array(
        [
            [
                sum(
                    cell.sign * integral_image.sum_over(cell.rectangle)
                    for cell in feature.cells_in(window)
                )
                for feature in bank
            ]
            for integral_image in integral_images
        ]
    )
    disagreeing = np.argwhere(plain != compensated)
    row, column = (int(one) for one in disagreeing[0])
    plain_value = float(plain[row, column])
    compensated_value = float(compensated[row, column])
    rule = ThresholdRule(bank[column], compensated_value, RuleDirection.ABOVE)
    return RoundingFinding(
        n_values=int(plain.size),
        n_disagreeing=len(disagreeing),
        largest_gap=float(np.abs(plain - compensated).max()),
        plain=repr(plain_value),
        compensated=repr(compensated_value),
        threshold=repr(compensated_value),
        plain_votes_yes=bool(rule.votes_for_value(plain_value)),
        compensated_votes_yes=bool(rule.votes_for_value(compensated_value)),
    )


@cache
def _arrangements() -> ArrangementsResponse:
    shapes = []
    level = []
    dim_image = IntegralImage.of(Picture(DIM))
    bright_image = IntegralImage.of(Picture(BRIGHT))
    window = Rectangle(0, 0, 4, 4)
    for arrangement, signs in CELL_SIGNS.items():
        sign_total = sum(sum(row) for row in signs)
        shapes.append(
            ArrangementShape(
                label=ARRANGEMENT_LABELS[arrangement],
                reads=ARRANGEMENT_READS[arrangement],
                rows_of_cells=len(signs),
                columns_of_cells=len(signs[0]),
                signs=[list(row) for row in signs],
                sign_total=sign_total,
                balanced=sign_total == 0,
            )
        )
        feature = RectangleFeature(arrangement, 0, 0, 1, 1)
        level.append(
            LevelReading(
                arrangement=ARRANGEMENT_LABELS[arrangement],
                on_dim=round(feature.value_in(dim_image, window), 6),
                on_bright=round(feature.value_in(bright_image, window), 6),
                balanced=sign_total == 0,
            )
        )

    worked = [
        _worked(
            "Bright above, dark below",
            FeatureArrangement.TWO_VERTICAL,
            BRIGHT_OVER_DARK,
            2,
            4,
        ),
        _worked(
            "The same picture, read side to side",
            FeatureArrangement.TWO_HORIZONTAL,
            BRIGHT_OVER_DARK,
            4,
            2,
        ),
        _worked(
            "One bright row between two dark ones",
            FeatureArrangement.THREE_VERTICAL,
            ONE_BRIGHT_ROW,
            1,
            2,
        ),
        _worked(
            "Two bright rows, so the band is as wide as its flank",
            FeatureArrangement.THREE_VERTICAL,
            TWO_BRIGHT_ROWS,
            1,
            2,
        ),
        _worked(
            "A chequer",
            FeatureArrangement.FOUR_CHEQUER,
            CHEQUER,
            1,
            1,
        ),
    ]

    counts = [
        WindowCount(
            side=side,
            n_pixels=side * side,
            n_features=FeatureBank.count_available(side, side),
            features_per_pixel=round(
                FeatureBank.count_available(side, side) / (side * side), 2
            ),
        )
        for side in WINDOW_SIDES_SHOWN
    ]
    doubling = FeatureBank.count_available(24, 24) / FeatureBank.count_available(12, 12)
    return ArrangementsResponse(
        arrangements=shapes,
        worked=worked,
        level=level,
        counts=counts,
        doubling_factor=round(doubling, 2),
        rounding=_rounding_finding(),
    )


@router.get("/arrangements", response_model=ArrangementsResponse)
def arrangements() -> ArrangementsResponse:
    """The five layouts, what they answer on hand-drawn pictures, and how many
    of them a window admits."""
    return _arrangements()


# --- the fitted detector ---------------------------------------------------


class RuleReading(BaseModel):
    """One feature, one threshold, one direction, and how loudly it votes."""

    arrangement: str
    top: int
    left: int
    cell_height: int
    cell_width: int
    threshold: float
    direction: str
    voice: float
    cells: list[SignedBox]


class StageReading(BaseModel):
    """One stage of the cascade, as it was fitted.

    ``lowest_positive_confidence`` is the least of the shares of the voice the
    training targets attracted, and ``demanded_share`` is what the stage asks
    for. A stage is only lowered as far as it has to be to keep every target,
    and never above the half a plain vote would ask for, so the two numbers
    together say whether the lowering ever bit.
    """

    number: int
    n_rules: int
    total_voice: float
    demanded_share: float
    lowest_positive_confidence: float
    was_lowered: bool
    negatives_reaching: int
    negatives_rejected: int
    negatives_surviving: int
    positives_kept: int
    rejection_share: float
    rules: list[RuleReading]


class HeldOutReading(BaseModel):
    """What the fitted detector does to a fresh draw of the same fixture."""

    n_backgrounds: int
    backgrounds_through_each_stage: list[int]
    background_shares: list[float]
    n_targets: int
    targets_through_each_stage: list[int]
    accuracy: float
    recall: float
    precision: float


class DetectorResponse(BaseModel):
    """The fit, what each stage did, and how it holds up on fresh windows."""

    window_side: int
    n_targets: int
    n_backgrounds: int
    n_candidate_features: int
    n_available_features: int
    n_stages: int
    n_rules: int
    stages: list[StageReading]
    held_out: HeldOutReading
    example_target: PictureGrid
    example_background: PictureGrid


def _rule_reading(rule: ThresholdRule, voice: float, window: Rectangle) -> RuleReading:
    feature = rule.feature
    return RuleReading(
        arrangement=ARRANGEMENT_LABELS[feature.arrangement],
        top=feature.top,
        left=feature.left,
        cell_height=feature.cell_height,
        cell_width=feature.cell_width,
        threshold=round(rule.threshold, 6),
        direction=DIRECTION_WORDS[rule.direction],
        voice=round(voice, 4),
        cells=[
            SignedBox(
                top=cell.rectangle.top,
                left=cell.rectangle.left,
                height=cell.rectangle.height,
                width=cell.rectangle.width,
                sign=cell.sign,
            )
            for cell in feature.cells_in(window)
        ],
    )


def _lowest_positive_confidence(
    stage: CascadeStage, positives: list[IntegralImage], window: Rectangle
) -> float:
    return min(stage.confidence_in(one, window) for one in positives)


@cache
def _detector() -> DetectorResponse:
    cascade = _fitted()
    window = _whole_window()
    pictures, labels = _training_set(TRAINING_SEED)
    positives = [
        IntegralImage.of(picture)
        for picture, label in zip(pictures, labels.values, strict=True)
        if label == 1.0
    ]

    stages = []
    for stage, outcome in zip(cascade.stages, cascade.stage_outcomes, strict=True):
        lowest = _lowest_positive_confidence(stage, positives, window)
        stages.append(
            StageReading(
                number=outcome.stage_number,
                n_rules=stage.n_rules,
                total_voice=round(stage.total_voice, 4),
                demanded_share=round(stage.acceptance_share, 4),
                lowest_positive_confidence=round(lowest, 4),
                was_lowered=stage.acceptance_share < 0.5,
                negatives_reaching=outcome.negatives_reaching,
                negatives_rejected=outcome.negatives_rejected,
                negatives_surviving=outcome.negatives_surviving,
                positives_kept=outcome.positives_kept,
                rejection_share=round(outcome.rejection_share, 4),
                rules=[
                    _rule_reading(one.rule, one.voice, window)
                    for one in stage.weighted_rules
                ],
            )
        )

    fresh, fresh_labels = _training_set(HELD_OUT_SEED)
    evaluation = cascade.evaluate(fresh, fresh_labels)
    backgrounds = [
        IntegralImage.of(picture)
        for picture, label in zip(fresh, fresh_labels.values, strict=True)
        if label == 0.0
    ]
    targets = [
        IntegralImage.of(picture)
        for picture, label in zip(fresh, fresh_labels.values, strict=True)
        if label == 1.0
    ]
    background_counts = [len(backgrounds)]
    target_counts = [len(targets)]
    for stage in cascade.stages:
        backgrounds = [one for one in backgrounds if stage.accepts(one, window)]
        targets = [one for one in targets if stage.accepts(one, window)]
        background_counts.append(len(backgrounds))
        target_counts.append(len(targets))

    generator = np.random.default_rng(EXAMPLE_SEED)
    example_target = _target(generator)
    example_background = _background(generator)

    return DetectorResponse(
        window_side=WINDOW_SIDE,
        n_targets=N_TARGETS,
        n_backgrounds=N_BACKGROUNDS,
        n_candidate_features=len(cascade.candidate_features),
        n_available_features=FeatureBank.count_available(WINDOW_SIDE, WINDOW_SIDE),
        n_stages=cascade.n_stages_fitted,
        n_rules=cascade.n_rules,
        stages=stages,
        held_out=HeldOutReading(
            n_backgrounds=background_counts[0],
            backgrounds_through_each_stage=background_counts,
            background_shares=[
                round(background_counts[position + 1] / background_counts[position], 4)
                for position in range(len(background_counts) - 1)
            ],
            n_targets=target_counts[0],
            targets_through_each_stage=target_counts,
            accuracy=round(evaluation.accuracy, 4),
            recall=round(evaluation.recall, 4),
            precision=round(evaluation.precision, 4),
        ),
        example_target=_as_grid(example_target),
        example_background=_as_grid(example_background),
    )


@router.get("/detector", response_model=DetectorResponse)
def detector() -> DetectorResponse:
    """The fitted detector, stage by stage, and what a fresh draw does to it."""
    return _detector()


# --- the sweep -------------------------------------------------------------


class StageReach(BaseModel):
    """How many windows entered one stage, and what they cost it."""

    number: int
    n_rules: int
    windows_reaching: int
    share_of_all_windows: float
    evaluations: int


class ScanResponse(BaseModel):
    """A sweep over the scene, what it found, and what it cost.

    ``survivors`` holds the window positions that reached the second stage and
    the third, in that order, since every window reaches the first.
    ``flat_evaluations`` is what the same five rules would cost applied to
    every window with no ordering at all; ``searched_bank`` and
    ``exhaustive_bank`` are what it would cost to measure every feature the fit
    looked at, and every feature the window admits, at every position.
    """

    scene: PictureGrid
    window_side: int
    planted: list[Position]
    windows_examined: int
    stages: list[StageReach]
    survivors: list[list[Position]]
    accepted: list[Position]
    n_accepted: int
    accepted_near_a_target: int
    rule_evaluations: int
    flat_evaluations: int
    searched_bank: int
    exhaustive_bank: int
    saving_over_flat: float
    saving_over_searched_bank: float
    saving_over_exhaustive_bank: float
    rules_per_window: float


def _near_a_target(scan: CascadeScan) -> int:
    return sum(
        any(
            abs(window.top - top) <= NEAR_ENOUGH
            and abs(window.left - left) <= NEAR_ENOUGH
            for top, left in PLANTED_AT
        )
        for window in scan.accepted
    )


def _survivors(cascade: HaarCascade, picture: Picture) -> list[list[Position]]:
    """Which window positions reached each stage after the first.

    Repeated from the sweep rather than returned by it, because a sweep counts
    windows and this needs to name them so the browser can draw them.
    """
    integral_image = IntegralImage.of(picture)
    height, width = cascade.window_shape
    reached: list[list[Position]] = [[] for _ in cascade.stages[1:]]
    for top in range(picture.height - height + 1):
        for left in range(picture.width - width + 1):
            window = Rectangle(top, left, height, width)
            for position, stage in enumerate(cascade.stages):
                if not stage.accepts(integral_image, window):
                    break
                if position + 1 < len(cascade.stages):
                    reached[position].append(Position(top=top, left=left))
    return reached


@cache
def _scan() -> ScanResponse:
    cascade = _fitted()
    picture = _planted_scene()
    scan = cascade.scan(picture)
    available = FeatureBank.count_available(WINDOW_SIDE, WINDOW_SIDE)
    searched = scan.windows_examined * len(cascade.candidate_features)
    exhaustive = scan.windows_examined * available
    return ScanResponse(
        scene=_as_grid(picture),
        window_side=WINDOW_SIDE,
        planted=[Position(top=top, left=left) for top, left in PLANTED_AT],
        windows_examined=scan.windows_examined,
        stages=[
            StageReach(
                number=position + 1,
                n_rules=stage.n_rules,
                windows_reaching=reaching,
                share_of_all_windows=round(reaching / scan.windows_examined, 4),
                evaluations=stage.n_rules * reaching,
            )
            for position, (stage, reaching) in enumerate(
                zip(cascade.stages, scan.windows_reaching, strict=True)
            )
        ],
        survivors=_survivors(cascade, picture),
        accepted=[
            Position(top=window.top, left=window.left) for window in scan.accepted
        ],
        n_accepted=scan.n_accepted,
        accepted_near_a_target=_near_a_target(scan),
        rule_evaluations=scan.rule_evaluations,
        flat_evaluations=scan.rule_evaluations_without_the_cascade,
        searched_bank=searched,
        exhaustive_bank=exhaustive,
        saving_over_flat=round(scan.saving, 3),
        saving_over_searched_bank=round(searched / scan.rule_evaluations, 1),
        saving_over_exhaustive_bank=round(exhaustive / scan.rule_evaluations, 1),
        rules_per_window=round(scan.rule_evaluations / scan.windows_examined, 4),
    )


@router.get("/scan", response_model=ScanResponse)
def scan() -> ScanResponse:
    """Sweep the scene, and count everything the sweep did."""
    return _scan()


# --- three scenes, one detector --------------------------------------------


class SceneCost(BaseModel):
    """What one scene cost the same detector."""

    label: str
    description: str
    scene: PictureGrid
    windows_examined: int
    windows_reaching: list[int]
    n_accepted: int
    accepted_share: float
    rule_evaluations: int
    flat_evaluations: int
    saving_over_flat: float
    rules_per_window: float


class ScenesResponse(BaseModel):
    """The same detector swept over three scenes that differ in one way.

    ``most_rules_per_window`` is what a window costs when nothing is rejected
    early, which is every rule in the cascade, and is the number the three
    measurements are read against.
    """

    scenes: list[SceneCost]
    most_rules_per_window: int


def _scene_cost(label: str, description: str, picture: Picture) -> SceneCost:
    cascade = _fitted()
    scan_of_it = cascade.scan(picture)
    return SceneCost(
        label=label,
        description=description,
        scene=_as_grid(picture),
        windows_examined=scan_of_it.windows_examined,
        windows_reaching=list(scan_of_it.windows_reaching),
        n_accepted=scan_of_it.n_accepted,
        accepted_share=round(scan_of_it.n_accepted / scan_of_it.windows_examined, 4),
        rule_evaluations=scan_of_it.rule_evaluations,
        flat_evaluations=scan_of_it.rule_evaluations_without_the_cascade,
        saving_over_flat=round(scan_of_it.saving, 3),
        rules_per_window=round(
            scan_of_it.rule_evaluations / scan_of_it.windows_examined, 4
        ),
    )


@cache
def _scenes() -> ScenesResponse:
    return ScenesResponse(
        scenes=[
            _scene_cost(
                "Nothing but ground",
                "The cluttered ground with no target anywhere in it.",
                _clutter_only_scene(),
            ),
            _scene_cost(
                "Three targets hidden in it",
                "The same ground with three targets planted in it.",
                _planted_scene(),
            ),
            _scene_cost(
                "Tiled edge to edge with targets",
                "The same ground covered with targets, so most windows are one.",
                _crowded_scene(),
            ),
        ],
        most_rules_per_window=_fitted().n_rules,
    )


@router.get("/scenes", response_model=ScenesResponse)
def scenes() -> ScenesResponse:
    """The same detector on three scenes, to see what the ordering is worth."""
    return _scenes()


# --- the poses it was never shown ------------------------------------------


class PoseReading(BaseModel):
    """How often the detector finds a target presented one way."""

    label: str
    description: str
    n_shown: int
    n_found: int
    share_found: float
    example: PictureGrid


class SignFlip(BaseModel):
    """The first rule's feature, on one target and on the same one relit."""

    arrangement: str
    on_the_target: float
    on_the_relit_target: float
    threshold: float
    direction: str
    votes_yes_on_the_target: bool
    votes_yes_on_the_relit_target: bool
    cells: list[SignedBox]


class PosesResponse(BaseModel):
    """The same targets shown three ways, and the reading behind the outcome."""

    poses: list[PoseReading]
    sign_flip: SignFlip


def _pose(
    label: str,
    description: str,
    lit_from_the_other_side: bool,
    turned_a_quarter: bool,
) -> PoseReading:
    cascade = _fitted()
    generator = np.random.default_rng(POSE_SEED)
    pictures = [
        _target(
            generator,
            lit_from_the_other_side=lit_from_the_other_side,
            turned_a_quarter=turned_a_quarter,
        )
        for _ in range(N_TARGETS)
    ]
    found = int(np.asarray(cascade.predict(pictures)).sum())
    return PoseReading(
        label=label,
        description=description,
        n_shown=len(pictures),
        n_found=found,
        share_found=round(found / len(pictures), 4),
        example=_as_grid(pictures[0]),
    )


@cache
def _poses() -> PosesResponse:
    cascade = _fitted()
    window = _whole_window()
    first = cascade.stages[0].weighted_rules[0].rule

    generator = np.random.default_rng(31)
    upright = IntegralImage.of(_target(generator))
    generator = np.random.default_rng(31)
    relit = IntegralImage.of(_target(generator, lit_from_the_other_side=True))
    on_target = first.feature.value_in(upright, window)
    on_relit = first.feature.value_in(relit, window)

    return PosesResponse(
        poses=[
            _pose(
                "Upright, as it was trained",
                "Fresh targets drawn the same way as the training ones.",
                False,
                False,
            ),
            _pose(
                "Lit from the other side",
                "The same targets with bright and dark exchanged.",
                True,
                False,
            ),
            _pose(
                "Turned a quarter",
                "The same targets with the band running down instead of across.",
                False,
                True,
            ),
        ],
        sign_flip=SignFlip(
            arrangement=ARRANGEMENT_LABELS[first.feature.arrangement],
            on_the_target=round(on_target, 4),
            on_the_relit_target=round(on_relit, 4),
            threshold=round(first.threshold, 4),
            direction=DIRECTION_WORDS[first.direction],
            votes_yes_on_the_target=bool(first.votes_for_value(on_target)),
            votes_yes_on_the_relit_target=bool(first.votes_for_value(on_relit)),
            cells=[
                SignedBox(
                    top=cell.rectangle.top,
                    left=cell.rectangle.left,
                    height=cell.rectangle.height,
                    width=cell.rectangle.width,
                    sign=cell.sign,
                )
                for cell in first.feature.cells_in(window)
            ],
        ),
    )


@router.get("/poses", response_model=PosesResponse)
def poses() -> PosesResponse:
    """The same targets upright, relit and turned, and how often each is found."""
    return _poses()
