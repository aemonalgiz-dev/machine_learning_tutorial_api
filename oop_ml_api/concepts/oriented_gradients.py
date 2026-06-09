"""Histogram of Oriented Gradients.

Every endpoint here is registered under
``/concepts/histogram-of-oriented-gradients``, and the router is created and
registered before any endpoint exists, so the pages of this section can be
built in parallel without many edits to one shared file.

The page is about describing a patch by which directions its edges point in.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, so that a reader meeting the workbench scene on one page
recognises it on the next. Three small drawn patches are added here rather
than there, because they exist to make one claim of this page checkable by
hand and no other page wants them: a six by six patch holding a single upright
edge, a bright bar on a dark ground together with its brightness-inverted
twin, and a pair of patches whose only difference is where inside one cell a
small square sits.

The linear ramp is the sharpest fixture of the lot. A picture whose brightness
is ``cos(a) * column + sin(a) * row`` has the constant gradient ``(cos a,
sin a)`` at every interior pixel, and exactly that, since the sweep is exact on
a straight ramp. So a cell that touches no border holds one direction, known
to the degree, and the two ways of casting a vote can be compared on it
without any ambiguity about what the right answer was.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a library
refusal needs, and a test asserts that no identifier survives it.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.computer_vision.edges import GradientField
from oop_ml.core.computer_vision.oriented_gradients import (
    CLIP_LIMIT,
    AngleRange,
    BlockNormalisation,
    HistogramOfOrientedGradients,
    OrientedGradientDescription,
    VoteSharing,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.vision_scenes import as_rows, hand_sized, relit, turned, workbench

router = APIRouter(
    prefix="/concepts/histogram-of-oriented-gradients",
    tags=["histogram-of-oriented-gradients"],
)

PictureName = Literal[
    "scene",
    "scene_relit",
    "scene_turned",
    "scene_inverted",
    "upright_edge",
    "bar",
    "bar_inverted",
    "even",
]

BAR_SIDE = 16
"""The side of the bar patch, in pixels.

Two whole cells across at the usual cell side, so the patch holds exactly one
block and its description is a single unit vector, which is what makes the
distance between the bar and its inverted twin readable without dividing by
anything.
"""

SQUARE_SIDE = 3
"""The side of the small square that is moved about inside one cell."""

RAMP_SIDE = 24
"""The side of the linear ramp, so its middle cell touches no border."""

PINNED_BEFORE = 19.0
PINNED_AFTER = 21.0
"""The two ramp angles either side of the boundary between the first two
buckets, which is where the two ways of casting a vote part company."""


def _bar_values(inverted: bool) -> np.ndarray:
    values = np.zeros((BAR_SIDE, BAR_SIDE))
    values[:, 5:11] = 1.0
    return 1.0 - values if inverted else values


def _square_at(row: int, column: int) -> Picture:
    values = np.zeros((BAR_SIDE, BAR_SIDE))
    values[row : row + SQUARE_SIDE, column : column + SQUARE_SIDE] = 1.0
    return Picture(values)


def _ramp(degrees: float) -> Picture:
    rows, columns = np.mgrid[0:RAMP_SIDE, 0:RAMP_SIDE]
    angle = np.deg2rad(degrees)
    return Picture(np.cos(angle) * columns + np.sin(angle) * rows)


def _scene_inverted() -> Picture:
    values = np.asarray(workbench())
    return Picture(values.max() - values)


PICTURES: dict[PictureName, tuple[str, str, int]] = {
    "scene": (
        "the workbench scene",
        (
            "A square, a disc, a diagonal bar, three small crosses and a "
            "left-to-right brightness ramp, drawn by arithmetic so that every "
            "claim on this page can be checked against what is actually there."
        ),
        8,
    ),
    "scene_relit": (
        "the same scene under a different lamp",
        (
            "Every brightness multiplied and then shifted, which is roughly "
            "what a brighter lamp and a second one do to a photograph. Nothing "
            "has moved."
        ),
        8,
    ),
    "scene_turned": (
        "the same scene turned a quarter circle",
        (
            "A quarter turn rather than an arbitrary angle, because every "
            "pixel lands on a pixel and nothing is guessed in between."
        ),
        8,
    ),
    "scene_inverted": (
        "the same scene with its brightness turned inside out",
        (
            "Every bright thing dark and every dark thing bright, so each edge "
            "keeps its line and reverses which side of it is the bright one."
        ),
        8,
    ),
    "upright_edge": (
        "a six by six patch with one upright edge",
        (
            "Three dark columns then three bright ones, small enough that "
            "every number it produces can be worked out on paper."
        ),
        3,
    ),
    "bar": (
        "a bright bar on a dark ground",
        (
            "Two upright edges, and they are the same edge seen from either "
            "side, which is what the choice of range is about."
        ),
        8,
    ),
    "bar_inverted": (
        "a dark bar on a bright ground",
        (
            "The bar with its brightness turned inside out, so both of its "
            "edges reverse and neither of them moves."
        ),
        8,
    ),
    "even": (
        "a patch of even brightness",
        (
            "Every pixel the same, so the brightness is not changing anywhere "
            "and there is no direction to record."
        ),
        8,
    ),
}
"""Each picture's name in plain words, what is in it, and the cell side that
divides it. The label and the description are rendered on the page, so neither
may carry an identifier."""


def picture_for(name: PictureName) -> Picture:
    """The picture that ``name`` stands for."""
    if name == "scene":
        return workbench()
    if name == "scene_relit":
        return relit()
    if name == "scene_turned":
        return turned()
    if name == "scene_inverted":
        return _scene_inverted()
    if name == "upright_edge":
        return hand_sized()
    if name == "bar":
        return Picture(_bar_values(inverted=False))
    if name == "bar_inverted":
        return Picture(_bar_values(inverted=True))
    return Picture(np.full((BAR_SIDE, BAR_SIDE), 0.37))


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, rewritten so no identifier reaches the page.

    The only refusals a reader can provoke here come from a cell side that
    does not divide the patch, or one so large that the patch does not hold a
    whole block. Both sentences are written to be read and carry no class
    names, but the second names the block twice by its side; a test asserts
    that no identifier survives this, so a message that grows one later fails
    there rather than reaching the page.
    """
    return str(refusal)


class DescribeRequest(BaseModel):
    """One picture, and every setting the three stages take."""

    picture: PictureName = "scene"
    cell_side: int = Field(default=8, ge=1, le=24)
    n_buckets: int = Field(default=9, ge=2, le=36)
    angle_range: Literal["unsigned", "signed"] = "unsigned"
    vote_sharing: Literal["whole_to_nearest", "split_between_neighbours"] = (
        "split_between_neighbours"
    )
    block_normalisation: Literal["none", "l1", "l1_square_root", "l2", "l2_clipped"] = (
        "l2_clipped"
    )

    model_config = {"extra": "forbid"}


class DescribeResponse(BaseModel):
    """A picture, its cell histograms, and how long its description came out.

    ``cells`` is the vote weight per cell per bucket, before any block has
    divided anything, which is where the method is readable. ``description``
    carries the finished answer's length and how many numbers it holds.
    ``distance_from_the_scene`` is how far this picture's description sits
    from the upright scene's under the same settings, and is absent for the
    pictures that are not the scene.
    """

    picture: PictureName
    picture_label: str
    picture_description: str
    height: int
    width: int
    n_pixels: int
    pixels: list[list[float]]
    cell_side: int
    n_cell_rows: int
    n_cell_columns: int
    n_buckets: int
    range_span_degrees: float
    bucket_span_degrees: float
    bucket_centre_degrees: list[float]
    signed: bool
    cells: list[list[list[float]]]
    cell_weights: list[list[float]]
    fullest_bucket: list[list[int]]
    total_weight: float
    heaviest_cell_weight: float
    n_values: int
    description_length: float
    largest_entry: float
    entries_above_the_clip_limit: int
    clip_limit: float
    n_block_rows: int
    n_block_columns: int
    cells_in_the_answer: float
    distance_from_the_scene: float | None


def _model(request: DescribeRequest) -> HistogramOfOrientedGradients:
    return HistogramOfOrientedGradients(
        cell_side=request.cell_side,
        n_buckets=request.n_buckets,
        angle_range=AngleRange(request.angle_range),
        vote_sharing=VoteSharing(request.vote_sharing),
        block_normalisation=BlockNormalisation(request.block_normalisation),
    )


def _rounded(values: np.ndarray, places: int = 6) -> list:
    return np.round(np.asarray(values, dtype=np.float64), places).tolist()


@router.post("/describe", response_model=DescribeResponse)
def describe(request: DescribeRequest) -> DescribeResponse:
    """Count every pixel's direction into its own cell, then finish the answer."""
    label, blurb, _ = PICTURES[request.picture]
    picture = picture_for(request.picture)
    model = _model(request)

    try:
        histograms = model.cell_histograms(picture)
        described = model.describe(picture)
    except MLLibError as refusal:
        raise type(refusal)(_in_plain_words(refusal)) from refusal

    counts = histograms.counts
    weights = counts.sum(axis=2)
    span = float(np.rad2deg(histograms.angle_range.span))

    comparable = request.picture in (
        "scene",
        "scene_relit",
        "scene_turned",
        "scene_inverted",
    )
    distance: float | None = None
    if comparable:
        distance = float(described.distance_to(model.describe(workbench())))

    return DescribeResponse(
        picture=request.picture,
        picture_label=label,
        picture_description=blurb,
        height=picture.height,
        width=picture.width,
        n_pixels=picture.n_pixels,
        pixels=as_rows(picture),
        cell_side=request.cell_side,
        n_cell_rows=histograms.n_cell_rows,
        n_cell_columns=histograms.n_cell_columns,
        n_buckets=histograms.n_buckets,
        range_span_degrees=span,
        bucket_span_degrees=round(span / histograms.n_buckets, 6),
        bucket_centre_degrees=[
            round(float(np.rad2deg(histograms.bucket_centre(bucket))), 6)
            for bucket in range(histograms.n_buckets)
        ],
        signed=histograms.angle_range is AngleRange.SIGNED,
        cells=_rounded(counts),
        cell_weights=_rounded(weights),
        fullest_bucket=[
            [
                histograms.fullest_bucket_at(row, column)
                for column in range(histograms.n_cell_columns)
            ]
            for row in range(histograms.n_cell_rows)
        ],
        total_weight=round(histograms.total_weight, 6),
        heaviest_cell_weight=round(float(weights.max()), 6),
        entries_above_the_clip_limit=int((np.asarray(described) > CLIP_LIMIT).sum()),
        clip_limit=CLIP_LIMIT,
        n_values=described.n_values,
        description_length=round(described.length, 12),
        largest_entry=round(float(np.asarray(described).max()), 6),
        n_block_rows=model.n_blocks_along(histograms.n_cell_rows),
        n_block_columns=model.n_blocks_along(histograms.n_cell_columns),
        cells_in_the_answer=described.n_values / histograms.n_buckets,
        distance_from_the_scene=None if distance is None else round(distance, 15),
    )


class GradientReading(BaseModel):
    """One pixel of the hand-sized patch, and what the sweep found there."""

    row: int
    column: int
    brightness: float
    across: float
    down: float
    magnitude: float
    direction_degrees: float


class HandWorkedResponse(BaseModel):
    """The six by six patch taken all the way to its finished description.

    Small enough to check on paper. The sweep answers the same number at every
    pixel of the two columns beside the edge and zero everywhere else, so each
    cell's whole weight is one direction, and the finished answer is one block
    of four cells.
    """

    pixels: list[list[float]]
    readings: list[GradientReading]
    cell_side: int
    n_cell_rows: int
    n_cell_columns: int
    bucket_centre_degrees: list[float]
    cells: list[list[list[float]]]
    cells_whole_vote: list[list[list[float]]]
    total_weight: float
    weight_per_cell: float
    unnormalised: list[float]
    unnormalised_length: float
    normalised: list[float]
    clipped: list[float]
    clip_limit: float
    entries_above_the_limit: int
    n_values: int


@router.get("/hand-worked", response_model=HandWorkedResponse)
def hand_worked() -> HandWorkedResponse:
    """The six by six patch with one upright edge, worked all the way through."""
    picture = hand_sized()
    field = GradientField.of(picture)
    magnitudes = field.magnitude.values
    directions = np.rad2deg(field.direction.values)

    readings = [
        GradientReading(
            row=row,
            column=column,
            brightness=round(float(picture.values[row, column]), 6),
            across=round(float(field.horizontal.values[row, column]), 6),
            down=round(float(field.vertical.values[row, column]), 6),
            magnitude=round(float(magnitudes[row, column]), 6),
            direction_degrees=round(float(directions[row, column]), 6),
        )
        for row in range(picture.height)
        for column in range(picture.width)
    ]

    model = HistogramOfOrientedGradients(cell_side=3)
    histograms = model.cell_histograms(picture)
    whole_vote = HistogramOfOrientedGradients(
        cell_side=3, vote_sharing=VoteSharing.WHOLE_TO_NEAREST
    ).cell_histograms(picture)
    finished = np.asarray(model.describe(picture))
    plain = np.asarray(
        HistogramOfOrientedGradients(
            cell_side=3, block_normalisation=BlockNormalisation.L2
        ).describe(picture)
    )
    raw = np.asarray(
        HistogramOfOrientedGradients(
            cell_side=3, block_normalisation=BlockNormalisation.NONE
        ).describe(picture)
    )

    return HandWorkedResponse(
        pixels=as_rows(picture),
        readings=readings,
        cell_side=3,
        n_cell_rows=histograms.n_cell_rows,
        n_cell_columns=histograms.n_cell_columns,
        bucket_centre_degrees=[
            round(float(np.rad2deg(histograms.bucket_centre(bucket))), 6)
            for bucket in range(histograms.n_buckets)
        ],
        cells=_rounded(histograms.counts),
        cells_whole_vote=_rounded(whole_vote.counts),
        total_weight=round(histograms.total_weight, 6),
        weight_per_cell=round(
            histograms.total_weight
            / (histograms.n_cell_rows * histograms.n_cell_columns),
            6,
        ),
        unnormalised=_rounded(raw),
        unnormalised_length=round(float(np.linalg.norm(raw)), 6),
        normalised=_rounded(plain),
        clipped=_rounded(finished),
        clip_limit=CLIP_LIMIT,
        entries_above_the_limit=int((finished > CLIP_LIMIT).sum()),
        n_values=int(finished.size),
    )


class SharingStep(BaseModel):
    """One ramp angle, and where its cell's weight went under each rule."""

    degrees: float
    whole: list[float]
    shared: list[float]


class SharingResponse(BaseModel):
    """The two ways of casting a vote, swept across a bucket boundary.

    ``steps`` walks a ramp whose gradient is one known direction everywhere,
    and reports the shares an interior cell put in each bucket under each
    rule. ``moved_whole`` and ``moved_shared`` are how far that cell's
    histogram travelled between the two pinned angles, which sit two degrees
    apart on either side of the boundary between the first two buckets.
    """

    bucket_centre_degrees: list[float]
    bucket_span_degrees: float
    steps: list[SharingStep]
    before_degrees: float
    after_degrees: float
    moved_whole: float
    moved_shared: float
    ratio: float
    whole_before: list[float]
    whole_after: list[float]
    shared_before: list[float]
    shared_after: list[float]
    upright_edge_whole: list[float]
    upright_edge_shared: list[float]


def _interior_cell(picture: Picture, sharing: VoteSharing) -> np.ndarray:
    model = HistogramOfOrientedGradients(vote_sharing=sharing)
    return np.asarray(model.cell_histograms(picture).histogram_at(1, 1))


@cache
def _sharing() -> SharingResponse:
    reference = HistogramOfOrientedGradients().cell_histograms(_ramp(10.0))
    centres = [
        round(float(np.rad2deg(reference.bucket_centre(bucket))), 6)
        for bucket in range(reference.n_buckets)
    ]

    steps: list[SharingStep] = []
    for whole_degrees in range(46):
        degrees = float(whole_degrees)
        picture = _ramp(degrees)
        whole = _interior_cell(picture, VoteSharing.WHOLE_TO_NEAREST)
        shared = _interior_cell(picture, VoteSharing.SPLIT_BETWEEN_NEIGHBOURS)
        steps.append(
            SharingStep(
                degrees=degrees,
                whole=_rounded(whole / whole.sum()),
                shared=_rounded(shared / shared.sum()),
            )
        )

    moved: dict[VoteSharing, float] = {}
    histograms: dict[tuple[VoteSharing, float], np.ndarray] = {}
    for sharing in VoteSharing:
        before = _interior_cell(_ramp(PINNED_BEFORE), sharing)
        after = _interior_cell(_ramp(PINNED_AFTER), sharing)
        histograms[(sharing, PINNED_BEFORE)] = before
        histograms[(sharing, PINNED_AFTER)] = after
        moved[sharing] = float(np.linalg.norm(after - before))

    upright = picture_for("scene")
    edge_cell = {
        sharing: np.asarray(
            HistogramOfOrientedGradients(vote_sharing=sharing)
            .cell_histograms(upright)
            .histogram_at(1, 0)
        )
        for sharing in VoteSharing
    }

    whole_moved = moved[VoteSharing.WHOLE_TO_NEAREST]
    shared_moved = moved[VoteSharing.SPLIT_BETWEEN_NEIGHBOURS]

    def shares(values: np.ndarray) -> list[float]:
        return _rounded(values / values.sum())

    return SharingResponse(
        bucket_centre_degrees=centres,
        bucket_span_degrees=round(180.0 / reference.n_buckets, 6),
        steps=steps,
        before_degrees=PINNED_BEFORE,
        after_degrees=PINNED_AFTER,
        moved_whole=round(whole_moved, 6),
        moved_shared=round(shared_moved, 6),
        ratio=whole_moved / shared_moved,
        whole_before=shares(histograms[(VoteSharing.WHOLE_TO_NEAREST, PINNED_BEFORE)]),
        whole_after=shares(histograms[(VoteSharing.WHOLE_TO_NEAREST, PINNED_AFTER)]),
        shared_before=shares(
            histograms[(VoteSharing.SPLIT_BETWEEN_NEIGHBOURS, PINNED_BEFORE)]
        ),
        shared_after=shares(
            histograms[(VoteSharing.SPLIT_BETWEEN_NEIGHBOURS, PINNED_AFTER)]
        ),
        upright_edge_whole=shares(edge_cell[VoteSharing.WHOLE_TO_NEAREST]),
        upright_edge_shared=shares(edge_cell[VoteSharing.SPLIT_BETWEEN_NEIGHBOURS]),
    )


@router.get("/vote-sharing", response_model=SharingResponse)
def vote_sharing() -> SharingResponse:
    """A ramp turned through a bucket boundary, under both ways of voting."""
    return _sharing()


RULE_LABELS: dict[str, str] = {
    "none": "left alone",
    "l1": "divided by the sum of its entries",
    "l1_square_root": "that, and then square-rooted",
    "l2": "divided by its own length",
    "l2_clipped": "divided by its length, cut back, divided again",
}
"""What each way of rescaling a block does, said rather than named."""


class LightingRow(BaseModel):
    """One way of rescaling a block, under one change of lighting."""

    rule: str
    rule_label: str
    description_length: float
    moved_by_the_shift: float
    moved_by_the_scale: float
    moved_by_both: float
    first_block_sum: float
    first_block_length: float


class LightingResponse(BaseModel):
    """What a change of lighting does to the pixels and to the description.

    ``pixel_length`` and ``pixel_moved`` are the scene read as a plain list of
    brightnesses, which is the description this method replaces.
    ``largest_direction_difference_degrees`` and the two magnitude ratios say
    what the change did to the gradients themselves, which is where the two
    halves of a lighting change part company.
    """

    scale: float
    shift: float
    n_pixels: int
    pixel_length: float
    pixel_moved: float
    pixel_share_moved: float
    n_pixels_changed: int
    largest_direction_difference_degrees: float
    smallest_magnitude_ratio: float
    largest_magnitude_ratio: float
    rows: list[LightingRow]


class LightingRequest(BaseModel):
    """How much brighter, and how much added on top."""

    scale: float = Field(default=1.6, gt=0.0, le=20.0)
    shift: float = Field(default=0.15, ge=-20.0, le=20.0)

    model_config = {"extra": "forbid"}


@router.post("/lighting", response_model=LightingResponse)
def lighting(request: LightingRequest) -> LightingResponse:
    """Relight the scene and measure what moved, at each stage of the answer."""
    here = workbench()
    values = np.asarray(here)
    shifted = Picture(values + request.shift)
    scaled = Picture(values * request.scale)
    both = Picture(values * request.scale + request.shift)

    flat_here = values.reshape(-1)
    flat_both = np.asarray(both).reshape(-1)
    pixel_length = float(np.linalg.norm(flat_here))
    pixel_moved = float(np.linalg.norm(flat_here - flat_both))

    field_here = GradientField.of(here)
    field_both = GradientField.of(both)
    direction_gap = float(
        np.abs(field_here.direction.values - field_both.direction.values).max()
    )
    lit = field_here.magnitude.values > 1e-12
    ratios = field_both.magnitude.values[lit] / field_here.magnitude.values[lit]

    rows: list[LightingRow] = []
    for rule in BlockNormalisation:
        model = HistogramOfOrientedGradients(block_normalisation=rule)
        described = model.describe(here)
        first_block = np.asarray(described)[: 4 * model.n_buckets]
        rows.append(
            LightingRow(
                rule=rule.value,
                rule_label=RULE_LABELS[rule.value],
                description_length=round(described.length, 6),
                moved_by_the_shift=float(
                    f"{described.distance_to(model.describe(shifted)):.6g}"
                ),
                moved_by_the_scale=float(
                    f"{described.distance_to(model.describe(scaled)):.6g}"
                ),
                moved_by_both=float(
                    f"{described.distance_to(model.describe(both)):.6g}"
                ),
                first_block_sum=round(float(np.abs(first_block).sum()), 6),
                first_block_length=round(float(np.linalg.norm(first_block)), 6),
            )
        )

    return LightingResponse(
        scale=request.scale,
        shift=request.shift,
        n_pixels=here.n_pixels,
        pixel_length=round(pixel_length, 6),
        pixel_moved=round(pixel_moved, 6),
        pixel_share_moved=round(pixel_moved / pixel_length, 6),
        n_pixels_changed=int(np.count_nonzero(flat_here != flat_both)),
        largest_direction_difference_degrees=float(f"{np.rad2deg(direction_gap):.6g}"),
        smallest_magnitude_ratio=round(float(ratios.min()), 9),
        largest_magnitude_ratio=round(float(ratios.max()), 9),
        rows=rows,
    )


class TurningRow(BaseModel):
    """One bucket count, and how far a turn and a lamp each move the answer."""

    n_buckets: int
    bucket_span_degrees: float
    whole_buckets_in_a_quarter_turn: bool
    n_values: int
    description_length: float
    moved_by_turning: float
    moved_by_relighting: float
    turned_share: float


class TurningRuleRow(BaseModel):
    """One way of rescaling a block, under a quarter turn."""

    rule: str
    rule_label: str
    description_length: float
    moved_by_turning: float
    turned_share: float


class TurningResponse(BaseModel):
    """A quarter turn measured beside a change of lighting.

    ``turned_share`` is how far the description moved as a fraction of its own
    length, so bucket counts whose answers are different lengths can be put on
    one axis.
    """

    rows: list[TurningRow]
    rules: list[TurningRuleRow]


@cache
def _turning() -> TurningResponse:
    here = workbench()
    the_turn = turned()
    the_lamp = relit()

    rows: list[TurningRow] = []
    for n_buckets in (4, 6, 9, 12, 18, 36):
        model = HistogramOfOrientedGradients(n_buckets=n_buckets)
        described = model.describe(here)
        moved = described.distance_to(model.describe(the_turn))
        rows.append(
            TurningRow(
                n_buckets=n_buckets,
                bucket_span_degrees=round(180.0 / n_buckets, 6),
                whole_buckets_in_a_quarter_turn=(90.0 * n_buckets) % 180.0 == 0.0,
                n_values=described.n_values,
                description_length=round(described.length, 6),
                moved_by_turning=round(moved, 6),
                moved_by_relighting=float(
                    f"{described.distance_to(model.describe(the_lamp)):.6g}"
                ),
                turned_share=round(moved / described.length, 6),
            )
        )

    rules: list[TurningRuleRow] = []
    for rule in BlockNormalisation:
        model = HistogramOfOrientedGradients(block_normalisation=rule)
        described = model.describe(here)
        moved = described.distance_to(model.describe(the_turn))
        rules.append(
            TurningRuleRow(
                rule=rule.value,
                rule_label=RULE_LABELS[rule.value],
                description_length=round(described.length, 6),
                moved_by_turning=round(moved, 6),
                turned_share=round(moved / described.length, 6),
            )
        )

    return TurningResponse(rows=rows, rules=rules)


@router.get("/turning", response_model=TurningResponse)
def turning() -> TurningResponse:
    """How far a quarter turn moves the answer, at six bucket counts."""
    return _turning()


class RangeRow(BaseModel):
    """One picture under one range, summed over its cells."""

    picture: PictureName
    picture_label: str
    shares: list[float]
    bucket_centre_degrees: list[float]
    occupied_buckets: list[int]


class RangePair(BaseModel):
    """A picture and its brightness-inverted twin, under one range."""

    angle_range: str
    n_buckets: int
    here: RangeRow
    inverted: RangeRow
    distance: float
    description_length: float


class RangeResponse(BaseModel):
    """The bar and the scene, each beside its own inverted twin.

    Under a half circle a direction and its reverse share a bucket, so
    inverting the brightness cannot be seen at all; under the whole circle it
    is the only thing that can be seen.
    """

    bar: list[RangePair]
    scene: list[RangePair]


def _range_row(name: PictureName, model: HistogramOfOrientedGradients) -> RangeRow:
    histograms = model.cell_histograms(picture_for(name))
    totals = histograms.counts.sum(axis=(0, 1))
    shares = totals / totals.sum()
    return RangeRow(
        picture=name,
        picture_label=PICTURES[name][0],
        shares=_rounded(shares),
        bucket_centre_degrees=[
            round(float(np.rad2deg(histograms.bucket_centre(bucket))), 6)
            for bucket in range(histograms.n_buckets)
        ],
        occupied_buckets=[int(bucket) for bucket in np.flatnonzero(shares > 1e-12)],
    )


def _range_pair(
    here: PictureName, inverted: PictureName, angle_range: AngleRange, n_buckets: int
) -> RangePair:
    model = HistogramOfOrientedGradients(
        angle_range=angle_range,
        n_buckets=n_buckets,
        vote_sharing=VoteSharing.WHOLE_TO_NEAREST,
    )
    described = HistogramOfOrientedGradients(
        angle_range=angle_range, n_buckets=n_buckets
    )
    first: OrientedGradientDescription = described.describe(picture_for(here))
    second = described.describe(picture_for(inverted))
    return RangePair(
        angle_range=angle_range.value,
        n_buckets=n_buckets,
        here=_range_row(here, model),
        inverted=_range_row(inverted, model),
        distance=float(f"{first.distance_to(second):.6g}"),
        description_length=round(first.length, 6),
    )


@cache
def _ranges() -> RangeResponse:
    return RangeResponse(
        bar=[
            _range_pair("bar", "bar_inverted", AngleRange.UNSIGNED, 9),
            _range_pair("bar", "bar_inverted", AngleRange.SIGNED, 18),
        ],
        scene=[
            _range_pair("scene", "scene_inverted", AngleRange.UNSIGNED, 9),
            _range_pair("scene", "scene_inverted", AngleRange.SIGNED, 18),
        ],
    )


@router.get("/range-choice", response_model=RangeResponse)
def range_choice() -> RangeResponse:
    """A bright thing and its opposite, under half a circle and the whole one."""
    return _ranges()


class SizeRequest(BaseModel):
    """A patch, and the arrangement the three stages are laid out in."""

    height: int = Field(default=128, ge=1, le=512)
    width: int = Field(default=64, ge=1, le=512)
    cell_side: int = Field(default=8, ge=1, le=64)
    cells_per_block_side: int = Field(default=2, ge=1, le=8)
    block_stride_in_cells: int = Field(default=1, ge=1, le=8)
    n_buckets: int = Field(default=9, ge=2, le=36)

    model_config = {"extra": "forbid"}


class SizeResponse(BaseModel):
    """How many numbers that arrangement answers, and which cells repeat.

    ``coverage`` counts, for every cell, how many blocks it belongs to and so
    how many times it appears in the answer under a different denominator.
    ``cells_in_the_answer`` is that grid's total, which is how the finished
    length exceeds the number of cells there actually are.
    """

    height: int
    width: int
    n_pixels: int
    cell_side: int
    cells_per_block_side: int
    block_stride_in_cells: int
    n_buckets: int
    n_cell_rows: int
    n_cell_columns: int
    n_cells: int
    n_block_rows: int
    n_block_columns: int
    n_blocks: int
    n_values: int
    cells_in_the_answer: int
    repetition: float
    numbers_per_pixel: float
    coverage: list[list[int]]
    uncovered: int


class ArrangementRefused(MLLibError):
    """An arrangement of cells and blocks that would leave cells undescribed.

    Raised in the page's own words rather than passed through, and of the
    library's own kind so the application's one handler answers it as a 422
    like every other refusal.
    """


@router.post("/size", response_model=SizeResponse)
def size(request: SizeRequest) -> SizeResponse:
    """How long the answer is for that arrangement, and which cells repeat."""
    if request.block_stride_in_cells > request.cells_per_block_side:
        raise ArrangementRefused(
            f"blocks {request.cells_per_block_side} cells across stepping "
            f"{request.block_stride_in_cells} cells at a time leave cells that no "
            f"block covers, so those cells would never reach the answer; a block "
            f"steps at most its own width"
        )
    model = HistogramOfOrientedGradients(
        cell_side=request.cell_side,
        cells_per_block_side=request.cells_per_block_side,
        block_stride_in_cells=request.block_stride_in_cells,
        n_buckets=request.n_buckets,
    )
    try:
        n_values = model.n_values_for(request.height, request.width)
    except MLLibError as refusal:
        raise type(refusal)(_in_plain_words(refusal)) from refusal

    n_cell_rows = model.n_cells_along(request.height)
    n_cell_columns = model.n_cells_along(request.width)
    n_block_rows = model.n_blocks_along(n_cell_rows)
    n_block_columns = model.n_blocks_along(n_cell_columns)

    side = request.cells_per_block_side
    stride = request.block_stride_in_cells
    covered = np.zeros((n_cell_rows, n_cell_columns), dtype=int)
    for top in range(0, n_cell_rows - side + 1, stride):
        for left in range(0, n_cell_columns - side + 1, stride):
            covered[top : top + side, left : left + side] += 1

    n_cells = n_cell_rows * n_cell_columns
    cells_in_the_answer = n_values // request.n_buckets

    return SizeResponse(
        height=request.height,
        width=request.width,
        n_pixels=request.height * request.width,
        cell_side=request.cell_side,
        cells_per_block_side=side,
        block_stride_in_cells=stride,
        n_buckets=request.n_buckets,
        n_cell_rows=n_cell_rows,
        n_cell_columns=n_cell_columns,
        n_cells=n_cells,
        n_block_rows=n_block_rows,
        n_block_columns=n_block_columns,
        n_blocks=n_block_rows * n_block_columns,
        n_values=n_values,
        cells_in_the_answer=cells_in_the_answer,
        repetition=round(cells_in_the_answer / n_cells, 6),
        numbers_per_pixel=round(n_values / (request.height * request.width), 6),
        coverage=covered.tolist(),
        uncovered=int((covered == 0).sum()),
    )


class InsideOneCellResponse(BaseModel):
    """Two patches a cell cannot tell apart, and one it can.

    A small square drawn in three places. The first two sit inside one cell,
    so the counts they produce are identical and the descriptions are exactly
    the same vector, however far apart the pixels are. The third crosses into
    a neighbouring cell, and the description parts company at once.
    """

    here: list[list[float]]
    moved_inside: list[list[float]]
    moved_across: list[list[float]]
    cell_side: int
    description_length: float
    pixels_moved_inside: float
    pixels_moved_across: float
    description_moved_inside: float
    description_moved_across: float
    counts_moved_inside: float


@cache
def _inside_one_cell() -> InsideOneCellResponse:
    model = HistogramOfOrientedGradients()
    here = _square_at(1, 1)
    inside = _square_at(4, 4)
    across = _square_at(1, 9)

    described = model.describe(here)

    def counts_gap(other: Picture) -> float:
        return float(
            np.linalg.norm(
                model.cell_histograms(here).counts - model.cell_histograms(other).counts
            )
        )

    def pixels_gap(other: Picture) -> float:
        return float(np.linalg.norm(np.asarray(here) - np.asarray(other)))

    return InsideOneCellResponse(
        here=as_rows(here),
        moved_inside=as_rows(inside),
        moved_across=as_rows(across),
        cell_side=model.cell_side,
        description_length=round(described.length, 6),
        pixels_moved_inside=round(pixels_gap(inside), 6),
        pixels_moved_across=round(pixels_gap(across), 6),
        description_moved_inside=float(
            f"{described.distance_to(model.describe(inside)):.6g}"
        ),
        description_moved_across=round(
            described.distance_to(model.describe(across)), 6
        ),
        counts_moved_inside=float(f"{counts_gap(inside):.6g}"),
    )


@router.get("/inside-one-cell", response_model=InsideOneCellResponse)
def inside_one_cell() -> InsideOneCellResponse:
    """A square moved inside its cell, and the same square moved out of it."""
    return _inside_one_cell()
