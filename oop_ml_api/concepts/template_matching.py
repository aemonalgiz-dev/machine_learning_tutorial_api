"""Template Matching.

Every endpoint here is registered under ``/concepts/template-matching``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about carrying a known picture across a larger one and scoring every position.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, so that a reader meeting the workbench scene on one page
recognises it on the next.

What the endpoints hand over
----------------------------
Pixels, mostly. The browser draws pictures on this page, which nothing else on
the site does, so a response carries whole grids rather than summary numbers:
the scene, the template and the score surface all travel as nested lists of
brightness, together with the darkest and brightest value in each, because a
greyscale drawing is meaningless until the reader is told what black and white
stand for.

Every score, position, ratio and count is the library's. Nothing here computes
a match; the only arithmetic in this module is arranging what came back into
the shapes the widgets draw, and the two counts of cost, which are the
multiplication the library already reports and this module repeats at picture
sizes too large to search in a web request.

The rules are named by their published names rather than by anything in the
code, because a reader has met "normalised cross-correlation" in a textbook and
has no reason to meet an identifier. ``_PLAIN_RULE_NAMES`` is the one place that
translation happens, and a test pins that no identifier survives it.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.computer_vision.matching import (
    DEFAULT_BELIEVABLE_RATIO,
    HIGHER_IS_BETTER,
    MatchRule,
    MatchScores,
    ScoredPosition,
    TemplateMatcher,
    placed,
    scaled_up,
    score_surface,
    turned_by_a_right_angle,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import UndefinedMetricError
from pydantic import BaseModel

from oop_ml_api.vision_scenes import (
    BACKGROUND,
    MOTIF_AT,
    RAMP_ACROSS,
    SCENE_SIDE,
    as_rows,
    motif,
    relit,
    workbench,
)

router = APIRouter(prefix="/concepts/template-matching", tags=["template-matching"])

SceneName = Literal["workbench", "unlit", "relit"]
"""Which of the three lightings of the one scene a request is asking about.

Three lightings rather than three scenes, so that a difference in the answer is
a difference the lighting made and nothing else.
"""

_SCENE_TITLES: dict[SceneName, str] = {
    "workbench": "The workbench",
    "unlit": "The workbench with the ramp taken away",
    "relit": "The workbench under a brighter lamp",
}

_PLAIN_RULE_NAMES: dict[MatchRule, str] = {
    MatchRule.SUM_OF_SQUARED_DIFFERENCES: "Sum of squared differences",
    MatchRule.CORRELATION: "Correlation",
    MatchRule.NORMALISED_CROSS_CORRELATION: "Normalised cross-correlation",
}
"""Each rule's published name, which is what a reader has met.

The values of the enum are lowercase and underscored, and one of the library's
own refusals opens on one of them. Neither belongs on a page, so the
translation happens here and a test asserts that nothing snake_cased or
CamelCased survives it.
"""

_RULE_ORDER = (
    MatchRule.SUM_OF_SQUARED_DIFFERENCES,
    MatchRule.CORRELATION,
    MatchRule.NORMALISED_CROSS_CORRELATION,
)
"""The order the page introduces them in, which is the order a reader tries them.

The literal reading of "how far apart are these", then the multiplication, then
the repair to the multiplication.
"""

RELIT_GAIN = 1.6
RELIT_OFFSET = 0.15
"""The lighting change the relit scene applies, repeated here so a response can
say what it was rather than leaving a reader to infer it from the pixels."""


class PictureGrid(BaseModel):
    """A picture as brightness values, with what to draw black and white as.

    A greyscale drawing is uninterpretable without the two ends of its scale,
    and the two ends differ from picture to picture here: the scene runs from
    empty ground to the brightest lit shape, a score surface under one rule runs
    over an entirely different range from the same surface under another. So
    every grid carries its own.
    """

    rows: list[list[float]]
    height: int
    width: int
    darkest: float
    brightest: float


def _grid(picture: Picture) -> PictureGrid:
    """One picture packed for drawing."""
    return PictureGrid(
        rows=as_rows(picture),
        height=picture.height,
        width=picture.width,
        darkest=round(picture.darkest, 6),
        brightest=round(picture.brightest, 6),
    )


class Position(BaseModel):
    """A position the template was tried at, and what it scored there."""

    row: int
    column: int
    score: float


def _position(scored: ScoredPosition) -> Position:
    return Position(row=scored.row, column=scored.column, score=round(scored.score, 10))


class Believability(BaseModel):
    """The winner set against the best position that does not overlap it.

    ``ratio`` is at least one whichever direction the rule runs in, and
    ``is_infinite`` is carried separately because a ratio with no second
    candidate to divide by is not a large number, it is the absence of one, and
    a page that printed it as a float would print the word for infinity in
    whatever spelling the transport chose.
    """

    ratio: float
    is_infinite: bool
    believable: bool
    runner_up: Position


def _believability(scores: MatchScores) -> Believability | None:
    """How far the winner stands clear, or nothing where there is no runner-up."""
    try:
        measured = scores.believability()
    except UndefinedMetricError:
        return None
    infinite = not np.isfinite(measured.ratio)
    return Believability(
        ratio=0.0 if infinite else round(measured.ratio, 10),
        is_infinite=infinite,
        believable=measured.is_believable(),
        runner_up=_position(measured.runner_up),
    )


class RuleResult(BaseModel):
    """One rule's whole answer on one picture.

    ``surface`` is every position's score, which is what the page draws beside
    the scene, and ``at_the_motifs`` the score at each of the three places the
    thing being looked for actually sits, in the order they were drawn. The two
    together are what makes the rules comparable: the winner alone says which
    position each rule chose, and the three scores say what each rule thought of
    the right answers.
    """

    rule: str
    higher_is_better: bool
    perfect_score: float
    best: Position
    at_the_motifs: list[Position]
    surface: PictureGrid
    believability: Believability | None


class SearchResponse(BaseModel):
    """One scene searched by all three rules.

    All three rather than one at a time, because every claim the page makes
    about a rule is a comparison with the other two on the same pixels, and
    three separate requests would let the scene drift between them.
    """

    scene_title: str
    scene: PictureGrid
    template: PictureGrid
    motif_positions: list[list[int]]
    n_positions: int
    n_pixels_read: int
    results: list[RuleResult]


class SearchRequest(BaseModel):
    """Which lighting of the shared scene to search."""

    scene: SceneName = "workbench"

    model_config = {"extra": "forbid"}


def _scene(name: SceneName) -> Picture:
    if name == "unlit":
        return workbench(with_ramp=False)
    if name == "relit":
        return relit(scale=RELIT_GAIN, shift=RELIT_OFFSET)
    return workbench()


@cache
def _searched(name: SceneName) -> SearchResponse:
    """The whole scene searched three ways, computed once and kept.

    The three scenes are fixed and so are their answers, and each response
    carries several thousand brightness values, so recomputing them per request
    would be arithmetic nobody asked for twice.
    """
    picture = _scene(name)
    template = motif()
    results = []
    for rule in _RULE_ORDER:
        matcher = TemplateMatcher(template=template, rule=rule)
        scores = matcher.scores_on(picture)
        results.append(
            RuleResult(
                rule=_PLAIN_RULE_NAMES[rule],
                higher_is_better=HIGHER_IS_BETTER[rule],
                perfect_score=round(matcher.score_of_an_exact_copy, 10),
                best=_position(scores.best),
                at_the_motifs=[
                    _position(scores.at(row, column)) for row, column in MOTIF_AT
                ],
                surface=_grid(scores.surface),
                believability=_believability(scores),
            )
        )
    first = MatchScores.of(picture, template, MatchRule.NORMALISED_CROSS_CORRELATION)
    return SearchResponse(
        scene_title=_SCENE_TITLES[name],
        scene=_grid(picture),
        template=_grid(template),
        motif_positions=[[row, column] for row, column in MOTIF_AT],
        n_positions=first.n_positions,
        n_pixels_read=first.n_pixels_read,
        results=results,
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Carry the motif across one lighting of the scene under each of three rules."""
    return _searched(request.scene)


class HandCell(BaseModel):
    """One position of the hand-sized search, under all three rules.

    Three scores per position rather than three surfaces, because the whole
    argument of that example is what the three rules say about the *same*
    position, and a reader checking it on paper works down a column.
    """

    row: int
    column: int
    scores: list[float]


class HandSizedResponse(BaseModel):
    """A search small enough to check with a pencil.

    Six pixels by six, holding one edge, with a three by three piece of itself
    carried across it. Sixteen positions, and every row of the picture is the
    same, so the sixteen scores are four distinct answers repeated four times
    and a reader need only check four numbers.
    """

    picture: PictureGrid
    template: PictureGrid
    template_from: Position
    rules: list[str]
    higher_is_better: list[bool]
    perfect_scores: list[float]
    cells: list[HandCell]
    n_positions: int
    n_pixels_read: int
    best: list[Position]
    correlation_ties: list[Position]


HAND_PICTURE = Picture([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] for _ in range(6)])
"""A six by six picture with one edge in it, dark on the left.

The shared hand-sized picture, repeated here rather than imported so that this
page can state its pixels: three dark columns and three bright ones, so that a
three by three template taken out of it has an exactly checkable score at every
position it can sit at.
"""

HAND_TEMPLATE_AT = (0, 1)
"""Where the hand-sized template is cut from.

Column one rather than column zero, so the template holds the edge itself
rather than three columns of dark, and so it has some variation to compare
with, which the rule that centres each patch requires of a template.
"""


@cache
def _hand_sized() -> HandSizedResponse:
    """The sixteen positions of the hand-sized search, scored three ways."""
    row, column = HAND_TEMPLATE_AT
    template = HAND_PICTURE.patch_at(row, column, 3, 3)
    surfaces = [score_surface(HAND_PICTURE, template, rule) for rule in _RULE_ORDER]
    every = [MatchScores.of(HAND_PICTURE, template, rule) for rule in _RULE_ORDER]
    height, width = surfaces[0].shape
    cells = [
        HandCell(
            row=surface_row,
            column=surface_column,
            scores=[
                round(float(surface.values[surface_row, surface_column]), 10)
                for surface in surfaces
            ],
        )
        for surface_row in range(height)
        for surface_column in range(width)
    ]

    correlation = every[_RULE_ORDER.index(MatchRule.CORRELATION)]
    top = correlation.best.score
    ties = [
        _position(position)
        for position in correlation
        if position.row == 0 and position.score == top
    ]

    return HandSizedResponse(
        picture=_grid(HAND_PICTURE),
        template=_grid(template),
        template_from=Position(row=row, column=column, score=0.0),
        rules=[_PLAIN_RULE_NAMES[rule] for rule in _RULE_ORDER],
        higher_is_better=[HIGHER_IS_BETTER[rule] for rule in _RULE_ORDER],
        perfect_scores=[
            round(
                TemplateMatcher(template=template, rule=rule).score_of_an_exact_copy,
                10,
            )
            for rule in _RULE_ORDER
        ],
        cells=cells,
        n_positions=every[0].n_positions,
        n_pixels_read=every[0].n_pixels_read,
        best=[_position(scores.best) for scores in every],
        correlation_ties=ties,
    )


@router.get("/hand-sized", response_model=HandSizedResponse)
def hand_sized() -> HandSizedResponse:
    """Sixteen positions of a six by six picture, scored by all three rules."""
    return _hand_sized()


GLYPH = Picture(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
"""A capital F, five pixels by five.

The workbench motif is a cross, and a cross is unchanged by a quarter turn, so
it cannot be used to show what a quarter turn costs. This shape is unchanged by
no rotation and no reflection, which is what the claim needs, and it has
twenty-five pixels, which is what the believability claim needs. It is the
shape the matching specs use for both.
"""

CORNER = Picture([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
"""An L. Doubling it turns out to produce an exact copy of it, which is the
whole point of the section it appears in."""

DIM = 0.1
"""Plain ground for the changed-thing pictures. Not zero, so that a patch which
scores well by being bright cannot be dismissed as scoring well against
nothing."""


class ChangedThing(BaseModel):
    """One alteration of the glyph, searched for with the unaltered template.

    ``at_the_truth`` is the score at the position the thing genuinely occupies
    and ``best`` the position reported instead, so the pair says both how badly
    the right answer scored and how confidently a wrong one was given.
    """

    change: str
    scene: PictureGrid
    template: PictureGrid
    drawn_at: list[int]
    at_the_truth: Position
    best: Position
    pixels_away: float
    surface: PictureGrid


class ChangedThingResponse(BaseModel):
    """An unaltered copy as the control, then a turned one, then a larger one."""

    cases: list[ChangedThing]
    doubled_corner: ChangedThing
    doubled_corner_contains_the_original: bool
    doubled_corner_copy_at: list[int]
    doubled_corner_ratio: float
    doubled_corner_believable: bool


def _changed(
    change: str, drawn: Picture, template: Picture, ground: Picture, at: tuple[int, int]
) -> ChangedThing:
    """Draw one altered thing into plain ground and search for the original."""
    scene = placed(ground, drawn, at[0], at[1])
    scores = MatchScores.of(scene, template, MatchRule.NORMALISED_CROSS_CORRELATION)
    truth = scores.at(at[0], at[1])
    return ChangedThing(
        change=change,
        scene=_grid(scene),
        template=_grid(template),
        drawn_at=[at[0], at[1]],
        at_the_truth=_position(truth),
        best=_position(scores.best),
        pixels_away=round(scores.best.pixels_away_from(truth), 10),
        surface=_grid(scores.surface),
    )


@cache
def _changed_thing() -> ChangedThingResponse:
    """The same glyph unaltered, turned three ways, and doubled."""
    plain_15 = Picture(np.full((15, 15), DIM))
    plain_20 = Picture(np.full((20, 20), DIM))
    plain_12 = Picture(np.full((12, 12), DIM))

    cases = [_changed("Unaltered", GLYPH, GLYPH, plain_15, (4, 4))]
    for turns, wording in (
        (1, "Turned a quarter"),
        (2, "Turned a half"),
        (3, "Turned three quarters"),
    ):
        cases.append(
            _changed(
                wording, turned_by_a_right_angle(GLYPH, turns), GLYPH, plain_15, (4, 4)
            )
        )
    cases.append(
        _changed("Doubled in size", scaled_up(GLYPH, 2), GLYPH, plain_20, (4, 4))
    )

    corner = _changed("Doubled in size", scaled_up(CORNER, 2), CORNER, plain_12, (3, 3))
    doubled = scaled_up(CORNER, 2)
    scene = placed(plain_12, doubled, 3, 3)
    scores = MatchScores.of(scene, CORNER, MatchRule.NORMALISED_CROSS_CORRELATION)
    measured = scores.believability()

    return ChangedThingResponse(
        cases=cases,
        doubled_corner=corner,
        doubled_corner_contains_the_original=doubled.patch_at(2, 1, 3, 3) == CORNER,
        doubled_corner_copy_at=[2, 1],
        doubled_corner_ratio=round(measured.ratio, 10),
        doubled_corner_believable=measured.is_believable(),
    )


@router.get("/changed-thing", response_model=ChangedThingResponse)
def changed_thing() -> ChangedThingResponse:
    """What a turned or enlarged copy scores, and what is reported instead."""
    return _changed_thing()


class SeedRatios(BaseModel):
    """One texture, searched with the thing drawn into it and without."""

    seed: int
    absent_ratio: float
    absent_best_score: float
    present_ratio: float
    present_best_score: float


class SizeRow(BaseModel):
    """One template size, and how easily a picture matches it by chance.

    ``mean_chance_agreement`` is the best score a template of this size reaches
    in twenty pictures that do not contain it, which is the number that decides
    whether any threshold can work. ``present_below_threshold`` counts how often
    a genuine match nonetheless failed to stand clear, and is empty where the
    size leaves no room for a second candidate at all.
    """

    side: int
    mean_chance_agreement: float
    highest_chance_agreement: float
    absent_low: float | None
    absent_high: float | None
    present_low: float | None
    present_high: float | None
    present_below_threshold: int | None
    n_seeds: int
    has_a_second_candidate: bool


class BelievabilityResponse(BaseModel):
    """Present against absent, and what template size does to the separation."""

    threshold: float
    template_side: int
    texture_side: int
    seeds: list[SeedRatios]
    absent_low: float
    absent_high: float
    present_low: float
    present_high: float
    absent_score_low: float
    absent_score_high: float
    sizes: list[SizeRow]
    n_chance_seeds: int


def _texture(seed: int, side: int = 15) -> Picture:
    """A picture with no particular thing in it, drawn the same way every time."""
    return Picture(np.random.default_rng(seed).random((side, side)))


TEXTURE_SIDE = 15
N_RATIO_SEEDS = 6
N_CHANCE_SEEDS = 20
"""How the believability measurement is run.

Fifteen pixels a side is the smallest picture with room for a five by five
template and a second candidate that does not overlap the first. Six seeds for
the ratio, which is enough to see two ranges that do not touch, and twenty for
the chance agreement, which is an average and wants more.
"""


@cache
def _believability_report() -> BelievabilityResponse:
    """The ratio present and absent, and the same experiment at three sizes."""
    matcher = TemplateMatcher(template=GLYPH)
    seeds = []
    for seed in range(N_RATIO_SEEDS):
        background = _texture(seed, TEXTURE_SIDE)
        absent = matcher.scores_on(background)
        present = matcher.scores_on(placed(background, GLYPH, 4, 4))
        seeds.append(
            SeedRatios(
                seed=seed,
                absent_ratio=round(absent.believability().ratio, 10),
                absent_best_score=round(absent.best.score, 10),
                present_ratio=round(present.believability().ratio, 10),
                present_best_score=round(present.best.score, 10),
            )
        )

    sizes = []
    for side in (3, 5, 7):
        thing = Picture(np.random.default_rng(99).random((side, side)))
        sized = TemplateMatcher(template=thing)
        bests = [
            sized.scores_on(_texture(seed, TEXTURE_SIDE)).best.score
            for seed in range(N_CHANCE_SEEDS)
        ]
        absent_ratios: list[float] = []
        present_ratios: list[float] = []
        for seed in range(N_RATIO_SEEDS):
            background = _texture(seed, TEXTURE_SIDE)
            try:
                absent_ratios.append(sized.scores_on(background).believability().ratio)
                present_ratios.append(
                    sized.scores_on(placed(background, thing, 4, 4))
                    .believability()
                    .ratio
                )
            except UndefinedMetricError:
                absent_ratios = []
                present_ratios = []
                break
        has_second = bool(present_ratios)
        sizes.append(
            SizeRow(
                side=side,
                mean_chance_agreement=round(float(np.mean(bests)), 10),
                highest_chance_agreement=round(max(bests), 10),
                absent_low=round(min(absent_ratios), 10) if has_second else None,
                absent_high=round(max(absent_ratios), 10) if has_second else None,
                present_low=round(min(present_ratios), 10) if has_second else None,
                present_high=round(max(present_ratios), 10) if has_second else None,
                present_below_threshold=(
                    sum(1 for r in present_ratios if r < DEFAULT_BELIEVABLE_RATIO)
                    if has_second
                    else None
                ),
                n_seeds=N_RATIO_SEEDS,
                has_a_second_candidate=has_second,
            )
        )

    absent_ratios_all = [seed.absent_ratio for seed in seeds]
    present_ratios_all = [seed.present_ratio for seed in seeds]
    absent_scores = [seed.absent_best_score for seed in seeds]
    return BelievabilityResponse(
        threshold=DEFAULT_BELIEVABLE_RATIO,
        template_side=GLYPH.height,
        texture_side=TEXTURE_SIDE,
        seeds=seeds,
        absent_low=min(absent_ratios_all),
        absent_high=max(absent_ratios_all),
        present_low=min(present_ratios_all),
        present_high=max(present_ratios_all),
        absent_score_low=min(absent_scores),
        absent_score_high=max(absent_scores),
        sizes=sizes,
        n_chance_seeds=N_CHANCE_SEEDS,
    )


@router.get("/believability", response_model=BelievabilityResponse)
def believability() -> BelievabilityResponse:
    """How far the winner stands clear, with the target present and absent."""
    return _believability_report()


class CostRow(BaseModel):
    """One picture and template size, and what searching it costs."""

    picture_side: int
    template_side: int
    n_positions: int
    n_pixels_read: int


class CostResponse(BaseModel):
    """The two counts at four sizes, and what searching several angles adds.

    The first two rows are searches this page actually runs, so their counts
    come back from the search itself rather than from a formula. The larger two
    are the multiplication carried out at a size no teaching request should
    perform, which is the whole point of quoting them.
    """

    rows: list[CostRow]
    n_angles: int
    n_scales: int
    largest_row: CostRow
    n_pixels_read_over_angles_and_scales: int


ANGLES_TRIED = 12
SCALES_TRIED = 5
"""A modest search over orientation and size.

Twelve angles is one every thirty degrees and five scales is a little under a
factor of two either way, both of which are coarse. The product of the two is
what the page multiplies the single-pass count by.
"""


@cache
def _cost() -> CostResponse:
    """What the four searches cost, the small two measured and the large two counted."""
    rows = []
    for picture, template in (
        (HAND_PICTURE, HAND_PICTURE.patch_at(0, 1, 3, 3)),
        (workbench(), motif()),
    ):
        scores = MatchScores.of(
            picture, template, MatchRule.NORMALISED_CROSS_CORRELATION
        )
        rows.append(
            CostRow(
                picture_side=picture.height,
                template_side=template.height,
                n_positions=scores.n_positions,
                n_pixels_read=scores.n_pixels_read,
            )
        )
    for picture_side, template_side in ((512, 64), (1024, 64)):
        positions = (picture_side - template_side + 1) ** 2
        rows.append(
            CostRow(
                picture_side=picture_side,
                template_side=template_side,
                n_positions=positions,
                n_pixels_read=positions * template_side * template_side,
            )
        )
    largest = rows[-2]
    return CostResponse(
        rows=rows,
        n_angles=ANGLES_TRIED,
        n_scales=SCALES_TRIED,
        largest_row=largest,
        n_pixels_read_over_angles_and_scales=(
            largest.n_pixels_read * ANGLES_TRIED * SCALES_TRIED
        ),
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """How many positions a search tries and how many pixels it reads."""
    return _cost()


class UndefinedResponse(BaseModel):
    """Where the normalised rule's quotient stops meaning anything.

    Three near-flat situations that are easy to run together, and one piece of
    geometry that has nothing to do with flatness. A patch with no variation at
    all leaves the template fine and this one position unscorable, and gets a
    stated convention. A patch that varies only by rounding gets the same answer
    for a reason worth knowing. A patch that varies by an amount too small to
    mean anything gets a real number, in full confidence, which is the one of
    the three with no protection at all.

    The geometry is separate. ``furthest_reach`` is how far the winner of the
    named search lies from the most distant position in it, and where that is
    smaller than the template's own side there is no position that fails to
    overlap the winner, so there is no second candidate to be compared against.
    """

    flat_patch_at_25_pixels: float
    flat_patch_at_9_pixels: float
    barely_varying_patch_spread: float
    barely_varying_patch_score: float
    no_second_candidate_template_side: int
    no_second_candidate_picture_side: int
    no_second_candidate_surface_side: int
    furthest_reach: int
    second_candidate_exists: bool


@cache
def _undefined() -> UndefinedResponse:
    """The three near-flat cases, and the geometry that leaves no runner-up."""
    cross = Picture([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    flat_25 = score_surface(
        Picture(np.full((15, 15), DIM)), GLYPH, MatchRule.NORMALISED_CROSS_CORRELATION
    )
    flat_9 = score_surface(
        Picture(np.full((9, 9), DIM)), cross, MatchRule.NORMALISED_CROSS_CORRELATION
    )

    barely = np.full((9, 9), DIM)
    barely[3:6, 3:6] = DIM + np.asarray(cross) * 1e-17
    barely_picture = Picture(barely)
    barely_surface = score_surface(
        barely_picture, cross, MatchRule.NORMALISED_CROSS_CORRELATION
    )
    barely_patch = barely_picture.patch_at(3, 3, 3, 3)

    template_side, picture_side = 7, 15
    crowded = Picture(np.random.default_rng(99).random((template_side, template_side)))
    scores = MatchScores.of(
        placed(_texture(0, picture_side), crowded, 4, 4),
        crowded,
        MatchRule.NORMALISED_CROSS_CORRELATION,
    )
    surface_side = scores.shape[0]
    winner = scores.best
    reach = max(
        winner.row,
        surface_side - 1 - winner.row,
        winner.column,
        surface_side - 1 - winner.column,
    )
    return UndefinedResponse(
        flat_patch_at_25_pixels=float(flat_25.values[0, 0]),
        flat_patch_at_9_pixels=float(flat_9.values[0, 0]),
        barely_varying_patch_spread=barely_patch.brightest - barely_patch.darkest,
        barely_varying_patch_score=round(float(barely_surface.values[3, 3]), 10),
        no_second_candidate_template_side=template_side,
        no_second_candidate_picture_side=picture_side,
        no_second_candidate_surface_side=surface_side,
        furthest_reach=reach,
        second_candidate_exists=_believability(scores) is not None,
    )


@router.get("/undefined", response_model=UndefinedResponse)
def undefined() -> UndefinedResponse:
    """What the quotient does where the patch has almost nothing to divide by."""
    return _undefined()


class RampResponse(BaseModel):
    """What the brightness ramp is, and what it does to a perfect match.

    The scene's three copies of the motif are identical drawings sitting under
    different amounts of light, so the ramp is the whole reason the three rules
    disagree about which of them is best. These are the numbers behind that.
    """

    ramp_across: float
    background: float
    shape_brightness: float
    scene_side: int
    motif_side: int
    patch_means: list[float]
    patch_sums: list[float]
    normalised_scores: list[float]
    scores_are_identical: bool
    ramp_part_length: float
    template_part_length: float
    ramp_dotted_with_template: float
    predicted_cosine: float
    relit_gain: float
    relit_offset: float
    largest_score_change_when_relit: float
    template_sum: float
    template_sum_of_squares: float
    decoy_at: list[int]
    decoy_darkest: float
    decoy_brightest: float
    decoy_mean: float
    decoy_correlation: float
    decoy_correlation_if_it_were_flat: float
    middle_copy_correlation: float


@cache
def _ramp() -> RampResponse:
    """The three copies, their brightness, and why they score alike."""
    scene = workbench()
    template = motif()
    values = np.asarray(scene)
    template_values = np.asarray(template)
    centred_template = template_values - template_values.mean()

    patch = values[
        MOTIF_AT[0][0] : MOTIF_AT[0][0] + template.height,
        MOTIF_AT[0][1] : MOTIF_AT[0][1] + template.width,
    ]
    ramp_part = patch - template_values
    centred_ramp = ramp_part - ramp_part.mean()
    ramp_length = float(np.sqrt((centred_ramp**2).sum()))
    template_length = float(np.sqrt((centred_template**2).sum()))

    scores = MatchScores.of(scene, template, MatchRule.NORMALISED_CROSS_CORRELATION)
    at_motifs = [scores.at(row, column).score for row, column in MOTIF_AT]

    lit = MatchScores.of(
        relit(scale=RELIT_GAIN, shift=RELIT_OFFSET),
        template,
        MatchRule.NORMALISED_CROSS_CORRELATION,
    )
    change = float(np.abs(np.asarray(scores.surface) - np.asarray(lit.surface)).max())

    multiplied = MatchScores.of(scene, template, MatchRule.CORRELATION)
    decoy = multiplied.best
    decoy_patch = scene.patch_at(
        decoy.row, decoy.column, template.height, template.width
    )
    decoy_mean = float(np.asarray(decoy_patch).mean())
    template_sum = float(template_values.sum())

    return RampResponse(
        ramp_across=RAMP_ACROSS,
        background=BACKGROUND,
        shape_brightness=template.brightest,
        template_sum=round(template_sum, 10),
        template_sum_of_squares=round(float((template_values**2).sum()), 10),
        decoy_at=[decoy.row, decoy.column],
        decoy_darkest=round(decoy_patch.darkest, 10),
        decoy_brightest=round(decoy_patch.brightest, 10),
        decoy_mean=round(decoy_mean, 10),
        decoy_correlation=round(decoy.score, 10),
        decoy_correlation_if_it_were_flat=round(decoy_mean * template_sum, 10),
        middle_copy_correlation=round(
            multiplied.at(MOTIF_AT[0][0], MOTIF_AT[0][1]).score, 10
        ),
        scene_side=SCENE_SIDE,
        motif_side=template.height,
        patch_means=[
            round(
                float(
                    values[
                        row : row + template.height, column : column + template.width
                    ].mean()
                ),
                10,
            )
            for row, column in MOTIF_AT
        ],
        patch_sums=[
            round(
                float(
                    values[
                        row : row + template.height, column : column + template.width
                    ].sum()
                ),
                10,
            )
            for row, column in MOTIF_AT
        ],
        normalised_scores=[round(score, 16) for score in at_motifs],
        scores_are_identical=len(set(at_motifs)) == 1,
        ramp_part_length=round(ramp_length, 10),
        template_part_length=round(template_length, 10),
        ramp_dotted_with_template=float((centred_ramp * centred_template).sum()),
        predicted_cosine=round(
            float(1.0 / np.sqrt(1.0 + (ramp_length / template_length) ** 2)), 16
        ),
        relit_gain=RELIT_GAIN,
        relit_offset=RELIT_OFFSET,
        largest_score_change_when_relit=change,
    )


@router.get("/ramp", response_model=RampResponse)
def ramp() -> RampResponse:
    """What the lighting across the scene does to three identical drawings."""
    return _ramp()
