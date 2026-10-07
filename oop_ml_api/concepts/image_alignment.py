"""Image Alignment.

Every endpoint here is registered under ``/concepts/image-alignment``, and the
router is created and registered before any endpoint exists, so the pages of
this section can be built in parallel without many edits to one shared file.

The page is about lining two pictures of one scene up: giving each keypoint a
direction of its own so that its description survives a turn, matching the
descriptions, and reading the camera's move off the matches.

The pictures come from :mod:`oop_ml_api.vision_scenes`, which every classical
vision page shares, placed in a larger frame so that the shapes near the edge
of the scene still have room for a descriptor's patch. The second picture of
each pair is the first carried through a known move, which is what lets the
page say which matches were right and not only which ones the method kept:
the method's inliers and outliers are its own verdict, and the truth sits
beside it.

One thing is computed here rather than in the library, and it is deliberate.
The RANSAC endpoint replays every pair of matches and reports what each pair
proposed and how many matches agreed, which the library does on the way to
its answer and does not keep. The page needs the whole distribution to show
why the winning pair wins, so the replay is written here with the library's
own transform fit and its own distance test, and the API test pins that the
winner it finds is the library's.
"""

from __future__ import annotations

import math
from functools import cache
from itertools import combinations
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.computer_vision.alignment import (
    AlignmentEstimate,
    BinaryDescriptor,
    Correspondences,
    DescriptorSet,
    DominantGradientRule,
    GradientHistogramDescriptor,
    IntensityCentroidRule,
    OrientationRule,
    OrientedKeypoint,
    OrientedKeypoints,
    Pyramid,
    SamplingPattern,
    SimilarityTransform,
    mean_absolute_difference,
    oriented_keypoints_of,
)
from oop_ml.core.computer_vision.edges import GradientField
from oop_ml.core.computer_vision.keypoints import Keypoint, PatchDescriptor
from oop_ml.core.computer_vision.matching import placed
from oop_ml.core.computer_vision.picture import Picture
from pydantic import BaseModel, Field

from oop_ml_api.vision_scenes import as_rows, workbench

router = APIRouter(prefix="/concepts/image-alignment", tags=["image-alignment"])

FRAME = 64
MARGIN = 8
"""The workbench scene sits in a 64 by 64 frame with eight pixels to spare.

A descriptor's patch, once turned to a keypoint's direction, reaches about
eleven pixels from it, so a keypoint on the scene's outermost shapes needs
room beyond the scene to be described at all.
"""

BORDER = 12
"""How close to the frame a keypoint may be reported. Twelve covers the
sixteen-wide histogram square at any angle, and the fifteen-wide pattern."""

LIMIT = 40
"""The strongest keypoints kept per picture, so the two lists are comparable."""

BAR_SIZE = 49
BAR_CORNERS = ((14, 14), (14, 34), (20, 23), (19, 34), (34, 14), (34, 22))
"""The bent bar the page works by hand, the library's own spec fixture.

An L rather than a square, because a square looks the same turned and a
descriptor that merely found a corner would seem to have found the corner.
Six corners, all at least twelve pixels from the frame.
"""

DEFAULT_ANGLE = 30.0
DEFAULT_ROW_SHIFT = 3.0
DEFAULT_COLUMN_SHIFT = -5.0
"""The playground's opening move: a thirty-degree turn and a small shift,
which lands no pixel on a pixel, so nothing about it is exact."""

MethodName = Literal["orb", "sift"]


def _scene() -> Picture:
    return placed(Picture(np.zeros((FRAME, FRAME))), workbench(), MARGIN, MARGIN)


def _bent_bar() -> Picture:
    values = np.zeros((BAR_SIZE, BAR_SIZE))
    values[14:35, 14:23] = 1.0
    values[14:20, 14:35] = 1.0
    return Picture(values)


def _about_the_centre(
    scale: float, degrees: float, row_shift: float, column_shift: float
) -> SimilarityTransform:
    """A move that turns and scales about the middle of the frame, then shifts."""
    angle = math.radians(degrees)
    centre = np.asarray([(FRAME - 1) / 2.0, (FRAME - 1) / 2.0])
    turn = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    shift = centre - scale * (turn @ centre) + np.asarray([row_shift, column_shift])
    return SimilarityTransform(scale, angle, float(shift[0]), float(shift[1]))


def _rule(method: MethodName) -> OrientationRule:
    return IntensityCentroidRule() if method == "orb" else DominantGradientRule()


def _described(
    method: MethodName, source: Pyramid | Picture, keypoints: OrientedKeypoints
) -> DescriptorSet:
    if method == "orb":
        return DescriptorSet.binary(source, keypoints)
    return DescriptorSet.gradient_histograms(source, keypoints)


def _degrees(angle: float) -> float:
    return round(math.degrees(angle), 2)


class KeypointReport(BaseModel):
    """Where a keypoint is, which way it faces, and how strongly it scored."""

    row: int
    column: int
    strength: float
    angle_degrees: float
    scale: float


class MoveReport(BaseModel):
    """The four numbers of a similarity, with the angle in degrees."""

    scale: float
    angle_degrees: float
    row_shift: float
    column_shift: float


def _reported(keypoints: OrientedKeypoints) -> list[KeypointReport]:
    return [
        KeypointReport(
            row=point.row,
            column=point.column,
            strength=round(point.strength, 6),
            angle_degrees=_degrees(point.angle),
            scale=round(point.scale, 4),
        )
        for point in keypoints
    ]


def _move(transform: SimilarityTransform) -> MoveReport:
    return MoveReport(
        scale=round(transform.scale, 4),
        angle_degrees=_degrees(transform.angle),
        row_shift=round(transform.row_shift, 3),
        column_shift=round(transform.column_shift, 3),
    )


class MatchReport(BaseModel):
    """One match: which keypoints, how it scored, and two verdicts on it.

    ``inlier`` is the method's verdict, whether the recovered move explains
    the match. ``right`` is the truth, whether the planted move does. The two
    disagree on a match the method refused although it was right, and on one
    it accepted although it was wrong, which is what the page counts.
    """

    first: int
    second: int
    distance: float
    ratio: float
    inlier: bool
    right: bool


class AlignRequest(BaseModel):
    """Which move to plant, and how to recover it.

    The move is the thing a reader changes: how far the second picture is
    turned, scaled and shifted. The method is which direction rule and which
    descriptor, the ORB pair or the SIFT pair. The ratio and the inlier
    distance are the two thresholds the method leaves open.
    """

    method: MethodName = "orb"
    angle_degrees: float = Field(default=DEFAULT_ANGLE, ge=-180.0, le=180.0)
    scale: float = Field(default=1.0, ge=0.6, le=1.6)
    row_shift: float = Field(default=DEFAULT_ROW_SHIFT, ge=-16.0, le=16.0)
    column_shift: float = Field(default=DEFAULT_COLUMN_SHIFT, ge=-16.0, le=16.0)
    maximum_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    inlier_distance: float = Field(default=1.5, gt=0.0, le=6.0)
    limit: int = Field(default=LIMIT, ge=4, le=80)

    model_config = {"extra": "forbid"}


class AlignResponse(BaseModel):
    """Both pictures, their keypoints, the matches and the recovered move.

    ``aligned`` is the second picture carried back by the recovered move, and
    the two differences are the mean pixel gap between it and the first
    picture before and after, which is the one number that says whether the
    alignment worked without reading a single match.
    """

    height: int
    width: int
    first: list[list[float]]
    second: list[list[float]]
    aligned: list[list[float]]
    first_keypoints: list[KeypointReport]
    second_keypoints: list[KeypointReport]
    matches: list[MatchReport]
    n_matches: int
    n_right: int
    n_inliers: int
    n_outliers: int
    inliers_right: int
    inliers_wrong: int
    outliers_right: int
    outliers_wrong: int
    truth: MoveReport
    estimate: MoveReport | None
    mean_residual: float | None
    angle_gap_degrees: float | None
    difference_before: float
    difference_after: float | None
    refusal: str | None


def _is_right(
    match_first: OrientedKeypoint,
    match_second: OrientedKeypoint,
    truth: SimilarityTransform,
    tolerance: float,
) -> bool:
    landed = truth.applied_to(match_first.position)[0]
    return bool(np.linalg.norm(landed - match_second.position) <= tolerance)


@router.post("/align", response_model=AlignResponse)
def align(request: AlignRequest) -> AlignResponse:
    """Plant a move, find it again, and say how well it was found."""
    first = _scene()
    truth = _about_the_centre(
        request.scale, request.angle_degrees, request.row_shift, request.column_shift
    )
    second = truth.warp(first)
    rule = _rule(request.method)
    before = oriented_keypoints_of(first, rule, border=BORDER, limit=request.limit)
    after = oriented_keypoints_of(second, rule, border=BORDER, limit=request.limit)
    # Keyed by direction as well as position, since the gradient rule can give
    # one position two directions and so two keypoints.
    index_before = {(k.row, k.column, k.angle): index for index, k in enumerate(before)}
    index_after = {(k.row, k.column, k.angle): index for index, k in enumerate(after)}

    matches = Correspondences([])
    refusal: str | None = None
    if len(before) >= 1 and len(after) >= 2:
        matches = _described(request.method, first, before).matched_to(
            _described(request.method, second, after), request.maximum_ratio
        )
    else:
        refusal = (
            "too few keypoints were found to match; the ratio test needs at "
            "least two in the second picture"
        )

    estimate: AlignmentEstimate | None = None
    if refusal is None:
        if len(matches) >= 2:
            try:
                estimate = AlignmentEstimate.of(matches, request.inlier_distance)
            except Exception as error:  # noqa: BLE001
                refusal = str(error)
        else:
            refusal = (
                f"{len(matches)} match(es) survived the ratio test, and two are "
                f"needed to fix a move"
            )

    inlier_keys = {(m.first, m.second) for m in estimate.inliers} if estimate else set()
    reports = [
        MatchReport(
            first=index_before[(m.first.row, m.first.column, m.first.angle)],
            second=index_after[(m.second.row, m.second.column, m.second.angle)],
            distance=round(m.distance, 4),
            ratio=round(m.ratio, 4),
            inlier=(m.first, m.second) in inlier_keys,
            right=_is_right(m.first, m.second, truth, request.inlier_distance),
        )
        for m in matches
    ]
    aligned = estimate.aligned(second) if estimate else None
    return AlignResponse(
        height=FRAME,
        width=FRAME,
        first=as_rows(first),
        second=as_rows(second),
        aligned=as_rows(aligned) if aligned else [],
        first_keypoints=_reported(before),
        second_keypoints=_reported(after),
        matches=reports,
        n_matches=len(reports),
        n_right=sum(r.right for r in reports),
        n_inliers=sum(r.inlier for r in reports),
        n_outliers=sum(not r.inlier for r in reports),
        inliers_right=sum(r.inlier and r.right for r in reports),
        inliers_wrong=sum(r.inlier and not r.right for r in reports),
        outliers_right=sum((not r.inlier) and r.right for r in reports),
        outliers_wrong=sum((not r.inlier) and not r.right for r in reports),
        truth=_move(truth),
        estimate=_move(estimate.transform) if estimate else None,
        mean_residual=round(estimate.mean_residual, 4) if estimate else None,
        angle_gap_degrees=(
            round(
                math.degrees(
                    abs(
                        (
                            (estimate.transform.angle - truth.angle + math.pi)
                            % (2 * math.pi)
                        )
                        - math.pi
                    )
                ),
                2,
            )
            if estimate
            else None
        ),
        difference_before=round(mean_absolute_difference(first, second), 4),
        difference_after=(
            round(mean_absolute_difference(first, aligned), 4) if aligned else None
        ),
        refusal=refusal,
    )


class TurnCostRow(BaseModel):
    """One corner of the bar against its quarter-turned self, three ways."""

    row: int
    column: int
    turned_row: int
    turned_column: int
    patch_distance: float
    patch_nearest_other: float
    binary_bits: float
    binary_nearest_other: float
    histogram_distance: float
    histogram_nearest_other: float


class TurnCostResponse(BaseModel):
    """What a quarter turn costs each kind of description.

    The patch descriptor is the keypoints page's, read in the picture's
    frame; the two oriented descriptors are read in the keypoint's. Each
    corner is compared with itself after the turn and with the nearest other
    corner, so a reader sees both halves of what matters: that the right pair
    is close, and that the wrong pairs are not.
    """

    size: int
    bar: list[list[float]]
    turned: list[list[float]]
    rows: list[TurnCostRow]
    patch_side: int
    pair_count: int


@cache
def _turn_cost() -> TurnCostResponse:
    bar = _bent_bar()
    turned = Picture(np.rot90(bar.values))
    rule = IntensityCentroidRule()
    before = oriented_keypoints_of(bar, rule, border=BORDER)
    after = oriented_keypoints_of(turned, rule, border=BORDER)
    where = {(k.row, k.column): k for k in after}
    patch_side = 5
    patches_before = [PatchDescriptor.of(bar, k.keypoint, patch_side) for k in before]
    patches_after = {
        (k.row, k.column): PatchDescriptor.of(turned, k.keypoint, patch_side)
        for k in after
    }
    binary_before = [BinaryDescriptor.of(bar, k) for k in before]
    binary_after = {(k.row, k.column): BinaryDescriptor.of(turned, k) for k in after}
    histogram_before = [GradientHistogramDescriptor.of(bar, k) for k in before]
    histogram_after = {
        (k.row, k.column): GradientHistogramDescriptor.of(turned, k) for k in after
    }
    rows = []
    for index, keypoint in enumerate(before):
        twin = (BAR_SIZE - 1 - keypoint.column, keypoint.row)
        others = [key for key in where if key != twin]
        rows.append(
            TurnCostRow(
                row=keypoint.row,
                column=keypoint.column,
                turned_row=twin[0],
                turned_column=twin[1],
                patch_distance=round(
                    patches_before[index].distance_to(patches_after[twin]), 6
                ),
                patch_nearest_other=round(
                    min(
                        patches_before[index].distance_to(patches_after[key])
                        for key in others
                    ),
                    6,
                ),
                binary_bits=binary_before[index].distance_to(binary_after[twin]),
                binary_nearest_other=min(
                    binary_before[index].distance_to(binary_after[key])
                    for key in others
                ),
                histogram_distance=round(
                    histogram_before[index].distance_to(histogram_after[twin]), 6
                ),
                histogram_nearest_other=round(
                    min(
                        histogram_before[index].distance_to(histogram_after[key])
                        for key in others
                    ),
                    6,
                ),
            )
        )
    return TurnCostResponse(
        size=BAR_SIZE,
        bar=as_rows(bar),
        turned=as_rows(turned),
        rows=rows,
        patch_side=patch_side,
        pair_count=SamplingPattern().pair_count,
    )


@router.get("/turn-cost", response_model=TurnCostResponse)
def turn_cost() -> TurnCostResponse:
    """A quarter turn, paid for by three descriptors."""
    return _turn_cost()


class OrientedCorner(BaseModel):
    """One corner's patch, and the direction each rule reads off it."""

    row: int
    column: int
    patch: list[list[float]]
    moment_right: float
    moment_down: float
    centroid_angle_degrees: float
    gradient_angle_degrees: float
    votes: list[float]
    fullest_bin: int


class OrientationResponse(BaseModel):
    """Every corner of the bar given a direction, two ways, with the workings."""

    size: int
    bar: list[list[float]]
    radius: int
    bins: int
    corners: list[OrientedCorner]


@cache
def _orientation() -> OrientationResponse:
    bar = _bent_bar()
    centroid = IntensityCentroidRule()
    gradient = DominantGradientRule()
    radius = centroid.radius
    corners = []
    for row, column in BAR_CORNERS:
        keypoint = Keypoint(row, column, 1.0)
        patch = bar.values[
            row - radius : row + radius + 1, column - radius : column + radius + 1
        ]
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
        inside = rows * rows + columns * columns <= radius * radius
        weighted = np.where(inside, patch, 0.0)
        field = GradientField.of(Picture(patch))
        bell = np.exp(-(rows * rows + columns * columns) / (2.0 * (radius / 2.0) ** 2))
        weights = np.where(inside, field.magnitude.values * bell, 0.0)
        which = (
            np.floor(
                (field.direction.values + np.pi) / (2 * np.pi) * gradient.bins
            ).astype(int)
            % gradient.bins
        )
        votes = np.bincount(
            which.ravel(), weights=weights.ravel(), minlength=gradient.bins
        )
        corners.append(
            OrientedCorner(
                row=row,
                column=column,
                patch=[[round(float(v), 4) for v in line] for line in patch],
                moment_right=round(float(np.sum(columns * weighted)), 4),
                moment_down=round(float(np.sum(rows * weighted)), 4),
                centroid_angle_degrees=_degrees(centroid.angle_at(bar, keypoint)),
                gradient_angle_degrees=_degrees(gradient.angle_at(bar, keypoint)),
                votes=[round(float(v), 4) for v in votes],
                fullest_bin=int(np.argmax(votes)),
            )
        )
    return OrientationResponse(
        size=BAR_SIZE,
        bar=as_rows(bar),
        radius=radius,
        bins=gradient.bins,
        corners=corners,
    )


@router.get("/orientation", response_model=OrientationResponse)
def orientation() -> OrientationResponse:
    """The direction of every corner of the bar, with the arithmetic shown."""
    return _orientation()


class PairOffsets(BaseModel):
    """One pair of the binary pattern, as offsets from the keypoint."""

    first_row: int
    first_column: int
    second_row: int
    second_column: int


class DescribedCorner(BaseModel):
    """A corner's two readings, with what each read."""

    row: int
    column: int
    angle_degrees: float
    pairs: list[PairOffsets]
    bits: list[bool]
    histogram: list[float]


class DescribeResponse(BaseModel):
    """The top-left corner of the bar, described before and after a quarter turn.

    The pairs are the pattern turned to each keypoint's own direction, so a
    reader can see that the second reading samples the turned picture at the
    turned positions, which is why the bits come out the same.
    """

    size: int
    bar: list[list[float]]
    turned: list[list[float]]
    before: DescribedCorner
    after: DescribedCorner
    hamming: float
    histogram_distance: float
    cells: int
    cell_side: int
    direction_bins: int
    pair_count: int
    patch_side: int


def _described_corner(picture: Picture, keypoint: OrientedKeypoint) -> DescribedCorner:
    pattern = SamplingPattern()
    first, second = pattern.turned_by(keypoint.angle)
    binary = BinaryDescriptor.of(picture, keypoint, pattern)
    histogram = GradientHistogramDescriptor.of(picture, keypoint)
    return DescribedCorner(
        row=keypoint.row,
        column=keypoint.column,
        angle_degrees=_degrees(keypoint.angle),
        pairs=[
            PairOffsets(
                first_row=int(a[0]),
                first_column=int(a[1]),
                second_row=int(b[0]),
                second_column=int(b[1]),
            )
            for a, b in zip(first, second, strict=True)
        ],
        bits=[bool(bit) for bit in binary.bits],
        histogram=[round(float(v), 4) for v in histogram.values],
    )


@cache
def _describe() -> DescribeResponse:
    bar = _bent_bar()
    turned = Picture(np.rot90(bar.values))
    rule = IntensityCentroidRule()
    corner = (14, 14)
    twin = (BAR_SIZE - 1 - corner[1], corner[0])
    before = next(
        k
        for k in oriented_keypoints_of(bar, rule, border=BORDER)
        if (k.row, k.column) == corner
    )
    after = next(
        k
        for k in oriented_keypoints_of(turned, rule, border=BORDER)
        if (k.row, k.column) == twin
    )
    described_before = _described_corner(bar, before)
    described_after = _described_corner(turned, after)
    pattern = SamplingPattern()
    return DescribeResponse(
        size=BAR_SIZE,
        bar=as_rows(bar),
        turned=as_rows(turned),
        before=described_before,
        after=described_after,
        hamming=BinaryDescriptor.of(bar, before).distance_to(
            BinaryDescriptor.of(turned, after)
        ),
        histogram_distance=round(
            GradientHistogramDescriptor.of(bar, before).distance_to(
                GradientHistogramDescriptor.of(turned, after)
            ),
            6,
        ),
        cells=4,
        cell_side=4,
        direction_bins=8,
        pair_count=pattern.pair_count,
        patch_side=pattern.patch_side,
    )


@router.get("/describe", response_model=DescribeResponse)
def describe() -> DescribeResponse:
    """One corner read in its own frame, before and after a quarter turn."""
    return _describe()


class Proposal(BaseModel):
    """What one pair of matches proposed, and how many matches agreed."""

    first_match: int
    second_match: int
    n_agreeing: int
    angle_degrees: float
    scale: float
    both_right: bool


class RansacResponse(BaseModel):
    """Every pair of matches tried on the opening move, and what each proposed.

    Most pairs explain only themselves. The pairs made of two right matches
    propose nearly the same move and explain nearly every right match, which
    is why the winner is found by counting rather than by averaging.
    """

    n_matches: int
    n_pairs: int
    proposals: list[Proposal]
    winner: int
    agreement_counts: list[int]
    truth: MoveReport
    estimate: MoveReport


@cache
def _ransac() -> RansacResponse:
    first = _scene()
    truth = _about_the_centre(
        1.0, DEFAULT_ANGLE, DEFAULT_ROW_SHIFT, DEFAULT_COLUMN_SHIFT
    )
    second = truth.warp(first)
    rule = _rule("orb")
    before = oriented_keypoints_of(first, rule, border=BORDER, limit=LIMIT)
    after = oriented_keypoints_of(second, rule, border=BORDER, limit=LIMIT)
    matches = _described("orb", first, before).matched_to(
        _described("orb", second, after)
    )
    estimate = AlignmentEstimate.of(matches)
    firsts = matches.first_positions
    seconds = matches.second_positions
    right = [_is_right(m.first, m.second, truth, 1.5) for m in matches]
    proposals: list[Proposal] = []
    best: tuple[int, float, int] | None = None
    for one, two in combinations(range(len(matches)), 2):
        if np.array_equal(firsts[one], firsts[two]) or np.array_equal(
            seconds[one], seconds[two]
        ):
            continue
        proposal = SimilarityTransform.from_pairs(
            firsts[[one, two]], seconds[[one, two]]
        )
        errors = np.linalg.norm(proposal.applied_to(firsts) - seconds, axis=1)
        agreeing = errors <= 1.5
        count = int(np.count_nonzero(agreeing))
        total = float(np.sum(errors[agreeing]))
        if best is None or count > best[0] or (count == best[0] and total < best[1]):
            best = (count, total, len(proposals))
        proposals.append(
            Proposal(
                first_match=one,
                second_match=two,
                n_agreeing=count,
                angle_degrees=_degrees(proposal.angle),
                scale=round(proposal.scale, 4),
                both_right=right[one] and right[two],
            )
        )
    counts = np.bincount([p.n_agreeing for p in proposals], minlength=len(matches) + 1)
    return RansacResponse(
        n_matches=len(matches),
        n_pairs=len(proposals),
        proposals=proposals,
        winner=best[2] if best else 0,
        agreement_counts=[int(c) for c in counts],
        truth=_move(truth),
        estimate=_move(estimate.transform),
    )


@router.get("/ransac", response_model=RansacResponse)
def ransac() -> RansacResponse:
    """Every pair of matches as a proposal, and the agreement each won."""
    return _ransac()


class PyramidRow(BaseModel):
    """One way of detecting, and what it did for matching the opening move."""

    levels: int
    factor: float
    n_first_keypoints: int
    n_second_keypoints: int
    n_matches: int
    n_right: int
    n_inliers: int
    recovered: bool
    estimate: MoveReport | None


class PyramidResponse(BaseModel):
    """Whether more sizes mean more right matches, measured on the opening move."""

    truth: MoveReport
    rows: list[PyramidRow]


@cache
def _pyramid() -> PyramidResponse:
    first = _scene()
    truth = _about_the_centre(
        1.0, DEFAULT_ANGLE, DEFAULT_ROW_SHIFT, DEFAULT_COLUMN_SHIFT
    )
    second = truth.warp(first)
    rule = _rule("orb")
    rows = []
    for levels, factor in ((1, 2.0), (3, 2.0), (6, 1.2)):
        one, two = Pyramid(first, levels, factor), Pyramid(second, levels, factor)
        before = oriented_keypoints_of(one, rule, border=BORDER, limit=LIMIT)
        after = oriented_keypoints_of(two, rule, border=BORDER, limit=LIMIT)
        matches = _described("orb", one, before).matched_to(
            _described("orb", two, after)
        )
        n_right = sum(_is_right(m.first, m.second, truth, 1.5) for m in matches)
        estimate = AlignmentEstimate.of(matches) if len(matches) >= 2 else None
        recovered = bool(
            estimate
            and abs(estimate.transform.scale - 1.0) < 0.05
            and abs(
                ((estimate.transform.angle - truth.angle + math.pi) % (2 * math.pi))
                - math.pi
            )
            < 0.05
        )
        rows.append(
            PyramidRow(
                levels=levels,
                factor=factor,
                n_first_keypoints=len(before),
                n_second_keypoints=len(after),
                n_matches=len(matches),
                n_right=n_right,
                n_inliers=estimate.n_inliers if estimate else 0,
                recovered=recovered,
                estimate=_move(estimate.transform) if estimate else None,
            )
        )
    return PyramidResponse(truth=_move(truth), rows=rows)


@router.get("/pyramid", response_model=PyramidResponse)
def pyramid() -> PyramidResponse:
    """One size against several, on the opening move."""
    return _pyramid()
