"""Centring on the mean, the half of standardising that is not about scale.

Every endpoint here is registered under ``/concepts/centring-on-the-mean``, and the
router is created and registered before any endpoint exists, so the pages of this
section can be built in parallel without many edits to one shared file.

The page is about subtracting the mean alone, which methods require it, and what
goes silently wrong when it is skipped. Each of those is measured rather than
asserted, on the crowd the neighbouring pages already use, eleven people by
height in centimetres and weight in kilograms.

Three endpoints serve it.

``/origin`` is the playground. It takes any list of people and a point to measure
them from, and reports what two origin-sensitive methods make of them there: the
longest direction of the rows measured from that origin, which is what a
principal component search finds when nothing centred the rows first, beside the
library's own principal component fit on the rows ``MeanCentrer`` centred; and a
least-squares line forced through that origin, beside the line with an
intercept. The uncentred direction is the leading eigenvector of the
second-moment matrix, computed here with numpy, because the library's principal
component fit always centres and has no reason to answer the uncentred question.
Every fit is the library's.

``/crowd`` is fixed and cached, and holds everything else the page quotes about
the crowd: the centring itself, the origin sweep, the lines and ridges, two
gradient walks with their checkpoints, the polynomial terms, the angles between
people, the methods that never notice, and a mean learned on one fold. The walk
checkpoints are separate fits capped at each pass count; a walk starts from zero
every time and takes the same steps, so the fit capped at ``k`` passes is where
the full walk stood after ``k``.

``/clock`` centres a list of times of day both ways, by the arithmetic mean the
scaler learns and by the circular mean, which is a definition computed here since
the library has no circular quantities at all. It exists for the page's last
part, where a mean stops being a meaningful place.

Refusals are rewritten by ``_in_plain_words`` on the way out, because the website
shows them to a reader who has no reason to meet a parameter name, and a test
pins that no identifier survives the rewriting.
"""

from __future__ import annotations

import math
import re
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DecisionNode,
    DecisionTreeClassifier,
    Feature,
    GradientDescentRegression,
    KMeans,
    KNearestNeighboursClassifier,
    MeanCentrer,
    MultipleLinearRegression,
    PolynomialFeatures,
    PrincipalComponentAnalysis,
    RidgeRegression,
    Standardizer,
)
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS

router = APIRouter(
    prefix="/concepts/centring-on-the-mean", tags=["centring-on-the-mean"]
)

CROWD_HEIGHTS: tuple[float, ...] = (
    147.0,
    156.0,
    145.0,
    159.0,
    162.0,
    120.0,
    122.0,
    118.0,
    180.0,
    183.0,
    178.0,
)
"""The eleven heights the neighbouring pages share, in centimetres."""

CROWD_WEIGHTS: tuple[float, ...] = (
    41.0,
    53.0,
    57.0,
    57.0,
    61.0,
    25.0,
    28.0,
    24.0,
    80.0,
    83.0,
    78.0,
)
"""Their weights, in kilograms."""

CROWD_LABELS: tuple[float, ...] = (0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1)
"""Child as 0 and adult as 1, as the classification pages have them."""

ORIGIN_SWEEP: tuple[float, ...] = (0.0, 50.0, 100.0, 118.0, 150.0)
"""Heights to measure the crowd from, the last two at its shortest and its middle."""

RIDGE_PENALTIES: tuple[float, ...] = (1.0, 100.0)
"""Two penalties, one gentle and one heavy, for the intercept comparison."""

WALK_PASS_LIMIT = 20_000
"""The cap the feature scaling page also uses, so the two pages' walks agree."""

WALK_CHECKPOINTS: tuple[int, ...] = (
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10_000,
    20_000,
)
"""Pass counts at which to read where each walk stood."""

HELD_OUT = (8, 9, 10)
"""The three tallest adults, held out of the fold the mean is learned on."""

VOTING_NEIGHBOURS = 3
"""How many people vote in the leave-one-out comparison of the two angles."""

HEIGHT = "height"
WEIGHT = "weight"


# ---------------------------------------------------------------------------
# Refusals, in the page's own words.
# ---------------------------------------------------------------------------

PLAIN_WORDS = (
    ("feature_values", "a column’s values"),
    ("total variance must be positive; got 0.0", "the people do not vary at all"),
)
"""Identifiers a reader has no reason to meet, and what to say instead."""

IDENTIFIER_PATTERN = re.compile(
    r"\b[a-z]+(?:_[a-z]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"
)
"""A snake_case or CamelCase word, which is a name from the code."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with every identifier taken out of it.

    The known ones are replaced by what they mean, and anything left that still
    looks like a name from the code is replaced by a neutral word, so a message
    that grows an identifier later still reaches the page as a sentence.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return IDENTIFIER_PATTERN.sub("a value", sentence)


def _refuse(message: str) -> HTTPException:
    """A typed 422 carrying a sentence the page may show as it is."""
    return HTTPException(status_code=422, detail=message)


# ---------------------------------------------------------------------------
# Shared arithmetic.
# ---------------------------------------------------------------------------


class Arrow(BaseModel):
    """A unit direction in the height-weight plane, and its angle in degrees."""

    dx: float
    dy: float
    angle_degrees: float


def _arrow(direction: np.ndarray) -> Arrow:
    """A direction as an arrow, its angle measured from the height axis."""
    return Arrow(
        dx=float(direction[0]),
        dy=float(direction[1]),
        angle_degrees=float(math.degrees(math.atan2(direction[1], direction[0]))),
    )


def _oriented(direction: np.ndarray, towards: np.ndarray | None) -> np.ndarray:
    """The same axis, signed to point along ``towards`` or else rightwards.

    A direction and its negative are one axis, so the sign is a convention; this
    one makes two arrows comparable by the cosine between them.
    """
    reference = towards if towards is not None else np.array([1.0, 0.0])
    if float(direction @ reference) < 0.0:
        return -direction
    if float(direction @ reference) == 0.0 and direction[0] < 0.0:
        return -direction
    return direction


def _centred(
    heights: np.ndarray, weights: np.ndarray
) -> tuple[MeanCentrer, np.ndarray, np.ndarray]:
    """Fit the scaler on both columns and hand back what it made of them."""
    features = [Feature(HEIGHT, heights), Feature(WEIGHT, weights)]
    centrer = MeanCentrer().fit(features)
    centred_height, centred_weight = centrer.transform(features)
    return (
        centrer,
        np.asarray(centred_height.values, dtype=np.float64),
        np.asarray(centred_weight.values, dtype=np.float64),
    )


def _uncentred_direction(rows: np.ndarray) -> tuple[np.ndarray, float]:
    """The longest direction of the rows measured from their origin, and its share.

    The leading eigenvector of the second-moment matrix, the mean of the outer
    products of the rows themselves, which is what a covariance matrix becomes
    when nothing subtracted the mean first. The share is its eigenvalue over the
    trace, the part of the rows' total squared length it accounts for.
    """
    second_moment = rows.T @ rows / len(rows)
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    total = float(np.sum(eigenvalues))
    share = float(eigenvalues[-1] / total) if total > 0.0 else 0.0
    return eigenvectors[:, -1], share


def _first_component(
    centred_height: np.ndarray, centred_weight: np.ndarray
) -> tuple[np.ndarray, float]:
    """The library's principal component fit on centred columns: direction, share."""
    fitted = PrincipalComponentAnalysis().fit(
        [Feature(HEIGHT, centred_height), Feature(WEIGHT, centred_weight)]
    )
    first = next(iter(fitted.components))
    return (
        np.asarray(first.direction, dtype=np.float64),
        float(fitted.components.variance_shares[0]),
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    """The cosine of the angle between two vectors, by its definition."""
    return float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))


def _check_varies(heights: np.ndarray, weights: np.ndarray) -> None:
    """Refuse a list whose heights or weights are all one value."""
    if np.ptp(heights) == 0.0:
        raise _refuse(
            "every height in the list is the same, so there is no line to fit "
            "through them and no direction they spread along"
        )
    if np.ptp(weights) == 0.0:
        raise _refuse(
            "every weight in the list is the same, so the people spread along "
            "height alone and a line has nothing to explain"
        )


# ---------------------------------------------------------------------------
# The playground: what two origin-sensitive methods make of one origin.
# ---------------------------------------------------------------------------


class OriginRequest(BaseModel):
    """The people, and the height and weight to measure them from.

    An origin of zero and zero is the usual one, a person of no height and no
    weight. Moving it is the same as recording every height as the distance
    above some mark on the wall, which changes no gap between two people.
    """

    points: list[PlainPoint] = Field(min_length=3, max_length=MAX_POINTS)
    height_origin: float = Field(default=0.0, ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    weight_origin: float = Field(default=0.0, ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)

    model_config = {"extra": "forbid"}


class DirectionReport(BaseModel):
    """One longest direction and the share of the total it claims.

    ``share`` is of the total squared length about the origin for the uncentred
    direction, and of the total variance for the centred one, which is why the
    first can look better while answering a different question.
    """

    arrow: Arrow
    share: float


class LineReport(BaseModel):
    """A least-squares line of weight on height, in the measured units.

    ``intercept`` is the weight the line gives at a measured height of zero,
    which is exactly zero for the line forced through the origin.
    """

    slope: float
    intercept: float
    r_squared: float


class OriginResponse(BaseModel):
    """What the rows look like from one origin, and what they look like centred.

    ``centre`` is what the scaler learned, the mean person measured from the
    chosen origin. ``mean_direction`` is the arrow from the origin to that
    person, absent when the origin is the mean person, since then there is no
    such arrow. ``second_moment_total`` is the mean squared length of the rows
    about the origin, which splits exactly into ``variance_total``, the spread
    about the mean, and ``location_total``, the squared length of the mean.
    ``refit_gap`` is how far the principal component fit moved when handed the
    raw rows rather than the centred ones, since that fit centres for itself.
    """

    origin: PlainPoint
    centre: PlainPoint
    uncentred: DirectionReport
    centred: DirectionReport
    mean_direction: Arrow | None
    uncentred_against_mean: float | None
    uncentred_against_centred: float
    second_moment_total: float
    variance_total: float
    location_total: float
    refit_gap: float
    through_origin: LineReport
    with_intercept: LineReport
    centred_through_origin: LineReport


def _line(
    model: MultipleLinearRegression, height: Feature, weight: Feature
) -> LineReport:
    """Read one fitted line back as slope, intercept and score."""
    return LineReport(
        slope=float(model.coefficients[HEIGHT]),
        intercept=float(model.intercept),
        r_squared=float(model.score([height], weight)),
    )


@router.post("/origin", response_model=OriginResponse)
def origin(request: OriginRequest) -> OriginResponse:
    """Measure the people from one origin and report both methods both ways."""
    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    weights = np.asarray([point.y for point in request.points], dtype=np.float64)
    _check_varies(heights, weights)

    measured_heights = heights - request.height_origin
    measured_weights = weights - request.weight_origin
    rows = np.column_stack([measured_heights, measured_weights])

    try:
        centrer, centred_height, centred_weight = _centred(
            measured_heights, measured_weights
        )
        centre = np.array(
            [centrer.scalings[HEIGHT].centre, centrer.scalings[WEIGHT].centre]
        )
        component, component_share = _first_component(centred_height, centred_weight)
        raw_fit = PrincipalComponentAnalysis().fit(
            [Feature(HEIGHT, measured_heights), Feature(WEIGHT, measured_weights)]
        )
        raw_component = np.asarray(
            next(iter(raw_fit.components)).direction, dtype=np.float64
        )

        height = Feature(HEIGHT, measured_heights)
        weight = Feature(WEIGHT, measured_weights)
        through = MultipleLinearRegression(fit_intercept=False).fit([height], weight)
        full = MultipleLinearRegression().fit([height], weight)
        centred_height_feature = Feature(HEIGHT, centred_height)
        centred_weight_feature = Feature(WEIGHT, centred_weight)
        centred_through = MultipleLinearRegression(fit_intercept=False).fit(
            [centred_height_feature], centred_weight_feature
        )
    except MLLibError as refusal:
        raise _refuse(_in_plain_words(refusal)) from refusal

    location = float(centre @ centre)
    mean_arrow = None if location == 0.0 else centre / math.sqrt(location)
    uncentred, uncentred_share = _uncentred_direction(rows)
    uncentred = _oriented(uncentred, mean_arrow)
    component = _oriented(component, None)
    raw_component = _oriented(raw_component, component)

    return OriginResponse(
        origin=PlainPoint(x=request.height_origin, y=request.weight_origin),
        centre=PlainPoint(x=float(centre[0]), y=float(centre[1])),
        uncentred=DirectionReport(arrow=_arrow(uncentred), share=uncentred_share),
        centred=DirectionReport(arrow=_arrow(component), share=component_share),
        mean_direction=None if mean_arrow is None else _arrow(mean_arrow),
        uncentred_against_mean=None
        if mean_arrow is None
        else _cosine(uncentred, mean_arrow),
        uncentred_against_centred=_cosine(uncentred, component),
        second_moment_total=float(np.sum(rows**2) / len(rows)),
        variance_total=float(np.var(measured_heights) + np.var(measured_weights)),
        location_total=location,
        refit_gap=float(np.max(np.abs(raw_component - component))),
        through_origin=_line(through, height, weight),
        with_intercept=_line(full, height, weight),
        centred_through_origin=_line(
            centred_through, centred_height_feature, centred_weight_feature
        ),
    )


# ---------------------------------------------------------------------------
# The crowd, measured once for every other widget on the page.
# ---------------------------------------------------------------------------


class CentringReport(BaseModel):
    """The crowd's two columns before and after the scaler.

    ``height_mean_after`` and ``weight_mean_after`` are what the centred columns
    average to in floating point, which is zero in the mathematics and a few
    units in the last place here. ``largest_gap_change`` is the largest change
    in the gap between any two people's heights, and ``standardized_gap`` the
    largest difference between standardising the raw columns and standardising
    the centred ones.
    """

    height_centre: float
    weight_centre: float
    height_spread: float
    centred_heights: list[float]
    centred_weights: list[float]
    height_mean_after: float
    weight_mean_after: float
    height_variance_before: float
    height_variance_after: float
    largest_gap_change: float
    standardized_gap: float


class SweepRow(BaseModel):
    """The two longest directions with height measured from one mark."""

    height_origin: float
    uncentred_angle: float
    uncentred_share: float
    centred_angle: float


class RidgeRow(BaseModel):
    """One penalty, fitted three ways.

    ``slope_with_intercept`` has an unpenalised intercept, the way the library's
    ridge always fits one. ``slope_centred_through_origin`` is the same penalty
    on centred columns with no intercept at all. ``slope_raw_through_origin`` is
    the raw columns with no intercept, the far end of penalising an intercept,
    where it is held at zero whatever it would cost.
    """

    penalty: float
    slope_with_intercept: float
    intercept: float
    slope_centred_through_origin: float
    slope_raw_through_origin: float
    r_squared_raw_through_origin: float


class LinesReport(BaseModel):
    """Weight on height, with and without an intercept, raw and centred."""

    least_squares: LineReport
    through_origin: LineReport
    centred_through_origin: LineReport
    centred_with_intercept: LineReport
    ridges: list[RidgeRow]


class WalkCheckpoint(BaseModel):
    """Where a walk stood after some number of passes."""

    passes: int
    slope: float
    level: float


class WalkReport(BaseModel):
    """One gradient walk to the least-squares line, on raw or centred heights.

    ``curvatures`` are the two eigenvalues of the design's mean outer product,
    smaller first, the bowl's curvature along its two natural directions up to
    a factor of two. ``ones_cosine`` is the cosine between the column of ones the
    intercept multiplies and the height column, which is what couples the two.
    ``learning_rate`` is half the divergence threshold, the rate the feature
    scaling page gives every walk.
    """

    reading: Literal["raw", "centred"]
    curvatures: list[float]
    condition_number: float
    ones_cosine: float
    learning_rate: float
    passes: int
    converged: bool
    slope: float
    level: float
    target_slope: float
    target_level: float
    checkpoints: list[WalkCheckpoint]


class PolynomialRow(BaseModel):
    """A height column and its square, fitted to weight with an intercept."""

    reading: Literal["raw", "centred"]
    correlation: float
    condition_number: float
    linear_coefficient: float
    square_coefficient: float
    intercept: float
    r_squared: float


class PairCosine(BaseModel):
    """The pair of people with the smallest cosine between them, and its value."""

    first: int
    second: int
    value: float


class VoteRow(BaseModel):
    """Leave-one-out accuracy of a three-person vote under one rule of nearness."""

    rule: Literal["straight-line", "angle"]
    raw_accuracy: float
    centred_accuracy: float


class AnglesReport(BaseModel):
    """Every cosine between two people, measured from zero and from the mean.

    ``column_cosine_raw`` is the cosine between the height and weight columns
    themselves, ``column_cosine_centred`` the same after centring, and
    ``correlation`` their correlation, which the second equals.
    """

    raw_cosines: list[list[float]]
    centred_cosines: list[list[float]]
    smallest_raw: PairCosine
    smallest_centred: PairCosine
    column_cosine_raw: float
    column_cosine_centred: float
    correlation: float
    votes: list[VoteRow]


class InvarianceReport(BaseModel):
    """The methods that never notice where zero is, raw beside centred."""

    distance_gap: float
    kmeans_inertia_raw: float
    kmeans_inertia_centred: float
    kmeans_same_groups: bool
    tree_threshold_raw: float
    tree_threshold_centred: float
    tree_feature: str
    tree_predictions_agree: bool
    slope_gap: float
    standardized_gap: float


class FoldReport(BaseModel):
    """A mean learned without the three tallest, then applied to them."""

    held_out_heights: list[float]
    training_mean: float
    by_training: list[float]
    by_training_mean: float
    own_mean: float
    by_own: list[float]


class CrowdResponse(BaseModel):
    """Everything the page quotes about the crowd, measured once."""

    centring: CentringReport
    sweep: list[SweepRow]
    lines: LinesReport
    walks: list[WalkReport]
    polynomial: list[PolynomialRow]
    angles: AnglesReport
    invariance: InvarianceReport
    fold: FoldReport


def _centring_report(heights: np.ndarray, weights: np.ndarray) -> CentringReport:
    centrer, centred_height, centred_weight = _centred(heights, weights)
    raw_features = [Feature(HEIGHT, heights), Feature(WEIGHT, weights)]
    centred_features = [
        Feature(HEIGHT, centred_height),
        Feature(WEIGHT, centred_weight),
    ]
    standardized_raw = Standardizer().fit(raw_features).transform(raw_features)
    standardized_centred = (
        Standardizer().fit(centred_features).transform(centred_features)
    )
    standardized_gap = max(
        float(np.max(np.abs(np.asarray(first.values) - np.asarray(second.values))))
        for first, second in zip(standardized_raw, standardized_centred, strict=True)
    )
    raw_gaps = heights[:, None] - heights[None, :]
    centred_gaps = centred_height[:, None] - centred_height[None, :]
    return CentringReport(
        height_centre=float(centrer.scalings[HEIGHT].centre),
        weight_centre=float(centrer.scalings[WEIGHT].centre),
        height_spread=float(centrer.scalings[HEIGHT].spread),
        centred_heights=[float(value) for value in centred_height],
        centred_weights=[float(value) for value in centred_weight],
        height_mean_after=float(np.mean(centred_height)),
        weight_mean_after=float(np.mean(centred_weight)),
        height_variance_before=float(np.var(heights)),
        height_variance_after=float(np.var(centred_height)),
        largest_gap_change=float(np.max(np.abs(raw_gaps - centred_gaps))),
        standardized_gap=standardized_gap,
    )


def _sweep(heights: np.ndarray, weights: np.ndarray) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for mark in ORIGIN_SWEEP:
        measured = heights - mark
        _, centred_height, centred_weight = _centred(measured, weights)
        mean = np.array([float(np.mean(measured)), float(np.mean(weights))])
        direction, share = _uncentred_direction(np.column_stack([measured, weights]))
        direction = _oriented(direction, mean)
        component, _ = _first_component(centred_height, centred_weight)
        rows.append(
            SweepRow(
                height_origin=mark,
                uncentred_angle=_arrow(direction).angle_degrees,
                uncentred_share=share,
                centred_angle=_arrow(_oriented(component, None)).angle_degrees,
            )
        )
    return rows


def _lines(heights: np.ndarray, weights: np.ndarray) -> LinesReport:
    _, centred_height, centred_weight = _centred(heights, weights)
    height = Feature(HEIGHT, heights)
    weight = Feature(WEIGHT, weights)
    centred_height_feature = Feature(HEIGHT, centred_height)
    centred_weight_feature = Feature(WEIGHT, centred_weight)

    ridges: list[RidgeRow] = []
    for penalty in RIDGE_PENALTIES:
        with_intercept = RidgeRegression(penalty=penalty).fit([height], weight)
        centred = RidgeRegression(penalty=penalty, fit_intercept=False).fit(
            [centred_height_feature], centred_weight_feature
        )
        through = RidgeRegression(penalty=penalty, fit_intercept=False).fit(
            [height], weight
        )
        ridges.append(
            RidgeRow(
                penalty=penalty,
                slope_with_intercept=float(with_intercept.coefficients[HEIGHT]),
                intercept=float(with_intercept.intercept),
                slope_centred_through_origin=float(centred.coefficients[HEIGHT]),
                slope_raw_through_origin=float(through.coefficients[HEIGHT]),
                r_squared_raw_through_origin=float(through.score([height], weight)),
            )
        )

    return LinesReport(
        least_squares=_line(
            MultipleLinearRegression().fit([height], weight), height, weight
        ),
        through_origin=_line(
            MultipleLinearRegression(fit_intercept=False).fit([height], weight),
            height,
            weight,
        ),
        centred_through_origin=_line(
            MultipleLinearRegression(fit_intercept=False).fit(
                [centred_height_feature], centred_weight_feature
            ),
            centred_height_feature,
            centred_weight_feature,
        ),
        centred_with_intercept=_line(
            MultipleLinearRegression().fit([centred_height_feature], weight),
            centred_height_feature,
            weight,
        ),
        ridges=ridges,
    )


def _walk(
    reading: Literal["raw", "centred"], column: np.ndarray, weights: np.ndarray
) -> WalkReport:
    design = np.column_stack([np.ones_like(column), column])
    curvatures = np.linalg.eigvalsh(design.T @ design / len(column))
    rate = 1.0 / (2.0 * float(curvatures[-1]))
    height = Feature(HEIGHT, column)
    weight = Feature(WEIGHT, weights)

    model = GradientDescentRegression(
        learning_rate=rate, max_epochs=WALK_PASS_LIMIT
    ).fit([height], weight)
    target = MultipleLinearRegression().fit([height], weight)

    checkpoints: list[WalkCheckpoint] = []
    for passes in WALK_CHECKPOINTS:
        capped = GradientDescentRegression(learning_rate=rate, max_epochs=passes).fit(
            [height], weight
        )
        checkpoints.append(
            WalkCheckpoint(
                passes=capped.epochs_run,
                slope=float(capped.coefficients[HEIGHT]),
                level=float(capped.intercept),
            )
        )

    return WalkReport(
        reading=reading,
        curvatures=[float(value) for value in curvatures],
        condition_number=float(curvatures[-1] / curvatures[0]),
        ones_cosine=_cosine(np.ones_like(column), column),
        learning_rate=rate,
        passes=model.epochs_run,
        converged=model.converged,
        slope=float(model.coefficients[HEIGHT]),
        level=float(model.intercept),
        target_slope=float(target.coefficients[HEIGHT]),
        target_level=float(target.intercept),
        checkpoints=checkpoints,
    )


def _polynomial(
    reading: Literal["raw", "centred"], column: np.ndarray, weights: np.ndarray
) -> PolynomialRow:
    height = [Feature(HEIGHT, column)]
    terms = PolynomialFeatures(degree=2).fit(height).transform(height)
    linear, square = (np.asarray(term.values, dtype=np.float64) for term in terms)
    design = np.column_stack([np.ones_like(column), linear, square])
    weight = Feature(WEIGHT, weights)
    fit = MultipleLinearRegression().fit(terms, weight)
    return PolynomialRow(
        reading=reading,
        correlation=float(np.corrcoef(linear, square)[0, 1]),
        condition_number=float(np.linalg.cond(design)),
        linear_coefficient=float(fit.coefficients[terms[0].name]),
        square_coefficient=float(fit.coefficients[terms[1].name]),
        intercept=float(fit.intercept),
        r_squared=float(fit.score(terms, weight)),
    )


def _cosines(rows: np.ndarray) -> np.ndarray:
    """Every cosine between two rows, by the definition."""
    units = rows / np.linalg.norm(rows, axis=1)[:, None]
    return np.clip(units @ units.T, -1.0, 1.0)


def _smallest_pair(cosines: np.ndarray) -> PairCosine:
    upper = np.triu_indices(len(cosines), k=1)
    position = int(np.argmin(cosines[upper]))
    first, second = int(upper[0][position]), int(upper[1][position])
    return PairCosine(first=first, second=second, value=float(cosines[first, second]))


def _leave_one_out(
    columns: list[np.ndarray], labels: np.ndarray, metric: DistanceMetric
) -> float:
    """How often three neighbours vote a held-out person's label back correctly."""
    right = 0
    for held in range(len(labels)):
        kept = [index for index in range(len(labels)) if index != held]
        model = KNearestNeighboursClassifier(
            n_neighbours=VOTING_NEIGHBOURS, metric=metric
        ).fit(
            [
                Feature(name, column[kept])
                for name, column in zip((HEIGHT, WEIGHT), columns, strict=True)
            ],
            Feature("label", labels[kept]),
        )
        answer = model.predict(
            [
                Feature(name, column[[held]])
                for name, column in zip((HEIGHT, WEIGHT), columns, strict=True)
            ]
        )
        right += int(float(np.asarray(answer)[0]) == float(labels[held]))
    return right / len(labels)


def _angles(
    heights: np.ndarray, weights: np.ndarray, labels: np.ndarray
) -> AnglesReport:
    _, centred_height, centred_weight = _centred(heights, weights)
    raw = _cosines(np.column_stack([heights, weights]))
    centred = _cosines(np.column_stack([centred_height, centred_weight]))
    votes = [
        VoteRow(
            rule=rule,
            raw_accuracy=_leave_one_out([heights, weights], labels, metric),
            centred_accuracy=_leave_one_out(
                [centred_height, centred_weight], labels, metric
            ),
        )
        for rule, metric in (
            ("straight-line", DistanceMetric.EUCLIDEAN),
            ("angle", DistanceMetric.COSINE),
        )
    ]
    return AnglesReport(
        raw_cosines=[[float(value) for value in row] for row in raw],
        centred_cosines=[[float(value) for value in row] for row in centred],
        smallest_raw=_smallest_pair(raw),
        smallest_centred=_smallest_pair(centred),
        column_cosine_raw=_cosine(heights, weights),
        column_cosine_centred=_cosine(centred_height, centred_weight),
        correlation=float(np.corrcoef(heights, weights)[0, 1]),
        votes=votes,
    )


def _same_groups(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two labellings put the same pairs of rows together."""
    return bool(
        np.array_equal(
            first[:, None] == first[None, :], second[:, None] == second[None, :]
        )
    )


def _all_distances(columns: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Every straight-line distance between two people, as a square table.

    Row ``i`` column ``j`` is how far person ``j`` is from person ``i``, read
    from the neighbour search with every person asked for every other.
    """
    features = [
        Feature(name, column)
        for name, column in zip((HEIGHT, WEIGHT), columns, strict=True)
    ]
    model = KNearestNeighboursClassifier(n_neighbours=len(labels)).fit(
        features, Feature("label", labels)
    )
    table = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row, search in enumerate(model.neighbour_search(features)):
        for index, distance in zip(
            search.chosen_indices, search.chosen_distances, strict=True
        ):
            table[row, int(index)] = float(distance)
    return table


def _invariance(
    heights: np.ndarray, weights: np.ndarray, labels: np.ndarray
) -> InvarianceReport:
    _, centred_height, centred_weight = _centred(heights, weights)
    raw_features = [Feature(HEIGHT, heights), Feature(WEIGHT, weights)]
    centred_features = [
        Feature(HEIGHT, centred_height),
        Feature(WEIGHT, centred_weight),
    ]
    label_feature = Feature("label", labels)

    # Each person's distance to themselves is left out. It is zero in the
    # mathematics and the square root of a rounding residue in the arithmetic,
    # which says nothing about where zero was put.
    between_two_people = ~np.eye(len(labels), dtype=bool)
    distance_gap = float(
        np.max(
            np.abs(
                _all_distances([heights, weights], labels)
                - _all_distances([centred_height, centred_weight], labels)
            )[between_two_people]
        )
    )

    raw_groups = KMeans(n_clusters=2, random_seed=0).fit(raw_features)
    centred_groups = KMeans(n_clusters=2, random_seed=0).fit(centred_features)

    raw_tree = DecisionTreeClassifier().fit(raw_features, label_feature)
    centred_tree = DecisionTreeClassifier().fit(centred_features, label_feature)
    raw_root = raw_tree.root
    centred_root = centred_tree.root
    if not isinstance(raw_root, DecisionNode) or not isinstance(
        centred_root, DecisionNode
    ):
        raise _refuse("the tree grown on the crowd asked no question at all")

    raw_line = MultipleLinearRegression().fit(raw_features, label_feature)
    centred_line = MultipleLinearRegression().fit(centred_features, label_feature)
    slope_gap = max(
        abs(float(raw_line.coefficients[name]) - float(centred_line.coefficients[name]))
        for name in (HEIGHT, WEIGHT)
    )

    return InvarianceReport(
        distance_gap=distance_gap,
        kmeans_inertia_raw=float(raw_groups.inertia),
        kmeans_inertia_centred=float(centred_groups.inertia),
        kmeans_same_groups=_same_groups(
            np.asarray(raw_groups.predict(raw_features), dtype=np.float64),
            np.asarray(centred_groups.predict(centred_features), dtype=np.float64),
        ),
        tree_threshold_raw=float(raw_root.split.threshold),
        tree_threshold_centred=float(centred_root.split.threshold),
        tree_feature=raw_root.split.feature_name,
        tree_predictions_agree=bool(
            np.array_equal(
                np.asarray(raw_tree.predict(raw_features)),
                np.asarray(centred_tree.predict(centred_features)),
            )
        ),
        slope_gap=slope_gap,
        standardized_gap=_centring_report(heights, weights).standardized_gap,
    )


def _fold(heights: np.ndarray) -> FoldReport:
    training = [index for index in range(len(heights)) if index not in HELD_OUT]
    held = list(HELD_OUT)
    held_feature = Feature(HEIGHT, heights[held])
    learned = MeanCentrer().fit([Feature(HEIGHT, heights[training])])
    by_training = np.asarray(learned.transform([held_feature])[0].values)
    own = MeanCentrer().fit([held_feature])
    by_own = np.asarray(own.transform([held_feature])[0].values)
    return FoldReport(
        held_out_heights=[float(value) for value in heights[held]],
        training_mean=float(learned.scalings[HEIGHT].centre),
        by_training=[float(value) for value in by_training],
        by_training_mean=float(np.mean(by_training)),
        own_mean=float(own.scalings[HEIGHT].centre),
        by_own=[float(value) for value in by_own],
    )


@cache
def _crowd() -> CrowdResponse:
    heights = np.asarray(CROWD_HEIGHTS, dtype=np.float64)
    weights = np.asarray(CROWD_WEIGHTS, dtype=np.float64)
    labels = np.asarray(CROWD_LABELS, dtype=np.float64)
    centred_height = heights - float(np.mean(heights))
    return CrowdResponse(
        centring=_centring_report(heights, weights),
        sweep=_sweep(heights, weights),
        lines=_lines(heights, weights),
        walks=[
            _walk("raw", heights, weights),
            _walk("centred", _centred(heights, weights)[1], weights),
        ],
        polynomial=[
            _polynomial("raw", heights, weights),
            _polynomial("centred", centred_height, weights),
        ],
        angles=_angles(heights, weights, labels),
        invariance=_invariance(heights, weights, labels),
        fold=_fold(heights),
    )


@router.get("/crowd", response_model=CrowdResponse)
def crowd() -> CrowdResponse:
    """Everything the page quotes about the crowd, measured once and kept."""
    return _crowd()


# ---------------------------------------------------------------------------
# A mean that is not a place: times of day.
# ---------------------------------------------------------------------------

HOURS_IN_A_DAY = 24.0

UNDEFINED_LENGTH = 1e-12
"""Below this resultant length the times cancel, and no circular mean exists."""

MIDNIGHT_TOLERANCE = 1e-9
"""A circular centre this close to either end of the day is reported as midnight.

The arc tangent of a sine that should be zero comes back as a few units in the
last place either side of it, which would print as 1e-16 or as 23.999999999.
"""


class ClockRequest(BaseModel):
    """Times of day as hours after midnight, so half past eleven at night is 23.5."""

    hours: list[float] = Field(min_length=1, max_length=48)

    model_config = {"extra": "forbid"}


class ClockResponse(BaseModel):
    """The same times centred on two different means.

    ``arithmetic_centre`` is what the scaler learns, the ordinary mean of the
    numbers. ``circular_centre`` is the direction of the mean of the times drawn
    as points on a clock face, and ``resultant_length`` is how long that mean
    arrow is, one when every time agrees and zero when they cancel. When they
    cancel there is no circular mean and the centre is absent. The centred
    lists hold each time's offset from the centre, the circular ones wrapped
    into the half day either side of it.
    """

    hours: list[float]
    arithmetic_centre: float
    arithmetic_offsets: list[float]
    circular_centre: float | None
    circular_offsets: list[float] | None
    resultant_length: float


def _wrapped(offset: float) -> float:
    """An offset in hours brought into the half day either side of zero."""
    wrapped = (offset + HOURS_IN_A_DAY / 2.0) % HOURS_IN_A_DAY - HOURS_IN_A_DAY / 2.0
    return float(wrapped)


@router.post("/clock", response_model=ClockResponse)
def clock(request: ClockRequest) -> ClockResponse:
    """Centre a list of times of day by the arithmetic and the circular mean."""
    if any(not 0.0 <= hour < HOURS_IN_A_DAY for hour in request.hours):
        raise _refuse(
            "a time of day is an hour from 0 up to but not including 24, so "
            "half past eleven at night is 23.5"
        )

    feature = Feature("hour", request.hours)
    try:
        centrer = MeanCentrer().fit([feature])
        arithmetic = float(centrer.scalings["hour"].centre)
        offsets = np.asarray(centrer.transform([feature])[0].values, dtype=np.float64)
    except MLLibError as refusal:
        raise _refuse(_in_plain_words(refusal)) from refusal

    angles = np.asarray(request.hours) / HOURS_IN_A_DAY * 2.0 * math.pi
    mean_sine = float(np.mean(np.sin(angles)))
    mean_cosine = float(np.mean(np.cos(angles)))
    length = math.hypot(mean_sine, mean_cosine)

    circular: float | None = None
    circular_offsets: list[float] | None = None
    if length > UNDEFINED_LENGTH:
        circular = (
            math.atan2(mean_sine, mean_cosine) / (2.0 * math.pi) * HOURS_IN_A_DAY
        ) % HOURS_IN_A_DAY
        if (
            circular < MIDNIGHT_TOLERANCE
            or HOURS_IN_A_DAY - circular < MIDNIGHT_TOLERANCE
        ):
            circular = 0.0
        circular_offsets = [_wrapped(hour - circular) for hour in request.hours]

    return ClockResponse(
        hours=list(request.hours),
        arithmetic_centre=arithmetic,
        arithmetic_offsets=[float(value) for value in offsets],
        circular_centre=circular,
        circular_offsets=circular_offsets,
        resultant_length=length,
    )
