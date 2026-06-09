"""Feature scaling, for the page that puts every column on one footing.

The first endpoint fits the library's ``Standardizer`` and its four affine
scalers, ``MinMaxScaler``, ``MaxAbsScaler``, ``RobustScaler`` and
``RootMeanSquareScaler``, on the same list of values, and returns each one's
centre, spread and scaled values keyed by a closed enum. The page needs all five
side by side because the lesson is not any one recipe but the fact that they are
one recipe, ``(value - centre) / spread``, differing only in how the two numbers
are read off the column, and that they disagree most when the column carries an
outlier. The widget drags one value far to the right and watches which rows
squash and which hold still.

The rest report why any of that matters. The neighbours endpoint reruns one
k-nearest-neighbours vote with the crowd's heights in millimetres, centimetres
and metres, and then standardized, so the page can show that the unit alone
decides who is nearest. The models endpoint fits a gradient walk, a ridge, a
least-squares line and a decision tree under the same four readings, and
reports which of them noticed. The hold-out endpoint standardizes the held-out
rows two ways, with the training numbers and with their own, and predicts from
both, which is the leak made into a number. The quartiles endpoint shows where
the robust scaler's two quartiles fall on a sorted column, since the rule that
places them is the one thing about that scaler a reader cannot see. The
contracts endpoint asks the library every edge the page's last part tabulates,
so the table is probed on every load rather than copied once.

Nothing here is seeded, because nothing here is random. A zero spread is left to
the library to refuse, since ``AllSameValuesError`` already becomes a 422
carrying the library's own words.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DecisionNode,
    DecisionTreeClassifier,
    Feature,
    FeatureScaler,
    GradientDescentRegression,
    KNearestNeighboursClassifier,
    MaxAbsScaler,
    MinMaxScaler,
    RidgeRegression,
    RobustScaler,
    RootMeanSquareScaler,
    SimpleLinearRegression,
    Standardizer,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    features_of,
)
from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["feature-scaling"])

FEATURE_NAME = "value"
"""The one column every request carries, named so the library can address it."""


class ScalingMethod(StrEnum):
    """The five readings of a centre and a spread the page compares."""

    STANDARDIZE = "standardize"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    ROOT_MEAN_SQUARE = "root_mean_square"


AFFINE_SCALERS: dict[ScalingMethod, type[FeatureScaler]] = {
    ScalingMethod.MIN_MAX: MinMaxScaler,
    ScalingMethod.MAX_ABS: MaxAbsScaler,
    ScalingMethod.ROBUST: RobustScaler,
    ScalingMethod.ROOT_MEAN_SQUARE: RootMeanSquareScaler,
}
"""The four methods that share the library's affine base.

``Standardizer`` predates that base and keeps its own vocabulary of a mean and a
standard deviation, so it is fitted separately below and read through its own
names.
"""

BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]


class ScaleRequest(BaseModel):
    """One feature's raw values, in the order the widget holds them."""

    values: list[BoundedValue] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class MethodScaling(BaseModel):
    """What one method learned, and what it did to every value.

    ``scaled`` is in the input order, so the widget can draw each raw dot's
    scaled twin without re-sorting anything.
    """

    centre: float
    spread: float
    scaled: list[float]


class ScaleResponse(BaseModel):
    """All five methods, keyed by name, fitted on the same column."""

    scalings: dict[ScalingMethod, MethodScaling]


@router.post("/concepts/feature-scaling/scale", response_model=ScaleResponse)
def scale(request: ScaleRequest) -> ScaleResponse:
    """Fit all five scalers on one column and report what each one did."""
    feature = Feature(FEATURE_NAME, request.values)

    standardizer = Standardizer().fit([feature])
    standard_scaling = standardizer.scalings[FEATURE_NAME]
    scalings = {
        ScalingMethod.STANDARDIZE: MethodScaling(
            centre=float(standard_scaling.mean),
            spread=float(standard_scaling.standard_deviation),
            scaled=_scaled_values(standardizer, feature),
        )
    }

    for method, scaler_type in AFFINE_SCALERS.items():
        scaler = scaler_type().fit([feature])
        affine_scaling = scaler.scalings.scaling_for(FEATURE_NAME)
        scalings[method] = MethodScaling(
            centre=float(affine_scaling.centre),
            spread=float(affine_scaling.spread),
            scaled=_scaled_values(scaler, feature),
        )

    return ScaleResponse(scalings=scalings)


def _scaled_values(
    scaler: Standardizer | FeatureScaler, feature: Feature
) -> list[float]:
    """The feature put through a fitted scaler, as plain floats in input order."""
    transformed = scaler.transform([feature])[0]
    return [float(value) for value in np.asarray(transformed.values, dtype=np.float64)]


# ---------------------------------------------------------------------------
# The unit the height was recorded in, and what four models make of it.
# ---------------------------------------------------------------------------

HEIGHT_UNITS: tuple[tuple[str, float], ...] = (
    ("millimetres", 10.0),
    ("centimetres", 1.0),
    ("metres", 0.01),
)
"""The crowd's heights rewritten in three units, each a factor on the same numbers.

The weights stay in kilograms throughout, so the factor changes only how the
height column compares with the weight column, which is the whole question.
"""

STANDARDIZED = "standardized"
"""The fourth reading, both columns divided by their own spread."""

MAX_K = 15
"""The most neighbours the widget may ask to consult."""

DESCENT_MAX_EPOCHS = 20_000
"""How many passes a gradient walk may take before the page calls it stuck.

Generous enough that centimetres finish and short enough that millimetres,
which would need over a million passes at their safe rate, do not hold the
request open.
"""

RIDGE_PENALTY = 1.0
"""The one penalty the ridge comparison uses, the same under every unit."""


class QueryPoint(BaseModel):
    """The new person to classify, height in centimetres and weight in kilograms."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)

    model_config = {"extra": "forbid"}


class UnitNeighboursRequest(BaseModel):
    """The crowd in centimetres and kilograms, one query, and how many vote."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    query: QueryPoint
    k: int = Field(ge=1, le=MAX_K)

    model_config = {"extra": "forbid"}


class UnitNeighbour(BaseModel):
    """One of the k nearest under one reading of the columns.

    ``height_share`` is the part of the squared distance that the height column
    contributed, in that reading's units. Near one the vote was decided by
    height alone; near zero, by weight alone.
    """

    index: int
    distance: float
    label: int
    height_share: float | None


class UnitVote(BaseModel):
    """The whole vote under one reading of the two columns."""

    unit: str
    factor: float
    prediction: int
    votes_for_one: int
    neighbours: list[UnitNeighbour]


class UnitNeighboursResponse(BaseModel):
    """The same vote taken four times, and the numbers the fourth divided by."""

    votes: list[UnitVote]
    height_mean: float
    height_deviation: float
    weight_mean: float
    weight_deviation: float


def _neighbour_vote(
    unit: str,
    factor: float,
    height_column: np.ndarray,
    weight_column: np.ndarray,
    labels: Feature,
    query_height: float,
    query_weight: float,
    k: int,
    points: list[LabelledPoint],
) -> UnitVote:
    """Fit on one reading of the columns and put the query to its k nearest."""
    model = KNearestNeighboursClassifier(n_neighbours=k).fit(
        [Feature("height", height_column), Feature("weight", weight_column)], labels
    )
    query = [Feature("height", [query_height]), Feature("weight", [query_weight])]
    prediction = int(np.asarray(model.predict(query), dtype=np.float64)[0])

    neighbours: list[UnitNeighbour] = []
    for search in model.neighbour_search(query):
        for index, distance in zip(
            search.chosen_indices, search.chosen_distances, strict=True
        ):
            height_gap = float(height_column[int(index)] - query_height)
            weight_gap = float(weight_column[int(index)] - query_weight)
            squared = height_gap**2 + weight_gap**2
            neighbours.append(
                UnitNeighbour(
                    index=int(index),
                    distance=float(distance),
                    label=points[int(index)].label,
                    height_share=None if squared == 0.0 else height_gap**2 / squared,
                )
            )

    neighbours.sort(key=lambda each: each.distance)
    votes_for_one = sum(1 for neighbour in neighbours if neighbour.label == 1)
    return UnitVote(
        unit=unit,
        factor=factor,
        prediction=prediction,
        votes_for_one=votes_for_one,
        neighbours=neighbours,
    )


@router.post(
    "/concepts/feature-scaling/units/neighbours",
    response_model=UnitNeighboursResponse,
)
def unit_neighbours(request: UnitNeighboursRequest) -> UnitNeighboursResponse:
    """Take one neighbour vote with the heights in each unit, then standardized."""
    if request.k > len(request.points):
        raise HTTPException(
            status_code=422,
            detail=(
                f"k is {request.k} but only {len(request.points)} people are "
                "stored, and a vote cannot have more voters than people"
            ),
        )

    heights, weights, labels = features_of(request.points)
    height_values = np.asarray(heights.values, dtype=np.float64)
    weight_values = np.asarray(weights.values, dtype=np.float64)

    votes = [
        _neighbour_vote(
            unit,
            factor,
            height_values * factor,
            weight_values,
            labels,
            request.query.x * factor,
            request.query.y,
            request.k,
            request.points,
        )
        for unit, factor in HEIGHT_UNITS
    ]

    standardizer = Standardizer().fit([heights, weights])
    scaled_height, scaled_weight = standardizer.transform([heights, weights])
    scaled_query_height, scaled_query_weight = standardizer.transform(
        [Feature("height", [request.query.x]), Feature("weight", [request.query.y])]
    )
    votes.append(
        _neighbour_vote(
            STANDARDIZED,
            1.0,
            np.asarray(scaled_height.values, dtype=np.float64),
            np.asarray(scaled_weight.values, dtype=np.float64),
            labels,
            float(np.asarray(scaled_query_height.values)[0]),
            float(np.asarray(scaled_query_weight.values)[0]),
            request.k,
            request.points,
        )
    )

    height_scaling = standardizer.scalings["height"]
    weight_scaling = standardizer.scalings["weight"]
    return UnitNeighboursResponse(
        votes=votes,
        height_mean=float(height_scaling.mean),
        height_deviation=float(height_scaling.standard_deviation),
        weight_mean=float(weight_scaling.mean),
        weight_deviation=float(weight_scaling.standard_deviation),
    )


class UnitModelsRequest(BaseModel):
    """The crowd in centimetres and kilograms, and nothing else."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DescentByUnit(BaseModel):
    """One gradient walk, weight against height measured from its mean.

    The heights are centred before the walk so that the level and the slope
    are independent directions, and the unit is then the only thing that
    differs between rows. ``curvature`` is the mean squared centred height,
    the bowl's curvature along the slope direction, where the level direction
    always has curvature one. ``learning_rate`` is half the divergence
    threshold, the rate that removes the stiffer direction's error in one
    step, so every row walks at its own best safe rate.
    """

    unit: str
    factor: float
    curvature: float
    condition_number: float
    learning_rate: float
    divergence_threshold: float
    epochs_run: int
    converged: bool
    slope: float
    level: float


class RidgeByUnit(BaseModel):
    """A ridge line, weight against height, at one penalty under one unit."""

    unit: str
    factor: float
    coefficient: float
    intercept: float
    r_squared: float


class LeastSquaresByUnit(BaseModel):
    """The least-squares line under one unit.

    ``slope_in_centimetres`` and ``intercept_in_centimetres`` translate the
    standardized fit back through the learned mean and deviation, and are the
    raw fit's own numbers for the other rows.
    """

    unit: str
    factor: float
    slope: float
    intercept: float
    r_squared: float
    slope_in_centimetres: float
    intercept_in_centimetres: float


class TreeByUnit(BaseModel):
    """A decision tree grown on the crowd under one unit."""

    unit: str
    factor: float
    depth: int
    n_leaves: int
    root_feature: str | None
    root_threshold: float | None
    predictions: list[int]


class InverseRoundTrip(BaseModel):
    """The height column scaled and restored, and the largest gap it left."""

    method: ScalingMethod
    max_gap: float


class UnitModelsResponse(BaseModel):
    """Four models under four readings, and the five scalers' round trips."""

    descents: list[DescentByUnit]
    ridges: list[RidgeByUnit]
    least_squares: list[LeastSquaresByUnit]
    trees: list[TreeByUnit]
    trees_agree: bool
    round_trips: list[InverseRoundTrip]


def _readings(
    heights: np.ndarray, standardizer: Standardizer
) -> list[tuple[str, float, np.ndarray]]:
    """The height column in each unit, then standardized, as plain arrays."""
    readings = [(unit, factor, heights * factor) for unit, factor in HEIGHT_UNITS]
    scaled = standardizer.transform([Feature("height", heights)])[0]
    readings.append((STANDARDIZED, 1.0, np.asarray(scaled.values, dtype=np.float64)))
    return readings


def _descent(
    unit: str, factor: float, column: np.ndarray, weights: Feature
) -> DescentByUnit:
    """Walk to the line on this reading of the heights, at its own safe rate."""
    centred = column - column.mean()
    curvature = float(np.mean(centred**2))
    stiffest = max(1.0, curvature)
    softest = min(1.0, curvature)
    threshold = 1.0 / stiffest
    rate = threshold / 2.0

    model = GradientDescentRegression(
        learning_rate=rate, max_epochs=DESCENT_MAX_EPOCHS
    ).fit([Feature("height", centred)], weights)

    return DescentByUnit(
        unit=unit,
        factor=factor,
        curvature=curvature,
        condition_number=stiffest / softest,
        learning_rate=rate,
        divergence_threshold=threshold,
        epochs_run=model.epochs_run,
        converged=model.converged,
        slope=float(model.coefficients["height"]),
        level=float(model.intercept),
    )


def _tree(
    unit: str, factor: float, column: np.ndarray, weights: Feature, labels: Feature
) -> TreeByUnit:
    """Grow the tree on this reading and read its root question."""
    features = [Feature("height", column), weights]
    model = DecisionTreeClassifier().fit(features, labels)
    root = model.root
    predictions = np.asarray(model.predict(features), dtype=np.float64)
    split = root.split if isinstance(root, DecisionNode) else None
    return TreeByUnit(
        unit=unit,
        factor=factor,
        depth=model.depth,
        n_leaves=model.n_leaves,
        root_feature=None if split is None else split.feature_name,
        root_threshold=None if split is None else float(split.threshold),
        predictions=[int(value) for value in predictions],
    )


@router.post(
    "/concepts/feature-scaling/units/models", response_model=UnitModelsResponse
)
def unit_models(request: UnitModelsRequest) -> UnitModelsResponse:
    """Fit four models with the heights in each unit and report which noticed."""
    heights, weights, labels = features_of(request.points)
    height_values = np.asarray(heights.values, dtype=np.float64)
    weight_values = np.asarray(weights.values, dtype=np.float64)

    standardizer = Standardizer().fit([heights])
    scaling = standardizer.scalings["height"]
    readings = _readings(height_values, standardizer)

    descents = [
        _descent(unit, factor, column, weights) for unit, factor, column in readings
    ]

    ridges: list[RidgeByUnit] = []
    lines: list[LeastSquaresByUnit] = []
    for unit, factor, column in readings:
        feature = Feature("height", column)
        ridge = RidgeRegression(penalty=RIDGE_PENALTY).fit([feature], weights)
        ridges.append(
            RidgeByUnit(
                unit=unit,
                factor=factor,
                coefficient=float(ridge.coefficients["height"]),
                intercept=float(ridge.intercept),
                r_squared=float(ridge.score([feature], weights)),
            )
        )

        line = SimpleLinearRegression().fit(column, weight_values)
        slope, intercept = float(line.slope), float(line.intercept)
        if unit == STANDARDIZED:
            slope_cm = slope / scaling.standard_deviation
            intercept_cm = intercept - slope_cm * scaling.mean
        else:
            slope_cm = slope * factor
            intercept_cm = intercept
        lines.append(
            LeastSquaresByUnit(
                unit=unit,
                factor=factor,
                slope=slope,
                intercept=intercept,
                r_squared=float(line.score(column, weight_values)),
                slope_in_centimetres=slope_cm,
                intercept_in_centimetres=intercept_cm,
            )
        )

    trees = [
        _tree(unit, factor, column, weights, labels)
        for unit, factor, column in readings
    ]
    trees_agree = all(tree.predictions == trees[0].predictions for tree in trees)

    round_trips = [
        InverseRoundTrip(
            method=ScalingMethod.STANDARDIZE,
            max_gap=float(
                np.max(
                    np.abs(
                        scaling.restore(
                            np.asarray(
                                standardizer.transform([heights])[0].values,
                                dtype=np.float64,
                            )
                        )
                        - height_values
                    )
                )
            ),
        )
    ]
    for method, scaler_type in AFFINE_SCALERS.items():
        scaler = scaler_type().fit([heights])
        restored = scaler.inverse_transform(scaler.transform([heights]))[0]
        gap = np.max(
            np.abs(np.asarray(restored.values, dtype=np.float64) - height_values)
        )
        round_trips.append(InverseRoundTrip(method=method, max_gap=float(gap)))

    return UnitModelsResponse(
        descents=descents,
        ridges=ridges,
        least_squares=lines,
        trees=trees,
        trees_agree=trees_agree,
        round_trips=round_trips,
    )


# ---------------------------------------------------------------------------
# Fit on the training rows, apply to the held-out ones.
# ---------------------------------------------------------------------------


class HoldOutRequest(BaseModel):
    """The crowd, and which of them are held out of the fit."""

    points: list[PlainPoint] = Field(min_length=4, max_length=MAX_POINTS)
    held_out: list[int] = Field(min_length=2, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class HoldOutResponse(BaseModel):
    """The held-out heights standardized two ways, and what a line makes of each.

    ``by_training`` uses the mean and deviation learned on the training rows,
    which is the contract; ``by_own`` refits on the held-out rows themselves,
    which is the leak. The line is fitted on the training rows' standardized
    heights, so ``predictions_by_own`` are that line reading numbers on a
    different scale from the one it learned.
    """

    training_indices: list[int]
    held_out_indices: list[int]
    training_mean: float
    training_deviation: float
    training_scaled: list[float]
    by_training: list[float]
    mean_by_training: float
    own_mean: float
    own_deviation: float
    by_own: list[float]
    slope: float
    intercept: float
    predictions_by_training: list[float]
    predictions_by_own: list[float]
    actual_weights: list[float]


@router.post("/concepts/feature-scaling/hold-out", response_model=HoldOutResponse)
def hold_out(request: HoldOutRequest) -> HoldOutResponse:
    """Standardize on the training rows and apply to the rest, both ways."""
    count = len(request.points)
    held = sorted(set(request.held_out))
    if len(held) != len(request.held_out):
        raise HTTPException(status_code=422, detail="held_out repeats an index")
    if held[0] < 0 or held[-1] >= count:
        raise HTTPException(
            status_code=422,
            detail=f"held_out names a person outside the {count} supplied",
        )
    training = [index for index in range(count) if index not in set(held)]
    if len(training) < 2:
        raise HTTPException(
            status_code=422,
            detail="at least two people must remain in training",
        )

    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    weights = np.asarray([point.y for point in request.points], dtype=np.float64)

    training_height = Feature("height", heights[training])
    held_height = Feature("height", heights[held])

    learned = Standardizer().fit([training_height])
    scaling = learned.scalings["height"]
    training_scaled = np.asarray(
        learned.transform([training_height])[0].values, dtype=np.float64
    )
    by_training = np.asarray(
        learned.transform([held_height])[0].values, dtype=np.float64
    )

    own = Standardizer().fit([held_height])
    own_scaling = own.scalings["height"]
    by_own = np.asarray(own.transform([held_height])[0].values, dtype=np.float64)

    line = SimpleLinearRegression().fit(training_scaled, weights[training])

    return HoldOutResponse(
        training_indices=training,
        held_out_indices=held,
        training_mean=float(scaling.mean),
        training_deviation=float(scaling.standard_deviation),
        training_scaled=[float(value) for value in training_scaled],
        by_training=[float(value) for value in by_training],
        mean_by_training=float(by_training.mean()),
        own_mean=float(own_scaling.mean),
        own_deviation=float(own_scaling.standard_deviation),
        by_own=[float(value) for value in by_own],
        slope=float(line.slope),
        intercept=float(line.intercept),
        predictions_by_training=[
            float(value) for value in np.asarray(line.predict(by_training))
        ],
        predictions_by_own=[float(value) for value in np.asarray(line.predict(by_own))],
        actual_weights=[float(value) for value in weights[held]],
    )


# ---------------------------------------------------------------------------
# Where the quartiles land.
# ---------------------------------------------------------------------------


class QuartilePosition(BaseModel):
    """One quartile: the fractional index it reads at, and the value there.

    ``position`` counts from zero along the sorted column, so a whole number
    names one value and a half names the midpoint of two.
    """

    position: float
    value: float


class QuartilesResponse(BaseModel):
    """The sorted column and the three cuts the robust scaler reads off it."""

    sorted_values: list[float]
    first_quartile: QuartilePosition
    median: QuartilePosition
    third_quartile: QuartilePosition
    interquartile_range: float
    robust_centre: float
    robust_spread: float


@router.post("/concepts/feature-scaling/quartiles", response_model=QuartilesResponse)
def quartiles(request: ScaleRequest) -> QuartilesResponse:
    """Sort the column and place the quartiles by the rule the library uses."""
    values = np.asarray(request.values, dtype=np.float64)
    ordered = np.sort(values)
    last = len(ordered) - 1
    first, middle, third = np.quantile(ordered, [0.25, 0.5, 0.75])

    scaler = RobustScaler().fit([Feature(FEATURE_NAME, values)])
    learned = scaler.scalings.scaling_for(FEATURE_NAME)

    return QuartilesResponse(
        sorted_values=[float(value) for value in ordered],
        first_quartile=QuartilePosition(position=0.25 * last, value=float(first)),
        median=QuartilePosition(position=0.5 * last, value=float(middle)),
        third_quartile=QuartilePosition(position=0.75 * last, value=float(third)),
        interquartile_range=float(third - first),
        robust_centre=float(learned.centre),
        robust_spread=float(learned.spread),
    )


# ---------------------------------------------------------------------------
# The contracts, asked again on every load.
# ---------------------------------------------------------------------------

Outcome = Literal["accepted", "refused"]


class ContractProbe(BaseModel):
    """One edge put to one scaler, and what came back.

    ``result`` is the number or count that came back when the edge was
    accepted, and ``reason`` is why it was refused when it was not. The
    reason is the refusal's own sentence with this package's identifiers
    taken out of it, because the website shows this table to a reader who has
    no reason to know a class name; ``_in_plain_words`` does that rewriting
    and a test pins that no identifier survives it.
    """

    edge: str
    method: str
    outcome: Outcome
    result: str | None
    reason: str | None


class ContractsResponse(BaseModel):
    """Every edge the page tabulates, probed."""

    probes: list[ContractProbe]


CROWD_HEIGHTS = [
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
]
"""The eleven heights the pages share, for the probes that need a real column."""

CROWD_WEIGHTS = [41.0, 53.0, 57.0, 57.0, 61.0, 25.0, 28.0, 24.0, 80.0, 83.0, 78.0]
"""Their weights, for the probes that need two named columns."""


NOT_FITTED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z]* must be fit before this is available"
)
"""A refusal that opens on the class name of whatever was asked too early."""

PLAIN_WORDS = (
    ("feature_values", "a column\u2019s values"),
    ("this standardizer knows", "this scaler knows"),
)
"""Identifiers a reader has no reason to meet, and what to say instead."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the two that name a parameter or a class and leaves the rest
    alone. The wording is otherwise the refusal's own, which is the point of
    showing it; a test asserts that no identifier survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    sentence = NOT_FITTED_PATTERN.sub(
        "this scaler must be fitted before that can be read", sentence
    )
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _probe(edge: str, method: str, attempt: Callable[[], object]) -> ContractProbe:
    """Run one attempt and record whether it was accepted or refused."""
    try:
        result = attempt()
    except MLLibError as refusal:
        return ContractProbe(
            edge=edge,
            method=method,
            outcome="refused",
            result=None,
            reason=_in_plain_words(refusal),
        )
    return ContractProbe(
        edge=edge,
        method=method,
        outcome="accepted",
        result=None if result is None else str(result),
        reason=None,
    )


def _spread_of(scaler: FeatureScaler, values: list[float]) -> float:
    """Fit one affine scaler on one column and read the spread it learned."""
    return float(scaler.fit([Feature("height", values)]).scalings["height"].spread)


def _deviation_of(values: list[float]) -> float:
    """Fit the standardizer on one column and read the deviation it learned."""
    return float(
        Standardizer()
        .fit([Feature("height", values)])
        .scalings["height"]
        .standard_deviation
    )


@cache
def _contract_probes() -> ContractsResponse:
    """The fixed list of edges, each asked of the library once."""
    heights = Feature("height", CROWD_HEIGHTS)
    weights = Feature("weight", CROWD_WEIGHTS)
    fitted_standardizer = Standardizer().fit([heights, weights])
    fitted_min_max = MinMaxScaler().fit([heights, weights])
    tiny = [index * 1e-16 for index in range(9)]
    constant = [170.0, 170.0, 170.0]
    seven_of_eight = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0]
    six_of_eight = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 9.0]
    ten_of_twelve = [5.0] * 10 + [6.0, 7.0]
    nine_of_twelve = [5.0] * 9 + [6.0, 7.0, 8.0]

    probes = [
        _probe("no features at all", "standardize", lambda: Standardizer().fit([])),
        _probe("no features at all", "min-max", lambda: MinMaxScaler().fit([])),
        _probe(
            "an empty column",
            "standardize",
            lambda: Standardizer().fit([Feature("height", [])]),
        ),
        _probe("one row", "standardize", lambda: _deviation_of([170.0])),
        _probe("one row", "min-max", lambda: _spread_of(MinMaxScaler(), [170.0])),
        _probe("one row", "robust", lambda: _spread_of(RobustScaler(), [170.0])),
        _probe("one row", "max-abs", lambda: _spread_of(MaxAbsScaler(), [170.0])),
        _probe(
            "one row",
            "root mean square",
            lambda: _spread_of(RootMeanSquareScaler(), [170.0]),
        ),
        _probe("a constant column", "standardize", lambda: _deviation_of(constant)),
        _probe(
            "a constant column", "min-max", lambda: _spread_of(MinMaxScaler(), constant)
        ),
        _probe(
            "a constant column", "robust", lambda: _spread_of(RobustScaler(), constant)
        ),
        _probe(
            "a constant column", "max-abs", lambda: _spread_of(MaxAbsScaler(), constant)
        ),
        _probe(
            "a constant column",
            "root mean square",
            lambda: _spread_of(RootMeanSquareScaler(), constant),
        ),
        _probe(
            "a column of zeros",
            "max-abs",
            lambda: _spread_of(MaxAbsScaler(), [0.0, 0.0, 0.0]),
        ),
        _probe(
            "a missing value",
            "standardize",
            lambda: _deviation_of([160.0, float("nan"), 180.0]),
        ),
        _probe(
            "an infinite value",
            "min-max",
            lambda: _spread_of(MinMaxScaler(), [160.0, float("inf"), 180.0]),
        ),
        _probe(
            "six identical values of eight",
            "robust",
            lambda: _spread_of(RobustScaler(), six_of_eight),
        ),
        _probe(
            "seven identical values of eight",
            "robust",
            lambda: _spread_of(RobustScaler(), seven_of_eight),
        ),
        _probe(
            "nine identical values of twelve",
            "robust",
            lambda: _spread_of(RobustScaler(), nine_of_twelve),
        ),
        _probe(
            "ten identical values of twelve",
            "robust",
            lambda: _spread_of(RobustScaler(), ten_of_twelve),
        ),
        _probe(
            "nine values one machine epsilon apart",
            "standardize",
            lambda: _deviation_of(tiny),
        ),
        _probe(
            "nine values one machine epsilon apart",
            "min-max",
            lambda: _spread_of(MinMaxScaler(), tiny),
        ),
        _probe(
            "transform with a subset of the fitted features",
            "standardize",
            lambda: len(fitted_standardizer.transform([heights])),
        ),
        _probe(
            "transform with a subset of the fitted features",
            "min-max",
            lambda: len(fitted_min_max.transform([weights])),
        ),
        _probe(
            "transform with a feature the fit never saw",
            "standardize",
            lambda: fitted_standardizer.transform([Feature("age", CROWD_HEIGHTS)]),
        ),
        _probe(
            "transform with a feature the fit never saw",
            "min-max",
            lambda: fitted_min_max.transform([Feature("age", CROWD_HEIGHTS)]),
        ),
        _probe(
            "inverse transform with a feature the fit never saw",
            "min-max",
            lambda: fitted_min_max.inverse_transform([Feature("age", CROWD_HEIGHTS)]),
        ),
        _probe(
            "transform before fit",
            "standardize",
            lambda: Standardizer().transform([heights]),
        ),
        _probe(
            "transform before fit",
            "min-max",
            lambda: MinMaxScaler().transform([heights]),
        ),
        _probe(
            "read the learned numbers before fit",
            "robust",
            lambda: RobustScaler().scalings,
        ),
        _probe(
            "two features with one name",
            "standardize",
            lambda: Standardizer().fit([heights, Feature("height", CROWD_WEIGHTS)]),
        ),
        _probe(
            "two features of different lengths",
            "min-max",
            lambda: MinMaxScaler().fit([heights, Feature("weight", CROWD_WEIGHTS[:5])]),
        ),
        _probe(
            "a value beyond the fitted range, 250 cm",
            "min-max",
            lambda: float(
                np.asarray(
                    fitted_min_max.transform([Feature("height", [250.0])])[0].values
                )[0]
            ),
        ),
    ]
    return ContractsResponse(probes=probes)


@router.get("/concepts/feature-scaling/contracts", response_model=ContractsResponse)
def contracts() -> ContractsResponse:
    """Every edge the page's last part tabulates, asked of the library."""
    return _contract_probes()
