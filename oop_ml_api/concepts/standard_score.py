"""The standard score, and what dividing by the spread buys.

Every endpoint here is registered under ``/concepts/the-standard-score``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about standardising a column to a mean of zero and a deviation of one, and what breaks without it.

The running example is an orchard of twenty-four apple trees, each measured by
trunk girth in millimetres and by the water it was given over a season in
metres, with the fruit it bore in kilograms as the target. The two predictor
columns are recorded in units a thousand apart on purpose, so the spread of one
is 659 times the spread of the other and any model that adds the two columns
together reads girth almost exclusively. Girth is very nearly unrelated to the
fruit and water very nearly decides it, so the unit choice does not merely blur
the answer, it points the model at the wrong column, and that shows up as a
negative coefficient of determination rather than as a slightly worse one.

Scoring is leave-one-out and pooled. Each row is held out in turn, the model is
refitted on the other twenty-three, and the twenty-four predictions are scored
once against the twenty-four true yields. Pooling is not a preference here: a
fold holding a single row has no spread in its target, so the coefficient of
determination for that fold alone would be a zero over a zero. Pooling also
removes every seed, so the whole page is deterministic and the tests pin exact
numbers rather than approximations.

Nothing here is random. The one refusal a caller can trigger is a constant
column, which ``Standardizer`` raises ``AllSameValuesError`` for, and which is
turned into a 422 with a sentence written here rather than the library's, since
the website shows these strings to a reader who has no reason to meet an
identifier.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DecisionTreeRegressor,
    Feature,
    KNearestNeighboursRegressor,
    MultipleLinearRegression,
    RegressionEvaluation,
    Regressor,
    RidgeRegression,
    Standardizer,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/the-standard-score", tags=["the-standard-score"])

GIRTH_NAME = "girth"
WATER_NAME = "water"
TARGET_NAME = "yield"

ORCHARD_GIRTH: list[float] = [
    620.0, 385.0, 226.0, 297.0, 515.0, 240.0, 289.0, 202.0,
    430.0, 202.0, 362.0, 302.0, 438.0, 439.0, 494.0, 354.0,
    369.0, 465.0, 426.0, 534.0, 359.0, 598.0, 418.0, 339.0,
]  # fmt: skip
"""Trunk girth in millimetres, twenty-four trees."""

ORCHARD_WATER: list[float] = [
    0.47, 0.55, 0.49, 0.93, 0.58, 0.45, 0.70, 0.79,
    0.66, 0.84, 0.59, 0.76, 0.41, 0.59, 0.51, 0.52,
    0.95, 0.88, 0.90, 0.83, 0.48, 0.84, 0.67, 0.91,
]  # fmt: skip
"""Water given over the season, in metres, for the same twenty-four trees."""

ORCHARD_YIELD: list[float] = [
    30.82, 36.40, 30.94, 53.00, 37.46, 30.61, 43.42, 47.20,
    41.40, 49.00, 36.57, 46.92, 27.78, 38.58, 37.01, 32.90,
    55.14, 53.50, 52.10, 48.77, 32.04, 51.50, 42.89, 53.08,
]  # fmt: skip
"""Fruit borne, in kilograms. Water very nearly decides it; girth barely moves it."""

NURSERY_GIRTH: list[float] = [210.0, 210.0, 260.0, 160.0, 310.0]
"""Five saplings, chosen so the mean is a whole number and the sums check by hand.

They are the whole-number column 1, 1, 2, 0, 3 shifted by 160 and stretched by
50, which is what lets the page assert that a shift and a stretch leave every
standard score where it was.
"""

WHOLE_NUMBER_COLUMN: list[float] = [1.0, 1.0, 2.0, 0.0, 3.0]
"""The nursery row before the shift and the stretch."""

MISTYPED_ROW = 12
"""Which tree's girth the page retypes with a stray digit, 438 becoming 4380."""

SKEWED_COLUMN: list[float] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
    2.0, 3.0, 3.0, 4.0, 6.0, 9.0, 20.0,
]  # fmt: skip
"""A long right tail, for the claim that standardising does not change a shape."""

NORMAL_WITHIN_ONE = 0.6826894921370859
"""The share of a normal curve inside one deviation of its centre.

Quoted beside a measured share so the page can say that the two do not have to
agree. It is a property of that curve rather than of any column, so it is a
constant rather than something computed from data.
"""

NORMAL_WITHIN_TWO = 0.9544997361036416
"""The same for two deviations."""

NORMAL_WITHIN_THREE = 0.9973002039367398
"""The same for three deviations."""

BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]


def _girth_water(rows: Sequence[int]) -> list[Feature]:
    """The two predictor columns, restricted to ``rows``, in their recorded units."""
    girth = np.asarray(ORCHARD_GIRTH, dtype=np.float64)[list(rows)]
    water = np.asarray(ORCHARD_WATER, dtype=np.float64)[list(rows)]
    return [Feature(GIRTH_NAME, girth), Feature(WATER_NAME, water)]


def _target(rows: Sequence[int]) -> Feature:
    """The yields of ``rows``, as the column every fit is scored against."""
    return Feature(TARGET_NAME, np.asarray(ORCHARD_YIELD, dtype=np.float64)[list(rows)])


ALL_ROWS = tuple(range(len(ORCHARD_GIRTH)))


@cache
def _whole_orchard_scaler() -> Standardizer:
    """The two numbers each column teaches, read from all twenty-four trees."""
    return Standardizer().fit(_girth_water(ALL_ROWS))


class ColumnSummary(BaseModel):
    """One column, the two numbers it taught, and every value it turned into."""

    name: str
    values: list[float]
    mean: float
    deviation: float
    scores: list[float]
    smallest: float
    largest: float


class OrchardResponse(BaseModel):
    """The running example, both columns, raw and in standard scores.

    ``spread_ratio`` is the girth deviation over the water deviation, and
    ``squared_spread_ratio`` is that number squared, which is the factor by
    which girth outweighs water inside a squared distance before anything is
    standardised. ``correlation`` is the ordinary correlation between the two
    columns and ``standardised_covariance`` is the covariance of the two
    standard-score columns, which is the same number; the page uses the pair to
    say what standardising a column does to the table of covariances.
    """

    yields: list[float]
    girth: ColumnSummary
    water: ColumnSummary
    spread_ratio: float
    squared_spread_ratio: float
    correlation: float
    standardised_covariance: float


def _summary(feature: Feature, scaler: Standardizer) -> ColumnSummary:
    """One column paired with what the fitted scaler did to it."""
    values = np.asarray(feature.values, dtype=np.float64)
    scaling = scaler.scalings[feature.name]
    scores = np.asarray(scaler.transform([feature])[0].values, dtype=np.float64)
    return ColumnSummary(
        name=feature.name,
        values=[float(value) for value in values],
        mean=float(scaling.mean),
        deviation=float(scaling.standard_deviation),
        scores=[float(score) for score in scores],
        smallest=float(values.min()),
        largest=float(values.max()),
    )


@cache
def _orchard() -> OrchardResponse:
    """The running example, computed once."""
    scaler = _whole_orchard_scaler()
    girth_feature, water_feature = _girth_water(ALL_ROWS)
    girth = _summary(girth_feature, scaler)
    water = _summary(water_feature, scaler)
    ratio = girth.deviation / water.deviation
    girth_scores = np.asarray(girth.scores, dtype=np.float64)
    water_scores = np.asarray(water.scores, dtype=np.float64)
    return OrchardResponse(
        yields=list(ORCHARD_YIELD),
        girth=girth,
        water=water,
        spread_ratio=ratio,
        squared_spread_ratio=ratio**2,
        correlation=float(np.corrcoef(girth.values, water.values)[0, 1]),
        standardised_covariance=float(np.mean(girth_scores * water_scores)),
    )


@router.get("/orchard", response_model=OrchardResponse)
def orchard() -> OrchardResponse:
    """The twenty-four trees, both columns, and what standardising does to each."""
    return _orchard()


class ColumnRequest(BaseModel):
    """One column of values, in the order the widget holds them."""

    values: list[BoundedValue] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class ColumnResponse(BaseModel):
    """What one column taught, and the check that a unit change is invisible.

    ``shifted_largest_gap`` is the largest difference between a value's standard
    score and the standard score of the same column with a constant added to
    every entry, and ``stretched_largest_gap`` is the same for the column
    multiplied by a tenth. Either can be ``None``, because a column whose values
    differ by only a few units in the last place collapses to a constant once a
    thousand is added to it, and a gap against a column that no longer has a
    spread is not a number. ``score_mean`` and ``score_variance`` are read back
    off the transformed column, so the page can show the two promises being
    kept rather than assert them.
    """

    mean: float
    deviation: float
    scores: list[float]
    score_mean: float
    score_variance: float
    sample_score_variance: float
    shifted_largest_gap: float | None
    stretched_largest_gap: float | None


def _scores_of(values: Sequence[float]) -> np.ndarray:
    """Standardise one column and hand back the scores as an array."""
    feature = Feature(GIRTH_NAME, list(values))
    scaler = Standardizer().fit([feature])
    return np.asarray(scaler.transform([feature])[0].values, dtype=np.float64)


CONSTANT_COLUMN_REFUSAL = (
    "every value in this column is the same, so it has no spread, and the "
    "standard score would be a division by zero"
)
"""What a caller is told when the column they sent carries no variation.

Written here rather than passed through, because the page shows this sentence
to a reader and the wording it replaces names a parameter.
"""


def _refuse(attempt: Callable[[], object]) -> object:
    """Run one attempt, turning a library refusal into a readable 422."""
    try:
        return attempt()
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=CONSTANT_COLUMN_REFUSAL
        ) from refusal


def _gap_against(moved: np.ndarray, scores: np.ndarray) -> float | None:
    """The largest disagreement between a moved column's scores and the original.

    ``None`` when the move destroyed the spread, which happens for a column whose
    values differ by a few units in the last place once a large constant is added
    to every one of them. There is no gap to report against a column that has no
    scores.
    """
    try:
        return float(np.max(np.abs(_scores_of(moved) - scores)))
    except MLLibError:
        return None


@router.post("/column", response_model=ColumnResponse)
def column(request: ColumnRequest) -> ColumnResponse:
    """Standardise one column and report both promises and both invariances."""
    values = np.asarray(request.values, dtype=np.float64)
    scores = np.asarray(_refuse(lambda: _scores_of(values)), dtype=np.float64)
    scaler = Standardizer().fit([Feature(GIRTH_NAME, values)])
    scaling = scaler.scalings[GIRTH_NAME]
    return ColumnResponse(
        mean=float(scaling.mean),
        deviation=float(scaling.standard_deviation),
        scores=[float(score) for score in scores],
        score_mean=float(scores.mean()),
        score_variance=float(scores.var()),
        sample_score_variance=float(scores.var(ddof=1)),
        shifted_largest_gap=_gap_against(values + 1000.0, scores),
        stretched_largest_gap=_gap_against(values * 0.1, scores),
    )


class NeighbourRequest(BaseModel):
    """Which tree to ask about, and how many neighbours it should listen to."""

    tree: int = Field(ge=0, le=len(ORCHARD_GIRTH) - 1)
    k: int = Field(default=3, ge=1, le=len(ORCHARD_GIRTH) - 1)

    model_config = {"extra": "forbid"}


class NeighbourRow(BaseModel):
    """One of the nearest trees, with how far away it is and what it bore."""

    tree: int
    distance: float
    girth_share: float
    girth: float
    water: float
    borne: float


class NeighbourReading(BaseModel):
    """The neighbours one reading of the columns produces, and their average."""

    reading: str
    neighbours: list[NeighbourRow]
    predicted: float
    error: float


class NeighbourResponse(BaseModel):
    """The same question asked of the raw columns and of the standard scores."""

    tree: int
    k: int
    girth: float
    water: float
    borne: float
    readings: list[NeighbourReading]


def _neighbours_under(
    label: str, first: np.ndarray, second: np.ndarray, tree: int, k: int
) -> NeighbourReading:
    """The k nearest trees to ``tree`` under one reading of the two columns."""
    first_gaps = (first - first[tree]) ** 2
    second_gaps = (second - second[tree]) ** 2
    squared = first_gaps + second_gaps
    squared[tree] = np.inf
    order = np.argsort(squared)[:k]
    borne = np.asarray(ORCHARD_YIELD, dtype=np.float64)
    predicted = float(borne[order].mean())
    return NeighbourReading(
        reading=label,
        neighbours=[
            NeighbourRow(
                tree=int(row),
                distance=float(np.sqrt(squared[row])),
                girth_share=(
                    0.0
                    if squared[row] == 0.0
                    else float(first_gaps[row] / squared[row])
                ),
                girth=ORCHARD_GIRTH[row],
                water=ORCHARD_WATER[row],
                borne=ORCHARD_YIELD[row],
            )
            for row in order
        ],
        predicted=predicted,
        error=predicted - ORCHARD_YIELD[tree],
    )


@router.post("/neighbours", response_model=NeighbourResponse)
def neighbours(request: NeighbourRequest) -> NeighbourResponse:
    """Which trees are nearest, read once in the recorded units and once in scores."""
    scaler = _whole_orchard_scaler()
    girth_feature, water_feature = _girth_water(ALL_ROWS)
    raw_girth = np.asarray(girth_feature.values, dtype=np.float64)
    raw_water = np.asarray(water_feature.values, dtype=np.float64)
    scored = scaler.transform([girth_feature, water_feature])
    return NeighbourResponse(
        tree=request.tree,
        k=request.k,
        girth=ORCHARD_GIRTH[request.tree],
        water=ORCHARD_WATER[request.tree],
        borne=ORCHARD_YIELD[request.tree],
        readings=[
            _neighbours_under(
                "recorded", raw_girth, raw_water, request.tree, request.k
            ),
            _neighbours_under(
                "standardised",
                np.asarray(scored[0].values, dtype=np.float64),
                np.asarray(scored[1].values, dtype=np.float64),
                request.tree,
                request.k,
            ),
        ],
    )


ModelMaker = Callable[[], Regressor[Sequence[Feature], Feature]]

MODEL_MAKERS: dict[str, ModelMaker] = {
    "nearest neighbours, three": lambda: KNearestNeighboursRegressor(n_neighbours=3),
    "nearest neighbours, five": lambda: KNearestNeighboursRegressor(n_neighbours=5),
    "ridge, penalty 1": lambda: RidgeRegression(penalty=1.0),
    "ridge, penalty 10": lambda: RidgeRegression(penalty=10.0),
    "a tree, depth 3": lambda: DecisionTreeRegressor(max_depth=3),
    "least squares": MultipleLinearRegression,
}
"""The six fits the page holds up against the same twenty-four trees."""


def _leave_one_out(
    make_model: ModelMaker,
    reading: str,
    names: tuple[str, ...] = (GIRTH_NAME, WATER_NAME),
) -> float:
    """Hold each tree out in turn, then score all twenty-four predictions at once.

    ``reading`` is ``recorded`` for the columns as measured, ``inside`` for two
    numbers learned from the twenty-three rows the fit saw, and ``outside`` for
    two numbers learned once from all twenty-four before any row was held back,
    which is the leak the page measures.
    """
    whole = _whole_orchard_scaler()
    predictions: list[float] = []
    for held in ALL_ROWS:
        training = [row for row in ALL_ROWS if row != held]
        training_columns = [
            column for column in _girth_water(training) if column.name in names
        ]
        held_columns = [
            column for column in _girth_water([held]) if column.name in names
        ]
        if reading == "recorded":
            fit_on, predict_on = training_columns, held_columns
        elif reading == "inside":
            scaler = Standardizer().fit(training_columns)
            fit_on = scaler.transform(training_columns)
            predict_on = scaler.transform(held_columns)
        else:
            fit_on = whole.transform(training_columns)
            predict_on = whole.transform(held_columns)
        model = make_model()
        model.fit(fit_on, _target(training))
        predictions.append(float(np.asarray(model.predict(predict_on))[0]))
    return float(
        RegressionEvaluation(_target(ALL_ROWS), np.array(predictions)).r2_score
    )


class ModelScores(BaseModel):
    """One model family scored three ways over the same twenty-four trees.

    ``leak`` is ``outside`` minus ``inside``, so a positive number means the fit
    that saw the held-out row inside its two learned numbers scored better for
    having seen it.
    """

    family: str
    recorded: float
    inside: float
    outside: float
    leak: float
    moved: bool


class NamedCoefficient(BaseModel):
    """One fitted weight, bound to the column it belongs to."""

    name: str
    value: float


class CoefficientReading(BaseModel):
    """A fit's weights under one reading of the columns."""

    reading: str
    coefficients: list[NamedCoefficient]
    intercept: float
    fitted_score: float


class TreeReading(BaseModel):
    """The question a tree roots on, under one reading of the columns."""

    reading: str
    root_column: str
    root_threshold: float
    depth: int
    leaves: int
    fitted_score: float


class ModelsResponse(BaseModel):
    """Which models noticed the unit, measured rather than argued.

    ``water_alone`` is the same nearest-neighbour question asked of the one
    column that carries the signal, which is the obvious alternative to
    standardising both and is better here. ``translated_back`` divides each
    standardised weight by its column's deviation, which should reproduce the
    weights fitted on the recorded columns.
    """

    scores: list[ModelScores]
    water_alone_three: float
    water_alone_five: float
    girth_alone_three: float
    least_squares: list[CoefficientReading]
    ridge_penalty_one: list[CoefficientReading]
    translated_back: list[NamedCoefficient]
    trees: list[TreeReading]
    tree_largest_prediction_gap: float
    tree_threshold_translated: float
    least_squares_score_gap: float


def _coefficient_readings(
    make_model: ModelMaker,
) -> list[CoefficientReading]:
    """One fit on the recorded columns and one on the standard scores."""
    scaler = _whole_orchard_scaler()
    raw_columns = _girth_water(ALL_ROWS)
    readings: list[CoefficientReading] = []
    for label, columns in (
        ("recorded", raw_columns),
        ("standardised", scaler.transform(raw_columns)),
    ):
        model = make_model()
        model.fit(columns, _target(ALL_ROWS))
        readings.append(
            CoefficientReading(
                reading=label,
                coefficients=[
                    NamedCoefficient(name=weight.name, value=float(weight.value))
                    for weight in model.coefficients
                ],
                intercept=float(model.intercept),
                fitted_score=float(model.score(columns, _target(ALL_ROWS))),
            )
        )
    return readings


@cache
def _models() -> ModelsResponse:
    """Every model the page compares, fitted once and cached."""
    scaler = _whole_orchard_scaler()
    raw_columns = _girth_water(ALL_ROWS)
    scored_columns = scaler.transform(raw_columns)

    scores = []
    for name, maker in MODEL_MAKERS.items():
        recorded = _leave_one_out(maker, "recorded")
        inside = _leave_one_out(maker, "inside")
        outside = _leave_one_out(maker, "outside")
        scores.append(
            ModelScores(
                family=name,
                recorded=recorded,
                inside=inside,
                outside=outside,
                leak=outside - inside,
                moved=abs(inside - recorded) > 1e-12,
            )
        )

    trees: list[TreeReading] = []
    predictions: list[np.ndarray] = []
    for label, columns in (
        ("recorded", raw_columns),
        ("standardised", scored_columns),
    ):
        tree = DecisionTreeRegressor(max_depth=3).fit(columns, _target(ALL_ROWS))
        trees.append(
            TreeReading(
                reading=label,
                root_column=tree.root.split.feature_name,
                root_threshold=float(tree.root.split.threshold),
                depth=tree.depth,
                leaves=tree.n_leaves,
                fitted_score=float(tree.score(columns, _target(ALL_ROWS))),
            )
        )
        predictions.append(np.asarray(tree.predict(columns), dtype=np.float64))

    standardised_root = trees[1]
    root_scaling = scaler.scalings[standardised_root.root_column]

    least_squares = _coefficient_readings(MultipleLinearRegression)
    standardised_line = least_squares[1]

    return ModelsResponse(
        scores=scores,
        water_alone_three=_leave_one_out(
            lambda: KNearestNeighboursRegressor(n_neighbours=3),
            "recorded",
            (WATER_NAME,),
        ),
        water_alone_five=_leave_one_out(
            lambda: KNearestNeighboursRegressor(n_neighbours=5),
            "recorded",
            (WATER_NAME,),
        ),
        girth_alone_three=_leave_one_out(
            lambda: KNearestNeighboursRegressor(n_neighbours=3),
            "recorded",
            (GIRTH_NAME,),
        ),
        least_squares=least_squares,
        ridge_penalty_one=_coefficient_readings(lambda: RidgeRegression(penalty=1.0)),
        translated_back=[
            NamedCoefficient(
                name=weight.name,
                value=weight.value / scaler.scalings[weight.name].standard_deviation,
            )
            for weight in standardised_line.coefficients
        ],
        trees=trees,
        tree_largest_prediction_gap=float(
            np.max(np.abs(predictions[0] - predictions[1]))
        ),
        tree_threshold_translated=float(
            standardised_root.root_threshold * root_scaling.standard_deviation
            + root_scaling.mean
        ),
        least_squares_score_gap=float(
            least_squares[1].fitted_score - least_squares[0].fitted_score
        ),
    )


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    """Which of six fits the unit changed, and by how much."""
    return _models()


class HeldOutReading(BaseModel):
    """The held-out trees read against one pair of numbers, and what a line said."""

    reading: str
    girth_mean: float
    girth_deviation: float
    water_mean: float
    water_deviation: float
    girth_scores: list[float]
    water_scores: list[float]
    predicted: list[float]
    largest_error: float


class HoldOutResponse(BaseModel):
    """The four wettest trees, held out and then read two ways.

    The first reading uses the numbers the twenty remaining trees taught, which
    is the rule; the second refits on the four held-out trees themselves, which
    is what measuring a held-out set against itself amounts to.
    """

    held_out: list[int]
    borne: list[float]
    water: list[float]
    girth: list[float]
    readings: list[HeldOutReading]


@cache
def _hold_out() -> HoldOutResponse:
    """The wettest four held out, standardised both ways, and a line read on each."""
    water = np.asarray(ORCHARD_WATER, dtype=np.float64)
    held = sorted(sorted(ALL_ROWS, key=lambda row: -water[row])[:4])
    training = [row for row in ALL_ROWS if row not in held]

    training_columns = _girth_water(training)
    held_columns = _girth_water(held)
    learned = Standardizer().fit(training_columns)
    own = Standardizer().fit(held_columns)
    line = MultipleLinearRegression().fit(
        learned.transform(training_columns), _target(training)
    )
    borne = np.asarray(ORCHARD_YIELD, dtype=np.float64)[held]

    readings: list[HeldOutReading] = []
    for label, scaler in (("training", learned), ("their own", own)):
        block = scaler.transform(held_columns)
        predicted = np.asarray(line.predict(block), dtype=np.float64)
        readings.append(
            HeldOutReading(
                reading=label,
                girth_mean=float(scaler.scalings[GIRTH_NAME].mean),
                girth_deviation=float(scaler.scalings[GIRTH_NAME].standard_deviation),
                water_mean=float(scaler.scalings[WATER_NAME].mean),
                water_deviation=float(scaler.scalings[WATER_NAME].standard_deviation),
                girth_scores=[float(value) for value in block[0].values],
                water_scores=[float(value) for value in block[1].values],
                predicted=[float(value) for value in predicted],
                largest_error=float(np.max(np.abs(predicted - borne))),
            )
        )

    return HoldOutResponse(
        held_out=held,
        borne=[float(value) for value in borne],
        water=[ORCHARD_WATER[row] for row in held],
        girth=[ORCHARD_GIRTH[row] for row in held],
        readings=readings,
    )


@router.get("/hold-out", response_model=HoldOutResponse)
def hold_out() -> HoldOutResponse:
    """The wettest four trees, measured against the rest and against themselves."""
    return _hold_out()


class ShareWithin(BaseModel):
    """How much of a column falls inside a given number of deviations."""

    column: str
    within_one: float
    within_two: float
    within_three: float


class MistypedReading(BaseModel):
    """What one stray digit does to the two numbers a column teaches."""

    before_mean: float
    before_deviation: float
    after_mean: float
    after_deviation: float
    mistyped_score: float
    widest_real_score_before: float
    widest_real_score_after: float
    others_lowest_after: float
    others_highest_after: float
    others_span_before: float
    others_span_after: float


class ShapeReading(BaseModel):
    """A column's lopsidedness and its ordering, before and after."""

    column: str
    skew_before: float
    skew_after: float
    order_kept: bool
    largest_score: float
    share_within_one: float
    share_within_two: float


class LimitsResponse(BaseModel):
    """The facts the closing part rests on, each one measured.

    ``tiny_deviation`` is the spread of nine values a machine epsilon apart,
    which is a real spread and is standardised without complaint;
    ``tiny_scores`` shows where those nine land. ``constant_defined`` is false
    because a column with no spread has no quotient to take, and
    ``single_row_defined`` is false for the same reason with one row, where
    every deviation from the mean is exactly zero.
    """

    tiny_deviation: float
    tiny_scores: list[float]
    constant_defined: bool
    single_row_defined: bool
    mistyped: MistypedReading
    shares: list[ShareWithin]
    normal_within_one: float
    normal_within_two: float
    normal_within_three: float
    chebyshev_within_two: float
    skewed: ShapeReading
    girth_shape: ShapeReading


def _skew(values: np.ndarray) -> float:
    """The third moment about the mean over the cube of the deviation."""
    centred = values - values.mean()
    return float((centred**3).mean() / (centred**2).mean() ** 1.5)


def _shares(name: str, scores: np.ndarray) -> ShareWithin:
    """How much of a standardised column sits inside one, two and three."""
    return ShareWithin(
        column=name,
        within_one=float(np.mean(np.abs(scores) <= 1.0)),
        within_two=float(np.mean(np.abs(scores) <= 2.0)),
        within_three=float(np.mean(np.abs(scores) <= 3.0)),
    )


def _is_defined(values: Sequence[float]) -> bool:
    """Whether a column has a standard score at all, asked rather than assumed."""
    try:
        Standardizer().fit([Feature(GIRTH_NAME, list(values))])
    except MLLibError:
        return False
    return True


def _shape_reading(name: str, values: np.ndarray) -> ShapeReading:
    """One column's shape before standardising and after."""
    scores = _scores_of(values)
    return ShapeReading(
        column=name,
        skew_before=_skew(values),
        skew_after=_skew(scores),
        order_kept=bool(np.all(np.argsort(values) == np.argsort(scores))),
        largest_score=float(scores.max()),
        share_within_one=float(np.mean(np.abs(scores) <= 1.0)),
        share_within_two=float(np.mean(np.abs(scores) <= 2.0)),
    )


@cache
def _limits() -> LimitsResponse:
    """Every degenerate reading the closing part quotes, asked once."""
    tiny = [index * 1e-16 for index in range(9)]
    tiny_scores = _scores_of(tiny)

    girth = np.asarray(ORCHARD_GIRTH, dtype=np.float64)
    before_scores = _scores_of(girth)
    mistyped = girth.copy()
    mistyped[MISTYPED_ROW] = girth[MISTYPED_ROW] * 10.0
    after_scaler = Standardizer().fit([Feature(GIRTH_NAME, mistyped)])
    after_scaling = after_scaler.scalings[GIRTH_NAME]
    after_scores = np.asarray(
        after_scaler.transform([Feature(GIRTH_NAME, mistyped)])[0].values,
        dtype=np.float64,
    )
    others_before = np.delete(before_scores, MISTYPED_ROW)
    others_after = np.delete(after_scores, MISTYPED_ROW)
    widest = int(np.argmax(girth))
    before_scaling = _whole_orchard_scaler().scalings[GIRTH_NAME]

    scored = _whole_orchard_scaler().transform(_girth_water(ALL_ROWS))
    return LimitsResponse(
        tiny_deviation=float(
            Standardizer()
            .fit([Feature(GIRTH_NAME, tiny)])
            .scalings[GIRTH_NAME]
            .standard_deviation
        ),
        tiny_scores=[float(score) for score in tiny_scores],
        constant_defined=_is_defined([300.0] * 9),
        single_row_defined=_is_defined([300.0]),
        mistyped=MistypedReading(
            before_mean=float(before_scaling.mean),
            before_deviation=float(before_scaling.standard_deviation),
            after_mean=float(after_scaling.mean),
            after_deviation=float(after_scaling.standard_deviation),
            mistyped_score=float(after_scores[MISTYPED_ROW]),
            widest_real_score_before=float(before_scores[widest]),
            widest_real_score_after=float(after_scores[widest]),
            others_lowest_after=float(others_after.min()),
            others_highest_after=float(others_after.max()),
            others_span_before=float(others_before.max() - others_before.min()),
            others_span_after=float(others_after.max() - others_after.min()),
        ),
        shares=[
            _shares(GIRTH_NAME, np.asarray(scored[0].values, dtype=np.float64)),
            _shares(WATER_NAME, np.asarray(scored[1].values, dtype=np.float64)),
        ],
        normal_within_one=NORMAL_WITHIN_ONE,
        normal_within_two=NORMAL_WITHIN_TWO,
        normal_within_three=NORMAL_WITHIN_THREE,
        chebyshev_within_two=0.75,
        skewed=_shape_reading(
            "a long tail", np.asarray(SKEWED_COLUMN, dtype=np.float64)
        ),
        girth_shape=_shape_reading(GIRTH_NAME, girth),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """Where the quotient stops existing, and what it never promised."""
    return _limits()


class NurseryResponse(BaseModel):
    """The five saplings taken through the arithmetic by hand.

    ``whole_number_scores`` is the same column before it was shifted by 160 and
    stretched by 50, and ``largest_gap`` is how far the two sets of scores are
    apart, which says whether the unit survived the transformation.
    """

    girths: list[float]
    mean: float
    deviation: float
    deviations_from_mean: list[float]
    squared_deviations: list[float]
    variance: float
    scores: list[float]
    whole_numbers: list[float]
    whole_number_scores: list[float]
    largest_gap: float
    score_mean: float
    score_variance: float
    sample_score_variance: float


@cache
def _nursery() -> NurseryResponse:
    """Every intermediate of the five-sapling sum, so the page can print them."""
    girths = np.asarray(NURSERY_GIRTH, dtype=np.float64)
    scaler = Standardizer().fit([Feature(GIRTH_NAME, girths)])
    scaling = scaler.scalings[GIRTH_NAME]
    scores = _scores_of(girths)
    whole_scores = _scores_of(np.asarray(WHOLE_NUMBER_COLUMN, dtype=np.float64))
    deviations = girths - scaling.mean
    return NurseryResponse(
        girths=[float(value) for value in girths],
        mean=float(scaling.mean),
        deviation=float(scaling.standard_deviation),
        deviations_from_mean=[float(value) for value in deviations],
        squared_deviations=[float(value) for value in deviations**2],
        variance=float(np.mean(deviations**2)),
        scores=[float(score) for score in scores],
        whole_numbers=list(WHOLE_NUMBER_COLUMN),
        whole_number_scores=[float(score) for score in whole_scores],
        largest_gap=float(np.max(np.abs(scores - whole_scores))),
        score_mean=float(scores.mean()),
        score_variance=float(scores.var()),
        sample_score_variance=float(scores.var(ddof=1)),
    )


@router.get("/nursery", response_model=NurseryResponse)
def nursery() -> NurseryResponse:
    """The five saplings, every intermediate of the sum, and the same column moved."""
    return _nursery()
