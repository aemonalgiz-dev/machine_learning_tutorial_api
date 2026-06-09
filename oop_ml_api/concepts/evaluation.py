"""Held-out scoring and cross-validation, for the evaluation page.

Every concept page ends on the same sentence, that the honest judge is data
the fit never saw, and this module is that sentence given machinery. One
endpoint fits a polynomial on a fixed seeded split of the points and scores
it twice, once on the rows it saw and once on the rows held back, so the
widget can colour the held-out people and show both numbers. One traces that
pair of scores across every degree, the curve that turns around. One runs the
library's five-fold cross-validation at each degree, so the page can show
every row taking a turn as the judge, and the spread that says how much the
mean deserves to be trusted.

The page's fixed split is drawn by index here, the same shuffle-and-cut the
library's ``TrainTestSplitter`` performs, because the widget needs to know
which of the caller's points were held out, and a ``DataSplit`` hands back
the rows themselves rather than their positions. The later endpoints want
the library's own splitters and still need positions, so they split a
dataset whose one predictor *is* the row position and read the positions
back off the held-out half.

Four more endpoints serve the rest of the page. One deals the same split
under a family of seeds, so the page can say how far a single held-out
verdict moves with the deal. One deals the folds and scores each on its own,
reporting the mean and spread beside the score of every held-out prediction
pooled, which is the summary that survives a fold of one row. One folds the
labelled crowd plain or stratified with a shallow tree and reports each
fold's class counts beside the averaged and the pooled figures, which is the
argument for pooling. And one measures the leak on the running example, a
standardiser fitted before the folds are dealt against one refitted inside
each, for a ridge fit that can feel it and a least-squares fit that cannot.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    ClassificationCrossValidationResult,
    ClassificationEvaluation,
    CrossValidation,
    Dataset,
    DecisionTreeClassifier,
    Feature,
    KFold,
    MultiClassEvaluation,
    MultipleLinearRegression,
    PipelineStep,
    PipelineSteps,
    PolynomialFeatures,
    RegressionCrossValidationResult,
    RegressionEvaluation,
    RegressionPipeline,
    Regressor,
    RidgeRegression,
    Standardizer,
    TrainTestSplitter,
    UndefinedMetricError,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint, Point
from oop_ml_api.concepts.split import HELD_OUT_FRACTION, SPLIT_SEED, held_out_indices
from oop_ml_api.concepts.two_feature_classification import LabelledPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/evaluation", tags=["evaluation"])

MAX_DEGREE = 9
"""The wildest curve the page lets a reader ask for."""

FOLD_COUNT = 5
"""How many folds the cross-validation section deals."""

CURVE_SAMPLES = 81


def _fit_on_train(
    x_values: np.ndarray, y_values: np.ndarray, held: np.ndarray, degree: int
):
    """Fit the degree on the training rows only, returning the fitted pair."""
    training = np.setdiff1d(np.arange(x_values.size), held)
    train_inputs = Feature("t", x_values[training])
    train_target = Feature("h", y_values[training])

    polynomial = PolynomialFeatures(degree=degree).fit([train_inputs])
    model = MultipleLinearRegression().fit(
        polynomial.transform([train_inputs]), train_target
    )
    return polynomial, model, training


class SplitFitRequest(BaseModel):
    """The points and the degree to fit on the training share alone."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}


class SplitFitResponse(BaseModel):
    """The fit, judged twice.

    ``held_out_indices`` are positions into the request's points, so the
    widget can ring the very dots the fit never saw.
    """

    held_out_indices: list[int]
    train_r_squared: float
    held_out_r_squared: float
    curve: list[CurvePoint]


@router.post("/split-fit", response_model=SplitFitResponse)
def split_fit(request: SplitFitRequest) -> SplitFitResponse:
    """Fit on the training share, then score on both shares."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    held = held_out_indices(x_values.size)

    n_training = x_values.size - held.size
    if n_training <= request.degree:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a degree {request.degree} curve has {request.degree + 1} "
                f"numbers to pin down and the training share holds only "
                f"{n_training} points"
            ),
        )

    polynomial, model, _ = _fit_on_train(x_values, y_values, held, request.degree)

    train_mask = np.ones(x_values.size, dtype=bool)
    train_mask[held] = False
    train_terms = polynomial.transform([Feature("t", x_values[train_mask])])
    held_terms = polynomial.transform([Feature("t", x_values[held])])

    grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), CURVE_SAMPLES)
    curve_heights = np.asarray(
        model.predict(polynomial.transform([Feature("t", grid)])), dtype=np.float64
    )

    return SplitFitResponse(
        held_out_indices=[int(index) for index in held],
        train_r_squared=float(
            model.score(train_terms, Feature("h", y_values[train_mask]))
        ),
        held_out_r_squared=float(model.score(held_terms, Feature("h", y_values[held]))),
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
    )


class GapCurveRequest(BaseModel):
    """The points whose two score curves are to be traced across degrees."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class GapCurveResponse(BaseModel):
    """One training score and one held-out score per degree.

    The sweep stops where the training share runs out of points to pin the
    next degree down.
    """

    degrees: list[int]
    train_scores: list[float]
    held_out_scores: list[float]


@router.post("/degree-curve", response_model=GapCurveResponse)
def degree_curve(request: GapCurveRequest) -> GapCurveResponse:
    """Trace both scores across every degree the training share can pin."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    held = held_out_indices(x_values.size)

    train_mask = np.ones(x_values.size, dtype=bool)
    train_mask[held] = False

    degrees: list[int] = []
    train_scores: list[float] = []
    held_out_scores: list[float] = []
    for degree in range(1, MAX_DEGREE + 1):
        if int(np.sum(train_mask)) <= degree:
            break
        polynomial, model, _ = _fit_on_train(x_values, y_values, held, degree)
        train_terms = polynomial.transform([Feature("t", x_values[train_mask])])
        held_terms = polynomial.transform([Feature("t", x_values[held])])
        degrees.append(degree)
        train_scores.append(
            float(model.score(train_terms, Feature("h", y_values[train_mask])))
        )
        held_out_scores.append(
            float(model.score(held_terms, Feature("h", y_values[held])))
        )

    return GapCurveResponse(
        degrees=degrees,
        train_scores=train_scores,
        held_out_scores=held_out_scores,
    )


class ValidationCurveRequest(BaseModel):
    """The points to cross-validate a polynomial on, degree by degree."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DegreeValidation(BaseModel):
    """One degree's five-fold verdict, the mean and how far the folds spread."""

    degree: int
    mean_r_squared: float
    spread: float


class ValidationCurveResponse(BaseModel):
    """The library's cross-validation run at every degree the folds can pin."""

    validations: list[DegreeValidation]


@router.post("/validation-curve", response_model=ValidationCurveResponse)
def validation_curve(request: ValidationCurveRequest) -> ValidationCurveResponse:
    """Five folds per degree, every row taking one turn as the judge.

    The candidate at each degree is a pipeline, the term manufacturing and
    the regression refit together inside every fold, so no fold ever sees a
    column built from rows it is about to judge.
    """
    dataset = Dataset(
        [Feature("t", [point.x for point in request.points])],
        Feature("h", [point.y for point in request.points]),
    )

    smallest_training = dataset.n_samples - dataset.n_samples // FOLD_COUNT - 1
    validations: list[DegreeValidation] = []
    for degree in range(1, MAX_DEGREE + 1):
        if smallest_training <= degree:
            break
        candidate = RegressionPipeline(
            steps=PipelineSteps(
                [
                    PipelineStep(
                        name="terms", transformer=PolynomialFeatures(degree=degree)
                    )
                ]
            ),
            model=MultipleLinearRegression(),
        )
        result = CrossValidation(
            folds=KFold(n_folds=FOLD_COUNT, random_seed=SPLIT_SEED)
        ).evaluate(candidate, dataset)
        validations.append(
            DegreeValidation(
                degree=degree,
                mean_r_squared=float(result.mean_r2_score),
                spread=float(result.r2_score_spread),
            )
        )

    return ValidationCurveResponse(validations=validations)


# --- The seed family, and what one held-out verdict is worth ---------------

FAMILY_SIZE = 30
"""How many seeds a family request may run.

One seed's held-out score is one draw of a few rows; the page needs the
whole family to say how far the verdict moves when the deal changes.
"""

MAX_FIRST_SEED = 10000
"""The largest first seed a family request may name."""

MAX_FOLDS = MAX_POINTS
"""Folds beyond the row count are the library's refusal, not the door's."""


def _throw_arrays(points: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """The request's points as two float arrays, in the request's order."""
    x_values = np.asarray([point.x for point in points], dtype=np.float64)
    y_values = np.asarray([point.y for point in points], dtype=np.float64)
    return x_values, y_values


def _positions_dataset(target: np.ndarray, target_name: str) -> Dataset:
    """A dataset whose one input column is each row's position.

    The library's splitters hand back rows rather than indices, and the
    widgets need to know *which* of the caller's dots were held out so they
    can ring them. Splitting a dataset whose only predictor is the row
    position turns the rows the splitter returns back into positions, and
    the actual fit is then made on the columns those positions name.
    """
    return Dataset(
        [Feature("position", np.arange(target.size, dtype=np.float64))],
        Feature(target_name, target),
    )


def _held_out_positions(testing: Dataset) -> np.ndarray:
    """The positions a split held out, read back off the position column."""
    return np.sort(np.asarray(testing.input_features[0].values, dtype=np.intp))


class _SplitVerdict:
    """One fit on the training rows, scored there and predicted on the rest.

    The held-out score is left to the caller, because a held-out share of one
    row has no spread for a ratio to explain and the fold loop wants the
    predictions anyway, to pool them.
    """

    __slots__ = ("held_out_predictions", "train_r_squared")

    def __init__(self, train_r_squared: float, held_out_predictions: np.ndarray):
        self.train_r_squared = train_r_squared
        self.held_out_predictions = held_out_predictions


def _verdict_on_split(
    x_values: np.ndarray, y_values: np.ndarray, held: np.ndarray, degree: int
) -> _SplitVerdict:
    """Fit on the rows not held out, score there, and predict the held-out rows."""
    polynomial, model, training = _fit_on_train(x_values, y_values, held, degree)
    train_terms = polynomial.transform([Feature("t", x_values[training])])
    held_terms = polynomial.transform([Feature("t", x_values[held])])
    return _SplitVerdict(
        train_r_squared=float(
            model.score(train_terms, Feature("h", y_values[training]))
        ),
        held_out_predictions=np.asarray(model.predict(held_terms), dtype=np.float64),
    )


def _refuse_if_unpinnable(n_training: int, degree: int) -> None:
    """A degree the training rows cannot pin is refused with the counts named."""
    if n_training <= degree:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a degree {degree} curve has {degree + 1} numbers to pin down "
                f"and the training share holds only {n_training} points"
            ),
        )


class SplitFamilyRequest(BaseModel):
    """The points, the degree, the share to hold out, and a family of seeds."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(ge=1, le=MAX_DEGREE)
    held_out_fraction: float = Field(default=HELD_OUT_FRACTION, gt=0.0, lt=1.0)
    first_seed: int = Field(default=0, ge=0, le=MAX_FIRST_SEED)
    n_seeds: int = Field(default=FAMILY_SIZE, ge=1, le=FAMILY_SIZE)

    model_config = {"extra": "forbid"}


class SeededSplit(BaseModel):
    """One seed's deal and the two scores it produced."""

    seed: int
    held_out_indices: list[int]
    train_r_squared: float
    held_out_r_squared: float


class SplitFamilyResponse(BaseModel):
    """The same fit judged under every seed in the family.

    ``held_out_spread`` is the highest held-out score less the lowest, the
    same reading the cross-validation result reports across folds.
    """

    n_training: int
    n_held_out: int
    splits: list[SeededSplit]
    mean_train_r_squared: float
    mean_held_out_r_squared: float
    lowest_held_out_r_squared: float
    highest_held_out_r_squared: float
    held_out_spread: float


@router.post("/split-family", response_model=SplitFamilyResponse)
def split_family(request: SplitFamilyRequest) -> SplitFamilyResponse:
    """Deal the split under each seed in turn and score the fit each time.

    The deal is the library's own shuffle-and-cut, and the wanted share is
    rounded to a row count and clamped so neither side is ever empty.
    """
    x_values, y_values = _throw_arrays(request.points)
    positions = _positions_dataset(y_values, "h")

    splits: list[SeededSplit] = []
    for seed in range(request.first_seed, request.first_seed + request.n_seeds):
        split = TrainTestSplitter(
            test_fraction=request.held_out_fraction, random_seed=seed
        ).split(positions)
        held = _held_out_positions(split.testing)
        _refuse_if_unpinnable(x_values.size - held.size, request.degree)
        verdict = _verdict_on_split(x_values, y_values, held, request.degree)
        held_out = RegressionEvaluation(y_values[held], verdict.held_out_predictions)
        splits.append(
            SeededSplit(
                seed=seed,
                held_out_indices=[int(index) for index in held],
                train_r_squared=verdict.train_r_squared,
                held_out_r_squared=float(held_out.r2_score),
            )
        )

    held_scores = [split.held_out_r_squared for split in splits]
    train_scores = [split.train_r_squared for split in splits]
    return SplitFamilyResponse(
        n_training=x_values.size - len(splits[0].held_out_indices),
        n_held_out=len(splits[0].held_out_indices),
        splits=splits,
        mean_train_r_squared=float(np.mean(train_scores)),
        mean_held_out_r_squared=float(np.mean(held_scores)),
        lowest_held_out_r_squared=float(min(held_scores)),
        highest_held_out_r_squared=float(max(held_scores)),
        held_out_spread=float(max(held_scores) - min(held_scores)),
    )


# --- The folds, dealt and scored one at a time ------------------------------


class FoldDealRequest(BaseModel):
    """The points, the degree, how many folds to deal and the seed to deal by."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(ge=1, le=MAX_DEGREE)
    n_folds: int = Field(default=FOLD_COUNT, ge=2, le=MAX_FOLDS)
    seed: int = Field(default=SPLIT_SEED, ge=0, le=MAX_FIRST_SEED)

    model_config = {"extra": "forbid"}


class FoldReading(BaseModel):
    """One fold's held-out rows and what the fit made of them.

    ``r_squared`` is ``None`` on a fold of one row, where the held-out share
    has no spread to explain and the library refuses the ratio rather than
    inventing a number; the squared error is always defined.
    """

    held_out_indices: list[int]
    n_held_out: int
    r_squared: float | None
    residual_sum_of_squares: float
    mean_squared_error: float


class FoldDealResponse(BaseModel):
    """Every fold's verdict and the two ways of summarising them.

    ``mean_r_squared`` and ``spread`` are the library's summary across folds
    and are ``None`` whenever any fold's score is undefined.
    ``pooled_r_squared`` scores every held-out prediction at once, against
    the mean of every held-out target, so it survives folds of one row.
    """

    folds: list[FoldReading]
    n_fits: int
    mean_r_squared: float | None
    spread: float | None
    pooled_r_squared: float
    pooled_mean_squared_error: float


@router.post("/fold-deal", response_model=FoldDealResponse)
def fold_deal(request: FoldDealRequest) -> FoldDealResponse:
    """Deal the folds, refit inside each, and score every fold's held-out rows.

    More folds than rows is the library's refusal and comes back as a typed
    422; a training fold too small to pin the degree is refused here with
    the counts named.
    """
    x_values, y_values = _throw_arrays(request.points)
    positions = _positions_dataset(y_values, "h")
    splits = KFold(n_folds=request.n_folds, random_seed=request.seed).split(positions)

    readings: list[FoldReading] = []
    evaluations: list[RegressionEvaluation] = []
    held_out_actuals: list[np.ndarray] = []
    held_out_predictions: list[np.ndarray] = []
    for split in splits:
        held = _held_out_positions(split.testing)
        _refuse_if_unpinnable(x_values.size - held.size, request.degree)
        predicted = _verdict_on_split(
            x_values, y_values, held, request.degree
        ).held_out_predictions
        evaluation = RegressionEvaluation(y_values[held], predicted)
        evaluations.append(evaluation)
        held_out_actuals.append(y_values[held])
        held_out_predictions.append(predicted)
        readings.append(
            FoldReading(
                held_out_indices=[int(index) for index in held],
                n_held_out=int(held.size),
                r_squared=_r_squared_or_none(evaluation),
                residual_sum_of_squares=float(evaluation.residual_sum_of_squares),
                mean_squared_error=float(evaluation.mean_squared_error),
            )
        )

    pooled = RegressionEvaluation(
        np.concatenate(held_out_actuals), np.concatenate(held_out_predictions)
    )
    every_fold_defined = all(reading.r_squared is not None for reading in readings)
    result = RegressionCrossValidationResult(evaluations)
    return FoldDealResponse(
        folds=readings,
        n_fits=len(readings),
        mean_r_squared=float(result.mean_r2_score) if every_fold_defined else None,
        spread=float(result.r2_score_spread) if every_fold_defined else None,
        pooled_r_squared=float(pooled.r2_score),
        pooled_mean_squared_error=float(pooled.mean_squared_error),
    )


def _r_squared_or_none(evaluation: RegressionEvaluation) -> float | None:
    """The fold's R squared, or ``None`` where its target has no spread."""
    try:
        return float(evaluation.r2_score)
    except UndefinedMetricError:
        return None


# --- Folding a classifier: the plain deal, the stratified deal, the pooling --

TREE_DEPTH = 2
"""The classifier the crowd is folded with is a tree two questions deep."""

N_CLASSES = 2
"""Stated to every fold's evaluation, so a fold missing a class still counts."""

POSITIVE_CLASS = 1
"""The class recall is read for, which the pages' crowd calls adult."""


class ClassifierFoldRequest(BaseModel):
    """The labelled crowd, how many folds, whether to stratify, and the seed."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_folds: int = Field(default=FOLD_COUNT, ge=2, le=MAX_FOLDS)
    stratified: bool = False
    seed: int = Field(default=SPLIT_SEED, ge=0, le=MAX_FIRST_SEED)

    model_config = {"extra": "forbid"}


class ClassifierFoldReading(BaseModel):
    """One fold's held-out people, their class counts and the two rates.

    ``recall`` is ``None`` on a fold holding no positive, where it is zero
    over zero; the counts it would be built from are given so the pooling
    can be checked by hand.
    """

    held_out_indices: list[int]
    n_held_out: int
    positives_held_out: int
    correct: int
    true_positives: int
    accuracy: float
    recall: float | None


class ClassifierFoldResponse(BaseModel):
    """Every fold, and the summaries that agree and the ones that do not.

    ``mean_fold_accuracy`` averages the folds' accuracies; ``pooled_accuracy``
    adds the folds' tables and divides once. ``mean_defined_recall`` averages
    recall over the folds where it is defined, which is the convention
    averaging has to adopt; ``pooled_recall`` needs none.
    """

    folds: list[ClassifierFoldReading]
    classes_missing_from_a_fold: int
    mean_fold_accuracy: float
    pooled_accuracy: float
    accuracy_spread: float
    folds_with_defined_recall: int
    mean_defined_recall: float | None
    pooled_recall: float | None
    pooled_true_positives: int
    pooled_actual_positives: int


@router.post("/classifier-folds", response_model=ClassifierFoldResponse)
def classifier_folds(request: ClassifierFoldRequest) -> ClassifierFoldResponse:
    """Deal the crowd plain or stratified, refit the tree in each fold, pool.

    A training fold holding one class only is still fitted, because the tree
    is told the class count and answers that one class everywhere; the fold's
    held-out rows then show what such a fit is worth.
    """
    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    weights = np.asarray([point.y for point in request.points], dtype=np.float64)
    labels = np.asarray([float(point.label) for point in request.points])

    positions = _positions_dataset(labels, "is_adult")
    splits = KFold(
        n_folds=request.n_folds,
        random_seed=request.seed,
        stratified=request.stratified,
    ).split(positions)

    readings: list[ClassifierFoldReading] = []
    evaluations: list[MultiClassEvaluation] = []
    held_out_actuals: list[np.ndarray] = []
    held_out_predictions: list[np.ndarray] = []
    for split in splits:
        held = _held_out_positions(split.testing)
        training = np.setdiff1d(np.arange(labels.size), held)
        model = DecisionTreeClassifier(
            max_depth=TREE_DEPTH, n_known_classes=N_CLASSES
        ).fit(
            [
                Feature("height", heights[training]),
                Feature("weight", weights[training]),
            ],
            Feature("is_adult", labels[training]),
        )
        predicted = np.asarray(
            model.predict(
                [Feature("height", heights[held]), Feature("weight", weights[held])]
            ),
            dtype=np.float64,
        )
        evaluation = MultiClassEvaluation(labels[held], predicted, n_classes=N_CLASSES)
        evaluations.append(evaluation)
        held_out_actuals.append(labels[held])
        held_out_predictions.append(predicted)
        readings.append(
            ClassifierFoldReading(
                held_out_indices=[int(index) for index in held],
                n_held_out=evaluation.n_samples,
                positives_held_out=int(np.sum(labels[held] == POSITIVE_CLASS)),
                correct=evaluation.confusion_matrix.correct,
                true_positives=evaluation.confusion_matrix.true_positives_for(
                    POSITIVE_CLASS
                ),
                accuracy=float(evaluation.accuracy),
                recall=_recall_or_none(evaluation),
            )
        )

    result = ClassificationCrossValidationResult(evaluations)
    pooled = ClassificationEvaluation(
        np.concatenate(held_out_actuals), np.concatenate(held_out_predictions)
    )
    defined_recalls = [
        reading.recall for reading in readings if reading.recall is not None
    ]
    return ClassifierFoldResponse(
        folds=readings,
        classes_missing_from_a_fold=splits.classes_missing_from_a_fold(),
        mean_fold_accuracy=float(np.mean([reading.accuracy for reading in readings])),
        pooled_accuracy=float(result.pooled_accuracy),
        accuracy_spread=float(result.accuracy_spread),
        folds_with_defined_recall=len(defined_recalls),
        mean_defined_recall=(
            float(np.mean(defined_recalls)) if defined_recalls else None
        ),
        pooled_recall=_pooled_recall_or_none(pooled),
        pooled_true_positives=pooled.confusion_matrix.true_positives,
        pooled_actual_positives=pooled.confusion_matrix.actual_positive,
    )


def _recall_or_none(evaluation: MultiClassEvaluation) -> float | None:
    """Recall for the positive class, or ``None`` where the fold had none."""
    try:
        return float(evaluation.recall_for(POSITIVE_CLASS))
    except UndefinedMetricError:
        return None


def _pooled_recall_or_none(evaluation: ClassificationEvaluation) -> float | None:
    """Recall off the pooled table, or ``None`` if the crowd had no positive."""
    try:
        return float(evaluation.recall)
    except UndefinedMetricError:
        return None


# --- The leak, measured on the running example ------------------------------


class LeakRequest(BaseModel):
    """The points, the degree, the ridge penalty and the family of fold seeds."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(default=2, ge=1, le=MAX_DEGREE)
    penalty: float = Field(default=1.0, ge=0.0)
    first_seed: int = Field(default=0, ge=0, le=MAX_FIRST_SEED)
    n_seeds: int = Field(default=FAMILY_SIZE, ge=1, le=FAMILY_SIZE)

    model_config = {"extra": "forbid"}


class ArrangementScores(BaseModel):
    """One model's two arrangements of the standardiser, seed by seed.

    ``outside_scores`` are the mean five-fold held-out R squared when the
    standardiser was fitted on every row before the folds were dealt,
    ``inside_scores`` when it was refitted inside each fold. ``flattered_by``
    is outside less inside, averaged over the seeds, and ``seeds_flattered``
    counts the seeds on which outside scored higher.
    """

    outside_scores: list[float]
    inside_scores: list[float]
    mean_outside: float
    mean_inside: float
    flattered_by: float
    seeds_flattered: int
    largest_absolute_gap: float


class LeakResponse(BaseModel):
    """The leak for a penalised fit and for the plain least-squares control."""

    n_rows: int
    n_folds: int
    degree: int
    penalty: float
    ridge: ArrangementScores
    least_squares: ArrangementScores


def _terms_dataset(points: list[Point], degree: int) -> Dataset:
    """The polynomial terms of every row, which learn nothing from the rows.

    The term manufacturing depends only on the names and the degree, so
    building it once over all the rows leaks nothing; what the two
    arrangements differ on is where the standardiser's mean and spread are
    learned.
    """
    x_values, y_values = _throw_arrays(points)
    input_feature = Feature("t", x_values)
    terms = (
        PolynomialFeatures(degree=degree)
        .fit([input_feature])
        .transform([input_feature])
    )
    return Dataset(list(terms), Feature("h", y_values))


def _standardised_outside(terms: Dataset) -> Dataset:
    """Every row standardised by a mean and spread learned from every row."""
    scaled = Standardizer().fit(terms.input_features).transform(terms.input_features)
    return Dataset(list(scaled), terms.target_feature)


def _arrangement(
    terms: Dataset,
    outside: Dataset,
    model_factory: Callable[[], Regressor[Sequence[Feature], Feature]],
    first_seed: int,
    n_seeds: int,
) -> ArrangementScores:
    """Cross-validate one model both ways over the family of seeds."""
    outside_scores: list[float] = []
    inside_scores: list[float] = []
    for seed in range(first_seed, first_seed + n_seeds):
        folds = KFold(n_folds=FOLD_COUNT, random_seed=seed)
        outside_scores.append(
            float(
                CrossValidation(folds=folds)
                .evaluate(model_factory(), outside)
                .mean_r2_score
            )
        )
        inside = RegressionPipeline(
            steps=PipelineSteps(
                [PipelineStep(name="scale", transformer=Standardizer())]
            ),
            model=model_factory(),
        )
        inside_scores.append(
            float(CrossValidation(folds=folds).evaluate(inside, terms).mean_r2_score)
        )

    gaps = np.asarray(outside_scores) - np.asarray(inside_scores)
    return ArrangementScores(
        outside_scores=outside_scores,
        inside_scores=inside_scores,
        mean_outside=float(np.mean(outside_scores)),
        mean_inside=float(np.mean(inside_scores)),
        flattered_by=float(np.mean(gaps)),
        seeds_flattered=int(np.sum(gaps > 0)),
        largest_absolute_gap=float(np.max(np.abs(gaps))),
    )


@router.post("/leak", response_model=LeakResponse)
def leak(request: LeakRequest) -> LeakResponse:
    """Standardise before folding and inside each fold, on the same seeds.

    The ridge fit reads the columns' scale through its penalty, so a mean and
    spread learned from the held-out rows can move it; the least-squares fit
    is unchanged by any shift or scaling of its columns, so its two
    arrangements agree to rounding and it is the control.
    """
    terms = _terms_dataset(request.points, request.degree)
    outside = _standardised_outside(terms)
    return LeakResponse(
        n_rows=terms.n_samples,
        n_folds=FOLD_COUNT,
        degree=request.degree,
        penalty=request.penalty,
        ridge=_arrangement(
            terms,
            outside,
            lambda: RidgeRegression(penalty=request.penalty),
            request.first_seed,
            request.n_seeds,
        ),
        least_squares=_arrangement(
            terms,
            outside,
            MultipleLinearRegression,
            request.first_seed,
            request.n_seeds,
        ),
    )
