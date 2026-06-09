"""Everything the judging-a-classifier page reads off a confusion matrix.

Accuracy is one number and it hides two different mistakes, so the page reads
a classifier off its confusion matrix instead. One endpoint fits the library's
``LogisticRegression`` to the labelled crowd, judges it at the threshold the
reader chose, and hands back everything the page's widgets draw from that one
fit: the four counts and the rates, each person's chance and call, the
majority-class baseline the fit has to beat, a sweep of the threshold through
every hundredth with precision, recall, false positive rate and the F-scores
at each, the receiver operating characteristic and precision-recall curves
read at every distinct chance so their areas are exact, a reliability reading
of the chances against the outcomes, and the boundary shaded across the plane.
A second endpoint fits the library's ``MultinomialLogisticRegression`` to a
three-class crowd and judges any subset of its rows through
``MultiClassEvaluation``, with the class count stated or inferred, which is
the page's section on why the width has to be said. The third folds the crowd
with ``KFold`` and pools the folds' tables, and is the held-out page's
argument applied here.

The library gives the counts and every ratio of them, precision, recall,
specificity, F1 and the per-class and macro figures, and refuses each one by
name when its denominator is empty. What it does not have is the sweep
bookkeeping, and that is arithmetic on its numbers rather than a fit: the area
under the ROC curve is a count of adult-child pairs ranked the right way, the
average precision is the step area under the precision-recall points, an
F-score at another beta is the same harmonic mean with one weight changed,
and a reliability bin is a count of outcomes against a mean of chances. All
of it is computed here from what the library's evaluation reported, and the
page says so where it quotes them.

The heights and weights are standardised before the ascent, because a gradient
walk on columns near 150 and 50 oscillates where the same walk on standardised
columns settles, and inside the fold endpoint the standardiser is fitted inside
each fold so a held-out row never lends its mean to the fit.

Precision is undefined when nothing was called positive, recall when no
positive was present, and the response says ``null`` there rather than
inventing a zero; a fold with no adults has the same ``0/0`` for recall, and
that gap is the whole argument for pooling the folds' tables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    ClassificationCrossValidationResult,
    ClassificationEvaluation,
    Dataset,
    Feature,
    KFold,
    LogisticRegression,
    MultiClassEvaluation,
    MultinomialLogisticRegression,
    Standardizer,
    UndefinedMetricError,
)
from oop_ml.core.data.predictions import Predictions
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(
    prefix="/concepts/judging-a-classifier", tags=["judging-a-classifier"]
)

LEARNING_RATE = 1.0
"""The stride of the ascent, on standardised columns where a stride of one is tame."""

MAX_EPOCHS = 2000
"""How long the ascent is allowed.

A separable crowd has no maximum likelihood, so the walk would otherwise run
its full ten thousand passes steepening a boundary it placed long before. Two
thousand passes put the boundary where a longer climb leaves it and cost a
twentieth of a second per fit, which is what lets the fold endpoint refit ten
times inside one request.
"""

TOLERANCE = 1e-6
"""Stop once no coefficient moves further than this in a whole pass."""

DEFAULT_THRESHOLD = 0.5
"""Where the log-odds are zero, the point a classifier commits at unless told."""

SWEEP_THRESHOLDS = [round(step / 100, 2) for step in range(1, 100)]
"""The sweep is read at every hundredth from 0.01 to 0.99."""

POSITIVE_CLASS = 1
"""The class the rates are read for, which the pages' crowd calls adult."""

N_CLASSES = 2
"""Stated to every fold's evaluation, so a fold missing a class still counts."""

FOLD_SEED = 6
"""The deal is seeded so the page can quote what the folding panel shows."""

MAX_FOLDS = 10
"""The most folds the panel may ask for."""

RELIABILITY_BINS = 5
"""The chances are binned into fifths for the reliability reading."""

F_BETAS = {"f_half": 0.5, "f_one": 1.0, "f_two": 2.0}
"""The three F-scores the page compares: precision-leaning, even, recall-leaning."""

THREE_CLASSES = 3
"""The three-class crowd has children, teenagers and adults."""

THREE_CLASS_NAMES = ("child", "teenager", "adult")
"""What the pages call classes 0, 1 and 2."""

MULTICLASS_MAX_EPOCHS = 500
"""The three-class ascent's allowance, the multiclass page's own figure."""


def _boundary(threshold: float = DEFAULT_THRESHOLD) -> LogisticRegression:
    """The page's classifier, configured once so every endpoint fits the same one."""
    return LogisticRegression(
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        tolerance=TOLERANCE,
        threshold=threshold,
    )


class _StandardisedModel:
    """A fitted model that standardises what it is asked about first.

    ``region_grid`` asks a model for its answer across a lattice in the raw
    units the points arrived in, and the models here are fitted on
    standardised columns, so this stands between the two. Plumbing rather
    than a model.
    """

    def __init__(self, standardizer: Standardizer, model: LogisticRegression) -> None:
        self._standardizer = standardizer
        self._model = model

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        return self._model.predict(self._standardizer.transform(input_values))


def _or_none(read: Callable[..., float], *arguments: int) -> float | None:
    """A rate, or ``None`` where the library says its denominator is empty."""
    try:
        return float(read(*arguments))
    except UndefinedMetricError:
        return None


def _f_beta(precision: float | None, recall: float | None, beta: float) -> float | None:
    """The F-score at one beta from a precision and a recall.

    The library's own ``f1_score`` is this at beta one, and the tests pin the
    two against each other. ``None`` where either rate is undefined or both
    are zero, which is the library's rule for F1 carried to the other betas.
    """
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return None
    weight = beta * beta
    return (1.0 + weight) * precision * recall / (weight * precision + recall)


class EvaluateRequest(BaseModel):
    """The labelled crowd, and the chance above which a person is called adult."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0.0, lt=1.0)

    model_config = {"extra": "forbid"}


class ConfusionCounts(BaseModel):
    """The four cells every rate on the page is a ratio of."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class Rates(BaseModel):
    """The ratios read off the counts.

    ``precision`` is ``None`` when the threshold called nobody positive, which
    leaves it 0/0 rather than zero, and the F-scores are ``None`` wherever
    precision is or where both rates are zero. ``f_one`` is the library's own
    F1; ``f_half`` and ``f_two`` are the same harmonic mean with recall
    weighted a quarter as much and four times as much.
    """

    accuracy: float
    precision: float | None
    recall: float
    specificity: float
    false_positive_rate: float
    f_one: float | None
    f_half: float | None
    f_two: float | None


class Baseline(BaseModel):
    """What calling everybody the majority class scores on the same crowd.

    ``recall`` is 1.0 if the majority class is adult and 0.0 otherwise, and
    ``precision`` is ``None`` when nobody was called adult.
    """

    majority_label: int
    accuracy: float
    precision: float | None
    recall: float


class ThresholdReading(BaseModel):
    """Every rate at one threshold of the sweep, with the counts behind them."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    accuracy: float
    precision: float | None
    recall: float
    false_positive_rate: float
    f_one: float | None
    f_half: float | None
    f_two: float | None


class RocPoint(BaseModel):
    """One corner of the receiver operating characteristic.

    Read at one distinct chance, calling everyone at or above it adult, so the
    curve turns exactly where a person changes side and nowhere else. The two
    ends, nobody called and everybody called, are included.
    """

    threshold: float | None
    false_positive_rate: float
    recall: float


class PrecisionRecallPoint(BaseModel):
    """One corner of the precision-recall curve, at one distinct chance."""

    threshold: float
    recall: float
    precision: float


class BestThreshold(BaseModel):
    """The threshold of the sweep where one F-score peaked, and its value."""

    threshold: float
    value: float


class ReliabilityBin(BaseModel):
    """People whose chance fell in one fifth, and how many really were adults.

    ``observed_share`` is ``None`` for an empty bin.
    """

    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_share: float | None


class Reliability(BaseModel):
    """The chances against the outcomes.

    ``expected_calibration_error`` is the mean over people of the gap between
    their bin's mean chance and its observed share; ``brier_score`` is the
    mean squared gap between each person's chance and their 0 or 1 outcome.
    """

    bins: list[ReliabilityBin]
    expected_calibration_error: float
    brier_score: float
    mean_probability: float
    observed_rate: float


class EvaluateResponse(BaseModel):
    """The fit judged at the chosen threshold, and everything the sweep draws.

    ``probabilities`` and ``predictions`` are one per person in request order,
    so the widget can colour each dot by which cell it fell in. ``roc_area``
    is the share of adult-child pairs where the adult's chance is the higher,
    ties counted a half, which is the number the ROC curve's area equals.
    ``average_precision`` is the step area under the precision-recall points.
    """

    probabilities: list[float]
    predictions: list[int]
    prevalence: float
    counts: ConfusionCounts
    rates: Rates
    baseline: Baseline
    curve: list[ThresholdReading]
    roc: list[RocPoint]
    roc_area: float
    precision_recall: list[PrecisionRecallPoint]
    average_precision: float
    best: dict[str, BestThreshold]
    reliability: Reliability
    epochs_run: int
    converged: bool
    regions: RegionGrid


def _rates_of(evaluation: ClassificationEvaluation) -> Rates:
    """Every rate the page reads at one threshold, from one evaluation."""
    precision = _or_none(lambda: evaluation.precision)
    recall = float(evaluation.recall)
    specificity = float(evaluation.specificity)
    return Rates(
        accuracy=float(evaluation.accuracy),
        precision=precision,
        recall=recall,
        specificity=specificity,
        false_positive_rate=1.0 - specificity,
        f_one=_or_none(lambda: evaluation.f1_score),
        f_half=_f_beta(precision, recall, F_BETAS["f_half"]),
        f_two=_f_beta(precision, recall, F_BETAS["f_two"]),
    )


def _reading_at(
    threshold: float, labels: Feature, chances: np.ndarray
) -> ThresholdReading:
    """The sweep's entry at one threshold, calling everyone at or above it adult."""
    evaluation = ClassificationEvaluation(
        labels, (chances >= threshold).astype(np.float64)
    )
    matrix = evaluation.confusion_matrix
    rates = _rates_of(evaluation)
    return ThresholdReading(
        threshold=threshold,
        true_positives=matrix.true_positives,
        false_positives=matrix.false_positives,
        false_negatives=matrix.false_negatives,
        true_negatives=matrix.true_negatives,
        accuracy=rates.accuracy,
        precision=rates.precision,
        recall=rates.recall,
        false_positive_rate=rates.false_positive_rate,
        f_one=rates.f_one,
        f_half=rates.f_half,
        f_two=rates.f_two,
    )


def _roc_points(labels: Feature, chances: np.ndarray) -> list[RocPoint]:
    """The ROC corners, one per distinct chance from the highest down.

    Starts at nobody called, which is the origin, and ends at everybody
    called, which is the top right corner.
    """
    points = [RocPoint(threshold=None, false_positive_rate=0.0, recall=0.0)]
    for threshold in sorted({float(chance) for chance in chances}, reverse=True):
        evaluation = ClassificationEvaluation(
            labels, (chances >= threshold).astype(np.float64)
        )
        points.append(
            RocPoint(
                threshold=threshold,
                false_positive_rate=1.0 - float(evaluation.specificity),
                recall=float(evaluation.recall),
            )
        )
    return points


def _roc_area(labels: np.ndarray, chances: np.ndarray) -> float:
    """The share of adult-child pairs ranked the right way, ties a half."""
    adults = chances[labels == POSITIVE_CLASS]
    children = chances[labels != POSITIVE_CLASS]
    above = np.sum(adults[:, None] > children[None, :])
    tied = np.sum(adults[:, None] == children[None, :])
    return float((above + 0.5 * tied) / (adults.size * children.size))


def _precision_recall_points(
    labels: Feature, chances: np.ndarray
) -> list[PrecisionRecallPoint]:
    """The precision-recall corners, one per distinct chance from the highest down.

    Every distinct chance calls at least one person adult, so precision is
    defined at every corner; the library refuses the point above the highest
    chance, where nobody is called, and the curve starts without it rather
    than with a conventional 1.0.
    """
    points = []
    for threshold in sorted({float(chance) for chance in chances}, reverse=True):
        evaluation = ClassificationEvaluation(
            labels, (chances >= threshold).astype(np.float64)
        )
        points.append(
            PrecisionRecallPoint(
                threshold=threshold,
                recall=float(evaluation.recall),
                precision=float(evaluation.precision),
            )
        )
    return points


def _average_precision(points: list[PrecisionRecallPoint]) -> float:
    """The step area under the precision-recall corners.

    Each corner contributes its precision times the recall it added over the
    corner before it, which is the usual definition and deliberately not a
    trapezoid, since precision between two corners is not on the line joining
    them.
    """
    total = 0.0
    previous_recall = 0.0
    for point in points:
        total += (point.recall - previous_recall) * point.precision
        previous_recall = point.recall
    return float(total)


def _best_thresholds(curve: list[ThresholdReading]) -> dict[str, BestThreshold]:
    """Where each F-score peaked over the sweep, the lowest threshold on a tie."""
    best: dict[str, BestThreshold] = {}
    for name in F_BETAS:
        defined = [
            (reading.threshold, getattr(reading, name))
            for reading in curve
            if getattr(reading, name) is not None
        ]
        threshold, value = max(defined, key=lambda pair: (pair[1], -pair[0]))
        best[name] = BestThreshold(threshold=threshold, value=value)
    return best


def _reliability(labels: np.ndarray, chances: np.ndarray) -> Reliability:
    """The chances binned into fifths against the share of adults in each."""
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    bins = []
    weighted_gap = 0.0
    for index in range(RELIABILITY_BINS):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == RELIABILITY_BINS - 1:
            inside = (chances >= lower) & (chances <= upper)
        else:
            inside = (chances >= lower) & (chances < upper)
        count = int(np.sum(inside))
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    count=0,
                    mean_probability=None,
                    observed_share=None,
                )
            )
            continue
        mean_probability = float(np.mean(chances[inside]))
        observed_share = float(np.mean(labels[inside] == POSITIVE_CLASS))
        weighted_gap += count * abs(mean_probability - observed_share)
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_probability=mean_probability,
                observed_share=observed_share,
            )
        )
    outcomes = (labels == POSITIVE_CLASS).astype(np.float64)
    return Reliability(
        bins=bins,
        expected_calibration_error=float(weighted_gap / labels.size),
        brier_score=float(np.mean((chances - outcomes) ** 2)),
        mean_probability=float(np.mean(chances)),
        observed_rate=float(np.mean(outcomes)),
    )


def _baseline(labels: Feature) -> Baseline:
    """Call everybody the majority class and judge that through the library."""
    values = np.asarray(labels.values)
    adults = int(np.sum(values == POSITIVE_CLASS))
    majority = POSITIVE_CLASS if adults > values.size - adults else 1 - POSITIVE_CLASS
    evaluation = ClassificationEvaluation(labels, np.full(values.size, float(majority)))
    return Baseline(
        majority_label=majority,
        accuracy=float(evaluation.accuracy),
        precision=_or_none(lambda: evaluation.precision),
        recall=float(evaluation.recall),
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Fit the boundary, judge it at the threshold, and sweep the threshold.

    A crowd holding a single class, or a constant column, is refused by the
    library with its own typed error, which the shared handler turns into a
    422.
    """
    heights, weights, labels = features_of(request.points)

    standardizer = Standardizer().fit([heights, weights])
    standardised = standardizer.transform([heights, weights])
    model = _boundary(request.threshold).fit(standardised, labels)

    probabilities = model.predict_probability(standardised)
    chances = np.asarray(probabilities.values, dtype=np.float64)
    label_values = np.asarray(labels.values, dtype=np.float64)
    predicted = model.predict(standardised)
    evaluation = ClassificationEvaluation(labels, predicted)
    matrix = evaluation.confusion_matrix

    curve = [_reading_at(threshold, labels, chances) for threshold in SWEEP_THRESHOLDS]
    precision_recall = _precision_recall_points(labels, chances)

    return EvaluateResponse(
        probabilities=[float(chance) for chance in chances],
        predictions=[int(label) for label in np.asarray(predicted)],
        prevalence=float(np.mean(label_values == POSITIVE_CLASS)),
        counts=ConfusionCounts(
            true_positives=matrix.true_positives,
            true_negatives=matrix.true_negatives,
            false_positives=matrix.false_positives,
            false_negatives=matrix.false_negatives,
        ),
        rates=_rates_of(evaluation),
        baseline=_baseline(labels),
        curve=curve,
        roc=_roc_points(labels, chances),
        roc_area=_roc_area(label_values, chances),
        precision_recall=precision_recall,
        average_precision=_average_precision(precision_recall),
        best=_best_thresholds(curve),
        reliability=_reliability(label_values, chances),
        epochs_run=int(model.epochs_run),
        converged=bool(model.converged),
        regions=region_grid(_StandardisedModel(standardizer, model), request.points),
    )


class ThreeClassPoint(BaseModel):
    """One person, their two measurements, and which of three classes they are.

    Not the shared ``LabelledPoint``, whose label is bounded to two classes.
    """

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=THREE_CLASSES - 1)


class MulticlassRequest(BaseModel):
    """The three-class crowd, which of its rows to judge, and whether to state the width.

    The model is always fitted to the whole crowd. ``judged`` names the rows
    the table is built from, in request order, and left out means every row;
    a held-out fold is a list like this. ``stated_width`` is the class count
    handed to the evaluation, and left out means the evaluation must infer
    it from the judged rows' true classes, which is the case the page shows
    refusing.
    """

    points: list[ThreeClassPoint] = Field(
        min_length=THREE_CLASSES, max_length=MAX_POINTS
    )
    judged: list[int] | None = Field(default=None, min_length=1)
    stated_width: int | None = Field(default=None, ge=1, le=10)

    model_config = {"extra": "forbid"}


class ClassReading(BaseModel):
    """One class's row and column of the table, and the rates read from them.

    ``precision`` is ``None`` when the model never called this class,
    ``recall`` when no judged row belongs to it, and ``f_one`` when either
    is or both are zero.
    """

    class_index: int
    name: str
    actually_are: int
    predicted_as: int
    true_positives: int
    precision: float | None
    recall: float | None
    f_one: float | None


class MulticlassResponse(BaseModel):
    """The table and every figure the page reads from it.

    ``counts`` has the true class down the rows and the predicted class
    across the columns. The macro figures are ``None`` whenever any class's
    rate is, because averaging over the classes that happen to be defined
    would change the question. ``micro_precision`` and ``micro_recall`` equal
    ``accuracy`` here, since every row gets exactly one call, and are
    reported so the page can show that they do.
    """

    n_classes: int
    class_names: list[str]
    width_was_stated: bool
    n_judged: int
    counts: list[list[int]]
    accuracy: float
    classes: list[ClassReading]
    macro_precision: float | None
    macro_recall: float | None
    macro_f_one: float | None
    micro_precision: float
    micro_recall: float
    predictions: list[int]
    judged_labels: list[int]


@router.post("/multiclass", response_model=MulticlassResponse)
def judge_multiclass(request: MulticlassRequest) -> MulticlassResponse:
    """Fit three classes to the whole crowd and judge the rows asked for.

    A crowd missing one of the three classes is refused here in words. Judged
    rows whose true classes leave a gap are refused by the library when the
    width is inferred and accepted when it is stated; a stated width smaller
    than a class the rows name is refused by the library by name. Each is a
    typed 422.
    """
    present = {point.label for point in request.points}
    missing = [
        name for index, name in enumerate(THREE_CLASS_NAMES) if index not in present
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"there is no {' or '.join(missing)} in the crowd, and a choice "
                "among three classes needs at least one person in each"
            ),
        )
    judged = (
        list(range(len(request.points))) if request.judged is None else request.judged
    )
    outside = [index for index in judged if not 0 <= index < len(request.points)]
    if outside:
        raise HTTPException(
            status_code=422,
            detail=(
                f"judged row {outside[0]} is outside a crowd of "
                f"{len(request.points)} people"
            ),
        )

    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])
    labels = Feature("age_group", [float(point.label) for point in request.points])

    standardizer = Standardizer().fit([heights, weights])
    standardised = standardizer.transform([heights, weights])
    model = MultinomialLogisticRegression(
        learning_rate=LEARNING_RATE,
        max_epochs=MULTICLASS_MAX_EPOCHS,
        tolerance=TOLERANCE,
    ).fit(standardised, labels)

    predicted = np.asarray(model.predict(standardised), dtype=np.float64)
    actual = np.asarray(labels.values, dtype=np.float64)
    evaluation = MultiClassEvaluation(
        actual[judged], predicted[judged], n_classes=request.stated_width
    )
    table = evaluation.confusion_matrix
    names = [
        THREE_CLASS_NAMES[index] if index < THREE_CLASSES else f"class {index}"
        for index in range(evaluation.n_classes)
    ]

    classes = [
        ClassReading(
            class_index=index,
            name=names[index],
            actually_are=table.actually_are(index),
            predicted_as=table.predicted_as(index),
            true_positives=table.true_positives_for(index),
            precision=_or_none(evaluation.precision_for, index),
            recall=_or_none(evaluation.recall_for, index),
            f_one=_or_none(evaluation.f1_for, index),
        )
        for index in range(evaluation.n_classes)
    ]

    return MulticlassResponse(
        n_classes=evaluation.n_classes,
        class_names=names,
        width_was_stated=request.stated_width is not None,
        n_judged=evaluation.n_samples,
        counts=[[int(value) for value in row] for row in table.counts],
        accuracy=float(evaluation.accuracy),
        classes=classes,
        macro_precision=_or_none(lambda: evaluation.macro_precision),
        macro_recall=_or_none(lambda: evaluation.macro_recall),
        macro_f_one=_or_none(lambda: evaluation.macro_f1_score),
        micro_precision=float(evaluation.micro_precision),
        micro_recall=float(evaluation.micro_recall),
        predictions=[int(value) for value in predicted[judged]],
        judged_labels=[int(value) for value in actual[judged]],
    )


class FoldRequest(BaseModel):
    """The crowd, how many folds to deal it into, and whether to stratify."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_folds: int = Field(default=5, ge=2, le=MAX_FOLDS)
    stratified: bool = False

    model_config = {"extra": "forbid"}


class FoldReading(BaseModel):
    """What one fold's held-out rows scored.

    ``precision`` is ``None`` when the fold's fit called nobody positive, and
    ``recall`` is ``None`` when the fold held no positive at all, which is the
    0/0 that averaging over folds has to paper over.
    """

    held_out: int
    positives_held_out: int
    accuracy: float
    precision: float | None
    recall: float | None


class FoldResponse(BaseModel):
    """Every fold's verdict, and the verdict pooled across them.

    ``mean_fold_accuracy`` averages the folds' accuracies, the route the
    library declines; ``pooled_accuracy`` adds the folds' tables first and
    divides once. They differ whenever the folds are not the same size. The
    pooled precision and recall are for the positive class; the macro figures
    average both classes' rates off the same pooled table.
    """

    folds: list[FoldReading]
    mean_fold_accuracy: float
    pooled_accuracy: float
    pooled_precision: float | None
    pooled_recall: float
    pooled_macro_precision: float | None
    pooled_macro_recall: float
    accuracy_spread: float
    classes_missing_from_a_fold: int


@router.post("/fold", response_model=FoldResponse)
def fold(request: FoldRequest) -> FoldResponse:
    """Deal the crowd into folds, refit inside each, and pool the verdicts.

    More folds than people, or a training fold holding a single class, is
    refused by the library with its own typed error and becomes a 422.
    """
    heights, weights, labels = features_of(request.points)
    dataset = Dataset([heights, weights], labels)

    splits = KFold(
        n_folds=request.n_folds,
        random_seed=FOLD_SEED,
        stratified=request.stratified,
    ).split(dataset)

    fold_evaluations: list[MultiClassEvaluation] = []
    readings: list[FoldReading] = []
    held_out_actuals: list[np.ndarray] = []
    held_out_predictions: list[np.ndarray] = []

    for split in splits:
        training, testing = split.training, split.testing

        standardizer = Standardizer().fit(training.input_features)
        model = _boundary().fit(
            standardizer.transform(training.input_features), training.target_feature
        )

        actual = np.asarray(testing.target_feature.values, dtype=np.float64)
        predicted = np.asarray(
            model.predict(standardizer.transform(testing.input_features)),
            dtype=np.float64,
        )
        evaluation = MultiClassEvaluation(actual, predicted, n_classes=N_CLASSES)

        fold_evaluations.append(evaluation)
        held_out_actuals.append(actual)
        held_out_predictions.append(predicted)
        readings.append(
            FoldReading(
                held_out=evaluation.n_samples,
                positives_held_out=int(np.sum(actual == POSITIVE_CLASS)),
                accuracy=float(evaluation.accuracy),
                precision=_or_none(evaluation.precision_for, POSITIVE_CLASS),
                recall=_or_none(evaluation.recall_for, POSITIVE_CLASS),
            )
        )

    result = ClassificationCrossValidationResult(fold_evaluations)
    pooled = ClassificationEvaluation(
        np.concatenate(held_out_actuals), np.concatenate(held_out_predictions)
    )

    return FoldResponse(
        folds=readings,
        mean_fold_accuracy=float(
            np.mean([evaluation.accuracy for evaluation in fold_evaluations])
        ),
        pooled_accuracy=float(result.pooled_accuracy),
        pooled_precision=_or_none(lambda: pooled.precision),
        pooled_recall=float(pooled.recall),
        pooled_macro_precision=_or_none(lambda: result.pooled_macro_precision),
        pooled_macro_recall=float(result.pooled_macro_recall),
        accuracy_spread=float(result.accuracy_spread),
        classes_missing_from_a_fold=splits.classes_missing_from_a_fold(),
    )
