"""The confusion matrix, its four rates, and folding, for the judging-a-classifier page.

Accuracy is one number and it hides two different mistakes, so the page reads
a classifier off its confusion matrix instead. One endpoint fits the library's
``LogisticRegression`` to the labelled crowd, with the threshold the reader
chose, and hands back the four counts, the four rates, each person's fitted
chance and label, the boundary shaded across the plane, and precision against
recall over a sweep of thresholds so the trade between them is a curve rather
than a sentence. The other folds the crowd with the library's ``KFold``, plain
or stratified, refits inside every fold, and reports what each fold scored
beside the figures pooled across them, which is the section on why pooling is
the undefined case rather than a preference.

Two facts about the library shape this module. The heights and weights are
standardised before the ascent, because a gradient walk on columns near 150
and 50 oscillates where the same walk on standardised columns settles, and the
standardiser is fitted inside each fold so a held-out row never lends its mean
to the fit. And ``CrossValidation.evaluate_classifier`` takes a
``MultiClassClassifier``, which the binary ``LogisticRegression`` is not, so
the fold endpoint runs the same loop over the public pieces itself, one
``MultiClassEvaluation`` per fold with the class count stated, gathered into
the library's own ``ClassificationCrossValidationResult`` for the pooling. It
is one run, recorded, which is the honest kind of loop.

Precision is the one rate that can be undefined after a fit, since a threshold
high enough calls nobody positive, and the response says ``null`` there rather
than inventing a zero; recall on a fold with no positives is the same ``0/0``
and is reported the same way, because that gap is the page's whole argument.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    ClassificationCrossValidationResult,
    ClassificationEvaluation,
    Dataset,
    Feature,
    KFold,
    LogisticRegression,
    MultiClassEvaluation,
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
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(
    prefix="/concepts/judging-a-classifier", tags=["judging-a-classifier"]
)

LEARNING_RATE = 1.0
"""The stride of the ascent, on standardised columns where a stride of one is tame."""

MAX_EPOCHS = 2000
"""How long the ascent is allowed.

The crowd is separable, and on separable data the likelihood has no maximum,
so the walk would otherwise run its full ten thousand passes steepening a
boundary it placed long before. Two thousand passes put the boundary where a
longer climb leaves it and cost a twentieth of a second per fit, which is what
lets the fold endpoint refit ten times inside one request.
"""

TOLERANCE = 1e-6
"""Stop once no coefficient moves further than this in a whole pass."""

DEFAULT_THRESHOLD = 0.5
"""Where the log-odds are zero, the point a classifier commits at unless told."""

SWEEP_THRESHOLDS = [round(step / 100, 2) for step in range(1, 100)]
"""The precision-recall curve is read at every hundredth from 0.01 to 0.99."""

POSITIVE_CLASS = 1
"""The class the four rates are read for, which the pages' crowd calls adult."""

N_CLASSES = 2
"""Stated to every fold's evaluation, so a fold missing a class still counts."""

FOLD_SEED = 6
"""The deal is seeded so the page can quote what the folding panel shows.

Under this seed, plain five-fold on the pages' eleven-person crowd deals one
fold holding no adult and another holding no child, which is the case the
section exists to show. Stratified folding under the same seed deals every
class into every fold.
"""

MAX_FOLDS = 10
"""The most folds the panel may ask for."""


def _boundary(threshold: float = DEFAULT_THRESHOLD) -> LogisticRegression:
    """The page's classifier, configured once so both endpoints fit the same one."""
    return LogisticRegression(
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        tolerance=TOLERANCE,
        threshold=threshold,
    )


class _StandardisedBoundary:
    """A fitted boundary that standardises what it is asked about first.

    ``region_grid`` asks a model for its answer across a lattice in the raw
    units the points arrived in, and the boundary was fitted on standardised
    columns, so this stands between the two. It is plumbing rather than a
    model, and it exists so the shared lattice is reused rather than rewritten
    with a standardiser threaded through it.
    """

    def __init__(self, standardizer: Standardizer, model: LogisticRegression) -> None:
        self._standardizer = standardizer
        self._model = model

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        return self._model.predict(self._standardizer.transform(input_values))


def _precision_or_none(evaluation: ClassificationEvaluation) -> float | None:
    """Precision, or ``None`` when nothing was called positive and it is 0/0."""
    try:
        return float(evaluation.precision)
    except UndefinedMetricError:
        return None


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
    """The four ratios read off the counts.

    ``precision`` is ``None`` when the threshold called nobody positive, which
    leaves it 0/0 rather than zero.
    """

    accuracy: float
    precision: float | None
    recall: float
    specificity: float


class ThresholdReading(BaseModel):
    """Precision and recall at one threshold of the sweep."""

    threshold: float
    precision: float | None
    recall: float


class EvaluateResponse(BaseModel):
    """The fit judged at the chosen threshold, and the curve of every threshold.

    ``probabilities`` and ``predictions`` are one per person in request order,
    so the widget can colour each dot by which cell it fell in.
    """

    probabilities: list[float]
    predictions: list[int]
    counts: ConfusionCounts
    rates: Rates
    curve: list[ThresholdReading]
    regions: RegionGrid


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Fit the boundary, judge it at the threshold, and sweep the threshold.

    A crowd holding a single class is refused by the library with its own
    typed error, which the shared handler turns into a 422.
    """
    heights, weights, labels = features_of(request.points)

    standardizer = Standardizer().fit([heights, weights])
    standardised = standardizer.transform([heights, weights])
    model = _boundary(request.threshold).fit(standardised, labels)

    probabilities = model.predict_probability(standardised)
    predicted = model.predict(standardised)
    evaluation = ClassificationEvaluation(labels, predicted)
    matrix = evaluation.confusion_matrix

    curve = []
    for threshold in SWEEP_THRESHOLDS:
        swept = ClassificationEvaluation(labels, probabilities.above(threshold))
        curve.append(
            ThresholdReading(
                threshold=threshold,
                precision=_precision_or_none(swept),
                recall=float(swept.recall),
            )
        )

    return EvaluateResponse(
        probabilities=[float(chance) for chance in np.asarray(probabilities)],
        predictions=[int(label) for label in np.asarray(predicted)],
        counts=ConfusionCounts(
            true_positives=matrix.true_positives,
            true_negatives=matrix.true_negatives,
            false_positives=matrix.false_positives,
            false_negatives=matrix.false_negatives,
        ),
        rates=Rates(
            accuracy=float(evaluation.accuracy),
            precision=_precision_or_none(evaluation),
            recall=float(evaluation.recall),
            specificity=float(evaluation.specificity),
        ),
        curve=curve,
        regions=region_grid(_StandardisedBoundary(standardizer, model), request.points),
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
                precision=_class_rate_or_none(evaluation.precision_for),
                recall=_class_rate_or_none(evaluation.recall_for),
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
        pooled_precision=_precision_or_none(pooled),
        pooled_recall=float(pooled.recall),
        pooled_macro_precision=_macro_precision_or_none(result),
        pooled_macro_recall=float(result.pooled_macro_recall),
        accuracy_spread=float(result.accuracy_spread),
        classes_missing_from_a_fold=splits.classes_missing_from_a_fold(),
    )


def _class_rate_or_none(rate_for) -> float | None:
    """A per-class rate for the positive class, or ``None`` where it is 0/0."""
    try:
        return float(rate_for(POSITIVE_CLASS))
    except UndefinedMetricError:
        return None


def _macro_precision_or_none(
    result: ClassificationCrossValidationResult,
) -> float | None:
    """Macro precision off the pooled table, or ``None`` if a class was never called."""
    try:
        return float(result.pooled_macro_precision)
    except UndefinedMetricError:
        return None
