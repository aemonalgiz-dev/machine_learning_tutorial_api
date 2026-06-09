"""Fit a pass-or-fail probability curve, for the logistic regression page.

One input, hours studied, and a yes-or-no outcome. The library's
``LogisticRegression`` fits by gradient ascent, which matters for a widget,
since a learner will drag the two classes fully apart and a Newton solver
would refuse at that point while the ascent simply steepens toward a step and
stops at its epoch cap. The response carries the fitted slope and intercept,
the boundary where the probability crosses one half, the accuracy, and a dense
sampling of the probability curve for the widget to draw.
"""

from __future__ import annotations

import math

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    ClassificationEvaluation,
    DesignMatrix,
    Feature,
    LogisticRegression,
    MultipleLinearRegression,
)
from oop_ml.core.logistic import stable_logistic
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/logistic-regression", tags=["logistic-regression"])

CURVE_SAMPLES = 81
"""How many points the probability curve is sampled at across the data."""

FLAT_SLOPE = 1e-9
"""Below this the curve is level and no boundary crossing exists to report."""


class Outcome(BaseModel):
    """One person, how long they studied, and whether they passed."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=1)


MAX_LEARNING_RATE = 50.0
"""The hottest climb the widget may ask for, hot enough to destabilise."""


class LogisticFitRequest(BaseModel):
    """The observed outcomes to fit the curve through, and how boldly to climb.

    The learning rate is exposed so the page can show the climb overshooting.
    A logistic gradient is bounded, so the weights never overflow to
    non-finite the way least squares can, though at a hot rate each stride
    flies past the top, the slope balloons with the epochs instead of
    settling, and the boundary wanders.
    """

    points: list[Outcome] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    learning_rate: float = Field(default=0.1, gt=0.0, le=MAX_LEARNING_RATE)

    model_config = {"extra": "forbid"}


class StraightLine(BaseModel):
    """The ordinary least-squares line through the zero-and-one labels.

    The page draws it to show why a straight line cannot be a probability:
    it leaves the band between zero and one on both sides of the data.
    """

    slope: float
    intercept: float


class LogisticFitResponse(BaseModel):
    """The fitted curve and everything the widget reads off it.

    ``boundary`` is where the probability crosses one half, ``None`` when the
    fitted curve is level and never crosses. ``converged`` says whether the
    climb met its tolerance or ran out of passes, which the page reports
    rather than hides. ``odds_multiplier`` is ``e`` to the slope, what one
    extra hour multiplies the odds by, and ``probability_at_zero`` is the
    sigmoid of the intercept.
    """

    slope: float
    intercept: float
    boundary: float | None
    accuracy: float
    curve: list[CurvePoint]
    gaps: list[float]
    gap_total: float
    probabilities: list[float]
    log_likelihood: float
    log_loss: float
    likelihood: float
    epochs_run: int
    converged: bool
    odds_multiplier: float
    probability_at_zero: float
    straight_line: StraightLine


@router.post("/fit", response_model=LogisticFitResponse)
def fit(request: LogisticFitRequest) -> LogisticFitResponse:
    """Fit the probability curve and sample it for drawing.

    A single-class target is refused by the library with its own typed error,
    which the shared handler turns into a 422; nothing here anticipates it.
    """
    hours = Feature("hours", [point.x for point in request.points])
    passed = Feature("passed", [float(point.label) for point in request.points])

    model = LogisticRegression(learning_rate=request.learning_rate).fit([hours], passed)
    slope = float(next(iter(model.coefficients)).value)
    intercept = float(model.intercept)

    boundary: float | None
    if abs(slope) < FLAT_SLOPE:
        boundary = None
    else:
        boundary = -intercept / slope

    x_values = [point.x for point in request.points]
    grid = np.linspace(min(x_values), max(x_values), CURVE_SAMPLES)
    probabilities = np.asarray(
        model.predict_probability([Feature("hours", grid)]), dtype=np.float64
    )

    fitted_probabilities = np.asarray(
        model.predict_probability([hours]), dtype=np.float64
    )
    gaps = [
        float(point.label - probability)
        for point, probability in zip(request.points, fitted_probabilities, strict=True)
    ]

    labels = np.asarray([point.label for point in request.points], dtype=np.float64)
    log_likelihood = _log_likelihood(labels, fitted_probabilities)
    straight = MultipleLinearRegression().fit([hours], passed)

    return LogisticFitResponse(
        slope=slope,
        intercept=intercept,
        boundary=boundary,
        accuracy=float(model.score([hours], passed)),
        curve=[
            CurvePoint(x=float(x), y=float(p))
            for x, p in zip(grid, probabilities, strict=True)
        ],
        gaps=gaps,
        gap_total=float(sum(gaps)),
        probabilities=[float(value) for value in fitted_probabilities],
        log_likelihood=log_likelihood,
        log_loss=-log_likelihood / labels.size,
        likelihood=math.exp(log_likelihood),
        epochs_run=int(model.epochs_run),
        converged=bool(model.converged),
        odds_multiplier=math.exp(slope),
        probability_at_zero=float(stable_logistic(np.asarray([intercept]))[0]),
        straight_line=StraightLine(
            slope=float(next(iter(straight.coefficients)).value),
            intercept=float(straight.intercept),
        ),
    )


def _log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """The log of the probability the curve assigns to what happened.

    Clipped a hair inside (0, 1) so a curve that has sharpened to a step on
    separated data reports a large finite number rather than minus infinity.
    """
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(np.sum(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


class EvaluateRequest(BaseModel):
    """One curve, named by its slope and intercept, and one threshold.

    The curve need not be the fitted one. The page hands over whatever the
    reader's sliders say, so a threshold can be moved without refitting and a
    deliberately worse curve can be scored beside the fitted one.
    """

    points: list[Outcome] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    slope: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    intercept: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    model_config = {"extra": "forbid"}


class ScoredOutcome(BaseModel):
    """One student under the curve: score, probability, decision, and loss.

    ``assigned_probability`` is the probability the curve gave to the outcome
    that actually happened, ``p`` for a pass and ``1 - p`` for a fail, and
    ``loss_contribution`` is minus its logarithm. ``gap`` is ``y - p`` and
    ``weighted_gap`` is that times the hours, the two per-student pieces of
    the gradient.
    """

    x: float
    label: int
    score: float
    probability: float
    predicted: int
    correct: bool
    assigned_probability: float
    loss_contribution: float
    gap: float
    weighted_gap: float


class ConfusionCounts(BaseModel):
    """The four cells of the confusion matrix at the chosen threshold."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class EvaluateResponse(BaseModel):
    """Everything a threshold and a curve decide together."""

    outcomes: list[ScoredOutcome]
    confusion: ConfusionCounts
    accuracy: float
    boundary: float | None
    log_likelihood: float
    log_loss: float
    likelihood: float
    gap_total: float
    weighted_gap_total: float
    curve: list[CurvePoint]


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Score the named curve at the named threshold, student by student."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    labels = np.asarray([point.label for point in request.points], dtype=np.float64)
    scores = request.intercept + request.slope * x_values
    probabilities = stable_logistic(scores)
    predicted = (probabilities >= request.threshold).astype(np.float64)

    evaluation = ClassificationEvaluation(labels, predicted)
    matrix = evaluation.confusion_matrix
    assigned = np.where(labels == 1, probabilities, 1 - probabilities)
    clipped = np.clip(assigned, 1e-15, 1 - 1e-15)
    gaps = labels - probabilities
    log_likelihood = float(np.sum(np.log(clipped)))

    boundary: float | None
    if abs(request.slope) < FLAT_SLOPE:
        boundary = None
    else:
        logit = math.log(request.threshold / (1 - request.threshold))
        boundary = (logit - request.intercept) / request.slope

    grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), CURVE_SAMPLES)
    grid_probabilities = stable_logistic(request.intercept + request.slope * grid)

    return EvaluateResponse(
        outcomes=[
            ScoredOutcome(
                x=float(x),
                label=int(label),
                score=float(score),
                probability=float(probability),
                predicted=int(guess),
                correct=bool(guess == label),
                assigned_probability=float(probability_assigned),
                loss_contribution=float(-np.log(probability_clipped)),
                gap=float(gap),
                weighted_gap=float(gap * x),
            )
            for x, label, score, probability, guess, probability_assigned, probability_clipped, gap in zip(
                x_values,
                labels,
                scores,
                probabilities,
                predicted,
                assigned,
                clipped,
                gaps,
                strict=True,
            )
        ],
        confusion=ConfusionCounts(
            true_positives=matrix.true_positives,
            true_negatives=matrix.true_negatives,
            false_positives=matrix.false_positives,
            false_negatives=matrix.false_negatives,
        ),
        accuracy=float(evaluation.accuracy),
        boundary=boundary,
        log_likelihood=log_likelihood,
        log_loss=-log_likelihood / labels.size,
        likelihood=math.exp(log_likelihood),
        gap_total=float(np.sum(gaps)),
        weighted_gap_total=float(np.sum(gaps * x_values)),
        curve=[
            CurvePoint(x=float(x), y=float(p))
            for x, p in zip(grid, grid_probabilities, strict=True)
        ],
    )


MAX_WALK_EPOCHS = 5000
"""The longest climb one recorded walk may take."""

SURFACE_SAMPLES = 41
"""The log loss is sampled over the two coefficients on a lattice this many per side."""


class LogisticWalkRequest(BaseModel):
    """The outcomes, how boldly to climb, and how many passes to record."""

    points: list[Outcome] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    learning_rate: float = Field(default=0.5, gt=0.0, le=MAX_LEARNING_RATE)
    max_epochs: int = Field(default=200, ge=1, le=MAX_WALK_EPOCHS)

    model_config = {"extra": "forbid"}


class LogisticPass(BaseModel):
    """Where the climb stood after one pass, and what it read there."""

    pass_number: int
    intercept: float
    slope: float
    log_loss: float
    accuracy: float
    gradient_intercept: float
    gradient_slope: float


class LogisticWalkResponse(BaseModel):
    """One recorded climb, and the log loss over the coefficients it crossed.

    The lattice is row-major over the slope then the intercept, centred so
    that both the start at zero and where the climb ended fit inside it.
    ``converged`` is the library's own verdict on this budget, and a climb on
    separated data never earns it.
    """

    passes: list[LogisticPass]
    start_log_loss: float
    converged: bool
    passes_run: int
    intercept_axis: list[float]
    slope_axis: list[float]
    log_losses: list[list[float]]


@router.post("/walk", response_model=LogisticWalkResponse)
def walk(request: LogisticWalkRequest) -> LogisticWalkResponse:
    """Record every pass of the climb, and sample the loss it climbs on."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    labels = np.asarray([point.label for point in request.points], dtype=np.float64)
    design_values = np.empty((x_values.size, 2), order="F")
    design_values[:, 0] = 1.0
    design_values[:, 1] = x_values
    design = DesignMatrix(design_values, ["hours"], True)
    target = Feature("passed", labels).column

    model = LogisticRegression(
        learning_rate=request.learning_rate, max_epochs=request.max_epochs
    )
    path = model.solver_path(design, target)

    def loss_at(intercept: float, slope: float) -> float:
        probabilities = stable_logistic(intercept + slope * x_values)
        return -_log_likelihood(labels, probabilities) / labels.size

    passes: list[LogisticPass] = []
    for step in path:
        intercept, slope = float(step.weights_after[0]), float(step.weights_after[1])
        probabilities = stable_logistic(intercept + slope * x_values)
        gaps = labels - probabilities
        passes.append(
            LogisticPass(
                pass_number=step.pass_number,
                intercept=intercept,
                slope=slope,
                log_loss=loss_at(intercept, slope),
                accuracy=float(np.mean((probabilities >= 0.5) == (labels == 1))),
                gradient_intercept=float(np.sum(gaps)),
                gradient_slope=float(np.sum(gaps * x_values)),
            )
        )

    final_intercept = passes[-1].intercept if passes else 0.0
    final_slope = passes[-1].slope if passes else 0.0
    intercept_low = min(0.0, final_intercept) - 0.3 * max(1.0, abs(final_intercept))
    intercept_high = max(0.0, final_intercept) + 0.3 * max(1.0, abs(final_intercept))
    slope_low = min(0.0, final_slope) - 0.3 * max(0.5, abs(final_slope))
    slope_high = max(0.0, final_slope) + 0.3 * max(0.5, abs(final_slope))
    intercept_axis = np.linspace(intercept_low, intercept_high, SURFACE_SAMPLES)
    slope_axis = np.linspace(slope_low, slope_high, SURFACE_SAMPLES)
    log_losses = [
        [loss_at(float(intercept), float(slope)) for intercept in intercept_axis]
        for slope in slope_axis
    ]

    return LogisticWalkResponse(
        passes=passes,
        start_log_loss=loss_at(0.0, 0.0),
        converged=bool(path.converged),
        passes_run=int(path.passes_run),
        intercept_axis=[float(value) for value in intercept_axis],
        slope_axis=[float(value) for value in slope_axis],
        log_losses=log_losses,
    )
