"""The five losses, for the loss-functions page.

A loss takes the last layer's raw output and the truth and answers two things,
what the miss costs and which way to move the output to shrink it. The page
asks that question six ways, and each endpoint here is one of them.

``measure`` and ``curve`` put one slider on a single raw output and ask all
five of the library's losses at once, so the reader watches five costs and
five slopes move together and sees where they agree and where they part.
``measure`` also nudges the raw output either side and divides, which is the
calculus primer's definition of the slope, so the page can print how far the
library's gradient is from the slope of its own value rather than assert that
they agree. Both endpoints carry one more reading beside the five, a sigmoid
output scored by squared error, which is the pairing the page argues against
and is computed by the library through a one-neuron layer so the argument is
measured.

``regression-batch`` and ``classification-batch`` score a whole batch of rows,
which is what a loss actually does inside a training loop. They report each
row's own cost and pull beside the batch's total, so the page can say what
share of the pull one person carries under each rule.

``softmax-row`` is one row of several class scores, because the softmax loss
is the only one here that reads across a row rather than down a column.

``descend-line`` fits one linear neuron to people's heights and weights by
gradient descent under each of the three regression losses in turn, using the
library's layers and losses and nothing else, so the page can show what a loss
does to a fitted line when one weight has been mistyped. The closed-form least
squares line travels with it as a check on the descent.

Two truths travel in the slider requests, because the losses ask two different
questions. The three regression losses compare the raw output to a number,
``target``. The two classification losses compare it to a yes or a no,
``label``, and apply their own squash first, which is why the request carries
both and why the response reports the squashed probability only for those two.

The library divides every loss by the row count, so on the slider endpoints,
which measure one row, the division is by one and the page's hand arithmetic
is the library's arithmetic. The batch endpoints are where that division does
something, and they report the sum of the row costs beside the value so a
reader can see it happen.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    AbsoluteError,
    BinaryCrossEntropy,
    DenseLayer,
    Feature,
    HuberError,
    Identity,
    LayerStack,
    Neuron,
    Sigmoid,
    SimpleLinearRegression,
    SoftmaxCrossEntropy,
    SquaredError,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["loss-functions"])

RAW_OUTPUT_LOW = -5.0
RAW_OUTPUT_HIGH = 5.0
"""The range the slider walks. Wide enough to show every loss's far behaviour."""

MAX_TARGET = 5.0
"""The furthest a regression truth may sit, matching the raw output's range."""

MAX_THRESHOLD = 10.0
"""The largest Huber knee the slider may ask for."""

MAX_BATCH_THRESHOLD = MAX_MAGNITUDE
"""The largest Huber knee a batch may ask for. A knee is in the target's
units, so on weights in kilograms it is a number of kilograms."""

MIN_SAMPLES = 21
MAX_SAMPLES = 201
"""How finely the curve endpoint may sample the raw output."""

FIRST_CLASS = 0
"""The softmax column that carries the raw output. The other column's logit is
held at zero, which is what makes a two-class softmax the sigmoid."""

NUDGE = 1e-5
"""Half the width of the nudge the finite-difference check uses.

Wide enough that a difference of two costs near one stands well above
rounding, narrow enough that the curvature past the linear term is negligible
everywhere except at a kink, where no slope exists and the check says so."""

MIN_CLASSES = 2
MAX_CLASSES = 6
"""How wide a softmax row may be. Two is the sigmoid; six is enough to see."""

MAX_EPOCHS = 400
"""How long the descent may walk. Four hundred epochs at the default rate
takes squared error to the closed form to within rounding."""

DEFAULT_EPOCHS = 400
DEFAULT_LEARNING_RATE = 0.1
MAX_LEARNING_RATE = 1.0


class LossSetting(BaseModel):
    """The two truths, and the one loss that takes a parameter."""

    target: float = Field(ge=-MAX_TARGET, le=MAX_TARGET)
    label: int = Field(ge=0, le=1)
    huber_threshold: float = Field(default=1.0, gt=0.0, le=MAX_THRESHOLD)

    model_config = {"extra": "forbid"}


class MeasureRequest(LossSetting):
    """The setting, plus the one raw output to measure at."""

    raw_output: float = Field(ge=RAW_OUTPUT_LOW, le=RAW_OUTPUT_HIGH)


class CurveRequest(LossSetting):
    """The setting, plus how finely to sample the sweep."""

    n_samples: int = Field(default=101, ge=MIN_SAMPLES, le=MAX_SAMPLES)


class LossReading(BaseModel):
    """What one loss answered at one raw output.

    ``probability`` is the squash the loss applied on its way to the value,
    and only the two classification losses apply one. ``slope_by_nudging`` is
    the value measured a nudge either side and divided by the gap, which the
    ``measure`` endpoint fills in and the ``curve`` endpoint leaves empty;
    ``disagreement`` is how far that is from ``gradient``.
    """

    value: float
    gradient: float
    probability: float | None = None
    slope_by_nudging: float | None = None
    disagreement: float | None = None


class FiveReadings(BaseModel):
    """All five losses at one raw output, and the pairing the page argues against.

    ``sigmoid_then_squared`` is a sigmoid output scored by squared error
    against the yes-or-no label. Its gradient at the raw output carries the
    sigmoid's own slope, which is the term the canonical pairing cancels.
    """

    squared_error: LossReading
    absolute_error: LossReading
    huber_error: LossReading
    binary_cross_entropy: LossReading
    softmax_cross_entropy: LossReading
    sigmoid_then_squared: LossReading


class MeasureResponse(BaseModel):
    """The raw output measured, and the six answers."""

    raw_output: float
    nudge: float
    readings: FiveReadings


class CurveSample(BaseModel):
    """One loss at one sample of the sweep."""

    raw_output: float
    value: float
    gradient: float
    probability: float | None = None


class LossCurves(BaseModel):
    """Six curves over the same sweep of raw outputs."""

    squared_error: list[CurveSample]
    absolute_error: list[CurveSample]
    huber_error: list[CurveSample]
    binary_cross_entropy: list[CurveSample]
    softmax_cross_entropy: list[CurveSample]
    sigmoid_then_squared: list[CurveSample]


READING_NAMES = (
    "squared_error",
    "absolute_error",
    "huber_error",
    "binary_cross_entropy",
    "softmax_cross_entropy",
    "sigmoid_then_squared",
)


def _sigmoid_then_squared(raw_output: float, label: int) -> LossReading:
    """A sigmoid output scored by squared error, through the library's layer.

    One neuron with weight one and no bias reading the raw output as its
    input, bent by the sigmoid, scored by squared error. The bias gradient the
    backward pass reports is then the slope of the loss at the raw score,
    which is the number the page wants beside the cross-entropy's.
    """
    stack = LayerStack([DenseLayer([Neuron([1.0], bias=0.0, activation=Sigmoid())])])
    inputs = np.array([[raw_output]])
    targets = np.array([[float(label)]])
    backward = stack.backward_pass(inputs, targets, SquaredError())
    gradient = backward[0]
    assert gradient is not None
    probability = float(stack.respond_to(inputs).outputs[0, 0])
    return LossReading(
        value=float(backward.loss),
        gradient=float(gradient.biases[0]),
        probability=probability,
    )


def _readings_at(
    raw_output: float, target: float, label: int, huber_threshold: float
) -> FiveReadings:
    """Ask all five losses about one raw output, one row each."""
    outputs = np.array([[raw_output]])
    number_truth = np.array([[target]])
    yes_or_no_truth = np.array([[float(label)]])

    squared = SquaredError().measure(outputs, number_truth)
    absolute = AbsoluteError().measure(outputs, number_truth)
    huber = HuberError(threshold=huber_threshold).measure(outputs, number_truth)
    binary = BinaryCrossEntropy().measure(outputs, yes_or_no_truth)

    # Two logits, the raw output in the first column and zero in the second,
    # so the first class's probability is the sigmoid of the raw output and
    # the page can show the two classification losses agreeing exactly.
    two_logits = np.array([[raw_output, 0.0]])
    true_column = FIRST_CLASS if label == 1 else 1 - FIRST_CLASS
    one_hot = SoftmaxCrossEntropy.one_hot([true_column], 2)
    softmax = SoftmaxCrossEntropy().measure(two_logits, one_hot)

    # The probabilities the two classification losses applied on the way to
    # their values. The library does not hand them back, so they are recomputed
    # here from the same definition, the sigmoid of the raw output, which is
    # arithmetic on the request and not modelling.
    yes_probability = float(1.0 / (1.0 + np.exp(-raw_output)))

    return FiveReadings(
        squared_error=LossReading(
            value=squared.value, gradient=float(squared.gradient[0, 0])
        ),
        absolute_error=LossReading(
            value=absolute.value, gradient=float(absolute.gradient[0, 0])
        ),
        huber_error=LossReading(
            value=huber.value, gradient=float(huber.gradient[0, 0])
        ),
        binary_cross_entropy=LossReading(
            value=binary.value,
            gradient=float(binary.gradient[0, 0]),
            probability=yes_probability,
        ),
        softmax_cross_entropy=LossReading(
            value=softmax.value,
            gradient=float(softmax.gradient[0, FIRST_CLASS]),
            probability=yes_probability,
        ),
        sigmoid_then_squared=_sigmoid_then_squared(raw_output, label),
    )


def _with_nudged_slopes(request: MeasureRequest) -> FiveReadings:
    """The readings at the raw output, each carrying its slope by nudging."""
    at = _readings_at(
        request.raw_output, request.target, request.label, request.huber_threshold
    )
    above = _readings_at(
        request.raw_output + NUDGE,
        request.target,
        request.label,
        request.huber_threshold,
    )
    below = _readings_at(
        request.raw_output - NUDGE,
        request.target,
        request.label,
        request.huber_threshold,
    )
    filled = {}
    for name in READING_NAMES:
        reading: LossReading = getattr(at, name)
        slope = (getattr(above, name).value - getattr(below, name).value) / (
            2.0 * NUDGE
        )
        filled[name] = reading.model_copy(
            update={
                "slope_by_nudging": float(slope),
                "disagreement": float(abs(slope - reading.gradient)),
            }
        )
    return FiveReadings(**filled)


@router.post("/concepts/loss-functions/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> MeasureResponse:
    """All five losses at one raw output, each checked against a nudge."""
    return MeasureResponse(
        raw_output=request.raw_output,
        nudge=NUDGE,
        readings=_with_nudged_slopes(request),
    )


@router.post("/concepts/loss-functions/curve", response_model=LossCurves)
def curve(request: CurveRequest) -> LossCurves:
    """All five losses across a sweep of the raw output."""
    sweep = np.linspace(RAW_OUTPUT_LOW, RAW_OUTPUT_HIGH, request.n_samples)
    readings = [
        _readings_at(
            float(raw_output), request.target, request.label, request.huber_threshold
        )
        for raw_output in sweep
    ]

    def samples_of(name: str) -> list[CurveSample]:
        return [
            CurveSample(
                raw_output=float(raw_output),
                value=getattr(reading, name).value,
                gradient=getattr(reading, name).gradient,
                probability=getattr(reading, name).probability,
            )
            for raw_output, reading in zip(sweep, readings, strict=True)
        ]

    return LossCurves(**{name: samples_of(name) for name in READING_NAMES})


# --- A whole batch of predictions of a number ------------------------------


class PredictedRow(BaseModel):
    """One person, the weight a line guessed for them, and the weight measured."""

    prediction: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    truth: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class RegressionBatchRequest(BaseModel):
    """The rows to score, and the one knee the batch needs.

    At least one row, because the library divides by the row count and a
    batch of nothing has no average; the library answers ``nan`` there with a
    warning rather than a refusal, so the refusal is made at this door.
    """

    rows: list[PredictedRow] = Field(min_length=1, max_length=MAX_POINTS)
    huber_threshold: float = Field(default=5.0, gt=0.0, le=MAX_BATCH_THRESHOLD)

    model_config = {"extra": "forbid"}


class BatchReading(BaseModel):
    """One loss on a whole batch, row by row and in total.

    ``row_costs`` are each row's own cost before the division by the row
    count, so they sum to ``total_row_cost`` and ``value`` is that total over
    ``n_rows``. ``gradients`` are the library's, one per row, carrying the
    division. ``pull_shares`` are each row's share of the batch's total pull,
    the size of its gradient over the sum of all the sizes, and are all zero
    when nothing pulls at all.
    """

    value: float
    row_costs: list[float]
    total_row_cost: float
    gradients: list[float]
    pull_shares: list[float]
    largest_pull_share: float


class RegressionBatchResponse(BaseModel):
    """Three losses on one batch of misses."""

    n_rows: int
    misses: list[float]
    inside_knee: list[bool]
    squared_error: BatchReading
    absolute_error: BatchReading
    huber_error: BatchReading


def _pull_shares(gradients: np.ndarray) -> list[float]:
    sizes = np.abs(gradients)
    total = float(sizes.sum())
    if total == 0.0:
        return [0.0 for _ in gradients]
    return [float(size / total) for size in sizes]


def _batch_reading(loss, outputs: np.ndarray, targets: np.ndarray) -> BatchReading:
    """Score a batch, and recover each row's own cost from the library's value.

    The library reports one averaged value. Each row's own cost is the value
    the library reports for that row alone, which is the same arithmetic on a
    batch of one, and the sum of those is the batch's total before division.
    """
    measured = loss.measure(outputs, targets)
    row_costs = [
        float(
            loss.measure(outputs[index : index + 1], targets[index : index + 1]).value
        )
        for index in range(outputs.shape[0])
    ]
    gradients = measured.gradient[:, 0]
    shares = _pull_shares(gradients)
    return BatchReading(
        value=float(measured.value),
        row_costs=row_costs,
        total_row_cost=float(sum(row_costs)),
        gradients=[float(value) for value in gradients],
        pull_shares=shares,
        largest_pull_share=float(max(shares)),
    )


@router.post(
    "/concepts/loss-functions/regression-batch",
    response_model=RegressionBatchResponse,
)
def regression_batch(request: RegressionBatchRequest) -> RegressionBatchResponse:
    """Score a batch of guessed weights against the measured ones, three ways."""
    outputs = np.array([[row.prediction] for row in request.rows])
    targets = np.array([[row.truth] for row in request.rows])
    misses = outputs - targets
    return RegressionBatchResponse(
        n_rows=len(request.rows),
        misses=[float(value) for value in misses[:, 0]],
        inside_knee=[
            bool(value) for value in (np.abs(misses[:, 0]) <= request.huber_threshold)
        ],
        squared_error=_batch_reading(SquaredError(), outputs, targets),
        absolute_error=_batch_reading(AbsoluteError(), outputs, targets),
        huber_error=_batch_reading(
            HuberError(threshold=request.huber_threshold), outputs, targets
        ),
    )


# --- A whole batch of yes-or-no scores -------------------------------------


class ScoredRow(BaseModel):
    """One person, the raw score a model gave them, and whether they are an adult."""

    score: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=1)


class ClassificationBatchRequest(BaseModel):
    """The scored rows. The label is a yes or a no and nothing in between."""

    rows: list[ScoredRow] = Field(min_length=1, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class ClassificationBatchResponse(BaseModel):
    """Binary cross-entropy on one batch, row by row and in total.

    ``accuracy`` is the share of rows whose probability lands on the label's
    side of one half, which is what a classifier is judged by and is not what
    the loss measures; the page keeps the two apart.
    """

    n_rows: int
    probabilities: list[float]
    binary_cross_entropy: BatchReading
    accuracy: float


@router.post(
    "/concepts/loss-functions/classification-batch",
    response_model=ClassificationBatchResponse,
)
def classification_batch(
    request: ClassificationBatchRequest,
) -> ClassificationBatchResponse:
    """Score a batch of yes-or-no answers by the log of the probability given."""
    outputs = np.array([[row.score] for row in request.rows])
    targets = np.array([[float(row.label)] for row in request.rows])
    probabilities = 1.0 / (1.0 + np.exp(-outputs[:, 0]))
    called = (probabilities >= 0.5).astype(np.float64)
    return ClassificationBatchResponse(
        n_rows=len(request.rows),
        probabilities=[float(value) for value in probabilities],
        binary_cross_entropy=_batch_reading(BinaryCrossEntropy(), outputs, targets),
        accuracy=float(np.mean(called == targets[:, 0])),
    )


# --- One row of several class scores ---------------------------------------


ClassScore = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]
"""One raw class score, inside the API's magnitude bound."""


class SoftmaxRowRequest(BaseModel):
    """One row of raw class scores and which class is the true one."""

    logits: list[ClassScore] = Field(min_length=MIN_CLASSES, max_length=MAX_CLASSES)
    true_class: int = Field(ge=0, le=MAX_CLASSES - 1)

    model_config = {"extra": "forbid"}


class SoftmaxRowResponse(BaseModel):
    """The squashed row, its cost, and the pull on every class score.

    ``gradient_sum`` is the pulls added across the classes, which the page
    says is zero because both the probabilities and the one-hot truth sum to
    one; it is reported rather than assumed.
    """

    probabilities: list[float]
    value: float
    gradient: list[float]
    gradient_sum: float


@router.post("/concepts/loss-functions/softmax-row", response_model=SoftmaxRowResponse)
def softmax_row(request: SoftmaxRowRequest) -> SoftmaxRowResponse:
    """Squash one row of scores across its classes and score the true one."""
    width = len(request.logits)
    if request.true_class >= width:
        # The library's one-hot builder would raise a bare index error here,
        # and a negative index would wrap to the last column without a word,
        # so the refusal is made at the door with the width named.
        raise HTTPException(
            status_code=422,
            detail=(
                f"the row holds {width} class scores, so there is no class "
                f"{request.true_class} to be the true one"
            ),
        )
    logits = np.array([request.logits], dtype=np.float64)
    one_hot = SoftmaxCrossEntropy.one_hot([request.true_class], width)
    measured = SoftmaxCrossEntropy().measure(logits, one_hot)
    # The squash the loss applied, recomputed from the same definition with the
    # row maximum subtracted first, since the library keeps it to itself.
    shifted = logits[0] - logits[0].max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    gradient = measured.gradient[0]
    return SoftmaxRowResponse(
        probabilities=[float(value) for value in probabilities],
        value=float(measured.value),
        gradient=[float(value) for value in gradient],
        gradient_sum=float(gradient.sum()),
    )


# --- One linear neuron, three losses, three lines --------------------------


class DescendLineRequest(BaseModel):
    """The people to fit, the knee for Huber, and how the walk is run.

    The knee is in kilograms, because the misses are. The learning rate is
    applied to standardised heights and centred weights, which is what keeps
    one rate sensible for every loss.
    """

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    huber_threshold: float = Field(default=5.0, gt=0.0, le=MAX_BATCH_THRESHOLD)
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE, gt=0.0, le=MAX_LEARNING_RATE
    )

    model_config = {"extra": "forbid"}


class DescendedLine(BaseModel):
    """Where one loss walked the line to, and how the walk went.

    ``final_largest_movement`` is the largest slope anywhere in the neuron on
    the last epoch, which is the convergence reading; a walk that has settled
    reports something near zero, and a walk that is still stepping reports
    the size of its step. ``loss_history`` is the loss before each epoch's
    step, so its first entry is the loss of the flat starting line.
    """

    slope: float
    intercept: float
    predictions: list[float]
    residuals: list[float]
    starting_loss: float
    final_loss: float
    final_largest_movement: float
    loss_history: list[float]


class LeastSquaresLine(BaseModel):
    """The closed-form least squares line, for checking the squared-error walk."""

    slope: float
    intercept: float


class DescendLineResponse(BaseModel):
    """Three lines from three losses on the same people, and the closed-form check.

    ``descent_gap`` is the largest difference between the squared-error
    walk's slope and intercept and the closed form's, which says whether the
    walk arrived.
    """

    n_rows: int
    squared_error: DescendedLine
    absolute_error: DescendedLine
    huber_error: DescendedLine
    least_squares: LeastSquaresLine
    descent_gap: float


def _descend(
    heights: np.ndarray, weights: np.ndarray, loss, epochs: int, rate: float
) -> DescendedLine:
    """Walk one linear neuron down the loss, then undo the standardising.

    The neuron reads height standardised to zero mean and unit spread and
    answers weight above the mean, both of which are arithmetic on the
    request. The slope and intercept are then converted back to kilograms
    per centimetre and kilograms so the line can be drawn over the people.
    """
    height_mean, height_spread = float(heights.mean()), float(heights.std())
    weight_mean = float(weights.mean())
    inputs = ((heights - height_mean) / height_spread).reshape(-1, 1)
    targets = (weights - weight_mean).reshape(-1, 1)

    stack = LayerStack([DenseLayer([Neuron([0.0], bias=0.0, activation=Identity())])])
    history: list[float] = []
    for _ in range(epochs):
        backward = stack.backward_pass(inputs, targets, loss)
        history.append(float(backward.loss))
        stack = stack.stepped_by(backward, rate)

    final = stack.backward_pass(inputs, targets, loss)
    layer = next(iter(stack))
    assert isinstance(layer, DenseLayer)
    weight = float(layer.weight_matrix[0, 0])
    bias = float(layer.bias_vector[0])
    slope = weight / height_spread
    intercept = weight_mean + bias - slope * height_mean
    predictions = slope * heights + intercept
    return DescendedLine(
        slope=slope,
        intercept=intercept,
        predictions=[float(value) for value in predictions],
        residuals=[float(value) for value in (weights - predictions)],
        starting_loss=history[0],
        final_loss=float(final.loss),
        final_largest_movement=float(final.largest_movement),
        loss_history=history,
    )


@router.post(
    "/concepts/loss-functions/descend-line", response_model=DescendLineResponse
)
def descend_line(request: DescendLineRequest) -> DescendLineResponse:
    """Fit a line by descent under each regression loss, and check one of them."""
    heights = np.array([point.x for point in request.points], dtype=np.float64)
    weights = np.array([point.y for point in request.points], dtype=np.float64)

    # The closed form first, so a set of people no line can be fitted through,
    # every height the same, is refused by the library in its own words before
    # the standardising above would divide by a spread of zero.
    closed = SimpleLinearRegression().fit(
        Feature("height", heights), Feature("weight", weights)
    )
    least_squares = LeastSquaresLine(
        slope=float(closed.slope), intercept=float(closed.intercept)
    )

    squared = _descend(
        heights, weights, SquaredError(), request.epochs, request.learning_rate
    )
    absolute = _descend(
        heights, weights, AbsoluteError(), request.epochs, request.learning_rate
    )
    huber = _descend(
        heights,
        weights,
        HuberError(threshold=request.huber_threshold),
        request.epochs,
        request.learning_rate,
    )
    return DescendLineResponse(
        n_rows=len(request.points),
        squared_error=squared,
        absolute_error=absolute,
        huber_error=huber,
        least_squares=least_squares,
        descent_gap=float(
            max(
                abs(squared.slope - least_squares.slope),
                abs(squared.intercept - least_squares.intercept),
            )
        ),
    )
