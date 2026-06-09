"""A small network trained end to end, for the training-a-network page.

Every other network page shows one piece standing still. This one runs the
loop: forward, score, walk the blame back down, step, and again, on a problem
no straight line can solve. What comes back is the record of one run, the loss
after every epoch and the decision region at a handful of them, so the page can
replay a single training run rather than refitting at each position of a
slider.

One run, recorded, and why that matters
----------------------------------------
The library records no history of its own, so this module owns the loop. It is
a dozen lines over public methods, and the important property is that it runs
ONCE per request and records what that one run did. A widget that refitted at
each slider position would be showing a different network at every step, and
the loss curve would not be a curve at all.

Where the arithmetic lives
--------------------------
Nothing here computes a gradient or a step. ``backward_pass`` scores the
answers and walks the blame down, ``stepped_by`` builds the moved stack, and
this module only decides when to write something down. The one calculation
that is this module's own is the sigmoid applied to the last layer's raw
output, which the loss applies internally and does not hand back, and it is
needed to turn a score into the yes-or-no the accuracy counts.

The control
-----------
Each response carries what the logistic page's model scores on the same
people, fitted with the same library. It is there because the page's claim is
that a straight boundary cannot do this, and a claim of that shape should be
measured rather than asserted.

The loop lives here and not in the library
-------------------------------------------
The library supplies the layers, the chain that refuses a bad join, the losses
and the backward pass, and it supplies the step. It supplies no epoch counter,
no batching, no recorded history and no optimiser beyond plain descent, so
every one of those is written in this module, over public methods, and is
deliberately dull. The endpoints below exist because a page that says "the
loop is yours to write" should show the numbers the writing of it produces.

Non-finite numbers travel as null
----------------------------------
A diverging walk reports a loss of infinity, and JSON has no way to write that
down that a browser will read back. Every loss that leaves this module goes
through :func:`_reportable`, which answers ``None`` for anything that is not
finite, so a widget plots a gap rather than receiving a token it cannot parse.
"""

from __future__ import annotations

import math
import time
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    BinaryCrossEntropy,
    DenseLayer,
    Dropout,
    Feature,
    HyperbolicTangent,
    Identity,
    LayerStack,
    LogisticRegression,
    Neuron,
    PassPurpose,
    SquaredError,
)
from oop_ml import Loss as LossFunction
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field, model_validator

from oop_ml_api.concepts.two_feature_classification import GRID_CELLS, RegionGrid

router = APIRouter(prefix="/concepts/training-a-network", tags=["training-a-network"])

MAX_EPOCHS = 300
"""How long the loop may run. Three hundred epochs on sixty rows is a few
milliseconds, and it is past where either preset has settled."""

MIN_HIDDEN = 2
MAX_HIDDEN = 8
"""How wide the hidden layer may be. Two is the narrowest that can bend a
boundary at all, and eight is well past what either preset needs."""

MAX_LEARNING_RATE = 2.0
"""The largest stride the replay accepts. Not a stability bound: measured on
this chain, a rate of five reaches a lower loss than a rate of a half, and the
walk only comes apart well above that. It is a bound on what the replay's
slider is useful for, and the sweep endpoint takes rates far above it."""

MAX_SWEPT_RATE = 100.0
"""The largest stride the sweep will walk with, which is well past where the
walk stops making progress on either preset."""

MAX_SWEEP_RATES = 8
"""How many rates one sweep may compare. Eight curves is already a crowded
picture."""

MAX_BATCH_EPOCHS = 120
"""How long a batch comparison may run. A batch of one takes sixty steps per
epoch, so this is the setting that decides how long the slowest row waits."""

MAX_BATCH_SIZES = 6
"""How many batch sizes one comparison may hold."""

MAX_GENERALISATION_EPOCHS = 800
"""How long the memorisation run may go. The held-out loss on the page's
fixture bottoms out well inside this and then climbs."""

DIRECTION_DRAWS = 200
"""How many batches to draw when measuring how far a small batch's direction
sits from the whole set's. Two hundred is enough for the mean to settle and
cheap enough to compute inside one request."""

FIRST_ORDER_RATES = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
"""The ladder the first-order check walks. Fixed rather than requested,
because the point of it is the trend down the column."""

Objective = Literal["log_loss", "squared_error"]
"""What to score the answers by. The page's own runs use log-loss, which is
what the loss-functions page argues for on a yes-or-no; squared error is
offered because it is the setting on which this chain can actually be made to
produce a number that is not a number."""

POINTS_PER_CLASS = 30
"""How many people each class gets, in both presets."""

RECORDED_REGIONS = 8
"""How many decision regions to record across the run. The lattice is 26 by 26
whole numbers, so eight of them is a few thousand values."""

FIXED_SEED = 7
"""The draw of the people, held fixed so the page can quote what it shows."""

Preset = Literal["rings", "exclusive_or"]


class TrainRequest(BaseModel):
    """Which problem, how wide the hidden layer, and how long to walk."""

    preset: Preset = "rings"
    hidden_width: int = Field(default=6, ge=MIN_HIDDEN, le=MAX_HIDDEN)
    learning_rate: float = Field(default=0.5, gt=0.0, le=MAX_LEARNING_RATE)
    max_epochs: int = Field(default=200, ge=1, le=MAX_EPOCHS)
    weight_seed: int = Field(default=1, ge=0, le=9999)

    model_config = {"extra": "forbid"}


class TrainedPoint(BaseModel):
    """One person and the class they belong to."""

    x: float
    y: float
    label: int


class EpochReading(BaseModel):
    """What the loop wrote down after one epoch.

    ``loss`` is the number the backward pass measured on the way forward, so
    it is the loss of the stack as it stood BEFORE this epoch's step, which is
    what makes the first reading the loss of the untrained network.
    """

    epoch: int
    loss: float
    accuracy: float
    largest_movement: float


class RecordedRegion(BaseModel):
    """The decision region as it stood after a particular epoch."""

    epoch: int
    regions: RegionGrid


class TrainResponse(BaseModel):
    """One recorded run, and the straight boundary that cannot do the same.

    ``readings`` holds one entry per epoch in order. ``regions`` holds a
    handful of snapshots across the same run, the first taken before any step
    was made so the page can show where the network started.
    """

    points: list[TrainedPoint]
    readings: list[EpochReading]
    regions: list[RecordedRegion]
    final_accuracy: float
    final_loss: float
    starting_loss: float
    n_parameters: int
    straight_line_accuracy: float


def _drawn_people(preset: Preset) -> tuple[np.ndarray, np.ndarray]:
    """The two presets, drawn once under the fixed seed.

    Rings are two circles about the same centre, which no straight line can
    separate because the inner one is surrounded. Exclusive-or is four clumps
    in the four quadrants, where diagonally opposite clumps share a class,
    which no straight line can separate for the reason the neuron page proves.
    """
    generator = np.random.default_rng(FIXED_SEED)

    if preset == "rings":
        rows = []
        labels = []
        for radius, label in ((0.55, 0), (1.5, 1)):
            angles = generator.uniform(0.0, 2.0 * np.pi, POINTS_PER_CLASS)
            spread = radius + generator.normal(0.0, 0.09, POINTS_PER_CLASS)
            rows.append(
                np.column_stack([spread * np.cos(angles), spread * np.sin(angles)])
            )
            labels.append(np.full(POINTS_PER_CLASS, float(label)))
        return np.vstack(rows), np.concatenate(labels)

    corners = ((1.0, 1.0, 0), (-1.0, -1.0, 0), (1.0, -1.0, 1), (-1.0, 1.0, 1))
    per_corner = POINTS_PER_CLASS // 2
    rows = []
    labels = []
    for centre_x, centre_y, label in corners:
        rows.append(
            np.column_stack(
                [
                    generator.normal(centre_x, 0.28, per_corner),
                    generator.normal(centre_y, 0.28, per_corner),
                ]
            )
        )
        labels.append(np.full(per_corner, float(label)))
    return np.vstack(rows), np.concatenate(labels)


def _fresh_stack(hidden_width: int, weight_seed: int) -> LayerStack:
    """Two dense layers, the hidden one bent and the output one straight.

    The output layer is left straight because the loss applies its own squash,
    which is the arrangement the loss-functions page derives. The weights are
    drawn with a spread of the square root of two over the fan-in, which is
    the initialisation the convolution layer already uses.
    """
    generator = np.random.default_rng(weight_seed)

    def layer(n_inputs: int, n_outputs: int, activation) -> DenseLayer:
        spread = (2.0 / n_inputs) ** 0.5
        return DenseLayer(
            [
                Neuron(
                    weights=generator.normal(0.0, spread, n_inputs),
                    bias=0.0,
                    activation=activation,
                )
                for _ in range(n_outputs)
            ]
        )

    return LayerStack(
        [
            layer(2, hidden_width, HyperbolicTangent()),
            layer(hidden_width, 1, Identity()),
        ]
    )


def _chances(stack: LayerStack, rows: np.ndarray) -> np.ndarray:
    """The network's answers, squashed into chances.

    The loss applies this squash internally and hands back only the scores, so
    turning a score into the yes-or-no the accuracy counts means applying it
    here. Written in the stable form, which is what the library uses too.
    """
    scores = np.asarray(stack.respond_to(rows).outputs, dtype=np.float64).ravel()
    return np.where(
        scores >= 0.0,
        1.0 / (1.0 + np.exp(-scores)),
        np.exp(scores) / (1.0 + np.exp(scores)),
    )


def _accuracy(stack: LayerStack, rows: np.ndarray, labels: np.ndarray) -> float:
    """The share of people the network calls correctly at the halfway mark."""
    return float(((_chances(stack, rows) >= 0.5).astype(np.float64) == labels).mean())


def _region_of(stack: LayerStack, rows: np.ndarray) -> RegionGrid:
    """The network's answer at every cell of a lattice over the people."""
    x_pad = 0.08 * float(rows[:, 0].max() - rows[:, 0].min())
    y_pad = 0.08 * float(rows[:, 1].max() - rows[:, 1].min())
    x_min, x_max = float(rows[:, 0].min()) - x_pad, float(rows[:, 0].max()) + x_pad
    y_min, y_max = float(rows[:, 1].min()) - y_pad, float(rows[:, 1].max()) + y_pad

    lattice_x, lattice_y = np.meshgrid(
        np.linspace(x_min, x_max, GRID_CELLS), np.linspace(y_min, y_max, GRID_CELLS)
    )
    cells = np.column_stack([lattice_x.ravel(), lattice_y.ravel()])
    called = (_chances(stack, cells) >= 0.5).astype(int).reshape(GRID_CELLS, GRID_CELLS)

    return RegionGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cells=GRID_CELLS,
        labels=called.tolist(),
    )


def _straight_line_accuracy(rows: np.ndarray, labels: np.ndarray) -> float:
    """What the logistic page's model manages on the same people."""
    features = [Feature("first", rows[:, 0]), Feature("second", rows[:, 1])]
    model = LogisticRegression().fit(features, Feature("label", labels))
    return float(model.score(features, Feature("label", labels)))


def _recorded_epochs(max_epochs: int) -> set[int]:
    """Which epochs get a decision region, spread across the run.

    Zero is always among them, so the page can show the network before it has
    learned anything, and so is the last epoch.
    """
    if max_epochs <= RECORDED_REGIONS:
        return set(range(max_epochs + 1))
    steps = np.linspace(0, max_epochs, RECORDED_REGIONS).round().astype(int)
    return {int(step) for step in steps}


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """Run one training loop and report every epoch of it.

    The loop is the whole of the page: measure and walk the blame down, step,
    repeat. Everything else here is writing down what happened.
    """
    rows, labels = _drawn_people(request.preset)
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()

    stack = _fresh_stack(request.hidden_width, request.weight_seed)
    wanted = _recorded_epochs(request.max_epochs)

    readings: list[EpochReading] = []
    regions: list[RecordedRegion] = []
    if 0 in wanted:
        regions.append(RecordedRegion(epoch=0, regions=_region_of(stack, rows)))

    starting_loss = stack.backward_pass(rows, targets, loss).loss

    for epoch in range(1, request.max_epochs + 1):
        backward = stack.backward_pass(rows, targets, loss)
        stack = stack.stepped_by(backward, request.learning_rate)

        readings.append(
            EpochReading(
                epoch=epoch,
                loss=backward.loss,
                accuracy=_accuracy(stack, rows, labels),
                largest_movement=backward.largest_movement,
            )
        )
        if epoch in wanted:
            regions.append(RecordedRegion(epoch=epoch, regions=_region_of(stack, rows)))

    n_parameters = sum(
        layer.shape.n_inputs * layer.shape.n_outputs + layer.shape.n_outputs
        for layer in stack
    )

    return TrainResponse(
        points=[
            TrainedPoint(x=float(row[0]), y=float(row[1]), label=int(label))
            for row, label in zip(rows, labels, strict=True)
        ],
        readings=readings,
        regions=regions,
        final_accuracy=_accuracy(stack, rows, labels),
        final_loss=stack.backward_pass(rows, targets, loss).loss,
        starting_loss=starting_loss,
        n_parameters=n_parameters,
        straight_line_accuracy=_straight_line_accuracy(rows, labels),
    )


# ---------------------------------------------------------------------------
# The loop itself: one step, the step size, the batch, and the two curves.
# ---------------------------------------------------------------------------


def _reportable(value: float) -> float | None:
    """A number JSON can carry, or ``None`` where it cannot.

    A diverging walk reports an infinity and a browser's parser refuses the
    token a Python encoder writes for it, so anything not finite leaves here
    as a null and the widget draws a gap.
    """
    return float(value) if math.isfinite(value) else None


def _objective(name: Objective) -> LossFunction:
    """The loss the request asked for."""
    return BinaryCrossEntropy() if name == "log_loss" else SquaredError()


def _threshold(name: Objective) -> float:
    """Where the raw output is cut into a yes or a no.

    Log-loss squashes the raw score itself, so its halfway mark sits at a raw
    score of zero. Squared error compares the raw score with the target
    directly, so its halfway mark is a half.
    """
    return 0.0 if name == "log_loss" else 0.5


def _accuracy_under(
    stack: LayerStack, rows: np.ndarray, labels: np.ndarray, objective: Objective
) -> float | None:
    """The share called correctly, or ``None`` once the walk has come apart.

    A stack whose weights have grown past what a float can hold answers with
    values the library refuses to read further, so the accuracy of a diverged
    run is genuinely unavailable rather than zero.
    """
    try:
        scores = np.asarray(stack.respond_to(rows).outputs, dtype=np.float64).ravel()
    except MLLibError:
        return None
    if not np.isfinite(scores).all():
        return None
    called = (scores >= _threshold(objective)).astype(np.float64)
    return float((called == labels).mean())


def _flat_gradient(backward) -> np.ndarray:
    """Every slope in the network laid end to end, one vector.

    The step moves each parameter against its own slope, so the movement of
    the whole network is this vector scaled by the rate, and its length is
    what turns a rate into a distance.
    """
    parts: list[np.ndarray] = []
    for gradient in backward:
        if gradient is None:
            continue
        parts.append(np.asarray(gradient.weights, dtype=np.float64).ravel())
        parts.append(np.asarray(gradient.biases, dtype=np.float64).ravel())
    return np.concatenate(parts)


class TracedParameter(BaseModel):
    """One weight before the step, the slope it was handed, and where it went."""

    description: str
    before: float
    slope: float
    after: float


class LayerStepSummary(BaseModel):
    """How much of the network one layer holds, and how hard it wants to move."""

    position: int
    n_weights: int
    n_biases: int
    largest_weight_slope: float
    largest_bias_slope: float


class FirstOrderRow(BaseModel):
    """What the step size predicted the loss would drop by, against what it did.

    ``predicted_drop`` is the rate times the squared length of the gradient,
    which is what the slope alone says the loss will fall by. ``ratio`` is the
    actual drop divided by that, so it approaches one as the step shrinks.
    """

    learning_rate: float
    predicted_drop: float
    actual_drop: float | None
    ratio: float | None
    loss_after: float | None


class StepRequest(BaseModel):
    """Which problem, how wide, which start, and how far to step once."""

    preset: Preset = "rings"
    hidden_width: int = Field(default=6, ge=MIN_HIDDEN, le=MAX_HIDDEN)
    learning_rate: float = Field(default=0.5, gt=0.0, le=MAX_LEARNING_RATE)
    weight_seed: int = Field(default=1, ge=0, le=9999)

    model_config = {"extra": "forbid"}


class StepResponse(BaseModel):
    """One step of learning, traced from the untrained start.

    ``distance_moved`` is the rate times the gradient's length, which is the
    distance in parameter space the step actually covered; the rate on its own
    is not a distance and moves a different way at every point of the walk.
    """

    n_parameters: int
    n_rows: int
    learning_rate: float
    loss_before: float
    loss_after: float | None
    actual_drop: float | None
    predicted_drop: float
    gradient_norm: float
    distance_moved: float
    largest_movement: float
    ratio: float | None
    layers: list[LayerStepSummary]
    traced: list[TracedParameter]
    first_order: list[FirstOrderRow]


@router.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Take one step from the untrained start and report everything it changed.

    The whole of learning is this, repeated. The endpoint measures the loss,
    reads the slope of every parameter, moves each one against its own slope,
    measures again, and reports the drop the slope predicted beside the drop
    that happened.
    """
    rows, labels = _drawn_people(request.preset)
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()

    stack = _fresh_stack(request.hidden_width, request.weight_seed)
    backward = stack.backward_pass(rows, targets, loss)
    slopes = _flat_gradient(backward)
    norm = float(np.linalg.norm(slopes))

    stepped = stack.stepped_by(backward, request.learning_rate)
    after = stepped.backward_pass(rows, targets, loss).loss
    predicted = request.learning_rate * norm * norm
    actual = backward.loss - after

    hidden_gradient = backward[0]
    output_gradient = backward[-1]
    assert hidden_gradient is not None and output_gradient is not None

    traced = [
        TracedParameter(
            description="hidden unit 1, its weight on the first measurement",
            before=float(stack[0].weight_matrix[0, 0]),
            slope=float(hidden_gradient.weights[0, 0]),
            after=float(stepped[0].weight_matrix[0, 0]),
        ),
        TracedParameter(
            description="hidden unit 1, its bias",
            before=float(stack[0].bias_vector[0]),
            slope=float(hidden_gradient.biases[0]),
            after=float(stepped[0].bias_vector[0]),
        ),
        TracedParameter(
            description="the output unit, its weight on hidden unit 1",
            before=float(stack[-1].weight_matrix[0, 0]),
            slope=float(output_gradient.weights[0, 0]),
            after=float(stepped[-1].weight_matrix[0, 0]),
        ),
    ]

    layers: list[LayerStepSummary] = []
    for position, gradient in enumerate(backward):
        if gradient is None:
            continue
        layers.append(
            LayerStepSummary(
                position=position,
                n_weights=int(np.asarray(gradient.weights).size),
                n_biases=int(np.asarray(gradient.biases).size),
                largest_weight_slope=float(np.max(np.abs(gradient.weights))),
                largest_bias_slope=float(np.max(np.abs(gradient.biases))),
            )
        )

    ladder: list[FirstOrderRow] = []
    for rate in FIRST_ORDER_RATES:
        tried = stack.stepped_by(backward, rate)
        reached = tried.backward_pass(rows, targets, loss).loss
        rate_predicted = rate * norm * norm
        rate_actual = backward.loss - reached
        ladder.append(
            FirstOrderRow(
                learning_rate=rate,
                predicted_drop=rate_predicted,
                actual_drop=_reportable(rate_actual),
                ratio=_reportable(rate_actual / rate_predicted),
                loss_after=_reportable(reached),
            )
        )

    return StepResponse(
        n_parameters=sum(row.n_weights + row.n_biases for row in layers),
        n_rows=int(rows.shape[0]),
        learning_rate=request.learning_rate,
        loss_before=backward.loss,
        loss_after=_reportable(after),
        actual_drop=_reportable(actual),
        predicted_drop=predicted,
        gradient_norm=norm,
        distance_moved=request.learning_rate * norm,
        largest_movement=backward.largest_movement,
        ratio=_reportable(actual / predicted),
        layers=layers,
        traced=traced,
        first_order=ladder,
    )


class SweptRateRow(BaseModel):
    """One step size, walked the whole way, and where it got to.

    ``losses`` holds the loss measured at the start of each epoch, so entry
    zero is the untrained loss every rate shares. ``first_non_finite_epoch``
    is where the walk stopped producing numbers, and ``refused_at_epoch``
    where the library refused to go on, which is later and is a different
    event.
    """

    learning_rate: float
    losses: list[float | None]
    epochs_run: int
    final_loss: float | None
    best_loss: float | None
    best_epoch: int | None
    final_accuracy: float | None
    first_non_finite_epoch: int | None
    refused_at_epoch: int | None
    refusal: str | None


class RateSweepRequest(BaseModel):
    """One start, one problem, several step sizes walked from it."""

    preset: Preset = "rings"
    hidden_width: int = Field(default=6, ge=MIN_HIDDEN, le=MAX_HIDDEN)
    weight_seed: int = Field(default=1, ge=0, le=9999)
    max_epochs: int = Field(default=200, ge=1, le=MAX_EPOCHS)
    objective: Objective = "log_loss"
    learning_rates: list[Annotated[float, Field(gt=0.0, le=MAX_SWEPT_RATE)]] = Field(
        default=[0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0],
        min_length=1,
        max_length=MAX_SWEEP_RATES,
    )

    model_config = {"extra": "forbid"}


class RateSweepResponse(BaseModel):
    """Every rate walked from the identical starting weights."""

    objective: Objective
    max_epochs: int
    starting_loss: float
    rows: list[SweptRateRow]


def _walk(
    stack: LayerStack,
    rows: np.ndarray,
    targets: np.ndarray,
    loss: LossFunction,
    learning_rate: float,
    max_epochs: int,
) -> tuple[LayerStack, list[float], int | None, int | None, str | None]:
    """The loop, with a record of where it stopped meaning anything.

    Two failures are distinguished on purpose. The loss going non-finite is
    the walk ceasing to say anything; the library refusing the step is the
    weights themselves ceasing to be numbers, and it happens later.
    """
    losses: list[float] = []
    first_non_finite: int | None = None
    refused_at: int | None = None
    refusal: str | None = None

    for epoch in range(1, max_epochs + 1):
        try:
            backward = stack.backward_pass(rows, targets, loss)
        except MLLibError as error:
            refused_at = epoch
            refusal = f"{type(error).__name__}: {error}"
            break

        losses.append(backward.loss)
        if first_non_finite is None and not math.isfinite(backward.loss):
            first_non_finite = epoch

        try:
            stack = stack.stepped_by(backward, learning_rate)
        except MLLibError as error:
            refused_at = epoch
            refusal = f"{type(error).__name__}: {error}"
            break

    return stack, losses, first_non_finite, refused_at, refusal


@router.post("/rate-sweep", response_model=RateSweepResponse)
def rate_sweep(request: RateSweepRequest) -> RateSweepResponse:
    """Walk the same start under several step sizes and report each curve."""
    rows, labels = _drawn_people(request.preset)
    targets = labels.reshape(-1, 1)
    loss = _objective(request.objective)

    starting_loss = (
        _fresh_stack(request.hidden_width, request.weight_seed)
        .backward_pass(rows, targets, loss)
        .loss
    )

    swept: list[SweptRateRow] = []
    for learning_rate in request.learning_rates:
        stack = _fresh_stack(request.hidden_width, request.weight_seed)
        stack, losses, first_non_finite, refused_at, refusal = _walk(
            stack, rows, targets, loss, learning_rate, request.max_epochs
        )
        finite = [
            (epoch, value)
            for epoch, value in enumerate(losses, start=1)
            if math.isfinite(value)
        ]
        best = min(finite, key=lambda pair: pair[1]) if finite else None

        swept.append(
            SweptRateRow(
                learning_rate=learning_rate,
                losses=[_reportable(value) for value in losses],
                epochs_run=len(losses),
                final_loss=_reportable(losses[-1]) if losses else None,
                best_loss=best[1] if best else None,
                best_epoch=best[0] if best else None,
                final_accuracy=_accuracy_under(stack, rows, labels, request.objective),
                first_non_finite_epoch=first_non_finite,
                refused_at_epoch=refused_at,
                refusal=refusal,
            )
        )

    return RateSweepResponse(
        objective=request.objective,
        max_epochs=request.max_epochs,
        starting_loss=starting_loss,
        rows=swept,
    )


class BatchSizeRow(BaseModel):
    """One batch size, run for the same number of epochs as every other.

    ``losses`` is measured on the whole training set once per epoch, so the
    rows are comparable to each other whatever they stepped on.
    """

    batch_size: int
    steps_per_epoch: int
    total_steps: int
    losses: list[float | None]
    final_loss: float | None
    final_accuracy: float | None
    milliseconds: float
    milliseconds_per_step: float


class DirectionAgreement(BaseModel):
    """How far one batch's direction sits from the whole set's.

    The cosine of the angle between them, measured at the untrained start over
    ``n_draws`` batches drawn without replacement. One is the same direction,
    zero is at right angles, and a negative number is a step that makes the
    loss worse on the rows the batch did not contain.
    """

    batch_size: int
    mean_cosine: float
    worst_cosine: float
    n_draws: int


class BatchSweepRequest(BaseModel):
    """One start, one step size, the same epochs, several batch sizes."""

    preset: Preset = "rings"
    hidden_width: int = Field(default=6, ge=MIN_HIDDEN, le=MAX_HIDDEN)
    weight_seed: int = Field(default=1, ge=0, le=9999)
    learning_rate: float = Field(default=0.5, gt=0.0, le=MAX_LEARNING_RATE)
    max_epochs: int = Field(default=60, ge=1, le=MAX_BATCH_EPOCHS)
    order_seed: int = Field(default=3, ge=0, le=9999)
    batch_sizes: list[Annotated[int, Field(ge=1, le=2 * POINTS_PER_CLASS)]] = Field(
        default=[1, 5, 15, 30, 60], min_length=1, max_length=MAX_BATCH_SIZES
    )

    model_config = {"extra": "forbid"}


class BatchSweepResponse(BaseModel):
    """Every batch size run from the identical starting weights."""

    n_rows: int
    learning_rate: float
    max_epochs: int
    starting_loss: float
    rows: list[BatchSizeRow]
    agreement: list[DirectionAgreement]


@router.post("/batch-sweep", response_model=BatchSweepResponse)
def batch_sweep(request: BatchSweepRequest) -> BatchSweepResponse:
    """Run the same network at several batch sizes and time each of them.

    Every row shuffles its rows afresh each epoch under the same order seed,
    walks the epoch one batch at a time, and measures the loss on the whole
    training set once per epoch so the curves can be laid over one another.
    """
    rows, labels = _drawn_people(request.preset)
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()
    n_rows = int(rows.shape[0])

    starting = _fresh_stack(request.hidden_width, request.weight_seed)
    starting_loss = starting.backward_pass(rows, targets, loss).loss

    measured: list[BatchSizeRow] = []
    for batch_size in request.batch_sizes:
        stack = _fresh_stack(request.hidden_width, request.weight_seed)
        shuffler = np.random.default_rng(request.order_seed)
        losses: list[float] = []
        steps = 0

        began = time.perf_counter()
        for _ in range(request.max_epochs):
            losses.append(stack.backward_pass(rows, targets, loss).loss)
            order = shuffler.permutation(n_rows)
            for start in range(0, n_rows, batch_size):
                picked = order[start : start + batch_size]
                backward = stack.backward_pass(rows[picked], targets[picked], loss)
                stack = stack.stepped_by(backward, request.learning_rate)
                steps += 1
        elapsed = (time.perf_counter() - began) * 1000.0

        final = stack.backward_pass(rows, targets, loss).loss
        measured.append(
            BatchSizeRow(
                batch_size=batch_size,
                steps_per_epoch=math.ceil(n_rows / batch_size),
                total_steps=steps,
                losses=[_reportable(value) for value in losses],
                final_loss=_reportable(final),
                final_accuracy=_accuracy_under(stack, rows, labels, "log_loss"),
                milliseconds=elapsed,
                milliseconds_per_step=elapsed / steps if steps else 0.0,
            )
        )

    whole = _flat_gradient(starting.backward_pass(rows, targets, loss))
    whole_length = float(np.linalg.norm(whole))
    drawer = np.random.default_rng(request.order_seed)
    agreement: list[DirectionAgreement] = []
    for batch_size in request.batch_sizes:
        cosines: list[float] = []
        for _ in range(DIRECTION_DRAWS):
            picked = drawer.choice(n_rows, size=batch_size, replace=False)
            part = _flat_gradient(
                starting.backward_pass(rows[picked], targets[picked], loss)
            )
            length = float(np.linalg.norm(part))
            cosines.append(float(part @ whole / (length * whole_length)))
        agreement.append(
            DirectionAgreement(
                batch_size=batch_size,
                mean_cosine=float(np.mean(cosines)),
                worst_cosine=float(np.min(cosines)),
                n_draws=DIRECTION_DRAWS,
            )
        )

    return BatchSweepResponse(
        n_rows=n_rows,
        learning_rate=request.learning_rate,
        max_epochs=request.max_epochs,
        starting_loss=starting_loss,
        rows=measured,
        agreement=agreement,
    )


class GeneralisationReading(BaseModel):
    """What the two sets of people cost at one epoch."""

    epoch: int
    training_loss: float | None
    held_out_loss: float | None
    training_accuracy: float
    held_out_accuracy: float


class GeneralisationRequest(BaseModel):
    """A training set small enough to memorise, and everybody else held back.

    ``mislabelled`` flips that many of the training labels, which is what
    gives the network something to memorise that the rest of the crowd does
    not share. At zero the held-out score stops improving; above zero it gets
    worse.
    """

    preset: Preset = "rings"
    n_training: int = Field(default=14, ge=4, le=40)
    mislabelled: int = Field(default=3, ge=0, le=6)
    hidden_width: int = Field(default=8, ge=MIN_HIDDEN, le=MAX_HIDDEN)
    learning_rate: float = Field(default=1.0, gt=0.0, le=MAX_LEARNING_RATE)
    max_epochs: int = Field(default=600, ge=1, le=MAX_GENERALISATION_EPOCHS)
    weight_seed: int = Field(default=1, ge=0, le=9999)
    split_seed: int = Field(default=5, ge=0, le=9999)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _leave_a_majority_labelled_correctly(self) -> GeneralisationRequest:
        """Refuse a split where the flipped labels could outvote the rest.

        A training set of four with three labels flipped is not a memorisation
        fixture, it is a differently labelled problem, so the flips are capped
        at a third of the rows.
        """
        if self.mislabelled * 3 > self.n_training:
            raise ValueError(
                "mislabelled must not exceed a third of the training rows, "
                f"got {self.mislabelled} of {self.n_training}"
            )
        return self


class GeneralisationResponse(BaseModel):
    """The training loss and the held-out score across the same epochs.

    ``best_held_out_loss_epoch`` is where the walk should have stopped, and
    the training loss keeps falling past it, which is the whole point of
    reporting both.
    """

    n_training: int
    n_held_out: int
    n_mislabelled: int
    n_parameters: int
    readings: list[GeneralisationReading]
    best_held_out_loss: float
    best_held_out_loss_epoch: int
    best_held_out_accuracy: float
    best_held_out_accuracy_epoch: int
    final_training_loss: float
    final_held_out_loss: float
    final_training_accuracy: float
    final_held_out_accuracy: float


def _stratified_split(
    labels: np.ndarray, n_training: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Half the training rows from each class, the rest held back.

    Drawing at random would leave a small training set able to miss a class
    entirely, and a fit on one class has nothing to discriminate between, so
    the two halves are drawn separately.
    """
    generator = np.random.default_rng(split_seed)
    wanted = n_training // 2
    training: list[np.ndarray] = []
    held: list[np.ndarray] = []

    for label in (0.0, 1.0):
        positions = np.flatnonzero(labels == label)
        shuffled = generator.permutation(positions)
        training.append(shuffled[:wanted])
        held.append(shuffled[wanted:])

    return np.concatenate(training), np.concatenate(held)


@router.post("/generalisation", response_model=GeneralisationResponse)
def generalisation(request: GeneralisationRequest) -> GeneralisationResponse:
    """Train on a handful of people and score everybody else at every epoch.

    Two losses are measured each epoch, one on the rows the step is taken
    from and one on the rows nothing is learned from. Only the first is
    available to a loop that has no held-out set, and the page's claim is that
    the first one falling says nothing about the second.
    """
    rows, labels = _drawn_people(request.preset)
    loss = BinaryCrossEntropy()

    training_index, held_index = _stratified_split(
        labels, request.n_training, request.split_seed
    )
    training_rows = rows[training_index]
    training_labels = labels[training_index].copy()
    if request.mislabelled:
        flipper = np.random.default_rng(request.split_seed + 1)
        flipped = flipper.choice(
            len(training_index), size=request.mislabelled, replace=False
        )
        training_labels[flipped] = 1.0 - training_labels[flipped]

    held_rows = rows[held_index]
    held_labels = labels[held_index]
    training_targets = training_labels.reshape(-1, 1)
    held_targets = held_labels.reshape(-1, 1)

    stack = _fresh_stack(request.hidden_width, request.weight_seed)
    readings: list[GeneralisationReading] = []

    for epoch in range(request.max_epochs + 1):
        backward = stack.backward_pass(training_rows, training_targets, loss)
        held = stack.backward_pass(held_rows, held_targets, loss)
        readings.append(
            GeneralisationReading(
                epoch=epoch,
                training_loss=_reportable(backward.loss),
                held_out_loss=_reportable(held.loss),
                training_accuracy=_accuracy(stack, training_rows, training_labels),
                held_out_accuracy=_accuracy(stack, held_rows, held_labels),
            )
        )
        stack = stack.stepped_by(backward, request.learning_rate)

    usable = [reading for reading in readings if reading.held_out_loss is not None]
    lowest = min(usable, key=lambda reading: reading.held_out_loss or 0.0)
    highest = max(readings, key=lambda reading: reading.held_out_accuracy)
    last = readings[-1]

    return GeneralisationResponse(
        n_training=len(training_index),
        n_held_out=len(held_index),
        n_mislabelled=request.mislabelled,
        n_parameters=sum(
            layer.shape.n_inputs * layer.shape.n_outputs + layer.shape.n_outputs
            for layer in stack
        ),
        readings=readings,
        best_held_out_loss=lowest.held_out_loss or 0.0,
        best_held_out_loss_epoch=lowest.epoch,
        best_held_out_accuracy=highest.held_out_accuracy,
        best_held_out_accuracy_epoch=highest.epoch,
        final_training_loss=last.training_loss or 0.0,
        final_held_out_loss=last.held_out_loss or 0.0,
        final_training_accuracy=last.training_accuracy,
        final_held_out_accuracy=last.held_out_accuracy,
    )


class RepeatedRun(BaseModel):
    """Two runs that differ in one setting, and whether they landed alike."""

    description: str
    first_final_loss: float
    second_final_loss: float
    identical: bool


class SeededStart(BaseModel):
    """One draw of the starting weights, and where the same loop took it."""

    weight_seed: int
    starting_loss: float
    final_loss: float
    final_accuracy: float | None


class DrawSequence(BaseModel):
    """Four consecutive draws from a layer that silences units at random.

    ``sheets`` holds which units survived on each pass, one row per pass.
    ``all_alike`` is true when every pass drew the same units, which is what a
    rebuilt layer does and what a stepped one must not.
    """

    description: str
    sheets: list[list[float]]
    all_alike: bool


class RepeatabilityResponse(BaseModel):
    """What a seed fixes, what it does not, and the draw that must not restart."""

    runs: list[RepeatedRun]
    starts: list[SeededStart]
    draws: list[DrawSequence]
    step_returns_the_same_layer: bool


def _batched_final_loss(
    batch_size: int, weight_seed: int, order_seed: int, epochs: int = 60
) -> float:
    """Run the batched loop once and report the loss it finished on."""
    rows, labels = _drawn_people("rings")
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()
    n_rows = int(rows.shape[0])

    stack = _fresh_stack(6, weight_seed)
    shuffler = np.random.default_rng(order_seed)
    for _ in range(epochs):
        order = shuffler.permutation(n_rows)
        for start in range(0, n_rows, batch_size):
            picked = order[start : start + batch_size]
            backward = stack.backward_pass(rows[picked], targets[picked], loss)
            stack = stack.stepped_by(backward, 0.5)
    return stack.backward_pass(rows, targets, loss).loss


@router.get("/repeatability", response_model=RepeatabilityResponse)
def repeatability() -> RepeatabilityResponse:
    """Run the same configuration twice, and vary one seed at a time.

    Three questions, each answered by running the loop rather than by
    reasoning about it. Does the same configuration land in the same place;
    does the order the rows are visited in matter; and does a layer that draws
    at random keep drawing after a step.
    """
    rows, labels = _drawn_people("rings")
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()

    small_first = _batched_final_loss(10, 1, 3)
    small_repeat = _batched_final_loss(10, 1, 3)
    small_second = _batched_final_loss(10, 1, 4)
    whole_first = _batched_final_loss(60, 1, 3)
    whole_second = _batched_final_loss(60, 1, 4)

    runs = [
        RepeatedRun(
            description="the identical request, sent twice, batches of ten",
            first_final_loss=small_first,
            second_final_loss=small_repeat,
            identical=small_first == small_repeat,
        ),
        RepeatedRun(
            description="the same weights, a different order of rows, batches of ten",
            first_final_loss=small_first,
            second_final_loss=small_second,
            identical=small_first == small_second,
        ),
        RepeatedRun(
            description="the same weights, a different order of rows, one whole batch",
            first_final_loss=whole_first,
            second_final_loss=whole_second,
            identical=whole_first == whole_second,
        ),
    ]

    starts: list[SeededStart] = []
    for weight_seed in (1, 2, 3, 4):
        stack = _fresh_stack(6, weight_seed)
        starting = stack.backward_pass(rows, targets, loss).loss
        for _ in range(200):
            backward = stack.backward_pass(rows, targets, loss)
            stack = stack.stepped_by(backward, 0.5)
        starts.append(
            SeededStart(
                weight_seed=weight_seed,
                starting_loss=starting,
                final_loss=stack.backward_pass(rows, targets, loss).loss,
                final_accuracy=_accuracy_under(stack, rows, labels, "log_loss"),
            )
        )

    sheet = np.ones((1, 6))
    kept_by_one_layer: list[list[float]] = []
    silencer = Dropout(reads=6, drop_probability=0.5, random_seed=11)
    same_object = silencer.stepped_by(None, 0.5) is silencer
    for _ in range(4):
        response = silencer.respond_to(sheet, PassPurpose.TRAINING)
        kept_by_one_layer.append(
            [float(value) for value in np.asarray(response.kept).ravel()]
        )

    kept_by_rebuilt: list[list[float]] = []
    for _ in range(4):
        rebuilt = Dropout(reads=6, drop_probability=0.5, random_seed=11)
        response = rebuilt.respond_to(sheet, PassPurpose.TRAINING)
        kept_by_rebuilt.append(
            [float(value) for value in np.asarray(response.kept).ravel()]
        )

    draws = [
        DrawSequence(
            description="one layer kept across the steps",
            sheets=kept_by_one_layer,
            all_alike=all(row == kept_by_one_layer[0] for row in kept_by_one_layer),
        ),
        DrawSequence(
            description="the layer rebuilt from its seed before every pass",
            sheets=kept_by_rebuilt,
            all_alike=all(row == kept_by_rebuilt[0] for row in kept_by_rebuilt),
        ),
    ]

    return RepeatabilityResponse(
        runs=runs,
        starts=starts,
        draws=draws,
        step_returns_the_same_layer=same_object,
    )


class ContractRow(BaseModel):
    """One thing a training loop can be asked to do, and what happened.

    ``verdict`` says whether the library refused or went ahead, and ``detail``
    is the refusal in the library's own words or the number that came back.
    """

    situation: str
    verdict: Literal["refused", "accepted"]
    detail: str


class ContractsResponse(BaseModel):
    """Every edge the page's last part quotes, probed when it is asked for."""

    rows: list[ContractRow]


def _probed(situation: str, attempt) -> ContractRow:
    """Run one edge case and report what the library did with it."""
    try:
        outcome = attempt()
    except MLLibError as error:
        return ContractRow(
            situation=situation,
            verdict="refused",
            detail=f"{type(error).__name__}: {error}",
        )
    return ContractRow(situation=situation, verdict="accepted", detail=str(outcome))


@router.get("/loop-contracts", response_model=ContractsResponse)
def loop_contracts() -> ContractsResponse:
    """Ask the library to do each awkward thing and report what it answered.

    Nothing here is written down from memory. Every row runs when the request
    arrives, so a row saying a thing is refused is a row that watched it be
    refused a moment ago.
    """
    rows, labels = _drawn_people("rings")
    targets = labels.reshape(-1, 1)
    loss = BinaryCrossEntropy()
    stack = _fresh_stack(6, 1)
    backward = stack.backward_pass(rows, targets, loss)

    def stepped_loss(rate: float) -> str:
        moved = stack.stepped_by(backward, rate)
        reached = moved.backward_pass(rows, targets, loss).loss
        direction = (
            "unchanged"
            if reached == backward.loss
            else ("lower" if reached < backward.loss else "higher")
        )
        return f"loss {backward.loss:.6f} becomes {reached:.6f}, {direction}"

    def no_epochs() -> str:
        walked, losses, _, _, _ = _walk(stack, rows, targets, loss, 0.5, 0)
        return (
            f"{len(losses)} losses recorded, and the chain that comes back is "
            f"the very chain that went in, {walked is stack}"
        )

    def oversized_batch() -> str:
        picked = rows[:200]
        value = stack.backward_pass(picked, targets[:200], loss).loss
        return (
            f"asking for 200 of 60 rows hands back {picked.shape[0]}, "
            f"loss {value:.6f}, the whole set"
        )

    def single_row() -> str:
        value = stack.backward_pass(rows[:1], targets[:1], loss).loss
        return f"loss {value:.6f}, measured on that one person"

    def empty_batch() -> float:
        return stack.backward_pass(rows[:0], targets[:0], loss).loss

    def wrong_width() -> float:
        doubled = np.column_stack([labels, labels])
        return stack.backward_pass(rows, doubled, loss).loss

    def wrong_length() -> float:
        return stack.backward_pass(rows, targets[:30], loss).loss

    def not_a_number() -> float:
        spoiled = rows.copy()
        spoiled[3, 1] = np.nan
        return stack.backward_pass(spoiled, targets, loss).loss

    def infinite_value() -> float:
        spoiled = rows.copy()
        spoiled[7, 0] = np.inf
        return stack.backward_pass(spoiled, targets, loss).loss

    def runaway() -> float:
        walking = _fresh_stack(6, 1)
        squared = SquaredError()
        for _ in range(MAX_EPOCHS):
            walking = walking.stepped_by(
                walking.backward_pass(rows, targets, squared), 2.0
            )
        return walking.backward_pass(rows, targets, squared).loss

    def runaway_watched() -> str:
        _, losses, first_bad, refused_at, _ = _walk(
            _fresh_stack(6, 1), rows, targets, SquaredError(), 2.0, MAX_EPOCHS
        )
        finite = [value for value in losses if math.isfinite(value)]
        return (
            f"the loss climbs to {finite[-1]:.3g} by epoch {len(finite)}, "
            f"is not a number at epoch {first_bad}, and the step is refused "
            f"at epoch {refused_at}"
        )

    def untrained_answer() -> str:
        fresh = _fresh_stack(6, 1)
        answer = np.asarray(fresh.respond_to(rows[:1]).outputs, dtype=np.float64)
        return f"answers {float(answer[0, 0]):.4f} about the first person"

    def bad_join() -> LayerStack:
        return LayerStack(
            [
                DenseLayer(
                    [
                        Neuron(
                            weights=[0.1, 0.2], bias=0.0, activation=HyperbolicTangent()
                        )
                        for _ in range(3)
                    ]
                ),
                DenseLayer(
                    [Neuron(weights=[0.1, 0.2], bias=0.0, activation=Identity())]
                ),
            ]
        )

    return ContractsResponse(
        rows=[
            _probed("a step size of zero", lambda: stepped_loss(0.0)),
            _probed("a negative step size", lambda: stepped_loss(-0.5)),
            _probed("zero epochs", no_epochs),
            _probed("a batch larger than the training set", oversized_batch),
            _probed("a batch of one row", single_row),
            _probed("a batch of no rows", empty_batch),
            _probed("a chain answering one number, targets two wide", wrong_width),
            _probed("targets of the wrong length", wrong_length),
            _probed("a value that is not a number in the rows", not_a_number),
            _probed("an infinite value in the rows", infinite_value),
            _probed("a walk that runs away, squared error at a rate of two", runaway),
            _probed("that same runaway walk, watched epoch by epoch", runaway_watched),
            _probed("two layers whose widths disagree", bad_join),
            _probed("asking the chain before it has been trained", untrained_answer),
        ]
    )
