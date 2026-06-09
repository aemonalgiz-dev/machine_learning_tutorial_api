"""Dropout, for the dropout page.

Three endpoints. The first takes one row of unit values and pushes it through
the library's ``Dropout`` layer in both of the layer's moods, once while
predicting, which the library answers with the row untouched, and as many
times while training as the request asks. The first draw is the one the page
draws as bars, together with the scaled mask the library carried on its
response, so the widget names the silenced units from the mask and never from
the outputs. The remaining draws are for the average. Inverted dropout claims
that what a unit sends while training averages to what it sends while
predicting, and a claim about an average is checked by averaging, so the
endpoint keeps every draw, returns the running mean after each one, and counts
how often each unit survived, read from the masks. Every draw comes from one
seeded layer whose generator advances between passes, the way it would inside
a training loop, so a shorter run is a prefix of a longer one.

The second takes a whole block, several rows of units, and runs one training
pass and one backward step through it. A block rather than a row because the
draw is independent per row, which a single row cannot show, and a backward
step because the mask is the whole of it: the blame passed down is the
arriving block multiplied by the same scaled mask the forward pass drew. The
endpoint also reports what reading the mask back off the outputs would have
said, so the page can count the entries that shortcut misfiles, and it checks
the passed-down block against a finite difference of the forward pass, which
is the oracle the library's own spec uses.

The third is an experiment on the classification pages' tangled crowd, twenty
five people by height and weight with the middle deliberately mixed, which is
where a wide network memorises. A small rectified network is trained with no
dropout and with dropout at two rates, held out fold by fold across several
seeds, and the endpoint returns training against held-out accuracy for every
combination, the accuracy curves of one seed, how much each hidden unit
matters when silenced on its own after each kind of training, and what
happens when the two purposes are confused. The library records no history of
its own, so this module owns the loop, as the training page's module does,
and nothing here computes a gradient or a step.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    BinaryCrossEntropy,
    DenseLayer,
    Dropout,
    DropoutResponse,
    Identity,
    LayerStack,
    Neuron,
    PassPurpose,
    RectifiedLinear,
)
from oop_ml.core.network.activation import Activation
from oop_ml.core.network.layer import Layer, LayerResponse
from pydantic import BaseModel, Field, model_validator

from oop_ml_api.concepts.two_feature_classification import LabelledPoint
from oop_ml_api.limits import MAX_MAGNITUDE

router = APIRouter(tags=["dropout"])

MIN_UNITS = 1
"""The fewest units a row may hold, since a layer needs something to read."""

MAX_UNITS = 12
"""The most units the widget's row may hold."""

MAX_DRAWS = 1000
"""The most training passes one request may ask for.

A pass is one row through a layer with no parameters, so a thousand of them
is a few milliseconds, and the running mean has settled long before then. The
bound exists so a crafted request cannot ask for a million.
"""

MAX_SEED = 2**32 - 1
"""The largest seed the generator is handed, one whole 32-bit word."""

WORKED_EXAMPLE_SEED = 10
"""The seed the page's worked example runs under, and the request's default.

Chosen because its first draw on four units at a drop probability of one half
keeps the first and third units and silences the second and fourth, so one row
of hand arithmetic shows both halves of the rule. The test module pins every
number the page quotes to this seed.
"""

WORKED_EXAMPLE_DRAWS = 400
"""How many training passes the worked example averages over, and the default.

Enough for every unit's mean to land within a few hundredths of its predicting
output, and few enough that the running mean still visibly wanders on the way.
"""

MAX_BLOCK_ROWS = 6
"""The most rows the block widget's grid may hold."""

FINITE_DIFFERENCE_STEP = 1e-6
"""How far each input is nudged when the backward step is checked.

The forward pass is linear in every input once the mask is drawn, so the
finite difference is exact up to rounding at any step; this one keeps the
rounding near 1e-10 on values of a few units.
"""

BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]


class ApplyRequest(BaseModel):
    """The row of unit values, the chance each is silenced, and how to draw.

    ``drop_probability`` is bounded at one on this side, not below it, so that
    a value of exactly one reaches the library and is refused in its own words,
    since silencing every unit leaves nothing to scale.
    """

    values: list[BoundedValue] = Field(min_length=MIN_UNITS, max_length=MAX_UNITS)
    drop_probability: float = Field(ge=0.0, le=1.0)
    random_seed: int = Field(default=WORKED_EXAMPLE_SEED, ge=0, le=MAX_SEED)
    n_draws: int = Field(default=WORKED_EXAMPLE_DRAWS, ge=1, le=MAX_DRAWS)

    model_config = {"extra": "forbid"}


class ApplyResponse(BaseModel):
    """One training draw, the predicting pass, and the average over many draws.

    ``scale`` is ``1 / (1 - p)``, the factor every survivor is multiplied by.
    ``mask`` is the scaled mask the library's response carried for the first
    draw, ``scale`` where a unit survived and zero where it did not, and
    ``kept`` is that mask read as a flag, so a unit that held zero and
    survived is reported as kept. ``training_output`` is the first draw's
    answer and ``predicting_output`` is the same row passed while predicting,
    which is the row itself.

    ``running_means`` holds, after each of the ``n_draws`` training passes,
    the mean so far of every unit's output, so its first entry is the first
    draw and its last is ``mean_output``. ``times_kept`` counts, per unit, how
    many passes the unit survived, read from the masks and never from the
    outputs. ``largest_gap`` is the widest distance between any unit's final
    mean and its predicting output. ``single_draw_low`` and
    ``single_draw_high`` are the smallest and largest output any one draw gave
    each unit, which is the spread the average hides.
    """

    scale: float
    training_output: list[float]
    mask: list[float]
    kept: list[bool]
    predicting_output: list[float]
    n_draws: int
    times_kept: list[int]
    running_means: list[list[float]]
    mean_output: list[float]
    largest_gap: float
    single_draw_low: list[float]
    single_draw_high: list[float]


@router.post("/concepts/dropout/apply", response_model=ApplyResponse)
def apply_dropout(request: ApplyRequest) -> ApplyResponse:
    """Push one row through a seeded dropout layer, once predicting, many times training."""
    width = len(request.values)
    row = np.asarray([request.values], dtype=np.float64)
    layer = Dropout(
        reads=width,
        drop_probability=request.drop_probability,
        random_seed=request.random_seed,
    )

    predicting_output = np.asarray(layer.respond_to(row).outputs[0], dtype=np.float64)

    outputs = np.empty((request.n_draws, width), dtype=np.float64)
    masks = np.empty((request.n_draws, width), dtype=np.float64)
    for draw_index in range(request.n_draws):
        response = _training_pass(layer, row)
        outputs[draw_index] = response.outputs[0]
        masks[draw_index] = response.kept[0]

    draws_so_far = np.arange(1, request.n_draws + 1, dtype=np.float64)
    running_means = np.cumsum(outputs, axis=0) / draws_so_far[:, None]
    mean_output = running_means[-1]
    survived = masks > 0.0

    return ApplyResponse(
        scale=float(1.0 / layer.keep_probability),
        training_output=[float(value) for value in outputs[0]],
        mask=[float(value) for value in masks[0]],
        kept=[bool(flag) for flag in survived[0]],
        predicting_output=[float(value) for value in predicting_output],
        n_draws=request.n_draws,
        times_kept=[int(count) for count in survived.sum(axis=0)],
        running_means=[[float(value) for value in means] for means in running_means],
        mean_output=[float(value) for value in mean_output],
        largest_gap=float(np.max(np.abs(mean_output - predicting_output))),
        single_draw_low=[float(value) for value in outputs.min(axis=0)],
        single_draw_high=[float(value) for value in outputs.max(axis=0)],
    )


def _training_pass(layer: Dropout, block: np.ndarray) -> DropoutResponse:
    """One training pass, narrowed to the response type that carries a mask.

    The library answers a training pass with a ``DropoutResponse`` and a
    predicting pass with a plain one, and only the first carries ``kept``.
    The check is the one the library's own backward step makes before it will
    read a mask, and failing it here is a server fault rather than a guess.
    """
    response = layer.respond_to(block, purpose=PassPurpose.TRAINING)
    if not isinstance(response, DropoutResponse):
        raise TypeError("a training pass through dropout must carry its mask")
    return response


class BlockRequest(BaseModel):
    """A block of rows, the chance each entry is silenced, and the blame arriving.

    ``arriving`` is the slope of some loss at the layer's outputs, one entry
    per output, and is what the backward step multiplies by the mask. It
    defaults to a block of ones, so the passed-down block is the mask itself.
    Every row must hold the same number of units, since a layer reads one
    width.
    """

    rows: list[list[BoundedValue]] = Field(min_length=1, max_length=MAX_BLOCK_ROWS)
    drop_probability: float = Field(ge=0.0, le=1.0)
    random_seed: int = Field(default=WORKED_EXAMPLE_SEED, ge=0, le=MAX_SEED)
    arriving: list[list[BoundedValue]] | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _one_width_throughout(self) -> BlockRequest:
        widths = {len(row) for row in self.rows}
        if len(widths) != 1:
            raise ValueError(
                "every row of the block must hold the same number of units"
            )
        width = widths.pop()
        if not MIN_UNITS <= width <= MAX_UNITS:
            raise ValueError(
                f"a row holds between {MIN_UNITS} and {MAX_UNITS} units, got {width}"
            )
        if self.arriving is not None and (
            len(self.arriving) != len(self.rows)
            or any(len(row) != width for row in self.arriving)
        ):
            raise ValueError(
                "the arriving block must be arranged exactly as the rows are"
            )
        return self


class BlockResponse(BaseModel):
    """One training pass through a block, and the backward step through its mask.

    ``mask`` is the scaled mask the library carried, ``scale`` where an entry
    survived and zero where it did not, drawn independently for every row.
    ``kept`` is that mask as a flag. ``mask_read_from_outputs`` is what the
    shortcut would record, kept wherever the output is not zero, and
    ``misfiled`` marks every entry where the two disagree, which is exactly
    the entries that held zero and survived; ``n_misfiled`` counts them.

    ``passed_down`` is the backward step's answer, the arriving block seen
    through the mask, and ``gradient_is_none`` says the layer reported nothing
    to learn. ``finite_difference`` is the slope of ``sum(arriving * outputs)``
    at every input, measured by nudging each input under the same seed, and
    ``largest_finite_difference_gap`` is how far it sits from ``passed_down``.
    ``n_kept_per_row`` counts the survivors in each row.
    """

    scale: float
    predicting_outputs: list[list[float]]
    training_outputs: list[list[float]]
    mask: list[list[float]]
    kept: list[list[bool]]
    mask_read_from_outputs: list[list[bool]]
    misfiled: list[list[bool]]
    n_misfiled: int
    n_kept_per_row: list[int]
    arriving: list[list[float]]
    passed_down: list[list[float]]
    gradient_is_none: bool
    finite_difference: list[list[float]]
    largest_finite_difference_gap: float


def _block_layer(width: int, drop_probability: float, random_seed: int) -> Dropout:
    return Dropout(
        reads=width, drop_probability=drop_probability, random_seed=random_seed
    )


@router.post("/concepts/dropout/block", response_model=BlockResponse)
def pass_block(request: BlockRequest) -> BlockResponse:
    """Run one training pass over a block and send blame back through its mask."""
    block = np.asarray(request.rows, dtype=np.float64)
    n_rows, width = block.shape
    arriving = (
        np.ones_like(block)
        if request.arriving is None
        else np.asarray(request.arriving, dtype=np.float64)
    )

    layer = _block_layer(width, request.drop_probability, request.random_seed)
    predicting = np.asarray(layer.respond_to(block).outputs, dtype=np.float64)

    layer = _block_layer(width, request.drop_probability, request.random_seed)
    response = _training_pass(layer, block)
    outputs = np.asarray(response.outputs, dtype=np.float64)
    mask = np.asarray(response.kept, dtype=np.float64)
    kept = mask > 0.0
    read_from_outputs = outputs != 0.0
    misfiled = kept != read_from_outputs

    correction = layer.correction_for(response, arriving)
    passed_down = np.asarray(correction.passed_down, dtype=np.float64)

    # The oracle: nudge one input, rerun the forward pass under the same seed
    # so the same mask is drawn, and difference the weighted total.
    finite_difference = np.empty_like(block)
    for row_index in range(n_rows):
        for unit_index in range(width):
            up = block.copy()
            up[row_index, unit_index] += FINITE_DIFFERENCE_STEP
            down = block.copy()
            down[row_index, unit_index] -= FINITE_DIFFERENCE_STEP
            finite_difference[row_index, unit_index] = (
                _weighted_total(width, request, up, arriving)
                - _weighted_total(width, request, down, arriving)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)

    return BlockResponse(
        scale=float(1.0 / layer.keep_probability),
        predicting_outputs=_rows_of(predicting),
        training_outputs=_rows_of(outputs),
        mask=_rows_of(mask),
        kept=_flags_of(kept),
        mask_read_from_outputs=_flags_of(read_from_outputs),
        misfiled=_flags_of(misfiled),
        n_misfiled=int(np.count_nonzero(misfiled)),
        n_kept_per_row=[int(count) for count in kept.sum(axis=1)],
        arriving=_rows_of(arriving),
        passed_down=_rows_of(passed_down),
        gradient_is_none=correction.gradient is None,
        finite_difference=_rows_of(finite_difference),
        largest_finite_difference_gap=float(
            np.max(np.abs(finite_difference - passed_down))
        ),
    )


def _weighted_total(
    width: int, request: BlockRequest, block: np.ndarray, arriving: np.ndarray
) -> float:
    """``sum(arriving * outputs)`` for a fresh layer under the request's seed."""
    layer = _block_layer(width, request.drop_probability, request.random_seed)
    outputs = np.asarray(_training_pass(layer, block).outputs, dtype=np.float64)
    return float(np.sum(arriving * outputs))


def _rows_of(block: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in block]


def _flags_of(block: np.ndarray) -> list[list[bool]]:
    return [[bool(value) for value in row] for row in block]


TANGLED_CROWD: tuple[tuple[float, float, int], ...] = (
    (147, 41, 0),
    (156, 53, 1),
    (145, 57, 0),
    (159, 57, 1),
    (162, 61, 1),
    (120, 25, 0),
    (122, 28, 0),
    (118, 24, 0),
    (180, 80, 1),
    (183, 83, 1),
    (178, 78, 1),
    (145, 45, 0),
    (145, 55, 1),
    (151, 45, 1),
    (151, 55, 0),
    (157, 45, 0),
    (157, 55, 1),
    (148, 50, 1),
    (154, 50, 0),
    (160, 50, 1),
    (147, 58, 0),
    (153, 58, 1),
    (150, 42, 1),
    (156, 42, 0),
    (143, 50, 0),
)
"""The bagging pages' twenty-five people, height in centimetres, weight in
kilograms, and whether they are an adult, with the middle of the crowd mixed
on purpose so that a model can only fit it by memorising who is who."""

HIDDEN_WIDTH = 16
"""How many rectified units the hidden layer holds.

Wide for twenty rows of two numbers, which is the point: the network has more
than enough capacity to memorise the mixed middle of the crowd, measured at a
training accuracy of 1.0 on the whole crowd with no dropout.
"""

EXPERIMENT_EPOCHS = 400
"""How many full-batch steps each training run takes.

Enough for the undropped network to reach a training accuracy near one on
its twenty rows; measured across the five seeds it averages 0.992.
"""

LEARNING_RATE = 1.0
"""The stride. Large, because plain descent on a rectified network of this
size walks slowly at the training page's half, and the point here is a
network that memorises rather than one that is still learning."""

DROP_RATES = (0.0, 0.2, 0.5)
"""The rates the sweep compares. Zero means no dropout layer at all."""

EXPERIMENT_SEEDS = (1, 2, 3, 4, 5)
"""The seeds the sweep runs under. Each seeds the weights, the fold deal and
the dropout draws together, so one seed is one repeatable experiment."""

N_FOLDS = 5
"""Five folds of the twenty-five people, so every person is held out once
and the held-out accuracy is counted over all twenty-five."""

CURVE_SEED = 1
"""The seed whose fold-by-fold accuracy curves are recorded."""

CURVE_EVERY = 4
"""Record the curves every this many epochs, so a curve holds a hundred points."""

ABLATION_RATE = 0.5
"""The rate the full-crowd network is trained at when each unit is silenced in turn."""

CONFUSION_DRAWS = 20
"""How many times the trained network is asked to predict while still training."""

AVERAGING_DRAWS = 400
"""How many thinned passes are averaged against the predicting pass."""

WITHIN_ROW_DROPOUT_SEED_OFFSET = 1000
"""The dropout layer's seed is the run's seed plus this, so the weights and
the draws come from different streams."""


class SeedReading(BaseModel):
    """One seed's result at one rate: mean training accuracy over the folds,
    and the held-out accuracy pooled over every person held out once."""

    seed: int
    training_accuracy: float
    held_out_accuracy: float


class RateReading(BaseModel):
    """One rate across every seed, with the two means the page compares."""

    drop_probability: float
    training_accuracy_mean: float
    held_out_accuracy_mean: float
    held_out_accuracy_low: float
    held_out_accuracy_high: float
    per_seed: list[SeedReading]


class CurveReading(BaseModel):
    """The two accuracies after one epoch, pooled over the folds of one seed.

    ``training_accuracy`` is the mean over the folds of each fold's training
    accuracy, read while predicting; ``held_out_accuracy`` is the share of all
    twenty-five people called correctly by the fold that held them out;
    ``thinned_loss`` is the mean over the folds of the loss the backward pass
    measured on the way forward, which under dropout is the loss of one
    thinned network rather than of the whole one.
    """

    epoch: int
    training_accuracy: float
    held_out_accuracy: float
    thinned_loss: float


class RateCurve(BaseModel):
    """The curves for one rate."""

    drop_probability: float
    readings: list[CurveReading]


class UnitAblation(BaseModel):
    """Training accuracy on the whole crowd with each hidden unit silenced in turn.

    ``intact_accuracy`` is with nothing silenced. ``silenced`` holds one
    accuracy per hidden unit. ``mean_drop`` and ``largest_drop`` summarise how
    much the network leans on any single unit. ``dead`` marks a unit whose
    rectified output is zero for every person in the crowd, which silencing
    cannot change, and ``n_dead`` counts them, so a network that leans on
    nothing can be told from one that has nothing left to lean on.
    """

    drop_probability: float
    intact_accuracy: float
    silenced: list[float]
    mean_drop: float
    largest_drop: float
    dead: list[bool]
    n_dead: int


class KeptZeros(BaseModel):
    """In one training pass through the trained network, how many hidden
    readings were exactly zero, and how many of those the mask kept."""

    n_readings: int
    n_zero: int
    n_zero_and_kept: int


class PurposeConfusion(BaseModel):
    """What forgetting each purpose costs, on the network trained at the ablation rate.

    ``predicting_accuracy`` is the trained network asked properly.
    ``training_purpose_accuracies`` are the same network asked while still
    thinning, once per draw, which is the answer a caller gets by forgetting
    to say predicting. ``identity_control_gap`` is the largest difference in
    output between a network trained with a dropout layer at rate zero and
    one trained with no dropout layer at all, same seed, which is what a
    caller who never said training would have built. The three ``swing``
    fields follow the one person whose chance of being called an adult moved
    most across the draws: what the network says of them while predicting,
    and the lowest and highest chance any thinned pass gave them.
    """

    predicting_accuracy: float
    training_purpose_accuracies: list[float]
    training_purpose_low: float
    training_purpose_high: float
    identity_control_gap: float
    swing_predicting_chance: float
    swing_low_chance: float
    swing_high_chance: float


class ScoreAveraging(BaseModel):
    """The averaged thinned output against the predicting output, on the
    trained network.

    ``largest_score_gap`` is the widest gap between any person's mean thinned
    score over the draws and their predicting score, where the score is the
    output layer's raw answer, which is linear in what dropout sends.
    ``largest_single_draw_gap`` is the widest gap any one draw produced.
    ``largest_chance_gap`` is the same comparison after the squash, where the
    mean of the squashed answers is not the squash of the mean.
    ``largest_predicting_score`` is the largest predicting score in magnitude,
    so the gaps can be read against the size of the numbers they are gaps in.
    """

    n_draws: int
    largest_score_gap: float
    largest_single_draw_gap: float
    largest_chance_gap: float
    largest_predicting_score: float


class ExperimentResponse(BaseModel):
    """The whole experiment, computed once and cached."""

    crowd: list[LabelledPoint]
    hidden_width: int
    epochs: int
    learning_rate: float
    n_folds: int
    seeds: list[int]
    sweep: list[RateReading]
    curves: list[RateCurve]
    ablations: list[UnitAblation]
    kept_zeros: KeptZeros
    confusion: PurposeConfusion
    averaging: ScoreAveraging


def _crowd_arrays() -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([[height, weight] for height, weight, _ in TANGLED_CROWD])
    labels = np.asarray([float(label) for _, _, label in TANGLED_CROWD])
    return rows, labels


def _standardised(fitted_on: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rows in standard units of the rows the network is fitted on.

    Fitted on the training rows only, so a held-out person's coordinates never
    reach the fit, which is the pipelines page's rule.
    """
    return (rows - fitted_on.mean(axis=0)) / fitted_on.std(axis=0)


def _fresh_stack(seed: int, drop_probability: float | None) -> LayerStack:
    """A rectified hidden layer, the dropout layer if asked for, and a straight output.

    The weights are drawn with a spread of the square root of two over the
    fan-in, the initialisation the convolution layer uses. ``None`` builds no
    dropout layer at all, where ``0.0`` builds one that silences nothing, and
    the two are compared in the experiment.
    """
    generator = np.random.default_rng(seed)

    def layer(n_inputs: int, n_outputs: int, activation: Activation) -> DenseLayer:
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

    layers: list[Layer] = [layer(2, HIDDEN_WIDTH, RectifiedLinear())]
    if drop_probability is not None:
        layers.append(
            Dropout(
                reads=HIDDEN_WIDTH,
                drop_probability=drop_probability,
                random_seed=seed + WITHIN_ROW_DROPOUT_SEED_OFFSET,
            )
        )
    layers.append(layer(HIDDEN_WIDTH, 1, Identity()))
    return LayerStack(layers)


def _scores(stack: LayerStack, rows: np.ndarray, purpose: PassPurpose) -> np.ndarray:
    return np.asarray(stack.respond_to(rows, purpose).outputs, dtype=np.float64).ravel()


def _squashed(scores: np.ndarray) -> np.ndarray:
    """The loss's own squash, applied here because the loss hands back only scores."""
    return np.where(
        scores >= 0.0,
        1.0 / (1.0 + np.exp(-scores)),
        np.exp(scores) / (1.0 + np.exp(scores)),
    )


def _called(scores: np.ndarray) -> np.ndarray:
    return (_squashed(scores) >= 0.5).astype(np.float64)


def _accuracy(stack: LayerStack, rows: np.ndarray, labels: np.ndarray) -> float:
    return float(
        (_called(_scores(stack, rows, PassPurpose.PREDICTING)) == labels).mean()
    )


def _folds(seed: int) -> list[np.ndarray]:
    """Five folds dealt from the seed, each a set of held-out row indices."""
    order = np.random.default_rng(seed).permutation(len(TANGLED_CROWD))
    return [order[fold::N_FOLDS] for fold in range(N_FOLDS)]


def _train(
    stack: LayerStack,
    rows: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    watch: Callable[[int, LayerStack, float], None] | None = None,
) -> LayerStack:
    """The loop: measure and walk the blame down, step, repeat.

    ``watch`` is called after each epoch with the epoch number, the stepped
    stack and the loss the backward pass measured, for the runs that record
    a curve.
    """
    loss = BinaryCrossEntropy()
    for epoch in range(1, epochs + 1):
        backward = stack.backward_pass(rows, targets, loss)
        stack = stack.stepped_by(backward, LEARNING_RATE)
        if watch is not None:
            watch(epoch, stack, backward.loss)
    return stack


def _rate_or_none(drop_probability: float) -> float | None:
    return None if drop_probability == 0.0 else drop_probability


class CurveTotals:
    """The running totals a curve is pooled from, one slot per recorded epoch."""

    __slots__ = ("held_correct", "loss", "training")

    def __init__(
        self, training: np.ndarray, held_correct: np.ndarray, loss: np.ndarray
    ) -> None:
        self.training = training
        self.held_correct = held_correct
        self.loss = loss


def _curve_watcher(
    totals: CurveTotals,
    training_rows: np.ndarray,
    training_labels: np.ndarray,
    held_rows: np.ndarray,
    held_labels: np.ndarray,
) -> Callable[[int, LayerStack, float], None]:
    """What one fold adds to the pooled curve after each recorded epoch."""

    def watch(epoch: int, stepped: LayerStack, thinned_loss: float) -> None:
        if epoch % CURVE_EVERY != 0:
            return
        position = epoch // CURVE_EVERY - 1
        totals.training[position] += _accuracy(stepped, training_rows, training_labels)
        held_scores = _scores(stepped, held_rows, PassPurpose.PREDICTING)
        totals.held_correct[position] += float(
            (_called(held_scores) == held_labels).sum()
        )
        totals.loss[position] += thinned_loss

    return watch


def _sweep_and_curves() -> tuple[list[RateReading], list[RateCurve]]:
    rows, labels = _crowd_arrays()
    n_people = len(TANGLED_CROWD)
    recorded_epochs = list(range(CURVE_EVERY, EXPERIMENT_EPOCHS + 1, CURVE_EVERY))

    sweep: list[RateReading] = []
    curves: list[RateCurve] = []
    for drop_probability in DROP_RATES:
        per_seed: list[SeedReading] = []
        curve_training = np.zeros(len(recorded_epochs))
        curve_held_correct = np.zeros(len(recorded_epochs))
        curve_loss = np.zeros(len(recorded_epochs))
        for seed in EXPERIMENT_SEEDS:
            held_correct = 0
            training_total = 0.0
            for held_out in _folds(seed):
                in_fold = np.ones(n_people, dtype=bool)
                in_fold[held_out] = False
                training_rows = _standardised(rows[in_fold], rows[in_fold])
                held_rows = _standardised(rows[in_fold], rows[held_out])
                training_labels = labels[in_fold]
                held_labels = labels[held_out]

                watch = (
                    _curve_watcher(
                        CurveTotals(curve_training, curve_held_correct, curve_loss),
                        training_rows,
                        training_labels,
                        held_rows,
                        held_labels,
                    )
                    if seed == CURVE_SEED
                    else None
                )

                stack = _train(
                    _fresh_stack(seed, _rate_or_none(drop_probability)),
                    training_rows,
                    training_labels.reshape(-1, 1),
                    EXPERIMENT_EPOCHS,
                    watch,
                )
                training_total += _accuracy(stack, training_rows, training_labels)
                held_scores = _scores(stack, held_rows, PassPurpose.PREDICTING)
                held_correct += int((_called(held_scores) == held_labels).sum())
            per_seed.append(
                SeedReading(
                    seed=seed,
                    training_accuracy=training_total / N_FOLDS,
                    held_out_accuracy=held_correct / n_people,
                )
            )
        held = [reading.held_out_accuracy for reading in per_seed]
        sweep.append(
            RateReading(
                drop_probability=drop_probability,
                training_accuracy_mean=float(
                    np.mean([reading.training_accuracy for reading in per_seed])
                ),
                held_out_accuracy_mean=float(np.mean(held)),
                held_out_accuracy_low=float(min(held)),
                held_out_accuracy_high=float(max(held)),
                per_seed=per_seed,
            )
        )
        curves.append(
            RateCurve(
                drop_probability=drop_probability,
                readings=[
                    CurveReading(
                        epoch=epoch,
                        training_accuracy=float(curve_training[position] / N_FOLDS),
                        held_out_accuracy=float(
                            curve_held_correct[position] / n_people
                        ),
                        thinned_loss=float(curve_loss[position] / N_FOLDS),
                    )
                    for position, epoch in enumerate(recorded_epochs)
                ],
            )
        )
    return sweep, curves


def _hidden_layer_response(stack: LayerStack, rows: np.ndarray) -> LayerResponse:
    return stack[0].respond_to(rows)


def _ablation(
    stack: LayerStack, rows: np.ndarray, labels: np.ndarray, rate: float
) -> UnitAblation:
    """Silence each hidden unit at prediction time and score the rest.

    The hidden layer answers, one column is zeroed, and the output layer reads
    the rest. The dropout layer, if the stack has one, is a pass-through while
    predicting and is skipped here for that reason.
    """
    hidden = np.asarray(_hidden_layer_response(stack, rows).outputs, dtype=np.float64)
    output_layer = stack[len(stack) - 1]
    intact = _accuracy(stack, rows, labels)
    silenced: list[float] = []
    for unit_index in range(HIDDEN_WIDTH):
        thinned = hidden.copy()
        thinned[:, unit_index] = 0.0
        scores = np.asarray(
            output_layer.respond_to(thinned).outputs, dtype=np.float64
        ).ravel()
        silenced.append(float((_called(scores) == labels).mean()))
    drops = [intact - accuracy for accuracy in silenced]
    dead = np.all(hidden == 0.0, axis=0)
    return UnitAblation(
        drop_probability=rate,
        intact_accuracy=intact,
        silenced=silenced,
        mean_drop=float(np.mean(drops)),
        largest_drop=float(max(drops)),
        dead=[bool(flag) for flag in dead],
        n_dead=int(np.count_nonzero(dead)),
    )


def _kept_zeros(stack: LayerStack, rows: np.ndarray) -> KeptZeros:
    """One training pass through the trained network's hidden and dropout layers."""
    hidden = np.asarray(_hidden_layer_response(stack, rows).outputs, dtype=np.float64)
    dropout = stack[1]
    if not isinstance(dropout, Dropout):
        raise TypeError("the ablation network holds its dropout layer second")
    response = _training_pass(dropout, hidden)
    kept = np.asarray(response.kept) > 0.0
    zero = hidden == 0.0
    return KeptZeros(
        n_readings=int(hidden.size),
        n_zero=int(np.count_nonzero(zero)),
        n_zero_and_kept=int(np.count_nonzero(zero & kept)),
    )


def _confusion(
    trained: LayerStack, rows: np.ndarray, labels: np.ndarray, targets: np.ndarray
) -> PurposeConfusion:
    thinned_chances = np.stack(
        [
            _squashed(_scores(trained, rows, PassPurpose.TRAINING))
            for _ in range(CONFUSION_DRAWS)
        ]
    )
    accuracies = [
        float(((chances >= 0.5).astype(np.float64) == labels).mean())
        for chances in thinned_chances
    ]
    swing = thinned_chances.max(axis=0) - thinned_chances.min(axis=0)
    widest = int(np.argmax(swing))
    predicting_chances = _squashed(_scores(trained, rows, PassPurpose.PREDICTING))
    with_idle_layer = _train(
        _fresh_stack(CURVE_SEED, 0.0), rows, targets, EXPERIMENT_EPOCHS
    )
    without_layer = _train(
        _fresh_stack(CURVE_SEED, None), rows, targets, EXPERIMENT_EPOCHS
    )
    gap = float(
        np.max(
            np.abs(
                _scores(with_idle_layer, rows, PassPurpose.PREDICTING)
                - _scores(without_layer, rows, PassPurpose.PREDICTING)
            )
        )
    )
    return PurposeConfusion(
        predicting_accuracy=_accuracy(trained, rows, labels),
        training_purpose_accuracies=accuracies,
        training_purpose_low=float(min(accuracies)),
        training_purpose_high=float(max(accuracies)),
        identity_control_gap=gap,
        swing_predicting_chance=float(predicting_chances[widest]),
        swing_low_chance=float(thinned_chances[:, widest].min()),
        swing_high_chance=float(thinned_chances[:, widest].max()),
    )


def _averaging(trained: LayerStack, rows: np.ndarray) -> ScoreAveraging:
    predicting = _scores(trained, rows, PassPurpose.PREDICTING)
    draws = np.stack(
        [_scores(trained, rows, PassPurpose.TRAINING) for _ in range(AVERAGING_DRAWS)]
    )
    mean_score = draws.mean(axis=0)
    mean_chance = _squashed(draws).mean(axis=0)
    return ScoreAveraging(
        n_draws=AVERAGING_DRAWS,
        largest_score_gap=float(np.max(np.abs(mean_score - predicting))),
        largest_single_draw_gap=float(np.max(np.abs(draws - predicting))),
        largest_chance_gap=float(np.max(np.abs(mean_chance - _squashed(predicting)))),
        largest_predicting_score=float(np.max(np.abs(predicting))),
    )


@cache
def _experiment() -> ExperimentResponse:
    sweep, curves = _sweep_and_curves()

    rows, labels = _crowd_arrays()
    standard_rows = _standardised(rows, rows)
    targets = labels.reshape(-1, 1)
    undropped = _train(
        _fresh_stack(CURVE_SEED, None), standard_rows, targets, EXPERIMENT_EPOCHS
    )
    dropped = _train(
        _fresh_stack(CURVE_SEED, ABLATION_RATE),
        standard_rows,
        targets,
        EXPERIMENT_EPOCHS,
    )

    return ExperimentResponse(
        crowd=[
            LabelledPoint(x=float(height), y=float(weight), label=label)
            for height, weight, label in TANGLED_CROWD
        ],
        hidden_width=HIDDEN_WIDTH,
        epochs=EXPERIMENT_EPOCHS,
        learning_rate=LEARNING_RATE,
        n_folds=N_FOLDS,
        seeds=list(EXPERIMENT_SEEDS),
        sweep=sweep,
        curves=curves,
        ablations=[
            _ablation(undropped, standard_rows, labels, 0.0),
            _ablation(dropped, standard_rows, labels, ABLATION_RATE),
        ],
        kept_zeros=_kept_zeros(dropped, standard_rows),
        confusion=_confusion(dropped, standard_rows, labels, targets),
        averaging=_averaging(dropped, standard_rows),
    )


@router.get("/concepts/dropout/experiment", response_model=ExperimentResponse)
def experiment() -> ExperimentResponse:
    """The tangled crowd, with and without dropout, held out fold by fold."""
    return _experiment()
