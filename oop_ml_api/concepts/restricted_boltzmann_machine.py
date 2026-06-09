"""A restricted Boltzmann machine's fit and reconstruction, for the generative page.

Nine endpoints. Two of them, ``fit`` and ``reconstruct``, are what the
playground reads; the other seven are what the page's later parts measure.
``step`` replays the contrastive divergence rule one update at a time and
reports both correlation tables beside the weights before and after, ``states``
scores any list of rows the browser builds against one fit, ``epoch-sweep``
refits at a list of epoch counts so a reconstruction can be watched improving,
``hidden-sweep`` refits at a list of hidden widths so the width can be chosen
from a measurement, ``seeds`` refits under several seeds so what a seed fixes
can be separated from what it does not, ``against-memory`` scores the same rows
under both this model and the associative memory of the page before it, and
``contracts`` runs every edge the last part of the page lists.

The first two fit the library's ``RestrictedBoltzmannMachine`` on the
request's patterns under a fixed seed and a constant learning rate. The fit
endpoint returns what the page reads off a fitted machine, per pattern the
hidden probabilities, the reconstruction and the free energy, alongside the
reconstruction error, the epochs run, and the whole of the learned state, the
weight matrix and both biases, so the page can show a weight the reader has
just watched being learned. The reconstruct endpoint takes the same fitting
inputs plus a probe, fits again from scratch under the same seed, and returns
the probe's hidden probabilities, reconstruction and free energy next to the
stored patterns' free energies, so the page can show the probe sitting higher
in free energy than anything the machine was shown. The API holds nothing
between calls, which is why the second endpoint refits rather than looking a
model up, and the fixed seed is what makes its fit the same machine.

Rows are patterns and units are the positions within one, so the request's
list of patterns is turned round on the way in. The library takes one
``Feature`` per visible unit, holding that unit's value in every pattern,
which is the same rows-are-samples arrangement every other model here uses.
The seed fixes the small random weights every fit starts from and every draw
the Gibbs chain makes afterwards, so the same patterns under the same widths
answer the same way every time and the page can quote what the widget shows.

``step`` is the one endpoint that does not go through ``fit``, because a fit
reports where it arrived and the page needs what happened on the way. It
rebuilds the epoch loop out of the library's own value objects, ``GibbsState``
for each end of the chain, ``ContrastiveDivergenceUpdate`` for the three
changes and ``BoltzmannParameters.shifted_by`` to apply them, so every number
it reports is the library's arithmetic and only the loop and the Bernoulli
draw are written here. That is a claim worth checking rather than asserting,
so a test fits the model for the same number of epochs under the same seed and
pins that the replayed weights and both biases agree with it to exactly 0.0.
The same fact is why a walk of ten epochs is the first ten epochs of a walk of
five hundred, which is what lets the sweep endpoints refit rather than
remember.

``contracts`` reports every refusal in plain words. The messages the library
writes name a column, a class or a field, and a reader of the website has no
reason to meet any of those, so ``_in_plain_words`` substitutes them out and a
test asserts that nothing identifier-shaped survives, which means a message
that grows one later fails there rather than reaching the page.

Both fitting endpoints also report the free energy of every row before any learning,
because that is the one figure on the page that can be checked by hand. With
zero weights and zero biases it is minus the hidden count times the natural
logarithm of two. The library starts a fit from normal weights of spread 0.01
and zero biases rather than from zeros, and exposes no way to read a model
before its first epoch, so the starting point is observed by fitting once more
at a learning rate of zero for a single epoch. A zero rate adds nothing to the
initial parameters, so that fit answers with exactly the weights the seed
drew, and its free energy differs from the hand figure only by the noise in
those weights, which the page states as the reason the readout is not exact.

The values are bounded here rather than by the library. The library accepts
any value in [0, 1] and reads one strictly between as a mean, which is a
legitimate input for it and not one the page's grids can draw, so the request
models allow only 0 and 1. What is checked in words is that the patterns are
all the same length and that the probe matches them, because the turning
round happens before the library sees them and a ragged list has no column to
read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BipolarPattern,
    BoltzmannParameters,
    ConstantSchedule,
    ContrastiveDivergenceUpdate,
    Feature,
    GibbsState,
    HopfieldNetwork,
    RestrictedBoltzmannMachine,
)
from oop_ml.core.exceptions import MLLibError
from oop_ml.numpy.generative.restricted_boltzmann_machine import INITIAL_WEIGHT_SPREAD
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(tags=["restricted-boltzmann-machine"])

MIN_PATTERNS = 1
"""The fewest patterns a machine can learn from. One is a distribution of one."""

MAX_PATTERNS = 12
"""The most patterns the widget may hand over.

Twelve five-by-five shapes is far more than the widget draws, and a fit over
twelve rows of thirty-six units at the widest hidden layer and the longest
walk still answers in well under a tenth of a second.
"""

MIN_UNITS = 4
"""The fewest visible units a pattern may have, the width of the smallest grid."""

MAX_UNITS = 36
"""The most visible units a pattern may have, a six-by-six grid."""

MIN_HIDDEN_UNITS = 1
"""The narrowest hidden layer. One unit can still tell some rows from others."""

MAX_HIDDEN_UNITS = 8
"""The widest hidden layer the widget may ask for.

Eight over three stored shapes is already wider than the shapes need, which is
the case the page wants the reader to be able to reach, a machine with more
units than distinctions to make.
"""

MIN_EPOCHS = 1
"""The shortest walk, a single update from the starting weights."""

MAX_EPOCHS = 500
"""The longest walk the widget may ask for.

At the fixed rate the three shapes are rebuilt to within a hundredth of mean
squared error by five hundred epochs, so the slider's far end is a machine that
has finished learning what it was shown, and the interesting direction is back
towards the start.
"""

LEARNING_RATE = 0.1
"""The constant rate every fit walks at, the library's own default."""

FIXED_SEED = 0
"""The starting weights and every Gibbs draw are seeded so the pages can quote
what the widget shows."""

BinaryValue = Annotated[int, Field(ge=0, le=1)]

PatternValues = Annotated[
    list[BinaryValue], Field(min_length=MIN_UNITS, max_length=MAX_UNITS)
]


class FitRequest(BaseModel):
    """The patterns to learn from, and how wide and how long to learn."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    n_hidden_units: int = Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class ReconstructRequest(FitRequest):
    """The same fit, plus the damaged row to push through it."""

    probe: PatternValues


class RowDocument(BaseModel):
    """One row as the fitted machine reads it.

    ``hidden_probabilities`` holds one entry per hidden unit, the probability
    that unit turns on given the row. ``reconstruction`` holds one entry per
    visible unit, the probability each cell is lit when the row is pushed
    through the hidden layer and back. ``free_energy`` is the machine's figure
    for the row after learning and ``initial_free_energy`` the same figure
    before any, from the fit at a rate of zero the module docstring describes.
    ``reconstruction_error`` is this row's own mean squared distance from its
    reconstruction.
    """

    hidden_probabilities: list[float]
    reconstruction: list[float]
    free_energy: float
    initial_free_energy: float
    reconstruction_error: float


class FitResponse(BaseModel):
    """Every stored pattern as the machine reads it, and everything it learned.

    ``reconstruction_error`` is the mean over all the patterns, the library's
    own diagnostic. ``converged`` means the weights stopped moving before the
    epoch cap, which is all it can mean under contrastive divergence, and
    ``epochs_run`` is how many passes the walk actually took. ``weights`` has
    one row per visible unit and one column per hidden unit.
    """

    n_visible_units: int
    n_hidden_units: int
    epochs_run: int
    converged: bool
    reconstruction_error: float
    patterns: list[RowDocument]
    weights: list[list[float]]
    visible_bias: list[float]
    hidden_bias: list[float]


class ReconstructResponse(BaseModel):
    """The probe as the machine reads it, beside the stored patterns' figures.

    ``stored_free_energies`` are the same numbers the fit endpoint reports for
    the same request, one per stored pattern in the order they were given, so
    the page can draw the probe's free energy against them.
    ``reconstruction_error`` is the stored patterns' mean, not the probe's,
    which lives on the probe's own document.
    """

    probe: RowDocument
    stored_free_energies: list[float]
    epochs_run: int
    reconstruction_error: float


@router.post("/concepts/rbm/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn the patterns and report how the machine now reads each of them."""
    n_units = _checked_width(request.patterns)
    units = _unit_features(request.patterns, n_units)

    model = _fitted(units, request.n_hidden_units, request.max_epochs)
    at_start = _before_learning(units, request.n_hidden_units)

    return FitResponse(
        n_visible_units=model.n_visible_units,
        n_hidden_units=model.n_hidden_units,
        epochs_run=model.epochs_run,
        converged=model.converged,
        reconstruction_error=float(model.reconstruction_error(units)),
        patterns=[
            _row_document(model, at_start, pattern) for pattern in request.patterns
        ],
        weights=[[float(value) for value in row] for row in model.weights],
        visible_bias=[float(value) for value in model.visible_bias],
        hidden_bias=[float(value) for value in model.hidden_bias],
    )


@router.post("/concepts/rbm/reconstruct", response_model=ReconstructResponse)
def reconstruct(request: ReconstructRequest) -> ReconstructResponse:
    """Learn the patterns again, then push the probe through and back."""
    n_units = _checked_width(request.patterns)
    if len(request.probe) != n_units:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the probe holds {len(request.probe)} values where the stored "
                f"patterns hold {n_units}, and a probe must cover the same units"
            ),
        )
    units = _unit_features(request.patterns, n_units)

    model = _fitted(units, request.n_hidden_units, request.max_epochs)
    at_start = _before_learning(units, request.n_hidden_units)

    return ReconstructResponse(
        probe=_row_document(model, at_start, request.probe),
        stored_free_energies=[
            _free_energy_of(model, _row_features(pattern))
            for pattern in request.patterns
        ],
        epochs_run=model.epochs_run,
        reconstruction_error=float(model.reconstruction_error(units)),
    )


def _checked_width(patterns: Sequence[Sequence[int]]) -> int:
    """How many units the patterns cover, refusing a ragged list in words."""
    n_units = len(patterns[0])
    for position, pattern in enumerate(patterns):
        if len(pattern) != n_units:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"pattern {position + 1} holds {len(pattern)} values where the "
                    f"first holds {n_units}, and every stored pattern must cover "
                    "the same units"
                ),
            )

    return n_units


def _unit_features(patterns: Sequence[Sequence[int]], n_units: int) -> list[Feature]:
    """The patterns turned round, one feature per visible unit."""
    return [
        Feature(
            _unit_name(position),
            [pattern[position] for pattern in patterns],
        )
        for position in range(n_units)
    ]


def _unit_name(position: int) -> str:
    """What the visible unit at ``position`` is called, counting from zero."""
    return f"unit_{position + 1}"


def _fitted(
    units: Sequence[Feature], n_hidden_units: int, max_epochs: int
) -> RestrictedBoltzmannMachine:
    """The machine the page describes, learned under the fixed seed and rate."""
    return RestrictedBoltzmannMachine(
        n_hidden_units=n_hidden_units,
        learning_rate=ConstantSchedule(value=LEARNING_RATE),
        max_epochs=max_epochs,
        random_seed=FIXED_SEED,
    ).fit(units)


def _before_learning(
    units: Sequence[Feature], n_hidden_units: int
) -> RestrictedBoltzmannMachine:
    """The same machine at its starting point, before any epoch has moved it.

    A single epoch at a rate of zero, which leaves the seeded initial weights
    and the zero biases exactly as the library drew them.
    """
    return RestrictedBoltzmannMachine(
        n_hidden_units=n_hidden_units,
        learning_rate=ConstantSchedule(value=0.0),
        max_epochs=1,
        random_seed=FIXED_SEED,
    ).fit(units)


def _row_document(
    model: RestrictedBoltzmannMachine,
    at_start: RestrictedBoltzmannMachine,
    row: Sequence[int],
) -> RowDocument:
    """One row read off the fitted machine, and off the machine before learning."""
    features = _row_features(row)

    return RowDocument(
        hidden_probabilities=[
            float(value) for value in model.hidden_probabilities(features).values[0]
        ],
        reconstruction=[
            float(feature.values[0]) for feature in model.reconstruct(features)
        ],
        free_energy=_free_energy_of(model, features),
        initial_free_energy=_free_energy_of(at_start, features),
        reconstruction_error=float(model.reconstruction_error(features)),
    )


def _row_features(row: Sequence[int]) -> list[Feature]:
    """A single row as one feature per visible unit, each holding one value."""
    return [
        Feature(_unit_name(position), [value]) for position, value in enumerate(row)
    ]


def _free_energy_of(model: RestrictedBoltzmannMachine, row: Sequence[Feature]) -> float:
    """One row's free energy, read the same way wherever a row is reported.

    Always read one row at a time, because a block of rows reaches the same
    figure by a different order of floating-point sums and the two endpoints
    would otherwise disagree in the last bit about the same pattern.
    """
    return float(np.asarray(model.free_energy(row))[0])


# --- One learning step, replayed rather than reported ------------------------

MAX_REPLAYED_STEPS = 12
"""The most updates the step endpoint replays in one request.

The page traces three on four cells and shows the first of them in full, so
twelve is already more than the widget draws and the replay is arithmetic on
blocks small enough that the whole request answers instantly.
"""


class StepRequest(BaseModel):
    """The patterns to learn from, and how many updates to replay."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    n_hidden_units: int = Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)
    n_steps: int = Field(ge=1, le=MAX_REPLAYED_STEPS)

    model_config = {"extra": "forbid"}


class StepDocument(BaseModel):
    """One contrastive divergence update, with both ends of the chain.

    ``data_correlations`` is how often each cell and each hidden unit are on
    together while the cells are held at the patterns, and
    ``chain_correlations`` is the same count after the machine has been let go
    for one alternation. Both have one row per cell and one column per hidden
    unit. ``weight_change`` is the rate times their difference, and
    ``weights_after`` is what the machine holds once it has been applied.
    ``chain_visible`` is the block of cells the chain actually drew, one row
    per pattern, which is what the second count was measured on.
    """

    step_number: int
    learning_rate: float
    data_correlations: list[list[float]]
    chain_correlations: list[list[float]]
    data_visible_means: list[float]
    chain_visible_means: list[float]
    chain_visible: list[list[float]]
    weight_change: list[list[float]]
    weights_after: list[list[float]]
    visible_bias_after: list[float]
    hidden_bias_after: list[float]
    largest_movement: float


class StepResponse(BaseModel):
    """The weights the seed drew, and every update applied to them in turn."""

    n_visible_units: int
    n_hidden_units: int
    learning_rate: float
    starting_weights: list[list[float]]
    steps: list[StepDocument]


@router.post("/concepts/rbm/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Replay the learning rule one update at a time, showing both counts."""
    n_units = _checked_width(request.patterns)
    rows = np.asarray(request.patterns, dtype=np.float64)

    generator = np.random.default_rng(FIXED_SEED)
    parameters = _starting_parameters(n_units, request.n_hidden_units, generator)
    starting_weights = _matrix(parameters.weights)

    replayed: list[StepDocument] = []
    for step_number in range(1, request.n_steps + 1):
        positive = GibbsState(rows, parameters.hidden_given(rows))
        negative = _chain_end(parameters, positive.hidden, generator)
        update = ContrastiveDivergenceUpdate(
            weight_change=LEARNING_RATE
            * (positive.correlations - negative.correlations),
            visible_bias_change=LEARNING_RATE
            * (positive.visible_means - negative.visible_means),
            hidden_bias_change=LEARNING_RATE
            * (positive.hidden_means - negative.hidden_means),
        )
        parameters = parameters.shifted_by(update)

        replayed.append(
            StepDocument(
                step_number=step_number,
                learning_rate=LEARNING_RATE,
                data_correlations=_matrix(positive.correlations),
                chain_correlations=_matrix(negative.correlations),
                data_visible_means=_row(positive.visible_means),
                chain_visible_means=_row(negative.visible_means),
                chain_visible=_matrix(negative.visible),
                weight_change=_matrix(update.weight_change),
                weights_after=_matrix(parameters.weights),
                visible_bias_after=_row(parameters.visible_bias),
                hidden_bias_after=_row(parameters.hidden_bias),
                largest_movement=float(update.largest_movement),
            )
        )

    return StepResponse(
        n_visible_units=n_units,
        n_hidden_units=request.n_hidden_units,
        learning_rate=LEARNING_RATE,
        starting_weights=starting_weights,
        steps=replayed,
    )


def _starting_parameters(
    n_visible_units: int, n_hidden_units: int, generator: np.random.Generator
) -> BoltzmannParameters:
    """The weights a fit starts from, drawn from the seed the same way it does.

    Small normal noise and zero biases. The spread is the library's own
    constant rather than a number copied here, so the replay cannot drift away
    from the fit it is claiming to reproduce.
    """
    return BoltzmannParameters(
        generator.normal(0.0, INITIAL_WEIGHT_SPREAD, (n_visible_units, n_hidden_units)),
        np.zeros(n_visible_units),
        np.zeros(n_hidden_units),
    )


def _chain_end(
    parameters: BoltzmannParameters,
    hidden_probabilities: np.ndarray,
    generator: np.random.Generator,
) -> GibbsState:
    """One alternation away from the data, sampling both layers on the way.

    Drawing rather than passing the probabilities forward is what makes this a
    chain over states instead of a deterministic map, and the block that comes
    back is paired with the probabilities those drawn cells imply.
    """
    hidden_states = _drawn(hidden_probabilities, generator)
    visible_states = _drawn(parameters.visible_given(hidden_states), generator)

    return GibbsState(visible_states, parameters.hidden_given(visible_states))


def _drawn(probabilities: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """One coin toss per entry, at the probability that entry holds."""
    return (generator.random(probabilities.shape) < probabilities).astype(np.float64)


def _matrix(values: np.ndarray) -> list[list[float]]:
    """A two-dimensional block as plain floats, row by row."""
    return [[float(value) for value in row] for row in values]


def _row(values: np.ndarray) -> list[float]:
    """A one-dimensional block as plain floats."""
    return [float(value) for value in values]


# --- Scoring rows the browser built ------------------------------------------

MAX_SCORED_STATES = 12
"""The most rows one request may score against a single fit."""


class StatesRequest(FitRequest):
    """The same fit, plus the rows to put to it."""

    states: list[PatternValues] = Field(min_length=1, max_length=MAX_SCORED_STATES)


class StatesResponse(BaseModel):
    """Every stored pattern and every supplied row, read off one fit.

    ``stored`` and ``states`` hold the same document, so a row that happens to
    be one of the stored patterns reports exactly what that pattern reports.
    """

    n_visible_units: int
    n_hidden_units: int
    epochs_run: int
    reconstruction_error: float
    stored: list[RowDocument]
    states: list[RowDocument]


@router.post("/concepts/rbm/states", response_model=StatesResponse)
def states(request: StatesRequest) -> StatesResponse:
    """Score every supplied row against one fit on the stored patterns."""
    n_units = _checked_width(request.patterns)
    _check_states_width(request.states, n_units)
    units = _unit_features(request.patterns, n_units)

    model = _fitted(units, request.n_hidden_units, request.max_epochs)
    at_start = _before_learning(units, request.n_hidden_units)

    return StatesResponse(
        n_visible_units=model.n_visible_units,
        n_hidden_units=model.n_hidden_units,
        epochs_run=model.epochs_run,
        reconstruction_error=float(model.reconstruction_error(units)),
        stored=[
            _row_document(model, at_start, pattern) for pattern in request.patterns
        ],
        states=[_row_document(model, at_start, state) for state in request.states],
    )


def _check_states_width(asked: Sequence[Sequence[int]], n_units: int) -> None:
    """Refuse, in words, a supplied row that does not cover the same cells."""
    for position, state in enumerate(asked):
        if len(state) != n_units:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"row {position + 1} holds {len(state)} values where the stored "
                    f"patterns hold {n_units}, and a row must cover the same units"
                ),
            )


# --- Watching a probe come back as the walk lengthens ------------------------

MAX_SWEEP_POINTS = 12
"""The most walk lengths or hidden widths one sweep may ask for."""


class EpochSweepRequest(BaseModel):
    """The patterns, a damaged probe, the shape it came from, and the lengths."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    n_hidden_units: int = Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)
    probe: PatternValues
    target: PatternValues
    epoch_counts: list[Annotated[int, Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)]] = Field(
        min_length=1, max_length=MAX_SWEEP_POINTS
    )

    model_config = {"extra": "forbid"}


class EpochSweepPoint(BaseModel):
    """One walk length, and what the probe comes back as after it.

    ``distance_to_target`` is the mean squared gap between what came back and
    the undamaged shape the probe was made from, which is the number that says
    whether the damage was repaired rather than merely smoothed.
    ``cells_matching_target`` rounds the rebuilt cells and counts how many land
    on the undamaged shape.
    """

    max_epochs: int
    epochs_run: int
    stored_reconstruction_error: float
    probe_reconstruction: list[float]
    probe_hidden_probabilities: list[float]
    probe_free_energy: float
    stored_free_energies: list[float]
    distance_to_target: float
    cells_matching_target: int


class EpochSweepResponse(BaseModel):
    """The probe rebuilt at every walk length asked for."""

    n_visible_units: int
    n_hidden_units: int
    target: list[int]
    points: list[EpochSweepPoint]


@router.post("/concepts/rbm/epoch-sweep", response_model=EpochSweepResponse)
def epoch_sweep(request: EpochSweepRequest) -> EpochSweepResponse:
    """Refit at each walk length and report what the probe comes back as."""
    n_units = _checked_width(request.patterns)
    _check_states_width([request.probe, request.target], n_units)
    units = _unit_features(request.patterns, n_units)
    probe = _row_features(request.probe)

    points: list[EpochSweepPoint] = []
    for max_epochs in request.epoch_counts:
        model = _fitted(units, request.n_hidden_units, max_epochs)
        rebuilt = [float(feature.values[0]) for feature in model.reconstruct(probe)]
        points.append(
            EpochSweepPoint(
                max_epochs=max_epochs,
                epochs_run=model.epochs_run,
                stored_reconstruction_error=float(model.reconstruction_error(units)),
                probe_reconstruction=rebuilt,
                probe_hidden_probabilities=[
                    float(value)
                    for value in model.hidden_probabilities(probe).values[0]
                ],
                probe_free_energy=_free_energy_of(model, probe),
                stored_free_energies=[
                    _free_energy_of(model, _row_features(pattern))
                    for pattern in request.patterns
                ],
                distance_to_target=_mean_squared_gap(rebuilt, request.target),
                cells_matching_target=_cells_matching(rebuilt, request.target),
            )
        )

    return EpochSweepResponse(
        n_visible_units=n_units,
        n_hidden_units=request.n_hidden_units,
        target=list(request.target),
        points=points,
    )


def _mean_squared_gap(rebuilt: Sequence[float], target: Sequence[int]) -> float:
    """Mean squared difference between a rebuilt row and the row it should be."""
    return float(
        np.mean(
            (
                np.asarray(rebuilt, dtype=np.float64)
                - np.asarray(target, dtype=np.float64)
            )
            ** 2
        )
    )


def _cells_matching(rebuilt: Sequence[float], target: Sequence[int]) -> int:
    """How many rebuilt cells round to the row they should have been."""
    return int(
        np.count_nonzero(
            np.round(np.asarray(rebuilt, dtype=np.float64))
            == np.asarray(target, dtype=np.float64)
        )
    )


# --- How wide the hidden layer needs to be -----------------------------------


class HiddenSweepRequest(BaseModel):
    """The patterns, the walk length, and the hidden widths to try."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    hidden_counts: list[
        Annotated[int, Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)]
    ] = Field(min_length=1, max_length=MAX_SWEEP_POINTS)

    model_config = {"extra": "forbid"}


class HiddenSweepPoint(BaseModel):
    """One hidden width, and how well the patterns come back under it.

    ``codes`` rounds each pattern's hidden probabilities to nought or one, and
    ``n_distinct_codes`` counts how many of those are different from each
    other, which is how many patterns the width can keep apart at all.
    """

    n_hidden_units: int
    reconstruction_error: float
    worst_pattern_error: float
    codes: list[list[int]]
    n_distinct_codes: int
    patterns_rebuilt_exactly: int


class HiddenSweepResponse(BaseModel):
    """Every hidden width asked for, refitted from the same seed."""

    n_visible_units: int
    n_patterns: int
    max_epochs: int
    points: list[HiddenSweepPoint]


@router.post("/concepts/rbm/hidden-sweep", response_model=HiddenSweepResponse)
def hidden_sweep(request: HiddenSweepRequest) -> HiddenSweepResponse:
    """Refit at each hidden width and report how well the patterns come back."""
    n_units = _checked_width(request.patterns)
    units = _unit_features(request.patterns, n_units)

    points: list[HiddenSweepPoint] = []
    for n_hidden_units in request.hidden_counts:
        model = _fitted(units, n_hidden_units, request.max_epochs)
        per_pattern = [_row_features(pattern) for pattern in request.patterns]
        errors = [
            float(model.reconstruction_error(features)) for features in per_pattern
        ]
        codes = [
            [
                round(float(value))
                for value in model.hidden_probabilities(features).values[0]
            ]
            for features in per_pattern
        ]
        rebuilt_exactly = sum(
            1
            for features, pattern in zip(per_pattern, request.patterns, strict=True)
            if [
                round(float(feature.values[0]))
                for feature in model.reconstruct(features)
            ]
            == list(pattern)
        )

        points.append(
            HiddenSweepPoint(
                n_hidden_units=n_hidden_units,
                reconstruction_error=float(model.reconstruction_error(units)),
                worst_pattern_error=max(errors),
                codes=codes,
                n_distinct_codes=len({tuple(code) for code in codes}),
                patterns_rebuilt_exactly=rebuilt_exactly,
            )
        )

    return HiddenSweepResponse(
        n_visible_units=n_units,
        n_patterns=len(request.patterns),
        max_epochs=request.max_epochs,
        points=points,
    )


# --- What a seed fixes, and what it does not ---------------------------------

MAX_SEEDS = 6
"""The most seeds one request may fit under."""

MAX_SEED_VALUE = 10_000
"""The largest seed a request may name, matching the memory page's bound."""


class SeedRequest(BaseModel):
    """The patterns, the width and the walk, and the seeds to fit under."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    n_hidden_units: int = Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    seeds: list[Annotated[int, Field(ge=0, le=MAX_SEED_VALUE)]] = Field(
        min_length=1, max_length=MAX_SEEDS
    )

    model_config = {"extra": "forbid"}


class SeedRun(BaseModel):
    """One seed's fit, in the few numbers that separate one from another."""

    seed: int
    reconstruction_error: float
    codes: list[list[int]]
    free_energies: list[float]
    largest_weight: float
    codes_all_distinct: bool


class SeedResponse(BaseModel):
    """Every seed's fit, and the two gaps that say what the seed decided.

    ``repeat_weight_gap`` is the largest difference between any two matching
    weights of two fits run under the first seed twice over, which is what a
    seed being a seed means. ``across_seed_weight_gap`` is the same
    measurement between the first two seeds, and ``error_spread`` is the range
    of the reconstruction errors across every seed asked for.
    """

    runs: list[SeedRun]
    repeat_weight_gap: float
    repeat_error_gap: float
    across_seed_weight_gap: float | None
    error_spread: float


@router.post("/concepts/rbm/seeds", response_model=SeedResponse)
def seeds(request: SeedRequest) -> SeedResponse:
    """Fit under each seed, and fit the first one twice over, and compare."""
    n_units = _checked_width(request.patterns)
    units = _unit_features(request.patterns, n_units)

    models = [
        _fitted_with_seed(units, request.n_hidden_units, request.max_epochs, seed)
        for seed in request.seeds
    ]
    repeated = _fitted_with_seed(
        units, request.n_hidden_units, request.max_epochs, request.seeds[0]
    )

    runs = [
        _seed_run(model, seed, units, request.patterns)
        for seed, model in zip(request.seeds, models, strict=True)
    ]
    errors = [run.reconstruction_error for run in runs]

    return SeedResponse(
        runs=runs,
        repeat_weight_gap=_largest_weight_gap(models[0], repeated),
        repeat_error_gap=abs(float(repeated.reconstruction_error(units)) - errors[0]),
        across_seed_weight_gap=(
            _largest_weight_gap(models[0], models[1]) if len(models) > 1 else None
        ),
        error_spread=max(errors) - min(errors),
    )


def _seed_run(
    model: RestrictedBoltzmannMachine,
    seed: int,
    units: Sequence[Feature],
    patterns: Sequence[Sequence[int]],
) -> SeedRun:
    """One fit reported in the few numbers that separate one seed from another."""
    codes = [
        [
            round(float(value))
            for value in model.hidden_probabilities(_row_features(pattern)).values[0]
        ]
        for pattern in patterns
    ]

    return SeedRun(
        seed=seed,
        reconstruction_error=float(model.reconstruction_error(units)),
        codes=codes,
        free_energies=[
            _free_energy_of(model, _row_features(pattern)) for pattern in patterns
        ],
        largest_weight=float(np.max(np.abs(model.weights))),
        codes_all_distinct=len({tuple(code) for code in codes}) == len(codes),
    )


def _fitted_with_seed(
    units: Sequence[Feature], n_hidden_units: int, max_epochs: int, random_seed: int
) -> RestrictedBoltzmannMachine:
    """The same machine as the page's, under a seed the caller chose."""
    return RestrictedBoltzmannMachine(
        n_hidden_units=n_hidden_units,
        learning_rate=ConstantSchedule(value=LEARNING_RATE),
        max_epochs=max_epochs,
        random_seed=random_seed,
    ).fit(units)


def _largest_weight_gap(
    first: RestrictedBoltzmannMachine, second: RestrictedBoltzmannMachine
) -> float:
    """The largest absolute difference between two fits' matching weights."""
    return float(np.max(np.abs(first.weights - second.weights)))


# --- Beside the memory that stores exactly -----------------------------------


class AgainstMemoryRequest(BaseModel):
    """The patterns both models are given, and the rows to put to both."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    n_hidden_units: int = Field(ge=MIN_HIDDEN_UNITS, le=MAX_HIDDEN_UNITS)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    states: list[PatternValues] = Field(min_length=1, max_length=MAX_SCORED_STATES)

    model_config = {"extra": "forbid"}


class BesideMemoryReport(BaseModel):
    """One row, scored and answered by both models.

    ``free_energy`` and ``reversed_free_energy`` are the machine's score for
    the row and for the row with every cell reversed; ``memory_energy`` and
    ``reversed_memory_energy`` are the memory's. ``rebuilt`` is what the
    machine gives back, one probability per cell, and ``settled`` is where the
    memory comes to rest from the same row, one value per cell.
    ``settled_into`` names which stored pattern that is, or none of them.
    """

    state: list[int]
    free_energy: float
    reversed_free_energy: float
    memory_energy: float
    reversed_memory_energy: float
    rebuilt: list[float]
    settled: list[int]
    settled_into: int | None
    settled_inverted: bool
    rebuilt_rounds_to: int | None


class AgainstMemoryResponse(BaseModel):
    """Both models on the same patterns, and the two reversal gaps.

    ``largest_reversal_gap`` is the largest difference across the supplied
    rows between a row's score and its reversal's under the machine, and
    ``largest_memory_reversal_gap`` is the same measurement under the memory's
    own energy, which the memory's arithmetic makes zero.
    """

    n_visible_units: int
    stored_free_energies: list[float]
    stored_memory_energies: list[float]
    reports: list[BesideMemoryReport]
    largest_reversal_gap: float
    largest_memory_reversal_gap: float


@router.post("/concepts/rbm/against-memory", response_model=AgainstMemoryResponse)
def against_memory(request: AgainstMemoryRequest) -> AgainstMemoryResponse:
    """Put the same patterns and the same rows to both models and compare."""
    n_units = _checked_width(request.patterns)
    _check_states_width(request.states, n_units)

    units = _unit_features(request.patterns, n_units)
    machine = _fitted(units, request.n_hidden_units, request.max_epochs)

    memory = HopfieldNetwork(random_seed=FIXED_SEED).fit(
        _unit_features([_as_bipolar(pattern) for pattern in request.patterns], n_units)
    )
    stored = [BipolarPattern(_as_bipolar(pattern)) for pattern in request.patterns]

    reports = [
        _beside_memory(machine, memory, stored, state) for state in request.states
    ]

    return AgainstMemoryResponse(
        n_visible_units=n_units,
        stored_free_energies=[
            _free_energy_of(machine, _row_features(pattern))
            for pattern in request.patterns
        ],
        stored_memory_energies=[
            float(memory.weights.energy_of(pattern)) for pattern in stored
        ],
        reports=reports,
        largest_reversal_gap=max(
            abs(report.free_energy - report.reversed_free_energy) for report in reports
        ),
        largest_memory_reversal_gap=max(
            abs(report.memory_energy - report.reversed_memory_energy)
            for report in reports
        ),
    )


def _beside_memory(
    machine: RestrictedBoltzmannMachine,
    memory: HopfieldNetwork,
    stored: Sequence[BipolarPattern],
    state: Sequence[int],
) -> BesideMemoryReport:
    """One row put to both models, with its reversal scored under each."""
    reversed_state = [1 - value for value in state]
    signs = BipolarPattern(_as_bipolar(state))
    settled = memory.recall(signs)
    rebuilt = [
        float(feature.values[0])
        for feature in machine.reconstruct(_row_features(state))
    ]
    resting_place = _named(settled, stored)

    return BesideMemoryReport(
        state=list(state),
        free_energy=_free_energy_of(machine, _row_features(state)),
        reversed_free_energy=_free_energy_of(machine, _row_features(reversed_state)),
        memory_energy=float(memory.weights.energy_of(signs)),
        reversed_memory_energy=float(memory.weights.energy_of(signs.flipped())),
        rebuilt=rebuilt,
        settled=[1 if value > 0 else 0 for value in settled.values],
        settled_into=resting_place,
        settled_inverted=resting_place is not None and settled != stored[resting_place],
        rebuilt_rounds_to=_rounds_to(rebuilt, stored),
    )


def _as_bipolar(pattern: Sequence[int]) -> list[int]:
    """A row of noughts and ones as the memory's downs and ups."""
    return [1 if value == 1 else -1 for value in pattern]


def _named(settled: BipolarPattern, stored: Sequence[BipolarPattern]) -> int | None:
    """Which stored pattern the memory came to rest in, or none of them."""
    for index, pattern in enumerate(stored):
        if settled == pattern or settled == pattern.flipped():
            return index
    return None


def _rounds_to(
    rebuilt: Sequence[float], stored: Sequence[BipolarPattern]
) -> int | None:
    """Which stored pattern the rebuilt cells round to, or none of them."""
    rounded = [1.0 if value >= 0.5 else -1.0 for value in rebuilt]
    for index, pattern in enumerate(stored):
        if rounded == [float(value) for value in pattern.values]:
            return index
    return None


# --- The contracts, every edge run rather than asserted ----------------------


class ContractProbe(BaseModel):
    """One edge put to the machine, and what came back.

    ``detail`` is a sentence in the words the website uses. Where the edge was
    refused it is the refusal's own wording with the names of columns, classes
    and settings substituted out, and where it was accepted it says what came
    back instead, so a reader meets the behaviour and never an identifier.
    """

    edge: str
    outcome: Literal["accepted", "refused"]
    detail: str


class ContractsResponse(BaseModel):
    """Every edge the last part of the page lists, probed."""

    probes: list[ContractProbe]


CONTRACT_PATTERNS = ([1, 1, 0, 0], [1, 1, 0, 1])
"""The four-cell pair the page traces its learning step on."""

CONTRACT_HIDDEN_UNITS = 2
"""The width the contract probes fit at, matching the four-cell trace."""

CONTRACT_EPOCHS = 500
"""The walk the page's own fits take, so an accepted edge reports the number a
finished machine gives rather than a half-learned one. Four cells over two
rows is small enough that the whole table still runs on the first request and
is remembered afterwards."""

NOT_FITTED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z]* must be fit before this is available"
)
"""A refusal that opens on the name of whatever was asked too early."""

FITTED_FEATURES_PATTERN = re.compile(
    r"expected exactly the fitted features \[(?P<expected>[^\]]*)\]; "
    r"got \[(?P<supplied>[^\]]*)\]"
)
"""A refusal that lists two sets of column names the page shows as counts."""

HIDDEN_UNITS_PATTERN = re.compile(
    r"expected exactly this model's hidden units \[(?P<expected>[^\]]*)\]; "
    r"got \[(?P<supplied>[^\]]*)\]"
)
"""The same refusal for the layer on the other side."""

RAGGED_PATTERN = re.compile(r"(\S+) has (\d+) rows, (\S+) has (\d+)")
"""A refusal that names two columns and their differing lengths."""

DUPLICATE_PATTERN = re.compile(r"feature names must be unique, got duplicates: (\S+)")
"""A refusal that names the column given twice."""

UNIT_PATTERN = re.compile(r"\bunit_(\d+)\b")
"""How a visible cell is named where the machine sees it."""

PLAIN_WORDS = (
    ("feature_values must contain only finite values", "every value must be finite"),
    ("at least one feature is required", "at least one cell is required"),
)
"""Identifiers a reader has no reason to meet, and what to say instead."""

PLAIN_FIELD_NAMES = {
    "n_hidden_units": "the hidden unit count",
    "max_epochs": "the number of learning passes",
    "n_gibbs_steps": "the number of alternations",
    "tolerance": "the settling tolerance",
    "value": "the learning rate",
}
"""What each refused setting is called on the page."""

PLAIN_MESSAGES = (("Input should be", "must be"),)
"""The stock wording, turned into a sentence about the setting."""

UNKNOWN_SETTING = "extra_forbidden"
"""The refusal a setting the machine does not have earns, which names that
setting and so cannot be reported with the name in it."""


def _in_plain_words(refusal: Exception) -> str:
    """One refusal's sentence, with the names of code taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the parts that name a column, a class or a field and leaves
    the rest alone. A test asserts that nothing identifier-shaped survives, so
    a message that grows one later fails there rather than reaching the page.
    """
    if isinstance(refusal, ValidationError):
        return _from_validation(refusal)

    sentence = str(refusal)
    sentence = NOT_FITTED_PATTERN.sub(
        "this machine must be learned from patterns before that can be read", sentence
    )
    sentence = FITTED_FEATURES_PATTERN.sub(_as_counts, sentence)
    sentence = HIDDEN_UNITS_PATTERN.sub(_as_hidden_counts, sentence)
    sentence = RAGGED_PATTERN.sub(_as_lengths, sentence)
    sentence = DUPLICATE_PATTERN.sub(r"two cells cannot share the name \1", sentence)
    sentence = UNIT_PATTERN.sub(r"cell \1", sentence)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)

    return sentence


def _as_lengths(match: re.Match[str]) -> str:
    """Two columns of differing lengths, said without the plural going wrong."""
    return (
        f"{match[1]} was given {_values(int(match[2]))} where "
        f"{match[3]} was given {_values(int(match[4]))}"
    )


def _values(count: int) -> str:
    """A count of values, singular where the count is one."""
    return f"{count} value" if count == 1 else f"{count} values"


def _as_counts(match: re.Match[str]) -> str:
    """Two lists of column names as the counts and mismatches a reader wants."""
    expected = _names_in(match["expected"])
    supplied = _names_in(match["supplied"])
    unknown = len(supplied - expected)
    if unknown:
        naming = "name cells that were" if unknown > 1 else "names a cell that was"
        return (
            f"expected exactly the {len(expected)} cells it learned from, and "
            f"{unknown} of the {len(supplied)} columns handed in {naming} never "
            "among them"
        )

    return (
        f"expected exactly the {len(expected)} cells it learned from, and was "
        f"handed {len(supplied)}"
    )


def _as_hidden_counts(match: re.Match[str]) -> str:
    """The same for the hidden layer, which is asked for by name too."""
    return (
        f"expected exactly the {len(_names_in(match['expected']))} hidden units this "
        f"machine has, and was handed {len(_names_in(match['supplied']))} columns that "
        "are not among them"
    )


def _names_in(listed: str) -> set[str]:
    """The names a refusal listed inside its brackets."""
    return {name.strip() for name in listed.split(",") if name.strip()}


def _from_validation(refusal: ValidationError) -> str:
    """A refused setting, named the way the page names it."""
    first = refusal.errors()[0]
    if first["type"] == UNKNOWN_SETTING:
        return (
            "a setting this machine does not have was passed, and it is refused "
            "rather than quietly ignored"
        )

    field = str(first["loc"][0]) if first["loc"] else ""
    message = first["msg"]
    for stock, plain in PLAIN_MESSAGES:
        message = message.replace(stock, plain)

    return f"{PLAIN_FIELD_NAMES.get(field, 'this setting')} {message}"


def _probe(edge: str, attempt: Callable[[], str]) -> ContractProbe:
    """Run one attempt and record whether it was accepted or refused."""
    try:
        detail = attempt()
    except (MLLibError, ValidationError) as refusal:
        return ContractProbe(
            edge=edge, outcome="refused", detail=_in_plain_words(refusal)
        )

    return ContractProbe(edge=edge, outcome="accepted", detail=detail)


def _contract_units(patterns: Sequence[Sequence[float]]) -> list[Feature]:
    """The contract table's patterns, turned round for the machine."""
    return _unit_features(patterns, len(patterns[0]))


def _contract_machine(
    patterns: Sequence[Sequence[float]] = CONTRACT_PATTERNS,
    n_hidden_units: int = CONTRACT_HIDDEN_UNITS,
    max_epochs: int = CONTRACT_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> RestrictedBoltzmannMachine:
    """One fitted machine for a probe to ask its question of."""
    return RestrictedBoltzmannMachine(
        n_hidden_units=n_hidden_units,
        learning_rate=ConstantSchedule(value=learning_rate),
        max_epochs=max_epochs,
        random_seed=FIXED_SEED,
    ).fit(_contract_units(patterns))


def _a_value_above_one() -> str:
    _contract_machine([[2, 1, 0, 0], [1, 1, 0, 1]])
    return ""


def _a_value_below_zero() -> str:
    _contract_machine([[-1, 1, 0, 0], [1, 1, 0, 1]])
    return ""


def _a_value_between() -> str:
    patterns = [[0.5, 1, 0, 0], [1, 1, 0, 1]]
    machine = _contract_machine(patterns)
    rebuilt = machine.reconstruct(_row_features([0.5, 1, 0, 0]))
    return (
        "accepted and read as a mean rather than a state; the half comes back at "
        f"{float(rebuilt[0].values[0]):.2f}"
    )


def _a_non_finite_value() -> str:
    _contract_machine([[float("nan"), 1, 0, 0], [1, 1, 0, 1]])
    return ""


def _one_pattern() -> str:
    patterns = [[1, 1, 0, 0]]
    machine = _contract_machine(patterns)
    return (
        "accepted, since a distribution over one row is still a distribution; the "
        f"row comes back at a mean squared gap of "
        f"{float(machine.reconstruction_error(_contract_units(patterns))):.4f} and the "
        f"walk stops after {machine.epochs_run} passes of {CONTRACT_EPOCHS} calling "
        "itself settled"
    )


def _patterns_of_differing_widths() -> str:
    RestrictedBoltzmannMachine(
        n_hidden_units=CONTRACT_HIDDEN_UNITS, max_epochs=CONTRACT_EPOCHS
    ).fit([Feature("unit_1", [1, 0]), Feature("unit_2", [1])])
    return ""


def _no_patterns_at_all() -> str:
    RestrictedBoltzmannMachine(
        n_hidden_units=CONTRACT_HIDDEN_UNITS, max_epochs=CONTRACT_EPOCHS
    ).fit([])
    return ""


def _more_hidden_units_than_cells() -> str:
    machine = _contract_machine(n_hidden_units=8)
    return (
        "accepted; eight hidden units over four cells is a wider description than the "
        "thing described, and the patterns come back at a mean squared gap of "
        f"{float(machine.reconstruction_error(_contract_units(CONTRACT_PATTERNS))):.4f}"
    )


def _zero_hidden_units() -> str:
    RestrictedBoltzmannMachine(n_hidden_units=0)
    return ""


def _zero_passes() -> str:
    RestrictedBoltzmannMachine(n_hidden_units=CONTRACT_HIDDEN_UNITS, max_epochs=0)
    return ""


def _a_learning_rate_of_zero() -> str:
    machine = _contract_machine(learning_rate=0.0)
    return (
        "accepted, and it settles immediately; nothing is added to the weights the "
        f"seed drew, so the walk stops after {machine.epochs_run} pass of "
        f"{CONTRACT_EPOCHS} and calls itself settled"
    )


def _a_negative_learning_rate() -> str:
    ConstantSchedule(value=-0.1)
    return ""


def _reading_a_weight_before_learning() -> str:
    unfitted = RestrictedBoltzmannMachine(n_hidden_units=CONTRACT_HIDDEN_UNITS)
    return str(unfitted.weights)


def _rebuilding_before_learning() -> str:
    RestrictedBoltzmannMachine(n_hidden_units=CONTRACT_HIDDEN_UNITS).reconstruct(
        _row_features([1, 1, 0, 0])
    )
    return ""


def _a_probe_of_the_wrong_width() -> str:
    _contract_machine().reconstruct(_row_features([1, 1, 0]))
    return ""


def _a_probe_with_an_unknown_cell() -> str:
    machine = _contract_machine()
    machine.reconstruct(
        [
            Feature("cell_1", [1]),
            Feature("unit_2", [1]),
            Feature("unit_3", [0]),
            Feature("unit_4", [0]),
        ]
    )
    return ""


def _a_probe_with_the_cells_reordered() -> str:
    machine = _contract_machine()
    straight = machine.reconstruct(_row_features([1, 1, 0, 1]))
    shuffled = machine.reconstruct(list(reversed(_row_features([1, 1, 0, 1]))))
    gap = max(
        abs(float(first.values[0]) - float(second.values[0]))
        for first, second in zip(straight, shuffled, strict=True)
    )
    return (
        "accepted; the cells are matched by name rather than by position, and the two "
        f"answers differ by {gap:.1f}"
    )


def _hidden_columns_where_cells_were_wanted() -> str:
    _contract_machine().visible_probabilities(
        [Feature("unit_1", [1]), Feature("unit_2", [0])]
    )
    return ""


def _a_cell_that_is_never_lit() -> str:
    patterns = [[1, 1, 0, 0], [1, 0, 0, 0]]
    machine = _contract_machine(patterns)
    rebuilt = machine.reconstruct(_row_features([1, 1, 0, 0]))
    return (
        "accepted, where a column holding one value throughout stops many fits; here "
        f"it is ordinary data, and after {machine.epochs_run} passes the cell comes "
        f"back at {float(rebuilt[2].values[0]):.3f}"
    )


def _an_unknown_setting() -> str:
    RestrictedBoltzmannMachine(hidden_units=3)  # type: ignore[call-arg]
    return ""


CONTRACT_PROBES: tuple[tuple[str, Callable[[], str]], ...] = (
    ("a cell holding 2", _a_value_above_one),
    ("a cell holding −1", _a_value_below_zero),
    ("a cell holding 0.5", _a_value_between),
    ("a cell holding a value that is not a number", _a_non_finite_value),
    ("one pattern only", _one_pattern),
    ("patterns of differing widths", _patterns_of_differing_widths),
    ("no patterns at all", _no_patterns_at_all),
    ("more hidden units than cells", _more_hidden_units_than_cells),
    ("no hidden units", _zero_hidden_units),
    ("no learning passes", _zero_passes),
    ("a learning rate of zero", _a_learning_rate_of_zero),
    ("a negative learning rate", _a_negative_learning_rate),
    ("a weight read before learning", _reading_a_weight_before_learning),
    ("a rebuild asked for before learning", _rebuilding_before_learning),
    ("a probe of the wrong width", _a_probe_of_the_wrong_width),
    ("a probe naming a cell the machine never saw", _a_probe_with_an_unknown_cell),
    ("a probe with its cells reordered", _a_probe_with_the_cells_reordered),
    (
        "hidden units handed in where cells were wanted",
        _hidden_columns_where_cells_were_wanted,
    ),
    ("a cell that is dark in every pattern", _a_cell_that_is_never_lit),
    ("a setting the machine does not have", _an_unknown_setting),
)
"""Every edge the page tabulates, in the order it tabulates them."""


@cache
def _contracts() -> ContractsResponse:
    """The fixed list of edges, each asked once and remembered."""
    return ContractsResponse(
        probes=[_probe(edge, attempt) for edge, attempt in CONTRACT_PROBES]
    )


@router.get("/concepts/rbm/contracts", response_model=ContractsResponse)
def contracts() -> ContractsResponse:
    """Every edge, run and reported in the words the page uses."""
    return _contracts()
