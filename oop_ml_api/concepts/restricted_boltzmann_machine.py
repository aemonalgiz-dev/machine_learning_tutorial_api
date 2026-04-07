"""A restricted Boltzmann machine's fit and reconstruction, for the generative page.

Two endpoints, and both fit the library's ``RestrictedBoltzmannMachine`` on the
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

Both endpoints also report the free energy of every row before any learning,
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

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import ConstantSchedule, Feature, RestrictedBoltzmannMachine
from pydantic import BaseModel, Field

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
