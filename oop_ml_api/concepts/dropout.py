"""Dropout, for the dropout page.

One endpoint, one row of unit values. The endpoint builds the library's
``Dropout`` layer over a row of that width, seeded as the request asks, and
pushes the row through it in both of the layer's moods. Once while predicting,
which the library answers with the row untouched, and the endpoint returns
that answer rather than asserting it, so the reader sees the pass-through
instead of being told about it. Then as many times while training as the
request asks. The first of those draws is the one the page draws as bars,
together with the scaled mask the library carried on its response, so the
widget names the silenced units from the mask and never from the outputs.
Reading a mask off the outputs is the quiet mistake the page's own section is
about, and the endpoint does not make it.

The remaining draws are for the average. Inverted dropout claims that what a
unit sends while training averages to what it sends while predicting, and a
claim about an average is checked by averaging. So the endpoint keeps every
draw, returns the running mean after each one, which is the curve the page
watches settle onto the predicting row, and counts how often each unit
survived, read from the masks, which is the number the derivation's
expectation turns into. Every draw comes from one seeded layer whose generator
advances between passes, the way it would inside a training loop, so the
sequence is one run of the library rather than a fresh layer per draw, and a
shorter run is a prefix of a longer one.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import Dropout, DropoutResponse, PassPurpose
from pydantic import BaseModel, Field

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
    mean and its predicting output.
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
    )


def _training_pass(layer: Dropout, row: np.ndarray) -> DropoutResponse:
    """One training pass, narrowed to the response type that carries a mask.

    The library answers a training pass with a ``DropoutResponse`` and a
    predicting pass with a plain one, and only the first carries ``kept``.
    The check is the one the library's own backward step makes before it will
    read a mask, and failing it here is a server fault rather than a guess.
    """
    response = layer.respond_to(row, purpose=PassPurpose.TRAINING)
    if not isinstance(response, DropoutResponse):
        raise TypeError("a training pass through dropout must carry its mask")
    return response
