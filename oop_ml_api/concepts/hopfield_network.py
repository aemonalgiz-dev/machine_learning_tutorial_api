"""A Hopfield network's recall, for the associative memory page.

One endpoint. It stores the request's patterns in the library's
``HopfieldNetwork`` under a fixed seed, hands the probe to ``recall_walk``, and
returns every sweep the walk took, the state after each, the energy before and
after it, and how many units moved. The page needs the whole walk rather than
the answer because the lesson is the settling itself, a staircase of energies
that only ever steps down, and the answer alone would hide it. Alongside the
walk go the settled state, whether it is a fixed point, which stored pattern it
equals if any, or whose negation it equals, the load, and the full weight
matrix, so the page can quote a weight the reader has just worked by hand.

Rows are patterns and units are the positions within one, so the request's
list of patterns is turned round on the way in. The library takes one
``Feature`` per unit, holding that unit's value in every stored pattern, which
is the same rows-are-samples arrangement every other model here uses and means
something different because of it. The seed fixes the order the units are
visited in during each sweep, which is the only draw the library makes, so the
same probe against the same patterns answers the same way every time and the
page can quote what the widget shows.

The values are refused by the library rather than here. A pattern holding
anything other than -1 and 1 is not a state the network can be in, and the
library says so in a sentence that names the unit at fault, so the endpoint
adds no check of its own. What it does check is that the patterns are all the
same length, because the turning round happens before the library sees them
and a ragged list has no column to read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml import BipolarPattern, Feature, HopfieldNetwork
from pydantic import BaseModel, Field

router = APIRouter(tags=["hopfield-network"])

MIN_PATTERNS = 1
"""The fewest patterns a network can store. One is a memory of one thing."""

MAX_PATTERNS = 8
"""The most patterns the widget may store.

Eight over the widget's twenty-five units is a load of 0.32, well past the
point where recall has stopped working, which is the failure the page wants
the reader to be able to reach.
"""

MIN_UNITS = 4
"""The fewest units a pattern may have, the width of the page's worked example."""

MAX_UNITS = 36
"""The most units a pattern may have, a six-by-six grid."""

MAX_PASSES = 20
"""The most sweeps one recall may take.

Under the library's asynchronous rule settling is guaranteed and a twenty-five
unit network rests in two or three sweeps, so this is a guard rather than the
stopping condition, and the response says which of the two stopped the walk.
"""

FIXED_SEED = 7
"""The unit visiting order is seeded so the pages can quote what the widget shows."""

PatternValues = Annotated[list[int], Field(min_length=MIN_UNITS, max_length=MAX_UNITS)]


class RecallRequest(BaseModel):
    """The patterns to store, and the damaged copy to hand the network."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    probe: PatternValues

    model_config = {"extra": "forbid"}


class RecallPassDocument(BaseModel):
    """One sweep of the update rule over every unit.

    ``state`` is where the sweep left the network. The state it started from is
    the previous pass's ``state``, or the probe for the first pass, so the walk
    can be replayed from the probe forward.
    """

    pass_number: int
    state: list[int]
    energy_before: float
    energy_after: float
    units_changed: int


class SettledInto(BaseModel):
    """Which stored pattern the network came to rest in, if any.

    ``pattern_index`` is the position of the stored pattern the settled state
    equals, or, with ``flipped`` set, the one whose negation it equals. Both are
    ``None`` and ``False`` when it equals nothing that was stored, which is a
    spurious resting state.
    """

    pattern_index: int | None
    flipped: bool


class RecallResponse(BaseModel):
    """The walk, where it ended, and everything the network learned.

    ``load`` is patterns stored per unit, the number that predicts failure.
    ``weights`` is the full matrix, unit by unit, with a zero diagonal.
    ``settled`` says a whole sweep went by without a unit moving, and
    ``is_fixed_point`` is the library's own check that no unit would move
    from the final state, which is the same claim asked of every visiting
    order at once.
    """

    n_units: int
    n_stored_patterns: int
    load: float
    weights: list[list[float]]
    initial_energy: float
    passes: list[RecallPassDocument]
    settled_state: list[int]
    settled: bool
    is_fixed_point: bool
    settled_into: SettledInto


@router.post("/concepts/hopfield/recall", response_model=RecallResponse)
def recall(request: RecallRequest) -> RecallResponse:
    """Store the patterns, then let the probe settle and report every sweep."""
    n_units = len(request.patterns[0])
    for position, pattern in enumerate(request.patterns):
        if len(pattern) != n_units:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"pattern {position + 1} holds {len(pattern)} values where the "
                    f"first holds {n_units}, and every stored pattern must cover "
                    "the same units"
                ),
            )

    units = [
        Feature(
            f"unit_{position + 1}",
            [pattern[position] for pattern in request.patterns],
        )
        for position in range(n_units)
    ]

    network = HopfieldNetwork(random_seed=FIXED_SEED, max_passes=MAX_PASSES).fit(units)
    stored = network.patterns_in(units)
    walk = network.recall_walk(BipolarPattern(request.probe))
    settled_state = walk.result

    return RecallResponse(
        n_units=network.n_units,
        n_stored_patterns=network.n_stored_patterns,
        load=float(network.load),
        weights=[[float(value) for value in row] for row in network.weights.values],
        initial_energy=float(walk.initial_energy),
        passes=[
            RecallPassDocument(
                pass_number=recall_pass.pass_number,
                state=_as_values(recall_pass.state_after),
                energy_before=float(recall_pass.energy_before),
                energy_after=float(recall_pass.energy_after),
                units_changed=recall_pass.n_units_changed,
            )
            for recall_pass in walk
        ],
        settled_state=_as_values(settled_state),
        settled=walk.settled,
        is_fixed_point=network.is_fixed_point(settled_state),
        settled_into=_settled_into(settled_state, stored),
    )


def _settled_into(
    settled_state: BipolarPattern, stored: Sequence[BipolarPattern]
) -> SettledInto:
    """Name the stored pattern the state equals, or whose negation it equals.

    The stored patterns are searched before any negation, so a state that is
    both one pattern and another's negation, which happens when a caller
    stores a pattern and its own negative, is reported as the pattern.
    """
    for index, pattern in enumerate(stored):
        if settled_state == pattern:
            return SettledInto(pattern_index=index, flipped=False)

    for index, pattern in enumerate(stored):
        if settled_state == pattern.flipped():
            return SettledInto(pattern_index=index, flipped=True)

    return SettledInto(pattern_index=None, flipped=False)


def _as_values(pattern: BipolarPattern) -> list[int]:
    """A bipolar state as plain integers, one per unit."""
    return [int(value) for value in pattern.values]
