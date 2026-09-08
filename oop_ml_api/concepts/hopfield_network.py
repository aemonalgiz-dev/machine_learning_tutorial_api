"""A Hopfield network's storage and recall, for the associative memory page.

The recall endpoint stores the request's patterns in the library's
``HopfieldNetwork`` under a fixed seed, hands the probe to ``recall_walk``, and
returns every sweep the walk took, the state after each, the energy before and
after it, and how many units moved. The page needs the whole walk rather than
the answer because the lesson is the settling itself, a staircase of energies
that only ever steps down, and the answer alone would hide it. Alongside the
walk go the settled state, whether it is a fixed point, which stored pattern it
equals if any, or whose negation it equals, the load, and the full weight
matrix, so the page can quote a weight the reader has just worked by hand.

The further endpoints report the road there. ``storage`` shows the outer
product each pattern contributes beside the matrix the library formed, so the
reader can see the sum and the cleared diagonal happen. ``walk-by-unit`` opens
a sweep up into its visits, one unit at a time, with the weighted sum the unit
read and the energy after it moved or did not, which is the granularity the
settling proof is stated at and the granularity ``recall_walk`` does not
record. The library's asynchronous rule visits the units in a permutation
drawn per sweep from the recall's own generator, so this endpoint draws the
same permutations from the same seed, reads every weighted sum and every
energy from the library's weights, and then checks its sweep-level states
against ``recall_walk`` and refuses to answer if they differ, which keeps the
replay honest without asking the library for a route it does not have.
``states`` scores any list of states the browser builds, a path of flips from
one shape to another or a mixture of three, with the library's energy and its
fixed-point verdict. ``capacity`` is the measurement the page quotes against
the literature's 0.138, drawn once and cached. ``self-connection`` puts a
weight back on the diagonal the storage rule cleared and reports what the
units then read. ``contracts`` runs the edges the last part of the page lists.

Rows are patterns and units are the positions within one, so a request's
list of patterns is turned round on the way in. The library takes one
``Feature`` per unit, holding that unit's value in every stored pattern, which
is the same rows-are-samples arrangement every other model here uses and means
something different because of it. The seed fixes the order the units are
visited in during each sweep, which is the only draw the library makes, so the
same probe against the same patterns answers the same way every time and the
page can quote what the widget shows.

The values are refused by the library rather than here. A pattern holding
anything other than -1 and 1 is not a state the network can be in, and the
library says so in a sentence that names the unit at fault, so the endpoints
add no check of their own. What they do check is that the patterns are all the
same length, because the turning round happens before the library sees them
and a ragged list has no column to read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BipolarPattern,
    Feature,
    HebbianWeights,
    HopfieldNetwork,
    RecallWalk,
    UpdateRule,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(tags=["hopfield-network"])

MIN_PATTERNS = 1
"""The fewest patterns a network can store. One is a memory of one thing."""

MAX_PATTERNS = 8
"""The most patterns the widget may store.

Eight over the widget's twenty-five units is a load of 0.32, well past the
point where recall has stopped working, which is the failure the page wants
the reader to be able to reach.
"""

MIN_UNITS = 2
"""The fewest units a pattern may have.

Two is the smallest network that has a connection, and it is the width of the
example that shows the synchronous rule oscillating.
"""

MAX_UNITS = 36
"""The most units a pattern may have, a six-by-six grid."""

MAX_PASSES = 20
"""The most sweeps one recall may take.

Under the library's asynchronous rule settling is guaranteed and a twenty-five
unit network rests in two or three sweeps, so this is a guard rather than the
stopping condition, and the response says which of the two stopped the walk.
Under the synchronous rule it is the only thing that stops an oscillation.
"""

MAX_STATES = 64
"""The most states one request may ask to have scored.

A path from a shape to its negation on twenty-five cells is twenty-six states,
and a path between two shapes is shorter, so this leaves room for both.
"""

FIXED_SEED = 7
"""The unit visiting order is seeded so the pages can quote what the widget shows."""

MAX_SEED = 10_000
"""Seeds are small whole numbers, so a reader can try a handful by hand."""

MAX_SELF_WEIGHT = 5.0
"""The largest self connection the dial may put back on the diagonal."""

PatternValues = Annotated[list[int], Field(min_length=MIN_UNITS, max_length=MAX_UNITS)]

UpdateRuleName = Literal["asynchronous", "synchronous"]


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


def _check_same_width(patterns: Sequence[Sequence[int]]) -> int:
    """The shared width of the patterns, refusing a ragged list in words."""
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


def _units_of(patterns: Sequence[Sequence[float]]) -> list[Feature]:
    """One feature per unit, holding that unit's value in every pattern."""
    n_units = len(patterns[0])
    return [
        Feature(
            f"unit_{position + 1}",
            [pattern[position] for pattern in patterns],
        )
        for position in range(n_units)
    ]


def _fitted(
    patterns: Sequence[Sequence[float]],
    random_seed: int = FIXED_SEED,
    update_rule: UpdateRule = UpdateRule.ASYNCHRONOUS,
    max_passes: int = MAX_PASSES,
) -> tuple[HopfieldNetwork, list[BipolarPattern]]:
    """Store the patterns and hand back the network with what it stored."""
    units = _units_of(patterns)
    network = HopfieldNetwork(
        random_seed=random_seed, update_rule=update_rule, max_passes=max_passes
    ).fit(units)
    return network, network.patterns_in(units)


@router.post("/concepts/hopfield/recall", response_model=RecallResponse)
def recall(request: RecallRequest) -> RecallResponse:
    """Store the patterns, then let the probe settle and report every sweep."""
    _check_same_width(request.patterns)
    network, stored = _fitted(request.patterns)
    walk = network.recall_walk(BipolarPattern(request.probe))
    settled_state = walk.result

    return RecallResponse(
        n_units=network.n_units,
        n_stored_patterns=network.n_stored_patterns,
        load=float(network.load),
        weights=_matrix(network.weights.values),
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


def _matrix(values: np.ndarray) -> list[list[float]]:
    """A square array as plain floats, row by row."""
    return [[float(value) for value in row] for row in values]


# --- Storage: the outer products, their sum, and the cleared diagonal -------


class StorageRequest(BaseModel):
    """The patterns to store, and nothing else."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )

    model_config = {"extra": "forbid"}


class PatternContribution(BaseModel):
    """What one stored pattern adds to every pair of units.

    ``products`` is the outer product of the pattern with itself, ``+1`` where
    two units agree in this pattern and ``-1`` where they disagree, before any
    division and before the diagonal is cleared. It is the definition written
    out, computed here rather than by the library, so the page can show the
    matrix being assembled; the library's own answer is ``weights`` on the
    response, and the two are held to agree.
    """

    pattern_index: int
    products: list[list[int]]


class StorageResponse(BaseModel):
    """The matrix assembled step by step, and the one the library formed.

    ``summed`` is the products added up and divided by the unit count, with
    the diagonal still holding the value every bipolar square gives it, which
    is the number of patterns over the unit count. ``weights`` is the library's
    matrix, which is ``summed`` with the diagonal cleared, and
    ``has_self_connections`` is the library's own check that it was.
    """

    n_units: int
    n_stored_patterns: int
    load: float
    contributions: list[PatternContribution]
    summed: list[list[float]]
    weights: list[list[float]]
    has_self_connections: bool


@router.post("/concepts/hopfield/storage", response_model=StorageResponse)
def storage(request: StorageRequest) -> StorageResponse:
    """Store the patterns and show the outer products that built the matrix."""
    n_units = _check_same_width(request.patterns)
    network, stored = _fitted(request.patterns)

    contributions = [
        PatternContribution(
            pattern_index=index,
            products=[
                [int(value) for value in row]
                for row in np.outer(pattern.values, pattern.values)
            ],
        )
        for index, pattern in enumerate(stored)
    ]
    summed = np.zeros((n_units, n_units))
    for contribution in contributions:
        summed += np.asarray(contribution.products, dtype=np.float64) / n_units

    return StorageResponse(
        n_units=network.n_units,
        n_stored_patterns=network.n_stored_patterns,
        load=float(network.load),
        contributions=contributions,
        summed=_matrix(summed),
        weights=_matrix(network.weights.values),
        has_self_connections=network.weights.has_self_connections,
    )


# --- The walk opened up into visits, one unit at a time ---------------------


class UnitWalkRequest(BaseModel):
    """The patterns, the probe, which rule to settle under, and the seed.

    ``random_seed`` fixes the order the units are visited in, which is the
    only draw the asynchronous rule makes; two seeds can settle an ambiguous
    probe into two different memories, which is the point of exposing it.
    ``max_passes`` matters under the synchronous rule, where it is the only
    thing that stops an oscillation.
    """

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    probe: PatternValues
    update_rule: UpdateRuleName = "asynchronous"
    random_seed: int = Field(default=FIXED_SEED, ge=0, le=MAX_SEED)
    max_passes: int = Field(default=MAX_PASSES, ge=1, le=MAX_PASSES)

    model_config = {"extra": "forbid"}


class UnitVisit(BaseModel):
    """One unit reading its connections and deciding whether to move.

    ``weighted_sum`` is what the unit's connections told it, the sum over the
    other units of the weight joining them times that unit's current value.
    ``flipped`` is whether the unit changed, and ``energy_after`` is the
    network's energy once it had, or had not; under the asynchronous rule it
    is never above ``energy_before``, and a flip lowers it by exactly twice the
    size of the weighted sum.
    """

    pass_number: int
    visit_number: int
    unit: int
    weighted_sum: float
    value_before: int
    value_after: int
    flipped: bool
    energy_before: float
    energy_after: float


class UnitWalkPass(BaseModel):
    """One sweep, as the visits add up to it.

    ``weighted_sums`` are what every unit read at the start of the sweep,
    which is what the synchronous rule acts on all at once and the
    asynchronous rule acts on only for the first unit it happens to visit.
    """

    pass_number: int
    state: list[int]
    energy_before: float
    energy_after: float
    units_changed: int
    weighted_sums: list[float]


class UnitWalkResponse(BaseModel):
    """Every visit and every sweep, under the rule that was asked for.

    ``visits`` is empty under the synchronous rule, which has no visits; every
    unit moves at once from the same old state. ``oscillation_period`` is 2
    when the walk ended by repeating the state it held two sweeps earlier
    while still moving, which is the synchronous failure, and ``None``
    otherwise. ``stopped_because`` is the library's own word for which exit
    the walk took.
    """

    update_rule: UpdateRuleName
    n_units: int
    n_stored_patterns: int
    load: float
    initial_energy: float
    visits: list[UnitVisit]
    passes: list[UnitWalkPass]
    settled_state: list[int]
    settled: bool
    stopped_because: str
    is_fixed_point: bool
    settled_into: SettledInto
    oscillation_period: int | None


@router.post("/concepts/hopfield/walk-by-unit", response_model=UnitWalkResponse)
def walk_by_unit(request: UnitWalkRequest) -> UnitWalkResponse:
    """Settle the probe and report the walk at the granularity of one unit."""
    _check_same_width(request.patterns)
    rule = UpdateRule(request.update_rule)
    network, stored = _fitted(
        request.patterns,
        random_seed=request.random_seed,
        update_rule=rule,
        max_passes=request.max_passes,
    )
    probe = BipolarPattern(request.probe)
    walk = network.recall_walk(probe)
    weights = network.weights

    visits = (
        _replayed_visits(network, walk, probe)
        if rule is UpdateRule.ASYNCHRONOUS
        else []
    )
    passes = [
        UnitWalkPass(
            pass_number=recall_pass.pass_number,
            state=_as_values(recall_pass.state_after),
            energy_before=float(recall_pass.energy_before),
            energy_after=float(recall_pass.energy_after),
            units_changed=recall_pass.n_units_changed,
            weighted_sums=[
                float(value)
                for value in weights.weighted_sums_for(recall_pass.state_before)
            ],
        )
        for recall_pass in walk
    ]
    settled_state = walk.result

    return UnitWalkResponse(
        update_rule=request.update_rule,
        n_units=network.n_units,
        n_stored_patterns=network.n_stored_patterns,
        load=float(network.load),
        initial_energy=float(walk.initial_energy),
        visits=visits,
        passes=passes,
        settled_state=_as_values(settled_state),
        settled=walk.settled,
        stopped_because=walk.stopped_because.value,
        is_fixed_point=network.is_fixed_point(settled_state),
        settled_into=_settled_into(settled_state, stored),
        oscillation_period=_oscillation_period(walk),
    )


def _replayed_visits(
    network: HopfieldNetwork, walk: RecallWalk, probe: BipolarPattern
) -> list[UnitVisit]:
    """Walk the units one at a time, in the order the library visited them.

    The library draws one permutation per sweep from a generator seeded with
    the network's seed, so drawing the same permutations from the same seed
    visits the same units in the same order. Every weighted sum is read from
    the library's weights and every energy is the library's; only the order
    and the sign are taken here. The sweep-level states are then held against
    the library's own walk, and a disagreement is a bug in this replay rather
    than a fact about the network, so it is raised as one.
    """
    weights = network.weights
    matrix = weights.values
    generator = np.random.default_rng(network.random_seed)
    state = np.array(probe.values, dtype=np.float64)
    energy = weights.energy_of(probe)

    visits: list[UnitVisit] = []
    visit_number = 0
    for recall_pass in walk:
        for unit in generator.permutation(state.size):
            visit_number += 1
            weighted_sum = float(matrix[unit] @ state)
            before = state[unit]
            if weighted_sum > 0.0:
                after = 1.0
            elif weighted_sum < 0.0:
                after = -1.0
            else:
                after = before
            state[unit] = after
            energy_after = weights.energy_of(
                BipolarPattern.already_checked(state.copy())
            )
            visits.append(
                UnitVisit(
                    pass_number=recall_pass.pass_number,
                    visit_number=visit_number,
                    unit=int(unit),
                    weighted_sum=weighted_sum,
                    value_before=int(before),
                    value_after=int(after),
                    flipped=after != before,
                    energy_before=energy,
                    energy_after=energy_after,
                )
            )
            energy = energy_after

        if not np.array_equal(state, recall_pass.state_after.values):
            raise RuntimeError(
                f"the unit-by-unit replay parted from the recorded walk on "
                f"sweep {recall_pass.pass_number}"
            )

    return visits


def _oscillation_period(walk: RecallWalk) -> int | None:
    """Two when the walk ended repeating the state from two sweeps back."""
    if walk.settled or len(walk) < 3:
        return None

    passes = list(walk)
    last, before_last, two_back = passes[-1], passes[-2], passes[-3]
    if (
        last.state_after == two_back.state_after
        and last.state_after != before_last.state_after
    ):
        return 2

    return None


# --- Scoring states the browser built -----------------------------------------


class StatesRequest(BaseModel):
    """The patterns to store, and the states to score against them."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    states: list[PatternValues] = Field(min_length=1, max_length=MAX_STATES)

    model_config = {"extra": "forbid"}


class StateReport(BaseModel):
    """One state's energy and whether the network would leave it.

    ``agreements`` counts, for each stored pattern in turn, how many units the
    state holds in common with it; the unit count means identical and zero
    means the state is that pattern's negation. ``units_wanting_to_move`` is
    how many units disagree with the sign of their own weighted sum, and the
    state is a fixed point exactly when that is zero.
    """

    state: list[int]
    energy: float
    is_fixed_point: bool
    units_wanting_to_move: int
    weighted_sums: list[float]
    agreements: list[int]
    equals: SettledInto


class StatesResponse(BaseModel):
    """Every state scored, with the stored patterns' own energies for scale."""

    n_units: int
    n_stored_patterns: int
    load: float
    stored_energies: list[float]
    reports: list[StateReport]


@router.post("/concepts/hopfield/states", response_model=StatesResponse)
def states(request: StatesRequest) -> StatesResponse:
    """Score each state's energy and fixed-point verdict under the stored patterns."""
    _check_same_width(request.patterns)
    network, stored = _fitted(request.patterns)
    weights = network.weights

    reports: list[StateReport] = []
    for values in request.states:
        state = BipolarPattern(values)
        weighted_sums = weights.weighted_sums_for(state)
        wanting_to_move = int(np.count_nonzero(state.values * weighted_sums < 0.0))
        reports.append(
            StateReport(
                state=_as_values(state),
                energy=float(weights.energy_of(state)),
                is_fixed_point=network.is_fixed_point(state),
                units_wanting_to_move=wanting_to_move,
                weighted_sums=[float(value) for value in weighted_sums],
                agreements=[state.agreements_with(pattern) for pattern in stored],
                equals=_settled_into(state, stored),
            )
        )

    return StatesResponse(
        n_units=network.n_units,
        n_stored_patterns=network.n_stored_patterns,
        load=float(network.load),
        stored_energies=[float(weights.energy_of(pattern)) for pattern in stored],
        reports=reports,
    )


# --- Capacity, measured rather than quoted -----------------------------------

CAPACITY_SEED = 11
"""The random patterns and the corruptions are drawn once, from this seed."""

CAPACITY_SETS = 10
"""How many independent sets of random patterns each load is averaged over."""

CAPACITY_SWEEPS: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (25, 5, (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)),
    (100, 10, (5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40)),
)
"""Each sweep is a unit count, how many units the corrupted probe flips, and
the pattern counts to try.

Twenty-five units is the page's grid, and five flips is what its scramble
does. A hundred units is where the literature's figure starts to mean
something, and ten flips is the corruption the library's own measurement
used.
"""

COMPARISON_UNITS = 25
"""The shapes-versus-random comparison is on the page's own grid."""

COMPARISON_FLIPS = 5
"""Five cells corrupted, as the page's scramble does."""

COMPARISON_DRAWS = 20
"""How many corruptions of each stored pattern the comparison averages over."""


def _shape(rows: Sequence[str]) -> list[int]:
    return [1 if cell == "#" else -1 for row in rows for cell in row]


PAGE_SHAPES: dict[str, list[int]] = {
    "the T": _shape(["#####", "..#..", "..#..", "..#..", "..#.."]),
    "the L": _shape(["#....", "#....", "#....", "#....", "#####"]),
    "the cross": _shape(["#...#", ".#.#.", "..#..", ".#.#.", "#...#"]),
    "the square": _shape(["#####", "#...#", "#...#", "#...#", "#####"]),
    "the Z": _shape(["#####", "...#.", "..#..", ".#...", "#####"]),
    "the diamond": _shape(["..#..", ".#.#.", "#...#", ".#.#.", "..#.."]),
}
"""The six shapes the page's playground can store, in the order it stores them."""


class LoadPoint(BaseModel):
    """Recall measured at one load.

    ``exact_share`` is the share of stored patterns that came back unchanged
    when presented exactly, averaged over the sets; ``corrupted_share`` the
    same with ``n_flipped`` units reversed first. ``fixed_point_share`` is the
    share of stored patterns the library reports as fixed points, which is
    the exact probe's question asked without settling.
    """

    n_patterns: int
    load: float
    exact_share: float
    corrupted_share: float
    fixed_point_share: float


class SizeSweep(BaseModel):
    """One unit count swept across loads."""

    n_units: int
    n_sets: int
    n_flipped: int
    points: list[LoadPoint]


class PatternSetTrial(BaseModel):
    """One set of patterns on the page's grid, and how well it recalls.

    ``mean_absolute_overlap`` is the mean over distinct pairs of stored
    patterns of the absolute value of their inner product over the unit
    count, so 0 is orthogonal and 1 is identical or opposite; it is the
    crosstalk the derivation names, measured. ``recovered_share`` is the share
    of corruptions, ``n_flipped`` cells reversed, that settled back into the
    pattern they came from.
    """

    name: str
    n_units: int
    n_patterns: int
    load: float
    mean_absolute_overlap: float
    fixed_point_share: float
    recovered_share: float
    n_flipped: int


class CapacityResponse(BaseModel):
    """The sweep at each size, and the shapes against random patterns."""

    sweeps: list[SizeSweep]
    comparisons: list[PatternSetTrial]


def _random_patterns(
    generator: np.random.Generator, n_patterns: int, n_units: int
) -> np.ndarray:
    return generator.choice([-1.0, 1.0], size=(n_patterns, n_units))


def _corrupted(
    pattern: BipolarPattern, generator: np.random.Generator, n_flipped: int
) -> BipolarPattern:
    values = np.array(pattern.values, dtype=np.float64)
    flipped = generator.choice(values.size, size=n_flipped, replace=False)
    values[flipped] *= -1.0
    return BipolarPattern.already_checked(values)


def _recall_shares(
    network: HopfieldNetwork,
    stored: Sequence[BipolarPattern],
    generator: np.random.Generator,
    n_flipped: int,
    draws_per_pattern: int,
) -> tuple[float, float, float]:
    """Exact recall, corrupted recall and fixed-point shares for one network."""
    exact = sum(network.recall(pattern) == pattern for pattern in stored)
    fixed = sum(network.is_fixed_point(pattern) for pattern in stored)
    recovered = 0
    for pattern in stored:
        for _ in range(draws_per_pattern):
            probe = _corrupted(pattern, generator, n_flipped)
            recovered += network.recall(probe) == pattern
    trials = len(stored) * draws_per_pattern
    return exact / len(stored), recovered / trials, fixed / len(stored)


def _mean_absolute_overlap(patterns: Sequence[BipolarPattern]) -> float:
    if len(patterns) < 2:
        return 0.0
    overlaps = [
        abs(float(first.values @ second.values)) / first.n_units
        for index, first in enumerate(patterns)
        for second in patterns[index + 1 :]
    ]
    return float(np.mean(overlaps))


def _swept(n_units: int, n_flipped: int, counts: Sequence[int]) -> SizeSweep:
    generator = np.random.default_rng(CAPACITY_SEED + n_units)
    points: list[LoadPoint] = []
    for n_patterns in counts:
        exact_total = corrupted_total = fixed_total = 0.0
        for _ in range(CAPACITY_SETS):
            network, stored = _fitted(_random_patterns(generator, n_patterns, n_units))
            exact, corrupted, fixed = _recall_shares(
                network, stored, generator, n_flipped, draws_per_pattern=1
            )
            exact_total += exact
            corrupted_total += corrupted
            fixed_total += fixed
        points.append(
            LoadPoint(
                n_patterns=n_patterns,
                load=n_patterns / n_units,
                exact_share=exact_total / CAPACITY_SETS,
                corrupted_share=corrupted_total / CAPACITY_SETS,
                fixed_point_share=fixed_total / CAPACITY_SETS,
            )
        )
    return SizeSweep(
        n_units=n_units, n_sets=CAPACITY_SETS, n_flipped=n_flipped, points=points
    )


def _shapes_trial(name: str, shapes: Sequence[Sequence[int]]) -> PatternSetTrial:
    generator = np.random.default_rng(CAPACITY_SEED)
    network, stored = _fitted(shapes)
    _, recovered, fixed = _recall_shares(
        network, stored, generator, COMPARISON_FLIPS, COMPARISON_DRAWS
    )
    return PatternSetTrial(
        name=name,
        n_units=COMPARISON_UNITS,
        n_patterns=len(shapes),
        load=len(shapes) / COMPARISON_UNITS,
        mean_absolute_overlap=_mean_absolute_overlap(stored),
        fixed_point_share=fixed,
        recovered_share=recovered,
        n_flipped=COMPARISON_FLIPS,
    )


def _random_trial(name: str, n_patterns: int) -> PatternSetTrial:
    generator = np.random.default_rng(CAPACITY_SEED)
    overlap_total = fixed_total = recovered_total = 0.0
    for _ in range(CAPACITY_SETS):
        network, stored = _fitted(
            _random_patterns(generator, n_patterns, COMPARISON_UNITS)
        )
        _, recovered, fixed = _recall_shares(
            network, stored, generator, COMPARISON_FLIPS, COMPARISON_DRAWS
        )
        overlap_total += _mean_absolute_overlap(stored)
        fixed_total += fixed
        recovered_total += recovered
    return PatternSetTrial(
        name=name,
        n_units=COMPARISON_UNITS,
        n_patterns=n_patterns,
        load=n_patterns / COMPARISON_UNITS,
        mean_absolute_overlap=overlap_total / CAPACITY_SETS,
        fixed_point_share=fixed_total / CAPACITY_SETS,
        recovered_share=recovered_total / CAPACITY_SETS,
        n_flipped=COMPARISON_FLIPS,
    )


@cache
def _capacity() -> CapacityResponse:
    shapes = list(PAGE_SHAPES.values())
    return CapacityResponse(
        sweeps=[_swept(*sweep) for sweep in CAPACITY_SWEEPS],
        comparisons=[
            _shapes_trial("three shapes", shapes[:3]),
            _random_trial("three random patterns", 3),
            _shapes_trial("six shapes", shapes),
            _random_trial("six random patterns", 6),
        ],
    )


@router.get("/concepts/hopfield/capacity", response_model=CapacityResponse)
def capacity() -> CapacityResponse:
    """How recall degrades with load, at two sizes, measured once and kept."""
    return _capacity()


# --- A weight put back on the diagonal ------------------------------------------


class SelfConnectionRequest(BaseModel):
    """The patterns, the probe, and the self weight to put on every diagonal entry."""

    patterns: list[PatternValues] = Field(
        min_length=MIN_PATTERNS, max_length=MAX_PATTERNS
    )
    probe: PatternValues
    self_weight: float = Field(ge=0.0, le=MAX_SELF_WEIGHT)

    model_config = {"extra": "forbid"}


class SelfConnectionResponse(BaseModel):
    """What the probe's units read with and without the self connection.

    ``units_wanting_to_move_without`` and ``..._with`` list the units whose
    value disagrees with the sign of their weighted sum under each matrix.
    ``energy_shift`` is the constant the self weight adds to every state's
    energy, the same for all of them, which is why the landscape's order is
    untouched; ``largest_row_sum`` is the largest sum of absolute weights in
    any row away from the diagonal, and a self weight above it stops every
    unit from ever moving.
    """

    self_weight: float
    has_self_connections: bool
    weighted_sums_without: list[float]
    weighted_sums_with: list[float]
    units_wanting_to_move_without: list[int]
    units_wanting_to_move_with: list[int]
    probe_energy_without: float
    probe_energy_with: float
    energy_shift: float
    largest_row_sum: float
    settled_state_without: list[int]
    settled_into_without: SettledInto


@router.post(
    "/concepts/hopfield/self-connection", response_model=SelfConnectionResponse
)
def self_connection(request: SelfConnectionRequest) -> SelfConnectionResponse:
    """Put a weight back on the diagonal and report what the units then read."""
    _check_same_width(request.patterns)
    network, stored = _fitted(request.patterns)
    probe = BipolarPattern(request.probe)
    without = network.weights

    with_values = np.array(without.values, dtype=np.float64)
    np.fill_diagonal(with_values, request.self_weight)
    with_self = HebbianWeights(with_values, without.unit_names)

    sums_without = without.weighted_sums_for(probe)
    sums_with = with_self.weighted_sums_for(probe)
    off_diagonal = np.abs(without.values).sum(axis=1)
    settled = network.recall(probe)

    return SelfConnectionResponse(
        self_weight=request.self_weight,
        has_self_connections=with_self.has_self_connections,
        weighted_sums_without=[float(value) for value in sums_without],
        weighted_sums_with=[float(value) for value in sums_with],
        units_wanting_to_move_without=_wanting_to_move(probe, sums_without),
        units_wanting_to_move_with=_wanting_to_move(probe, sums_with),
        probe_energy_without=float(without.energy_of(probe)),
        probe_energy_with=float(with_self.energy_of(probe)),
        energy_shift=float(with_self.energy_of(probe) - without.energy_of(probe)),
        largest_row_sum=float(off_diagonal.max()),
        settled_state_without=_as_values(settled),
        settled_into_without=_settled_into(settled, stored),
    )


def _wanting_to_move(probe: BipolarPattern, weighted_sums: np.ndarray) -> list[int]:
    return [int(unit) for unit in np.flatnonzero(probe.values * weighted_sums < 0.0)]


# --- The contracts ----------------------------------------------------------------


class ContractProbe(BaseModel):
    """One edge, and what the library did with it."""

    edge: str
    outcome: Literal["accepted", "refused"]
    error: str | None
    detail: str


class ContractsResponse(BaseModel):
    """Every edge the page's last part lists, probed rather than asserted."""

    probes: list[ContractProbe]


FOUR_CELL_PATTERNS = ([1, 1, -1, -1], [1, 1, -1, 1])
"""The page's hand-worked pair."""

TWO_UNIT_PATTERN = [1, -1]
"""The pattern whose network the synchronous rule oscillates in."""


def _probe(edge: str, attempt: Callable[[], str]) -> ContractProbe:
    """Run one attempt and record whether the library accepted or refused."""
    try:
        detail = attempt()
    except (MLLibError, ValidationError) as error:
        return ContractProbe(
            edge=edge, outcome="refused", error=type(error).__name__, detail=str(error)
        )
    return ContractProbe(edge=edge, outcome="accepted", error=None, detail=detail)


def _four_cell_network() -> tuple[HopfieldNetwork, list[BipolarPattern]]:
    return _fitted(FOUR_CELL_PATTERNS)


def _fit_units(units: list[Feature]) -> HopfieldNetwork:
    return HopfieldNetwork(random_seed=FIXED_SEED).fit(units)


def _no_units() -> str:
    _fit_units([])
    return ""


def _one_unit() -> str:
    _fit_units([Feature("unit_1", [1.0, -1.0])])
    return ""


def _one_pattern() -> str:
    network, _ = _fitted([[1, 1, -1, -1]])
    return (
        f"fits; every weight off the diagonal is +1/4 or -1/4, and the load is "
        f"{network.load:.2f}"
    )


def _constant_unit() -> str:
    network, _ = _fitted([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    first_row = network.weights.values[0]
    return (
        "fits; a unit lit in every pattern is wired to each other unit by that "
        "unit's values summed over the patterns and divided by the unit count, "
        f"here {', '.join(f'{value:+.2f}' for value in first_row)}"
    )


def _a_zero() -> str:
    _fitted([[1, 1, 0, -1], [1, 1, -1, 1]])
    return ""


def _a_nan() -> str:
    _fit_units(
        [
            Feature("unit_1", [1.0, 1.0]),
            Feature("unit_2", [1.0, float("nan")]),
            Feature("unit_3", [-1.0, -1.0]),
        ]
    )
    return ""


def _ragged_units() -> str:
    _fit_units(
        [
            Feature("unit_1", [1.0, 1.0]),
            Feature("unit_2", [1.0, -1.0, 1.0]),
        ]
    )
    return ""


def _duplicate_names() -> str:
    _fit_units(
        [
            Feature("unit_1", [1.0, 1.0]),
            Feature("unit_1", [1.0, -1.0]),
        ]
    )
    return ""


def _more_patterns_than_units() -> str:
    patterns = [
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
        [-1, 1, 1, -1],
        [1, 1, 1, -1],
        [-1, -1, 1, 1],
    ]
    network, stored = _fitted(patterns)
    fixed = sum(network.is_fixed_point(pattern) for pattern in stored)
    return (
        f"fits at a load of {network.load:.2f}; {fixed} of the {len(stored)} "
        "stored patterns are still fixed points"
    )


def _same_pattern_twice() -> str:
    once, _ = _fitted([[1, 1, -1, -1]])
    twice, _ = _fitted([[1, 1, -1, -1], [1, 1, -1, -1]])
    doubled = np.allclose(twice.weights.values, 2.0 * once.weights.values)
    return (
        "fits; every weight is exactly doubled"
        if doubled
        else "fits; the weights are not simply doubled"
    )


def _pattern_and_its_negation() -> str:
    once, _ = _fitted([[1, 1, -1, -1]])
    both, stored = _fitted([[1, 1, -1, -1], [-1, -1, 1, 1]])
    doubled = np.allclose(both.weights.values, 2.0 * once.weights.values)
    reported = _settled_into(both.recall(stored[1]), stored)
    return (
        "fits, and the weights are exactly those of storing the first pattern "
        "twice, since the negation contributes the same products; the second "
        f"pattern recalls as pattern {reported.pattern_index} with flipped set "
        f"to {reported.flipped}"
        if doubled
        else "fits; the weights differ from the doubled ones"
    )


def _recall_before_fit() -> str:
    HopfieldNetwork().recall(BipolarPattern([1, 1, -1, -1]))
    return ""


def _probe_of_the_wrong_width() -> str:
    network, _ = _four_cell_network()
    network.recall(BipolarPattern([1, 1, -1, -1, 1]))
    return ""


def _transform_missing_a_unit() -> str:
    network, _ = _four_cell_network()
    network.transform(
        [
            Feature("unit_1", [1.0]),
            Feature("unit_2", [1.0]),
            Feature("unit_3", [-1.0]),
        ]
    )
    return ""


def _transform_with_an_unknown_unit() -> str:
    network, _ = _four_cell_network()
    network.transform(
        [
            Feature("unit_1", [1.0]),
            Feature("unit_2", [1.0]),
            Feature("unit_3", [-1.0]),
            Feature("unit_5", [-1.0]),
        ]
    )
    return ""


def _transform_reordered() -> str:
    network, _ = _four_cell_network()
    recalled = network.transform(
        [
            Feature("unit_4", [-1.0]),
            Feature("unit_3", [-1.0]),
            Feature("unit_2", [1.0]),
            Feature("unit_1", [-1.0]),
        ]
    )
    values = [int(feature.values[0]) for feature in recalled]
    names = [feature.name for feature in recalled]
    return (
        f"accepted; the units are matched by name, and come back in the fitted "
        f"order {names} holding {values}"
    )


def _a_probe_already_at_rest() -> str:
    network, _ = _four_cell_network()
    walk = network.recall_walk(BipolarPattern([1, 1, -1, -1]))
    return (
        f"one sweep, moving {next(iter(walk)).n_units_changed} units, and the walk "
        f"reports {walk.stopped_because.value}"
    )


def _a_weighted_sum_of_zero() -> str:
    network, _ = _four_cell_network()
    state = BipolarPattern([1, 1, -1, 1])
    sums = network.weights.weighted_sums_for(state)
    zeros = [int(unit) + 1 for unit in np.flatnonzero(sums == 0.0)]
    return (
        f"cell {zeros} reads a weighted sum of exactly zero and is left as it is; "
        f"the state is reported as a fixed point: {network.is_fixed_point(state)}"
    )


def _pass_limit_below_one() -> str:
    HopfieldNetwork(max_passes=0)
    return ""


def _unknown_keyword() -> str:
    HopfieldNetwork(passes=5)  # type: ignore[call-arg]
    return ""


def _synchronous_at_the_pass_limit() -> str:
    network, _ = _fitted(
        [TWO_UNIT_PATTERN], update_rule=UpdateRule.SYNCHRONOUS, max_passes=6
    )
    walk = network.recall_walk(BipolarPattern([1, 1]))
    return (
        f"completes rather than raises; {walk.passes_run} sweeps, every one moving "
        f"both units, and the walk reports {walk.stopped_because.value} with the "
        f"energy at {walk.energies[-1]:+.1f} where it started"
    )


def _self_connected_container() -> str:
    weights = HebbianWeights(
        [
            [3.0, 0.5, -0.5, 0.0],
            [0.5, 3.0, -0.5, 0.0],
            [-0.5, -0.5, 3.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ],
        ("unit_1", "unit_2", "unit_3", "unit_4"),
    )
    return (
        "accepted on purpose, and reported: has_self_connections is "
        f"{weights.has_self_connections}"
    )


def _asymmetric_container() -> str:
    HebbianWeights(
        [
            [0.0, 0.5, -0.5, 0.0],
            [0.4, 0.0, -0.5, 0.0],
            [-0.5, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        ("unit_1", "unit_2", "unit_3", "unit_4"),
    )
    return ""


CONTRACT_PROBES: tuple[tuple[str, Callable[[], str]], ...] = (
    ("no units at all", _no_units),
    ("one unit", _one_unit),
    ("one pattern", _one_pattern),
    ("a unit lit in every pattern", _constant_unit),
    ("a value of 0 in a pattern", _a_zero),
    ("a non-finite value", _a_nan),
    ("units holding different numbers of patterns", _ragged_units),
    ("two units with one name", _duplicate_names),
    ("more patterns than units", _more_patterns_than_units),
    ("the same pattern stored twice", _same_pattern_twice),
    ("a pattern and its negation stored together", _pattern_and_its_negation),
    ("recall before fit", _recall_before_fit),
    ("a probe of the wrong width", _probe_of_the_wrong_width),
    ("a table missing a fitted unit", _transform_missing_a_unit),
    ("a table with an unknown unit", _transform_with_an_unknown_unit),
    ("a table with the units reordered", _transform_reordered),
    ("a probe that is already at rest", _a_probe_already_at_rest),
    ("a weighted sum of exactly zero", _a_weighted_sum_of_zero),
    ("a pass limit below one", _pass_limit_below_one),
    ("an unknown keyword", _unknown_keyword),
    ("synchronous update on an oscillating probe", _synchronous_at_the_pass_limit),
    ("a self-connected matrix handed to the container", _self_connected_container),
    ("an asymmetric matrix handed to the container", _asymmetric_container),
)


@cache
def _contracts() -> ContractsResponse:
    return ContractsResponse(
        probes=[_probe(edge, attempt) for edge, attempt in CONTRACT_PROBES]
    )


@router.get("/concepts/hopfield/contracts", response_model=ContractsResponse)
def contracts() -> ContractsResponse:
    """Every edge, run against the library and reported as it answered."""
    return _contracts()
