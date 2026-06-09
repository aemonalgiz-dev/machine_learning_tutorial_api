"""A self-organising map's resting state, for the map page.

The first endpoint fits the library's ``SelfOrganisingMap`` on the request's
people under a fixed seed, with the grid the request names and the epoch
budget it sets, and returns where every unit came to rest, which unit won each
person, how many epochs ran, how far the map moved on its last epoch, and the
quantisation error. The page needs the resting map rather than a walk because
the library records none. Both schedules are fractions of the epoch budget,
so a fit stopped at epoch ten of a hundred is not the hundred-epoch fit
caught early, it is a different fit with a faster decay, and drawing six
budgets as six frames of one walk would be a small lie. The widget says
instead that every change is a fresh fit, and the epochs slider stretches the
whole walk.

Five further endpoints report the parts of that fit the resting map hides.
The presentation endpoint runs the library's own competition, cooperation and
adaptation once, on units the request places by hand, so a reader can check
every unit's move with a calculator. The schedules endpoint reads the two
curves the library walks under, epoch by epoch, and counts how many units
still move at each radius. The ablation endpoint refits the same people four
times, holding the rate, the radius, both or neither, which is the only way
to say what each of the two decays does on its own. The topology endpoint
reports every pair of units with its grid distance beside its weight
distance, once with the neighbourhood and once without, which is where the
claim that neighbouring units hold similar people is either true or is not.
The seeds endpoint fits the same people from several seeds, because a map is
an arrangement and a differently seeded fit can lay the same arrangement out
backwards. The versus-k-means endpoint fits a chain with its reach switched
off beside the grouping method itself, which is the claim that a map with no
reach is that method, tested rather than asserted. The edges endpoint runs
every row of the page's contract table against the library and reports what
came back, so a refusal quoted on the page is a refusal the library made
while the page was being served rather than one somebody remembered.

The grid is the model, so the response carries it twice over. Every unit
comes back with its row and column as well as its resting height and weight,
in the row-major order the library keeps, and each person's winner is
reported as a row and column rather than an index, so the widget can drape
the net and colour the people without knowing how the library numbers units.

The quantisation error is computed here and is arithmetic rather than
modelling. It is the mean straight-line distance from each person to the
unit that won them, read off the positions the library answered with, and it
is the figure a resting map can be judged by, the way inertia judges a
resting k-means. The seed fixes both where the units start, on people drawn
from the request, and the order the people are presented in each epoch, so
the same people on the same grid answer the same way every time and the page
can quote what the widget shows.

The two schedules are fixed here rather than requested, because the page
teaches the shape of a schedule and not its tuning. The rate falls
geometrically from a half to a hundredth, the library's own default. The
radius falls in a straight line from half the grid's longest side, which the
library's docstring names as the place to start and cannot default to
because a field default cannot read another field, down to a quarter of a
grid step, where a unit's nearest neighbour on the grid moves by less than a
thousandth of the winner's step and the map is, for practical purposes,
refining each unit alone.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    Feature,
    GridPosition,
    KMeans,
    LinearDecaySchedule,
    Schedule,
    SelfOrganisingMap,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["self-organising-map"])

MIN_GRID_SIDE = 1
"""The narrowest a grid may be along either side. A one-by-one map is one unit."""

MAX_GRID_SIDE = 6
"""The widest a grid may be along either side.

Thirty-six units over a hundred people at the longest walk answers in under
two seconds, and a net of six by six is already as much as the widget can
draw legibly over a scatter of that size.
"""

MIN_EPOCHS = 1
"""The shortest walk. One epoch presents every person once at the starting rate."""

MAX_EPOCHS = 200
"""The longest walk the widget may ask for."""

FIXED_SEED = 3
"""The seeding is fixed so the page can quote what the widget shows."""

LEARNING_RATE = ExponentialDecaySchedule(start=0.5, end=0.01)
"""How far a winner moves toward a person, as a fraction of the gap.

The library's default, a geometric fall from a half to a hundredth, because
what matters about a rate is its order of magnitude and a geometric fall
spends most of the walk at small values.
"""

RADIUS_END = 0.25
"""Where the neighbourhood radius ends, in grid steps.

At a quarter of a step the nearest neighbour on the grid scores
exp(-8), about a third of a thousandth, so by the end of the walk the
neighbourhood has shrunk to nothing that matters and only the winner moves.
"""


MAX_RADIUS = 10.0
"""The widest neighbourhood a request may ask for, in grid steps.

Larger than the diagonal of the largest grid this API will build, so a
request can ask for a reach that never shrinks below the whole map and see
what that does, and small enough that the Gaussian cannot underflow into
anything strange.
"""


class OrganiseRequest(BaseModel):
    """The people, the grid to drape over them, and how long to walk.

    ``radius_start`` and ``radius_end`` override the neighbourhood schedule,
    which is otherwise half the grid's longest side falling to a quarter of a
    step. Both zero switches the neighbourhood off entirely, which is the
    grouping method with extra machinery; both large leaves every unit moving
    with every winner for the whole walk.
    """

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    radius_start: float | None = Field(default=None, ge=0.0, le=MAX_RADIUS)
    radius_end: float | None = Field(default=None, ge=0.0, le=MAX_RADIUS)

    model_config = {"extra": "forbid"}


class MapUnitDocument(BaseModel):
    """One unit, where it sits on the grid and where its weights came to rest."""

    row: int
    column: int
    x: float
    y: float


class WinningUnit(BaseModel):
    """The place on the grid of the unit that won one person."""

    row: int
    column: int


class OrganiseResponse(BaseModel):
    """The resting map, who won whom, and how the walk ended.

    ``units`` are in row-major order, the first ``grid_width`` of them the top
    row of the grid. ``winners`` has one entry per person, in the order the
    people were sent. ``epochs_run`` is how many passes the walk took, and
    ``final_movement`` is the furthest any unit moved during the last of them.
    ``quantisation_error`` is the mean distance from each person to the unit
    that won them.
    """

    grid_width: int
    grid_height: int
    units: list[MapUnitDocument]
    winners: list[WinningUnit]
    epochs_run: int
    final_movement: float
    quantisation_error: float


@router.post("/concepts/self-organising-map/fit", response_model=OrganiseResponse)
def fit(request: OrganiseRequest) -> OrganiseResponse:
    """Drape the grid over the people and report where it came to rest."""
    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    model = SelfOrganisingMap(
        grid_width=request.grid_width,
        grid_height=request.grid_height,
        learning_rate=LEARNING_RATE,
        neighbourhood_radius=_radius_schedule(
            request.grid_width,
            request.grid_height,
            request.radius_start,
            request.radius_end,
        ),
        max_epochs=request.max_epochs,
        random_seed=FIXED_SEED,
    ).fit([heights, weights])

    units = [
        MapUnitDocument(
            row=unit.position.row,
            column=unit.position.column,
            x=float(unit.weight_for("height")),
            y=float(unit.weight_for("weight")),
        )
        for unit in model.units
    ]

    winning_indices = [int(value) for value in model.predict([heights, weights]).values]
    winners = [
        WinningUnit(row=units[index].row, column=units[index].column)
        for index in winning_indices
    ]

    return OrganiseResponse(
        grid_width=request.grid_width,
        grid_height=request.grid_height,
        units=units,
        winners=winners,
        epochs_run=int(model.epochs_run),
        final_movement=float(model.final_epoch_movement),
        quantisation_error=_quantisation_error(request.points, units, winning_indices),
    )


def _radius_schedule(
    grid_width: int,
    grid_height: int,
    start: float | None = None,
    end: float | None = None,
) -> LinearDecaySchedule:
    """The neighbourhood radius, from half the longest side down to a quarter step.

    Linear rather than geometric, because a radius is measured in the grid's
    own steps and the midpoint of the walk should be the middle of the range.
    A request may name either end itself, which is how the page switches the
    neighbourhood off and how it holds the reach wide for the whole walk.
    """
    return LinearDecaySchedule(
        start=max(grid_width, grid_height) / 2.0 if start is None else start,
        end=RADIUS_END if end is None else end,
    )


def _quantisation_error(
    points: list[PlainPoint],
    units: list[MapUnitDocument],
    winning_indices: list[int],
) -> float:
    """The mean straight-line distance from each person to the unit that won them."""
    distances = [
        math.hypot(point.x - units[index].x, point.y - units[index].y)
        for point, index in zip(points, winning_indices, strict=True)
    ]

    return float(sum(distances) / len(distances))


# --- The further endpoints the page reads ----------------------------------

MOVE_THRESHOLD = 0.001
"""How much of the winner's step a unit must take to count as moving.

Every unit moves a little at every presentation, since a Gaussian never
reaches zero, so counting the units that move needs a threshold rather than a
test against zero. A thousandth of the winner's step is below anything a
picture of the map could show.
"""

MAX_SEEDS = 6
"""The most seeds the gallery may fit the same people from."""


def _features(points: list[PlainPoint]) -> list[Feature]:
    """The two features every fit here reads, in the order the library keeps."""
    return [
        Feature("height", [point.x for point in points]),
        Feature("weight", [point.y for point in points]),
    ]


def _matrix(points: list[PlainPoint]) -> np.ndarray:
    """The people as a two-column block, height first."""
    return np.asarray([[point.x, point.y] for point in points], dtype=np.float64)


def _fitted(
    points: list[PlainPoint],
    grid_width: int,
    grid_height: int,
    max_epochs: int,
    learning_rate: Schedule,
    neighbourhood_radius: Schedule,
    seed: int = FIXED_SEED,
) -> SelfOrganisingMap:
    """One fit, with both schedules named by the caller rather than defaulted."""
    return SelfOrganisingMap(
        grid_width=grid_width,
        grid_height=grid_height,
        learning_rate=learning_rate,
        neighbourhood_radius=neighbourhood_radius,
        max_epochs=max_epochs,
        random_seed=seed,
    ).fit(_features(points))


def _unit_documents(model: SelfOrganisingMap) -> list[MapUnitDocument]:
    """Every unit's place on the grid and where its weights came to rest."""
    return [
        MapUnitDocument(
            row=unit.position.row,
            column=unit.position.column,
            x=float(unit.weight_for("height")),
            y=float(unit.weight_for("weight")),
        )
        for unit in model.units
    ]


def _winning_indices(model: SelfOrganisingMap, points: list[PlainPoint]) -> list[int]:
    """Which unit won each person, as an index into the row-major order."""
    return [int(value) for value in model.predict(_features(points)).values]


class UnitPair(BaseModel):
    """Two units, how far apart on the grid, and how far apart in the data."""

    first_row: int
    first_column: int
    second_row: int
    second_column: int
    grid_distance: float
    weight_distance: float


class CellContents(BaseModel):
    """One unit, and the people it won.

    ``people`` holds their positions in the order the request sent them, so a
    widget can find them again in its own scatter. ``mean_x`` and ``mean_y``
    are the average of those people and are absent for a unit that won nobody,
    which is an ordinary outcome on a map rather than a failure.
    """

    row: int
    column: int
    x: float
    y: float
    n_people: int
    people: list[int]
    mean_x: float | None
    mean_y: float | None


class MapReport(BaseModel):
    """One fitted map, read three ways.

    ``units`` is where every unit came to rest, ``cells`` is what each of them
    won, and ``pairs`` is every pair of units with its grid distance beside
    its weight distance. ``correlation`` is how well the first predicts the
    second across those pairs, which is the topology claim as a number.
    """

    grid_width: int
    grid_height: int
    units: list[MapUnitDocument]
    cells: list[CellContents]
    pairs: list[UnitPair]
    correlation: float | None
    quantisation_error: float
    final_movement: float
    epochs_run: int


def _unit_pairs(units: list[MapUnitDocument]) -> list[UnitPair]:
    """Every pair of units twice over, on the grid and in the input space.

    The grid distance comes from the library's own ``GridPosition``, which is
    the one type that measures the map rather than the people, so the two
    distances in a pair cannot be reached by one expression here either.
    """
    pairs: list[UnitPair] = []
    for first in range(len(units)):
        for second in range(first + 1, len(units)):
            here, there = units[first], units[second]
            pairs.append(
                UnitPair(
                    first_row=here.row,
                    first_column=here.column,
                    second_row=there.row,
                    second_column=there.column,
                    grid_distance=GridPosition(here.row, here.column).distance_to(
                        GridPosition(there.row, there.column)
                    ),
                    weight_distance=math.hypot(here.x - there.x, here.y - there.y),
                )
            )
    return pairs


def _topology_correlation(pairs: list[UnitPair]) -> float | None:
    """How well grid distance predicts weight distance, over every pair.

    ``None`` when either list has no spread at all, which is a grid of one
    unit, where there are no pairs, and a map whose units all came to rest in
    one place, where every weight distance is zero and the correlation would
    divide by nothing.
    """
    if len(pairs) < 2:
        return None
    grid = np.asarray([pair.grid_distance for pair in pairs], dtype=np.float64)
    weights = np.asarray([pair.weight_distance for pair in pairs], dtype=np.float64)
    if grid.std() == 0.0 or weights.std() == 0.0:
        return None
    return float(np.corrcoef(grid, weights)[0, 1])


def _map_report(model: SelfOrganisingMap, points: list[PlainPoint]) -> MapReport:
    """Everything the page reads off one fitted map."""
    units = _unit_documents(model)
    winners = _winning_indices(model, points)

    cells: list[CellContents] = []
    for index, unit in enumerate(units):
        won = [position for position, winner in enumerate(winners) if winner == index]
        held = _matrix([points[position] for position in won]) if won else None
        cells.append(
            CellContents(
                row=unit.row,
                column=unit.column,
                x=unit.x,
                y=unit.y,
                n_people=len(won),
                people=won,
                mean_x=None if held is None else float(held[:, 0].mean()),
                mean_y=None if held is None else float(held[:, 1].mean()),
            )
        )

    pairs = _unit_pairs(units)
    return MapReport(
        grid_width=model.grid_width,
        grid_height=model.grid_height,
        units=units,
        cells=cells,
        pairs=pairs,
        correlation=_topology_correlation(pairs),
        quantisation_error=_quantisation_error(points, units, winners),
        final_movement=float(model.final_epoch_movement),
        epochs_run=int(model.epochs_run),
    )


def _same_partition(first: list[int], second: list[int]) -> bool:
    """Whether two labellings put the same people together, whatever the numbers."""
    return all(
        (first[i] == first[j]) == (second[i] == second[j])
        for i in range(len(first))
        for j in range(i + 1, len(first))
    )


class PresentationRequest(BaseModel):
    """One person shown to a grid of units placed by hand.

    ``units`` are the weight vectors before the presentation, in row-major
    order, which is what lets the page put them at round numbers and check the
    arithmetic with a calculator rather than reporting whatever a walk
    happened to leave behind.
    """

    units: list[PlainPoint] = Field(min_length=1, max_length=MAX_GRID_SIDE**2)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    person: PlainPoint
    rate: float = Field(gt=0.0, le=1.0)
    radius: float = Field(ge=0.0, le=MAX_RADIUS)

    model_config = {"extra": "forbid"}


class PresentedUnit(BaseModel):
    """What one presentation did to one unit.

    ``grid_distance`` is measured on the grid and ``distance_before`` in the
    input space, and holding both in one document is the point of it, since
    the score is a function of the first and the gap is a function of the
    second.
    """

    row: int
    column: int
    is_winner: bool
    grid_distance: float
    score: float
    before_x: float
    before_y: float
    after_x: float
    after_y: float
    moved: float
    distance_before: float
    distance_after: float


class PresentationResponse(BaseModel):
    """The winner, and every unit's share of the step.

    ``moving_units`` counts the units that took at least a thousandth of the
    winner's step, which is the count the schedules endpoint reports across a
    whole walk.
    """

    winner_row: int
    winner_column: int
    units: list[PresentedUnit]
    moving_units: int


@router.post(
    "/concepts/self-organising-map/presentation", response_model=PresentationResponse
)
def presentation(request: PresentationRequest) -> PresentationResponse:
    """Show one person to the grid once and report what each unit did."""
    if len(request.units) != request.grid_width * request.grid_height:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a {request.grid_height} by {request.grid_width} grid holds "
                f"{request.grid_width * request.grid_height} units; got "
                f"{len(request.units)}"
            ),
        )

    # The library's competition, cooperation and adaptation, called one step at
    # a time rather than through a fit, so the page can show the three halves
    # of a single presentation apart from one another.
    model = SelfOrganisingMap(
        grid_width=request.grid_width,
        grid_height=request.grid_height,
        max_epochs=1,
        random_seed=FIXED_SEED,
    )
    weights = _matrix(request.units)
    presented = np.asarray([request.person.x, request.person.y], dtype=np.float64)

    winner = model._winner_for(presented, weights)
    scores = model._neighbourhood_at(winner, request.radius)
    moved = model._updated_weights(
        weights, presented, winner, request.rate, request.radius
    )
    winning_position = model._position_at(winner)

    units: list[PresentedUnit] = []
    for index, score in enumerate(scores):
        position = model._position_at(index)
        before, after = weights[index], moved[index]
        units.append(
            PresentedUnit(
                row=position.row,
                column=position.column,
                is_winner=index == winner,
                grid_distance=position.distance_to(winning_position),
                score=float(score),
                before_x=float(before[0]),
                before_y=float(before[1]),
                after_x=float(after[0]),
                after_y=float(after[1]),
                moved=float(math.hypot(*(after - before))),
                distance_before=float(math.hypot(*(presented - before))),
                distance_after=float(math.hypot(*(presented - after))),
            )
        )

    return PresentationResponse(
        winner_row=winning_position.row,
        winner_column=winning_position.column,
        units=units,
        moving_units=int(sum(1 for score in scores if score >= MOVE_THRESHOLD)),
    )


class ScheduleRequest(BaseModel):
    """The grid the schedules are read against, and the epoch budget."""

    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class EpochSchedule(BaseModel):
    """What the two curves hold on one epoch, and what that does to the grid.

    ``neighbour_score`` is the share of the winner's step taken by a unit one
    grid step away and ``diagonal_score`` the share taken by one across a
    diagonal; ``moving_units`` counts the units taking at least a thousandth
    of that step when the middle unit of the grid wins.
    """

    epoch: int
    rate: float
    radius: float
    neighbour_score: float
    diagonal_score: float
    moving_units: int


class ScheduleResponse(BaseModel):
    """The two curves across the whole walk, and the grid they were read on."""

    epochs: list[EpochSchedule]
    n_units: int
    move_threshold: float
    middle_row: int
    middle_column: int


@router.post("/concepts/self-organising-map/schedules", response_model=ScheduleResponse)
def schedules(request: ScheduleRequest) -> ScheduleResponse:
    """Read the rate and the radius at every epoch of the walk."""
    model = SelfOrganisingMap(
        grid_width=request.grid_width,
        grid_height=request.grid_height,
        max_epochs=request.max_epochs,
        random_seed=FIXED_SEED,
    )
    radius_schedule = _radius_schedule(request.grid_width, request.grid_height)
    middle = (request.grid_height // 2) * request.grid_width + request.grid_width // 2
    middle_position = model._position_at(middle)

    epochs: list[EpochSchedule] = []
    for epoch in range(1, request.max_epochs + 1):
        radius = radius_schedule.value_at(epoch, request.max_epochs)
        scores = model._neighbourhood_at(middle, radius)
        epochs.append(
            EpochSchedule(
                epoch=epoch,
                rate=LEARNING_RATE.value_at(epoch, request.max_epochs),
                radius=radius,
                neighbour_score=_gaussian_at(1.0, radius),
                diagonal_score=_gaussian_at(math.sqrt(2.0), radius),
                moving_units=int(sum(1 for score in scores if score >= MOVE_THRESHOLD)),
            )
        )

    return ScheduleResponse(
        epochs=epochs,
        n_units=request.grid_width * request.grid_height,
        move_threshold=MOVE_THRESHOLD,
        middle_row=middle_position.row,
        middle_column=middle_position.column,
    )


def _gaussian_at(grid_distance: float, radius: float) -> float:
    """The neighbourhood score at one grid distance, which is zero at no reach."""
    if radius <= 0.0:
        return 0.0
    return float(math.exp(-(grid_distance**2) / (2.0 * radius**2)))


class AblationRequest(BaseModel):
    """The people, the grid, and how long each of the four walks runs."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class AblationRun(BaseModel):
    """One of the four walks, and what it left behind.

    ``rate_decays`` and ``radius_decays`` say which of the two curves the walk
    was allowed. ``final_movement`` is the furthest any unit moved on the last
    epoch, which says whether the map settled, and ``correlation`` is the
    topology claim, which says whether it arranged itself.
    """

    name: str
    rate_decays: bool
    radius_decays: bool
    final_movement: float
    quantisation_error: float
    correlation: float | None
    units: list[MapUnitDocument]


class AblationResponse(BaseModel):
    """The same people fitted four times, one curve held still at a time."""

    runs: list[AblationRun]


@router.post("/concepts/self-organising-map/ablation", response_model=AblationResponse)
def ablation(request: AblationRequest) -> AblationResponse:
    """Refit the people with each of the two decays held still in turn."""
    held_rate = ConstantSchedule(value=LEARNING_RATE.start)
    decaying_radius = _radius_schedule(request.grid_width, request.grid_height)
    held_radius = ConstantSchedule(value=decaying_radius.start)

    settings = (
        ("both shrink", True, True, LEARNING_RATE, decaying_radius),
        ("the step held at a half", False, True, held_rate, decaying_radius),
        ("the reach held wide", True, False, LEARNING_RATE, held_radius),
        ("neither shrinks", False, False, held_rate, held_radius),
    )

    runs: list[AblationRun] = []
    for name, rate_decays, radius_decays, rate, radius in settings:
        model = _fitted(
            request.points,
            request.grid_width,
            request.grid_height,
            request.max_epochs,
            rate,
            radius,
        )
        units = _unit_documents(model)
        winners = _winning_indices(model, request.points)
        runs.append(
            AblationRun(
                name=name,
                rate_decays=rate_decays,
                radius_decays=radius_decays,
                final_movement=float(model.final_epoch_movement),
                quantisation_error=_quantisation_error(request.points, units, winners),
                correlation=_topology_correlation(_unit_pairs(units)),
                units=units,
            )
        )

    return AblationResponse(runs=runs)


class TopologyRequest(BaseModel):
    """The people, the grid, and how long both walks run."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class TopologyResponse(BaseModel):
    """The same people fitted with the neighbourhood and without it.

    The second fit is the control. It quantises the people just as well and
    carries no arrangement at all, so its correlation is noise around zero
    where the first fit's is near one.
    """

    organised: MapReport
    without_neighbourhood: MapReport


@router.post("/concepts/self-organising-map/topology", response_model=TopologyResponse)
def topology(request: TopologyRequest) -> TopologyResponse:
    """Fit twice, with the reach and without it, and pair up the distances."""
    organised = _fitted(
        request.points,
        request.grid_width,
        request.grid_height,
        request.max_epochs,
        LEARNING_RATE,
        _radius_schedule(request.grid_width, request.grid_height),
    )
    without = _fitted(
        request.points,
        request.grid_width,
        request.grid_height,
        request.max_epochs,
        LEARNING_RATE,
        ConstantSchedule(value=0.0),
    )

    return TopologyResponse(
        organised=_map_report(organised, request.points),
        without_neighbourhood=_map_report(without, request.points),
    )


class VersusKMeansRequest(BaseModel):
    """The people, how many units the chain holds, and how long the walk runs."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_units: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class VersusKMeansResponse(BaseModel):
    """A chain with its reach switched off, beside the grouping method itself.

    ``same_partition`` compares which people were put together rather than
    which numbers they were given, because nothing ever told either model
    which group deserved which label. ``largest_prototype_gap`` matches each
    unit to its nearest centre and reports the furthest of those distances,
    and ``matched_one_to_one`` says whether that matching used every centre
    once, which is what makes the gap worth reading.
    """

    map_labels: list[int]
    kmeans_labels: list[int]
    same_partition: bool
    map_units: list[PlainPoint]
    kmeans_centres: list[PlainPoint]
    largest_prototype_gap: float
    matched_one_to_one: bool
    map_quantisation_error: float
    kmeans_quantisation_error: float
    with_reach_labels: list[int]
    with_reach_same_partition: bool


@router.post(
    "/concepts/self-organising-map/versus-k-means", response_model=VersusKMeansResponse
)
def versus_k_means(request: VersusKMeansRequest) -> VersusKMeansResponse:
    """Fit a chain with no reach and the grouping method on the same people."""
    if request.n_units > len(request.points):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{request.n_units} groups were asked for but only "
                f"{len(request.points)} people are here, and the grouping method "
                "refuses a group it can find nobody for"
            ),
        )

    features = _features(request.points)
    without_reach = _fitted(
        request.points,
        request.n_units,
        1,
        request.max_epochs,
        LEARNING_RATE,
        ConstantSchedule(value=0.0),
    )
    with_reach = _fitted(
        request.points,
        request.n_units,
        1,
        request.max_epochs,
        LEARNING_RATE,
        _radius_schedule(request.n_units, 1),
    )
    grouping = KMeans(n_clusters=request.n_units, random_seed=FIXED_SEED).fit(features)

    map_units = _unit_documents(without_reach)
    map_labels = _winning_indices(without_reach, request.points)
    reach_labels = _winning_indices(with_reach, request.points)
    kmeans_labels = [
        int(value) for value in np.asarray(grouping.predict(features), dtype=np.float64)
    ]
    centres = [
        PlainPoint(
            x=float(centroid.coordinate_for("height")),
            y=float(centroid.coordinate_for("weight")),
        )
        for centroid in grouping.centroids
    ]

    positions = np.asarray([[unit.x, unit.y] for unit in map_units])
    centre_positions = np.asarray([[centre.x, centre.y] for centre in centres])
    gaps = np.linalg.norm(positions[:, None, :] - centre_positions[None, :, :], axis=2)
    nearest = np.argmin(gaps, axis=1)
    kmeans_error = float(
        np.mean(
            np.linalg.norm(
                _matrix(request.points) - centre_positions[np.asarray(kmeans_labels)],
                axis=1,
            )
        )
    )

    return VersusKMeansResponse(
        map_labels=map_labels,
        kmeans_labels=kmeans_labels,
        same_partition=_same_partition(map_labels, kmeans_labels),
        map_units=[PlainPoint(x=unit.x, y=unit.y) for unit in map_units],
        kmeans_centres=centres,
        largest_prototype_gap=float(np.max(gaps[np.arange(len(map_units)), nearest])),
        matched_one_to_one=len({int(value) for value in nearest}) == len(map_units),
        map_quantisation_error=_quantisation_error(
            request.points, map_units, map_labels
        ),
        kmeans_quantisation_error=kmeans_error,
        with_reach_labels=reach_labels,
        with_reach_same_partition=_same_partition(reach_labels, kmeans_labels),
    )


class SeedsRequest(BaseModel):
    """The people, the grid, the epoch budget, and how many seeds to try."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    n_seeds: int = Field(default=4, ge=1, le=MAX_SEEDS)

    model_config = {"extra": "forbid"}


class SeededMap(BaseModel):
    """One seeded fit, and how it compares with the first.

    ``reading_order`` lists the units by index, shortest resting height first,
    so a fit that laid the same arrangement out backwards shows up as the
    reversal of another fit's list rather than as a disagreement about the
    people.
    """

    seed: int
    units: list[MapUnitDocument]
    labels: list[int]
    same_partition_as_first: bool
    identical_to_first: bool
    correlation: float | None
    quantisation_error: float
    reading_order: list[int]


class SeedsResponse(BaseModel):
    """The same people fitted from each seed in turn."""

    fits: list[SeededMap]


@router.post("/concepts/self-organising-map/seeds", response_model=SeedsResponse)
def seeds(request: SeedsRequest) -> SeedsResponse:
    """Fit the same people from several seeds and compare the arrangements."""
    radius = _radius_schedule(request.grid_width, request.grid_height)

    fits: list[SeededMap] = []
    first_labels: list[int] = []
    first_units: list[MapUnitDocument] = []
    for seed in range(request.n_seeds):
        model = _fitted(
            request.points,
            request.grid_width,
            request.grid_height,
            request.max_epochs,
            LEARNING_RATE,
            radius,
            seed=seed,
        )
        units = _unit_documents(model)
        labels = _winning_indices(model, request.points)
        if not fits:
            first_labels, first_units = labels, units
        fits.append(
            SeededMap(
                seed=seed,
                units=units,
                labels=labels,
                same_partition_as_first=_same_partition(labels, first_labels),
                identical_to_first=all(
                    math.isclose(here.x, there.x, abs_tol=1e-9)
                    and math.isclose(here.y, there.y, abs_tol=1e-9)
                    for here, there in zip(units, first_units, strict=True)
                ),
                correlation=_topology_correlation(_unit_pairs(units)),
                quantisation_error=_quantisation_error(request.points, units, labels),
                reading_order=[
                    index
                    for index, _ in sorted(enumerate(units), key=lambda pair: pair[1].x)
                ],
            )
        )

    return SeedsResponse(fits=fits)


# --- The edges the page's contract table quotes ----------------------------

EDGE_CROWD = [
    PlainPoint(x=147, y=41),
    PlainPoint(x=156, y=53),
    PlainPoint(x=145, y=57),
    PlainPoint(x=159, y=57),
    PlainPoint(x=162, y=61),
    PlainPoint(x=120, y=25),
    PlainPoint(x=122, y=28),
    PlainPoint(x=118, y=24),
    PlainPoint(x=180, y=80),
    PlainPoint(x=183, y=83),
    PlainPoint(x=178, y=78),
]
"""The eleven people the rest of the page works on, with no labels."""

EDGE_CLUMPS = [
    PlainPoint(x=118, y=24),
    PlainPoint(x=120, y=25),
    PlainPoint(x=122, y=28),
    PlainPoint(x=124, y=27),
    PlainPoint(x=178, y=78),
    PlainPoint(x=180, y=80),
    PlainPoint(x=182, y=83),
    PlainPoint(x=184, y=79),
]
"""The eight people in two clumps, whose two clump means are whole numbers."""

EDGE_EPOCHS = 100
"""The walk length every edge is probed at, which is the page's own budget."""

GRAND_MEAN = (151.0, 53.0)
"""The mean of the eight people in two clumps, which a one-cell map rests on."""

FLAT_COLUMN_VALUE = 60.0
"""The value the column that never changes holds at every row."""


class EdgeOutcome(BaseModel):
    """One edge run against the library, and what came back.

    ``accepted`` is whether the library answered at all; ``detail`` is either
    the refusal's own message or a description of what was accepted; and
    ``figure`` is the one number the page quotes from that case, where there
    is one.
    """

    edge: str
    accepted: bool
    detail: str
    figure: float | None


class EdgesResponse(BaseModel):
    """Every edge the page's contract table quotes, probed live."""

    outcomes: list[EdgeOutcome]


NOT_FITTED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z]* must be fit before this is available"
)
"""A refusal that opens on the class name of whatever was asked too early."""

PLAIN_WORDS = (("feature_values", "a column\u2019s values"),)
"""Identifiers a reader has no reason to meet, and what to say instead."""


def _in_plain_words(sentence: str) -> str:
    """One refusal's sentence, with the identifiers taken out of it.

    The website shows these sentences to somebody learning what a map is, and
    a class name or a parameter name means nothing to them. Most of the
    messages are already written to be read and are left alone; these are the
    two that name something internal. A test asserts that no identifier
    survives, so a message that grows one later fails there rather than
    reaching a reader.
    """
    sentence = NOT_FITTED_PATTERN.sub(
        "a map must be fitted before it can answer", sentence
    )
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _outcome(edge: str, attempt: Callable[[], tuple[str, float | None]]) -> EdgeOutcome:
    """Run one case and record what came of it.

    Two kinds of refusal are caught rather than one, because the model refuses
    on two levels. A grid extent or an epoch budget below its minimum is a
    field bound and so is pydantic's to reject, and everything about the data
    is the model's own. Both are the caller's request being wrong, and the
    table reads them the same way.
    """
    try:
        detail, figure = attempt()
    except MLLibError as error:
        return EdgeOutcome(
            edge=edge, accepted=False, detail=_in_plain_words(str(error)), figure=None
        )
    except ValidationError as error:
        return EdgeOutcome(
            edge=edge,
            accepted=False,
            detail=_in_plain_words(str(error.errors()[0]["msg"])),
            figure=None,
        )

    return EdgeOutcome(edge=edge, accepted=True, detail=detail, figure=figure)


def _widest_unit_gap(model: SelfOrganisingMap) -> float:
    """How far apart the two furthest-separated units of a fitted map rest."""
    weights = np.asarray([unit.weights for unit in model.units], dtype=np.float64)

    return float(np.max(np.linalg.norm(weights[:, None] - weights[None, :], axis=2)))


def _plain_fit(
    points: list[PlainPoint], width: int, height: int, radius: Schedule | None = None
) -> SelfOrganisingMap:
    """One fit at the page's own settings, with only the reach ever varied."""
    return _fitted(
        points,
        width,
        height,
        EDGE_EPOCHS,
        LEARNING_RATE,
        _radius_schedule(width, height) if radius is None else radius,
    )


def _edge_cases() -> list[EdgeOutcome]:
    """Every row of the page's contract table, run rather than remembered."""
    crowd = _features(EDGE_CROWD)
    clumps = _features(EDGE_CLUMPS)

    def more_cells_than_people() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CROWD, 6, 6)
        winners = set(_winning_indices(model, EDGE_CROWD))
        detail = (
            f"fitted; {len(winners)} of the 36 cells won somebody, and the rest "
            "came to rest along the ways between them"
        )
        return detail, float(len(winners))

    def more_groups_than_people() -> tuple[str, float | None]:
        KMeans(n_clusters=12, random_seed=FIXED_SEED).fit(crowd)
        return "fitted", None

    def one_cell() -> tuple[str, float | None]:
        unit = _unit_documents(_plain_fit(EDGE_CLUMPS, 1, 1))[0]
        detail = (
            f"fitted; the one cell rests at ({unit.x:.2f}, {unit.y:.2f}), which is "
            "the mean of everybody"
        )
        gap = math.hypot(unit.x - GRAND_MEAN[0], unit.y - GRAND_MEAN[1])
        return detail, float(gap)

    def no_epochs() -> tuple[str, float | None]:
        SelfOrganisingMap(grid_width=2, grid_height=1, max_epochs=0)
        return "constructed", None

    def reach_wider_than_the_grid() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CROWD, 4, 1, ConstantSchedule(value=MAX_RADIUS))
        detail = (
            "fitted, and every cell came to rest in one knot, since a reach that "
            "never shrinks moves every cell with every winner for the whole walk"
        )
        return detail, _widest_unit_gap(model)

    def reach_of_zero() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CROWD, 3, 1, ConstantSchedule(value=0.0))
        grouping = KMeans(n_clusters=3, random_seed=FIXED_SEED).fit(crowd)
        agrees = _same_partition(
            _winning_indices(model, EDGE_CROWD),
            [int(value) for value in np.asarray(grouping.predict(crowd))],
        )
        detail = (
            "fitted; only the winner ever moves, and on these eleven people it put "
            f"the same people together as the grouping method: {agrees}"
        )
        return detail, None

    def flat_column() -> tuple[str, float | None]:
        model = SelfOrganisingMap(
            grid_width=3,
            grid_height=1,
            learning_rate=LEARNING_RATE,
            neighbourhood_radius=_radius_schedule(3, 1),
            max_epochs=EDGE_EPOCHS,
            random_seed=FIXED_SEED,
        ).fit(
            [
                Feature("height", [160.0, 170.0, 180.0, 175.0]),
                Feature("weight", [FLAT_COLUMN_VALUE] * 4),
            ]
        )
        drift = max(
            abs(float(unit.weight_for("weight")) - FLAT_COLUMN_VALUE)
            for unit in model.units
        )
        detail = (
            "fitted; every cell rests at exactly 60 along the flat column, because "
            "the gap along it is zero at every presentation"
        )
        return detail, float(drift)

    def not_a_number() -> tuple[str, float | None]:
        SelfOrganisingMap(
            grid_width=2, grid_height=1, max_epochs=10, random_seed=FIXED_SEED
        ).fit(
            [
                Feature("height", [160.0, float("nan"), 180.0]),
                Feature("weight", [60.0, 61.0, 62.0]),
            ]
        )
        return "fitted", None

    def unfitted() -> tuple[str, float | None]:
        SelfOrganisingMap(grid_width=2, grid_height=1).predict(clumps)
        return "answered", None

    def unknown_column() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CLUMPS, 2, 1)
        model.predict([Feature("height", [170.0]), Feature("age", [30.0])])
        return "answered", None

    def missing_column() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CLUMPS, 2, 1)
        model.predict([Feature("height", [170.0])])
        return "answered", None

    def reordered_columns() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CLUMPS, 2, 1)
        forward = np.asarray(model.predict(clumps), dtype=np.float64)
        backward = np.asarray(model.predict(list(reversed(clumps))), dtype=np.float64)
        detail = (
            "answered, since the columns are matched by name rather than by the "
            "order they arrived in"
        )
        return detail, float(np.max(np.abs(forward - backward)))

    def one_name_twice() -> tuple[str, float | None]:
        model = _plain_fit(EDGE_CLUMPS, 2, 1)
        model.predict([Feature("height", [170.0]), Feature("height", [30.0])])
        return "answered", None

    def already_on_their_cells() -> tuple[str, float | None]:
        model = _plain_fit([PlainPoint(x=150, y=50)] * 3, 2, 1)
        detail = (
            f"fitted, and stopped after {model.epochs_run} epoch of the "
            f"{EDGE_EPOCHS} asked for, because nothing moved at all"
        )
        return detail, float(model.final_epoch_movement)

    return [
        _outcome(
            "a grid with more cells than there are people", more_cells_than_people
        ),
        _outcome(
            "twelve groups asked of the grouping method, for contrast",
            more_groups_than_people,
        ),
        _outcome("a grid of one cell", one_cell),
        _outcome("a walk of no epochs", no_epochs),
        _outcome("a reach wider than the grid, held there", reach_wider_than_the_grid),
        _outcome("a reach of zero", reach_of_zero),
        _outcome("a column that never changes", flat_column),
        _outcome("a value that is not a number", not_a_number),
        _outcome("an unfitted map asked to answer", unfitted),
        _outcome("a query carrying a column the fit never saw", unknown_column),
        _outcome("a query missing one of the fitted columns", missing_column),
        _outcome(
            "a query with the fitted columns in the other order", reordered_columns
        ),
        _outcome("a query carrying one name twice", one_name_twice),
        _outcome("people who already sit on their cells", already_on_their_cells),
    ]


@router.get("/concepts/self-organising-map/edges", response_model=EdgesResponse)
def edges() -> EdgesResponse:
    """Run every edge the page's contract table quotes and report what happened."""
    return EdgesResponse(outcomes=_edge_cases())
