"""A self-organising map's resting state, for the map page.

One endpoint. It fits the library's ``SelfOrganisingMap`` on the request's
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

from fastapi import APIRouter
from oop_ml import (
    ExponentialDecaySchedule,
    Feature,
    LinearDecaySchedule,
    SelfOrganisingMap,
)
from pydantic import BaseModel, Field

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


class OrganiseRequest(BaseModel):
    """The people, the grid to drape over them, and how long to walk."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    grid_width: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    grid_height: int = Field(ge=MIN_GRID_SIDE, le=MAX_GRID_SIDE)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

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
        neighbourhood_radius=_radius_schedule(request.grid_width, request.grid_height),
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


def _radius_schedule(grid_width: int, grid_height: int) -> LinearDecaySchedule:
    """The neighbourhood radius, from half the longest side down to a quarter step.

    Linear rather than geometric, because a radius is measured in the grid's
    own steps and the midpoint of the walk should be the middle of the range.
    """
    return LinearDecaySchedule(start=max(grid_width, grid_height) / 2.0, end=RADIUS_END)


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
