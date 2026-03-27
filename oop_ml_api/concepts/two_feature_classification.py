"""The shapes the two-feature classifier pages share.

Both k-nearest neighbours and the decision tree classify people by height and
weight, so both endpoints take the same labelled points and both hand back the
same picture of their decision, a lattice of predictions over the plane that
the widgets shade as regions. The lattice is predicted by the fitted model
itself, one batch call through the library, so the shading is the model's
actual answer at every cell rather than a re-implementation of it.
"""

from __future__ import annotations

import numpy as np
from oop_ml import Feature
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE

GRID_CELLS = 26
"""The lattice is this many cells on each side."""


class LabelledPoint(BaseModel):
    """One person, their two measurements, and which class they belong to."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=1)


class RegionGrid(BaseModel):
    """The model's answer at every cell of a lattice over the plane.

    ``labels`` is row-major from the bottom-left, each entry the class the
    fitted model predicts at that cell's centre.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cells: int
    labels: list[list[int]]


def features_of(points: list[LabelledPoint]) -> tuple[Feature, Feature, Feature]:
    """The three columns the library models take, named for the pages' example."""
    heights = Feature("height", [point.x for point in points])
    weights = Feature("weight", [point.y for point in points])
    labels = Feature("is_adult", [float(point.label) for point in points])
    return heights, weights, labels


def region_grid(model, points: list[LabelledPoint]) -> RegionGrid:
    """Ask the fitted model for its answer across the plane the points span.

    The window pads the data's extent a little so every point sits inside it.
    """
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
    y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad

    x_centres = np.linspace(x_min, x_max, GRID_CELLS)
    y_centres = np.linspace(y_min, y_max, GRID_CELLS)
    lattice_x, lattice_y = np.meshgrid(x_centres, y_centres)

    predictions = np.asarray(
        model.predict(
            [
                Feature("height", lattice_x.ravel()),
                Feature("weight", lattice_y.ravel()),
            ]
        ),
        dtype=np.float64,
    ).reshape(GRID_CELLS, GRID_CELLS)

    return RegionGrid(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        cells=GRID_CELLS,
        labels=[[int(value) for value in row] for row in predictions],
    )
