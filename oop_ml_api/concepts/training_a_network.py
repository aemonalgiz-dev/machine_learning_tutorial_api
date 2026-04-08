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
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    BinaryCrossEntropy,
    DenseLayer,
    Feature,
    HyperbolicTangent,
    Identity,
    LayerStack,
    LogisticRegression,
    Neuron,
)
from pydantic import BaseModel, Field

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
"""The largest stride. Above this the walk runs away, which the gradient
descent page is the place to explore."""

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
