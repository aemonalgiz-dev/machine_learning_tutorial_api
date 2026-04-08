"""One neuron, for the neurons-and-activations page.

A neuron is a weighted sum and a bend. The page puts two inputs on a plane,
lets the reader set the two weights and the bias, and asks the library's
``Neuron`` what it answers at every cell of a lattice over that plane, so the
shaded map is the neuron's actual response rather than a drawing of a formula.
Beside the map sits the bend itself, the chosen activation sampled across a
fixed range of scores together with its slope, because the slope is what a
gradient has to pass through on its way back and is the whole reason the four
bends on the list differ.

One endpoint. ``respond`` takes the weights, the bias, the activation's name,
one probe row and the lattice, and answers with the neuron's response at the
probe, its score and output at every lattice cell, the line along which the
score is zero, and the activation's curve and slope over the fixed range. The
probe is what the worked example quotes, weights (2, -1) and bias 0.5 at the
input (1, 1), which every activation scores at 1.5 and each then bends its own
way.

The zero line is reported because it is the one thing a single neuron knows
about the plane. The score is an affine function of the inputs, so it is zero
along a straight line and grows in proportion to the distance from that line,
and every bend on the list is monotone, so the output is a function of that
distance alone. Which side of the line, and how far, is all one neuron can say,
and the page's softmax section and the next page's layer both start from there.

Every cell of the surface and the probe itself go through ``Neuron.respond_to``
one row at a time, which is the public route and the one the page describes.
The curve goes through the activation's own ``of`` and ``derivative_at`` on the
sampled scores, since a curve is the bend asked about a score with no neuron in
front of it.
"""

from __future__ import annotations

from itertools import combinations
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Activation,
    HyperbolicTangent,
    Identity,
    Neuron,
    RectifiedLinear,
    Sigmoid,
)
from pydantic import BaseModel, Field

router = APIRouter(tags=["neurons-and-activations"])

MAX_PARAMETER = 10.0
"""How large a weight or the bias may be, in absolute value.

Ten is already enough to drive a sigmoid flat across the whole window the
widget draws, which is the saturation the page wants the reader to see, and a
teaching neuron has no use for anything larger.
"""

LATTICE_EXTENT = 10.0
"""How far from the origin the lattice window, and the probe, may reach."""

MIN_LATTICE_CELLS = 2
MAX_LATTICE_CELLS = 41
"""How finely the lattice may sample the plane, in cells per side.

Every cell is one call to the neuron, so the ceiling keeps a request at a
couple of thousand calls, which is a few milliseconds.
"""

CURVE_LOW = -6.0
CURVE_HIGH = 6.0
CURVE_SAMPLES = 121
"""The activation curve is sampled every tenth from -6 to 6.

Wide enough that a sigmoid has flattened to a hundredth of its peak slope at
either end, which is the vanishing the page names, and an odd count so that a
score of exactly zero is one of the samples and the sigmoid's peak slope of one
quarter is read off the curve rather than approached.
"""

ActivationName = Literal[
    "identity", "rectified_linear", "sigmoid", "hyperbolic_tangent"
]

ACTIVATIONS: dict[str, Activation] = {
    "identity": Identity(),
    "rectified_linear": RectifiedLinear(),
    "sigmoid": Sigmoid(),
    "hyperbolic_tangent": HyperbolicTangent(),
}
"""The four bends the library offers, by the name the page uses for each."""

Parameter = Annotated[float, Field(ge=-MAX_PARAMETER, le=MAX_PARAMETER)]


class PlanePoint(BaseModel):
    """One row of two inputs, which is a point on the plane the neuron reads."""

    first_input: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    second_input: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)

    model_config = {"extra": "forbid"}


class Lattice(BaseModel):
    """The window over the plane, and how finely to sample it."""

    first_input_low: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    first_input_high: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    second_input_low: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    second_input_high: float = Field(ge=-LATTICE_EXTENT, le=LATTICE_EXTENT)
    cells: int = Field(ge=MIN_LATTICE_CELLS, le=MAX_LATTICE_CELLS)

    model_config = {"extra": "forbid"}


class RespondRequest(BaseModel):
    """The neuron's three parameters, its bend, one probe row and the lattice."""

    weights: list[Parameter] = Field(min_length=2, max_length=2)
    bias: Parameter
    activation: ActivationName
    probe: PlanePoint
    lattice: Lattice

    model_config = {"extra": "forbid"}


class NeuronReading(BaseModel):
    """What the neuron did with one row, and the slope of the bend there.

    ``score`` is the weighted sum plus the bias, before the bend. ``output`` is
    the bend applied to it, and ``slope`` is the bend's derivative at that
    score, which is the factor a gradient arriving from above is multiplied by
    on its way through this neuron.
    """

    probe: PlanePoint
    score: float
    output: float
    slope: float


class ResponseSurface(BaseModel):
    """The neuron's score and output at every cell of the lattice.

    Both tables are row-major from the low corner, so ``scores[row][column]``
    is the score at the ``row``-th second input and the ``column``-th first
    input, counting up from the window's low edges.
    """

    first_input_low: float
    first_input_high: float
    second_input_low: float
    second_input_high: float
    cells: int
    scores: list[list[float]]
    outputs: list[list[float]]


class ZeroLine(BaseModel):
    """The line along which the score is zero, clipped to the lattice window."""

    start: PlanePoint
    end: PlanePoint


class ActivationCurve(BaseModel):
    """The bend and its slope, sampled over the fixed range of scores.

    ``peak_slope`` is the largest slope among the samples, which is the most a
    gradient can hope to keep on its way through one such bend.
    """

    scores: list[float]
    outputs: list[float]
    slopes: list[float]
    peak_slope: float


class RespondResponse(BaseModel):
    """Everything the page draws about one neuron with one bend."""

    activation: ActivationName
    formula: str
    reading: NeuronReading
    surface: ResponseSurface
    zero_line: ZeroLine | None
    curve: ActivationCurve


@router.post("/concepts/neurons/respond", response_model=RespondResponse)
def respond(request: RespondRequest) -> RespondResponse:
    """Ask one neuron about a probe row, a lattice, and its own bend."""
    lattice = request.lattice
    if lattice.first_input_low >= lattice.first_input_high:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the lattice's first input runs from {lattice.first_input_low} "
                f"to {lattice.first_input_high}, and a window has to open upward"
            ),
        )
    if lattice.second_input_low >= lattice.second_input_high:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the lattice's second input runs from {lattice.second_input_low} "
                f"to {lattice.second_input_high}, and a window has to open upward"
            ),
        )

    activation = ACTIVATIONS[request.activation]
    neuron = Neuron(request.weights, request.bias, activation)

    return RespondResponse(
        activation=request.activation,
        formula=activation.description,
        reading=_reading_at(neuron, request.probe),
        surface=_surface_over(neuron, lattice),
        zero_line=_zero_line(request.weights, request.bias, lattice),
        curve=_curve_of(activation),
    )


def _reading_at(neuron: Neuron, probe: PlanePoint) -> NeuronReading:
    """The neuron's response to one row, with the bend's slope at its score."""
    response = neuron.respond_to([probe.first_input, probe.second_input])
    slope = neuron.activation.derivative_at(np.asarray(response.score))

    return NeuronReading(
        probe=probe,
        score=float(response.score),
        output=float(response.output),
        slope=float(slope),
    )


def _surface_over(neuron: Neuron, lattice: Lattice) -> ResponseSurface:
    """The neuron asked at every cell of the lattice, one row at a time."""
    first_inputs = np.linspace(
        lattice.first_input_low, lattice.first_input_high, lattice.cells
    )
    second_inputs = np.linspace(
        lattice.second_input_low, lattice.second_input_high, lattice.cells
    )

    scores: list[list[float]] = []
    outputs: list[list[float]] = []
    for second_input in second_inputs:
        score_row: list[float] = []
        output_row: list[float] = []
        for first_input in first_inputs:
            response = neuron.respond_to([float(first_input), float(second_input)])
            score_row.append(float(response.score))
            output_row.append(float(response.output))
        scores.append(score_row)
        outputs.append(output_row)

    return ResponseSurface(
        first_input_low=float(lattice.first_input_low),
        first_input_high=float(lattice.first_input_high),
        second_input_low=float(lattice.second_input_low),
        second_input_high=float(lattice.second_input_high),
        cells=lattice.cells,
        scores=scores,
        outputs=outputs,
    )


def _zero_line(weights: list[float], bias: float, lattice: Lattice) -> ZeroLine | None:
    """The segment of the zero-score line that lies inside the window.

    The score is ``first_weight * first + second_weight * second + bias``, so
    it is zero along a straight line unless both weights are zero, when the
    score is the bias everywhere and there is no line to draw. The line is met
    where it crosses each of the window's four edges, and the two crossings
    inside the window, furthest apart, are the segment. A line through a
    corner crosses two edges at the same point, which is why the furthest pair
    is taken rather than the first two found, and a line that only touches
    the window at one corner has no length and is not drawn.
    """
    first_weight, second_weight = weights
    if first_weight == 0.0 and second_weight == 0.0:
        return None

    slack = 1e-9
    crossings: list[tuple[float, float]] = []

    if second_weight != 0.0:
        for first_input in (lattice.first_input_low, lattice.first_input_high):
            second_input = -(first_weight * first_input + bias) / second_weight
            if (
                lattice.second_input_low - slack
                <= second_input
                <= lattice.second_input_high + slack
            ):
                crossings.append((first_input, second_input))

    if first_weight != 0.0:
        for second_input in (lattice.second_input_low, lattice.second_input_high):
            first_input = -(second_weight * second_input + bias) / first_weight
            if (
                lattice.first_input_low - slack
                <= first_input
                <= lattice.first_input_high + slack
            ):
                crossings.append((first_input, second_input))

    if len(crossings) < 2:
        return None

    start, end = max(
        combinations(crossings, 2),
        key=lambda pair: (
            (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2
        ),
    )
    if start == end:
        return None

    return ZeroLine(
        start=_point_inside(start, lattice),
        end=_point_inside(end, lattice),
    )


def _point_inside(crossing: tuple[float, float], lattice: Lattice) -> PlanePoint:
    """A crossing a rounding error put a hair outside the window, put back."""
    first_input, second_input = crossing
    return PlanePoint(
        first_input=_clamped(
            first_input, lattice.first_input_low, lattice.first_input_high
        ),
        second_input=_clamped(
            second_input, lattice.second_input_low, lattice.second_input_high
        ),
    )


def _clamped(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _curve_of(activation: Activation) -> ActivationCurve:
    """The bend and its slope, sampled across the fixed range of scores."""
    scores = np.linspace(CURVE_LOW, CURVE_HIGH, CURVE_SAMPLES)
    outputs = activation.of(scores)
    slopes = activation.derivative_at(scores)

    return ActivationCurve(
        scores=[float(value) for value in scores],
        outputs=[float(value) for value in outputs],
        slopes=[float(value) for value in slopes],
        peak_slope=float(np.max(slopes)),
    )
