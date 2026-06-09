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

Four more endpoints, each for one claim the page measures rather than states.
``logistic-twin`` standardises the classification pages' crowd of twenty-five
people, fits the library's ``LogisticRegression`` on the two standardised
columns, builds a ``Neuron`` from the fitted weights and bias with a sigmoid
bend, and reports the two routes' probabilities side by side, because a sigmoid
neuron is that model and the page wants the gap read rather than asserted.
``crowd-census`` asks a neuron of the reader's choosing about every person in
the same crowd and counts how many of them land where the bend's slope is
exactly zero or under a hundredth of its peak, which is where saturation and
dead units stop being adjectives. ``chain`` feeds one neuron's output into a
second and fits a plane through the composite surface with the library's
``MultipleLinearRegression``, so that the collapse of two bendless neurons into
one is an R squared of exactly 1.0 and the difference a bend makes is a
residual. ``softmax-contrast`` is the one place softmax appears, moving one
score of three and reporting what happened to the others, which is the
property no elementwise bend can have.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Activation,
    Feature,
    HyperbolicTangent,
    Identity,
    LogisticRegression,
    MultipleLinearRegression,
    Neuron,
    RectifiedLinear,
    Sigmoid,
    Standardizer,
)
from oop_ml.core.exceptions import UndefinedMetricError
from oop_ml.core.logistic import stable_softmax
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
    _check_window_opens_upward(lattice)

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


CROWD: tuple[tuple[float, float, int], ...] = (
    (147, 41, 0),
    (156, 53, 1),
    (145, 57, 0),
    (159, 57, 1),
    (162, 61, 1),
    (120, 25, 0),
    (122, 28, 0),
    (118, 24, 0),
    (180, 80, 1),
    (183, 83, 1),
    (178, 78, 1),
    (145, 45, 0),
    (145, 55, 1),
    (151, 45, 1),
    (151, 55, 0),
    (157, 45, 0),
    (157, 55, 1),
    (148, 50, 1),
    (154, 50, 0),
    (160, 50, 1),
    (147, 58, 0),
    (153, 58, 1),
    (150, 42, 1),
    (156, 42, 0),
    (143, 50, 0),
)
"""The classification pages' crowd: height in centimetres, weight in kilograms,
and whether the person is an adult.

The same twenty-five people the bagging and random-forest pages grow trees on,
kept here as a fixed fixture because the page's neuron reads them in
standardised units and the standardisation is part of what is measured.
"""

STANDARD_WINDOW = Lattice(
    first_input_low=-3.0,
    first_input_high=3.0,
    second_input_low=-3.0,
    second_input_high=3.0,
    cells=25,
)
"""The window the page draws the crowd in, three standard deviations each way."""

CENSUS_MAX_PARAMETER = 100.0
"""How large a census neuron's weights and bias may be.

Ten times the playground's ceiling, because the census exists to show what a
sigmoid does when its weights are scaled up until the crowd sits on its flat
ends, and the fitted weights are already above two.
"""

CensusParameter = Annotated[
    float, Field(ge=-CENSUS_MAX_PARAMETER, le=CENSUS_MAX_PARAMETER)
]

SATURATION_SHARE = 0.01
"""A slope under this share of the bend's peak counts as saturated.

One hundredth is where the sigmoid has reached by a score of six, which is the
edge of the curve the page draws, so the count says how many people have been
pushed off the picture.
"""

SOFTMAX_SCORES = (2.0, 1.0, 0.1)
SOFTMAX_MOVED_SCORES = (2.0, 1.0, 5.0)
"""Three scores, and the same three with only the last one moved."""


class CrowdPerson(BaseModel):
    """One person of the crowd, in the units measured and in standard units."""

    height: float
    weight: float
    is_adult: int
    standardised_height: float
    standardised_weight: float


class TwinPerson(CrowdPerson):
    """One person scored by the logistic model and by the neuron built from it.

    ``model_probability`` is what the fitted logistic model answers,
    ``neuron_output`` what a neuron carrying its weights and bias under a
    sigmoid bend answers to the same standardised row, and ``gap`` the absolute
    difference. ``predicted_adult`` is the model's call at a threshold of one
    half.
    """

    score: float
    model_probability: float
    neuron_output: float
    gap: float
    predicted_adult: int


class LogisticTwinResponse(BaseModel):
    """The logistic model fitted to the crowd, and the neuron that equals it.

    ``fitted_weights`` are the model's coefficients on standardised height and
    standardised weight, in that order, and ``fitted_bias`` its intercept, both
    inside the playground's parameter range so the page can load them. The
    means and deviations are what the standardisation subtracted and divided
    by. ``zero_line`` is where the fitted score is zero across the standard
    window, which is the boundary where the probability is one half.
    """

    mean_height: float
    deviation_height: float
    mean_weight: float
    deviation_weight: float
    fitted_weights: list[float]
    fitted_bias: float
    epochs_run: int
    converged: bool
    accuracy: float
    n_correct: int
    n_people: int
    largest_gap: float
    zero_line: ZeroLine | None
    people: list[TwinPerson]


class _StandardisedCrowd:
    """The crowd's two columns standardised once, with the labels beside them."""

    __slots__ = ("labels", "raw", "scaler", "standardised")

    def __init__(self) -> None:
        heights = Feature("height", [float(row[0]) for row in CROWD])
        weights = Feature("weight", [float(row[1]) for row in CROWD])
        self.raw = [heights, weights]
        self.labels = Feature("is_adult", [float(row[2]) for row in CROWD])
        self.scaler = Standardizer().fit(self.raw)
        self.standardised = self.scaler.transform(self.raw)

    def row(self, index: int) -> list[float]:
        """One person's standardised height and weight, in that order."""
        return [
            float(self.standardised[0].values[index]),
            float(self.standardised[1].values[index]),
        ]

    def person(self, index: int) -> CrowdPerson:
        standardised = self.row(index)
        return CrowdPerson(
            height=float(CROWD[index][0]),
            weight=float(CROWD[index][1]),
            is_adult=CROWD[index][2],
            standardised_height=standardised[0],
            standardised_weight=standardised[1],
        )


@cache
def _standardised_crowd() -> _StandardisedCrowd:
    return _StandardisedCrowd()


@cache
def _logistic_twin() -> LogisticTwinResponse:
    crowd = _standardised_crowd()
    model = LogisticRegression().fit(crowd.standardised, crowd.labels)
    fitted_weights = [float(coefficient.value) for coefficient in model.coefficients]
    fitted_bias = float(model.intercept)
    neuron = Neuron(fitted_weights, fitted_bias, Sigmoid())

    probabilities = np.asarray(
        model.predict_probability(crowd.standardised), dtype=np.float64
    )
    calls = np.asarray(model.predict(crowd.standardised), dtype=np.float64)

    people: list[TwinPerson] = []
    for index in range(len(CROWD)):
        response = neuron.respond_to(crowd.row(index))
        people.append(
            TwinPerson(
                **crowd.person(index).model_dump(),
                score=float(response.score),
                model_probability=float(probabilities[index]),
                neuron_output=float(response.output),
                gap=abs(float(response.output) - float(probabilities[index])),
                predicted_adult=int(calls[index]),
            )
        )

    n_correct = int(sum(person.predicted_adult == person.is_adult for person in people))
    height_scaling = crowd.scaler.scalings.scaling_for("height")
    weight_scaling = crowd.scaler.scalings.scaling_for("weight")

    return LogisticTwinResponse(
        mean_height=float(height_scaling.mean),
        deviation_height=float(height_scaling.standard_deviation),
        mean_weight=float(weight_scaling.mean),
        deviation_weight=float(weight_scaling.standard_deviation),
        fitted_weights=fitted_weights,
        fitted_bias=fitted_bias,
        epochs_run=int(model.epochs_run),
        converged=bool(model.converged),
        accuracy=float(model.score(crowd.standardised, crowd.labels)),
        n_correct=n_correct,
        n_people=len(CROWD),
        largest_gap=max(person.gap for person in people),
        zero_line=_zero_line(fitted_weights, fitted_bias, STANDARD_WINDOW),
        people=people,
    )


@router.get("/concepts/neurons/logistic-twin", response_model=LogisticTwinResponse)
def logistic_twin() -> LogisticTwinResponse:
    """Fit the logistic model to the crowd and read the same neuron off it."""
    return _logistic_twin()


class CensusRequest(BaseModel):
    """A neuron of any weights, to be shown every person in the crowd."""

    weights: list[CensusParameter] = Field(min_length=2, max_length=2)
    bias: CensusParameter
    activation: ActivationName

    model_config = {"extra": "forbid"}


class CensusPerson(CrowdPerson):
    """One person's score, output and the bend's slope at that score."""

    score: float
    output: float
    slope: float


class CensusResponse(BaseModel):
    """How much of the crowd a neuron has stopped learning from.

    ``n_dead`` counts people at whose score the slope is exactly zero, which
    only the rectifier produces. ``n_saturated`` counts people at whose score
    the slope is under ``saturation_threshold``, one hundredth of the bend's
    peak, and it includes the dead. The identity saturates nowhere and is never
    dead, so both counts are zero for it whatever the weights.
    """

    activation: ActivationName
    peak_slope: float
    saturation_threshold: float
    n_people: int
    n_dead: int
    n_saturated: int
    smallest_slope: float
    largest_slope: float
    people: list[CensusPerson]


@router.post("/concepts/neurons/crowd-census", response_model=CensusResponse)
def crowd_census(request: CensusRequest) -> CensusResponse:
    """Show one neuron every person in the crowd and count the flat ones."""
    crowd = _standardised_crowd()
    activation = ACTIVATIONS[request.activation]
    neuron = Neuron(request.weights, request.bias, activation)
    peak_slope = _curve_of(activation).peak_slope
    threshold = SATURATION_SHARE * peak_slope

    people: list[CensusPerson] = []
    for index in range(len(CROWD)):
        response = neuron.respond_to(crowd.row(index))
        slope = float(activation.derivative_at(np.asarray(response.score)))
        people.append(
            CensusPerson(
                **crowd.person(index).model_dump(),
                score=float(response.score),
                output=float(response.output),
                slope=slope,
            )
        )

    slopes = [person.slope for person in people]
    return CensusResponse(
        activation=request.activation,
        peak_slope=peak_slope,
        saturation_threshold=threshold,
        n_people=len(CROWD),
        n_dead=sum(slope == 0.0 for slope in slopes),
        n_saturated=sum(slope < threshold for slope in slopes),
        smallest_slope=min(slopes),
        largest_slope=max(slopes),
        people=people,
    )


class ChainRequest(BaseModel):
    """Two neurons in a chain, the first reading the plane and the second
    reading the first's output through one bend.

    The second neuron has one weight because it reads one number. Its own
    bend is the identity, so what the plane fit measures is only what the
    bend between the two did.
    """

    first_weights: list[Parameter] = Field(min_length=2, max_length=2)
    first_bias: Parameter
    between: ActivationName
    second_weight: Parameter
    second_bias: Parameter
    probe: PlanePoint
    lattice: Lattice

    model_config = {"extra": "forbid"}


class PlaneFit(BaseModel):
    """The least-squares plane through the chain's surface.

    ``r_squared`` is ``None`` when the surface is flat, since a flat surface
    leaves nothing for a plane to explain and the score is undefined rather
    than perfect. ``largest_residual`` is the furthest any cell sits from the
    plane, which is zero exactly when the chain is one neuron in disguise.
    """

    height_weight: float
    weight_weight: float
    intercept: float
    r_squared: float | None
    largest_residual: float


class ChainReading(BaseModel):
    """The probe row followed through both neurons, and through the collapse.

    ``collapsed_output`` is what the single bendless neuron with the
    collapsed weights answers at the probe, which agrees with ``output``
    only when the bend between is the identity.
    """

    probe: PlanePoint
    first_score: float
    first_output: float
    second_score: float
    output: float
    collapsed_output: float


class ChainResponse(BaseModel):
    """Everything the page draws about two neurons in a chain.

    ``collapsed_weights`` and ``collapsed_bias`` are the second weight times
    the first neuron's parameters, plus the second bias, which is the single
    neuron the chain equals when nothing bends between them. ``outputs`` is
    the chain's answer at every lattice cell, row-major from the low corner
    like the response surface.
    """

    between: ActivationName
    collapsed_weights: list[float]
    collapsed_bias: float
    cells: int
    outputs: list[list[float]]
    reading: ChainReading
    plane: PlaneFit


@router.post("/concepts/neurons/chain", response_model=ChainResponse)
def chain(request: ChainRequest) -> ChainResponse:
    """Feed one neuron into another and fit a plane through what comes out."""
    lattice = request.lattice
    _check_window_opens_upward(lattice)

    first = Neuron(
        request.first_weights, request.first_bias, ACTIVATIONS[request.between]
    )
    second = Neuron([request.second_weight], request.second_bias, Identity())
    collapsed_weights = [
        request.second_weight * weight for weight in request.first_weights
    ]
    collapsed_bias = request.second_weight * request.first_bias + request.second_bias
    collapsed = Neuron(collapsed_weights, collapsed_bias, Identity())

    first_inputs = np.linspace(
        lattice.first_input_low, lattice.first_input_high, lattice.cells
    )
    second_inputs = np.linspace(
        lattice.second_input_low, lattice.second_input_high, lattice.cells
    )
    outputs: list[list[float]] = []
    heights: list[float] = []
    weights: list[float] = []
    flat_outputs: list[float] = []
    for second_input in second_inputs:
        row: list[float] = []
        for first_input in first_inputs:
            middle = first.respond_to([float(first_input), float(second_input)])
            answer = second.respond_to([middle.output])
            row.append(float(answer.output))
            heights.append(float(first_input))
            weights.append(float(second_input))
            flat_outputs.append(float(answer.output))
        outputs.append(row)

    probe_row = [request.probe.first_input, request.probe.second_input]
    probe_middle = first.respond_to(probe_row)
    probe_answer = second.respond_to([probe_middle.output])
    probe_collapsed = collapsed.respond_to(probe_row)

    return ChainResponse(
        between=request.between,
        collapsed_weights=collapsed_weights,
        collapsed_bias=collapsed_bias,
        cells=lattice.cells,
        outputs=outputs,
        reading=ChainReading(
            probe=request.probe,
            first_score=float(probe_middle.score),
            first_output=float(probe_middle.output),
            second_score=float(probe_answer.score),
            output=float(probe_answer.output),
            collapsed_output=float(probe_collapsed.output),
        ),
        plane=_plane_through(heights, weights, flat_outputs),
    )


def _check_window_opens_upward(lattice: Lattice) -> None:
    """Refuse a window whose low edge is not below its high edge, in words."""
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


def _plane_through(
    heights: list[float], weights: list[float], outputs: list[float]
) -> PlaneFit:
    """The library's least-squares plane through the surface, and its misses."""
    features = [Feature("height", heights), Feature("weight", weights)]
    target = Feature("output", outputs)
    plane = MultipleLinearRegression().fit(features, target)
    predicted = np.asarray(plane.predict(features), dtype=np.float64)
    residuals = np.abs(predicted - np.asarray(outputs, dtype=np.float64))
    coefficients = {
        coefficient.name: float(coefficient.value) for coefficient in plane.coefficients
    }

    r_squared: float | None
    try:
        r_squared = float(plane.score(features, target))
    except UndefinedMetricError:
        r_squared = None

    return PlaneFit(
        height_weight=coefficients["height"],
        weight_weight=coefficients["weight"],
        intercept=float(plane.intercept),
        r_squared=r_squared,
        largest_residual=float(np.max(residuals)),
    )


class SoftmaxRow(BaseModel):
    """Three scores and what softmax makes of them together."""

    scores: list[float]
    outputs: list[float]


class SoftmaxContrastResponse(BaseModel):
    """One score moved, and what happened to the outputs that did not move.

    ``first_output_ratio`` is the first output before over the first output
    after, which is how many times an unchanged neuron's answer fell because
    a neighbour's score rose.
    """

    before: SoftmaxRow
    after: SoftmaxRow
    first_output_ratio: float


@router.get(
    "/concepts/neurons/softmax-contrast", response_model=SoftmaxContrastResponse
)
def softmax_contrast() -> SoftmaxContrastResponse:
    """Move one of three scores and read what softmax did to the other two."""
    before = stable_softmax(np.asarray([SOFTMAX_SCORES], dtype=np.float64))[0]
    after = stable_softmax(np.asarray([SOFTMAX_MOVED_SCORES], dtype=np.float64))[0]
    return SoftmaxContrastResponse(
        before=SoftmaxRow(
            scores=list(SOFTMAX_SCORES),
            outputs=[float(value) for value in before],
        ),
        after=SoftmaxRow(
            scores=list(SOFTMAX_MOVED_SCORES),
            outputs=[float(value) for value in after],
        ),
        first_output_ratio=float(before[0] / after[0]),
    )
