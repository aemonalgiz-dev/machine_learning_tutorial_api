"""One backward pass through a 2-3-1 network, for the backpropagation page.

The network is the dense-layers page's, three hidden neurons reading a row of
two numbers and one linear neuron reading those three, scored by squared error
on a single row against a single target. The endpoint runs the row forward,
measures the loss, walks the blame back down the two layers, takes one step
against it, and measures the loss again on the stepped network. Everything the
widget writes on a node or an edge is reported, layer by layer, in the order
the stack holds its layers.

Two walks, one arithmetic. ``LayerStack.backward_pass`` is the library's route
and the step is taken from what it returns, but it keeps only each layer's
gradient. The page also wants what each layer *passed down*, the slope of the
loss at that layer's inputs, which is the arriving block for the layer beneath
and the thing the induction in the derivation is about. So the endpoint runs
the stack's own four lines a second time over the public ``correction_for`` of
each layer, threading ``arriving`` downward exactly as the stack does, and
reports both halves of every correction. The gradients agree with the library's
route because they are the same calls in the same order.

The finite-difference check is arithmetic on library answers. The endpoint
rebuilds the chosen layer with one weight nudged up, then down, through
``with_parameters``, measures the loss on each, and divides the difference by
the width of the nudge. A central difference is exact on a quadratic, and the
squared-error loss is a quadratic in any one weight of a linear output layer,
so on the page's worked example the two figures agree to rounding and the page
can print the disagreement rather than assert it.

One row throughout, so the loss's division by the row count is a division by
one, and every slope the page works by hand is the slope the library reports.

The output neuron is always linear. The library's losses report their slope at
the last layer's raw output, and the classification losses apply their own
squash on the way, which is the canonical-link argument the loss-functions
page makes. A bend on the output neuron here would have its slope applied
twice, so the request chooses the hidden bend and nothing else.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Activation,
    DenseLayer,
    HyperbolicTangent,
    Identity,
    LayerStack,
    Neuron,
    PassPurpose,
    RectifiedLinear,
    Sigmoid,
    SquaredError,
)
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE

router = APIRouter(tags=["backpropagation"])

INPUT_WIDTH = 2
"""How many numbers the row holds, and how many weights each hidden neuron has."""

HIDDEN_WIDTH = 3
"""How many neurons the hidden layer holds, which is what the output neuron reads."""

OUTPUT_WIDTH = 1
"""How many neurons the output layer holds. One, so the network answers a number."""

HIDDEN_LAYER = 0
"""The hidden layer's position in the stack, counting from the input end."""

OUTPUT_LAYER = 1
"""The output layer's position in the stack."""

MAX_LEARNING_RATE = 1.0
"""The largest step the widget may ask for. Past one the walk is a leap."""

FINITE_DIFFERENCE_EPSILON = 1e-6
"""Half the width of the nudge the finite-difference check uses.

Small enough that the curvature past the quadratic term is negligible, large
enough that the difference of two losses near eight still stands well above
rounding. On the worked example the two figures disagree by about 2e-9.
"""


class HiddenActivation(StrEnum):
    """Which bend the hidden neurons carry. The output neuron is always linear."""

    IDENTITY = "identity"
    RECTIFIED_LINEAR = "rectified_linear"
    SIGMOID = "sigmoid"
    HYPERBOLIC_TANGENT = "hyperbolic_tangent"


ACTIVATIONS: dict[HiddenActivation, Activation] = {
    HiddenActivation.IDENTITY: Identity(),
    HiddenActivation.RECTIFIED_LINEAR: RectifiedLinear(),
    HiddenActivation.SIGMOID: Sigmoid(),
    HiddenActivation.HYPERBOLIC_TANGENT: HyperbolicTangent(),
}
"""One instance per name. The bends carry no state, so sharing them is safe."""


BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]
"""One hand-entered number, inside the API's magnitude bound."""

InputRow = Annotated[
    list[BoundedValue], Field(min_length=INPUT_WIDTH, max_length=INPUT_WIDTH)
]

HiddenWeightRow = Annotated[
    list[BoundedValue], Field(min_length=INPUT_WIDTH, max_length=INPUT_WIDTH)
]

OutputWeightRow = Annotated[
    list[BoundedValue], Field(min_length=HIDDEN_WIDTH, max_length=HIDDEN_WIDTH)
]


class HiddenNeuron(BaseModel):
    """One hidden neuron, a weight per input in the order the row arrives, and a bias."""

    weights: HiddenWeightRow
    bias: BoundedValue

    model_config = {"extra": "forbid"}


class OutputNeuron(BaseModel):
    """The output neuron, a weight per hidden neuron in layer order, and a bias."""

    weights: OutputWeightRow
    bias: BoundedValue

    model_config = {"extra": "forbid"}


class CheckedWeight(BaseModel):
    """Which single weight the finite-difference check nudges.

    Layer 0 is the hidden layer and layer 1 the output layer. The neuron and
    input indices are the row and column of that layer's weight matrix, so the
    hidden layer has three rows of two and the output layer one row of three.
    """

    layer_index: int = Field(ge=HIDDEN_LAYER, le=OUTPUT_LAYER)
    neuron_index: int = Field(ge=0, le=HIDDEN_WIDTH - 1)
    input_index: int = Field(ge=0, le=HIDDEN_WIDTH - 1)

    model_config = {"extra": "forbid"}


class StepRequest(BaseModel):
    """The network's parameters, one row, its target, and how far to step."""

    inputs: InputRow
    target: BoundedValue
    hidden_activation: HiddenActivation
    hidden_neurons: list[HiddenNeuron] = Field(
        min_length=HIDDEN_WIDTH, max_length=HIDDEN_WIDTH
    )
    output_neuron: OutputNeuron
    learning_rate: float = Field(gt=0.0, le=MAX_LEARNING_RATE)
    checked_weight: CheckedWeight

    model_config = {"extra": "forbid"}


class LayerStep(BaseModel):
    """Everything one layer did, forward and backward, and where it stepped to.

    ``inputs``, ``scores`` and ``outputs`` are the layer's forward response on
    the one row. ``arriving`` is the slope of the loss at this layer's outputs,
    handed down by the layer above or by the loss itself. ``slopes`` is the
    bend's derivative at each score. ``bias_gradient`` is the delta, arriving
    times slope, because on one row the sum over rows is the row. ``passed_down``
    is the slope of the loss at this layer's inputs, which is what the layer
    beneath receives as its own ``arriving``.
    """

    inputs: list[float]
    scores: list[float]
    outputs: list[float]
    arriving: list[float]
    slopes: list[float]
    weight_gradient: list[list[float]]
    bias_gradient: list[float]
    passed_down: list[float]
    weights_after: list[list[float]]
    biases_after: list[float]


class FiniteDifference(BaseModel):
    """The chain rule's slope for one weight beside a nudge-and-remeasure figure."""

    layer_index: int
    neuron_index: int
    input_index: int
    analytic: float
    numerical: float
    disagreement: float
    epsilon: float


class StepResponse(BaseModel):
    """One backward pass, one step, and the loss on either side of it."""

    layers: list[LayerStep]
    prediction: float
    loss_before: float
    loss_gradient: float
    prediction_after: float
    loss_after: float
    largest_movement: float
    finite_difference: FiniteDifference


def _as_floats(block: np.ndarray) -> list[float]:
    """One row of a block, or a vector, as plain floats."""
    return [float(value) for value in np.asarray(block, dtype=np.float64).ravel()]


def _as_rows(block: np.ndarray) -> list[list[float]]:
    """A matrix as plain floats, row by row."""
    return [_as_floats(row) for row in np.asarray(block, dtype=np.float64)]


def _hidden_layer(request: StepRequest) -> DenseLayer:
    """The three hidden neurons, each reading the two inputs under the chosen bend."""
    activation = ACTIVATIONS[request.hidden_activation]
    return DenseLayer(
        [
            Neuron(neuron.weights, bias=neuron.bias, activation=activation)
            for neuron in request.hidden_neurons
        ]
    )


def _output_layer(request: StepRequest) -> DenseLayer:
    """The one linear neuron reading the three hidden outputs."""
    return DenseLayer(
        [
            Neuron(
                request.output_neuron.weights,
                bias=request.output_neuron.bias,
                activation=Identity(),
            )
        ]
    )


def _refuse_missing_weight(checked: CheckedWeight) -> None:
    """Refuse a checked weight at a position the chosen layer does not have.

    The library never sees the index, so it cannot refuse it. Left alone, numpy
    would raise an index error the learner cannot read. Better to say which
    layer was asked and how wide it is.
    """
    if checked.layer_index == HIDDEN_LAYER:
        neurons, inputs, name = HIDDEN_WIDTH, INPUT_WIDTH, "the hidden layer"
    else:
        neurons, inputs, name = OUTPUT_WIDTH, HIDDEN_WIDTH, "the output layer"

    if checked.neuron_index >= neurons or checked.input_index >= inputs:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{name} holds {neurons} neurons reading {inputs} inputs, so "
                f"there is no weight at neuron {checked.neuron_index}, input "
                f"{checked.input_index}"
            ),
        )


def _loss_with_nudged_weight(
    stack: LayerStack,
    inputs: np.ndarray,
    targets: np.ndarray,
    checked: CheckedWeight,
    shift: float,
) -> float:
    """The loss after moving one weight by ``shift`` and nothing else."""
    layers = list(stack)
    layer = layers[checked.layer_index]
    assert isinstance(layer, DenseLayer)

    nudged = np.array(layer.weight_matrix, dtype=np.float64, copy=True)
    nudged[checked.neuron_index, checked.input_index] += shift
    layers[checked.layer_index] = layer.with_parameters(nudged, layer.bias_vector)

    rebuilt = LayerStack(layers)
    return SquaredError().measure(rebuilt.respond_to(inputs).outputs, targets).value


@router.post("/concepts/backpropagation/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Run one row forward, walk the blame back, step once, and measure again."""
    _refuse_missing_weight(request.checked_weight)

    stack = LayerStack([_hidden_layer(request), _output_layer(request)])
    inputs = np.array([request.inputs], dtype=np.float64)
    targets = np.array([[request.target]], dtype=np.float64)
    loss = SquaredError()

    # The library's route, which is what the step is taken from.
    backward = stack.backward_pass(inputs, targets, loss)
    stepped = stack.stepped_by(backward, request.learning_rate)

    # The same walk again, layer by layer, keeping what each layer passed down.
    forward = stack.respond_to(inputs, PassPurpose.TRAINING)
    measurement = loss.measure(forward.outputs, targets)
    arriving = measurement.gradient

    layer_steps: list[LayerStep] = []
    for layer, response, stepped_layer in zip(
        reversed(stack), reversed(forward), reversed(stepped), strict=True
    ):
        assert isinstance(layer, DenseLayer)
        assert isinstance(stepped_layer, DenseLayer)
        correction = layer.correction_for(response, arriving)
        gradient = correction.gradient
        assert gradient is not None

        layer_steps.append(
            LayerStep(
                inputs=_as_floats(response.inputs),
                scores=_as_floats(response.scores),
                outputs=_as_floats(response.outputs),
                arriving=_as_floats(arriving),
                slopes=_as_floats(layer.slopes_at(response.scores)),
                weight_gradient=_as_rows(gradient.weights),
                bias_gradient=_as_floats(gradient.biases),
                passed_down=_as_floats(correction.passed_down),
                weights_after=_as_rows(stepped_layer.weight_matrix),
                biases_after=_as_floats(stepped_layer.bias_vector),
            )
        )
        arriving = correction.passed_down

    # Walked top to bottom; reported in layer order, like the stack itself.
    layer_steps.reverse()

    after = stepped.respond_to(inputs)
    loss_after = loss.measure(after.outputs, targets).value

    checked = request.checked_weight
    checked_gradient = backward[checked.layer_index]
    assert checked_gradient is not None
    analytic = float(
        checked_gradient.weights[checked.neuron_index, checked.input_index]
    )
    nudged_up = _loss_with_nudged_weight(
        stack, inputs, targets, checked, FINITE_DIFFERENCE_EPSILON
    )
    nudged_down = _loss_with_nudged_weight(
        stack, inputs, targets, checked, -FINITE_DIFFERENCE_EPSILON
    )
    numerical = (nudged_up - nudged_down) / (2.0 * FINITE_DIFFERENCE_EPSILON)

    return StepResponse(
        layers=layer_steps,
        prediction=float(forward.outputs[0, 0]),
        loss_before=float(backward.loss),
        loss_gradient=float(measurement.gradient[0, 0]),
        prediction_after=float(after.outputs[0, 0]),
        loss_after=float(loss_after),
        largest_movement=float(backward.largest_movement),
        finite_difference=FiniteDifference(
            layer_index=checked.layer_index,
            neuron_index=checked.neuron_index,
            input_index=checked.input_index,
            analytic=analytic,
            numerical=float(numerical),
            disagreement=float(abs(numerical - analytic)),
            epsilon=FINITE_DIFFERENCE_EPSILON,
        ),
    )
