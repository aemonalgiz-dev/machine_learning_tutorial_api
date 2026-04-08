"""A dense layer and the forward pass, for the dense-layers page.

Two endpoints. ``forward`` takes one row of width two and the weights, biases
and bends of a two-to-three-to-one stack. It builds the library's ``Neuron``
objects, groups them into two ``DenseLayer`` objects, joins those into a
``LayerStack`` and pushes the row through once, saying
``PassPurpose.PREDICTING`` out loud even though it is the default, because the
page names the purpose and the number it prints should come from a pass that
stated one. Everything reported is read straight off the ``StackResponse``,
one ``LayerResponse`` per layer, so the page can show what each layer read,
the score every neuron formed and the output each bend handed upward, along
with every layer's shape and the stack's shape end to end, as the library
records them.

The derivation closes on what the two layers become when both bends are
removed, a single affine map with matrix W2 W1 and bias W2 b1 + b2. No layer
in the library ever collapses, so nothing there reports that product, and the
endpoint forms it from the two layers' ``weight_matrix`` and ``bias_vector``
properties, one matrix multiply and one addition. Beside it the same row is
pushed through a second stack whose bends are both the identity, so the
derivation's claim, that the product answers what the bendless stack answers,
is checked by the library on every request rather than asserted by the page.

``join`` takes the widths of two dense layers and nothing else. It builds each
one from placeholder neurons whose weights are all zero, asks the library to
stack them, and reports the stack's shape. A pair that does not join is
refused by the library's own ``ShapeMismatchError``, whose message names the
seam and both arrangements, and the shared error handling already turns that
into a 422, so the endpoint catches nothing and rewords nothing. No row is
sent on either route, which is the point. The verdict is a comparison of two
tuples of integers, settled before any data exists.

The worked example is the row (2, 1) through hidden rows (1, 2), (-1, 3) and
(1, -3) with biases 0, 1 and 0 under the rectifier, then output weights
(1, -1, 2) with bias 1 under the identity. The hidden scores are 4, 2 and -1,
the rectifier turns the last to 0, and the output is 3. Under a sigmoid the
third hidden neuron answers 0.2689 instead and the output moves to 1.6391.
Without either bend the stack is the single map with weights (4, -7) and bias
0, which answers 1 at that row.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter
from oop_ml import (
    Activation,
    DenseLayer,
    HyperbolicTangent,
    Identity,
    LayerResponse,
    LayerShape,
    LayerStack,
    Neuron,
    PassPurpose,
    RectifiedLinear,
    Sigmoid,
)
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE

router = APIRouter(tags=["dense-layers"])

INPUT_WIDTH = 2
"""How many numbers the row holds, and so how many weights each hidden neuron has."""

HIDDEN_WIDTH = 3
"""How many neurons the hidden layer holds, which is the width it answers with."""

OUTPUT_WIDTH = 1
"""How many neurons the output layer holds. One, so the network answers a number."""

MAX_WIDTH = 8
"""The widest a layer in the join check may read or answer.

Each placeholder layer is built from real neurons, one per answered width with
one weight per read width, so the bound keeps a check to a few dozen zeros.
"""

PLACEHOLDER_WEIGHT = 0.0
"""What every weight in a join-check layer holds. The check never reads them."""


class ActivationName(StrEnum):
    """The four bends the library offers, by name.

    Softmax is deliberately absent, for the reason the neurons page gives. It
    reads a whole layer's scores at once, and these four each read one number.
    """

    IDENTITY = "identity"
    RECTIFIED_LINEAR = "rectified_linear"
    SIGMOID = "sigmoid"
    HYPERBOLIC_TANGENT = "hyperbolic_tangent"


ACTIVATIONS: dict[ActivationName, Activation] = {
    ActivationName.IDENTITY: Identity(),
    ActivationName.RECTIFIED_LINEAR: RectifiedLinear(),
    ActivationName.SIGMOID: Sigmoid(),
    ActivationName.HYPERBOLIC_TANGENT: HyperbolicTangent(),
}
"""One instance per name. A bend carries no state, so sharing it is safe."""


BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]

InputRow = Annotated[
    list[BoundedValue], Field(min_length=INPUT_WIDTH, max_length=INPUT_WIDTH)
]

HiddenWeightRow = Annotated[
    list[BoundedValue], Field(min_length=INPUT_WIDTH, max_length=INPUT_WIDTH)
]

OutputWeightRow = Annotated[
    list[BoundedValue], Field(min_length=HIDDEN_WIDTH, max_length=HIDDEN_WIDTH)
]


class HiddenLayerRequest(BaseModel):
    """The hidden layer's three neurons, one weight row and one bias each."""

    weights: list[HiddenWeightRow] = Field(
        min_length=HIDDEN_WIDTH, max_length=HIDDEN_WIDTH
    )
    biases: list[BoundedValue] = Field(min_length=HIDDEN_WIDTH, max_length=HIDDEN_WIDTH)
    activation: ActivationName

    model_config = {"extra": "forbid"}


class OutputLayerRequest(BaseModel):
    """The output layer's one neuron, reading the three hidden answers."""

    weights: list[OutputWeightRow] = Field(
        min_length=OUTPUT_WIDTH, max_length=OUTPUT_WIDTH
    )
    biases: list[BoundedValue] = Field(min_length=OUTPUT_WIDTH, max_length=OUTPUT_WIDTH)
    activation: ActivationName

    model_config = {"extra": "forbid"}


class ForwardRequest(BaseModel):
    """One row, and the two layers it is to be pushed through."""

    inputs: InputRow
    hidden: HiddenLayerRequest
    output: OutputLayerRequest

    model_config = {"extra": "forbid"}


class Extents(BaseModel):
    """What a layer, or a whole stack, reads and answers with.

    Each side is the tuple of extents the library keeps, not a count. A dense
    layer's tuple has one entry, so ``reads`` of ``[2]`` is the library's
    ``(2,)``, and the list form is there so a picture-shaped layer on another
    page can report ``[8, 26, 26]`` through the same document.
    """

    reads: list[int]
    answers: list[int]


class LayerPass(BaseModel):
    """What one layer did with the row, straight off its ``LayerResponse``.

    ``inputs`` is the row the layer read, ``scores`` the weighted sums plus
    biases before the bend, and ``outputs`` the scores after it, which is the
    row the next layer reads. ``formula`` is the bend's own description, in
    the library's words.
    """

    reads: list[int]
    answers: list[int]
    activation: ActivationName
    formula: str
    inputs: list[float]
    scores: list[float]
    outputs: list[float]


class CollapsedMap(BaseModel):
    """The single affine map the two layers become when both bends go.

    ``weights`` and ``bias`` are the product of the two weight matrices and
    the bias that product carries. ``output`` is what the library answers when
    the same row goes through the same two layers with identity bends, which
    the derivation says is the collapsed map evaluated at that row.
    """

    weights: list[float]
    bias: float
    output: float


class ForwardResponse(BaseModel):
    """The whole pass, layer by layer, and the stack's shape end to end."""

    purpose: str
    shape: Extents
    layers: list[LayerPass]
    output: float
    without_bends: CollapsedMap


def _extents_of(shape: LayerShape) -> Extents:
    """The library's shape as two lists, so the page can print the tuples."""
    return Extents(reads=list(shape.reads), answers=list(shape.answers))


def _dense_layer(
    weights: list[list[float]], biases: list[float], activation: Activation
) -> DenseLayer:
    """One neuron per weight row, all sharing the bend, grouped into a layer."""
    return DenseLayer(
        [
            Neuron(weight_row, bias=bias, activation=activation)
            for weight_row, bias in zip(weights, biases, strict=True)
        ]
    )


def _pass_through(
    layer: DenseLayer, name: ActivationName, response: LayerResponse
) -> LayerPass:
    """Read the first, and only, row of one layer's response."""
    return LayerPass(
        reads=list(layer.shape.reads),
        answers=list(layer.shape.answers),
        activation=name,
        formula=ACTIVATIONS[name].description,
        inputs=[float(value) for value in response.inputs[0]],
        scores=[float(value) for value in response.scores[0]],
        outputs=[float(value) for value in response.outputs[0]],
    )


@router.post("/concepts/dense-layers/forward", response_model=ForwardResponse)
def forward(request: ForwardRequest) -> ForwardResponse:
    """Push one row through the two layers and report what each one did."""
    hidden = _dense_layer(
        request.hidden.weights,
        request.hidden.biases,
        ACTIVATIONS[request.hidden.activation],
    )
    output = _dense_layer(
        request.output.weights,
        request.output.biases,
        ACTIVATIONS[request.output.activation],
    )
    stack = LayerStack([hidden, output])
    response = stack.respond_to([request.inputs], PassPurpose.PREDICTING)

    names = (request.hidden.activation, request.output.activation)
    layers = [
        _pass_through(layer, name, layer_response)
        for layer, name, layer_response in zip(
            (hidden, output), names, response, strict=True
        )
    ]

    bendless = LayerStack(
        [
            _dense_layer(request.hidden.weights, request.hidden.biases, Identity()),
            _dense_layer(request.output.weights, request.output.biases, Identity()),
        ]
    )
    bendless_response = bendless.respond_to([request.inputs], PassPurpose.PREDICTING)

    collapsed_weights = output.weight_matrix @ hidden.weight_matrix
    collapsed_bias = output.weight_matrix @ hidden.bias_vector + output.bias_vector

    return ForwardResponse(
        purpose=PassPurpose.PREDICTING.value,
        shape=_extents_of(stack.shape),
        layers=layers,
        output=float(response.outputs[0, 0]),
        without_bends=CollapsedMap(
            weights=[float(value) for value in collapsed_weights[0]],
            bias=float(collapsed_bias[0]),
            output=float(bendless_response.outputs[0, 0]),
        ),
    )


class DenseShape(BaseModel):
    """What one dense layer reads and answers with, as the two widths."""

    reads: int = Field(ge=1, le=MAX_WIDTH)
    answers: int = Field(ge=1, le=MAX_WIDTH)

    model_config = {"extra": "forbid"}


class JoinRequest(BaseModel):
    """Two dense layers, the one beneath and the one above, by shape alone."""

    beneath: DenseShape
    above: DenseShape

    model_config = {"extra": "forbid"}


class JoinResponse(BaseModel):
    """The two layers stacked, which they only are when the join holds."""

    layers: list[Extents]
    stack: Extents


def _placeholder_layer(shape: DenseShape) -> DenseLayer:
    """A dense layer of the asked shape whose weights are all zero.

    The join check reads shapes and never weights, so any finite numbers would
    do. Zero is the one that says so.
    """
    return DenseLayer(
        [
            Neuron(
                [PLACEHOLDER_WEIGHT] * shape.reads,
                bias=PLACEHOLDER_WEIGHT,
                activation=Identity(),
            )
            for _neuron in range(shape.answers)
        ]
    )


@router.post("/concepts/dense-layers/join", response_model=JoinResponse)
def join(request: JoinRequest) -> JoinResponse:
    """Ask the library whether the layer above can read what the one beneath answers.

    A join that fails raises the library's ``ShapeMismatchError`` from inside
    ``LayerStack``, and the shared handler answers 422 with that message. No
    data is involved on either route.
    """
    beneath = _placeholder_layer(request.beneath)
    above = _placeholder_layer(request.above)
    stack = LayerStack([beneath, above])

    return JoinResponse(
        layers=[_extents_of(beneath.shape), _extents_of(above.shape)],
        stack=_extents_of(stack.shape),
    )
