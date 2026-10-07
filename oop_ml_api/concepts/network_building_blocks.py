"""Small, inspectable SDK passes for the sequence and network lessons.

Parameters are deliberately chosen examples, not trained language models. The
SDK owns every forward and backward operation; this module selects fixtures and
serializes their responses. Requests are bounded so each control remains quick.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    DenseLayer,
    Embedding,
    GatedRecurrentUnit,
    HyperbolicTangent,
    LongShortTermMemory,
    MultiHeadAttention,
    Neuron,
    PositionalEncoding,
    RadialBasisLayer,
    Residual,
    SimpleRecurrent,
    SquaredError,
)
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/concepts", tags=["network-building-blocks"])

Matrix = list[list[float]]
Colour = Literal["red", "blue", "green"]
Word = Literal["cat", "dog", "mat"]
SmallNumber = Annotated[float, Field(ge=-3, le=3, allow_inf_nan=False)]


class ExampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttentionRequest(ExampleRequest):
    tokens: list[Colour] = Field(
        default=["red", "blue", "red"], min_length=1, max_length=8
    )
    heads: Literal[1, 2] = 1
    causal: bool = False
    positions: bool = False


class AttentionView(BaseModel):
    tokens: list[Colour]
    inputs: Matrix
    queries: list[Matrix]
    keys: list[Matrix]
    values: list[Matrix]
    shares: list[Matrix]
    outputs: Matrix
    head_size: int


@router.post("/attention/respond", response_model=AttentionView)
def attend(request: AttentionRequest) -> AttentionView:
    vocabulary = ["red", "blue", "green"]
    identifiers = np.array([[vocabulary.index(token) for token in request.tokens]])
    embedding = Embedding(3, 4, len(request.tokens)).with_parameters(np.eye(3, 4))
    inputs = embedding.respond_to(identifiers).outputs
    if request.positions:
        inputs = PositionalEncoding((len(request.tokens), 4)).respond_to(inputs).outputs
    layer = MultiHeadAttention(
        reads=(len(request.tokens), 4),
        n_heads=request.heads,
        looks_ahead=not request.causal,
    ).with_parameters(np.tile(np.eye(4), (4, 1)), np.zeros(16))
    response = layer.respond_to(inputs)
    return AttentionView(
        tokens=request.tokens,
        inputs=inputs[0].tolist(),
        queries=response.queries[0].tolist(),
        keys=response.keys[0].tolist(),
        values=response.values[0].tolist(),
        shares=response.attention[0].tolist(),
        outputs=response.outputs[0].tolist(),
        head_size=layer.head_size,
    )


class EmbeddingRequest(ExampleRequest):
    tokens: list[Word] = Field(
        default=["cat", "dog", "cat"], min_length=1, max_length=8
    )
    learning_rate: float = Field(default=0.1, ge=0, le=0.1, allow_inf_nan=False)


class EmbeddingView(BaseModel):
    vocabulary: list[str]
    identifiers: list[int]
    before: Matrix
    looked_up: Matrix
    targets: Matrix
    gradient: Matrix
    after: Matrix
    loss_before: float
    loss_after: float


@router.post("/embedding-layers/step", response_model=EmbeddingView)
def update_embedding(request: EmbeddingRequest) -> EmbeddingView:
    vocabulary = ["cat", "dog", "mat"]
    identifiers = np.array([[vocabulary.index(token) for token in request.tokens]])
    layer = Embedding(3, 2, len(request.tokens)).with_parameters(
        np.array([[0.2, 0.0], [-0.2, 0.0], [0.0, -0.2]])
    )
    target_table = np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]])
    targets = target_table[identifiers]
    response = layer.respond_to(identifiers)
    measurement = SquaredError().measure(
        response.outputs.reshape(1, -1), targets.reshape(1, -1)
    )
    correction = layer.correction_for(
        response, measurement.gradient.reshape(response.outputs.shape)
    )
    assert correction.gradient is not None
    updated = layer.stepped_by(correction.gradient, request.learning_rate)
    after = SquaredError().measure(
        updated.respond_to(identifiers).outputs.reshape(1, -1), targets.reshape(1, -1)
    )
    return EmbeddingView(
        vocabulary=vocabulary,
        identifiers=identifiers[0].tolist(),
        before=layer.table.tolist(),
        looked_up=response.outputs[0].tolist(),
        targets=targets[0].tolist(),
        gradient=correction.gradient.weights.tolist(),
        after=updated.table.tolist(),
        loss_before=measurement.value,
        loss_after=after.value,
    )


class PositionRequest(ExampleRequest):
    steps: int = Field(default=4, ge=1, le=8)
    wavelength: float = Field(default=10000, ge=2, le=10000, allow_inf_nan=False)


class PositionView(BaseModel):
    inputs: Matrix
    pattern: Matrix
    outputs: Matrix


@router.post("/positional-encoding/respond", response_model=PositionView)
def encode_positions(request: PositionRequest) -> PositionView:
    inputs = np.tile(np.array([1.0, 0.0, 1.0, 0.0]), (1, request.steps, 1))
    layer = PositionalEncoding((request.steps, 4), wavelength=request.wavelength)
    return PositionView(
        inputs=inputs[0].tolist(),
        pattern=layer.pattern.tolist(),
        outputs=layer.respond_to(inputs).outputs[0].tolist(),
    )


class RecurrentRequest(ExampleRequest):
    kind: Literal["simple", "lstm", "gru"] = "lstm"
    steps: int = Field(default=6, ge=2, le=12)
    signal: SmallNumber = 1.0
    forget_bias: SmallNumber = 1.0


class RecurrentView(BaseModel):
    inputs: list[float]
    states: list[float]
    cells: list[float]
    gate_names: list[str]
    gates: Matrix
    input_slopes: list[float]


@router.post("/recurrent-layers/respond", response_model=RecurrentView)
def recur(request: RecurrentRequest) -> RecurrentView:
    classes = {
        "simple": SimpleRecurrent,
        "lstm": LongShortTermMemory,
        "gru": GatedRecurrentUnit,
    }
    layer = classes[request.kind](
        reads=(request.steps, 1), n_units=1, answers_every_step=True
    )
    weights = np.zeros_like(layer.weights)
    biases = np.zeros_like(layer.biases)
    if request.kind == "simple":
        weights[0] = [1.0, 0.5]
        names = ["state after tanh"]
    elif request.kind == "lstm":
        weights[LongShortTermMemory.CANDIDATE] = [1.0, 0.0]
        biases[LongShortTermMemory.FORGET] = request.forget_bias
        names = ["input gate", "forget gate", "candidate", "output gate"]
    else:
        weights[GatedRecurrentUnit.CANDIDATE] = [1.0, 0.5]
        names = ["reset gate", "update gate", "candidate"]
    layer = layer.with_parameters(weights, biases)
    inputs = np.zeros((1, request.steps, 1))
    inputs[0, 0, 0] = request.signal
    response = layer.respond_to(inputs)
    arriving = np.zeros_like(response.outputs)
    arriving[0, -1, 0] = 1.0
    correction = layer.correction_for(response, arriving)
    return RecurrentView(
        inputs=inputs[0, :, 0].tolist(),
        states=response.outputs[0, :, 0].tolist(),
        cells=[float(step.cell[0, 0]) for step in response.steps],
        gate_names=names,
        gates=[step.gates[0].tolist() for step in response.steps],
        input_slopes=correction.passed_down[0, :, 0].tolist(),
    )


class RadialRequest(ExampleRequest):
    probe: SmallNumber = 1.0
    width: float = Field(default=1.0, ge=0.2, le=3, allow_inf_nan=False)


class RadialView(BaseModel):
    centres: list[float]
    outputs: list[float]
    distances_squared: list[float]
    curve_inputs: list[float]
    curve_outputs: Matrix
    centre_slopes: list[float]
    width_slopes: list[float]


@router.post("/radial-basis-networks/respond", response_model=RadialView)
def radial(request: RadialRequest) -> RadialView:
    layer = RadialBasisLayer(reads=1, n_units=3).with_parameters(
        centres=np.array([[-1.0], [0.0], [1.0]]),
        widths=np.full(3, request.width),
    )
    response = layer.respond_to(np.array([[request.probe]]))
    correction = layer.correction_for(response, np.ones_like(response.outputs))
    assert correction.gradient is not None
    curve_inputs = np.linspace(-3, 3, 61).reshape(-1, 1)
    return RadialView(
        centres=layer.centres[:, 0].tolist(),
        outputs=response.outputs[0].tolist(),
        distances_squared=response.scores[0].tolist(),
        curve_inputs=curve_inputs[:, 0].tolist(),
        curve_outputs=layer.respond_to(curve_inputs).outputs.tolist(),
        centre_slopes=correction.gradient.weights[:, 0].tolist(),
        width_slopes=correction.gradient.biases.tolist(),
    )


class ResidualRequest(ExampleRequest):
    value: SmallNumber = 1.0
    weight: SmallNumber = 0.5


class ResidualView(BaseModel):
    inner_output: float
    output: float
    inner_slope: float
    slope: float


@router.post("/residual-connections/respond", response_model=ResidualView)
def residual(request: ResidualRequest) -> ResidualView:
    inner = DenseLayer([Neuron([request.weight], 0.0, HyperbolicTangent())])
    layer = Residual(inner)
    inputs = np.array([[request.value]])
    inner_response = inner.respond_to(inputs)
    response = layer.respond_to(inputs)
    arriving = np.ones_like(response.outputs)
    return ResidualView(
        inner_output=float(inner_response.outputs[0, 0]),
        output=float(response.outputs[0, 0]),
        inner_slope=float(
            inner.correction_for(inner_response, arriving).passed_down[0, 0]
        ),
        slope=float(layer.correction_for(response, arriving).passed_down[0, 0]),
    )
