"""A dense layer and the forward pass, for the dense-layers page.

Six endpoints, most on the same 2-3-1 network the backpropagation page works
by hand: the row (1, 2), hidden neurons (1, 1), (1, -1) and (-1, 1) with no
biases under the rectifier, and an output neuron (1, -1, 2) with no bias under
the identity. The hidden scores are 3, -1 and 1, the rectifier turns the
middle one to 0, and the output is 3 - 0 + 2 = 5.

``forward`` pushes one row through the two layers and reports what each layer
did, straight off the library's ``LayerResponse``: the block it read, the
scores it formed, the outputs it handed upward, and the shape of every layer
and of the stack end to end. Beside the matrix route it also runs the
library's observed route, ``DenseLayer.neuron_responses``, which asks each
neuron for its own answer one at a time, and reports the largest gap between
the two, which is how the page can say the loop and the multiply are one
calculation rather than assert it. The derivation closes on what the two
layers become when both bends are removed, one affine map with matrix W2 W1
and bias W2 b1 + b2; no layer ever collapses, so the endpoint forms the
product from ``weight_matrix`` and ``bias_vector`` and checks it against a
second stack whose bends are both the identity.

``block`` pushes several rows through the same two layers at once, because
the matrix form's real payoff is a block of rows in one multiply, and reports
the shape of every block on the way up. Its rows may be one to four wide, so
that a row of the wrong width reaches the library and is refused by the first
layer in the library's own words rather than by the request's bounds.

``routes`` builds one layer of any width up to sixty-four from a seed and
asks it for the same row both ways, because on the page's own two- and
three-wide layers the two routes agree to the last bit and the claim worth
making is that they part, by a few multiples of machine precision, once the
matrix multiply sums enough terms in an order the loop does not.

``join`` takes the widths of two dense layers and nothing else, builds each
from zero weights, asks the library to stack them, and reports the stack's
shape; a pair that does not join is refused by the library's
``ShapeMismatchError``, whose message names the seam. ``chain`` is the same
question over up to six layers, so the page can show that a whole network is
checked seam by seam at construction, before any row exists, and that the
refusal names which seam.

``carve`` trains rectifier networks on the classification pages' crowd, the
same twenty-five people the neurons page standardises, at whatever hidden
widths the request names, under several starting seeds, for a fixed number of
epochs at a fixed rate. It reports every run's accuracy, loss and count of
units that never switched on, and the best run's decision region and the fold
line of every first-layer unit, which is what one hidden layer of rectifiers
carves the plane with. The loop is the training page's, a dozen lines over
``backward_pass`` and ``stepped_by``; nothing here computes a gradient. It
exists so the page can measure what width and depth buy on this crowd rather
than quote a theorem about what a wide enough layer could do.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Activation,
    BinaryCrossEntropy,
    DenseLayer,
    Feature,
    HyperbolicTangent,
    Identity,
    LayerResponse,
    LayerShape,
    LayerStack,
    LogisticRegression,
    Neuron,
    PassPurpose,
    RectifiedLinear,
    Sigmoid,
    Standardizer,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.neurons import CROWD
from oop_ml_api.concepts.two_feature_classification import GRID_CELLS, RegionGrid
from oop_ml_api.limits import MAX_MAGNITUDE

router = APIRouter(tags=["dense-layers"])

INPUT_WIDTH = 2
"""How many numbers the row holds, and so how many weights each hidden neuron has."""

HIDDEN_WIDTH = 3
"""How many neurons the hidden layer holds, which is the width it answers with."""

OUTPUT_WIDTH = 1
"""How many neurons the output layer holds. One, so the network answers a number."""

MAX_WIDTH = 8
"""The widest a layer in the join and chain checks may read or answer.

Each placeholder layer is built from real neurons, one per answered width with
one weight per read width, so the bound keeps a check to a few dozen zeros.
"""

MAX_CHAIN_LAYERS = 6
"""How many layers a chain check may hold. Five seams is enough to name one."""

PLACEHOLDER_WEIGHT = 0.0
"""What every weight in a join-check layer holds. The check never reads them."""

MAX_BLOCK_ROWS = 6
"""How many rows the block endpoint accepts. Enough to see a block, not a dataset."""

MAX_ROW_WIDTH = 4
"""How wide a block row may be. Wider than the layer reads on purpose, so that a
row of the wrong width reaches the library and is refused in its words."""

MAX_HIDDEN_LAYERS = 3
"""How deep a carved network may go. Three hidden layers is past where the
crowd needs any."""

MAX_SEEDS = 6
"""How many starting seeds a carve may run. Six runs of 1500 epochs on
twenty-five rows is under two seconds, and it is enough to see the spread."""

CARVE_LEARNING_RATE = 0.5
"""The stride every carved network walks at, fixed so the numbers are pinned."""

CARVE_EPOCHS = 1500
"""How many epochs every carved network walks, fixed for the same reason."""


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

BlockRow = Annotated[list[BoundedValue], Field(min_length=1, max_length=MAX_ROW_WIDTH)]


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


class NeuronRoute(BaseModel):
    """The observed route: every neuron asked one at a time for the same row.

    ``scores`` and ``outputs`` come from ``DenseLayer.neuron_responses``, which
    walks the definition as a loop, and ``products`` holds each neuron's
    weight-times-input terms in input order, so the page can print the dot
    product a neuron formed term by term.
    """

    scores: list[float]
    outputs: list[float]
    products: list[list[float]]


class LayerPass(BaseModel):
    """What one layer did with the row, straight off its ``LayerResponse``.

    ``inputs`` is the row the layer read, ``scores`` the weighted sums plus
    biases before the bend, and ``outputs`` the scores after it, which is the
    row the next layer reads. ``formula`` is the bend's own description, in
    the library's words. ``weight_matrix`` is the layer's matrix with one row
    per neuron and one column per input, ``bias_vector`` one bias per neuron,
    and ``n_parameters`` the count of both. ``neuron_route`` is the same
    layer asked neuron by neuron, and ``route_gap`` the largest absolute
    difference between the two routes' scores and outputs.
    """

    reads: list[int]
    answers: list[int]
    activation: ActivationName
    formula: str
    inputs: list[float]
    scores: list[float]
    outputs: list[float]
    weight_matrix: list[list[float]]
    bias_vector: list[float]
    n_parameters: int
    neuron_route: NeuronRoute
    route_gap: float


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
    n_layers: int
    n_parameters: int
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


def _floats(block: np.ndarray) -> list[float]:
    """A vector, or one row of a block, as plain floats."""
    return [float(value) for value in np.asarray(block, dtype=np.float64).ravel()]


def _rows(block: np.ndarray) -> list[list[float]]:
    """A matrix as plain floats, row by row."""
    return [_floats(row) for row in np.asarray(block, dtype=np.float64)]


def _parameter_count(layer: DenseLayer) -> int:
    """One weight per input per neuron, plus one bias per neuron."""
    return layer.shape.n_inputs * layer.shape.n_outputs + layer.shape.n_outputs


def _pass_through(
    layer: DenseLayer, name: ActivationName, response: LayerResponse
) -> LayerPass:
    """Read the first, and only, row of one layer's response, both routes."""
    row = response.inputs[0]
    neuron_responses = layer.neuron_responses(row)
    loop_scores = [neuron_response.score for neuron_response in neuron_responses]
    loop_outputs = [neuron_response.output for neuron_response in neuron_responses]
    matrix_scores = _floats(response.scores[0])
    matrix_outputs = _floats(response.outputs[0])
    route_gap = max(
        max(
            abs(fast - slow)
            for fast, slow in zip(matrix_scores, loop_scores, strict=True)
        ),
        max(
            abs(fast - slow)
            for fast, slow in zip(matrix_outputs, loop_outputs, strict=True)
        ),
    )

    return LayerPass(
        reads=list(layer.shape.reads),
        answers=list(layer.shape.answers),
        activation=name,
        formula=ACTIVATIONS[name].description,
        inputs=_floats(row),
        scores=matrix_scores,
        outputs=matrix_outputs,
        weight_matrix=_rows(layer.weight_matrix),
        bias_vector=_floats(layer.bias_vector),
        n_parameters=_parameter_count(layer),
        neuron_route=NeuronRoute(
            scores=loop_scores,
            outputs=loop_outputs,
            products=[_floats(neuron.weights.values * row) for neuron in layer],
        ),
        route_gap=float(route_gap),
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
        n_layers=len(stack),
        n_parameters=sum(layer.n_parameters for layer in layers),
        layers=layers,
        output=float(response.outputs[0, 0]),
        without_bends=CollapsedMap(
            weights=_floats(collapsed_weights[0]),
            bias=float(collapsed_bias[0]),
            output=float(bendless_response.outputs[0, 0]),
        ),
    )


class BlockRequest(BaseModel):
    """Several rows at once, and the two layers they go through together.

    The rows may be one to four wide. The hidden layer reads two, so any other
    width is a real disagreement the library refuses, and the request lets it
    through so that the refusal is the library's own.
    """

    rows: list[BlockRow] = Field(min_length=1, max_length=MAX_BLOCK_ROWS)
    hidden: HiddenLayerRequest
    output: OutputLayerRequest

    model_config = {"extra": "forbid"}


class BlockPass(BaseModel):
    """One layer's response to a block: three blocks and their shapes.

    ``inputs_shape`` is ``(n_rows, n_inputs)`` and ``scores_shape`` is
    ``(n_rows, n_neurons)``, which is also the outputs' shape, since a score
    and its output describe the same neuron on the same row.
    """

    reads: list[int]
    answers: list[int]
    inputs_shape: list[int]
    scores_shape: list[int]
    inputs: list[list[float]]
    scores: list[list[float]]
    outputs: list[list[float]]


class BlockResponse(BaseModel):
    """The block on its way up, and the last layer's outputs as the answer."""

    n_rows: int
    shape: Extents
    layers: list[BlockPass]
    outputs: list[float]


@router.post("/concepts/dense-layers/block", response_model=BlockResponse)
def block(request: BlockRequest) -> BlockResponse:
    """Push a block of rows through the two layers in one pass per layer.

    A row of the wrong width is refused by the first layer's own check, as a
    ``ShapeMismatchError`` naming what the layer reads and what arrived, and
    the shared handler answers 422 with that message.
    """
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
    response = stack.respond_to(request.rows, PassPurpose.PREDICTING)

    return BlockResponse(
        n_rows=len(request.rows),
        shape=_extents_of(stack.shape),
        layers=[
            BlockPass(
                reads=list(layer.shape.reads),
                answers=list(layer.shape.answers),
                inputs_shape=list(layer_response.inputs.shape),
                scores_shape=list(layer_response.scores.shape),
                inputs=_rows(layer_response.inputs),
                scores=_rows(layer_response.scores),
                outputs=_rows(layer_response.outputs),
            )
            for layer, layer_response in zip(stack, response, strict=True)
        ],
        outputs=_floats(response.outputs),
    )


MAX_ROUTE_WIDTH = 64
"""The widest a route-agreement layer may be on either side.

Sixty-four inputs is wide enough for the matrix multiply's summation order to
part from the loop's in the last bits, which is the whole point of the check.
"""


class RoutesRequest(BaseModel):
    """One seeded layer of any width, to compare its two routes on one row.

    The weights, biases and row are drawn from the seed as standard normals,
    so the request carries no numbers of its own and every figure the page
    quotes is one the library computed.
    """

    n_inputs: int = Field(ge=1, le=MAX_ROUTE_WIDTH)
    n_neurons: int = Field(ge=1, le=MAX_ROUTE_WIDTH)
    activation: ActivationName = ActivationName.HYPERBOLIC_TANGENT
    seed: int = Field(default=0, ge=0, le=9999)

    model_config = {"extra": "forbid"}


class RoutesResponse(BaseModel):
    """The two routes on one row, and how far apart they came out.

    ``score_gap`` is the largest absolute difference between a score formed
    by the matrix multiply and the same score formed by the neuron's own dot
    product, over every neuron; ``output_gap`` the same after the bend. Both
    are zero or a few multiples of machine precision, never more, and which
    of the two it is depends on the summation order and nothing else.
    """

    n_inputs: int
    n_neurons: int
    n_parameters: int
    score_gap: float
    output_gap: float
    first_scores_matrix: list[float]
    first_scores_loop: list[float]


@router.post("/concepts/dense-layers/routes", response_model=RoutesResponse)
def routes(request: RoutesRequest) -> RoutesResponse:
    """Ask one seeded layer for the same row both ways and report the gap."""
    generator = np.random.default_rng(request.seed)
    activation = ACTIVATIONS[request.activation]
    layer = DenseLayer(
        [
            Neuron(
                weights=generator.normal(size=request.n_inputs),
                bias=float(generator.normal()),
                activation=activation,
            )
            for _ in range(request.n_neurons)
        ]
    )
    row = generator.normal(size=request.n_inputs)

    matrix_response = layer.respond_to([row], PassPurpose.PREDICTING)
    loop_responses = layer.neuron_responses(row)
    matrix_scores = _floats(matrix_response.scores[0])
    matrix_outputs = _floats(matrix_response.outputs[0])
    loop_scores = [item.score for item in loop_responses]
    loop_outputs = [item.output for item in loop_responses]

    shown = min(request.n_neurons, HIDDEN_WIDTH)
    return RoutesResponse(
        n_inputs=request.n_inputs,
        n_neurons=request.n_neurons,
        n_parameters=_parameter_count(layer),
        score_gap=max(
            abs(fast - slow)
            for fast, slow in zip(matrix_scores, loop_scores, strict=True)
        ),
        output_gap=max(
            abs(fast - slow)
            for fast, slow in zip(matrix_outputs, loop_outputs, strict=True)
        ),
        first_scores_matrix=matrix_scores[:shown],
        first_scores_loop=loop_scores[:shown],
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


class ChainRequest(BaseModel):
    """Up to six dense layers by shape alone, bottom to top."""

    layers: list[DenseShape] = Field(min_length=1, max_length=MAX_CHAIN_LAYERS)

    model_config = {"extra": "forbid"}


class ChainResponse(BaseModel):
    """The whole chain stacked, with its shape end to end and its seam count."""

    layers: list[Extents]
    stack: Extents
    n_seams: int
    n_parameters: int


@router.post("/concepts/dense-layers/chain", response_model=ChainResponse)
def chain(request: ChainRequest) -> ChainResponse:
    """Stack every layer at once and let the library check every seam.

    The library walks the seams in order and refuses at the first that does
    not hold, naming its position and both arrangements, so a chain whose
    third seam is wrong is refused with the words ``layer 2 answers with``.
    """
    layers = [_placeholder_layer(shape) for shape in request.layers]
    stack = LayerStack(layers)

    return ChainResponse(
        layers=[_extents_of(layer.shape) for layer in layers],
        stack=_extents_of(stack.shape),
        n_seams=len(layers) - 1,
        n_parameters=sum(_parameter_count(layer) for layer in layers),
    )


HiddenWidths = Annotated[
    list[Annotated[int, Field(ge=1, le=MAX_WIDTH)]],
    Field(min_length=1, max_length=MAX_HIDDEN_LAYERS),
]


class CarveRequest(BaseModel):
    """Which hidden widths to build, bottom to top, and how many starts to try."""

    hidden_widths: HiddenWidths
    n_seeds: int = Field(default=MAX_SEEDS, ge=1, le=MAX_SEEDS)

    model_config = {"extra": "forbid"}


class CarvedRun(BaseModel):
    """One training run from one seed, as it ended.

    ``dead_units`` has one entry per hidden layer and counts the units whose
    score was at or below zero on every one of the twenty-five people, so
    the rectifier switched them off for the whole crowd and they contribute a
    constant. ``starting_loss`` is the loss before any step.
    """

    seed: int
    accuracy: float
    n_correct: int
    starting_loss: float
    final_loss: float
    dead_units: list[int]


class FoldLine(BaseModel):
    """One first-layer rectifier's fold, in standardised units.

    The unit's score is ``weight_height * h + weight_weight * w + bias``, and
    the fold is where that is zero. On one side the unit answers zero and on
    the other it answers the score, so the line is where the plane creases.
    ``dead`` says the unit never switched on for anyone in the crowd.
    """

    weight_height: float
    weight_weight: float
    bias: float
    dead: bool


class CarvedPerson(BaseModel):
    """One person, their measurements, and what the best run called them."""

    height: float
    weight: float
    is_adult: int
    called_adult: int


class CarveResponse(BaseModel):
    """Every run at one architecture, and the best run's region and folds.

    ``regions`` is the best run's answer at every cell of a lattice over the
    crowd, in raw units, and ``folds`` its first hidden layer's creases, in
    standardised units, with the means and deviations needed to draw them
    over the raw plane. ``straight_line_accuracy`` is what the logistic
    page's model manages on the same people, fitted with the same library.
    """

    hidden_widths: list[int]
    n_parameters: int
    learning_rate: float
    epochs: int
    runs: list[CarvedRun]
    best_seed: int
    best_accuracy: float
    mean_accuracy: float
    seeds_reaching_everyone: int
    regions: RegionGrid
    folds: list[FoldLine]
    people: list[CarvedPerson]
    mean_height: float
    deviation_height: float
    mean_weight: float
    deviation_weight: float
    straight_line_accuracy: float


class _StandardisedCrowd:
    """The crowd's two columns standardised once, with the labels beside them."""

    __slots__ = ("labels", "raw", "rows", "scaler", "standardised")

    def __init__(self) -> None:
        heights = Feature("height", [float(row[0]) for row in CROWD])
        weights = Feature("weight", [float(row[1]) for row in CROWD])
        self.raw = [heights, weights]
        self.labels = Feature("is_adult", [float(row[2]) for row in CROWD])
        self.scaler = Standardizer().fit(self.raw)
        self.standardised = self.scaler.transform(self.raw)
        self.rows = np.column_stack(
            [np.asarray(feature.values) for feature in self.standardised]
        )

    def standardise(self, heights: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Any block of raw measurements, through the crowd's own scaling."""
        scaled = self.scaler.transform(
            [Feature("height", heights), Feature("weight", weights)]
        )
        return np.column_stack([np.asarray(feature.values) for feature in scaled])


@cache
def _standardised_crowd() -> _StandardisedCrowd:
    return _StandardisedCrowd()


@cache
def _straight_line_accuracy() -> float:
    """What the logistic page's model manages on the same standardised people."""
    crowd = _standardised_crowd()
    model = LogisticRegression().fit(crowd.standardised, crowd.labels)
    return float(model.score(crowd.standardised, crowd.labels))


def _fresh_stack(hidden_widths: tuple[int, ...], seed: int) -> LayerStack:
    """Rectifier hidden layers under a straight output layer, drawn from a seed.

    The weights are drawn with a spread of the square root of two over the
    fan-in, which is the initialisation the training page uses, and every
    bias starts at zero. The output layer is left straight because the loss
    applies its own squash, which is the loss-functions page's argument.
    """
    generator = np.random.default_rng(seed)

    def layer(n_inputs: int, n_outputs: int, activation: Activation) -> DenseLayer:
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

    layers: list[DenseLayer] = []
    n_inputs = INPUT_WIDTH
    for width in hidden_widths:
        layers.append(layer(n_inputs, width, RectifiedLinear()))
        n_inputs = width
    layers.append(layer(n_inputs, OUTPUT_WIDTH, Identity()))
    return LayerStack(layers)


def _calls(stack: LayerStack, rows: np.ndarray) -> np.ndarray:
    """Which class the network calls each row, at the halfway mark.

    The loss squashes the last layer's score through a sigmoid, and a
    sigmoid is above one half exactly where its score is above zero, so the
    call is the sign of the raw score.
    """
    scores = np.asarray(stack.respond_to(rows).outputs, dtype=np.float64).ravel()
    return (scores >= 0.0).astype(np.float64)


class _TrainedRun:
    """One trained stack and what was measured about it."""

    __slots__ = ("run", "stack")

    def __init__(self, run: CarvedRun, stack: LayerStack) -> None:
        self.run = run
        self.stack = stack


def _train(hidden_widths: tuple[int, ...], seed: int) -> _TrainedRun:
    """The training page's loop, on the crowd, from one seed."""
    crowd = _standardised_crowd()
    rows = crowd.rows
    targets = np.asarray(crowd.labels.values).reshape(-1, 1)
    loss = BinaryCrossEntropy()

    stack = _fresh_stack(hidden_widths, seed)
    starting_loss = stack.backward_pass(rows, targets, loss).loss
    for _ in range(CARVE_EPOCHS):
        backward = stack.backward_pass(rows, targets, loss)
        stack = stack.stepped_by(backward, CARVE_LEARNING_RATE)

    final_loss = stack.backward_pass(rows, targets, loss).loss
    responses = stack.respond_to(rows)
    dead_units = [
        int(np.sum(np.all(responses[position].scores <= 0.0, axis=0)))
        for position in range(len(hidden_widths))
    ]
    labels = np.asarray(crowd.labels.values)
    n_correct = int(np.sum(_calls(stack, rows) == labels))

    return _TrainedRun(
        CarvedRun(
            seed=seed,
            accuracy=n_correct / len(CROWD),
            n_correct=n_correct,
            starting_loss=float(starting_loss),
            final_loss=float(final_loss),
            dead_units=dead_units,
        ),
        stack,
    )


def _region_of(stack: LayerStack) -> RegionGrid:
    """The network's call at every cell of a lattice over the crowd, raw units."""
    heights = np.array([row[0] for row in CROWD], dtype=np.float64)
    weights = np.array([row[1] for row in CROWD], dtype=np.float64)
    x_pad = 0.05 * float(heights.max() - heights.min())
    y_pad = 0.05 * float(weights.max() - weights.min())
    x_min, x_max = float(heights.min()) - x_pad, float(heights.max()) + x_pad
    y_min, y_max = float(weights.min()) - y_pad, float(weights.max()) + y_pad

    lattice_x, lattice_y = np.meshgrid(
        np.linspace(x_min, x_max, GRID_CELLS), np.linspace(y_min, y_max, GRID_CELLS)
    )
    cells = _standardised_crowd().standardise(lattice_x.ravel(), lattice_y.ravel())
    called = _calls(stack, cells).astype(int).reshape(GRID_CELLS, GRID_CELLS)

    return RegionGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cells=GRID_CELLS,
        labels=called.tolist(),
    )


def _folds_of(stack: LayerStack) -> list[FoldLine]:
    """Every first-layer unit's crease, with whether it ever switched on."""
    first = stack[0]
    assert isinstance(first, DenseLayer)
    scores = stack.respond_to(_standardised_crowd().rows)[0].scores
    return [
        FoldLine(
            weight_height=float(neuron.weights.values[0]),
            weight_weight=float(neuron.weights.values[1]),
            bias=float(neuron.bias),
            dead=bool(np.all(scores[:, position] <= 0.0)),
        )
        for position, neuron in enumerate(first)
    ]


@cache
def _carved(hidden_widths: tuple[int, ...], n_seeds: int) -> CarveResponse:
    crowd = _standardised_crowd()
    trained = [_train(hidden_widths, seed) for seed in range(n_seeds)]
    runs = [item.run for item in trained]
    best = max(trained, key=lambda item: (item.run.n_correct, -item.run.final_loss))
    calls = _calls(best.stack, crowd.rows)

    means = crowd.scaler.scalings
    return CarveResponse(
        hidden_widths=list(hidden_widths),
        n_parameters=sum(
            _parameter_count(layer)
            for layer in best.stack
            if isinstance(layer, DenseLayer)
        ),
        learning_rate=CARVE_LEARNING_RATE,
        epochs=CARVE_EPOCHS,
        runs=runs,
        best_seed=best.run.seed,
        best_accuracy=best.run.accuracy,
        mean_accuracy=float(np.mean([run.accuracy for run in runs])),
        seeds_reaching_everyone=sum(run.n_correct == len(CROWD) for run in runs),
        regions=_region_of(best.stack),
        folds=_folds_of(best.stack),
        people=[
            CarvedPerson(
                height=float(row[0]),
                weight=float(row[1]),
                is_adult=row[2],
                called_adult=int(call),
            )
            for row, call in zip(CROWD, calls, strict=True)
        ],
        mean_height=float(means.scaling_for("height").mean),
        deviation_height=float(means.scaling_for("height").standard_deviation),
        mean_weight=float(means.scaling_for("weight").mean),
        deviation_weight=float(means.scaling_for("weight").standard_deviation),
        straight_line_accuracy=_straight_line_accuracy(),
    )


@router.post("/concepts/dense-layers/carve", response_model=CarveResponse)
def carve(request: CarveRequest) -> CarveResponse:
    """Train rectifier networks of the asked widths on the crowd, several starts each."""
    return _carved(tuple(request.hidden_widths), request.n_seeds)
