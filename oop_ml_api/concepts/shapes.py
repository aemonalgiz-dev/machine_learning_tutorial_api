"""The shape guarantee, for the shapes-and-flattening page.

A network that cannot work is refused before it reads a single row. Every
layer in the library states what it reads and what it answers with as a tuple
of extents, and ``LayerStack`` refuses to exist unless each layer reads exactly
what the layer beneath it answers. The page's widget is a stack builder, and
this module is the one endpoint behind it. It takes an ordered list of layer
specifications from a closed set, convolution, pooling, flatten and dense,
builds each one with the library, and reports every layer's reads and answers,
every join's verdict, and the library's own message for the first thing that
went wrong.

What a layer reads, when the request does not say
--------------------------------------------------
The first layer must state what it reads, because there is nothing beneath it
to read from. Every layer above it may leave ``reads`` unstated, and then it
is told what arrives. A convolution, a pooling layer and a flatten are told
the arrangement the layer beneath answers with, extents and all. A dense layer
is told only a count, as many numbers as arrive, because a dense layer is
built from neurons and a neuron holds a width, not a picture. That is the
natural mistake the page is about. A dense layer placed straight after a
pooling layer that answers ``(4, 3, 3)`` reads ``(36,)``, holds the right
number of weights, and is still refused, because the arrangement disagrees. A
flatten between them is the join said out loud.

A stated ``reads`` overrides what arrives, so the reader can also build the
other refusal, a dense layer reading ``(784,)`` after a convolution answering
``(8, 26, 26)``, where the counts disagree as well and the message says less.

How the verdicts are found
--------------------------
Each join's verdict is the library's own predicate, ``LayerShape.follows``,
asked one join at a time. The message is the library's too. The stack is
attempted only up to the first join that fails, so the refusal it raises names
that join with the same numbering the widget shows, and no message is composed
here. The library checks joins from the bottom and stops at the first bad one,
which is why only that join carries a message and the joins above it carry
their verdict alone.

A layer that cannot be built at all, a window wider than the picture it reads
or a convolution handed a row, is reported with the constructor's refusal, and
every layer above it that was going to read what it answered is left
unsettled rather than guessed at.

Why the dense layer's weights are zeros
---------------------------------------
Nothing here reads a weight. A dense layer's shape is its neuron count over
its neurons' width, both settled before any weight is, so the neurons hold
zeros and the layer is built only to be asked its shape and to be stacked.

The four endpoints beside the stack check
------------------------------------------
The stack check answers what a whole chain does. Three questions the page also
asks are narrower than a chain, and one is about what a layer does to numbers
rather than to arrangements.

``/concepts/shapes/extents`` builds a bare ``LayerShape`` and reports its two
sides and their counts, or the library's refusal. It is the one endpoint that
can be handed an extent of zero or a side with no extents at all, because a
layer constructor would never get that far.

``/concepts/shapes/join`` asks the library's own join predicate about two
arrangements, and then asks a stack of two layers so the refusal is the
library's words rather than this module's. A layer answering an arbitrary
arrangement has to be synthesised: a row of ``n`` is what a flattening layer of
``(n,)`` answers with, and a picture ``(c, h, w)`` is what a convolution of
``c`` kernels of one by one over a picture ``(1, h, w)`` answers with. Two
extents have no such layer, and are refused here in this module's words.

``/concepts/shapes/tied-weights`` is the weakness the library records rather
than defends. A stack steps by rebuilding each position from that position's
gradient, so one layer object placed twice comes back as two layers with
different weights after a single step. This endpoint builds exactly that stack
and reports both positions' weights afterwards.

``/concepts/shapes/whole-numbers`` is the other recorded weakness. Every block
a layer reads is coerced to float64 and scanned for non-finite entries, so a
value meant as an index is never checked for being whole or for being in range,
and two whole numbers too large for float64 to tell apart arrive as one. The
endpoint hands a flattening layer a row of numbers and reports what comes back.
"""

from __future__ import annotations

from math import prod
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    AveragePool2d,
    Conv2d,
    DenseLayer,
    Flatten,
    FloatArray,
    Identity,
    Layer,
    LayerShape,
    LayerStack,
    MaxPool2d,
    MLLibError,
    Neuron,
    RectifiedLinear,
    SquaredError,
)
from pydantic import BaseModel, Field

router = APIRouter(tags=["shapes-and-flattening"])

MAX_LAYERS = 8
"""The most layers one stack may hold. Enough for a picture stage, a bridge and
a row stage twice over."""

MAX_EXTENT = 32
"""The largest channel count, height or width a stated picture may have. A
teaching picture is small, and nothing here ever allocates one, but a
convolution's kernels are allocated and their count grows with the extents."""

MAX_WIDTH = 16_384
"""The widest row a dense layer may be told to read, stated or arrived, which
is the widest row a picture inside the bounds flattens to, sixteen channels of
thirty-two by thirty-two. A dense layer is the one layer here that allocates
its width, once per neuron."""

MAX_FILTERS = 16
"""The most kernels one convolution may sweep."""

MAX_NEURONS = 64
"""The most neurons one dense layer may hold."""

MAX_PADDING = 8
"""The most rows and columns of zeros a convolution may add at each edge."""

PictureExtent = Annotated[int, Field(ge=1, le=MAX_EXTENT)]
RowExtent = Annotated[int, Field(ge=1, le=MAX_WIDTH)]


class ConvSpec(BaseModel):
    """A bank of kernels swept across a picture.

    ``reads`` is ``(channels, height, width)`` for the first layer and may be
    left unstated above it. Between one and three extents are accepted rather
    than exactly three, because the library refuses any other arrangement in
    its own words and the page wants those words. The bound here is only on
    how large an extent may be.
    """

    kind: Literal["conv"]
    reads: list[PictureExtent] | None = Field(default=None, min_length=1, max_length=3)
    n_filters: int = Field(ge=1, le=MAX_FILTERS)
    kernel_size: int = Field(ge=1, le=MAX_EXTENT)
    stride: int = Field(default=1, ge=1, le=MAX_EXTENT)
    padding: int = Field(default=0, ge=0, le=MAX_PADDING)

    model_config = {"extra": "forbid"}


class PoolSpec(BaseModel):
    """A window slid over each channel, keeping one number per window."""

    kind: Literal["pool"]
    reads: list[PictureExtent] | None = Field(default=None, min_length=1, max_length=3)
    summary: Literal["max", "average"] = "max"
    window: int = Field(default=2, ge=1, le=MAX_EXTENT)
    stride: int = Field(default=2, ge=1, le=MAX_EXTENT)

    model_config = {"extra": "forbid"}


class FlattenSpec(BaseModel):
    """The bridge from an arrangement to a row. It has no settings.

    Its extents are bounded as a row's rather than a picture's, so the page
    can state a rearrangement of the same count, ``(4, 52, 26)`` against a
    convolution's ``(8, 26, 26)``, and hear the library refuse it.
    """

    kind: Literal["flatten"]
    reads: list[RowExtent] | None = Field(default=None, min_length=1, max_length=3)

    model_config = {"extra": "forbid"}


class DenseSpec(BaseModel):
    """Neurons that all read the same row.

    ``reads`` is one extent, the width of the row. Left unstated, the layer
    reads as many numbers as arrive, whatever their arrangement, which is the
    mistake the page exists to show.
    """

    kind: Literal["dense"]
    reads: list[RowExtent] | None = Field(default=None, min_length=1, max_length=3)
    n_neurons: int = Field(ge=1, le=MAX_NEURONS)

    model_config = {"extra": "forbid"}


LayerSpec = Annotated[
    ConvSpec | PoolSpec | FlattenSpec | DenseSpec, Field(discriminator="kind")
]


class CheckRequest(BaseModel):
    """The layers, in order, from the input end.

    An empty list is allowed through rather than bounded away, so that the page
    can quote the library's own words for a stack with nothing in it instead of
    this module's.
    """

    layers: list[LayerSpec] = Field(min_length=0, max_length=MAX_LAYERS)

    model_config = {"extra": "forbid"}


class LayerReport(BaseModel):
    """What one layer reads and answers with, or why it could not be built.

    ``reads`` is what the layer was told to read, stated or arrived. Both
    extents and counts are reported, because the page's whole point is that
    the two are different facts. A layer the library refused has ``refusal``
    set and no answers. A layer left unsettled, because the layer beneath it
    was refused and it had nothing to read, has neither.
    """

    kind: Literal["conv", "pool", "flatten", "dense"]
    reads: list[int] | None
    answers: list[int] | None
    n_reads: int | None
    n_answers: int | None
    refusal: str | None


class JoinReport(BaseModel):
    """The seam between layer ``position`` and the layer above it.

    ``message`` is the library's refusal, carried only by the join the stack
    stopped at. The joins above a failed one still get a verdict, from the
    same predicate the stack uses, but the library never reaches them.
    """

    position: int
    verdict: Literal["holds", "fails", "unsettled"]
    message: str | None


class CheckResponse(BaseModel):
    """Every layer, every join, and whether the stack exists.

    ``reads`` and ``answers`` are the built stack's own report of itself, the
    first layer's arrangement and the last layer's, and are absent when it
    was refused. ``refusal`` is the library's message for the first thing that
    went wrong, walking up from the input end.
    """

    layers: list[LayerReport]
    joins: list[JoinReport]
    holds: bool
    reads: list[int] | None
    answers: list[int] | None
    refusal: str | None


@router.post("/concepts/shapes/check", response_model=CheckResponse)
def check(request: CheckRequest) -> CheckResponse:
    """Build every layer, ask every join, and report what the library said.

    One walk from the input end. Each layer is told what it reads, built or
    refused, and its join with the layer beneath is settled before the next
    layer is looked at, so the first thing wrong is met first and is the only
    thing that gets a message.
    """
    if not request.layers:
        return CheckResponse(
            layers=[],
            joins=[],
            holds=False,
            reads=None,
            answers=None,
            refusal=_stack_refusal([]),
        )

    if request.layers[0].reads is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "the first layer has nothing beneath it to read from, so it must "
                "state what it reads"
            ),
        )

    built: list[Layer | None] = []
    reports: list[LayerReport] = []
    joins: list[JoinReport] = []
    refusal: str | None = None

    for position, specification in enumerate(request.layers):
        beneath = built[-1] if built else None
        reads = _reads_for(specification, beneath)

        layer: Layer | None = None
        if reads is None:
            report = _unsettled(specification)
        else:
            try:
                layer = _build(specification, reads)
                report = _built(specification, layer)
            except MLLibError as error:
                report = _refused(specification, reads, str(error))

        if position > 0:
            join = _join(position - 1, beneath, layer)
            if join.verdict == "fails" and refusal is None and layer is not None:
                refusal = _stack_refusal([*_settled(built), layer])
                join = JoinReport(
                    position=join.position, verdict="fails", message=refusal
                )
            joins.append(join)

        if report.refusal is not None and refusal is None:
            refusal = report.refusal

        built.append(layer)
        reports.append(report)

    if refusal is not None:
        return CheckResponse(
            layers=reports,
            joins=joins,
            holds=False,
            reads=None,
            answers=None,
            refusal=refusal,
        )

    stack = LayerStack(_settled(built))
    return CheckResponse(
        layers=reports,
        joins=joins,
        holds=True,
        reads=list(stack.shape.reads),
        answers=list(stack.shape.answers),
        refusal=None,
    )


def _reads_for(
    specification: LayerSpec, beneath: Layer | None
) -> tuple[int, ...] | None:
    """What a layer is told to read, stated or arrived.

    A stated ``reads`` wins. Otherwise a dense layer is told the count of what
    the layer beneath answers, and every other kind is told the arrangement.
    Nothing arrives from a layer that was not built, so the answer is ``None``
    and the layer is left unsettled.

    Two refusals are this module's rather than the library's, because the
    library cannot be asked either question. A dense layer is built from
    neurons, and a neuron holds a width, so there is no way to hand one a
    picture. And a row wider than the bound would be built happily, one
    allocation per neuron, which is the one cost in this module that grows
    with the request.
    """
    if specification.reads is not None:
        if specification.kind == "dense" and len(specification.reads) != 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    "a dense layer reads a row, so its reads is one extent, not "
                    f"{tuple(specification.reads)}"
                ),
            )
        return tuple(specification.reads)

    if beneath is None:
        return None

    if specification.kind == "dense":
        arriving = beneath.shape.n_outputs
        if arriving > MAX_WIDTH:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"a dense layer here reads a row of at most {MAX_WIDTH} "
                    f"numbers, and {arriving} arrive"
                ),
            )
        return (arriving,)

    return beneath.shape.answers


def _build(specification: LayerSpec, reads: tuple[int, ...]) -> Layer:
    """The library's layer for one specification, or the library's refusal."""
    if specification.kind == "conv":
        return Conv2d(
            reads=reads,
            n_filters=specification.n_filters,
            kernel_size=specification.kernel_size,
            activation=RectifiedLinear(),
            stride=specification.stride,
            padding=specification.padding,
        )

    if specification.kind == "pool":
        pooling = MaxPool2d if specification.summary == "max" else AveragePool2d
        return pooling(
            reads=reads, window=specification.window, stride=specification.stride
        )

    if specification.kind == "flatten":
        return Flatten(reads=reads)

    width = reads[0]
    return DenseLayer(
        [
            Neuron(weights=np.zeros(width), bias=0.0, activation=Identity())
            for _ in range(specification.n_neurons)
        ]
    )


def _built(specification: LayerSpec, layer: Layer) -> LayerReport:
    """A layer the library accepted, described by its own shape."""
    return LayerReport(
        kind=specification.kind,
        reads=list(layer.shape.reads),
        answers=list(layer.shape.answers),
        n_reads=layer.shape.n_inputs,
        n_answers=layer.shape.n_outputs,
        refusal=None,
    )


def _refused(
    specification: LayerSpec, reads: tuple[int, ...], refusal: str
) -> LayerReport:
    """A layer the library would not build, with what it was told to read."""
    return LayerReport(
        kind=specification.kind,
        reads=list(reads),
        answers=None,
        n_reads=prod(reads),
        n_answers=None,
        refusal=refusal,
    )


def _unsettled(specification: LayerSpec) -> LayerReport:
    """A layer that had nothing to read, because the one beneath was refused."""
    return LayerReport(
        kind=specification.kind,
        reads=None,
        answers=None,
        n_reads=None,
        n_answers=None,
        refusal=None,
    )


def _join(position: int, beneath: Layer | None, above: Layer | None) -> JoinReport:
    """One seam's verdict, from the predicate the stack itself uses."""
    if beneath is None or above is None:
        return JoinReport(position=position, verdict="unsettled", message=None)

    holds = above.shape.follows(beneath.shape)
    return JoinReport(
        position=position, verdict="holds" if holds else "fails", message=None
    )


def _settled(built: list[Layer | None]) -> list[Layer]:
    """The layers the library built, in order, with the gaps left out."""
    return [layer for layer in built if layer is not None]


def _stack_refusal(layers: list[Layer]) -> str | None:
    """What the library says when asked to stack these layers, if anything.

    Called only for a prefix whose every layer was built and whose every join
    but the last holds, so the refusal names the last join and the numbering
    in it is the numbering the widget shows.
    """
    try:
        LayerStack(layers)
    except MLLibError as error:
        return str(error)

    return None


MAX_SIDE_EXTENTS = 4
"""The most extents one side of a stated shape may carry. Three is a picture,
one is a row, and the fourth is there so a reader can overshoot and still be
answered rather than bounded away."""

MIN_STATED_EXTENT = -8
"""How far below one a stated extent may go. The library refuses anything under
one in its own words, and this is far enough for the page to hear it say so
about zero and about a negative."""

MAX_TIED_WIDTH = 6
"""The widest square dense layer the weight-tying probe may build. Square,
because the layer stands at two positions of one chain and so must read exactly
what it answers with."""

MAX_TIED_ROWS = 8
"""The most rows one step of the weight-tying probe may be handed."""

MAX_TIED_MAGNITUDE = 100.0
"""How large any weight, bias, input or target in that probe may be."""

MAX_PROBED_NUMBERS = 8
"""The most numbers one coercion probe may hand a layer."""

MAX_WHOLE_NUMBER = 2**60
"""The largest whole number the coercion probe accepts.

Well past the 2 to the 53 where float64 stops being able to tell consecutive
whole numbers apart, which is what the probe is for, and well short of anything
that would make the arithmetic slow."""

MAX_FRACTIONAL_MAGNITUDE = 1_000_000.0
"""How large a fractional value handed to that probe may be."""

StatedExtent = Annotated[int, Field(ge=MIN_STATED_EXTENT, le=MAX_WIDTH)]
JoinExtent = Annotated[int, Field(ge=1, le=MAX_WIDTH)]
TiedValue = Annotated[float, Field(ge=-MAX_TIED_MAGNITUDE, le=MAX_TIED_MAGNITUDE)]
WholeNumber = Annotated[int, Field(ge=-MAX_WHOLE_NUMBER, le=MAX_WHOLE_NUMBER)]
FractionalValue = Annotated[
    float, Field(ge=-MAX_FRACTIONAL_MAGNITUDE, le=MAX_FRACTIONAL_MAGNITUDE)
]


class ExtentsRequest(BaseModel):
    """Two sides of a shape, stated as extents and nothing else.

    Both sides may be empty and both may carry an extent below one, because
    those are two of the refusals the page wants in the library's own words and
    no layer constructor would let a request reach them.
    """

    reads: list[StatedExtent] = Field(min_length=0, max_length=MAX_SIDE_EXTENTS)
    answers: list[StatedExtent] = Field(min_length=0, max_length=MAX_SIDE_EXTENTS)

    model_config = {"extra": "forbid"}


class ExtentsResponse(BaseModel):
    """What the shape holds, or why the library would not build it.

    ``n_reads`` and ``n_answers`` are the products of the extents, which is how
    many numbers one input and one output hold. They are the fact the page
    keeps apart from the arrangement, and they are absent when the shape was
    refused.
    """

    reads: list[int] | None
    answers: list[int] | None
    n_reads: int | None
    n_answers: int | None
    refusal: str | None


class JoinRequest(BaseModel):
    """One seam, stated as the arrangement each side of it carries.

    ``beneath`` is what the lower layer answers with and ``above`` is what the
    upper layer reads. One extent is a row and three are a picture. Two are
    refused, because no layer here answers with a two-extent arrangement and
    the seam has to be built out of layers for the refusal to be the
    library's.
    """

    beneath: list[JoinExtent] = Field(min_length=1, max_length=3)
    above: list[JoinExtent] = Field(min_length=1, max_length=3)

    model_config = {"extra": "forbid"}


class JoinResponse(BaseModel):
    """The verdict, both counts, and the library's words when it refuses.

    ``counts_agree`` is reported beside ``holds`` rather than folded into it,
    because the pair of them is the page's subject. A seam where the counts
    agree and the verdict is still false is the case the whole check exists
    for.
    """

    beneath: list[int]
    above: list[int]
    n_beneath: int
    n_above: int
    counts_agree: bool
    holds: bool
    refusal: str | None


class TiedWeightsRequest(BaseModel):
    """One square dense layer, placed at two positions of one chain.

    The weights are square so the layer reads exactly what it answers with,
    which is what lets one object occupy both positions. The rows and targets
    are one step's worth of data, and the step is taken once.
    """

    weights: list[list[TiedValue]] = Field(min_length=1, max_length=MAX_TIED_WIDTH)
    biases: list[TiedValue] = Field(min_length=1, max_length=MAX_TIED_WIDTH)
    rows: list[list[TiedValue]] = Field(min_length=1, max_length=MAX_TIED_ROWS)
    targets: list[list[TiedValue]] = Field(min_length=1, max_length=MAX_TIED_ROWS)
    learning_rate: float = Field(default=0.1, gt=0.0, le=10.0)

    model_config = {"extra": "forbid"}


class TiedWeightsResponse(BaseModel):
    """What one shared layer looked like before the step and after it.

    ``same_object_before`` and ``same_object_after`` are identity rather than
    equality. The chain really does hold one object at both positions going in,
    and the question is whether it still does coming out.
    """

    width: int
    loss: float
    same_object_before: bool
    same_object_after: bool
    weights_before: list[list[float]]
    first_after: list[list[float]]
    second_after: list[list[float]]
    largest_gap: float


class NumberReport(BaseModel):
    """One number as it was asked for and as the layer answered with it.

    Both are written as decimal strings. A whole number past 2 to the 53 cannot
    survive a browser's own numbers either, so text is the only way the page can
    show two of them apart.
    """

    asked: str
    answered: str
    unchanged: bool
    whole: bool


class WholeNumberRequest(BaseModel):
    """Numbers to hand a layer, in two groups that go wrong differently.

    ``whole_numbers`` are exact integers on the way in, so anything lost is
    lost by the coercion rather than by the request. ``fractional_values`` are
    what an index is not, and are handed over to see whether anything objects.
    """

    whole_numbers: list[WholeNumber] = Field(
        min_length=1, max_length=MAX_PROBED_NUMBERS
    )
    fractional_values: list[FractionalValue] = Field(
        default_factory=list, max_length=MAX_PROBED_NUMBERS
    )

    model_config = {"extra": "forbid"}


class WholeNumberResponse(BaseModel):
    """What came back, and how many distinct values are left.

    ``n_distinct_after`` against ``n_whole_asked`` is the measurement, since two
    whole numbers the request tells apart may be two the layer cannot.
    """

    whole_numbers: list[NumberReport]
    fractional_values: list[NumberReport]
    n_whole_asked: int
    n_distinct_after: int
    refusal: str | None


@router.post("/concepts/shapes/extents", response_model=ExtentsResponse)
def extents(request: ExtentsRequest) -> ExtentsResponse:
    """Build a bare shape and report its sides, or the library's refusal."""
    try:
        shape = LayerShape(n_inputs=request.reads, n_outputs=request.answers)
    except MLLibError as error:
        return ExtentsResponse(
            reads=None, answers=None, n_reads=None, n_answers=None, refusal=str(error)
        )

    return ExtentsResponse(
        reads=list(shape.reads),
        answers=list(shape.answers),
        n_reads=shape.n_inputs,
        n_answers=shape.n_outputs,
        refusal=None,
    )


@router.post("/concepts/shapes/join", response_model=JoinResponse)
def join(request: JoinRequest) -> JoinResponse:
    """Ask the library whether one arrangement may follow another.

    The verdict is the shape's own predicate. The refusal is a stack of the two
    synthesised layers, so the words are the ones a reader would meet building
    the same seam out of layers they chose themselves.
    """
    beneath = _layer_answering(request.beneath)
    above = Flatten(reads=request.above)

    return JoinResponse(
        beneath=list(beneath.shape.answers),
        above=list(above.shape.reads),
        n_beneath=beneath.shape.n_outputs,
        n_above=above.shape.n_inputs,
        counts_agree=beneath.shape.n_outputs == above.shape.n_inputs,
        holds=above.shape.follows(beneath.shape),
        refusal=_stack_refusal([beneath, above]),
    )


@router.post("/concepts/shapes/tied-weights", response_model=TiedWeightsResponse)
def tied_weights(request: TiedWeightsRequest) -> TiedWeightsResponse:
    """Step a chain holding one layer twice, and report both positions after.

    The chain is built, run forward, scored, walked back and stepped once, all
    through the library. Nothing here compares the two positions until the
    library has handed back the stepped chain.
    """
    width = len(request.weights)
    if len(request.biases) != width:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a layer of {width} neurons carries {width} biases, got "
                f"{len(request.biases)}"
            ),
        )
    if any(len(row) != width for row in request.weights):
        raise HTTPException(
            status_code=422,
            detail=(
                "this layer stands at two positions of one chain, so it reads "
                f"exactly what it answers with and its weights are {width} by "
                f"{width}"
            ),
        )
    if len(request.rows) != len(request.targets):
        raise HTTPException(
            status_code=422,
            detail=(
                f"one target per row, got {len(request.rows)} rows and "
                f"{len(request.targets)} targets"
            ),
        )
    if any(len(row) != width for row in [*request.rows, *request.targets]):
        raise HTTPException(
            status_code=422,
            detail=f"every row and every target holds {width} numbers",
        )

    shared = DenseLayer(
        [
            Neuron(weights=weights, bias=bias, activation=Identity())
            for weights, bias in zip(request.weights, request.biases, strict=True)
        ]
    )
    chain = LayerStack([shared, shared])

    backward = chain.backward_pass(
        np.array(request.rows, dtype=np.float64),
        np.array(request.targets, dtype=np.float64),
        SquaredError(),
    )
    stepped = chain.stepped_by(backward, request.learning_rate)

    first = stepped[0]
    second = stepped[1]
    assert isinstance(first, DenseLayer)
    assert isinstance(second, DenseLayer)

    return TiedWeightsResponse(
        width=width,
        loss=float(backward.loss),
        same_object_before=chain[0] is chain[1],
        same_object_after=first is second,
        weights_before=_as_rows(shared.weight_matrix),
        first_after=_as_rows(first.weight_matrix),
        second_after=_as_rows(second.weight_matrix),
        largest_gap=float(np.max(np.abs(first.weight_matrix - second.weight_matrix))),
    )


@router.post("/concepts/shapes/whole-numbers", response_model=WholeNumberResponse)
def whole_numbers(request: WholeNumberRequest) -> WholeNumberResponse:
    """Hand a layer a row of numbers and report what it answers with.

    A flattening layer over a row changes nothing about the numbers, so
    whatever differs between what was asked for and what came back was done by
    the coercion at the layer's own boundary and by nothing else.
    """
    whole = _through_a_layer([float(value) for value in request.whole_numbers])
    fractional = _through_a_layer(list(request.fractional_values))

    return WholeNumberResponse(
        whole_numbers=[
            _number_report(str(asked), answered)
            for asked, answered in zip(request.whole_numbers, whole, strict=True)
        ],
        fractional_values=[
            _number_report(_decimal(asked), answered)
            for asked, answered in zip(
                request.fractional_values, fractional, strict=True
            )
        ],
        n_whole_asked=len(set(request.whole_numbers)),
        n_distinct_after=len(set(whole)),
        refusal=None,
    )


def _layer_answering(arrangement: list[int]) -> Layer:
    """A layer whose answers are exactly this arrangement.

    A row of ``n`` is what a flattening layer of ``(n,)`` answers with. A
    picture ``(c, h, w)`` is what a convolution of ``c`` kernels of one by one
    over a picture ``(1, h, w)`` answers with, since a window of one at stride
    one reaches every position and leaves both sides the length they were.
    """
    if len(arrangement) == 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "a seam here is stated as a row of numbers or as a picture of "
                "channels, height and width, and no layer answers with two "
                "extents"
            ),
        )

    if len(arrangement) == 1:
        return Flatten(reads=arrangement)

    channels, height, width = arrangement
    if channels > MAX_FILTERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a picture here carries at most {MAX_FILTERS} channels, got {channels}"
            ),
        )
    if height > MAX_EXTENT or width > MAX_EXTENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a picture here is at most {MAX_EXTENT} by {MAX_EXTENT}, got "
                f"{height} by {width}"
            ),
        )

    return Conv2d(
        reads=(1, height, width),
        n_filters=channels,
        kernel_size=1,
        activation=Identity(),
        stride=1,
        padding=0,
    )


def _through_a_layer(values: list[float]) -> list[float]:
    """One row pushed through a flattening layer, which changes no number."""
    if not values:
        return []

    layer = Flatten(reads=len(values))
    response = layer.respond_to(np.array([values], dtype=np.float64))
    return [float(value) for value in response.outputs[0]]


def _decimal(value: float) -> str:
    """A float as the shortest decimal text that means it.

    A whole number loses its trailing zero, so that a number asked for as an
    integer and answered with unchanged reads the same on both sides and the
    ones that did change are the only ones that differ.
    """
    return str(int(value)) if value.is_integer() else repr(value)


def _number_report(asked: str, answered: float) -> NumberReport:
    """One number as asked for and as answered with, and what survived."""
    text = _decimal(answered)
    return NumberReport(
        asked=asked, answered=text, unchanged=text == asked, whole=answered.is_integer()
    )


def _as_rows(block: FloatArray) -> list[list[float]]:
    """A weight matrix as plain rows of Python floats."""
    return [[float(value) for value in row] for row in block]
