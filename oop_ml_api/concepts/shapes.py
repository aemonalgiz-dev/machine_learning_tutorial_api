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
    Identity,
    Layer,
    LayerStack,
    MaxPool2d,
    MLLibError,
    Neuron,
    RectifiedLinear,
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
    """The layers, in order, from the input end."""

    layers: list[LayerSpec] = Field(min_length=1, max_length=MAX_LAYERS)

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
