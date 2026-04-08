"""One kernel swept across a small picture, for the convolution page.

The page hands the reader an eight by eight picture to paint and a three by
three kernel to edit, and asks the library's ``Conv2d`` what the sweep
answers. The layer is built with one filter, an identity bend and a zero bias,
then handed the reader's kernel through ``with_parameters``, so what it
answers is the bare cross-correlation sum and nothing else, which is the sum
the page works by hand. Every cell of the answer comes back with the window
that produced it, where that window sat on the picture, the nine values it
read and the nine products it summed, so the widget can light a window up on
hover and lay the arithmetic out without doing any of it.

The same request also counts parameters, because the count is a fact about
the layer that was just built rather than a separate question. A convolution
holds its kernel bank plus one bias per filter, read off the layer itself. Two
layers are costed beside it, both read off the convolution's own
``LayerShape``. The dense layer of equal width reads the same numbers and
answers the same numbers, so it holds one weight per input per output plus
one bias per output. The unshared layer keeps the window but gives every
answer cell its own kernel and its own bias, which is locality without
sharing, and it exists so the page can say which of the two ideas buys what.
At the page's eight by eight picture the dense layer holds 234 times what the
convolution does. The second endpoint counts any small picture the same way
without running it, so the page can reach the library's own example, a 28 by
28 picture through eight filters, where the figure is 53,066.

Stride and padding are settings of the sweep rather than of the picture, so
both endpoints take them. A window that does not fit is refused by the library
in its own words with the arithmetic written out, and the API adds no refusal
of its own, because there is no ask here the library would accept and the
page would not.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import Conv2d, Identity
from pydantic import BaseModel, Field

router = APIRouter(tags=["convolution"])

PICTURE_SIDE = 8
"""The picture the reader paints is this many cells along each side."""

KERNEL_SIDE = 3
"""The kernel the reader edits is this many cells along each side."""

CHANNELS = 1
"""The picture is a single channel, a grey level per cell."""

MAX_KERNEL_WEIGHT = 10.0
"""How far from zero one kernel entry may sit.

Not a numerical limit, since the library would sum far larger products without
complaint, but a bound on what a teaching widget draws. The presets all stay
within five, and a sharpening kernel at ten is already past anything a reader
would want to see on an eight by eight picture.
"""

MAX_SWEEP_STRIDE = 2
"""The largest stride the picture endpoint accepts.

At three the eight by eight picture answers with two cells a side, which
leaves nothing to hover, so the widget stops at two.
"""

MAX_SWEEP_PADDING = 1
"""The largest padding the picture endpoint accepts.

One row of zeros is what makes the answer the same size as the picture, which
is the case the page teaches. Two rows would put windows reading nothing but
border on the map.
"""

FIXED_SEED = 0
"""The seed the layer draws its starting kernels from.

The draw is replaced by the reader's kernel before anything is computed, so
the seed decides nothing the page shows. It is fixed so that the layer built
is always the same layer.
"""

MIN_COUNTED_SIDE = 4
MAX_COUNTED_SIDE = 32
MAX_COUNTED_CHANNELS = 3
MAX_COUNTED_FILTERS = 16
MAX_COUNTED_KERNEL_SIDE = 7
MAX_COUNTED_STRIDE = 3
MAX_COUNTED_PADDING = 3
"""The bounds on the counting endpoint.

Wide enough to reach the library's own example of a 28 by 28 picture through
eight filters of three by three, and a colour picture with three channels. A
kernel side of seven against a picture side of four is inside the bounds on
purpose, because that is the window the library refuses and the page shows
the refusal.
"""

PixelValue = Annotated[float, Field(ge=0.0, le=1.0)]
PictureRow = Annotated[
    list[PixelValue], Field(min_length=PICTURE_SIDE, max_length=PICTURE_SIDE)
]
KernelWeight = Annotated[float, Field(ge=-MAX_KERNEL_WEIGHT, le=MAX_KERNEL_WEIGHT)]
KernelRow = Annotated[
    list[KernelWeight], Field(min_length=KERNEL_SIDE, max_length=KERNEL_SIDE)
]


class ApplyRequest(BaseModel):
    """The picture, the kernel, and how the window moves across the picture."""

    picture: list[PictureRow] = Field(min_length=PICTURE_SIDE, max_length=PICTURE_SIDE)
    kernel: list[KernelRow] = Field(min_length=KERNEL_SIDE, max_length=KERNEL_SIDE)
    stride: int = Field(default=1, ge=1, le=MAX_SWEEP_STRIDE)
    padding: int = Field(default=0, ge=0, le=MAX_SWEEP_PADDING)

    model_config = {"extra": "forbid"}


class WindowReading(BaseModel):
    """The window behind one cell of the answer.

    ``top`` and ``left`` are the picture row and column of the window's first
    cell, and are negative where the window begins on the border of zeros
    that padding adds. ``patch`` is what the window read, border zeros
    included, and ``products`` is that patch times the kernel, entry by
    entry, whose sum is the answer cell.
    """

    top: int
    left: int
    patch: list[list[float]]
    products: list[list[float]]


class ParameterCount(BaseModel):
    """What one convolution costs beside two layers answering the same size.

    ``reads`` and ``answers`` are the layer's own extents, and ``n_inputs``
    and ``n_outputs`` are how many numbers each side holds, the products of
    those extents. ``convolution`` is the kernel bank plus one bias per
    filter. ``unshared`` is what a layer would hold if it kept the window but
    gave every answer cell its own kernel and its own bias, which is locality
    without sharing. ``dense`` is a dense layer reading the same numbers and
    answering the same numbers, one weight per input per output plus one bias
    per output. ``ratio`` is ``dense`` over ``convolution``.
    """

    reads: list[int]
    answers: list[int]
    n_inputs: int
    n_outputs: int
    convolution: int
    unshared: int
    dense: int
    ratio: float


class ApplyResponse(BaseModel):
    """The answer, every window behind it, and what the layer cost."""

    output: list[list[float]]
    output_height: int
    output_width: int
    windows: list[list[WindowReading]]
    parameters: ParameterCount


def _parameter_count(layer: Conv2d) -> ParameterCount:
    """Count the layer's parameters and those of the two layers beside it."""
    shape = layer.shape
    fan_in = shape.reads[0] * layer.kernel_size * layer.kernel_size
    convolution = int(layer.kernels.size + layer.bias_vector.size)
    unshared = int(shape.n_outputs * (fan_in + 1))
    dense = int(shape.n_outputs * shape.n_inputs + shape.n_outputs)
    return ParameterCount(
        reads=[int(extent) for extent in shape.reads],
        answers=[int(extent) for extent in shape.answers],
        n_inputs=int(shape.n_inputs),
        n_outputs=int(shape.n_outputs),
        convolution=convolution,
        unshared=unshared,
        dense=dense,
        ratio=float(dense / convolution),
    )


def _windows_behind(
    picture: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    output_height: int,
    output_width: int,
) -> list[list[WindowReading]]:
    """Every window the sweep read, laid out like the answer.

    The border of zeros is added here the way the library adds it, so a
    window that begins on the border reads zeros there and reports a negative
    row or column for where it began.
    """
    bordered = np.pad(picture, padding)
    windows: list[list[WindowReading]] = []
    for output_row in range(output_height):
        top = output_row * stride
        row_of_windows: list[WindowReading] = []
        for output_column in range(output_width):
            left = output_column * stride
            patch = bordered[top : top + KERNEL_SIDE, left : left + KERNEL_SIDE]
            # A zero pixel under a negative weight gives a negative zero, and
            # adding zero turns it back into the plain zero the page prints.
            products = patch * kernel + 0.0
            row_of_windows.append(
                WindowReading(
                    top=top - padding,
                    left=left - padding,
                    patch=[[float(value) for value in row] for row in patch],
                    products=[[float(value) for value in row] for row in products],
                )
            )
        windows.append(row_of_windows)
    return windows


@router.post("/concepts/convolution/apply", response_model=ApplyResponse)
def apply(request: ApplyRequest) -> ApplyResponse:
    """Sweep the kernel across the picture and report every window it read."""
    picture = np.asarray(request.picture, dtype=np.float64)
    kernel = np.asarray(request.kernel, dtype=np.float64)

    layer = Conv2d(
        reads=(CHANNELS, PICTURE_SIDE, PICTURE_SIDE),
        n_filters=1,
        kernel_size=KERNEL_SIDE,
        activation=Identity(),
        stride=request.stride,
        padding=request.padding,
        random_seed=FIXED_SEED,
    ).with_parameters(
        kernel.reshape(1, CHANNELS, KERNEL_SIDE, KERNEL_SIDE),
        np.zeros(1, dtype=np.float64),
    )

    block = picture.reshape(1, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)
    output = np.asarray(layer.respond_to(block).outputs, dtype=np.float64)[0, 0]
    output_height, output_width = layer.shape.answers[1:]

    return ApplyResponse(
        output=[[float(value) for value in row] for row in output],
        output_height=int(output_height),
        output_width=int(output_width),
        windows=_windows_behind(
            picture,
            kernel,
            request.stride,
            request.padding,
            int(output_height),
            int(output_width),
        ),
        parameters=_parameter_count(layer),
    )


class ParameterCountRequest(BaseModel):
    """A square picture and a kernel bank, to be costed rather than run."""

    channels: int = Field(default=CHANNELS, ge=1, le=MAX_COUNTED_CHANNELS)
    side: int = Field(ge=MIN_COUNTED_SIDE, le=MAX_COUNTED_SIDE)
    n_filters: int = Field(ge=1, le=MAX_COUNTED_FILTERS)
    kernel_size: int = Field(ge=1, le=MAX_COUNTED_KERNEL_SIDE)
    stride: int = Field(default=1, ge=1, le=MAX_COUNTED_STRIDE)
    padding: int = Field(default=0, ge=0, le=MAX_COUNTED_PADDING)

    model_config = {"extra": "forbid"}


@router.post("/concepts/convolution/parameter-count", response_model=ParameterCount)
def parameter_count(request: ParameterCountRequest) -> ParameterCount:
    """Build the layer and count it beside the two layers of equal width."""
    layer = Conv2d(
        reads=(request.channels, request.side, request.side),
        n_filters=request.n_filters,
        kernel_size=request.kernel_size,
        activation=Identity(),
        stride=request.stride,
        padding=request.padding,
        random_seed=FIXED_SEED,
    )
    return _parameter_count(layer)
