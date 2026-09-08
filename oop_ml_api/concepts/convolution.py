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

The arrangement endpoint is the page's opening claim, measured. It shuffles
the picture's sixty-four cells with a fixed permutation, rewires a dense
layer's weights by the same permutation, and reports that the dense layer's
answers move by 2.2e-16, which is one rounding step and not a difference of
meaning. The same shuffled picture through the sweep changes twenty-seven of
its thirty-six answers by as much as six. A dense layer was never told the
arrangement; the sweep is built out of it.

The shift endpoint measures translation equivariance rather than describing
it. The picture is moved sideways a column at a time and swept again, and the
answer is compared against the resting answer moved by the same amount, which
comes out equal to the last bit whatever kernel the layer holds. Beside it sit
two dense layers of the same width. One holds a single detector, the page's
kernel written into one window position and zeros everywhere else, and it
answers 4 where the square's edge sits under it and 0 at every other shift.
The other is an ordinary draw, and its answers are compared the same way after
being laid out as the sweep's map, which is a courtesy the layer has not
earned, since nothing in its arithmetic makes that numbering spatial.

The bank endpoint sweeps several kernels at once, so the page can say that one
layer holds a bank and answers with one map per filter, and the geometry
endpoint builds a layer for every kernel size, stride and padding the page
asks about and reads the arrangement off the layer's own shape without running
anything. It also counts how many windows read each cell of the picture, which
is what makes the border argument for padding a count rather than a claim.

The backward endpoint does two things the page quotes. It works the backward
pass on the page's own square with a slope of one arriving at every answer
cell, where every kernel entry's slope comes out as the number of lit cells
and every interior cell's blame comes out as the sum of the kernel entries, so
the accumulation can be checked by hand. And it checks the whole backward pass
against a central difference of the loss on three fixtures, plain, padded and
strided, over the kernels, the biases and the inputs alike.

The cost endpoint times three forward passes at three sizes. The first is the
library's convolution, which lays every window out as a view of the padded
picture and sums over channels and kernel positions in one contraction. The
second is the definition written as seven nested loops, held in this module
and sharing nothing with the layer but the numbers, and its answer is compared
with the layer's, which agree to 1.8e-15 at the largest size. The third is the
dense layer of equal width. The sweep's time barely moves between the page's
picture and the library's own example, because at these sizes its fixed
overhead is most of it, while the loops grow with the 324, 5,184 and 48,672
products they add up and the dense layer with its weights. So the sweep is
several times slower than the dense layer on the page's picture and several
times quicker at 28 by 28 through eight filters. Those figures are a
measurement of whichever machine served the request and move between runs,
which is why the tests pin the orderings, the agreement and the arrangement
rather than the milliseconds.

The edge-case endpoint probes every refusal the page's last part quotes,
including the join a stack refuses when a convolution is placed directly
beneath a layer that reads a row, whose message names both arrangements and
says that the counts agree.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Conv2d,
    DenseLayer,
    Flatten,
    Identity,
    LayerStack,
    Neuron,
    SquaredError,
)
from oop_ml.core.exceptions import MLLibError
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

MAX_GEOMETRY_CHOICES = 6
"""How many kernel sizes, strides and paddings one sweep may ask about.

Six of each would be two hundred and sixteen layers, which is more than a
reader looks down, so the page asks for far fewer; the bound is here so that
nothing can ask for more.
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


MAX_BANK_FILTERS = 6
"""How many kernels one bank may hold.

Six is enough to show that a bank answers with one map per filter and that the
maps disagree, and few enough that the maps fit beside one another on a page.
"""

MAX_SHIFTS = 6
"""How many sideways positions the shift endpoint sweeps the picture at."""

DENSE_DRAW_SEED = 3
"""Which draw the ordinary dense layer of equal width starts from.

Fixed, so that the equivariance comparison is the same comparison on every
page load. The layer is never trained, so the draw decides its answers
entirely, and pinning it is what lets a test quote them.
"""

SHUFFLE_SEED = 11
"""Which shuffle the arrangement endpoint scrambles the picture with."""

SQUARE_TOP = 2
SQUARE_SIDE = 4
"""Where the page's square sits, and how wide it is.

Rows and columns two through five of the eight by eight picture, which is the
square the playground opens with and the one every worked number on the page
is taken from.
"""

VERTICAL_EDGE_KERNEL = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
"""The kernel the page works by hand, negative left and positive right."""

DETECTOR_TOP = 2
DETECTOR_LEFT = 0
"""Where the single fixed detector's window sits inside the dense layer.

Rows two to four, columns zero to two, which is the window the page works by
hand and the one that answers 4 on the resting picture.
"""

ARRIVING_SLOPE = 1.0
"""The slope sent to every answer cell in the worked backward pass.

One rather than a measured loss gradient, so that a kernel entry's slope is
the plain sum of what it multiplied and can be checked against a hand count.
"""

DIFFERENCE_STEP = 1e-6
"""How far each parameter is moved either way for the central difference.

Small enough that the difference is the slope and large enough that
subtracting two nearly equal losses in float64 still leaves signal, which is
the trade every finite-difference check makes.
"""

COST_REPEATS = 5
"""How many times each forward pass is timed, the median being reported."""

COST_SIZES = ((8, 1), (14, 4), (28, 8))
"""The picture sides and filter counts the cost endpoint times.

Eight by eight is the page's own picture, twenty-eight by twenty-eight through
eight filters is the library's own example, and fourteen sits between them so
the growth is visible rather than assumed.
"""


def _square_picture() -> np.ndarray:
    """The page's eight by eight picture with its four by four square lit."""
    picture = np.zeros((PICTURE_SIDE, PICTURE_SIDE), dtype=np.float64)
    picture[
        SQUARE_TOP : SQUARE_TOP + SQUARE_SIDE, SQUARE_TOP : SQUARE_TOP + SQUARE_SIDE
    ] = 1.0
    return picture


def _sweeping_layer(
    picture_side: int = PICTURE_SIDE,
    n_filters: int = 1,
    kernel_size: int = KERNEL_SIDE,
    stride: int = 1,
    padding: int = 0,
    channels: int = CHANNELS,
) -> Conv2d:
    """A convolution over a square picture, with an identity bend."""
    return Conv2d(
        reads=(channels, picture_side, picture_side),
        n_filters=n_filters,
        kernel_size=kernel_size,
        activation=Identity(),
        stride=stride,
        padding=padding,
        random_seed=FIXED_SEED,
    )


def _carrying(layer: Conv2d, kernels: np.ndarray) -> Conv2d:
    """The same layer holding these kernels and a zero bias per filter."""
    n_filters = kernels.shape[0]
    return layer.with_parameters(kernels, np.zeros(n_filters, dtype=np.float64))


def _swept(layer: Conv2d, picture: np.ndarray) -> np.ndarray:
    """One picture through one layer, answering ``(n_filters, height, width)``."""
    block = picture.reshape(1, CHANNELS, picture.shape[0], picture.shape[1])
    return np.asarray(layer.respond_to(block).outputs, dtype=np.float64)[0]


def _dense_of(weights: np.ndarray) -> DenseLayer:
    """A dense layer of one neuron per row of ``weights``, every bias zero."""
    return DenseLayer(
        [
            Neuron(weights=row, bias=0.0, activation=Identity())
            for row in np.asarray(weights, dtype=np.float64)
        ]
    )


def _moved_right(picture: np.ndarray, offset: int) -> np.ndarray:
    """The picture slid ``offset`` columns to the right, zeros entering at the left."""
    moved = np.zeros_like(picture)
    width = picture.shape[1]
    if offset < width:
        moved[:, offset:] = picture[:, : width - offset]
    return moved


def _grid_of(values: np.ndarray) -> list[list[float]]:
    """A two-dimensional array as plain floats, rows down and columns across."""
    return [[float(value) for value in row] for row in values]


def _largest_gap_after_moving(
    settled: np.ndarray, answer: np.ndarray, columns: int
) -> float:
    """How far the answer is from the resting answer moved ``columns`` across.

    Only the columns the move leaves inside the map are compared, since the
    ones pushed off the right edge have nothing to be compared against and the
    ones entering at the left were never read.
    """
    width = settled.shape[-1]
    if columns >= width:
        return 0.0
    return float(
        np.max(np.abs(answer[..., columns:] - settled[..., : width - columns]))
    )


class ArrangementResponse(BaseModel):
    """One picture and one shuffle of it, read by a dense layer and by a sweep.

    ``scrambled`` is the picture with its cells moved by a fixed permutation,
    and ``dense_rewired`` is the same permutation applied to every neuron's
    weights, which is the whole of what a dense layer would have to be told to
    read the shuffled picture as it read the original. ``dense_largest_gap``
    is how far its answers moved, and ``swept_largest_gap`` how far the
    convolution's did, on the identical pair of pictures.
    """

    picture: list[list[float]]
    scrambled: list[list[float]]
    picture_cells_moved: int
    n_picture_cells: int
    dense_answer: list[float]
    dense_answer_rewired: list[float]
    dense_largest_gap: float
    dense_identical: bool
    swept: list[list[float]]
    swept_scrambled: list[list[float]]
    swept_largest_gap: float
    swept_cells_changed: int
    n_answer_cells: int
    dense_parameters: int
    convolution_parameters: int


@router.get("/concepts/convolution/arrangement", response_model=ArrangementResponse)
def arrangement() -> ArrangementResponse:
    """Shuffle the picture, rewire a dense layer to match, and compare both readers."""
    picture = _square_picture()
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(picture.size)
    scrambled = picture.reshape(-1)[permutation].reshape(picture.shape)

    layer = _carrying(
        _sweeping_layer(),
        np.asarray(VERTICAL_EDGE_KERNEL, dtype=np.float64).reshape(
            1, CHANNELS, KERNEL_SIDE, KERNEL_SIDE
        ),
    )
    swept = _swept(layer, picture)[0]
    swept_scrambled = _swept(layer, scrambled)[0]

    weights = np.random.default_rng(DENSE_DRAW_SEED).normal(
        size=(swept.size, picture.size)
    ) / float(PICTURE_SIDE)
    plain = _dense_of(weights)
    rewired = _dense_of(weights[:, permutation])

    answer = np.asarray(
        plain.respond_to(picture.reshape(1, -1)).outputs, dtype=np.float64
    )[0]
    answer_rewired = np.asarray(
        rewired.respond_to(scrambled.reshape(1, -1)).outputs, dtype=np.float64
    )[0]

    return ArrangementResponse(
        picture=_grid_of(picture),
        scrambled=_grid_of(scrambled),
        picture_cells_moved=int(np.count_nonzero(scrambled != picture)),
        n_picture_cells=int(picture.size),
        dense_answer=[float(value) for value in answer],
        dense_answer_rewired=[float(value) for value in answer_rewired],
        dense_largest_gap=float(np.max(np.abs(answer - answer_rewired))),
        dense_identical=bool(np.array_equal(answer, answer_rewired)),
        swept=_grid_of(swept),
        swept_scrambled=_grid_of(swept_scrambled),
        swept_largest_gap=float(np.max(np.abs(swept - swept_scrambled))),
        swept_cells_changed=int(np.count_nonzero(swept != swept_scrambled)),
        n_answer_cells=int(swept.size),
        dense_parameters=int(weights.size + swept.size),
        convolution_parameters=int(layer.kernels.size + layer.bias_vector.size),
    )


class ShiftRequest(BaseModel):
    """A kernel, a sweep setting, and how far to slide the picture sideways.

    The picture is the page's own square rather than a parameter, because the
    two dense layers beside the sweep are built to read exactly it and the
    comparison is only a comparison while all three read the same thing.
    """

    kernel: list[KernelRow] = Field(min_length=KERNEL_SIDE, max_length=KERNEL_SIDE)
    stride: int = Field(default=1, ge=1, le=MAX_SWEEP_STRIDE)
    shifts: list[Annotated[int, Field(ge=0, le=PICTURE_SIDE)]] = Field(
        min_length=1, max_length=MAX_SHIFTS
    )

    model_config = {"extra": "forbid"}


class ShiftedSweep(BaseModel):
    """One sideways position, read three ways.

    ``output_shift`` is how many answer positions the picture's move works out
    to, which is the move divided by the stride and is ``None`` when the
    stride does not divide it, since the answer has nowhere to move to.
    ``largest_gap`` compares the answer against the resting answer moved by
    that many positions, over the columns the move leaves inside the map.
    ``detector_answer`` is what the single fixed detector says, and
    ``dense_largest_gap`` is the same comparison made of the ordinary dense
    layer's thirty-six answers laid out as the sweep's map.
    """

    shift: int
    picture: list[list[float]]
    swept: list[list[float]]
    output_shift: int | None
    largest_gap: float | None
    matches_moved_answer: bool
    picture_cells_changed: int
    swept_cells_changed: int
    detector_answer: float
    dense_largest_gap: float | None
    dense_largest_answer: float
    dense_cells_changed: int


class ShiftResponse(BaseModel):
    """Every shift asked for, and what the three readers made of it."""

    stride: int
    reads: list[int]
    answers: list[int]
    detector_top: int
    detector_left: int
    detector_at_rest: float
    resting_swept: list[list[float]]
    n_answer_cells: int
    shifts: list[ShiftedSweep]


@router.post("/concepts/convolution/shift", response_model=ShiftResponse)
def shift(request: ShiftRequest) -> ShiftResponse:
    """Move the picture sideways and ask whether each reader's answer moved with it."""
    picture = _square_picture()
    kernel = np.asarray(request.kernel, dtype=np.float64)
    layer = _carrying(
        _sweeping_layer(stride=request.stride),
        kernel.reshape(1, CHANNELS, KERNEL_SIDE, KERNEL_SIDE),
    )
    _, out_height, out_width = layer.shape.answers
    resting = _swept(layer, picture)[0]

    detector = np.zeros((PICTURE_SIDE, PICTURE_SIDE), dtype=np.float64)
    detector[
        DETECTOR_TOP : DETECTOR_TOP + KERNEL_SIDE,
        DETECTOR_LEFT : DETECTOR_LEFT + KERNEL_SIDE,
    ] = kernel
    one_detector = _dense_of(detector.reshape(1, -1))

    n_answers = out_height * out_width
    weights = np.random.default_rng(DENSE_DRAW_SEED).normal(
        size=(n_answers, picture.size)
    ) / float(PICTURE_SIDE)
    equal_width = _dense_of(weights)
    dense_resting = np.asarray(
        equal_width.respond_to(picture.reshape(1, -1)).outputs, dtype=np.float64
    )[0].reshape(out_height, out_width)

    return ShiftResponse(
        stride=layer.stride,
        reads=[int(extent) for extent in layer.shape.reads],
        answers=[int(extent) for extent in layer.shape.answers],
        detector_top=DETECTOR_TOP,
        detector_left=DETECTOR_LEFT,
        detector_at_rest=float(
            one_detector.respond_to(picture.reshape(1, -1)).outputs[0, 0]
        ),
        resting_swept=_grid_of(resting),
        n_answer_cells=n_answers,
        shifts=[
            _one_shift(
                offset,
                picture,
                layer,
                resting,
                one_detector,
                equal_width,
                dense_resting,
            )
            for offset in request.shifts
        ],
    )


def _one_shift(
    offset: int,
    picture: np.ndarray,
    layer: Conv2d,
    resting: np.ndarray,
    one_detector: DenseLayer,
    equal_width: DenseLayer,
    dense_resting: np.ndarray,
) -> ShiftedSweep:
    """One sideways position, swept and read by both dense layers."""
    moved = _moved_right(picture, offset)
    swept = _swept(layer, moved)[0]
    dense_answer = np.asarray(
        equal_width.respond_to(moved.reshape(1, -1)).outputs, dtype=np.float64
    )[0].reshape(dense_resting.shape)

    divides = offset % layer.stride == 0
    output_shift = offset // layer.stride if divides else None
    largest_gap = (
        _largest_gap_after_moving(resting, swept, output_shift)
        if output_shift is not None
        else None
    )
    dense_gap = (
        _largest_gap_after_moving(dense_resting, dense_answer, output_shift)
        if output_shift is not None
        else None
    )

    return ShiftedSweep(
        shift=offset,
        picture=_grid_of(moved),
        swept=_grid_of(swept),
        output_shift=output_shift,
        largest_gap=largest_gap,
        matches_moved_answer=largest_gap == 0.0,
        picture_cells_changed=int(np.count_nonzero(moved != picture)),
        swept_cells_changed=int(np.count_nonzero(swept != resting)),
        detector_answer=float(
            one_detector.respond_to(moved.reshape(1, -1)).outputs[0, 0]
        ),
        dense_largest_gap=dense_gap,
        dense_largest_answer=float(np.max(np.abs(dense_answer))),
        dense_cells_changed=int(np.count_nonzero(dense_answer != dense_resting)),
    )


class BankRequest(BaseModel):
    """Several kernels, swept across the page's picture at once."""

    picture: list[PictureRow] = Field(min_length=PICTURE_SIDE, max_length=PICTURE_SIDE)
    kernels: list[list[KernelRow]] = Field(min_length=1, max_length=MAX_BANK_FILTERS)
    stride: int = Field(default=1, ge=1, le=MAX_SWEEP_STRIDE)
    padding: int = Field(default=0, ge=0, le=MAX_SWEEP_PADDING)

    model_config = {"extra": "forbid"}


class FilterMap(BaseModel):
    """One filter's own map, and where it answered most strongly.

    ``strongest_row`` and ``strongest_column`` are the answer position holding
    the largest value, so the page can say what each filter in the bank turned
    out to answer to rather than describing kernels in words.
    """

    position: int
    output: list[list[float]]
    largest: float
    smallest: float
    strongest_row: int
    strongest_column: int


class BankResponse(BaseModel):
    """The bank's arrangement, one map per filter, and what the bank costs."""

    reads: list[int]
    answers: list[int]
    n_filters: int
    kernel_shape: list[int]
    maps: list[FilterMap]
    parameters: ParameterCount


@router.post("/concepts/convolution/bank", response_model=BankResponse)
def bank(request: BankRequest) -> BankResponse:
    """Sweep a bank of kernels and report one map per filter."""
    picture = np.asarray(request.picture, dtype=np.float64)
    kernels = np.asarray(request.kernels, dtype=np.float64).reshape(
        len(request.kernels), CHANNELS, KERNEL_SIDE, KERNEL_SIDE
    )
    layer = _carrying(
        _sweeping_layer(
            n_filters=len(request.kernels),
            stride=request.stride,
            padding=request.padding,
        ),
        kernels,
    )
    maps = _swept(layer, picture)

    return BankResponse(
        reads=[int(extent) for extent in layer.shape.reads],
        answers=[int(extent) for extent in layer.shape.answers],
        n_filters=layer.n_filters,
        kernel_shape=[int(extent) for extent in layer.kernels.shape],
        maps=[_filter_map(position, one) for position, one in enumerate(maps)],
        parameters=_parameter_count(layer),
    )


def _filter_map(position: int, answered: np.ndarray) -> FilterMap:
    """One filter's map, with the position of its strongest answer."""
    strongest_row, strongest_column = np.unravel_index(
        int(np.argmax(answered)), answered.shape
    )
    return FilterMap(
        position=position,
        output=_grid_of(answered),
        largest=float(np.max(answered)),
        smallest=float(np.min(answered)),
        strongest_row=int(strongest_row),
        strongest_column=int(strongest_column),
    )


class GeometryRequest(BaseModel):
    """One picture's arrangement, and the settings to try on it.

    Every kernel size is paired with every stride and every padding, so three
    of each describes twenty-seven layers. Nothing is swept; each layer is
    built and its shape read, which is the whole of what this endpoint is for.
    """

    channels: int = Field(default=CHANNELS, ge=1, le=MAX_COUNTED_CHANNELS)
    side: int = Field(ge=MIN_COUNTED_SIDE, le=MAX_COUNTED_SIDE)
    n_filters: int = Field(default=1, ge=1, le=MAX_COUNTED_FILTERS)
    kernel_sizes: list[Annotated[int, Field(ge=1, le=MAX_COUNTED_KERNEL_SIDE)]] = Field(
        min_length=1, max_length=MAX_GEOMETRY_CHOICES
    )
    strides: list[Annotated[int, Field(ge=1, le=MAX_COUNTED_STRIDE)]] = Field(
        min_length=1, max_length=MAX_GEOMETRY_CHOICES
    )
    paddings: list[Annotated[int, Field(ge=0, le=MAX_COUNTED_PADDING)]] = Field(
        min_length=1, max_length=MAX_GEOMETRY_CHOICES
    )

    model_config = {"extra": "forbid"}


class GeometryRow(BaseModel):
    """What one setting makes of the picture, before any data arrives.

    ``refused`` marks a setting the library will not build and ``detail`` is
    its own message, in which case every measured field is empty.
    ``corner_windows`` and ``centre_windows`` count how many window positions
    read the picture's top left cell and its middle cell, which is the border
    argument for padding as a count. ``unread_cells`` counts the cells no
    window position reaches at all, which a stride that overruns the side
    leaves behind.
    """

    kernel_size: int
    stride: int
    padding: int
    refused: bool
    detail: str | None
    reads: list[int] | None
    answers: list[int] | None
    n_inputs: int | None
    n_outputs: int | None
    parameters: int | None
    corner_windows: int | None
    centre_windows: int | None
    unread_cells: int | None
    keeps_the_side: bool


class GeometryResponse(BaseModel):
    """One row per setting, in the order the settings were paired."""

    channels: int
    side: int
    n_filters: int
    rows: list[GeometryRow]


@router.post("/concepts/convolution/geometry", response_model=GeometryResponse)
def geometry(request: GeometryRequest) -> GeometryResponse:
    """Build a layer for every setting, and read the arrangement off its shape."""
    rows = [
        _geometry_row(request, kernel_size, stride, padding)
        for kernel_size in request.kernel_sizes
        for stride in request.strides
        for padding in request.paddings
    ]
    return GeometryResponse(
        channels=request.channels,
        side=request.side,
        n_filters=request.n_filters,
        rows=rows,
    )


def _geometry_row(
    request: GeometryRequest, kernel_size: int, stride: int, padding: int
) -> GeometryRow:
    """One setting, refusal included, with nothing swept."""
    try:
        layer = _sweeping_layer(
            picture_side=request.side,
            n_filters=request.n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            channels=request.channels,
        )
    except MLLibError as refusal:
        return GeometryRow(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            refused=True,
            detail=str(refusal),
            reads=None,
            answers=None,
            n_inputs=None,
            n_outputs=None,
            parameters=None,
            corner_windows=None,
            centre_windows=None,
            unread_cells=None,
            keeps_the_side=False,
        )

    _, out_height, out_width = layer.shape.answers
    coverage = _window_coverage(request.side, out_height, out_width, layer)
    middle = request.side // 2

    return GeometryRow(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        refused=False,
        detail=None,
        reads=[int(extent) for extent in layer.shape.reads],
        answers=[int(extent) for extent in layer.shape.answers],
        n_inputs=int(layer.shape.n_inputs),
        n_outputs=int(layer.shape.n_outputs),
        parameters=int(layer.kernels.size + layer.bias_vector.size),
        corner_windows=int(coverage[0, 0]),
        centre_windows=int(coverage[middle, middle]),
        unread_cells=int(np.count_nonzero(coverage == 0)),
        keeps_the_side=bool(out_height == request.side and out_width == request.side),
    )


def _window_coverage(
    side: int, out_height: int, out_width: int, layer: Conv2d
) -> np.ndarray:
    """How many window positions read each cell of the picture itself.

    The sweep is the layer's own, the corner of the window at output position
    ``k`` sitting at ``k`` times the stride inside the bordered picture, so
    this counts what the layer will actually read rather than what the formula
    suggests it might. Positions falling on the border of zeros are dropped,
    since the border belongs to nobody.
    """
    coverage = np.zeros((side, side), dtype=np.int64)
    for output_row in range(out_height):
        top = output_row * layer.stride - layer.padding
        for output_column in range(out_width):
            left = output_column * layer.stride - layer.padding
            for kernel_row in range(layer.kernel_size):
                for kernel_column in range(layer.kernel_size):
                    row = top + kernel_row
                    column = left + kernel_column
                    if 0 <= row < side and 0 <= column < side:
                        coverage[row, column] += 1
    return coverage


class WorkedBackward(BaseModel):
    """The backward pass on the page's square, with a slope of one everywhere.

    ``kernel_slopes`` is what the layer says each of the nine weights wants,
    and ``kernel_slopes_by_hand`` is the same nine worked independently, each
    entry being the picture summed over the patch of answer-sized extent that
    starts at that entry's offset. ``blame`` is what each cell of the picture
    receives. ``kernel_total`` is the nine kernel entries added up, which is
    what a cell read by every window position collects.
    """

    picture: list[list[float]]
    arriving: float
    n_output_positions: int
    n_lit_cells: int
    kernel_slopes: list[list[float]]
    kernel_slopes_by_hand: list[list[float]]
    slopes_agree: bool
    bias_slope: float
    blame: list[list[float]]
    kernel_total: float
    corner_blame: float
    centre_blame: float
    n_cells_receiving_nothing: int


class GradientCheck(BaseModel):
    """One fixture's backward pass against a central difference of the loss.

    ``largest_kernel_gap``, ``largest_bias_gap`` and ``largest_input_gap`` are
    the worst disagreement over each block, and ``largest_true_slope`` is the
    biggest slope the layer reported, so the gaps can be read against
    something rather than in isolation.
    """

    name: str
    reads: list[int]
    n_filters: int
    kernel_size: int
    stride: int
    padding: int
    n_rows: int
    n_checked: int
    largest_kernel_gap: float
    largest_bias_gap: float
    largest_input_gap: float
    largest_true_slope: float


class BackwardResponse(BaseModel):
    """The worked pass, and the three fixtures checked against differences."""

    worked: WorkedBackward
    checks: list[GradientCheck]
    step: float
    largest_gap_anywhere: float


@router.get("/concepts/convolution/backward", response_model=BackwardResponse)
def backward() -> BackwardResponse:
    """Work the backward pass by hand, then check it against finite differences."""
    checks = [
        _gradient_check(name, **settings) for name, settings in _CHECK_FIXTURES.items()
    ]
    return BackwardResponse(
        worked=_worked_backward(),
        checks=checks,
        step=DIFFERENCE_STEP,
        largest_gap_anywhere=max(
            max(
                check.largest_kernel_gap,
                check.largest_bias_gap,
                check.largest_input_gap,
            )
            for check in checks
        ),
    )


def _worked_backward() -> WorkedBackward:
    """One slope arriving at every answer cell, over the page's own square."""
    picture = _square_picture()
    kernel = np.asarray(VERTICAL_EDGE_KERNEL, dtype=np.float64)
    layer = _carrying(
        _sweeping_layer(), kernel.reshape(1, CHANNELS, KERNEL_SIDE, KERNEL_SIDE)
    )

    block = picture.reshape(1, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)
    response = layer.respond_to(block)
    arriving = np.full_like(np.asarray(response.outputs), ARRIVING_SLOPE)
    correction = layer.correction_for(response, arriving)

    slopes = np.asarray(correction.gradient.weights).reshape(
        1, CHANNELS, KERNEL_SIDE, KERNEL_SIDE
    )[0, 0]
    blame = np.asarray(correction.passed_down)[0, 0]
    _, out_height, out_width = layer.shape.answers

    by_hand = np.array(
        [
            [
                float(
                    picture[
                        kernel_row : kernel_row + out_height,
                        kernel_column : kernel_column + out_width,
                    ].sum()
                )
                for kernel_column in range(KERNEL_SIDE)
            ]
            for kernel_row in range(KERNEL_SIDE)
        ]
    )
    middle = PICTURE_SIDE // 2

    return WorkedBackward(
        picture=_grid_of(picture),
        arriving=ARRIVING_SLOPE,
        n_output_positions=int(out_height * out_width),
        n_lit_cells=int(np.count_nonzero(picture)),
        kernel_slopes=_grid_of(slopes),
        kernel_slopes_by_hand=_grid_of(by_hand),
        slopes_agree=bool(np.array_equal(slopes, by_hand)),
        bias_slope=float(correction.gradient.biases[0]),
        blame=_grid_of(blame),
        kernel_total=float(kernel.sum()),
        corner_blame=float(blame[0, 0]),
        centre_blame=float(blame[middle, middle]),
        n_cells_receiving_nothing=int(np.count_nonzero(blame == 0.0)),
    )


_CHECK_FIXTURES: dict[str, dict[str, int | tuple[int, int, int]]] = {
    "a plain sweep": {
        "reads": (1, 5, 5),
        "n_filters": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "n_rows": 2,
    },
    "padded by one": {
        "reads": (1, 5, 5),
        "n_filters": 2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "n_rows": 2,
    },
    "two channels at a stride of two": {
        "reads": (2, 6, 6),
        "n_filters": 2,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "n_rows": 2,
    },
}
"""The three fixtures the check runs on, plain, padded and strided.

The padded one is where a version that forgets to strip the border shows up,
and the strided one is where a version that walks the input rather than the
output positions loses terms.
"""


def _gradient_check(
    name: str,
    reads: tuple[int, int, int],
    n_filters: int,
    kernel_size: int,
    stride: int,
    padding: int,
    n_rows: int,
) -> GradientCheck:
    """One fixture's three gradients against a central difference of the loss."""
    layer = Conv2d(
        reads=reads,
        n_filters=n_filters,
        kernel_size=kernel_size,
        activation=Identity(),
        stride=stride,
        padding=padding,
        random_seed=FIXED_SEED,
    )
    generator = np.random.default_rng(FIXED_SEED + 1)
    inputs = generator.normal(size=(n_rows, *reads))
    targets = generator.normal(size=(n_rows, *layer.shape.answers))
    loss = SquaredError()

    response = layer.respond_to(inputs)
    correction = layer.correction_for(
        response, loss.measure(response.outputs, targets).gradient
    )
    kernel_slopes = np.asarray(correction.gradient.weights).reshape(layer.kernels.shape)
    bias_slopes = np.asarray(correction.gradient.biases)
    input_slopes = np.asarray(correction.passed_down)

    def cost(moved: Conv2d, block: np.ndarray) -> float:
        return float(loss.measure(moved.respond_to(block).outputs, targets).value)

    largest_kernel_gap = _worst_difference(
        kernel_slopes,
        np.asarray(layer.kernels),
        lambda block: cost(layer.with_parameters(block, layer.bias_vector), inputs),
    )
    largest_bias_gap = _worst_difference(
        bias_slopes,
        np.asarray(layer.bias_vector),
        lambda block: cost(layer.with_parameters(layer.kernels, block), inputs),
    )
    largest_input_gap = _worst_difference(
        input_slopes, inputs, lambda block: cost(layer, block)
    )

    return GradientCheck(
        name=name,
        reads=list(reads),
        n_filters=n_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_rows=n_rows,
        n_checked=int(kernel_slopes.size + bias_slopes.size + input_slopes.size),
        largest_kernel_gap=largest_kernel_gap,
        largest_bias_gap=largest_bias_gap,
        largest_input_gap=largest_input_gap,
        largest_true_slope=float(np.max(np.abs(kernel_slopes))),
    )


def _worst_difference(
    slopes: np.ndarray, values: np.ndarray, cost: Callable[[np.ndarray], float]
) -> float:
    """The worst gap between a reported slope and the difference that measures it.

    Central rather than one-sided, so the error falls as the square of the
    step rather than in step with it, which is what makes a disagreement at
    1e-09 evidence about the implementation rather than about the difference.
    """
    worst = 0.0
    for index in np.ndindex(values.shape):
        raised = np.array(values, dtype=np.float64)
        raised[index] += DIFFERENCE_STEP
        lowered = np.array(values, dtype=np.float64)
        lowered[index] -= DIFFERENCE_STEP
        measured = (cost(raised) - cost(lowered)) / (2 * DIFFERENCE_STEP)
        worst = max(worst, abs(measured - float(slopes[index])))
    return worst


def _definitional_sweep(
    picture: np.ndarray,
    kernels: np.ndarray,
    biases: np.ndarray,
    stride: int,
    padding: int,
) -> np.ndarray:
    """The forward pass as its definition reads, one term at a time.

    Seven nested loops, over the rows of the block, the filters, the two
    output axes, the channels and the two kernel axes, with no view and no
    contraction anywhere, so that it shares nothing with the layer but the
    numbers it is handed. It is written here from the definition rather than
    taken from anywhere else, and it exists for the cost endpoint to time
    beside the layer and to check the layer's answer against.
    """
    n_rows, channels, height, width = picture.shape
    n_filters, _, side, _ = kernels.shape
    bordered = np.pad(picture, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_height = (height - side + 2 * padding) // stride + 1
    out_width = (width - side + 2 * padding) // stride + 1
    answers = np.zeros((n_rows, n_filters, out_height, out_width))
    for row in range(n_rows):
        for filter_index in range(n_filters):
            for out_row in range(out_height):
                for out_column in range(out_width):
                    total = float(biases[filter_index])
                    for channel in range(channels):
                        for kernel_row in range(side):
                            for kernel_column in range(side):
                                total += float(
                                    kernels[
                                        filter_index, channel, kernel_row, kernel_column
                                    ]
                                    * bordered[
                                        row,
                                        channel,
                                        out_row * stride + kernel_row,
                                        out_column * stride + kernel_column,
                                    ]
                                )
                    answers[row, filter_index, out_row, out_column] = total
    return answers


class CostRow(BaseModel):
    """One size, timed three ways.

    ``sweep_milliseconds``, ``loop_milliseconds`` and ``dense_milliseconds``
    are each the median of ``COST_REPEATS`` forward passes over a single
    picture, through the layer, through the seven loops of the definition,
    and through a dense layer reading and answering the same numbers. ``ratio``
    is the sweep over the dense layer, so below one means the sweep was the
    quicker, and ``loop_over_sweep`` is the loops over the sweep. ``n_terms``
    is how many products one pass adds up, and ``largest_disagreement`` is the
    worst gap between the loops' answer and the layer's. The times are a
    measurement of the machine that served the request rather than a property
    of the arrangement; the disagreement is not.
    """

    side: int
    n_filters: int
    answers: list[int]
    n_inputs: int
    n_outputs: int
    n_terms: int
    convolution_parameters: int
    dense_parameters: int
    sweep_milliseconds: float
    loop_milliseconds: float
    dense_milliseconds: float
    ratio: float
    loop_over_sweep: float
    largest_disagreement: float


class CostResponse(BaseModel):
    """Every size timed, and how many passes each figure is the median of."""

    repeats: int
    rows: list[CostRow]
    largest_disagreement_anywhere: float


@router.get("/concepts/convolution/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """Time the sweep against its own definition and a dense layer, at three sizes."""
    rows = [_cost_row(side, n_filters) for side, n_filters in COST_SIZES]
    return CostResponse(
        repeats=COST_REPEATS,
        rows=rows,
        largest_disagreement_anywhere=max(row.largest_disagreement for row in rows),
    )


def _cost_row(side: int, n_filters: int) -> CostRow:
    """One size, all three built and all three forward passes timed.

    The layer is given drawn biases in place of its starting zeros, so that a
    sweep which forgot the bias could not agree with the loops by accident.
    """
    generator = np.random.default_rng(FIXED_SEED)
    unbiased = _sweeping_layer(picture_side=side, n_filters=n_filters)
    layer = unbiased.with_parameters(
        unbiased.kernels, generator.normal(size=unbiased.bias_vector.shape)
    )
    n_inputs = int(layer.shape.n_inputs)
    n_outputs = int(layer.shape.n_outputs)

    weights = np.random.default_rng(DENSE_DRAW_SEED).normal(
        size=(n_outputs, n_inputs)
    ) / np.sqrt(n_inputs)
    equal_width = _dense_of(weights)

    picture = generator.normal(size=(1, CHANNELS, side, side))
    row = picture.reshape(1, n_inputs)
    kernels = np.asarray(layer.kernels, dtype=np.float64)
    biases = np.asarray(layer.bias_vector, dtype=np.float64)

    def by_definition() -> np.ndarray:
        return _definitional_sweep(
            picture, kernels, biases, layer.stride, layer.padding
        )

    sweep_time = _median_time(lambda: layer.respond_to(picture))
    loop_time = _median_time(by_definition)
    dense_time = _median_time(lambda: equal_width.respond_to(row))

    swept = np.asarray(layer.respond_to(picture).outputs, dtype=np.float64)

    return CostRow(
        side=side,
        n_filters=n_filters,
        answers=[int(extent) for extent in layer.shape.answers],
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_terms=int(n_outputs * kernels[0].size),
        convolution_parameters=int(kernels.size + biases.size),
        dense_parameters=int(weights.size + n_outputs),
        sweep_milliseconds=sweep_time * 1000.0,
        loop_milliseconds=loop_time * 1000.0,
        dense_milliseconds=dense_time * 1000.0,
        ratio=sweep_time / dense_time,
        loop_over_sweep=loop_time / sweep_time,
        largest_disagreement=float(np.max(np.abs(swept - by_definition()))),
    )


def _median_time(action: Callable[[], object]) -> float:
    """The median of ``COST_REPEATS`` runs, in seconds.

    The median rather than the mean, because one run interrupted by whatever
    else the machine was doing would move a mean and cannot move a median.
    """
    samples = []
    for _ in range(COST_REPEATS):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return sorted(samples)[len(samples) // 2]


class EdgeCase(BaseModel):
    """One thing asked of a convolution, and what the library did about it.

    ``raised`` says whether anything was refused at all, ``error`` names the
    class of the refusal and ``detail`` carries its message. ``library_error``
    says whether that refusal is one of the library's own rather than a bare
    builtin failure, which is the difference between a message written for a
    learner and one written for an interpreter. ``outcome`` describes what
    came back when nothing was refused, built from the values themselves.
    """

    name: str
    attempt: str
    raised: bool
    error: str | None
    detail: str | None
    library_error: bool
    outcome: str


class JoinAttempt(BaseModel):
    """One chain of layers offered to a stack, and what the stack said.

    ``beneath`` and ``above`` are the two arrangements at the join in
    question, ``both_hold`` how many numbers each of them carries, and
    ``detail`` the stack's own refusal where there was one.
    """

    name: str
    beneath: list[int]
    above: list[int]
    both_hold: int
    counts_agree: bool
    refused: bool
    error: str | None
    detail: str | None
    stack_reads: list[int] | None
    stack_answers: list[int] | None
    n_layers: int | None


class EdgeCasesResponse(BaseModel):
    """Every probe the page's last part quotes, run against the library."""

    cases: list[EdgeCase]
    n_refused: int
    n_accepted: int
    n_in_the_library_words: int
    joins: list[JoinAttempt]


def _probe(name: str, attempt: str, action: Callable[[], str]) -> EdgeCase:
    """Run one attempt and report the refusal, or what came back instead.

    Every failure is caught, not only the library's own, so a bare builtin
    escaping the hierarchy shows up as a row saying so rather than as a server
    error the page cannot describe.
    """
    try:
        outcome = action()
    except MLLibError as refusal:
        return EdgeCase(
            name=name,
            attempt=attempt,
            raised=True,
            error=type(refusal).__name__,
            detail=str(refusal),
            library_error=True,
            outcome="refused, in words written to be read",
        )
    except Exception as failure:  # noqa: BLE001 - reporting it is the point
        return EdgeCase(
            name=name,
            attempt=attempt,
            raised=True,
            error=type(failure).__name__,
            detail=str(failure),
            library_error=False,
            outcome="refused, though not in words written to be read",
        )
    return EdgeCase(
        name=name,
        attempt=attempt,
        raised=False,
        error=None,
        detail=None,
        library_error=False,
        outcome=outcome,
    )


def _kernel_larger_than_the_picture() -> str:
    _sweeping_layer(picture_side=4, kernel_size=7)
    return "accepted"


def _bare_integer_extents() -> str:
    Conv2d(reads=PICTURE_SIDE, n_filters=1, kernel_size=3, activation=Identity())  # type: ignore[arg-type]
    return "accepted"


def _two_extents() -> str:
    Conv2d(
        reads=(PICTURE_SIDE, PICTURE_SIDE),
        n_filters=1,
        kernel_size=3,
        activation=Identity(),
    )
    return "accepted"


def _kernel_size_of_zero() -> str:
    _sweeping_layer(kernel_size=0)
    return "accepted"


def _stride_of_zero() -> str:
    _sweeping_layer(stride=0)
    return "accepted"


def _padding_below_zero() -> str:
    _sweeping_layer(padding=-1)
    return "accepted"


def _filters_given_as_a_flag() -> str:
    Conv2d(
        reads=(CHANNELS, PICTURE_SIDE, PICTURE_SIDE),
        n_filters=True,
        kernel_size=3,
        activation=Identity(),
    )
    return "accepted"


def _kernel_size_given_as_a_fraction() -> str:
    _sweeping_layer(kernel_size=3.5)  # type: ignore[arg-type]
    return "accepted"


def _picture_handed_in_as_a_row() -> str:
    _sweeping_layer().respond_to(np.zeros((1, PICTURE_SIDE * PICTURE_SIDE)))
    return "accepted"


def _picture_of_the_wrong_arrangement() -> str:
    _sweeping_layer().respond_to(np.zeros((1, CHANNELS, 6, 6)))
    return "accepted"


def _picture_with_a_missing_value() -> str:
    spoiled = _square_picture()
    spoiled[0, 0] = np.nan
    _sweeping_layer().respond_to(
        spoiled.reshape(1, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)
    )
    return "accepted"


def _block_holding_no_rows() -> str:
    _sweeping_layer().respond_to(np.zeros((0, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)))
    return "accepted"


def _arriving_block_of_the_wrong_arrangement() -> str:
    layer = _sweeping_layer()
    response = layer.respond_to(
        _square_picture().reshape(1, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)
    )
    layer.correction_for(response, np.ones((1, CHANNELS, 5, 5)))
    return "accepted"


def _arriving_block_of_the_wrong_row_count() -> str:
    layer = _sweeping_layer()
    response = layer.respond_to(
        _square_picture().reshape(1, CHANNELS, PICTURE_SIDE, PICTURE_SIDE)
    )
    layer.correction_for(response, np.ones((3, CHANNELS, 6, 6)))
    return "accepted"


def _a_kernel_bank_of_the_wrong_shape() -> str:
    _sweeping_layer().with_parameters(np.zeros((1, CHANNELS, 5, 5)), np.zeros(1))
    return "accepted"


def _a_step_with_no_gradient() -> str:
    _sweeping_layer().stepped_by(None, 0.1)
    return "accepted"


def _a_kernel_the_size_of_the_picture() -> str:
    layer = _sweeping_layer(kernel_size=PICTURE_SIDE)
    return (
        f"accepted, answering {layer.shape.answers} from one window, which is "
        f"{layer.kernels.size + layer.bias_vector.size} parameters reading "
        f"{layer.shape.n_inputs}"
    )


def _a_stride_larger_than_the_kernel() -> str:
    layer = _sweeping_layer(stride=4)
    coverage = _window_coverage(
        PICTURE_SIDE, layer.shape.answers[1], layer.shape.answers[2], layer
    )
    return (
        f"accepted, answering {layer.shape.answers} and leaving "
        f"{int(np.count_nonzero(coverage == 0))} of {coverage.size} cells "
        "outside every window"
    )


def _padding_wider_than_the_kernel() -> str:
    layer = _sweeping_layer(padding=4)
    _, out_height, out_width = layer.shape.answers
    return (
        f"accepted, answering {layer.shape.answers}, of which "
        f"{_positions_reading_only_border(PICTURE_SIDE, out_height, out_width, layer)} "
        f"of {out_height * out_width} positions read nothing but border"
    )


def _positions_reading_only_border(
    side: int, out_height: int, out_width: int, layer: Conv2d
) -> int:
    """How many output positions sit entirely on the zeros padding added.

    A padding wider than the kernel puts whole windows outside the picture, and
    each of them answers with the filter's bias and no evidence at all. Nothing
    refuses it, so the page counts it instead.
    """
    stranded = 0
    for output_row in range(out_height):
        top = output_row * layer.stride - layer.padding
        for output_column in range(out_width):
            left = output_column * layer.stride - layer.padding
            touches = (
                top + layer.kernel_size > 0
                and top < side
                and left + layer.kernel_size > 0
                and left < side
            )
            if not touches:
                stranded += 1
    return stranded


_EDGE_PROBES: tuple[Callable[[], EdgeCase], ...] = (
    lambda: _probe(
        "a kernel larger than the picture",
        "a kernel of 7 over a picture 4 by 4",
        _kernel_larger_than_the_picture,
    ),
    lambda: _probe(
        "a bare integer where extents are wanted",
        "reads given as 8 rather than as three extents",
        _bare_integer_extents,
    ),
    lambda: _probe(
        "two extents rather than three",
        "reads given as (8, 8), with no channel count",
        _two_extents,
    ),
    lambda: _probe(
        "a kernel of zero", "a kernel size of 0 at a stride of 1", _kernel_size_of_zero
    ),
    lambda: _probe("a stride of zero", "a stride of 0", _stride_of_zero),
    lambda: _probe("a padding below zero", "a padding of −1", _padding_below_zero),
    lambda: _probe(
        "a flag where a whole number is wanted",
        "a filter count given as True, which indexes as 1",
        _filters_given_as_a_flag,
    ),
    lambda: _probe(
        "a fraction where a whole number is wanted",
        "a kernel size of 3.5",
        _kernel_size_given_as_a_fraction,
    ),
    lambda: _probe(
        "the picture handed in as a row",
        "a block arranged (1, 64) into a layer reading (1, 8, 8)",
        _picture_handed_in_as_a_row,
    ),
    lambda: _probe(
        "a picture of the wrong arrangement",
        "a block arranged (1, 1, 6, 6) into a layer reading (1, 8, 8)",
        _picture_of_the_wrong_arrangement,
    ),
    lambda: _probe(
        "a picture with a missing value",
        "the square with its top left cell set to a missing value",
        _picture_with_a_missing_value,
    ),
    lambda: _probe(
        "a block holding no rows",
        "a block arranged (0, 1, 8, 8)",
        _block_holding_no_rows,
    ),
    lambda: _probe(
        "an arriving block of the wrong arrangement",
        "a block arranged (1, 1, 5, 5) into a layer answering (1, 6, 6)",
        _arriving_block_of_the_wrong_arrangement,
    ),
    lambda: _probe(
        "an arriving block with the wrong number of rows",
        "three rows of blame for a response holding one",
        _arriving_block_of_the_wrong_row_count,
    ),
    lambda: _probe(
        "a kernel bank of the wrong shape",
        "kernels arranged (1, 1, 5, 5) given to a layer whose kernels are 3 by 3",
        _a_kernel_bank_of_the_wrong_shape,
    ),
    lambda: _probe(
        "a step asked for with no gradient",
        "stepped by nothing at a rate of 0.1",
        _a_step_with_no_gradient,
    ),
    lambda: _probe(
        "a kernel the size of the whole picture",
        "a kernel of 8 over a picture 8 by 8",
        _a_kernel_the_size_of_the_picture,
    ),
    lambda: _probe(
        "a stride larger than the kernel",
        "a kernel of 3 at a stride of 4",
        _a_stride_larger_than_the_kernel,
    ),
    lambda: _probe(
        "padding wider than the kernel",
        "a kernel of 3 padded by 4",
        _padding_wider_than_the_kernel,
    ),
)
"""Every probe the page quotes, in the order the page's table lists them."""


def _dense_reading(n_inputs: int, n_outputs: int) -> DenseLayer:
    """A dense layer of the given width, its weights all zero.

    Nothing here reads an answer, so the weights decide nothing. What is being
    asked is whether a stack will let the layer sit where it is put.
    """
    return _dense_of(np.zeros((n_outputs, n_inputs)))


def _join(
    name: str, beneath: Conv2d, layers: list[object], above_reads: tuple[int, ...]
) -> JoinAttempt:
    """Offer one chain to a stack and report what it said about the join."""
    both_hold = int(beneath.shape.n_outputs)
    try:
        stack = LayerStack(layers)  # type: ignore[arg-type]
    except MLLibError as refusal:
        return JoinAttempt(
            name=name,
            beneath=[int(extent) for extent in beneath.shape.answers],
            above=[int(extent) for extent in above_reads],
            both_hold=both_hold,
            counts_agree=int(np.prod(above_reads)) == both_hold,
            refused=True,
            error=type(refusal).__name__,
            detail=str(refusal),
            stack_reads=None,
            stack_answers=None,
            n_layers=None,
        )
    return JoinAttempt(
        name=name,
        beneath=[int(extent) for extent in beneath.shape.answers],
        above=[int(extent) for extent in above_reads],
        both_hold=both_hold,
        counts_agree=int(np.prod(above_reads)) == both_hold,
        refused=False,
        error=None,
        detail=None,
        stack_reads=[int(extent) for extent in stack.shape.reads],
        stack_answers=[int(extent) for extent in stack.shape.answers],
        n_layers=len(stack),
    )


def _join_attempts() -> list[JoinAttempt]:
    """The three chains the page's last part quotes."""
    page_layer = _sweeping_layer()
    page_answers = page_layer.shape.n_outputs
    library_layer = _sweeping_layer(picture_side=28, n_filters=8)
    library_answers = library_layer.shape.n_outputs

    return [
        _join(
            "a convolution with a layer that reads a row placed straight on top",
            page_layer,
            [page_layer, _dense_reading(page_answers, 3)],
            (page_answers,),
        ),
        _join(
            "the same chain with a flattening layer between them",
            page_layer,
            [
                page_layer,
                Flatten(page_layer.shape.answers),
                _dense_reading(page_answers, 3),
            ],
            page_layer.shape.answers,
        ),
        _join(
            "the full-size example, a 28 by 28 picture, joined the same wrong way",
            library_layer,
            [library_layer, _dense_reading(library_answers, 10)],
            (library_answers,),
        ),
    ]


@router.get("/concepts/convolution/edge-cases", response_model=EdgeCasesResponse)
def edge_cases() -> EdgeCasesResponse:
    """Ask the library for each of the page's edges, and report what it answered."""
    cases = [probe() for probe in _EDGE_PROBES]
    return EdgeCasesResponse(
        cases=cases,
        n_refused=sum(1 for case in cases if case.raised),
        n_accepted=sum(1 for case in cases if not case.raised),
        n_in_the_library_words=sum(1 for case in cases if case.library_error),
        joins=_join_attempts(),
    )
