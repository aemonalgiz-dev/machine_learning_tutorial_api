"""Pooling, for the page that shrinks a picture by summarising each window.

The apply endpoint takes one small picture, a window, a stride and a kind,
builds the library's ``MaxPool2d`` or ``AveragePool2d`` to read it as a block
of one row and one channel, and answers with two grids. The first is the
pooled map, what the layer said on the way up. The second is what the layer
hands back down when a slope of one arrives at every cell of that map, which
is the whole lesson of the page. A maximum routes each window's slope to the
single position that won the window, so the grid is one at the winners and
zero everywhere else. An average hands every position the same fraction, a
quarter each under a window of two. Where the stride is smaller than the
window the windows overlap, and the library adds rather than assigns, so a
position that won three windows shows three and an interior position under an
average shows four quarters.

The answer also lists every window the layer swept, where it sat, what it
answered, which position won it under a maximum, and what the shares inside
it add up to. Those shares come from the layer's own ``shares_of``, the method
its backward pass calls on the same view of the same picture, so a winner
named here is the winner the layer routed the slope to and not a second
reading of the geometry made in this module.

The share-totals endpoint exists for one figure. An average's shares are
``1 / n`` each and there are ``n`` of them, and in float64 that adds up to one
only most of the time. The endpoint builds a layer whose window is the whole
picture at every side up to the one asked for, sends a slope of one through
its backward pass, and reports the total handed down, so the page can say that
side 7 comes to 0.9999999999999999, that sixteen of the first thirty-two sides
miss, and that the worst is side 31 at 4.4e-16, with every one of those
numbers read off the layer rather than off a sentence.

The geometry endpoint runs no data at all. It builds a layer for every window
and stride the page asks about and reads the arrangement straight off the
layer's own shape, which is the claim the page makes about pooling and the
reason it can be made before a picture exists. It also reports which rows and
columns no window position ever reaches, worked from the layer's own window and
stride, and it builds both kinds at every combination so the page's statement
that the two share their geometry exactly is a comparison rather than an
assertion. A combination the library refuses comes back as a refused row
carrying the library's message, rather than failing the whole sweep, so the
table can show a window that does not fit beside the ones that do.

The window endpoint takes one small square and asks each kind what it makes of
it, through ``summarise`` and ``shares_of``, which are the two methods a kind of
pooling consists of. The shift endpoint pools one picture, then the same picture
moved sideways a cell at a time, and counts how many pooled cells changed, which
is the tolerance claim measured rather than repeated.

The edge-case endpoint probes every refusal the page's last part quotes,
including the two things that are not refusals. A stride larger than the window
is accepted and leaves cells no window reaches, and two layers of one shape
built from different windows and strides accept each other's responses, which
the library documents rather than defends; the endpoint pairs them deliberately
and hands back both blame grids so the page can show where the arriving values
landed instead.

A ragged picture is refused here in words, before the library sees it. The
library would refuse it too, as a block it cannot read as numbers, which is
true and is not the mistake the learner made. A window larger than the picture
is left to the library, whose refusal already names both sizes.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import AveragePool2d, MaxPool2d, Pool2d
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE

router = APIRouter(tags=["pooling"])

MAX_SIDE = 8
"""The largest extent a picture may have on either side.

Eight by eight is enough to show overlapping windows and a window that does
not divide the side evenly, and small enough that every value on the grid can
be read, which is the point of a picture the reader paints by hand.
"""

MAX_SHARE_SIDE = 32
"""The largest window side the share-totals table is asked for.

Thirty-two is where the library's own measurement of the inexact sides stops,
and far enough to show that they are scattered rather than a pattern.
"""

CHANNELS = 1
"""A picture here has one channel.

Pooling never mixes channels, so a second one would only repeat the first.
"""

ARRIVING_SLOPE = 1.0
"""The slope sent to every pooled cell on the way back down.

One rather than a measured loss gradient, so that what each input cell
receives is exactly its share, counted once per window it belongs to.
"""

MAX_COUNTED_SIDE = 64
"""The largest extent the geometry endpoint will describe.

That endpoint runs no data through a layer. It constructs one and reads the
arrangement off its shape, so it is free to describe a picture far larger than
the one a reader paints by hand. Sixty-four reaches the 26 by 26 map the
convolution page's own example answers with, and a little beyond it.
"""

MAX_COUNTED_CHANNELS = 64
"""The most channels the geometry endpoint will describe, for the same reason."""

MAX_GEOMETRY_CHOICES = 8
"""How many window sides, and how many strides, one sweep may ask about.

Eight of each is sixty-four rows, which is a table a reader can look down.
"""

MAX_SHIFTS = 8
"""How many sideways positions the shift endpoint will pool a picture at."""

CountedSide = Annotated[int, Field(ge=1, le=MAX_COUNTED_SIDE)]

Shift = Annotated[int, Field(ge=0, le=MAX_SIDE)]


class PoolingKind(StrEnum):
    """The two ways a window is summarised, named by what each keeps."""

    MAX = "max"
    AVERAGE = "average"


PictureValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]

PictureRow = Annotated[list[PictureValue], Field(min_length=1, max_length=MAX_SIDE)]


class ApplyRequest(BaseModel):
    """The picture, rows down, and the window, stride and kind to pool it with."""

    picture: list[PictureRow] = Field(min_length=1, max_length=MAX_SIDE)
    window: int = Field(ge=1, le=MAX_SIDE)
    stride: int = Field(ge=1, le=MAX_SIDE)
    kind: PoolingKind

    model_config = {"extra": "forbid"}


class PicturePosition(BaseModel):
    """One cell of the picture, row down and column across, both from zero."""

    row: int
    column: int


class SweptWindow(BaseModel):
    """One window the layer swept, and what it made of it.

    ``top`` and ``left`` locate the window's first cell in the picture; the
    window extends the request's ``window`` cells down and across from there.
    ``winner`` is the position a maximum routed the whole arriving slope to,
    the first in row-major order on a tie, and is ``None`` under an average,
    which prefers no position. ``share_total`` is what the layer's shares
    inside this window add up to, one to within float64.
    """

    output_row: int
    output_column: int
    top: int
    left: int
    answer: float
    winner: PicturePosition | None
    share_total: float


class ApplyResponse(BaseModel):
    """The pooled map, and the blame each input cell receives for a slope of one.

    ``pooled`` is the layer's answer, ``pooled_height`` rows of
    ``pooled_width`` cells. ``blame`` is the layer's ``passed_down`` when a
    slope of one arrives at every pooled cell, laid over the picture, and
    ``corrected`` marks the cells whose blame is not zero, which under a
    maximum is the winners alone and under an average is every cell a window
    reached. Overlap accumulates, so an entry can exceed one.
    """

    kind: PoolingKind
    height: int
    width: int
    window: int
    stride: int
    pooled_height: int
    pooled_width: int
    pooled: list[list[float]]
    blame: list[list[float]]
    corrected: list[list[bool]]
    n_corrected: int
    windows: list[SweptWindow]


@router.post("/concepts/pooling/apply", response_model=ApplyResponse)
def apply(request: ApplyRequest) -> ApplyResponse:
    """Pool the picture, then hand a slope of one back down over it."""
    picture = _rectangle_of(request.picture)
    height, width = picture.shape
    layer = _layer_for(request.kind, height, width, request.window, request.stride)

    response = layer.respond_to(picture[None, None, :, :])
    arriving = np.full_like(response.outputs, ARRIVING_SLOPE)
    correction = layer.correction_for(response, arriving)

    pooled = response.outputs[0, 0]
    blame = correction.passed_down[0, 0]
    corrected = blame != 0.0

    return ApplyResponse(
        kind=request.kind,
        height=int(height),
        width=int(width),
        window=layer.window,
        stride=layer.stride,
        pooled_height=int(pooled.shape[0]),
        pooled_width=int(pooled.shape[1]),
        pooled=_rows_of(pooled),
        blame=_rows_of(blame),
        corrected=[[bool(flag) for flag in row] for row in corrected],
        n_corrected=int(np.count_nonzero(corrected)),
        windows=_swept_windows(layer, request.kind, picture, pooled),
    )


def _rectangle_of(rows: list[list[float]]) -> np.ndarray:
    """The picture as a float array, refusing one whose rows disagree in length."""
    width = len(rows[0])
    for position, row in enumerate(rows):
        if len(row) != width:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"row {position + 1} holds {len(row)} values where the first "
                    f"row holds {width}, and a picture has to be a rectangle "
                    "before a window can slide over it"
                ),
            )
    return np.asarray(rows, dtype=np.float64)


def _layer_for(
    kind: PoolingKind, height: int, width: int, window: int, stride: int
) -> Pool2d:
    """The library layer for this request. It refuses a window that does not fit."""
    reads = (CHANNELS, height, width)
    if kind is PoolingKind.MAX:
        return MaxPool2d(reads=reads, window=window, stride=stride)
    return AveragePool2d(reads=reads, window=window, stride=stride)


def _swept_windows(
    layer: Pool2d, kind: PoolingKind, picture: np.ndarray, pooled: np.ndarray
) -> list[SweptWindow]:
    """Every window the layer swept, read back through its own ``shares_of``.

    The corner of each window is the output position times the stride, which
    is the layer's own sweep, and the shares come from the method its backward
    pass calls on the same view, so the winner reported here is the winner
    that received the slope.
    """
    windows: list[SweptWindow] = []
    for output_row in range(pooled.shape[0]):
        top = output_row * layer.stride
        for output_column in range(pooled.shape[1]):
            left = output_column * layer.stride
            view = picture[top : top + layer.window, left : left + layer.window]
            shares = layer.shares_of(view)

            winner: PicturePosition | None = None
            if kind is PoolingKind.MAX:
                winning_row, winning_column = np.unravel_index(
                    int(np.argmax(shares)), shares.shape
                )
                winner = PicturePosition(
                    row=top + int(winning_row), column=left + int(winning_column)
                )

            windows.append(
                SweptWindow(
                    output_row=output_row,
                    output_column=output_column,
                    top=top,
                    left=left,
                    answer=float(pooled[output_row, output_column]),
                    winner=winner,
                    share_total=float(np.sum(shares)),
                )
            )
    return windows


class ShareTotalsRequest(BaseModel):
    """Which kind of pooling to ask, and the largest window side to ask it at."""

    kind: PoolingKind
    largest_side: int = Field(ge=1, le=MAX_SHARE_SIDE)

    model_config = {"extra": "forbid"}


class ShareTotal(BaseModel):
    """What one window's shares add up to at one side.

    ``positions`` is the side squared, how many shares were summed, and
    ``exact`` says whether the total is one to the last bit.
    """

    side: int
    positions: int
    total: float
    exact: bool


class ShareTotalsResponse(BaseModel):
    """One total per side, from one up to the side asked for.

    ``n_inexact`` counts the sides whose total is not exactly one, and
    ``worst_side`` and ``worst_gap`` name the side furthest from one and by
    how much. When every total is exact the gap is zero and the side is one.
    """

    kind: PoolingKind
    totals: list[ShareTotal]
    n_inexact: int
    worst_side: int
    worst_gap: float


@router.post("/concepts/pooling/share-totals", response_model=ShareTotalsResponse)
def share_totals(request: ShareTotalsRequest) -> ShareTotalsResponse:
    """Sum one window's shares at every side, through the layer's backward pass."""
    totals: list[ShareTotal] = []
    for side in range(1, request.largest_side + 1):
        layer = _layer_for(request.kind, side, side, side, side)
        response = layer.respond_to(np.zeros((1, CHANNELS, side, side)))
        arriving = np.full_like(response.outputs, ARRIVING_SLOPE)
        correction = layer.correction_for(response, arriving)
        total = float(np.sum(correction.passed_down[0, 0]))
        totals.append(
            ShareTotal(
                side=side, positions=side * side, total=total, exact=total == 1.0
            )
        )

    worst = max(totals, key=lambda entry: (abs(entry.total - 1.0), -entry.side))

    return ShareTotalsResponse(
        kind=request.kind,
        totals=totals,
        n_inexact=sum(1 for entry in totals if not entry.exact),
        worst_side=worst.side,
        worst_gap=abs(worst.total - 1.0),
    )


def _rows_of(grid: np.ndarray) -> list[list[float]]:
    """A two-dimensional array as plain floats, rows down and columns across."""
    return [[float(value) for value in row] for row in grid]


def _layer_reading(
    kind: PoolingKind, channels: int, height: int, width: int, window: int, stride: int
) -> Pool2d:
    """A layer over any arrangement, for the endpoints that count rather than pool."""
    reads = (channels, height, width)
    if kind is PoolingKind.MAX:
        return MaxPool2d(reads=reads, window=window, stride=stride)
    return AveragePool2d(reads=reads, window=window, stride=stride)


class GeometryRequest(BaseModel):
    """One picture's arrangement, and the windows and strides to try on it.

    Every window is paired with every stride, so eight of each describes
    sixty-four layers. Nothing is pooled; each layer is built and its shape
    read, which is the whole of what this endpoint is for.
    """

    channels: int = Field(ge=1, le=MAX_COUNTED_CHANNELS)
    height: CountedSide
    width: CountedSide
    windows: list[CountedSide] = Field(min_length=1, max_length=MAX_GEOMETRY_CHOICES)
    strides: list[CountedSide] = Field(min_length=1, max_length=MAX_GEOMETRY_CHOICES)

    model_config = {"extra": "forbid"}


class GeometryRow(BaseModel):
    """What one window and stride make of the picture, before any data arrives.

    ``refused`` marks a combination the library will not build, and ``detail``
    is its own message; every other field is empty on such a row. ``answers``
    is the arrangement one output has, ``n_inputs`` and ``n_outputs`` are how
    many numbers one input and one output hold, and ``kept_share`` is the
    second over the first. ``unvisited_rows`` and ``unvisited_columns`` are the
    picture rows and columns no window position reaches, which a stride that
    does not divide the side leaves behind. ``kinds_agree`` says whether a
    maximum and an average built on these settings answer with the same
    arrangement.
    """

    window: int
    stride: int
    refused: bool
    detail: str | None
    reads: list[int] | None
    answers: list[int] | None
    n_inputs: int | None
    n_outputs: int | None
    values_dropped: int | None
    kept_share: float | None
    overlaps: bool
    unvisited_rows: list[int]
    unvisited_columns: list[int]
    n_unvisited_cells: int
    kinds_agree: bool


class GeometryResponse(BaseModel):
    """One row per window and stride, in the order they were asked for."""

    channels: int
    height: int
    width: int
    rows: list[GeometryRow]


@router.post("/concepts/pooling/geometry", response_model=GeometryResponse)
def geometry(request: GeometryRequest) -> GeometryResponse:
    """Build a layer for every window and stride, and read the arrangement back."""
    rows = [
        _geometry_row(request, window, stride)
        for window in request.windows
        for stride in request.strides
    ]
    return GeometryResponse(
        channels=request.channels,
        height=request.height,
        width=request.width,
        rows=rows,
    )


def _geometry_row(request: GeometryRequest, window: int, stride: int) -> GeometryRow:
    """One combination, refusal included, with nothing pooled."""
    try:
        layer = _layer_reading(
            PoolingKind.MAX,
            request.channels,
            request.height,
            request.width,
            window,
            stride,
        )
        twin = _layer_reading(
            PoolingKind.AVERAGE,
            request.channels,
            request.height,
            request.width,
            window,
            stride,
        )
    except MLLibError as refusal:
        return GeometryRow(
            window=window,
            stride=stride,
            refused=True,
            detail=str(refusal),
            reads=None,
            answers=None,
            n_inputs=None,
            n_outputs=None,
            values_dropped=None,
            kept_share=None,
            overlaps=stride < window,
            unvisited_rows=[],
            unvisited_columns=[],
            n_unvisited_cells=0,
            kinds_agree=True,
        )

    _, out_height, out_width = layer.shape.answers
    unvisited_rows = _unvisited(request.height, out_height, layer.window, layer.stride)
    unvisited_columns = _unvisited(request.width, out_width, layer.window, layer.stride)
    visited_cells = (request.height - len(unvisited_rows)) * (
        request.width - len(unvisited_columns)
    )

    return GeometryRow(
        window=window,
        stride=stride,
        refused=False,
        detail=None,
        reads=list(layer.shape.reads),
        answers=list(layer.shape.answers),
        n_inputs=layer.shape.n_inputs,
        n_outputs=layer.shape.n_outputs,
        values_dropped=layer.shape.n_inputs - layer.shape.n_outputs,
        kept_share=layer.shape.n_outputs / layer.shape.n_inputs,
        overlaps=layer.stride < layer.window,
        unvisited_rows=unvisited_rows,
        unvisited_columns=unvisited_columns,
        n_unvisited_cells=request.height * request.width - visited_cells,
        kinds_agree=layer.shape.answers == twin.shape.answers,
    )


def _unvisited(extent: int, out_extent: int, window: int, stride: int) -> list[int]:
    """Which positions along one side no window position covers.

    The sweep is the layer's own, the corner of window ``k`` sitting at ``k``
    times the stride, so this counts what the layer will actually read rather
    than what the formula suggests it might.
    """
    covered: set[int] = set()
    for position in range(out_extent):
        top = position * stride
        covered.update(range(top, top + window))
    return sorted(set(range(extent)) - covered)


WindowRow = Annotated[list[PictureValue], Field(min_length=1, max_length=MAX_SIDE)]


class WindowRequest(BaseModel):
    """One square window, small enough to check with a pencil."""

    window: list[WindowRow] = Field(min_length=1, max_length=MAX_SIDE)

    model_config = {"extra": "forbid"}


class WindowSummary(BaseModel):
    """What one kind of pooling makes of the window, both ways.

    ``answer`` is what it keeps and ``shares`` is what each position is
    responsible for, the derivative of that answer with respect to that
    position. ``n_receiving`` counts the positions whose share is not zero,
    which is one under a maximum and every position under an average.
    """

    kind: PoolingKind
    answer: float
    shares: list[list[float]]
    share_total: float
    winner: PicturePosition | None
    n_receiving: int


class WindowResponse(BaseModel):
    """Both kinds on the one window, so the pair can be read side by side."""

    side: int
    positions: int
    values: list[list[float]]
    total: float
    summaries: list[WindowSummary]


@router.post("/concepts/pooling/window", response_model=WindowResponse)
def window(request: WindowRequest) -> WindowResponse:
    """Ask each kind what it keeps from one window, and what it owes each position."""
    values = _square_of(request.window)
    side = int(values.shape[0])

    return WindowResponse(
        side=side,
        positions=side * side,
        values=_rows_of(values),
        total=float(np.sum(values)),
        summaries=[_summary_of(kind, values) for kind in PoolingKind],
    )


def _square_of(rows: list[list[float]]) -> np.ndarray:
    """The window as a float array, refusing one that is not square."""
    height = len(rows)
    for position, row in enumerate(rows):
        if len(row) != height:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"a pooling window is square, and row {position + 1} holds "
                    f"{len(row)} values where the window is {height} rows deep"
                ),
            )
    return np.asarray(rows, dtype=np.float64)


def _summary_of(kind: PoolingKind, values: np.ndarray) -> WindowSummary:
    """One kind's answer and shares, taken from the layer's own two methods."""
    side = int(values.shape[0])
    layer = _layer_reading(kind, CHANNELS, side, side, side, side)
    shares = layer.shares_of(values)

    winner: PicturePosition | None = None
    if kind is PoolingKind.MAX:
        winning_row, winning_column = np.unravel_index(
            int(np.argmax(shares)), shares.shape
        )
        winner = PicturePosition(row=int(winning_row), column=int(winning_column))

    return WindowSummary(
        kind=kind,
        answer=layer.summarise(values),
        shares=_rows_of(shares),
        share_total=float(np.sum(shares)),
        winner=winner,
        n_receiving=int(np.count_nonzero(shares)),
    )


class ShiftRequest(BaseModel):
    """A picture, a window and stride, and how far to slide the picture sideways."""

    picture: list[PictureRow] = Field(min_length=1, max_length=MAX_SIDE)
    window: int = Field(ge=1, le=MAX_SIDE)
    stride: int = Field(ge=1, le=MAX_SIDE)
    shifts: list[Shift] = Field(min_length=1, max_length=MAX_SHIFTS)

    model_config = {"extra": "forbid"}


class ShiftedMap(BaseModel):
    """The pooled map at one sideways position, against the map at no shift.

    ``picture_cells_changed`` counts how many of the picture's own cells differ
    from the unshifted picture, and ``pooled_cells_changed`` how many of the
    pooled map's do, which is the comparison the tolerance claim rests on.
    """

    shift: int
    picture: list[list[float]]
    pooled: list[list[float]]
    picture_cells_changed: int
    pooled_cells_changed: int
    largest_pooled_change: float


class KindShifts(BaseModel):
    """One kind's pooled map at every shift asked for."""

    kind: PoolingKind
    maps: list[ShiftedMap]


class ShiftResponse(BaseModel):
    """Both kinds, at every shift, over one picture."""

    window: int
    stride: int
    height: int
    width: int
    pooled_height: int
    pooled_width: int
    n_picture_cells: int
    n_pooled_cells: int
    by_kind: list[KindShifts]


@router.post("/concepts/pooling/shift", response_model=ShiftResponse)
def shift(request: ShiftRequest) -> ShiftResponse:
    """Pool the picture, then pool it again moved sideways, and count what moved."""
    picture = _rectangle_of(request.picture)
    height, width = picture.shape
    layer = _layer_for(
        PoolingKind.MAX, int(height), int(width), request.window, request.stride
    )
    _, pooled_height, pooled_width = layer.shape.answers

    return ShiftResponse(
        window=layer.window,
        stride=layer.stride,
        height=int(height),
        width=int(width),
        pooled_height=int(pooled_height),
        pooled_width=int(pooled_width),
        n_picture_cells=int(height * width),
        n_pooled_cells=int(pooled_height * pooled_width),
        by_kind=[
            _shifts_under(kind, picture, request.window, request.stride, request.shifts)
            for kind in PoolingKind
        ],
    )


def _shifts_under(
    kind: PoolingKind,
    picture: np.ndarray,
    window_side: int,
    stride: int,
    shifts: list[int],
) -> KindShifts:
    """Every requested shift under one kind, each compared with no shift at all."""
    height, width = picture.shape
    layer = _layer_for(kind, int(height), int(width), window_side, stride)
    settled = layer.respond_to(picture[None, None, :, :]).outputs[0, 0]

    maps: list[ShiftedMap] = []
    for offset in shifts:
        moved = _moved_right(picture, offset)
        pooled = layer.respond_to(moved[None, None, :, :]).outputs[0, 0]
        maps.append(
            ShiftedMap(
                shift=offset,
                picture=_rows_of(moved),
                pooled=_rows_of(pooled),
                picture_cells_changed=int(np.count_nonzero(moved != picture)),
                pooled_cells_changed=int(np.count_nonzero(pooled != settled)),
                largest_pooled_change=float(np.max(np.abs(pooled - settled))),
            )
        )
    return KindShifts(kind=kind, maps=maps)


def _moved_right(picture: np.ndarray, offset: int) -> np.ndarray:
    """The picture slid ``offset`` columns to the right, zeros entering the left."""
    moved = np.zeros_like(picture)
    width = picture.shape[1]
    if offset < width:
        moved[:, offset:] = picture[:, : width - offset]
    return moved


class EdgeCase(BaseModel):
    """One thing asked of a pooling layer, and what the library did about it.

    ``raised`` says whether anything was refused at all, ``error`` names the
    class of the refusal and ``detail`` carries its message. ``library_error``
    says whether that refusal is one of the library's own rather than a bare
    builtin failure, which is the difference between a message written for a
    learner and one written for an interpreter. ``outcome`` describes in words
    what came back when nothing was refused, and it is built from the values
    that came back rather than from a sentence about them.
    """

    name: str
    attempt: str
    raised: bool
    error: str | None
    detail: str | None
    library_error: bool
    outcome: str


class PairingMistake(BaseModel):
    """Two layers of one shape, and a response handed to the wrong one.

    Both read the same arrangement and both answer with the same arrangement,
    so neither can tell the other's response from its own. The arriving block
    is the same either way and so is the total handed down; where it lands is
    not.
    """

    reads: list[int]
    answers: list[int]
    producer_window: int
    producer_stride: int
    receiver_window: int
    receiver_stride: int
    arriving: list[list[float]]
    producer_blame: list[list[float]]
    receiver_blame: list[list[float]]
    producer_total: float
    receiver_total: float
    n_cells_differing: int
    raised: bool


class EdgeCasesResponse(BaseModel):
    """Every probe the page's last part quotes, run against the library."""

    cases: list[EdgeCase]
    n_refused: int
    n_accepted: int
    pairing_mistake: PairingMistake


EDGE_PICTURE = np.array(
    [
        [1.0, 5.0, 1.0, 3.0],
        [4.0, 2.0, 4.0, 0.0],
        [0.0, 1.0, 3.0, 1.0],
        [2.0, 1.0, 7.0, 5.0],
    ]
)
"""The page's own four by four patch, so every edge is probed on familiar cells."""


@router.get("/concepts/pooling/edge-cases", response_model=EdgeCasesResponse)
def edge_cases() -> EdgeCasesResponse:
    """Ask the library for each of the page's edges, and report what it answered."""
    cases = [probe() for probe in _EDGE_PROBES]
    return EdgeCasesResponse(
        cases=cases,
        n_refused=sum(1 for case in cases if case.raised),
        n_accepted=sum(1 for case in cases if not case.raised),
        pairing_mistake=_pairing_mistake(),
    )


def _probe(name: str, attempt: str, action: Callable[[], str]) -> EdgeCase:
    """Run one attempt and report the refusal, or what came back instead.

    Every failure is caught, not only the library's own, so a bare builtin
    escaping the hierarchy shows up as a row saying so rather than as a
    server error the page cannot describe.
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
            outcome="refused before anything was read",
        )
    except Exception as failure:  # noqa: BLE001 - reporting it is the point
        return EdgeCase(
            name=name,
            attempt=attempt,
            raised=True,
            error=type(failure).__name__,
            detail=str(failure),
            library_error=False,
            outcome="refused, though not in the library's own words",
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


def _blame_under(layer: Pool2d, picture: np.ndarray) -> np.ndarray:
    """What one layer hands down over a picture when a slope of one arrives."""
    response = layer.respond_to(picture[None, None, :, :])
    arriving = np.full_like(response.outputs, ARRIVING_SLOPE)
    return layer.correction_for(response, arriving).passed_down[0, 0]


def _window_too_large() -> str:
    MaxPool2d(reads=(CHANNELS, 4, 4), window=5, stride=1)
    return "accepted"


def _stride_past_the_window() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=3)
    blame = _blame_under(layer, EDGE_PICTURE)
    untouched = int(np.count_nonzero(blame == 0.0))
    return (
        f"accepted, answering {layer.shape.answers} and leaving {untouched} of "
        f"{blame.size} cells outside every window"
    )


def _bare_integer_extents() -> str:
    MaxPool2d(reads=4, window=2, stride=2)  # type: ignore[arg-type]
    return "accepted"


def _two_extents() -> str:
    MaxPool2d(reads=(4, 4), window=2, stride=2)
    return "accepted"


def _window_of_zero() -> str:
    MaxPool2d(reads=(CHANNELS, 4, 4), window=0, stride=2)
    return "accepted"


def _stride_of_zero() -> str:
    MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=0)
    return "accepted"


def _extent_of_zero() -> str:
    MaxPool2d(reads=(CHANNELS, 0, 4), window=2, stride=2)
    return "accepted"


def _window_given_as_a_flag() -> str:
    MaxPool2d(reads=(CHANNELS, 4, 4), window=True, stride=2)
    return "accepted"


def _block_of_no_rows() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    layer.respond_to(np.zeros((0, CHANNELS, 4, 4)))
    return "accepted"


def _block_with_a_missing_value() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    spoiled = EDGE_PICTURE.copy()
    spoiled[0, 0] = np.nan
    layer.respond_to(spoiled[None, None, :, :])
    return "accepted"


def _block_of_the_wrong_arrangement() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    layer.respond_to(np.zeros((1, CHANNELS, 5, 5)))
    return "accepted"


def _arriving_block_of_the_wrong_arrangement() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    response = layer.respond_to(EDGE_PICTURE[None, None, :, :])
    layer.correction_for(response, np.ones((1, CHANNELS, 3, 3)))
    return "accepted"


def _arriving_block_of_the_wrong_row_count() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    response = layer.respond_to(EDGE_PICTURE[None, None, :, :])
    layer.correction_for(response, np.ones((3, CHANNELS, 2, 2)))
    return "accepted"


def _response_from_another_shape() -> str:
    wider = MaxPool2d(reads=(CHANNELS, 6, 6), window=2, stride=2)
    narrower = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    response = wider.respond_to(np.zeros((1, CHANNELS, 6, 6)))
    narrower.correction_for(response, np.ones((1, CHANNELS, 2, 2)))
    return "accepted"


def _window_of_one() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=1, stride=1)
    blame = _blame_under(layer, EDGE_PICTURE)
    return (
        f"accepted, answering {layer.shape.answers}, which is what it read, and "
        f"handing every one of {blame.size} cells exactly {blame.min():g}"
    )


def _a_step_with_nothing_to_step() -> str:
    layer = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)
    stepped = layer.stepped_by(None, 0.1)
    same = "the same layer object" if stepped is layer else "a different layer"
    return f"accepted, and answers with {same}"


_EDGE_PROBES: tuple[Callable[[], EdgeCase], ...] = (
    lambda: _probe(
        "a window larger than the picture",
        "a window of 5 over a picture 4 by 4",
        _window_too_large,
    ),
    lambda: _probe(
        "a stride larger than the window",
        "a window of 2 at a stride of 3 over a picture 4 by 4",
        _stride_past_the_window,
    ),
    lambda: _probe(
        "a bare integer where extents are wanted",
        "reads given as 4 rather than as three extents",
        _bare_integer_extents,
    ),
    lambda: _probe(
        "two extents rather than three",
        "reads given as (4, 4), with no channel count",
        _two_extents,
    ),
    lambda: _probe(
        "a window of zero", "a window of 0 at a stride of 2", _window_of_zero
    ),
    lambda: _probe(
        "a stride of zero", "a window of 2 at a stride of 0", _stride_of_zero
    ),
    lambda: _probe(
        "an extent of zero", "a picture 0 rows deep and 4 across", _extent_of_zero
    ),
    lambda: _probe(
        "a flag where a whole number is wanted",
        "a window given as True, which indexes as 1",
        _window_given_as_a_flag,
    ),
    lambda: _probe(
        "a block holding no rows",
        "a block arranged (0, 1, 4, 4)",
        _block_of_no_rows,
    ),
    lambda: _probe(
        "a picture with a missing value",
        "the patch with its first cell set to a missing value",
        _block_with_a_missing_value,
    ),
    lambda: _probe(
        "a block of the wrong arrangement",
        "a block arranged (1, 1, 5, 5) into a layer reading (1, 4, 4)",
        _block_of_the_wrong_arrangement,
    ),
    lambda: _probe(
        "an arriving block of the wrong arrangement",
        "a block arranged (1, 1, 3, 3) into a layer answering (1, 2, 2)",
        _arriving_block_of_the_wrong_arrangement,
    ),
    lambda: _probe(
        "an arriving block with the wrong number of rows",
        "three rows of blame for a response holding one",
        _arriving_block_of_the_wrong_row_count,
    ),
    lambda: _probe(
        "a response from a layer of another shape",
        "a response from a layer reading (1, 6, 6) into one reading (1, 4, 4)",
        _response_from_another_shape,
    ),
    lambda: _probe(
        "a window of one",
        "a window of 1 at a stride of 1 over a picture 4 by 4",
        _window_of_one,
    ),
    lambda: _probe(
        "a step asked of a layer with nothing to learn",
        "stepped by no gradient at a rate of 0.1",
        _a_step_with_nothing_to_step,
    ),
)
"""Every probe the page quotes, in the order the page's table lists them."""

PAIRING_ARRIVING = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
"""Four distinguishable arriving values, so where each landed can be told apart.

A block of ones would hide the mistake entirely, since both layers hand down
one per window either way and the two grids come out equal.
"""

PAIRING_PICTURE = np.arange(16, dtype=np.float64).reshape(4, 4)
"""Sixteen increasing values, so each window has one plain winner and no tie."""


def _pairing_mistake() -> PairingMistake:
    """Hand one layer's response to another of the same shape, and see where it lands."""
    producer = MaxPool2d(reads=(CHANNELS, 4, 4), window=3, stride=1)
    receiver = MaxPool2d(reads=(CHANNELS, 4, 4), window=2, stride=2)

    response = producer.respond_to(PAIRING_PICTURE[None, None, :, :])
    producer_blame = producer.correction_for(response, PAIRING_ARRIVING).passed_down[
        0, 0
    ]
    receiver_blame = receiver.correction_for(response, PAIRING_ARRIVING).passed_down[
        0, 0
    ]

    return PairingMistake(
        reads=list(producer.shape.reads),
        answers=list(producer.shape.answers),
        producer_window=producer.window,
        producer_stride=producer.stride,
        receiver_window=receiver.window,
        receiver_stride=receiver.stride,
        arriving=_rows_of(PAIRING_ARRIVING[0, 0]),
        producer_blame=_rows_of(producer_blame),
        receiver_blame=_rows_of(receiver_blame),
        producer_total=float(np.sum(producer_blame)),
        receiver_total=float(np.sum(receiver_blame)),
        n_cells_differing=int(np.count_nonzero(producer_blame != receiver_blame)),
        raised=False,
    )
