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

A ragged picture is refused here in words, before the library sees it. The
library would refuse it too, as a block it cannot read as numbers, which is
true and is not the mistake the learner made. A window larger than the picture
is left to the library, whose refusal already names both sizes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import AveragePool2d, MaxPool2d, Pool2d
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
