"""Codebook Quantisation.

Every endpoint here is registered under ``/concepts/codebook-quantisation``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about CodebookQuantizer.

What this module adds to the library
------------------------------------
Nothing mathematical. The lookup is the library's ``CodebookQuantizer``, the
tables are the library's ``KMeans`` centroids handed over by
``Codebook.from_centroids``, and the cloud's shape is the library's
``PrincipalComponentAnalysis``. What lives here is the running example: four
sixteen-by-sixteen greyscale pictures, each cut into pairs of side-by-side
pixels so that every vector on the page has two coordinates and can therefore be
drawn in the plane and checked by hand. A real image coder uses a larger patch;
two is what fits on a page.

The four pictures are chosen so that the same sweep says four different things.
The photograph is smooth, so its pairs lie close to the diagonal and a table
buys more than the high-rate prediction for two coordinates says it should. The
poster has four flat tones, so four entries very nearly finish the job and
everything after that is ordinary. The printed chart has hard edges and only
five different pairs, so a table larger than five holds entries nothing ever
uses and a table larger than the picture's variety cannot be built at all. The
speckled square is brightness drawn independently per pixel, so its pairs fill
the plane and it is the control the photograph is measured against.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a library refusal
needs, and a test asserts that no identifier survives it.
"""

from __future__ import annotations

import math
import re
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import Feature, KMeans, PrincipalComponentAnalysis
from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.distance.calculations import MinkowskiDistance
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    Codebook,
    CodebookQuantizer,
)
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/codebook-quantisation", tags=["codebook-quantisation"]
)

SIDE = 16
"""How many pixels down and across each picture is."""

COORDINATE_NAMES = ("left", "right")
"""What the two numbers of a piece are, which is what makes the plane drawable."""

FIT_SEED = 7
"""The one seed every table on the page is chosen under, so nothing drifts."""

TABLE_SIZES = (1, 2, 4, 8, 16, 32)
"""The table sizes the sweep walks, each twice the one before it."""

MAX_CODES = 64
"""The largest table a request may ask for, well above anything the page shows."""

EXACT_BELOW = 1e-12
"""Below this a rounding error is zero to every bit the arithmetic keeps.

A table holding every different piece of a picture reproduces it exactly, and
the sum of squared gaps then comes out as accumulated arithmetic noise around
1e-35 rather than as a clean zero. Dividing the previous error by that produces
a ratio in the hundreds of thousands of billions, which says nothing about the
trade, so the step is marked exact and no ratio is reported for it.
"""

Picture = Literal["photograph", "poster", "chart", "speckle"]

PICTURE_LABELS: dict[str, str] = {
    "photograph": "the photograph",
    "poster": "the poster",
    "chart": "the printed chart",
    "speckle": "the speckled square",
}
"""What each picture is called on the page. No name here is an identifier."""

PICTURE_DESCRIPTIONS: dict[str, str] = {
    "photograph": (
        "a lit shape on a background that brightens from left to right, so "
        "neighbouring pixels are nearly equal almost everywhere"
    ),
    "poster": (
        "four flat bands of tone with a faint grain over them, so the pieces "
        "gather into four tight clumps"
    ),
    "chart": (
        "hard-edged stripes at two levels, so a piece is either flat or steps "
        "across an edge and there are only five different pieces in all"
    ),
    "speckle": (
        "brightness drawn independently for every pixel, so a piece says nothing "
        "about the piece beside it and the pairs fill the square"
    ),
}
"""One sentence per picture, for the widget that lets a reader choose between them."""

_ROWS = np.arange(SIDE)[:, None]
_COLUMNS = np.arange(SIDE)[None, :]


def _photograph() -> FloatArray:
    """A smooth picture: a left-to-right gradient with a dark blot on it."""
    background = 0.20 + 0.60 * (_COLUMNS / (SIDE - 1))
    reach = np.sqrt((_ROWS - 5.0) ** 2 + (_COLUMNS - 6.0) ** 2)
    blot = 0.55 * np.exp(-(reach**2) / 12.0)
    return np.round(np.clip(background - blot, 0.0, 1.0), 2)


def _poster() -> FloatArray:
    """Four flat tones in bands, with a small seeded grain so they are not exact."""
    bands = ((_ROWS // 4) * np.ones_like(_COLUMNS)) % 4
    tones = np.array([0.08, 0.34, 0.61, 0.88])
    grain = np.random.default_rng(3).uniform(-0.03, 0.03, size=(SIDE, SIDE))
    return np.round(np.clip(tones[bands] + grain, 0.0, 1.0), 2)


def _printed_chart() -> FloatArray:
    """Hard vertical stripes, bright above and mid-grey below."""
    stripes = ((_COLUMNS // 3) % 2 == 0).astype(float)
    level = np.where(_ROWS < 8, 0.9, 0.5)
    return np.round(np.where(stripes > 0, level, 0.05), 2)


def _speckle() -> FloatArray:
    """Brightness drawn independently per pixel under a fixed seed."""
    return np.round(np.random.default_rng(11).uniform(0.0, 1.0, size=(SIDE, SIDE)), 2)


@cache
def picture_pixels(picture: Picture) -> FloatArray:
    """The brightness of every pixel of one picture, ``(SIDE, SIDE)``."""
    builders = {
        "photograph": _photograph,
        "poster": _poster,
        "chart": _printed_chart,
        "speckle": _speckle,
    }
    return builders[picture]()


@cache
def picture_pieces(picture: Picture) -> FloatArray:
    """One picture cut into pairs of side-by-side pixels, ``(SIDE * SIDE / 2, 2)``.

    Reading order, so piece ``i`` is columns ``2i`` and ``2i + 1`` of row
    ``i // (SIDE / 2)``. The widget draws the pieces back into that arrangement.
    """
    return picture_pixels(picture).reshape(-1, 2)


DUPLICATE_PATTERN = re.compile(r"^codes \d+ and \d+ are the same vector.*$")
"""The one refusal a reader can provoke here, which opens on two table positions."""

PLAIN_WORDS = (
    ("cannot find", "cannot choose"),
    ("groups in", "entries from"),
)
"""Wording from the group-finding step, said in the page's own words instead."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, rewritten so no identifier reaches the page.

    The messages here are written to be read and carry no class names, but two
    of them are phrased for somebody grouping rows rather than for somebody
    choosing a table, and the duplicate refusal names two positions in a table
    the reader never asked to see inside. A test asserts that no identifier
    survives this, so a message that grows one later fails there rather than
    reaching the page.
    """
    sentence = str(refusal)
    if DUPLICATE_PATTERN.match(sentence):
        return (
            "two entries of that table would be the same vector, which would be "
            "two numbers meaning one thing, so a table that size cannot be "
            "chosen for this picture"
        )
    for identifier, plain in PLAIN_WORDS:
        sentence = sentence.replace(identifier, plain)
    return sentence


class TableRefused(MLLibError):
    """A table of the asked-for size cannot be chosen for the asked-for picture.

    Raised in the page's own words rather than passed through, and an
    ``MLLibError`` so the application's one handler answers it as a 422 like
    every other refusal.
    """


@cache
def table_for(picture: Picture, n_codes: int) -> Codebook:
    """The table of ``n_codes`` entries chosen from ``picture``'s own pieces.

    The entries are the centres a grouping of the pieces settles on, which is
    where a table of representative vectors comes from throughout this page.

    Raises
    ------
    TableRefused
        If no table of that size exists for that picture, either because the
        picture holds fewer pieces than the table has entries or because two of
        the centres land on one point.
    """
    pieces = picture_pieces(picture)
    features = [
        Feature(name, pieces[:, position])
        for position, name in enumerate(COORDINATE_NAMES)
    ]
    try:
        grouping = KMeans(n_clusters=n_codes, random_seed=FIT_SEED).fit(features)
        return Codebook.from_centroids(grouping.centroids)
    except MLLibError as refusal:
        raise TableRefused(_in_plain_words(refusal)) from refusal


def _cloud_spread(pieces: FloatArray) -> tuple[float, float]:
    """How far the pieces spread along their leading direction, and across it."""
    analysis = PrincipalComponentAnalysis(n_components=2).fit(
        [
            Feature(name, pieces[:, position])
            for position, name in enumerate(COORDINATE_NAMES)
        ]
    )
    along, across = analysis.components
    return float(along.variance), float(across.variance)


# ---------------------------------------------------------------------------
# The four vectors and three entries the page works by hand.
# ---------------------------------------------------------------------------

HAND_CODES = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
"""Three entries in the plane, the same three the tests for this lookup pin."""

HAND_VECTORS = np.array([[1.0, 0.0], [3.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
"""Four vectors to quantise, the last of them exactly between the first two entries."""

REORDERED_CODES = np.array([[4.0, 0.0], [0.0, 0.0], [0.0, 3.0]])
"""The same three entries in a different order, which moves the tied vector."""


class HandRow(BaseModel):
    """One of the four vectors, with its distance to every entry and its answer."""

    vector: list[float]
    squared_distances: list[float]
    code_id: int
    reconstruction: list[float]
    squared_gap: float
    tied: bool


class HandResponse(BaseModel):
    """Three entries and four vectors, small enough to check with a pencil.

    ``reordered_ids`` is what the same four vectors become when the same three
    entries are listed in a different order, which changes nothing except which
    number the tied vector is given.
    """

    codes: list[list[float]]
    rows: list[HandRow]
    distortion: float
    reordered_codes: list[list[float]]
    reordered_ids: list[int]
    reordered_distortion: float


@router.get("/nearest-code", response_model=HandResponse)
def nearest_code() -> HandResponse:
    """Quantise four vectors against three entries, showing every distance."""
    quantizer = CodebookQuantizer(codebook=Codebook(HAND_CODES))
    assignment = quantizer.quantize(HAND_VECTORS)
    reconstruction = np.asarray(assignment.reconstruction)
    gaps = np.sum((HAND_VECTORS - reconstruction) ** 2, axis=1)

    rows: list[HandRow] = []
    for position, vector in enumerate(HAND_VECTORS):
        distances = np.sum((HAND_CODES - vector) ** 2, axis=1)
        smallest = float(np.min(distances))
        rows.append(
            HandRow(
                vector=vector.tolist(),
                squared_distances=[float(value) for value in distances],
                code_id=assignment.ids[position],
                reconstruction=reconstruction[position].tolist(),
                squared_gap=float(gaps[position]),
                tied=int(np.sum(distances == smallest)) > 1,
            )
        )

    reordered = CodebookQuantizer(codebook=Codebook(REORDERED_CODES)).quantize(
        HAND_VECTORS
    )
    return HandResponse(
        codes=HAND_CODES.tolist(),
        rows=rows,
        distortion=assignment.distortion,
        reordered_codes=REORDERED_CODES.tolist(),
        reordered_ids=list(reordered.ids),
        reordered_distortion=reordered.distortion,
    )


# ---------------------------------------------------------------------------
# One picture quantised against one table.
# ---------------------------------------------------------------------------


class QuantiseRequest(BaseModel):
    """Which picture to quantise, against a table chosen from which picture."""

    picture: Picture
    n_codes: Annotated[int, Field(ge=1, le=MAX_CODES)]
    table_from: Picture | None = None

    model_config = {"extra": "forbid"}


class PieceView(BaseModel):
    """One pair of side-by-side pixels, and what it became."""

    left: float
    right: float
    code_id: int
    squared_gap: float


class QuantiseResponse(BaseModel):
    """A whole picture said in numbers, and everything that cost.

    ``pixels`` and ``reconstruction`` are both ``SIDE`` rows of ``SIDE``
    brightnesses, the second built by replacing every piece with the entry it
    was given. ``usage`` counts how many pieces each entry won, so the entries
    nothing uses are the zeros in it. ``bits_per_piece`` is what a number
    costs to write down once the table is agreed, and it is zero for a table of
    one entry, which is the honest reading of a number that could not have been
    anything else.
    """

    picture: Picture
    picture_label: str
    picture_description: str
    table_from: Picture
    table_from_label: str
    side: int
    n_codes: int
    n_pieces: int
    n_distinct_pieces: int
    pixels: list[list[float]]
    reconstruction: list[list[float]]
    codes: list[list[float]]
    pieces: list[PieceView]
    usage: list[int]
    n_unused: int
    distortion: float
    root_mean_square_gap: float
    worst_squared_gap: float
    worst_piece: list[float]
    best_squared_gap: float
    best_piece: list[float]
    bits_per_piece: float
    total_id_bits: float


def _quantised(picture: Picture, table_from: Picture, n_codes: int) -> QuantiseResponse:
    """Everything one picture becomes under one table, in one object."""
    pieces = picture_pieces(picture)
    codebook = table_for(table_from, n_codes)
    assignment = CodebookQuantizer(codebook=codebook).quantize(pieces)
    reconstruction = np.asarray(assignment.reconstruction)
    gaps = np.sum((pieces - reconstruction) ** 2, axis=1)
    ids = np.asarray(assignment.code_ids)

    worst = int(np.argmax(gaps))
    best = int(np.argmin(gaps))
    usage = np.bincount(ids, minlength=n_codes)

    return QuantiseResponse(
        picture=picture,
        picture_label=PICTURE_LABELS[picture],
        picture_description=PICTURE_DESCRIPTIONS[picture],
        table_from=table_from,
        table_from_label=PICTURE_LABELS[table_from],
        side=SIDE,
        n_codes=n_codes,
        n_pieces=int(pieces.shape[0]),
        n_distinct_pieces=len(np.unique(pieces, axis=0)),
        pixels=picture_pixels(picture).tolist(),
        reconstruction=reconstruction.reshape(SIDE, SIDE).tolist(),
        codes=np.asarray(codebook.vectors).tolist(),
        pieces=[
            PieceView(
                left=float(piece[0]),
                right=float(piece[1]),
                code_id=int(ids[position]),
                squared_gap=float(gaps[position]),
            )
            for position, piece in enumerate(pieces)
        ],
        usage=[int(count) for count in usage],
        n_unused=int(np.sum(usage == 0)),
        distortion=assignment.distortion,
        root_mean_square_gap=math.sqrt(assignment.distortion / pieces.shape[1]),
        worst_squared_gap=float(gaps[worst]),
        worst_piece=pieces[worst].tolist(),
        best_squared_gap=float(gaps[best]),
        best_piece=pieces[best].tolist(),
        bits_per_piece=math.log2(n_codes),
        total_id_bits=pieces.shape[0] * math.log2(n_codes),
    )


@router.post("/quantise", response_model=QuantiseResponse)
def quantise(request: QuantiseRequest) -> QuantiseResponse:
    """Give every piece of one picture the number of the entry nearest to it."""
    return _quantised(
        request.picture, request.table_from or request.picture, request.n_codes
    )


# ---------------------------------------------------------------------------
# What a larger table buys, on all four pictures.
# ---------------------------------------------------------------------------


class SweepStep(BaseModel):
    """One table size on one picture, or the reason no such table exists.

    ``cut_by`` is how many times smaller the rounding error is than at half
    this table size, which is the number the whole trade turns on, and is
    absent at the first size because there is nothing before it and at a size
    that reproduces the picture exactly, where the ratio is arithmetic noise
    divided into a real number. ``exact`` marks that case.
    """

    n_codes: int
    bits_per_piece: float
    distortion: float | None
    cut_by: float | None
    exact: bool
    n_unused: int | None
    refused: str | None


class PictureSweep(BaseModel):
    """One picture across every table size, with the shape of its cloud.

    ``spread_along`` and ``spread_across`` are how far the pieces reach along
    their strongest direction and at right angles to it. Their ratio says
    whether the pieces fill the square or lie close to a line, which is what
    decides how much a doubled table buys.
    """

    picture: Picture
    label: str
    description: str
    n_pieces: int
    n_distinct_pieces: int
    spread_along: float
    spread_across: float
    spread_ratio: float | None
    steps: list[SweepStep]


class SweepResponse(BaseModel):
    """Every picture swept over every table size."""

    sizes: list[int]
    pictures: list[PictureSweep]


@cache
def _sweep() -> SweepResponse:
    """The whole sweep, computed once because nothing about it varies."""
    pictures: list[PictureSweep] = []
    for name in PICTURE_LABELS:
        picture: Picture = name  # type: ignore[assignment]
        pieces = picture_pieces(picture)
        along, across = _cloud_spread(pieces)
        steps: list[SweepStep] = []
        previous: float | None = None
        for n_codes in TABLE_SIZES:
            try:
                view = _quantised(picture, picture, n_codes)
            except MLLibError as refusal:
                steps.append(
                    SweepStep(
                        n_codes=n_codes,
                        bits_per_piece=math.log2(n_codes),
                        distortion=None,
                        cut_by=None,
                        exact=False,
                        n_unused=None,
                        refused=_in_plain_words(refusal),
                    )
                )
                previous = None
                continue

            exact = view.distortion < EXACT_BELOW
            cut_by = None if previous is None or exact else previous / view.distortion
            steps.append(
                SweepStep(
                    n_codes=n_codes,
                    bits_per_piece=view.bits_per_piece,
                    distortion=view.distortion,
                    cut_by=cut_by,
                    exact=exact,
                    n_unused=view.n_unused,
                    refused=None,
                )
            )
            previous = view.distortion

        pictures.append(
            PictureSweep(
                picture=picture,
                label=PICTURE_LABELS[picture],
                description=PICTURE_DESCRIPTIONS[picture],
                n_pieces=int(pieces.shape[0]),
                n_distinct_pieces=len(np.unique(pieces, axis=0)),
                spread_along=along,
                spread_across=across,
                spread_ratio=None if across == 0.0 else along / across,
                steps=steps,
            )
        )

    return SweepResponse(sizes=list(TABLE_SIZES), pictures=pictures)


@router.get("/table-size-sweep", response_model=SweepResponse)
def table_size_sweep() -> SweepResponse:
    """Every picture at every table size, with the rounding error each time."""
    return _sweep()


# ---------------------------------------------------------------------------
# Where one number stands for more than one piece.
# ---------------------------------------------------------------------------


class Collision(BaseModel):
    """Two different pieces that came back with the same number.

    ``separation`` is the straight-line distance between the two pieces, and
    ``reconstruction`` is the single vector both of them come back as.
    """

    n_codes: int
    code_id: int
    n_pieces_sharing: int
    n_distinct_sharing: int
    first: list[float]
    second: list[float]
    separation: float
    reconstruction: list[float]


class MatchProbe(BaseModel):
    """One piece of one picture put to a table, and how well it landed."""

    piece: list[float]
    code_id: int
    reconstruction: list[float]
    squared_gap: float


class SameNumberResponse(BaseModel):
    """Two ways the numbers say less than the pieces did.

    ``collisions`` are pieces that share a number inside the picture the table
    was chosen from. ``elsewhere_best`` and ``elsewhere_worst`` are the pieces
    of a different picture that the same table matched best and worst, and the
    point of showing them together is that both are one whole number and
    neither carries how far it had to move.
    """

    collisions: list[Collision]
    home_picture: Picture
    home_label: str
    home_distortion: float
    elsewhere_picture: Picture
    elsewhere_label: str
    elsewhere_distortion: float
    elsewhere_n_codes: int
    elsewhere_ratio: float
    elsewhere_best: MatchProbe
    elsewhere_worst: MatchProbe


def _widest_collision(picture: Picture, n_codes: int) -> Collision:
    """The two pieces furthest apart that still came back with one number."""
    pieces = picture_pieces(picture)
    codebook = table_for(picture, n_codes)
    assignment = CodebookQuantizer(codebook=codebook).quantize(pieces)
    ids = np.asarray(assignment.code_ids)

    widest = 0.0
    found: Collision | None = None
    for code_id in range(n_codes):
        members = pieces[ids == code_id]
        distinct = np.unique(members, axis=0)
        if len(distinct) < 2:
            continue
        separations = np.sum((distinct[:, None, :] - distinct[None, :, :]) ** 2, axis=2)
        first, second = np.unravel_index(int(np.argmax(separations)), separations.shape)
        if float(separations[first, second]) > widest:
            widest = float(separations[first, second])
            found = Collision(
                n_codes=n_codes,
                code_id=code_id,
                n_pieces_sharing=len(members),
                n_distinct_sharing=len(distinct),
                first=distinct[first].tolist(),
                second=distinct[second].tolist(),
                separation=math.sqrt(widest),
                reconstruction=np.asarray(codebook.vectors)[code_id].tolist(),
            )

    if found is None:  # pragma: no cover - every table on the page has one
        raise TableRefused("no two pieces of this picture share a number")
    return found


ELSEWHERE_CODES = 16
"""The table size the two pictures are compared at, in the middle of the sweep."""


@cache
def _same_number() -> SameNumberResponse:
    """The collisions and the misplaced picture, computed once."""
    home: Picture = "photograph"
    elsewhere: Picture = "chart"

    at_home = _quantised(home, home, ELSEWHERE_CODES)
    away = _quantised(elsewhere, home, ELSEWHERE_CODES)

    return SameNumberResponse(
        collisions=[_widest_collision(home, size) for size in (2, 4, 8, 16)],
        home_picture=home,
        home_label=PICTURE_LABELS[home],
        home_distortion=at_home.distortion,
        elsewhere_picture=elsewhere,
        elsewhere_label=PICTURE_LABELS[elsewhere],
        elsewhere_distortion=away.distortion,
        elsewhere_n_codes=ELSEWHERE_CODES,
        elsewhere_ratio=away.distortion / at_home.distortion,
        elsewhere_best=MatchProbe(
            piece=away.best_piece,
            code_id=min(away.pieces, key=lambda view: view.squared_gap).code_id,
            reconstruction=away.codes[
                min(away.pieces, key=lambda view: view.squared_gap).code_id
            ],
            squared_gap=away.best_squared_gap,
        ),
        elsewhere_worst=MatchProbe(
            piece=away.worst_piece,
            code_id=max(away.pieces, key=lambda view: view.squared_gap).code_id,
            reconstruction=away.codes[
                max(away.pieces, key=lambda view: view.squared_gap).code_id
            ],
            squared_gap=away.worst_squared_gap,
        ),
    )


@router.get("/same-number", response_model=SameNumberResponse)
def same_number() -> SameNumberResponse:
    """Pieces that share a number, and a picture the table was never chosen for."""
    return _same_number()


# ---------------------------------------------------------------------------
# Choosing the table, and the entries nothing uses.
# ---------------------------------------------------------------------------


class UnusedRow(BaseModel):
    """One table size, measured on the half it was chosen from and the other half.

    ``held_out_unused`` is the count of entries the second half never asks for,
    which is capacity the numbers pay for and the picture does not receive.
    """

    n_codes: int
    fitted_distortion: float
    fitted_unused: int
    held_out_distortion: float
    held_out_unused: int
    elsewhere_distortion: float | None
    elsewhere_unused: int | None


class StartRow(BaseModel):
    """One starting draw for the grouping that chooses the table, and what it reached."""

    seed: int
    distortion: float


class MetricDisagreement(BaseModel):
    """One piece two notions of near send to two different entries."""

    piece: list[float]
    straight_line_id: int
    straight_line_code: list[float]
    coordinate_sum_id: int
    coordinate_sum_code: list[float]
    straight_line_distances: list[float]
    coordinate_sum_distances: list[float]


class ChoosingResponse(BaseModel):
    """How the table is chosen, and what the choice leaves unused.

    ``rows`` fits the table on the top half of the photograph and measures it on
    both halves, and on the printed chart as well. ``starts`` is the same size
    of table chosen from six different starting draws. ``disagreements`` are the
    pieces that go to different entries when near is measured by adding the two
    coordinate differences rather than by the straight line.
    """

    n_pieces: int
    n_fitting_pieces: int
    n_held_out_pieces: int
    rows: list[UnusedRow]
    start_size: int
    starts: list[StartRow]
    disagreement_size: int
    n_disagreements: int
    disagreements: list[MetricDisagreement]
    largest_table: int
    largest_table_distortion: float
    refusal_size: int
    refusal: str
    one_entry_distortion: float
    spread_about_the_mean: float


@cache
def _half_table(n_codes: int) -> Codebook:
    """A table chosen from the top half of the photograph only."""
    top = picture_pieces("photograph")[: (SIDE * SIDE // 4)]
    features = [
        Feature(name, top[:, position])
        for position, name in enumerate(COORDINATE_NAMES)
    ]
    grouping = KMeans(n_clusters=n_codes, random_seed=FIT_SEED).fit(features)
    return Codebook.from_centroids(grouping.centroids)


def _measured(codebook: Codebook, pieces: FloatArray) -> tuple[float, int]:
    """The rounding error on ``pieces``, and how many entries they never ask for."""
    assignment = CodebookQuantizer(codebook=codebook).quantize(pieces)
    used = np.bincount(np.asarray(assignment.code_ids), minlength=codebook.n_codes)
    return assignment.distortion, int(np.sum(used == 0))


@cache
def _choosing() -> ChoosingResponse:
    """The whole choosing story, computed once."""
    pieces = picture_pieces("photograph")
    half = SIDE * SIDE // 4
    top, bottom = pieces[:half], pieces[half:]
    chart = picture_pieces("chart")

    rows: list[UnusedRow] = []
    for n_codes in (2, 4, 8, 16, 32):
        codebook = _half_table(n_codes)
        fitted_distortion, fitted_unused = _measured(codebook, top)
        held_distortion, held_unused = _measured(codebook, bottom)
        try:
            elsewhere = _measured(table_for("photograph", n_codes), chart)
        except MLLibError:  # pragma: no cover - the photograph never refuses here
            elsewhere = None
        rows.append(
            UnusedRow(
                n_codes=n_codes,
                fitted_distortion=fitted_distortion,
                fitted_unused=fitted_unused,
                held_out_distortion=held_distortion,
                held_out_unused=held_unused,
                elsewhere_distortion=None if elsewhere is None else elsewhere[0],
                elsewhere_unused=None if elsewhere is None else elsewhere[1],
            )
        )

    start_size = 8
    starts: list[StartRow] = []
    features = [
        Feature(name, pieces[:, position])
        for position, name in enumerate(COORDINATE_NAMES)
    ]
    for seed in range(6):
        grouping = KMeans(n_clusters=start_size, random_seed=seed).fit(features)
        book = Codebook.from_centroids(grouping.centroids)
        starts.append(
            StartRow(
                seed=seed,
                distortion=CodebookQuantizer(codebook=book).quantize(pieces).distortion,
            )
        )

    disagreement_size = 8
    codebook = table_for("photograph", disagreement_size)
    codes = np.asarray(codebook.vectors)
    straight_ids = np.asarray(
        CodebookQuantizer(codebook=codebook).quantize(pieces).code_ids
    )
    coordinate_sums = MinkowskiDistance(1.0).between(
        RowBlock(pieces, COORDINATE_NAMES), RowBlock(codes, COORDINATE_NAMES)
    )
    sum_ids = np.argmin(coordinate_sums, axis=1)
    differing = np.flatnonzero(straight_ids != sum_ids)

    disagreements = [
        MetricDisagreement(
            piece=pieces[position].tolist(),
            straight_line_id=int(straight_ids[position]),
            straight_line_code=codes[straight_ids[position]].tolist(),
            coordinate_sum_id=int(sum_ids[position]),
            coordinate_sum_code=codes[sum_ids[position]].tolist(),
            straight_line_distances=[
                float(value)
                for value in np.sqrt(np.sum((codes - pieces[position]) ** 2, axis=1))
            ],
            coordinate_sum_distances=[
                float(value) for value in coordinate_sums[position]
            ],
        )
        for position in differing
    ]

    largest = max(
        size for size in range(1, MAX_CODES + 1) if _table_exists("photograph", size)
    )
    refusal_size = largest + 1
    try:
        table_for("photograph", refusal_size)
        refusal = ""  # pragma: no cover - a table above the largest always refuses
    except MLLibError as error:
        refusal = str(error)

    return ChoosingResponse(
        n_pieces=int(pieces.shape[0]),
        n_fitting_pieces=int(top.shape[0]),
        n_held_out_pieces=int(bottom.shape[0]),
        rows=rows,
        start_size=start_size,
        starts=starts,
        disagreement_size=disagreement_size,
        n_disagreements=int(differing.size),
        disagreements=disagreements,
        largest_table=largest,
        largest_table_distortion=_quantised(
            "photograph", "photograph", largest
        ).distortion,
        refusal_size=refusal_size,
        refusal=refusal,
        one_entry_distortion=_quantised("photograph", "photograph", 1).distortion,
        spread_about_the_mean=float(
            np.mean(np.sum((pieces - pieces.mean(axis=0)) ** 2, axis=1))
        ),
    )


def _table_exists(picture: Picture, n_codes: int) -> bool:
    """Whether a table of that size can be chosen from that picture at all."""
    try:
        table_for(picture, n_codes)
    except MLLibError:
        return False
    return True


@router.get("/choosing-the-table", response_model=ChoosingResponse)
def choosing_the_table() -> ChoosingResponse:
    """Where the table comes from, what it leaves unused, and what it cannot decide."""
    return _choosing()
