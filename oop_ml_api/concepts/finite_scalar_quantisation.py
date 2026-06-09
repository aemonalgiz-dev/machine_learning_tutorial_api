"""Finite Scalar Quantisation.

Every endpoint here is registered under ``/concepts/finite-scalar-quantisation``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about FiniteScalarQuantizer.

What this module computes
-------------------------
Five endpoints, one per family of widgets rather than one per number.
``/quantise`` takes one vector and one set of level counts and returns the whole
per-coordinate ladder, the digits, the composed id and the reconstruction, so
the playground and every worked instance on the page read one response.
``/words`` carries the seventy-two word vectors the page's tangible example is
taken from, already quantised at the published levels, together with which
pieces of the section's running sentence the corpus turns out to hold.
``/levels`` is the finding the page is built around, swept from two levels to
twelve: how many distinct rungs the obvious rounding rule reaches against how
many the half-offset rule reaches, and the same contrast on whole
configurations over the wider collection. ``/cost`` is the rounding error
against the number of codes, beside a table of the same size fitted to the same
vectors by k-means, and how evenly each of the two uses what it has.
``/limits`` carries the closing part: the scale sweep, the collection whose
first two coordinates lean on each other, the point at which the squashing
reaches one exactly, and the two ways of spending sixty-four codes.

The naive rounding rule is written out here in three lines rather than taken
from the library, because the library implements the corrected rule and the
whole point of the sweep is to contrast the two. Every number on the corrected
side comes from the quantizer itself, so the two sides are independent routes
and not one route reporting on itself.

The word vectors are read off the same eighteen English sentences the byte pair
encoding, bytes-and-characters and patching pages are built on, so the corpus a
reader meets here is the corpus they met there. Fitting them costs about ten
milliseconds and the k-means comparisons about one and a half seconds in total,
so every sweep is cached and only the one-vector endpoint recomputes per call.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a refusal needs,
and a test asserts that no identifier survives it.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import cache
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (
    LatentSemanticAnalysis,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    Codebook,
    CodebookQuantizer,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.finite_scalar import (
    FiniteScalarQuantizer,
)
from oop_ml.numpy.clustering.k_means import KMeans
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/finite-scalar-quantisation", tags=["finite-scalar-quantisation"]
)


# --------------------------------------------------------------------------
# The collections and the settings every endpoint works from.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries, and no corpus here holds."""

CORPUS: tuple[str, ...] = (
    "The report was expected on Monday.",
    "The costs were lower than the first estimate.",
    "We reviewed the results and rewrote the summary.",
    "Dr. Bell asked for the analysis of the samples.",
    "A second analysis agreed with the first analysis.",
    "The low readings weren't expected.",
    "The team rechecked the costing and the totals.",
    "Every report carries the date and the analyst's name.",
    "The revised estimate was lower again.",
    "Nobody expected the samples to arrive early.",
    "The cost of the analysis was the reason.",
    "The analysts reran the tests on Tuesday.",
    "The lowest cost was the reason the report was late.",
    "The reviewers expected a lower estimate.",
    "The high-cost option was dropped.",
    "The size of the August batch was fixed.",
    "Dr. Bell rewrote the costing and the report.",
    "The analysis was expected to cost less.",
)
"""Eighteen sentences of ordinary English, none of them the running sentence."""

EMBEDDING_DIMENSION = 4
"""How many coordinates each word gets, which is the published width."""

WORD_DIVISOR = 4.0
"""Every word vector is divided by this before it is squashed.

Chosen so that no coordinate of the collection reaches 0.99 after squashing,
which is the only thing that stops the whole vocabulary landing on corners.
Nothing in the method chooses it, which is the closing part's first claim.
"""

PAPER_LEVELS: tuple[int, ...] = (8, 5, 5, 5)
"""The published configuration: a thousand codes over four coordinates."""

NAMED_WORDS: tuple[str, ...] = (
    "cost",
    "analysis",
    "report",
    "expected",
    "lower",
    "estimate.",
    "the",
)
"""The words the page walks through, including two that collide on one code."""

HAND_WORD = "cost"
"""The one word taken through the arithmetic a reader can check with a pencil."""

WIDE_ROWS = 600
"""How many vectors the sweeps are measured over.

Seventy-two words cannot fill a grid of a thousand codes, and a table fitted to
seventy-two vectors at sixty-four codes memorises them, so every comparison
between the grid and a fitted table is taken over this wider collection instead.
"""

WIDE_SEED = 7
WIDE_SPREAD = 1.5
"""The draw behind the wider collection, fixed so every number here repeats."""

SWEEP_LEVELS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
"""The level counts the two rounding rules are contrasted at, one coordinate."""

RULE_CONFIGS: tuple[tuple[int, ...], ...] = (
    (2, 2, 2, 2),
    (4, 4, 4, 4),
    (5, 5, 5, 5),
    (8, 5, 5, 5),
    (8, 8, 8, 8),
)
"""Whole configurations the two rules are contrasted at, over the collection."""

COST_CONFIGS: tuple[tuple[int, ...], ...] = (
    (2, 2, 2, 2),
    (3, 2, 2, 2),
    (3, 3, 2, 2),
    (3, 3, 3, 2),
    (4, 4, 2, 2),
    (3, 3, 3, 3),
    (5, 4, 4, 2),
)
"""Sixteen codes up to a hundred and sixty, every one fitted against as well."""

SPENDING_CONFIGS: tuple[tuple[int, ...], ...] = (
    (8, 2, 2, 2),
    (2, 2, 2, 8),
    (4, 4, 2, 2),
    (3, 3, 3, 2),
)
"""Three ways to spend sixty-four codes, and one that spends fifty-four."""

SCALE_FACTORS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""What the collection is multiplied by before it is squashed."""

LEAN_SEED = 21
LEAN_STRENGTH = 0.95
"""How hard the second coordinate is made to follow the first."""

CLUSTER_SEED = 11
CLUSTER_RESTARTS = 3
"""The fitted table's own settings, pinned so the comparison repeats."""

MAX_LEVELS_PER_COORDINATE = 16
MAX_DIMENSION = 6
MAX_CODES = 4096
MAX_MAGNITUDE = 50.0
"""The bounds a request may ask for, so no call can turn into a compute job."""

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("code id", "code number"),
    ("code_ids", "code numbers"),
    ("n_codes", "the number of codes"),
    ("levels", "level counts"),
    ("vectors", "vectors"),
    ("dimension", "width"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the vocabulary of the code taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _refuse(refusal: MLLibError) -> HTTPException:
    return HTTPException(status_code=422, detail=_in_plain_words(refusal))


# --------------------------------------------------------------------------
# The two collections.
# --------------------------------------------------------------------------


@cache
def _word_table() -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """The vocabulary of the eighteen sentences and one vector per word."""
    fitted = LatentSemanticAnalysis(dimension=EMBEDDING_DIMENSION).fit(list(CORPUS))
    embeddings = fitted.embeddings
    table = np.asarray(embeddings.table) / WORD_DIVISOR
    words = tuple(str(word) for word in embeddings.vocabulary)
    return words, tuple(tuple(float(value) for value in row) for row in table)


def _word_block() -> np.ndarray:
    return np.array(_word_table()[1], dtype=np.float64)


@cache
def _wide_collection() -> tuple[tuple[float, ...], ...]:
    """Six hundred four-coordinate vectors, drawn once and reused everywhere."""
    block = np.random.default_rng(WIDE_SEED).normal(
        0.0, WIDE_SPREAD, size=(WIDE_ROWS, EMBEDDING_DIMENSION)
    )
    return tuple(tuple(float(value) for value in row) for row in block)


def _wide_block() -> np.ndarray:
    return np.array(_wide_collection(), dtype=np.float64)


@cache
def _leaning_collection() -> tuple[tuple[float, ...], ...]:
    """The same shape of draw, with the second coordinate following the first."""
    drawn = np.random.default_rng(LEAN_SEED).normal(
        0.0, WIDE_SPREAD, size=(WIDE_ROWS, EMBEDDING_DIMENSION)
    )
    leaning = drawn.copy()
    leaning[:, 1] = (
        LEAN_STRENGTH * drawn[:, 0] + math.sqrt(1.0 - LEAN_STRENGTH**2) * drawn[:, 1]
    )
    return tuple(tuple(float(value) for value in row) for row in leaning)


def _fitted_table_distortion(
    bounded: np.ndarray, n_codes: int
) -> tuple[float, int, int]:
    """What a table of ``n_codes`` fitted to these rows costs, and how it is used.

    Returns the mean squared gap, how many of its codes are ever chosen, and how
    many rows the busiest of them holds.
    """
    features = [
        Feature(name=f"coordinate {index}", values=bounded[:, index].tolist())
        for index in range(bounded.shape[1])
    ]
    clustering = KMeans(
        n_clusters=n_codes,
        random_seed=CLUSTER_SEED,
        n_initialisations=CLUSTER_RESTARTS,
    ).fit(features)
    assignment = CodebookQuantizer(
        codebook=Codebook.from_centroids(clustering.centroids)
    ).quantize(bounded)
    used = Counter(assignment.ids)
    return float(assignment.distortion), len(used), used.most_common(1)[0][1]


# --------------------------------------------------------------------------
# The two rounding rules, side by side.
# --------------------------------------------------------------------------


def _half_widths(levels: tuple[int, ...]) -> np.ndarray:
    return (np.array(levels, dtype=np.float64) - 1.0) / 2.0


def _offsets(levels: tuple[int, ...]) -> np.ndarray:
    counts = np.array(levels)
    return np.where(counts % 2 == 0, 0.5, 0.0)


def _naive_rounding(bounded: np.ndarray, levels: tuple[int, ...]) -> np.ndarray:
    """The rule anybody writes first: scale to the half width and round.

    Written out here rather than taken from the library, because the library
    rounds the corrected way and this is the thing it is corrected against.
    """
    return np.rint(bounded * _half_widths(levels))


def _naive_reach(levels: tuple[int, ...]) -> int:
    """How many distinct codes the naive rule can produce on ordinary inputs."""
    return math.prod(count - 1 if count % 2 == 0 else count for count in levels)


# --------------------------------------------------------------------------
# One vector, taken all the way through.
# --------------------------------------------------------------------------


class CoordinateStep(BaseModel):
    """One coordinate's whole journey, from the number to the rung it took."""

    index: int
    n_levels: int
    half_width: float
    offset: float
    value: float
    squashed: float
    scaled: float
    position: int
    digit: int
    rung: float
    squared_gap: float
    place_value: int


class QuantiseRequest(BaseModel):
    """One vector and the level counts to round each of its coordinates to."""

    values: list[Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )
    levels: list[Annotated[int, Field(ge=2, le=MAX_LEVELS_PER_COORDINATE)]] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    model_config = {"extra": "forbid"}


class QuantiseResponse(BaseModel):
    """The ladder, the digits, the composed number and what comes back out."""

    values: list[float]
    levels: list[int]
    n_codes: int
    steps: list[CoordinateStep]
    digits: list[int]
    code_id: int
    reconstruction: list[float]
    distortion: float
    id_expression: str
    product_expression: str
    bits: float


def _quantise(values: list[float], levels: tuple[int, ...]) -> QuantiseResponse:
    """The one place a vector is taken through the whole arithmetic."""
    quantizer = FiniteScalarQuantizer(levels=levels)
    block = np.array([values], dtype=np.float64)
    try:
        assignment = quantizer.quantize(block)
    except MLLibError as refusal:
        raise _refuse(refusal) from refusal

    code_id = int(assignment.ids[0])
    digits = quantizer.digits_of(code_id)
    reconstruction = [float(value) for value in assignment.reconstruction[0]]

    bounded = np.tanh(block)[0]
    half = _half_widths(levels)
    offset = _offsets(levels)
    scaled = bounded * half - offset

    steps: list[CoordinateStep] = []
    place = 1
    for index, count in enumerate(levels):
        steps.append(
            CoordinateStep(
                index=index,
                n_levels=count,
                half_width=float(half[index]),
                offset=float(offset[index]),
                value=float(values[index]),
                squashed=float(bounded[index]),
                scaled=float(scaled[index]),
                position=int(digits[index]) - count // 2,
                digit=int(digits[index]),
                rung=reconstruction[index],
                squared_gap=float((bounded[index] - reconstruction[index]) ** 2),
                place_value=place,
            )
        )
        place *= count

    terms = " + ".join(
        f"{digit} × {step.place_value}" for digit, step in zip(digits, steps)
    )
    return QuantiseResponse(
        values=[float(value) for value in values],
        levels=list(levels),
        n_codes=quantizer.n_codes,
        steps=steps,
        digits=[int(digit) for digit in digits],
        code_id=code_id,
        reconstruction=reconstruction,
        distortion=float(assignment.distortion),
        id_expression=f"{terms} = {code_id}",
        product_expression=(
            " × ".join(str(count) for count in levels) + f" = {quantizer.n_codes}"
        ),
        bits=math.log2(quantizer.n_codes),
    )


@router.post("/quantise", response_model=QuantiseResponse)
def quantise(request: QuantiseRequest) -> QuantiseResponse:
    """Squash, round, read the rungs as digits, and compose them into one number."""
    if len(request.values) != len(request.levels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"there are {len(request.values)} numbers and "
                f"{len(request.levels)} level counts, and each coordinate needs "
                f"exactly one of each"
            ),
        )
    levels = tuple(request.levels)
    if math.prod(levels) > MAX_CODES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"those level counts multiply to {math.prod(levels)} codes, and "
                f"this page works up to {MAX_CODES}"
            ),
        )
    return _quantise(request.values, levels)


# --------------------------------------------------------------------------
# The words, which are what the page's tangible example is taken from.
# --------------------------------------------------------------------------


class QuantisedWord(BaseModel):
    """One word, its four numbers, the rungs they took and the code they made."""

    word: str
    values: list[float]
    squashed: list[float]
    digits: list[int]
    code_id: int
    reconstruction: list[float]
    squared_gap: float


class SentencePiece(BaseModel):
    """One piece of the running sentence, and whether the corpus holds it."""

    piece: str
    in_the_collection: bool


class WordsResponse(BaseModel):
    """The seventy-two words, already quantised at the published level counts."""

    levels: list[int]
    n_codes: int
    n_words: int
    words: list[QuantisedWord]
    named: list[QuantisedWord]
    n_distinct_vectors: int
    n_distinct_codes: int
    n_empty_codes: int
    largest_group: int
    sharing_a_code: list[list[str]]
    sentence: str
    pieces: list[SentencePiece]
    n_pieces_in_the_collection: int
    corpus: list[str]


@cache
def _words_response() -> WordsResponse:
    words, rows = _word_table()
    block = np.array(rows, dtype=np.float64)
    quantizer = FiniteScalarQuantizer(levels=PAPER_LEVELS)
    assignment = quantizer.quantize(block)
    bounded = np.tanh(block)

    quantised = [
        QuantisedWord(
            word=word,
            values=[float(value) for value in block[index]],
            squashed=[float(value) for value in bounded[index]],
            digits=[int(digit) for digit in quantizer.digits_of(int(code))],
            code_id=int(code),
            reconstruction=[float(value) for value in assignment.reconstruction[index]],
            squared_gap=float(
                np.sum((bounded[index] - assignment.reconstruction[index]) ** 2)
            ),
        )
        for index, (word, code) in enumerate(zip(words, assignment.ids))
    ]
    by_word = {entry.word: entry for entry in quantised}

    groups: dict[int, list[str]] = {}
    for entry in quantised:
        groups.setdefault(entry.code_id, []).append(entry.word)

    # Words that never occur outside one sentence come out of the corpus with
    # the same vector, so the count that separates the corpus's doing from the
    # rounding's is how many distinct vectors there were to begin with.
    distinct_vectors = {tuple(round(value, 12) for value in row) for row in rows}

    # A group worth showing is one where different vectors met, not one where
    # the corpus had already made them the same point.
    vectors_by_code: dict[int, set[tuple[float, ...]]] = {}
    for entry in quantised:
        vectors_by_code.setdefault(entry.code_id, set()).add(
            tuple(round(value, 12) for value in entry.values)
        )
    sharing = sorted(
        (
            sorted(members)
            for code_id, members in groups.items()
            if len(vectors_by_code[code_id]) > 1
        ),
        key=len,
        reverse=True,
    )

    pieces = [
        SentencePiece(piece=piece, in_the_collection=piece in by_word)
        for piece in RUNNING_SENTENCE.split()
    ]

    return WordsResponse(
        levels=list(PAPER_LEVELS),
        n_codes=quantizer.n_codes,
        n_words=len(words),
        words=quantised,
        named=[by_word[word] for word in NAMED_WORDS if word in by_word],
        n_distinct_vectors=len(distinct_vectors),
        n_distinct_codes=len(groups),
        n_empty_codes=quantizer.n_codes - len(groups),
        largest_group=max(len(members) for members in groups.values()),
        sharing_a_code=sharing[:6],
        sentence=RUNNING_SENTENCE,
        pieces=pieces,
        n_pieces_in_the_collection=sum(
            1 for piece in pieces if piece.in_the_collection
        ),
        corpus=list(CORPUS),
    )


@router.get("/words", response_model=WordsResponse)
def words() -> WordsResponse:
    """The word vectors the page walks through, and how they land on the grid."""
    return _words_response()


# --------------------------------------------------------------------------
# The finding: what the obvious rounding rule gives at an even level count.
# --------------------------------------------------------------------------


class LevelRow(BaseModel):
    """One level count, under both rounding rules."""

    n_levels: int
    half_width: float
    offset: float
    asked_for: int
    naive_rungs: int
    naive_lowest: int
    naive_highest: int
    naive_with_saturation: int
    corrected_rungs: int
    corrected_lowest: int
    corrected_highest: int
    rungs: list[float]
    naive_rung_positions: list[float]


class ConfigurationRow(BaseModel):
    """One whole configuration, under both rules, over the wider collection."""

    levels: list[int]
    asked_for: int
    naive_reach: int
    naive_distortion: float
    corrected_distortion: float
    naive_distinct: int
    corrected_distinct: int


class LevelsResponse(BaseModel):
    """The sweep the page's central claim rests on."""

    rows: list[LevelRow]
    configurations: list[ConfigurationRow]
    saturating_input: float
    n_rows_measured: int


@cache
def _levels_response() -> LevelsResponse:
    finite = np.linspace(-10.0, 10.0, 200_001)
    saturating = np.array([-1.0e6, 1.0e6])
    both = np.concatenate([finite, saturating])

    rows: list[LevelRow] = []
    for count in SWEEP_LEVELS:
        levels = (count,)
        naive_finite = np.unique(_naive_rounding(np.tanh(finite), levels))
        naive_both = np.unique(_naive_rounding(np.tanh(both), levels))

        quantizer = FiniteScalarQuantizer(levels=levels)
        produced = np.unique(quantizer.quantize(both.reshape(-1, 1)).code_ids)
        half = float(_half_widths(levels)[0])
        offset = float(_offsets(levels)[0])
        lowest = -(count // 2)

        rows.append(
            LevelRow(
                n_levels=count,
                half_width=half,
                offset=offset,
                asked_for=count,
                naive_rungs=len(naive_finite),
                naive_lowest=int(naive_finite.min()),
                naive_highest=int(naive_finite.max()),
                naive_with_saturation=len(naive_both),
                corrected_rungs=len(produced),
                corrected_lowest=lowest,
                corrected_highest=lowest + count - 1,
                rungs=[float(value) for value in quantizer.codebook.vectors.ravel()],
                naive_rung_positions=[
                    float(position / half) for position in naive_finite
                ],
            )
        )

    block = _wide_block()
    bounded = np.tanh(block)
    configurations: list[ConfigurationRow] = []
    for levels in RULE_CONFIGS:
        quantizer = FiniteScalarQuantizer(levels=levels)
        corrected = quantizer.quantize(block)

        naive_positions = _naive_rounding(bounded, levels)
        naive_rungs = naive_positions / _half_widths(levels)
        naive_gaps = bounded - naive_rungs
        naive_codes = {tuple(row) for row in naive_positions.astype(int)}

        configurations.append(
            ConfigurationRow(
                levels=list(levels),
                asked_for=quantizer.n_codes,
                naive_reach=_naive_reach(levels),
                naive_distortion=float(
                    np.mean(np.sum(naive_gaps * naive_gaps, axis=1))
                ),
                corrected_distortion=float(corrected.distortion),
                naive_distinct=len(naive_codes),
                corrected_distinct=len(set(corrected.ids)),
            )
        )

    low, high = 18.0, 20.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if np.tanh(middle) == 1.0:
            high = middle
        else:
            low = middle

    return LevelsResponse(
        rows=rows,
        configurations=configurations,
        saturating_input=float(high),
        n_rows_measured=WIDE_ROWS,
    )


@router.get("/levels", response_model=LevelsResponse)
def levels() -> LevelsResponse:
    """How many rungs each rounding rule actually reaches, two levels to twelve."""
    return _levels_response()


# --------------------------------------------------------------------------
# What giving up the fit costs, and how evenly the codes are used.
# --------------------------------------------------------------------------


class CostRow(BaseModel):
    """One code count, rounded to a fixed grid and to a table fitted to the rows."""

    levels: list[int]
    n_codes: int
    bits: float
    grid_distortion: float
    table_distortion: float
    ratio: float
    grid_codes_used: int
    table_codes_used: int
    grid_busiest: int
    table_busiest: int


class UsageRow(BaseModel):
    """How many codes hold exactly this many of the vectors."""

    n_vectors: int
    n_codes: int


class CostResponse(BaseModel):
    """The rounding error against the number of codes, and where the rows land."""

    n_rows: int
    rows: list[CostRow]
    published_levels: list[int]
    published_codes: int
    published_distortion: float
    published_distinct: int
    published_empty: int
    published_busiest: int
    published_usage: list[UsageRow]


@cache
def _cost_response() -> CostResponse:
    block = _wide_block()
    bounded = np.tanh(block)

    rows: list[CostRow] = []
    for levels in COST_CONFIGS:
        quantizer = FiniteScalarQuantizer(levels=levels)
        grid = quantizer.quantize(block)
        grid_used = Counter(grid.ids)
        table_distortion, table_used, table_busiest = _fitted_table_distortion(
            bounded, quantizer.n_codes
        )
        rows.append(
            CostRow(
                levels=list(levels),
                n_codes=quantizer.n_codes,
                bits=math.log2(quantizer.n_codes),
                grid_distortion=float(grid.distortion),
                table_distortion=table_distortion,
                ratio=float(grid.distortion) / table_distortion,
                grid_codes_used=len(grid_used),
                table_codes_used=table_used,
                grid_busiest=grid_used.most_common(1)[0][1],
                table_busiest=table_busiest,
            )
        )
    rows.sort(key=lambda row: row.n_codes)

    published = FiniteScalarQuantizer(levels=PAPER_LEVELS)
    assignment = published.quantize(block)
    used = Counter(assignment.ids)
    histogram = Counter(used.values())

    return CostResponse(
        n_rows=WIDE_ROWS,
        rows=rows,
        published_levels=list(PAPER_LEVELS),
        published_codes=published.n_codes,
        published_distortion=float(assignment.distortion),
        published_distinct=len(used),
        published_empty=published.n_codes - len(used),
        published_busiest=max(used.values()),
        published_usage=[
            UsageRow(n_vectors=held, n_codes=count)
            for held, count in sorted(histogram.items())
        ],
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """The rounding error against a table of the same size fitted to the rows."""
    return _cost_response()


# --------------------------------------------------------------------------
# Where the method stops being defined.
# --------------------------------------------------------------------------


class ScaleRow(BaseModel):
    """The same vectors multiplied by a factor before they are squashed."""

    factor: float
    distortion: float
    codes_used: int
    share_past_the_edge: float


class LeanRow(BaseModel):
    """One configuration on the leaning collection and on the same draw without it."""

    levels: list[int]
    n_codes: int
    leaning_grid: float
    leaning_table: float
    leaning_grid_used: int
    leaning_table_used: int
    straight_grid: float
    straight_table: float
    straight_grid_used: int


class SpendingRow(BaseModel):
    """One way of spending a fixed number of codes across four coordinates."""

    levels: list[int]
    n_codes: int
    distortion: float
    codes_used: int


class LimitsResponse(BaseModel):
    """The measured facts the closing part rests on."""

    n_rows: int
    correlation: float
    scale_rows: list[ScaleRow]
    lean_rows: list[LeanRow]
    spending_rows: list[SpendingRow]
    saturating_input: float
    zero_under_even: QuantiseResponse
    zero_under_odd: QuantiseResponse


@cache
def _limits_response() -> LimitsResponse:
    block = _wide_block()
    published = FiniteScalarQuantizer(levels=PAPER_LEVELS)

    scale_rows: list[ScaleRow] = []
    for factor in SCALE_FACTORS:
        scaled = block * factor
        assignment = published.quantize(scaled)
        scale_rows.append(
            ScaleRow(
                factor=factor,
                distortion=float(assignment.distortion),
                codes_used=len(set(assignment.ids)),
                share_past_the_edge=float(np.mean(np.abs(np.tanh(scaled)) > 0.99)),
            )
        )

    leaning = np.array(_leaning_collection(), dtype=np.float64)
    straight = np.random.default_rng(LEAN_SEED).normal(
        0.0, WIDE_SPREAD, size=(WIDE_ROWS, EMBEDDING_DIMENSION)
    )
    lean_rows: list[LeanRow] = []
    for levels in ((4, 4, 2, 2), (3, 3, 3, 3)):
        quantizer = FiniteScalarQuantizer(levels=levels)

        leaning_grid = quantizer.quantize(leaning)
        leaning_table, leaning_table_used, _ = _fitted_table_distortion(
            np.tanh(leaning), quantizer.n_codes
        )
        straight_grid = quantizer.quantize(straight)
        straight_table, _, _ = _fitted_table_distortion(
            np.tanh(straight), quantizer.n_codes
        )

        lean_rows.append(
            LeanRow(
                levels=list(levels),
                n_codes=quantizer.n_codes,
                leaning_grid=float(leaning_grid.distortion),
                leaning_table=leaning_table,
                leaning_grid_used=len(set(leaning_grid.ids)),
                leaning_table_used=leaning_table_used,
                straight_grid=float(straight_grid.distortion),
                straight_table=straight_table,
                straight_grid_used=len(set(straight_grid.ids)),
            )
        )

    spending_rows: list[SpendingRow] = []
    for levels in SPENDING_CONFIGS:
        quantizer = FiniteScalarQuantizer(levels=levels)
        assignment = quantizer.quantize(block)
        spending_rows.append(
            SpendingRow(
                levels=list(levels),
                n_codes=quantizer.n_codes,
                distortion=float(assignment.distortion),
                codes_used=len(set(assignment.ids)),
            )
        )

    return LimitsResponse(
        n_rows=WIDE_ROWS,
        correlation=float(np.corrcoef(leaning[:, 0], leaning[:, 1])[0, 1]),
        scale_rows=scale_rows,
        lean_rows=lean_rows,
        spending_rows=spending_rows,
        saturating_input=_levels_response().saturating_input,
        zero_under_even=_quantise([0.0], (8,)),
        zero_under_odd=_quantise([0.0], (5,)),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The scale, the lean and the ties the closing part is built on."""
    return _limits_response()
