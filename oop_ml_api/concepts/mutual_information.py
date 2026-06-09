"""Pointwise Mutual Information.

Every endpoint here is registered under ``/concepts/pointwise-mutual-information``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about PointwiseMutualInformationEmbeddings.

Two corpora, both the ones the library's own spec for this model is pinned to.
The three sentences ``the cat sat``, ``the dog sat``, ``the cat ran`` are small
enough that every count, every rate and every score can be checked with a
pencil, and the module docstring of the model works them by hand. The
twenty-four documents are a designed corpus, twelve about cooking and twelve
about sailing, whose content words never mix and which share only ``and``,
``the`` and ``we``; document ``i`` of a topic uses that topic's words ``i`` to
``i + 4`` modulo ten, so every topic word keeps the same company as every other.
Both are reproduced here rather than imported, because ``test`` is not part of
the installed package, and ``test_mutual_information.py`` asserts the two agree
sentence for sentence with the library's own fixtures.

One quantity is computed here rather than in the library, and it is the reason
the page can say anything at all about the sign. ``PointwiseMutualInformation
Matrix.from_cooccurrence`` clips at zero, so the negative half never leaves it
and a pair that never co-occurred is indistinguishable from a pair seen below
chance. ``_scores`` computes the unclipped value from the library's own
co-occurrence totals, marks the cells where the logarithm does not exist, and
``test_mutual_information.py`` asserts that clipping and shifting it reproduces
the library's matrix entry for entry. That is what makes the comparison a
comparison of clipped against unclipped rather than of two implementations.

Nothing a widget renders may name a class, a parameter or a module, so the four
ways of reading the corpus cross the wire as the page's own words and
``_READING_LABELS`` is the only place any of them meets an identifier.
"""

from __future__ import annotations

import math
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (
    TruncatedSingularValueDecomposition,
)
from oop_ml.core.natural_language_processing.embeddings.counts.pointwise_mutual_information import (
    PointwiseMutualInformationEmbeddings,
    PointwiseMutualInformationMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/pointwise-mutual-information",
    tags=["pointwise-mutual-information"],
)


# --- The two corpora ---------------------------------------------------------

THREE_SENTENCES: tuple[str, ...] = ("the cat sat", "the dog sat", "the cat ran")
"""Five words and twelve counted neighbours, checkable with a pencil."""

COOKING: tuple[str, ...] = (
    "flour sugar and butter eggs oven",
    "sugar butter eggs the oven bake",
    "we butter eggs oven bake stir",
    "eggs oven and bake stir whisk",
    "oven bake stir the whisk dough",
    "we bake stir whisk dough pan",
    "stir whisk and dough pan flour",
    "whisk dough pan the flour sugar",
    "we dough pan flour sugar butter",
    "pan flour and sugar butter eggs",
    "flour sugar butter the eggs oven",
    "we sugar butter eggs oven bake",
)
"""Twelve documents about cooking, each five topic words and one shared one."""

SAILING: tuple[str, ...] = (
    "sail wind and boat harbour anchor",
    "wind boat harbour the anchor tide",
    "we boat harbour anchor tide mast",
    "harbour anchor and tide mast rope",
    "anchor tide mast the rope deck",
    "we tide mast rope deck crew",
    "mast rope and deck crew sail",
    "rope deck crew the sail wind",
    "we deck crew sail wind boat",
    "crew sail and wind boat harbour",
    "sail wind boat the harbour anchor",
    "we wind boat harbour anchor tide",
)
"""Twelve documents about sailing, sharing no topic word with the others."""

TWENTY_FOUR: list[str] = list(COOKING) + list(SAILING)

COOKING_WORDS: tuple[str, ...] = (
    "flour",
    "sugar",
    "butter",
    "eggs",
    "oven",
    "bake",
    "stir",
    "whisk",
    "dough",
    "pan",
)
SAILING_WORDS: tuple[str, ...] = (
    "sail",
    "wind",
    "boat",
    "harbour",
    "anchor",
    "tide",
    "mast",
    "rope",
    "deck",
    "crew",
)
SHARED_WORDS: frozenset[str] = frozenset({"and", "the", "we"})
"""The three words both halves use, which are the only overlap there is."""

TOPIC_OF: dict[str, str] = {
    **{word: "cooking" for word in COOKING_WORDS},
    **{word: "sailing" for word in SAILING_WORDS},
    **{word: "shared" for word in SHARED_WORDS},
}

CorpusChoice = Literal["three-sentences", "twenty-four-documents"]

_CORPORA: dict[CorpusChoice, list[str]] = {
    "three-sentences": list(THREE_SENTENCES),
    "twenty-four-documents": TWENTY_FOUR,
}


# --- What the page fixes -----------------------------------------------------

PAGE_WINDOW = 5
"""How far the window reaches in every fixed measurement on the larger corpus.

Five is the usual default and, measured on the ``/readings`` sweep, the value
at which every document's words all reach each other.
"""

PENCIL_WINDOW = 1
"""How far the window reaches on the three sentences: the word next door only."""

PAGE_DIMENSION = 4
"""How many numbers a word ends up with in every fixed measurement here."""

PAPER_SMOOTHING = 0.75
"""The exponent the published adjustment raises every context count to."""

PAPER_EXPONENT = 0.5
"""The published power of the singular values in the word vectors."""

MAX_SHIFT = 2.0
"""The largest amount a request may subtract before clipping.

The whole matrix on the larger corpus is zeroed a little above 1.6, so this
bound leaves the reader room to reach that and stop.
"""

SMOOTHING_STEPS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.75, 0.6, 0.5, 0.25)
"""The exponents the smoothing sweep reports, the published one among them."""

SHIFT_STEPS: tuple[float, ...] = (
    0.0,
    0.2,
    0.4,
    0.6931471805599453,
    1.0,
    1.6094379124341003,
)
"""The amounts the clipping sweep subtracts, two of them a logarithm of a count."""

WINDOW_STEPS: tuple[int, ...] = (1, 2, 3, 5, 8)
DIMENSION_STEPS: tuple[int, ...] = (2, 3, 4, 6, 8, 12, 23)
EXPONENT_STEPS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

RARE_CONTEXT_PAIR: tuple[str, str] = ("flour", "whisk")
"""A pair whose context is one of the least common words in the corpus."""

FREQUENT_CONTEXT_PAIR: tuple[str, str] = ("sugar", "the")
"""A pair whose context is one of the three commonest words in the corpus."""

BELOW_CHANCE_PAIR: tuple[str, str] = ("and", "bake")
"""A pair the corpus puts together less often than chance alone would."""


# --- Shared pieces -----------------------------------------------------------


@cache
def _tokenised(corpus: CorpusChoice) -> TokenisedCorpus:
    """One of the two corpora, split into words, once."""
    return TokenisedCorpus.from_texts(_CORPORA[corpus], WhitespacePreTokenizer())


@cache
def _vocabulary(corpus: CorpusChoice) -> Vocabulary:
    """The words of one corpus, commonest first with ties alphabetical."""
    return _tokenised(corpus).vocabulary(1)


@cache
def _cooccurrence(corpus: CorpusChoice, window: int) -> CooccurrenceMatrix:
    """How often each word appeared within ``window`` positions of each other."""
    tokenised = _tokenised(corpus)
    vocabulary = _vocabulary(corpus)
    return CooccurrenceMatrix.from_id_sequences(
        vocabulary,
        tokenised.id_sequences(vocabulary),
        window,
        ContextWeighting.UNIFORM,
    )


class Scores:
    """Every pair's score before anything was clipped, and where it exists.

    Held together rather than returned as a tuple, because a value of zero in
    ``values`` means one thing where ``observed`` is true and something else
    entirely where it is false, and separating the two arrays would let a caller
    read one without the other.
    """

    __slots__ = ("_context_rates", "_observed", "_values", "_word_rates")

    def __init__(
        self,
        values: FloatArray,
        observed: np.ndarray,
        word_rates: FloatArray,
        context_rates: FloatArray,
    ) -> None:
        self._values = values
        self._observed = observed
        self._word_rates = word_rates
        self._context_rates = context_rates

    @property
    def values(self) -> FloatArray:
        """The unclipped score of every pair; meaningless where unobserved."""
        return self._values

    @property
    def observed(self) -> np.ndarray:
        """True where the two words were seen together, so a score exists."""
        return self._observed

    @property
    def word_rates(self) -> FloatArray:
        """How much of all the counted context each word had, as a share."""
        return self._word_rates

    @property
    def context_rates(self) -> FloatArray:
        """How much of it each word served as, smoothed if it was asked for."""
        return self._context_rates


def _scores(cooccurrence: CooccurrenceMatrix, smoothing: float) -> Scores:
    """The logarithm of the observed rate over the rate chance alone predicts.

    The library clips this at zero, which is what the page's third part is
    about, so the unclipped value is worked out here from the library's own row
    totals, column totals and grand total. Nothing is computed twice: clipping
    and shifting what comes back reproduces the library's matrix entry for
    entry, and a test asserts it.
    """
    counts = np.asarray(cooccurrence)
    total = cooccurrence.total
    word_rates = cooccurrence.word_totals / total
    raised = cooccurrence.context_totals**smoothing
    context_rates = raised / raised.sum()
    expected = np.outer(word_rates, context_rates)
    observed = counts > 0.0
    values = np.zeros_like(counts)
    values[observed] = np.log((counts[observed] / total) / expected[observed])
    return Scores(values, observed, word_rates, context_rates)


def _matrix(
    corpus: CorpusChoice, window: int, smoothing: float, shift: float
) -> PointwiseMutualInformationMatrix:
    """What the library keeps: the score shifted and clipped at zero."""
    return PointwiseMutualInformationMatrix.from_cooccurrence(
        _cooccurrence(corpus, window), smoothing, shift
    )


def _fitted(
    window: int,
    dimension: int,
    smoothing: float,
    shift: float,
    exponent: float,
) -> PointwiseMutualInformationEmbeddings:
    """One fit of the twenty-four documents at the settings asked for."""
    return PointwiseMutualInformationEmbeddings(
        dimension=dimension,
        window=window,
        weighting=ContextWeighting.UNIFORM,
        context_distribution_smoothing=smoothing,
        shift=shift,
        singular_value_exponent=exponent,
    ).fit(TWENTY_FOUR)


def _cosine(rows: FloatArray, first: int, second: int) -> float | None:
    """The cosine between two rows, or nothing where one has no direction."""
    length = float(np.linalg.norm(rows[first]) * np.linalg.norm(rows[second]))
    if length == 0.0:
        return None
    return float(rows[first] @ rows[second] / length)


def _separation(
    rows: FloatArray, words: list[str]
) -> tuple[float | None, float | None]:
    """Mean cosine inside one topic, and mean cosine between the two topics.

    Only topic words count, since the three shared words belong to neither half
    and comparing them against anything says nothing about whether the halves
    came apart. Where every vector involved has collapsed to no direction at
    all there is nothing to average and the answer is nothing.
    """
    inside: list[float] = []
    for group in (COOKING_WORDS, SAILING_WORDS):
        for position, first in enumerate(group):
            for second in group[position + 1 :]:
                value = _cosine(rows, words.index(first), words.index(second))
                if value is not None:
                    inside.append(value)
    between: list[float] = []
    for first in COOKING_WORDS:
        for second in SAILING_WORDS:
            value = _cosine(rows, words.index(first), words.index(second))
            if value is not None:
                between.append(value)
    within = float(np.mean(inside)) if inside else None
    across = float(np.mean(between)) if between else None
    return within, across


class Neighbour(BaseModel):
    """One nearby word, how near it is, and which half of the corpus it is from."""

    word: str
    similarity: float
    topic: str


def _nearest(
    rows: FloatArray, words: list[str], word: str, n_results: int
) -> list[Neighbour]:
    """The nearest words by cosine, skipping any that has no direction at all."""
    if word not in words:
        return []
    origin = words.index(word)
    scored: list[tuple[float, str]] = []
    for position, other in enumerate(words):
        if position == origin:
            continue
        value = _cosine(rows, origin, position)
        if value is not None:
            scored.append((value, other))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [
        Neighbour(
            word=other,
            similarity=similarity,
            topic=TOPIC_OF.get(other, "shared"),
        )
        for similarity, other in scored[:n_results]
    ]


# --- The table of scores -----------------------------------------------------


class ScoredPair(BaseModel):
    """One pair of words, with every quantity its score was built from.

    ``score`` is the unclipped value and is nothing at all where the two words
    were never seen together, since the logarithm of zero is not a number.
    ``kept`` is what survives the shift and the clip, which is a number in every
    case.
    """

    word: str
    context: str
    count: float
    joint_rate: float
    word_rate: float
    context_rate: float
    expected_rate: float
    ratio: float | None
    score: float | None
    kept: float
    ceiling: float


class TableRequest(BaseModel):
    """Which corpus to count, how far to reach, and how to score what is there."""

    corpus: CorpusChoice = "twenty-four-documents"
    window: int = Field(default=PAGE_WINDOW, ge=1, le=10)
    context_smoothing: float = Field(default=1.0, gt=0.0, le=1.0)
    shift: float = Field(default=0.0, ge=0.0, le=MAX_SHIFT)

    model_config = {"extra": "forbid"}


class TableResponse(BaseModel):
    """Every count, every rate and every score for one corpus at one setting.

    ``scores`` holds nothing in the positions where the two words never met, so
    a reader of this response cannot confuse an absent score with a low one;
    ``kept`` holds a number everywhere, which is exactly the confusion the
    page's third part is about.
    """

    corpus: CorpusChoice
    words: list[str]
    topics: list[str]
    n_texts: int
    n_occurrences: int
    total: float
    counts: list[list[float]]
    word_totals: list[float]
    context_totals: list[float]
    word_rates: list[float]
    context_rates: list[float]
    scores: list[list[float | None]]
    kept: list[list[float]]
    n_cells: int
    n_observed: int
    n_undefined: int
    n_above_chance: int
    n_below_chance: int
    n_kept: int
    n_zero: int
    pairs: list[ScoredPair]
    highest: ScoredPair | None
    nearest_chance: ScoredPair | None
    lowest: ScoredPair | None


def _scored_pair(
    words: list[str],
    counts: FloatArray,
    scores: Scores,
    kept: FloatArray,
    total: float,
    word: int,
    context: int,
) -> ScoredPair:
    count = float(counts[word, context])
    joint_rate = count / total
    word_rate = float(scores.word_rates[word])
    context_rate = float(scores.context_rates[context])
    expected_rate = word_rate * context_rate
    seen = bool(scores.observed[word, context])
    return ScoredPair(
        word=words[word],
        context=words[context],
        count=count,
        joint_rate=joint_rate,
        word_rate=word_rate,
        context_rate=context_rate,
        expected_rate=expected_rate,
        ratio=joint_rate / expected_rate if seen and expected_rate > 0.0 else None,
        score=float(scores.values[word, context]) if seen else None,
        kept=float(kept[word, context]),
        ceiling=-math.log(max(word_rate, context_rate)),
    )


@router.post("/table", response_model=TableResponse)
def table(request: TableRequest) -> TableResponse:
    """Count one corpus, score every pair, and report what the scoring left."""
    cooccurrence = _cooccurrence(request.corpus, request.window)
    scores = _scores(cooccurrence, request.context_smoothing)
    matrix = _matrix(
        request.corpus, request.window, request.context_smoothing, request.shift
    )
    words = list(cooccurrence.vocabulary)
    counts = np.asarray(cooccurrence)
    kept = np.asarray(matrix)
    total = cooccurrence.total

    observed = scores.observed
    above = observed & (scores.values > 0.0)
    below = observed & (scores.values < 0.0)

    pairs = [
        _scored_pair(words, counts, scores, kept, total, row, column)
        for row, column in zip(*np.nonzero(observed), strict=True)
    ]
    ranked = sorted(pairs, key=lambda pair: pair.score if pair.score else 0.0)

    return TableResponse(
        corpus=request.corpus,
        words=words,
        topics=[TOPIC_OF.get(word, "shared") for word in words],
        n_texts=_tokenised(request.corpus).n_sentences,
        n_occurrences=_tokenised(request.corpus).n_words,
        total=total,
        counts=[[float(value) for value in row] for row in counts],
        word_totals=[float(value) for value in cooccurrence.word_totals],
        context_totals=[float(value) for value in cooccurrence.context_totals],
        word_rates=[float(value) for value in scores.word_rates],
        context_rates=[float(value) for value in scores.context_rates],
        scores=[
            [
                float(scores.values[row, column]) if observed[row, column] else None
                for column in range(len(words))
            ]
            for row in range(len(words))
        ],
        kept=[[float(value) for value in row] for row in kept],
        n_cells=int(counts.size),
        n_observed=int(observed.sum()),
        n_undefined=int((~observed).sum()),
        n_above_chance=int(above.sum()),
        n_below_chance=int(below.sum()),
        n_kept=matrix.n_positive,
        n_zero=int(counts.size - matrix.n_positive),
        pairs=pairs,
        highest=ranked[-1] if ranked else None,
        nearest_chance=min(
            pairs, key=lambda pair: abs(pair.score if pair.score else 0.0), default=None
        ),
        lowest=ranked[0] if ranked else None,
    )


# --- What clipping the negative half does ------------------------------------


class ShiftRow(BaseModel):
    """One amount subtracted before clipping, and what it left standing."""

    shift: float
    n_kept: int
    n_zero: int
    within_topic: float | None
    across_topic: float | None
    gap: float | None


class ClippingResponse(BaseModel):
    """The census of one matrix's cells, and what a larger shift does to it.

    ``n_undefined`` and ``n_below_chance`` are counted apart here and are the
    same number after clipping, which is the whole of the section this feeds.
    """

    n_cells: int
    n_observed: int
    n_undefined: int
    n_above_chance: int
    n_below_chance: int
    n_zero_after_clipping: int
    lowest_score: float
    mean_below_chance: float
    furthest_falls: list[ScoredPair]
    shifts: list[ShiftRow]


@router.get("/clipping", response_model=ClippingResponse)
def clipping() -> ClippingResponse:
    """How much of the larger corpus's table the clip and the shift set to zero."""
    cooccurrence = _cooccurrence("twenty-four-documents", PAGE_WINDOW)
    scores = _scores(cooccurrence, 1.0)
    matrix = _matrix("twenty-four-documents", PAGE_WINDOW, 1.0, 0.0)
    words = list(cooccurrence.vocabulary)
    counts = np.asarray(cooccurrence)
    kept = np.asarray(matrix)
    total = cooccurrence.total

    observed = scores.observed
    below = observed & (scores.values < 0.0)
    falls = sorted(
        (
            _scored_pair(words, counts, scores, kept, total, row, column)
            for row, column in zip(*np.nonzero(below), strict=True)
        ),
        key=lambda pair: (pair.score if pair.score else 0.0, pair.word, pair.context),
    )

    shifts: list[ShiftRow] = []
    for shift in SHIFT_STEPS:
        shifted = _matrix("twenty-four-documents", PAGE_WINDOW, PAPER_SMOOTHING, shift)
        model = _fitted(
            PAGE_WINDOW, PAGE_DIMENSION, PAPER_SMOOTHING, shift, PAPER_EXPONENT
        )
        within, across = _separation(np.asarray(model.embeddings.table), words)
        shifts.append(
            ShiftRow(
                shift=shift,
                n_kept=shifted.n_positive,
                n_zero=int(kept.size - shifted.n_positive),
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
            )
        )

    return ClippingResponse(
        n_cells=int(counts.size),
        n_observed=int(observed.sum()),
        n_undefined=int((~observed).sum()),
        n_above_chance=int((observed & (scores.values > 0.0)).sum()),
        n_below_chance=int(below.sum()),
        n_zero_after_clipping=int((~observed).sum() + below.sum()),
        lowest_score=float(scores.values[below].min()),
        mean_below_chance=float(scores.values[below].mean()),
        furthest_falls=falls[:8],
        shifts=shifts,
    )


# --- Smoothing the context rates ---------------------------------------------


class SmoothingRow(BaseModel):
    """One exponent, the two pairs it moves in opposite directions, and the cost.

    ``rare_context`` is a pair whose context is one of the least common words
    and ``frequent_context`` one whose context is among the three commonest, so
    the two columns move apart as the exponent falls.
    """

    exponent: float
    rare_context: float
    frequent_context: float
    below_chance: float
    n_kept: int
    n_asymmetric_entries: int
    n_asymmetric_pairs: int
    within_topic: float | None
    across_topic: float | None
    gap: float | None


class SmoothingResponse(BaseModel):
    """What raising every context count to a fractional power does, swept."""

    rare_context_pair: list[str]
    frequent_context_pair: list[str]
    below_chance_pair: list[str]
    rare_context_total: float
    frequent_context_total: float
    rows: list[SmoothingRow]
    largest_asymmetry: list[ScoredPair]


@router.get("/smoothing", response_model=SmoothingResponse)
def smoothing() -> SmoothingResponse:
    """Sweep the exponent and report which way each of three pairs moved."""
    cooccurrence = _cooccurrence("twenty-four-documents", PAGE_WINDOW)
    words = list(cooccurrence.vocabulary)
    counts = np.asarray(cooccurrence)
    total = cooccurrence.total
    rare = (words.index(RARE_CONTEXT_PAIR[0]), words.index(RARE_CONTEXT_PAIR[1]))
    frequent = (
        words.index(FREQUENT_CONTEXT_PAIR[0]),
        words.index(FREQUENT_CONTEXT_PAIR[1]),
    )
    low = (words.index(BELOW_CHANCE_PAIR[0]), words.index(BELOW_CHANCE_PAIR[1]))

    rows: list[SmoothingRow] = []
    for exponent in SMOOTHING_STEPS:
        scores = _scores(cooccurrence, exponent)
        matrix = _matrix("twenty-four-documents", PAGE_WINDOW, exponent, 0.0)
        kept = np.asarray(matrix)
        both = (kept > 0.0) & (kept.T > 0.0)
        differs = both & (np.abs(kept - kept.T) > 1e-9)
        model = _fitted(PAGE_WINDOW, PAGE_DIMENSION, exponent, 0.0, PAPER_EXPONENT)
        within, across = _separation(np.asarray(model.embeddings.table), words)
        rows.append(
            SmoothingRow(
                exponent=exponent,
                rare_context=float(scores.values[rare]),
                frequent_context=float(scores.values[frequent]),
                below_chance=float(scores.values[low]),
                n_kept=matrix.n_positive,
                n_asymmetric_entries=int(differs.sum()),
                n_asymmetric_pairs=int(differs.sum()) // 2,
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
            )
        )

    published = _scores(cooccurrence, PAPER_SMOOTHING)
    kept = np.asarray(
        _matrix("twenty-four-documents", PAGE_WINDOW, PAPER_SMOOTHING, 0.0)
    )
    both = (kept > 0.0) & (kept.T > 0.0)
    gaps = np.where(both, np.abs(kept - kept.T), 0.0)
    row, column = np.unravel_index(int(np.argmax(gaps)), gaps.shape)

    return SmoothingResponse(
        rare_context_pair=list(RARE_CONTEXT_PAIR),
        frequent_context_pair=list(FREQUENT_CONTEXT_PAIR),
        below_chance_pair=list(BELOW_CHANCE_PAIR),
        rare_context_total=float(cooccurrence.context_totals[rare[1]]),
        frequent_context_total=float(cooccurrence.context_totals[frequent[1]]),
        rows=rows,
        largest_asymmetry=[
            _scored_pair(words, counts, published, kept, total, int(row), int(column)),
            _scored_pair(words, counts, published, kept, total, int(column), int(row)),
        ],
    )


# --- From a table of scores to positions -------------------------------------


ReadingName = Literal["counts", "plain-scores", "smoothed-scores", "vectors"]

_READING_LABELS: dict[ReadingName, str] = {
    "counts": "the counts themselves",
    "plain-scores": "the scores, unsmoothed",
    "smoothed-scores": "the scores, smoothed",
    "vectors": "four numbers a word",
}
"""The page's own words for the four ways of reading the same corpus."""


class Reading(BaseModel):
    """One way of describing a word, and how far apart it puts the two topics."""

    name: ReadingName
    label: str
    numbers_per_word: int
    within_topic: float | None
    across_topic: float | None
    gap: float | None
    nearest: list[Neighbour]


class SweepRow(BaseModel):
    """One value of one setting, and the separation the fit reached under it."""

    value: float
    numbers_per_word: int
    n_kept: int
    within_topic: float | None
    across_topic: float | None
    gap: float | None


class CollapsedPair(BaseModel):
    """Two words the squeeze has made indistinguishable, and how alike they were.

    ``full_row_similarity`` is the cosine between the two words' whole rows of
    scores and ``vector_similarity`` the cosine after the squeeze, so a pair
    listed here is one the compression has placed on exactly one ray.
    """

    first: str
    second: str
    full_row_similarity: float
    vector_similarity: float


class ReadingsResponse(BaseModel):
    """Four descriptions of the same corpus compared, then three settings swept.

    The word the nearest lists are taken around is ``word``, and every reading
    answers it, so the four lists are comparable line for line.
    """

    word: str
    readings: list[Reading]
    collapsed: list[CollapsedPair]
    windows: list[SweepRow]
    dimensions: list[SweepRow]
    exponents: list[SweepRow]
    numbers_in_the_table: int
    numbers_in_the_vectors: int
    counting_milliseconds: float
    scoring_milliseconds: float
    decomposing_milliseconds: float


def _timed(work, repeats: int = 20) -> float:
    """How long one call takes, in milliseconds, over a few repetitions."""
    started = time.perf_counter()
    for _ in range(repeats):
        work()
    return (time.perf_counter() - started) * 1000.0 / repeats


@router.get("/readings", response_model=ReadingsResponse)
def readings(word: str = "flour", n_neighbours: int = 5) -> ReadingsResponse:
    """Compare four descriptions of the twenty-four documents, and sweep three settings."""
    n_neighbours = max(1, min(n_neighbours, 12))
    cooccurrence = _cooccurrence("twenty-four-documents", PAGE_WINDOW)
    words = list(cooccurrence.vocabulary)
    counts = np.asarray(cooccurrence)
    plain = np.asarray(_matrix("twenty-four-documents", PAGE_WINDOW, 1.0, 0.0))
    smoothed = np.asarray(
        _matrix("twenty-four-documents", PAGE_WINDOW, PAPER_SMOOTHING, 0.0)
    )
    model = _fitted(PAGE_WINDOW, PAGE_DIMENSION, PAPER_SMOOTHING, 0.0, PAPER_EXPONENT)
    vectors = np.asarray(model.embeddings.table)

    entries: list[Reading] = []
    for name, rows in (
        ("counts", counts),
        ("plain-scores", plain),
        ("smoothed-scores", smoothed),
        ("vectors", vectors),
    ):
        within, across = _separation(rows, words)
        entries.append(
            Reading(
                name=name,
                label=_READING_LABELS[name],
                numbers_per_word=int(rows.shape[1]),
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
                nearest=_nearest(rows, words, word, n_neighbours),
            )
        )

    windows: list[SweepRow] = []
    for window in WINDOW_STEPS:
        swept = _fitted(window, PAGE_DIMENSION, PAPER_SMOOTHING, 0.0, PAPER_EXPONENT)
        within, across = _separation(np.asarray(swept.embeddings.table), words)
        windows.append(
            SweepRow(
                value=float(window),
                numbers_per_word=PAGE_DIMENSION,
                n_kept=swept.pointwise_mutual_information.n_positive,
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
            )
        )

    dimensions: list[SweepRow] = []
    for dimension in DIMENSION_STEPS:
        swept = _fitted(PAGE_WINDOW, dimension, PAPER_SMOOTHING, 0.0, PAPER_EXPONENT)
        within, across = _separation(np.asarray(swept.embeddings.table), words)
        dimensions.append(
            SweepRow(
                value=float(dimension),
                numbers_per_word=dimension,
                n_kept=swept.pointwise_mutual_information.n_positive,
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
            )
        )

    exponents: list[SweepRow] = []
    for exponent in EXPONENT_STEPS:
        swept = _fitted(PAGE_WINDOW, PAGE_DIMENSION, PAPER_SMOOTHING, 0.0, exponent)
        within, across = _separation(np.asarray(swept.embeddings.table), words)
        exponents.append(
            SweepRow(
                value=exponent,
                numbers_per_word=PAGE_DIMENSION,
                n_kept=swept.pointwise_mutual_information.n_positive,
                within_topic=within,
                across_topic=across,
                gap=None if within is None or across is None else within - across,
            )
        )

    collapsed: list[CollapsedPair] = []
    for group in (COOKING_WORDS, SAILING_WORDS):
        for position, first in enumerate(group):
            for second in group[position + 1 :]:
                left = words.index(first)
                right = words.index(second)
                squeezed = _cosine(vectors, left, right)
                if squeezed is not None and squeezed > 0.9999:
                    collapsed.append(
                        CollapsedPair(
                            first=first,
                            second=second,
                            full_row_similarity=_cosine(smoothed, left, right) or 0.0,
                            vector_similarity=squeezed,
                        )
                    )

    return ReadingsResponse(
        word=word,
        readings=entries,
        collapsed=collapsed,
        windows=windows,
        dimensions=dimensions,
        exponents=exponents,
        numbers_in_the_table=int(smoothed.size),
        numbers_in_the_vectors=int(vectors.size),
        counting_milliseconds=_timed(
            lambda: CooccurrenceMatrix.from_id_sequences(
                _vocabulary("twenty-four-documents"),
                _tokenised("twenty-four-documents").id_sequences(
                    _vocabulary("twenty-four-documents")
                ),
                PAGE_WINDOW,
                ContextWeighting.UNIFORM,
            )
        ),
        scoring_milliseconds=_timed(
            lambda: PointwiseMutualInformationMatrix.from_cooccurrence(
                cooccurrence, PAPER_SMOOTHING, 0.0
            )
        ),
        decomposing_milliseconds=_timed(
            lambda: TruncatedSingularValueDecomposition.of(smoothed, PAGE_DIMENSION)
        ),
    )


# --- One fit, on request -----------------------------------------------------


class FitRequest(BaseModel):
    """One fit of the twenty-four documents, and the word to look around after."""

    window: int = Field(default=PAGE_WINDOW, ge=1, le=8)
    dimension: int = Field(default=PAGE_DIMENSION, ge=2, le=23)
    context_smoothing: float = Field(default=PAPER_SMOOTHING, gt=0.0, le=1.0)
    shift: float = Field(default=0.0, ge=0.0, le=MAX_SHIFT)
    singular_value_exponent: float = Field(default=PAPER_EXPONENT, ge=0.0, le=1.0)
    word: str = Field(default="flour", min_length=1, max_length=32)
    n_neighbours: int = Field(default=6, ge=1, le=12)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """A fit, what the scoring left it to work with, and where one word landed.

    ``note`` is set only where the shift has zeroed so much of the table that a
    word has no direction left, in which case the separations are nothing rather
    than a number, since there is no angle to average.
    """

    n_words: int
    n_cells: int
    n_observed: int
    n_kept: int
    n_zero: int
    singular_values: list[float]
    within_topic: float | None
    across_topic: float | None
    gap: float | None
    neighbours: list[Neighbour]
    coordinates: list[float]
    numbers_in_the_table: int
    numbers_in_the_vectors: int
    reconstruction_error: float
    n_words_without_a_direction: int
    seconds: float
    note: str | None


NO_DIRECTION_NOTE = (
    "so much of the table has been set to zero that some words are left with no "
    "direction at all, and an angle to a word that points nowhere does not exist"
)
"""What the page says when a shift has emptied a word's row entirely.

Written here rather than in the browser because it is the answer to a question
about the arithmetic; it names no identifier, and a test asserts that.
"""


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Score the corpus at these settings, decompose it, and report both halves."""
    started = time.perf_counter()
    model = _fitted(
        request.window,
        request.dimension,
        request.context_smoothing,
        request.shift,
        request.singular_value_exponent,
    )
    seconds = time.perf_counter() - started

    cooccurrence = model.cooccurrence
    words = list(cooccurrence.vocabulary)
    counts = np.asarray(cooccurrence)
    kept = np.asarray(model.pointwise_mutual_information)
    vectors = np.asarray(model.embeddings.table)
    lengths = np.linalg.norm(vectors, axis=1)
    without = int(np.count_nonzero(lengths == 0.0))
    within, across = _separation(vectors, words)
    approximation = model.decomposition.reconstruction()
    scale = float(np.linalg.norm(kept))

    return FitResponse(
        n_words=cooccurrence.n_words,
        n_cells=int(counts.size),
        n_observed=int(np.count_nonzero(counts)),
        n_kept=model.pointwise_mutual_information.n_positive,
        n_zero=int(counts.size - model.pointwise_mutual_information.n_positive),
        singular_values=[float(value) for value in model.singular_values],
        within_topic=within,
        across_topic=across,
        gap=None if within is None or across is None else within - across,
        neighbours=_nearest(vectors, words, request.word, request.n_neighbours),
        coordinates=[
            float(value)
            for value in np.asarray(model.embeddings.vector_of(request.word).values)
        ]
        if request.word in words
        else [],
        numbers_in_the_table=int(kept.size),
        numbers_in_the_vectors=int(vectors.size),
        reconstruction_error=float(np.linalg.norm(approximation - kept) / scale)
        if scale > 0.0
        else 0.0,
        n_words_without_a_direction=without,
        seconds=seconds,
        note=NO_DIRECTION_NOTE if without else None,
    )
