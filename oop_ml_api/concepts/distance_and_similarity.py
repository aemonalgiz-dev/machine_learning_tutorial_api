"""Distance and similarity, the premise an embedding space rests on.

Every endpoint here is registered under ``/concepts/distance-and-similarity``,
and the router is created and registered before any endpoint exists, so the
pages of this section can be built in parallel without many edits to one shared
file.

The page is about what near means for two vectors, the six rules that answer it
differently, and why nearness is the whole claim a space of word positions
makes.

Three tables are served, because the page needs all three. The *hand table* is
three words against two contexts, whose entries are genuine co-occurrence counts
from the corpus below and small enough that every rule's answer can be checked
with a pencil. The *counting fit* is the same corpus scored by how much more
often two words occur together than chance would explain and squeezed to four
numbers per word, which is the table the rest of this topic uses. The
*predicting fit* is word2vec on the same corpus, and it is here for one reason:
the two fits disagree about the sign of the relationship between a word's use
count and its vector's length, which is a fact about how the vectors were
learned rather than about vectors.

Refusal messages are rewritten by ``_in_plain_words`` on the way out, because a
reader has no reason to meet a parameter name or a class, and a test pins that
no identifier survives the rewriting.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from functools import cache

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.distance.calculations import MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings import (
    PointwiseMutualInformationEmbeddings,
    Word2Vec,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

from oop_ml_api.concepts.word_vectors import TWO_TOPIC_DOCUMENTS

router = APIRouter(
    prefix="/concepts/distance-and-similarity", tags=["distance-and-similarity"]
)


WINDOW = 2
"""How many positions either side of a word count as its company."""

DIMENSION = 4
"""How many numbers the counting fit gives each word."""

PREDICTING_DIMENSION = 8
PREDICTING_EPOCHS = 5
PREDICTING_RATE = 0.05
PREDICTING_SEED = 7
"""The predicting fit's settings. The rate is twice the published default,
because at the published one every vector on a corpus this small comes out
pointing the same way."""

HOME_WORD = "sail"
"""The word the page stands at wherever one word has to be chosen."""

METRIC_NAMES: tuple[str, ...] = tuple(metric.value for metric in DistanceMetric)
"""The six rules, in the order the page reads them."""

PLAIN_NAMES: dict[str, str] = {
    "euclidean": "straight line",
    "manhattan": "summed gaps",
    "chebyshev": "worst coordinate",
    "cosine": "angle",
    "hamming": "coordinates that differ",
    "canberra": "gaps against their size",
}
"""What each rule is called on the page, since a reader meets the behaviour
rather than the name a program uses for it."""

BOUNDED: frozenset[str] = frozenset({"cosine", "hamming"})
"""The two rules whose answer cannot exceed a fixed ceiling."""

HAND_CONTEXTS: tuple[str, str] = ("oven", "wind")
HAND_WORDS: tuple[str, str, str] = ("bake", "stir", "sail")
"""Three words counted against two contexts. Small enough to work by hand, and
every entry a real count taken from the corpus."""


IDENTIFIER_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"token '(.+?)' is not in this vocabulary of (\d+).*", re.DOTALL),
        (
            r"the corpus never used the word \1, so there is no position for it "
            r"among the \2 positions this table holds"
        ),
    ),
    (
        re.compile(r"cosine similarity is undefined for a zero vector.*", re.DOTALL),
        (
            "a position at the origin points nowhere, so there is no angle to "
            "measure from it"
        ),
    ),
    (
        re.compile(r"the zero vector has no direction.*", re.DOTALL),
        (
            "a position at the origin points nowhere, so nothing can be nearest "
            "to it by angle"
        ),
    ),
)
"""Refusals rewritten in the page's own words on the way out. Anything not
matched here is passed through, and a test asserts that what comes through
carries no identifier."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifiers taken out of it."""
    sentence = str(refusal)
    for pattern, replacement in IDENTIFIER_REWRITES:
        rewritten, changes = pattern.subn(replacement, sentence)
        if changes:
            return rewritten
    return sentence


def _block(values: FloatArray) -> RowBlock:
    """These rows as the block every rule takes, so the orientation is fixed."""
    array = np.asarray(values, dtype=np.float64)
    return RowBlock(array, [f"number_{index}" for index in range(array.shape[1])])


@cache
def _counting_fit() -> WordEmbeddings:
    """The four-number table the rest of this topic works on."""
    return (
        PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=DIMENSION)
        .fit(TWO_TOPIC_DOCUMENTS)
        .embeddings
    )


@cache
def _predicting_fit() -> WordEmbeddings:
    """The same corpus, learned by predicting neighbours rather than counting."""
    return (
        Word2Vec(
            dimension=PREDICTING_DIMENSION,
            window=WINDOW,
            epochs=PREDICTING_EPOCHS,
            learning_rate=PREDICTING_RATE,
            random_seed=PREDICTING_SEED,
        )
        .fit(TWO_TOPIC_DOCUMENTS)
        .embeddings
    )


@cache
def _use_counts() -> Counter[str]:
    """How often each word was used across the whole corpus."""
    return Counter(" ".join(TWO_TOPIC_DOCUMENTS).split())


@cache
def _company_counts() -> dict[tuple[str, str], int]:
    """How often each word stood within the window of each other word."""
    counts: Counter[tuple[str, str]] = Counter()
    for document in TWO_TOPIC_DOCUMENTS:
        sentence = document.split()
        for position, word in enumerate(sentence):
            for offset in range(-WINDOW, WINDOW + 1):
                neighbour = position + offset
                if offset == 0 or not 0 <= neighbour < len(sentence):
                    continue
                counts[(word, sentence[neighbour])] += 1
    return dict(counts)


@cache
def _hand_rows() -> FloatArray:
    """The three-word table, read straight off the company counts."""
    counts = _company_counts()
    return np.asarray(
        [
            [float(counts.get((word, context), 0)) for context in HAND_CONTEXTS]
            for word in HAND_WORDS
        ],
        dtype=np.float64,
    )


def _distance_under(name: str, first: FloatArray, second: FloatArray) -> float:
    """One rule's answer for one pair of positions."""
    metric = DistanceMetric(name)
    return float(metric.between(_block(first[None, :]), _block(second[None, :]))[0, 0])


class WordRow(BaseModel):
    """One word, its position, and how much of the corpus it accounts for."""

    word: str
    use_count: int
    vector: list[float]
    length: float


class SpaceResponse(BaseModel):
    """The corpus, the table fitted to it, and what the table costs to hold.

    ``table_numbers`` is one number per word per coordinate, which is what a
    query has to read; ``n_pairs`` is how many distinct pairs of words the
    table can be asked about.
    """

    documents: list[str]
    n_documents: int
    n_words: int
    dimension: int
    window: int
    words: list[WordRow]
    table_numbers: int
    n_pairs: int
    metrics: list[str]
    plain_names: dict[str, str]


def _word_rows(space: WordEmbeddings) -> list[WordRow]:
    counts = _use_counts()
    table = np.asarray(space.table)
    return [
        WordRow(
            word=word,
            use_count=int(counts[word]),
            vector=[float(value) for value in table[token_id]],
            length=float(np.linalg.norm(table[token_id])),
        )
        for token_id, word in enumerate(space.vocabulary)
    ]


@router.get("/space", response_model=SpaceResponse)
def space() -> SpaceResponse:
    """The corpus and the table of positions fitted to it."""
    fitted = _counting_fit()
    return SpaceResponse(
        documents=list(TWO_TOPIC_DOCUMENTS),
        n_documents=len(TWO_TOPIC_DOCUMENTS),
        n_words=fitted.n_words,
        dimension=fitted.dimension,
        window=WINDOW,
        words=_word_rows(fitted),
        table_numbers=fitted.n_words * fitted.dimension,
        n_pairs=fitted.n_words * (fitted.n_words - 1) // 2,
        metrics=list(METRIC_NAMES),
        plain_names=dict(PLAIN_NAMES),
    )


class HandRequest(BaseModel):
    """The three-word table, with one word's counts multiplied.

    ``times`` at one is the table as counted. Above one it is the same word
    seen in a corpus that repeated itself, which is the thing a rule either
    notices or ignores.
    """

    times: float = Field(default=1.0, ge=1.0, le=10.0)
    scaled_word: str = Field(default="bake", min_length=1, max_length=32)

    model_config = {"extra": "forbid"}


class HandPair(BaseModel):
    """Two of the three words, and every rule's answer for them."""

    first: str
    second: str
    first_vector: list[float]
    second_vector: list[float]
    answers: dict[str, float]
    cosine_similarity: float


class HandResponse(BaseModel):
    """The three-word table and the fifteen numbers it produces."""

    contexts: list[str]
    words: list[str]
    rows: list[list[float]]
    lengths: list[float]
    times: float
    scaled_word: str
    pairs: list[HandPair]


@router.post("/by-hand", response_model=HandResponse)
def by_hand(request: HandRequest) -> HandResponse:
    """Every rule on every pair of three words counted against two contexts."""
    if request.scaled_word not in HAND_WORDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"only {', '.join(HAND_WORDS)} are counted in this small table, "
                f"and {request.scaled_word} is not one of them"
            ),
        )

    rows = _hand_rows().copy()
    rows[HAND_WORDS.index(request.scaled_word)] *= request.times

    pairs: list[HandPair] = []
    for first in range(len(HAND_WORDS)):
        for second in range(first + 1, len(HAND_WORDS)):
            try:
                similarity = cosine_similarity(rows[first], rows[second])
            except MLLibError:
                similarity = 0.0
            pairs.append(
                HandPair(
                    first=HAND_WORDS[first],
                    second=HAND_WORDS[second],
                    first_vector=[float(value) for value in rows[first]],
                    second_vector=[float(value) for value in rows[second]],
                    answers={
                        name: _distance_under(name, rows[first], rows[second])
                        for name in METRIC_NAMES
                    },
                    cosine_similarity=similarity,
                )
            )

    return HandResponse(
        contexts=list(HAND_CONTEXTS),
        words=list(HAND_WORDS),
        rows=[[float(value) for value in row] for row in rows],
        lengths=[float(np.linalg.norm(row)) for row in rows],
        times=request.times,
        scaled_word=request.scaled_word,
        pairs=pairs,
    )


class NeighbourRow(BaseModel):
    """One candidate word, and what one rule said about it."""

    word: str
    reading: float
    length: float


class RankedList(BaseModel):
    """One rule's ranking of the whole vocabulary, nearest first."""

    metric: str
    plain_name: str
    bounded: bool
    ceiling: float | None
    rows: list[NeighbourRow]


class NeighboursRequest(BaseModel):
    """Which word to stand at, and how many neighbours to name."""

    word: str = Field(min_length=1, max_length=64)
    n_results: int = Field(default=6, ge=1, le=22)

    model_config = {"extra": "forbid"}


class NeighboursResponse(BaseModel):
    """The same word's neighbours under every rule.

    ``comparisons`` is how many other positions had to be read to answer once,
    and ``multiplications`` how many products that came to.
    """

    word: str
    vector: list[float]
    length: float
    rankings: list[RankedList]
    n_rules_agreeing_on_first: int
    first_places: dict[str, str]
    comparisons: int
    multiplications: int


def _ranked_lists(space: WordEmbeddings, word: str, n_results: int) -> list[RankedList]:
    words = list(space.vocabulary)
    table = np.asarray(space.table)
    lengths = np.linalg.norm(table, axis=1)
    position = words.index(word)

    rankings: list[RankedList] = []
    for name in METRIC_NAMES:
        readings = DistanceMetric(name).between(
            _block(table[position][None, :]), _block(table)
        )[0]
        order = [
            index
            for index in np.argsort(readings, kind="stable")
            if int(index) != position
        ]
        rankings.append(
            RankedList(
                metric=name,
                plain_name=PLAIN_NAMES[name],
                bounded=name in BOUNDED,
                ceiling=(2.0 if name == "cosine" else 1.0) if name in BOUNDED else None,
                rows=[
                    NeighbourRow(
                        word=words[int(index)],
                        reading=float(readings[int(index)]),
                        length=float(lengths[int(index)]),
                    )
                    for index in order[:n_results]
                ],
            )
        )
    return rankings


@router.post("/neighbours", response_model=NeighboursResponse)
def neighbours(request: NeighboursRequest) -> NeighboursResponse:
    """Rank the vocabulary from one word under each of the six rules."""
    fitted = _counting_fit()
    if request.word not in fitted:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the corpus never used the word {request.word}, so there is no "
                f"position for it among the {fitted.n_words} positions this "
                f"table holds"
            ),
        )

    rankings = _ranked_lists(fitted, request.word, request.n_results)
    first_places = {
        ranking.metric: ranking.rows[0].word for ranking in rankings if ranking.rows
    }
    most_popular = Counter(first_places.values()).most_common(1)
    table = np.asarray(fitted.table)
    position = list(fitted.vocabulary).index(request.word)

    return NeighboursResponse(
        word=request.word,
        vector=[float(value) for value in table[position]],
        length=float(np.linalg.norm(table[position])),
        rankings=rankings,
        n_rules_agreeing_on_first=most_popular[0][1] if most_popular else 0,
        first_places=first_places,
        comparisons=fitted.n_words - 1,
        multiplications=(fitted.n_words - 1) * fitted.dimension,
    )


class LengthRow(BaseModel):
    """One word's use count beside the length its vector came out at."""

    word: str
    use_count: int
    counting_length: float
    predicting_length: float


class LengthsResponse(BaseModel):
    """Two fits of one corpus, and what each made a word's length mean.

    ``counting_correlation`` and ``predicting_correlation`` are the ordinary
    correlation between a word's use count and its vector's length under each
    fit. They come out with opposite signs, which is the finding.
    """

    rows: list[LengthRow]
    counting_correlation: float
    predicting_correlation: float
    counting_dimension: int
    predicting_dimension: int
    commonest_words: list[str]
    commonest_count: int


@cache
def _lengths() -> LengthsResponse:
    counting = _counting_fit()
    predicting = _predicting_fit()
    counts = _use_counts()

    words = list(counting.vocabulary)
    counting_lengths = np.linalg.norm(np.asarray(counting.table), axis=1)
    predicting_table = np.asarray(predicting.table)
    predicting_words = list(predicting.vocabulary)
    predicting_lengths = np.asarray(
        [
            float(np.linalg.norm(predicting_table[predicting_words.index(word)]))
            for word in words
        ]
    )
    use_counts = np.asarray([float(counts[word]) for word in words])

    commonest = max(counts.values())
    return LengthsResponse(
        rows=[
            LengthRow(
                word=word,
                use_count=int(counts[word]),
                counting_length=float(counting_lengths[index]),
                predicting_length=float(predicting_lengths[index]),
            )
            for index, word in enumerate(words)
        ],
        counting_correlation=float(np.corrcoef(counting_lengths, use_counts)[0, 1]),
        predicting_correlation=float(np.corrcoef(predicting_lengths, use_counts)[0, 1]),
        counting_dimension=counting.dimension,
        predicting_dimension=predicting.dimension,
        commonest_words=sorted(word for word in words if counts[word] == commonest),
        commonest_count=int(commonest),
    )


@router.get("/lengths", response_model=LengthsResponse)
def lengths() -> LengthsResponse:
    """A word's use count against the length of its vector, under two fits."""
    return _lengths()


class Disagreement(BaseModel):
    """A word two rules place differently, and where each of them put it."""

    word: str
    first_choice: str
    second_choice: str


class CoordinateMatch(BaseModel):
    """Two words, and how many coordinates they hold in common exactly."""

    first: str
    second: str
    n_matching: int


class AgreementResponse(BaseModel):
    """Which rules pick the same nearest word, over the whole vocabulary.

    ``grid[i][j]`` is how many of the ``n_words`` words the rules at positions
    ``i`` and ``j`` give the same nearest word. ``unit_orders_agree`` says
    whether ranking by angle and ranking by straight line coincide once every
    position is scaled to length one, which is the identity the page rests the
    section on.
    """

    metrics: list[str]
    plain_names: list[str]
    grid: list[list[int]]
    n_words: int
    nearest: dict[str, list[str]]
    words: list[str]
    angle_against_line: list[Disagreement]
    n_all_but_labels_agree: int
    exact_coordinate_matches: int
    n_pairs: int
    best_label_match: CoordinateMatch
    unit_orders_agree: bool
    largest_identity_gap: float
    raw_order_matches_unit_order: bool


@cache
def _agreement() -> AgreementResponse:
    fitted = _counting_fit()
    words = list(fitted.vocabulary)
    table = np.asarray(fitted.table)

    nearest: dict[str, list[str]] = {}
    for name in METRIC_NAMES:
        readings = DistanceMetric(name).between(_block(table), _block(table))
        np.fill_diagonal(readings, np.inf)
        nearest[name] = [words[int(index)] for index in np.argmin(readings, axis=1)]

    grid = [
        [
            sum(
                1
                for index in range(len(words))
                if nearest[first][index] == nearest[second][index]
            )
            for second in METRIC_NAMES
        ]
        for first in METRIC_NAMES
    ]

    disagreements = [
        Disagreement(
            word=word,
            first_choice=nearest["euclidean"][index],
            second_choice=nearest["cosine"][index],
        )
        for index, word in enumerate(words)
        if nearest["euclidean"][index] != nearest["cosine"][index]
    ]

    without_labels = [name for name in METRIC_NAMES if name != "hamming"]
    n_agreeing = sum(
        1
        for index in range(len(words))
        if len({nearest[name][index] for name in without_labels}) == 1
    )

    matches: dict[tuple[str, str], int] = {}
    for first in range(len(words)):
        for second in range(first + 1, len(words)):
            matches[(words[first], words[second])] = int(
                np.sum(table[first] == table[second])
            )
    best = max(matches.items(), key=lambda entry: entry[1])

    unit = table / np.linalg.norm(table, axis=1)[:, None]
    position = words.index(HOME_WORD)
    cosines = unit @ unit[position]
    identity = np.sqrt(np.maximum(2.0 - 2.0 * cosines, 0.0))
    straight = DistanceMetric.EUCLIDEAN.between(
        _block(unit[position][None, :]), _block(unit)
    )[0]
    by_cosine = [
        words[int(index)]
        for index in np.argsort(-cosines, kind="stable")
        if int(index) != position
    ]
    by_line = [
        words[int(index)]
        for index in np.argsort(straight, kind="stable")
        if int(index) != position
    ]
    raw = DistanceMetric.EUCLIDEAN.between(
        _block(table[position][None, :]), _block(table)
    )[0]
    by_raw = [
        words[int(index)]
        for index in np.argsort(raw, kind="stable")
        if int(index) != position
    ]

    return AgreementResponse(
        metrics=list(METRIC_NAMES),
        plain_names=[PLAIN_NAMES[name] for name in METRIC_NAMES],
        grid=grid,
        n_words=len(words),
        nearest=nearest,
        words=words,
        angle_against_line=disagreements,
        n_all_but_labels_agree=n_agreeing,
        exact_coordinate_matches=sum(1 for value in matches.values() if value > 0),
        n_pairs=len(matches),
        best_label_match=CoordinateMatch(
            first=best[0][0], second=best[0][1], n_matching=best[1]
        ),
        unit_orders_agree=by_cosine == by_line,
        largest_identity_gap=float(np.max(np.abs(identity - straight))),
        raw_order_matches_unit_order=by_raw == by_line,
    )


@router.get("/agreement", response_model=AgreementResponse)
def agreement() -> AgreementResponse:
    """How often two rules name the same nearest word, over every word."""
    return _agreement()


class MovedRow(BaseModel):
    """One shift, and the three answers it produces."""

    shift: float
    definition: float
    expanded: float
    shifted_first: float
    digits_lost: int


class MovedRequest(BaseModel):
    """How far to move both positions before measuring the gap between them."""

    shifts: list[float] = Field(
        default=[0.0, 1.0e02, 1.0e04, 1.0e06, 1.0e08], min_length=1, max_length=12
    )

    model_config = {"extra": "forbid"}


class MovedResponse(BaseModel):
    """One pair of word positions, moved away from the origin and remeasured.

    ``definition`` subtracts first and squares afterwards; ``expanded`` uses
    the identity that turns the whole table into one matrix multiply, without
    moving anything first; ``shifted_first`` uses the same identity after
    taking the pair's own mean out. The gap between the second and the other
    two is the whole point.
    """

    first: str
    second: str
    first_vector: list[float]
    second_vector: list[float]
    rows: list[MovedRow]
    largest_relative_gap: float
    n_random_pairs: int


WORKING_DIGITS = 16
"""How many decimal digits a double-precision number carries, near enough."""


def _digits_lost(truth: float, answer: float) -> int:
    """How many of the sixteen working digits the shortcut gave up."""
    if truth == 0.0:
        return 0
    relative = abs(answer - truth) / abs(truth)
    if relative <= 0.0:
        return 0
    matching = max(0, min(WORKING_DIGITS, round(-float(np.log10(relative)))))
    return WORKING_DIGITS - matching


@cache
def _moved(shifts: tuple[float, ...]) -> MovedResponse:
    fitted = _counting_fit()
    words = list(fitted.vocabulary)
    table = np.asarray(fitted.table)
    pair = table[[words.index(HOME_WORD), words.index("mast")]]

    rows: list[MovedRow] = []
    for shift in shifts:
        moved = pair + shift
        definition = float(np.sqrt(np.sum((moved[0] - moved[1]) ** 2)))
        expanded_square = float(
            np.dot(moved[0], moved[0])
            - 2.0 * np.dot(moved[0], moved[1])
            + np.dot(moved[1], moved[1])
        )
        expanded = float(np.sqrt(max(expanded_square, 0.0)))
        shifted = float(
            DistanceMetric.EUCLIDEAN.between(_block(moved[0:1]), _block(moved))[0, 1]
        )
        rows.append(
            MovedRow(
                shift=float(shift),
                definition=definition,
                expanded=expanded,
                shifted_first=shifted,
                digits_lost=_digits_lost(definition, expanded),
            )
        )

    generator = np.random.default_rng(3)
    queries = generator.normal(size=(40, 12))
    remembered = generator.normal(size=(60, 12))
    fast = DistanceMetric.EUCLIDEAN.between(_block(queries), _block(remembered))
    slow = MinkowskiDistance(2).between(_block(queries), _block(remembered))
    largest = float(np.max(np.abs(fast - slow) / np.maximum(slow, 1.0e-300)))

    return MovedResponse(
        first=HOME_WORD,
        second="mast",
        first_vector=[float(value) for value in pair[0]],
        second_vector=[float(value) for value in pair[1]],
        rows=rows,
        largest_relative_gap=largest,
        n_random_pairs=queries.shape[0] * remembered.shape[0],
    )


@router.post("/moved", response_model=MovedResponse)
def moved(request: MovedRequest) -> MovedResponse:
    """Measure one gap three ways, with both positions moved from the origin."""
    return _moved(tuple(request.shifts))


class ContrastRow(BaseModel):
    """One width, and how much further the farthest word is than the nearest."""

    dimension: int
    fitted: float
    uniform: float


class ContrastResponse(BaseModel):
    """The fitted table's contrast against random positions of the same shape.

    Contrast is ``(farthest - nearest) / nearest``, taken from each word in
    turn and reported as the median over the vocabulary. A random table of the
    same shape is the control, because the usual demonstration draws its
    positions independently and a fitted table is not independent.
    """

    rows: list[ContrastRow]
    n_words: int
    crossover_dimension: int | None


@cache
def _contrast() -> ContrastResponse:
    rows: list[ContrastRow] = []
    crossover: int | None = None
    for dimension in (2, 4, 8, 16):
        fitted = (
            PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=dimension)
            .fit(TWO_TOPIC_DOCUMENTS)
            .embeddings
        )
        real = np.asarray(fitted.table)
        generator = np.random.default_rng(0)
        drawn = generator.uniform(size=real.shape)

        readings = [_median_contrast(block) for block in (real, drawn)]
        rows.append(
            ContrastRow(dimension=dimension, fitted=readings[0], uniform=readings[1])
        )
        if crossover is None and readings[0] < readings[1]:
            crossover = dimension

    return ContrastResponse(
        rows=rows, n_words=_counting_fit().n_words, crossover_dimension=crossover
    )


def _median_contrast(rows: FloatArray) -> float:
    """The median over rows of how much further the farthest is than the nearest."""
    readings = DistanceMetric.EUCLIDEAN.between(_block(rows), _block(rows))
    np.fill_diagonal(readings, np.inf)
    nearest = readings.min(axis=1)
    readings[~np.isfinite(readings)] = -np.inf
    farthest = readings.max(axis=1)
    return float(np.median((farthest - nearest) / nearest))


@router.get("/contrast", response_model=ContrastResponse)
def contrast() -> ContrastResponse:
    """How much nearest and farthest part as the number of coordinates grows."""
    return _contrast()


class CostRow(BaseModel):
    """One vocabulary size, and what one question costs against it."""

    n_words: int
    dimension: int
    multiplications: int
    milliseconds: float | None


class CostResponse(BaseModel):
    """What one nearest-word question costs as a vocabulary grows.

    The largest row carries no timing, because timing it on demand would make
    the page slow for the sake of a number the arithmetic already gives.
    """

    rows: list[CostRow]
    repeats: int


COST_SHAPES: tuple[tuple[int, int, bool], ...] = (
    (23, 4, True),
    (1_000, 100, True),
    (10_000, 300, True),
    (50_000, 300, True),
    (200_000, 300, False),
)
"""Vocabulary sizes the page walks up, and whether each one is timed."""

COST_REPEATS = 5


@cache
def _cost() -> CostResponse:
    generator = np.random.default_rng(1)
    rows: list[CostRow] = []
    for n_words, dimension, timed in COST_SHAPES:
        milliseconds: float | None = None
        if timed:
            table = generator.normal(size=(n_words, dimension))
            query = _block(table[0:1])
            whole = _block(table)
            DistanceMetric.COSINE.between(query, whole)
            started = time.perf_counter()
            for _ in range(COST_REPEATS):
                DistanceMetric.COSINE.between(query, whole)
            milliseconds = (time.perf_counter() - started) / COST_REPEATS * 1000.0
        rows.append(
            CostRow(
                n_words=n_words,
                dimension=dimension,
                multiplications=n_words * dimension,
                milliseconds=milliseconds,
            )
        )
    return CostResponse(rows=rows, repeats=COST_REPEATS)


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """What one question costs against vocabularies of rising size."""
    return _cost()


class UndefinedCase(BaseModel):
    """One degenerate input, what the mathematics says, and what came back."""

    situation: str
    what_is_undefined: str
    outcome: str


class AsymmetricPair(BaseModel):
    """A word, its nearest, and the word that one names in return."""

    word: str
    nearest: str
    returned: str


class UndefinedResponse(BaseModel):
    """The cases where nearness has no answer, or has one only by convention."""

    cases: list[UndefinedCase]
    asymmetric: list[AsymmetricPair]
    n_words: int
    n_asymmetric: int
    symmetry_gap: float


@cache
def _undefined() -> UndefinedResponse:
    fitted = _counting_fit()
    words = list(fitted.vocabulary)
    table = np.asarray(fitted.table)
    home = table[words.index(HOME_WORD)]
    origin = np.zeros(fitted.dimension)

    with_origin = WordEmbeddings(
        Vocabulary([*words, "silent"]), np.vstack([table, origin[None, :]])
    )

    cases: list[UndefinedCase] = []

    try:
        with_origin.similarity(HOME_WORD, "silent")
        angle_outcome = "an angle was returned"
    except MLLibError as refusal:
        angle_outcome = _in_plain_words(refusal)
    cases.append(
        UndefinedCase(
            situation="a word whose position is the origin, compared by angle",
            what_is_undefined=(
                "the cosine divides by both lengths, and one of them is zero, so "
                "the quotient does not exist"
            ),
            outcome=angle_outcome,
        )
    )

    cases.append(
        UndefinedCase(
            situation="the same word, compared by straight line",
            what_is_undefined=(
                "nothing, since a straight-line gap needs no division and the "
                "origin is an ordinary point for it"
            ),
            outcome=(
                f"{_distance_under('euclidean', origin, home):.4f} away, which "
                f"is exactly the length of the other word's position"
            ),
        )
    )

    cases.append(
        UndefinedCase(
            situation="the origin compared with itself by angle",
            what_is_undefined=(
                "a rule that answers zero only for identical inputs cannot, "
                "since neither input points anywhere"
            ),
            outcome=(
                f"{_distance_under('cosine', origin, origin):.4f}, the "
                f"convention for a direction that does not exist, and not a "
                f"measurement"
            ),
        )
    )

    try:
        fitted.similarity(HOME_WORD, "kettle")
        unknown_outcome = "an answer was returned"
    except MLLibError as refusal:
        unknown_outcome = _in_plain_words(refusal)
    cases.append(
        UndefinedCase(
            situation="a word the corpus never used",
            what_is_undefined=(
                "there is no position to compare, so the question names "
                "something that is not there"
            ),
            outcome=unknown_outcome,
        )
    )

    single = WordEmbeddings(
        Vocabulary([HOME_WORD]), table[words.index(HOME_WORD)][None, :]
    )
    cases.append(
        UndefinedCase(
            situation="the nearest word in a vocabulary of one",
            what_is_undefined=(
                "nothing is undefined, though the set the question ranges over "
                "is empty once the word itself is struck out"
            ),
            outcome=f"{single.most_similar(HOME_WORD).n_words} words came back",
        )
    )

    duplicated = np.vstack([home, home, table[words.index("mast")]])
    readings = DistanceMetric.EUCLIDEAN.between(
        _block(duplicated[0:1]), _block(duplicated)
    )[0]
    cases.append(
        UndefinedCase(
            situation="two words at exactly the same position",
            what_is_undefined=(
                "which of them is nearer, since a tie has no answer and the "
                "order has to be settled by something outside the rule"
            ),
            outcome=(
                f"both read {readings[1]:.4f} and the earlier one is named "
                f"first, which is a choice rather than a measurement"
            ),
        )
    )

    cases.append(
        UndefinedCase(
            situation="a word compared with its own position reversed",
            what_is_undefined=(
                "nothing, and it is the far end of the angle rule's range "
                "rather than an edge"
            ),
            outcome=(
                f"an angle reading of {_distance_under('cosine', home, -home):.4f}, "
                f"the largest this rule can produce"
            ),
        )
    )

    asymmetric: list[AsymmetricPair] = []
    for word in words:
        nearest = fitted.most_similar(word, 1).words[0]
        returned = fitted.most_similar(nearest, 1).words[0]
        if returned != word:
            asymmetric.append(
                AsymmetricPair(word=word, nearest=nearest, returned=returned)
            )

    forwards = cosine_similarity(home, table[words.index("mast")])
    backwards = cosine_similarity(table[words.index("mast")], home)

    return UndefinedResponse(
        cases=cases,
        asymmetric=asymmetric,
        n_words=fitted.n_words,
        n_asymmetric=len(asymmetric),
        symmetry_gap=abs(forwards - backwards),
    )


@router.get("/undefined", response_model=UndefinedResponse)
def undefined() -> UndefinedResponse:
    """The inputs on which nearness has no answer, or one only by convention."""
    return _undefined()


class TriangleResponse(BaseModel):
    """Whether a detour can be shorter than the direct route, under each rule.

    ``violations`` counts ordered triples of distinct words where going by way
    of a third word reads shorter than going directly, which cannot happen for
    a rule that keeps the triangle inequality.
    """

    metrics: list[str]
    plain_names: list[str]
    violations: list[int]
    n_triples: int
    worst_excess: list[float]
    worst_route: dict[str, list[str]]
    worst_legs: dict[str, list[float]]


@cache
def _triangle() -> TriangleResponse:
    fitted = _counting_fit()
    words = list(fitted.vocabulary)
    table = np.asarray(fitted.table)
    count = len(words)

    violations: list[int] = []
    excesses: list[float] = []
    routes: dict[str, list[str]] = {}
    legs: dict[str, list[float]] = {}
    n_triples = count * (count - 1) * (count - 2)

    # Positions are (start, waypoint, finish), so every triple of distinct
    # words is looked at once and the mask strikes out the ones that repeat a
    # word, where the comparison is trivially true.
    index = np.arange(count)
    distinct = (
        (index[:, None, None] != index[None, :, None])
        & (index[None, :, None] != index[None, None, :])
        & (index[:, None, None] != index[None, None, :])
    )

    for name in METRIC_NAMES:
        readings = DistanceMetric(name).between(_block(table), _block(table))
        detour = readings[:, :, None] + readings[None, :, :]
        slack = detour - readings[:, None, :]
        broken = (slack < -1.0e-12) & distinct
        violations.append(int(broken.sum()))
        if bool(broken.any()):
            flattened = int(np.argmin(np.where(distinct, slack, np.inf)))
            start, waypoint, finish = (
                int(value) for value in np.unravel_index(flattened, slack.shape)
            )
            excesses.append(float(-slack[start, waypoint, finish]))
            routes[name] = [words[start], words[waypoint], words[finish]]
            legs[name] = [
                float(readings[start, waypoint]),
                float(readings[waypoint, finish]),
                float(readings[start, finish]),
            ]
        else:
            excesses.append(0.0)
            routes[name] = []
            legs[name] = []

    return TriangleResponse(
        metrics=list(METRIC_NAMES),
        plain_names=[PLAIN_NAMES[name] for name in METRIC_NAMES],
        violations=violations,
        n_triples=n_triples,
        worst_excess=excesses,
        worst_route=routes,
        worst_legs=legs,
    )


@router.get("/triangle", response_model=TriangleResponse)
def triangle() -> TriangleResponse:
    """Whether a detour through a third word can read shorter than the direct route."""
    return _triangle()
