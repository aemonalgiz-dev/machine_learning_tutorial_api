"""A Vector for a Word.

Every endpoint here is registered under ``/concepts/a-vector-for-a-word``, and
the router is created and registered before any endpoint exists, so the pages of
this section can be built in parallel without many edits to one shared file.

The page is about the object every embedding technique answers with: a
vocabulary paired with one vector per word, and the three questions that object
can be asked. How near are two words, which words are nearest this one, and
which word completes an analogy.

Two spaces are served, because the page needs both. The *documents* space is a
real fit on twenty-four short texts, twelve about cooking and twelve about
sailing, copied from the count-based embedding specs so every number here is
pinned by tests in the library's own tree; its vectors are found by scoring how
much more often two words occur together in a small window than chance would
explain, and squeezing that table down to four numbers per word. The *sketch*
space is the five-word two-dimensional table the vector specs themselves use,
where every cosine can be read off a drawing and the analogy can be worked all
the way through by hand.

Nothing on the page depends on which technique produced the documents space, and
the rotation endpoint is the proof of that: turning the whole table leaves every
angle unchanged to the last bits while moving every coordinate.

Refusal messages are rewritten by ``_in_plain_words`` on the way out, because a
reader has no reason to meet a parameter name, and a test pins that no
identifier survives the rewriting.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Callable
from functools import cache

import numpy as np
from fastapi import APIRouter
from oop_ml import Feature, PrincipalComponentAnalysis
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings import (
    CooccurrenceMatrix,
    PointwiseMutualInformationEmbeddings,
    TokenisedCorpus,
    Word2Vec,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/a-vector-for-a-word", tags=["a-vector-for-a-word"])


# The count-based embedding specs' two-topic corpus, copied so the page's
# numbers are the same ones those tests pin. Document i of a topic uses that
# topic's words i .. i + 4 modulo ten, so every topic word keeps the same
# company as every other; the three function words appear one per document.
COOKING_DOCUMENTS: list[str] = [
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
]

SAILING_DOCUMENTS: list[str] = [
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
]

TWO_TOPIC_DOCUMENTS: list[str] = COOKING_DOCUMENTS + SAILING_DOCUMENTS

COOKING_WORDS: list[str] = [
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
]

SAILING_WORDS: list[str] = [
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
]

# The vector specs' own table: king and queen point one way and differ only in
# length, man and woman point at right angles to them and differ only in
# length, and the fifth word has no direction at all.
SKETCH_WORDS: list[str] = ["king", "queen", "man", "woman", "silent"]
SKETCH_TABLE: list[list[float]] = [
    [1.0, 0.0],
    [2.0, 0.0],
    [0.0, 1.0],
    [0.0, 3.0],
    [0.0, 0.0],
]

WINDOW = 2
DIMENSION = 4

DOCUMENTS_SPACE = "documents"
SKETCH_SPACE = "sketch"

# The word substituted into both topics for the two-sense probe, and the two
# words it replaces.
AMBIGUOUS_WORD = "sheet"
AMBIGUOUS_SOURCES: list[str] = ["pan", "rope"]
UNAMBIGUOUS_COMPARISON = "dough"

HOME_WORD = "sail"
"""The word the page stands at wherever one word has to be chosen."""

MISSING_WORD = "kettle"

PLAIN_WORDS = (("n_results", "the number of neighbours asked for"),)
"""Identifiers a reader has no reason to meet, and what to say instead."""

RESULT_COUNT_PATTERN = re.compile(
    r"the number of neighbours asked for must be at least 1, got (-?\d+)"
)
"""The one refusal whose whole sentence is easier to rewrite than to patch."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the one that names a parameter and leaves the rest alone. A
    test asserts that no identifier survives, so a message that grows one later
    fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return RESULT_COUNT_PATTERN.sub(
        lambda match: (
            f"a list of {match.group(1)} nearest words is not a list, so at "
            f"least one must be asked for"
        ),
        sentence,
    )


@cache
def _documents_space() -> WordEmbeddings:
    """The fit on the twenty-four documents, computed once and reused."""
    return (
        PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=DIMENSION)
        .fit(TWO_TOPIC_DOCUMENTS)
        .embeddings
    )


@cache
def _sketch_space() -> WordEmbeddings:
    """The five-word drawing the vector specs use."""
    return WordEmbeddings(
        Vocabulary(SKETCH_WORDS), np.asarray(SKETCH_TABLE, dtype=np.float64)
    )


@cache
def _merged_space() -> WordEmbeddings:
    """The same documents with one spelling standing for two different things."""
    merged = [
        document.replace(AMBIGUOUS_SOURCES[0], AMBIGUOUS_WORD)
        for document in COOKING_DOCUMENTS
    ] + [
        document.replace(AMBIGUOUS_SOURCES[1], AMBIGUOUS_WORD)
        for document in SAILING_DOCUMENTS
    ]
    return (
        PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=DIMENSION)
        .fit(merged)
        .embeddings
    )


def _space_for(name: str) -> WordEmbeddings:
    return _sketch_space() if name == SKETCH_SPACE else _documents_space()


def _counts() -> Counter[str]:
    return Counter(" ".join(TWO_TOPIC_DOCUMENTS).split())


def _vector_of(space: WordEmbeddings, word: str) -> np.ndarray:
    return np.asarray(space.vector_of(word).values, dtype=np.float64)


class WordRow(BaseModel):
    """One word, where it sits, and how often the texts used it."""

    word: str
    token_id: int
    count: int | None
    vector: list[float]
    length: float


class PairRow(BaseModel):
    """Two words, how far apart their numbers are, and how alike they are."""

    first: str
    second: str
    id_gap: int
    cosine: float
    gap: float


class SpaceResponse(BaseModel):
    """A whole table of positions, with what can be said about it as a whole.

    ``length_count_correlation`` is the correlation across the vocabulary
    between how often a word was used and how long its vector came out, and
    ``id_gap_cosine_correlation`` the correlation between the gap in the two
    words' numbers and their cosine, which is the measurement that says a
    number is only a name. ``pairs`` carries both readings of every pair, so a
    picture of either correlation needs no second request.
    """

    name: str
    documents: list[str]
    dimension: int
    n_words: int
    words: list[WordRow]
    pairs: list[PairRow]
    table_numbers: int
    n_pairs: int
    length_count_correlation: float | None
    id_gap_cosine_correlation: float | None


class NeighbourRow(BaseModel):
    """One candidate neighbour, by both readings of near."""

    word: str
    cosine: float
    gap: float
    length: float


class NeighbourRequest(BaseModel):
    """Which word to stand at, in which table, and how many to list."""

    space: str = Field(default=DOCUMENTS_SPACE, pattern="^(documents|sketch)$")
    word: str = Field(min_length=1, max_length=40)
    n_results: int = Field(default=6, ge=1, le=50)

    model_config = {"extra": "forbid"}


class NeighbourAnswer(BaseModel):
    """One word's whole row, ordered by angle and again by straight-line gap.

    ``comparisons`` is how many other words had to be scored to answer, which
    is the whole vocabulary less the word itself.
    """

    word: str
    vector: list[float]
    length: float
    by_cosine: list[NeighbourRow]
    by_gap: list[NeighbourRow]
    comparisons: int


class AnalogyTerm(BaseModel):
    """One word of the question, with the unit vector it contributes."""

    word: str
    added: bool
    vector: list[float]
    length: float
    unit: list[float]


class AnalogyAnswer(BaseModel):
    """The question worked through, and what it lands on.

    ``answers`` is the ranking after whichever exclusion was asked for, so a
    caller can ask the same question twice and see the difference.
    """

    terms: list[AnalogyTerm]
    combination: list[float]
    combination_length: float
    excluded: list[str]
    answers: list[NeighbourRow]


class AnalogyRequest(BaseModel):
    """The words added, the words subtracted, and whether they may answer."""

    space: str = Field(default=SKETCH_SPACE, pattern="^(documents|sketch)$")
    positive: list[str] = Field(min_length=1, max_length=3)
    negative: list[str] = Field(default_factory=list, max_length=3)
    exclude_question_words: bool = True
    n_results: int = Field(default=4, ge=1, le=50)

    model_config = {"extra": "forbid"}


class ExclusionExample(BaseModel):
    """One question asked both ways."""

    positive_first: str
    negative: str
    positive_second: str
    with_exclusion: str
    without_exclusion: str


class ExclusionResponse(BaseModel):
    """Every three-word question the documents allow, asked without exclusion.

    ``share`` is how often the top answer is one of the three words already in
    the question, and the two counts below split that between the words that
    were added and the word that was subtracted.
    """

    n_questions: int
    n_returning_a_question_word: int
    share: float
    n_added_word: int
    n_subtracted_word: int
    examples: list[ExclusionExample]


class RotationRequest(BaseModel):
    """How far to turn the first two coordinates, and whose row to watch."""

    angle_degrees: float = Field(ge=0.0, le=360.0)
    word: str = Field(default="sail", min_length=1, max_length=40)

    model_config = {"extra": "forbid"}


class TurnedRow(BaseModel):
    """One word before and after the turn."""

    word: str
    before: list[float]
    after: list[float]
    length_before: float
    length_after: float


class RotationResponse(BaseModel):
    """What the turn moved and what it left alone.

    ``worst_cosine_change`` is the largest change in any pair's cosine and
    ``worst_length_change`` the largest change in any word's length, both of
    which are zero up to rounding; ``largest_coordinate_move`` is how far the
    first coordinate of some word travelled. ``n_orders_unchanged`` counts the
    words whose whole ranking survives and ``n_top_unchanged`` those whose
    nearest word does; the two can differ, because a pair of words whose
    cosines are exactly equal may come back in either order.
    """

    angle_degrees: float
    rows: list[TurnedRow]
    watched: TurnedRow
    neighbours_before: list[NeighbourRow]
    neighbours_after: list[NeighbourRow]
    worst_cosine_change: float
    worst_length_change: float
    largest_coordinate_move: float
    n_orders_unchanged: int
    n_top_unchanged: int
    n_words: int


class SenseRow(BaseModel):
    """A word and the nearest thing to it, in whichever table it was fitted."""

    word: str
    nearest: str
    cosine: float


class SensesResponse(BaseModel):
    """One spelling standing for two things, against the two it replaced.

    ``mean_to_cooking`` and ``mean_to_sailing`` are the ambiguous word's mean
    cosine to each topic's words, and the comparison word is an ordinary word
    of one topic measured the same way.
    """

    merged_word: str
    replaced: list[str]
    merged_documents: list[str]
    neighbours: list[NeighbourRow]
    best_cosine: float
    separate: list[SenseRow]
    mean_to_cooking: float
    mean_to_sailing: float
    comparison_word: str
    comparison_to_cooking: float
    comparison_to_sailing: float
    lowest_best_cosine: SenseRow


class ContractProbe(BaseModel):
    """One edge of the definition, tried, with what came back.

    ``reason`` is the refusal in plain words, because a reader has no reason to
    know a parameter name; ``_in_plain_words`` does that rewriting and a test
    pins that no identifier survives it.
    """

    edge: str
    outcome: str
    result: str | None
    reason: str | None


class ContractsResponse(BaseModel):
    """Every edge the page tabulates, probed."""

    probes: list[ContractProbe]


def _word_rows(space: WordEmbeddings, counts: Counter[str] | None) -> list[WordRow]:
    table = np.asarray(space.table, dtype=np.float64)
    rows: list[WordRow] = []
    for token_id, word in enumerate(space.vocabulary):
        rows.append(
            WordRow(
                word=word,
                token_id=token_id,
                count=None if counts is None else counts[word],
                vector=[float(value) for value in table[token_id]],
                length=float(np.linalg.norm(table[token_id])),
            )
        )
    return rows


class SpaceSummary(BaseModel):
    """What is true of a whole table rather than of one word in it."""

    pairs: list[PairRow]
    length_count_correlation: float | None
    id_gap_cosine_correlation: float | None


def _summarise(space: WordEmbeddings, rows: list[WordRow]) -> SpaceSummary:
    """Length against use, and number gap against cosine, over every pair."""
    lengths = np.asarray([row.length for row in rows])
    counts = [row.count for row in rows]
    length_correlation: float | None = None
    if all(count is not None for count in counts):
        as_array = np.asarray(counts, dtype=np.float64)
        if float(as_array.std()) > 0.0 and float(lengths.std()) > 0.0:
            length_correlation = float(np.corrcoef(as_array, lengths)[0, 1])

    pairs: list[PairRow] = []
    for first, second in itertools.combinations(rows, 2):
        if first.length == 0.0 or second.length == 0.0:
            continue
        pairs.append(
            PairRow(
                first=first.word,
                second=second.word,
                id_gap=abs(first.token_id - second.token_id),
                cosine=space.similarity(first.word, second.word),
                gap=float(
                    np.linalg.norm(
                        _vector_of(space, first.word) - _vector_of(space, second.word)
                    )
                ),
            )
        )

    id_correlation: float | None = None
    id_gaps = [pair.id_gap for pair in pairs]
    cosines = [pair.cosine for pair in pairs]
    if len(pairs) > 1 and float(np.std(id_gaps)) > 0.0 and float(np.std(cosines)) > 0.0:
        id_correlation = float(np.corrcoef(id_gaps, cosines)[0, 1])
    return SpaceSummary(
        pairs=pairs,
        length_count_correlation=length_correlation,
        id_gap_cosine_correlation=id_correlation,
    )


@router.get("/space", response_model=SpaceResponse)
def documents_space() -> SpaceResponse:
    """Every word of the twenty-four documents, with where it sits."""
    space = _documents_space()
    rows = _word_rows(space, _counts())
    summary = _summarise(space, rows)
    return SpaceResponse(
        name="the cooking and sailing documents",
        documents=TWO_TOPIC_DOCUMENTS,
        dimension=space.dimension,
        n_words=space.n_words,
        words=rows,
        pairs=summary.pairs,
        table_numbers=space.n_words * space.dimension,
        n_pairs=len(summary.pairs),
        length_count_correlation=summary.length_count_correlation,
        id_gap_cosine_correlation=summary.id_gap_cosine_correlation,
    )


@router.get("/sketch", response_model=SpaceResponse)
def sketch_space() -> SpaceResponse:
    """The five-word drawing, where every cosine can be checked by hand."""
    space = _sketch_space()
    rows = _word_rows(space, None)
    summary = _summarise(space, rows)
    return SpaceResponse(
        name="the five-word sketch",
        documents=[],
        dimension=space.dimension,
        n_words=space.n_words,
        words=rows,
        pairs=summary.pairs,
        table_numbers=space.n_words * space.dimension,
        n_pairs=len(summary.pairs),
        length_count_correlation=None,
        id_gap_cosine_correlation=summary.id_gap_cosine_correlation,
    )


@router.post("/neighbours", response_model=NeighbourAnswer)
def neighbours(request: NeighbourRequest) -> NeighbourAnswer:
    """One word's row, ordered by angle and again by straight-line gap."""
    space = _space_for(request.space)
    query = _vector_of(space, request.word)
    length = float(np.linalg.norm(query))
    if length == 0.0:
        # The library says this the moment it is asked, and says it better.
        space.most_similar(request.word, n_results=request.n_results)

    rows: list[NeighbourRow] = []
    for other in space.vocabulary:
        if other == request.word:
            continue
        other_vector = _vector_of(space, other)
        other_length = float(np.linalg.norm(other_vector))
        if other_length == 0.0:
            continue
        rows.append(
            NeighbourRow(
                word=other,
                cosine=space.similarity(request.word, other),
                gap=float(np.linalg.norm(query - other_vector)),
                length=other_length,
            )
        )

    by_cosine = sorted(rows, key=lambda row: -row.cosine)[: request.n_results]
    by_gap = sorted(rows, key=lambda row: row.gap)[: request.n_results]
    return NeighbourAnswer(
        word=request.word,
        vector=[float(value) for value in query],
        length=length,
        by_cosine=by_cosine,
        by_gap=by_gap,
        comparisons=space.n_words - 1,
    )


@router.post("/analogy", response_model=AnalogyAnswer)
def analogy(request: AnalogyRequest) -> AnalogyAnswer:
    """Add and subtract unit vectors, then rank the whole table against the sum."""
    space = _space_for(request.space)

    terms: list[AnalogyTerm] = []
    combination = np.zeros(space.dimension)
    for word, added in [(word, True) for word in request.positive] + [
        (word, False) for word in request.negative
    ]:
        vector = _vector_of(space, word)
        length = float(np.linalg.norm(vector))
        if length == 0.0:
            # Let the library raise the sentence it already has for this.
            space.analogy(request.positive, request.negative, n_results=1)
        unit = vector / length
        combination = combination + unit if added else combination - unit
        terms.append(
            AnalogyTerm(
                word=word,
                added=added,
                vector=[float(value) for value in vector],
                length=length,
                unit=[float(value) for value in unit],
            )
        )

    excluding = (
        [*request.positive, *request.negative] if request.exclude_question_words else []
    )
    ranked = space.similar_to_vector(
        combination, n_results=request.n_results, excluding=excluding
    )
    combination_length = float(np.linalg.norm(combination))
    answers = [
        NeighbourRow(
            word=similar.word,
            cosine=similar.similarity,
            gap=float(np.linalg.norm(combination - _vector_of(space, similar.word))),
            length=float(np.linalg.norm(_vector_of(space, similar.word))),
        )
        for similar in ranked
    ]
    return AnalogyAnswer(
        terms=terms,
        combination=[float(value) for value in combination],
        combination_length=combination_length,
        excluded=excluding,
        answers=answers,
    )


EXCLUSION_EXAMPLES: list[tuple[str, str, str]] = [
    ("sail", "boat", "flour"),
    ("oven", "bake", "mast"),
    ("crew", "deck", "sugar"),
]
"""Three questions shown in full beside the sweep, one from each shape."""


@cache
def _exclusion_sweep() -> ExclusionResponse:
    space = _documents_space()
    table = np.asarray(space.table, dtype=np.float64)
    lengths = np.linalg.norm(table, axis=1)
    words = [word for word in space.vocabulary if lengths[space.vocabulary.id_of(word)]]
    units = {
        word: table[space.vocabulary.id_of(word)]
        / lengths[space.vocabulary.id_of(word)]
        for word in words
    }

    n_questions = 0
    added = 0
    subtracted = 0
    for first, middle, last in itertools.permutations(words, 3):
        n_questions += 1
        combination = units[first] - units[middle] + units[last]
        top = space.similar_to_vector(combination, n_results=1).words
        if not top:
            continue
        if top[0] in (first, last):
            added += 1
        elif top[0] == middle:
            subtracted += 1

    examples: list[ExclusionExample] = []
    for first, middle, last in EXCLUSION_EXAMPLES:
        combination = units[first] - units[middle] + units[last]
        without = space.similar_to_vector(combination, n_results=1)
        with_exclusion = space.similar_to_vector(
            combination, n_results=1, excluding=(first, middle, last)
        )
        examples.append(
            ExclusionExample(
                positive_first=first,
                negative=middle,
                positive_second=last,
                with_exclusion=with_exclusion.words[0],
                without_exclusion=without.words[0],
            )
        )

    hits = added + subtracted
    return ExclusionResponse(
        n_questions=n_questions,
        n_returning_a_question_word=hits,
        share=hits / n_questions,
        n_added_word=added,
        n_subtracted_word=subtracted,
        examples=examples,
    )


@router.get("/analogy-exclusion", response_model=ExclusionResponse)
def analogy_exclusion() -> ExclusionResponse:
    """Ask every three-word question without exclusion and count the echoes."""
    return _exclusion_sweep()


def _neighbour_rows(
    space: WordEmbeddings, word: str, n_results: int
) -> list[NeighbourRow]:
    query = _vector_of(space, word)
    return [
        NeighbourRow(
            word=similar.word,
            cosine=similar.similarity,
            gap=float(np.linalg.norm(query - _vector_of(space, similar.word))),
            length=float(np.linalg.norm(_vector_of(space, similar.word))),
        )
        for similar in space.most_similar(word, n_results=n_results)
    ]


@router.post("/rotation", response_model=RotationResponse)
def rotation(request: RotationRequest) -> RotationResponse:
    """Turn the first two coordinates of every word and measure what moved."""
    space = _documents_space()
    table = np.asarray(space.table, dtype=np.float64)
    radians = np.deg2rad(request.angle_degrees)
    turn = np.eye(space.dimension)
    turn[0, 0] = np.cos(radians)
    turn[0, 1] = -np.sin(radians)
    turn[1, 0] = np.sin(radians)
    turn[1, 1] = np.cos(radians)
    turned_table = table @ turn.T
    turned = WordEmbeddings(space.vocabulary, turned_table)

    rows: list[TurnedRow] = []
    worst_length = 0.0
    largest_move = 0.0
    for token_id, word in enumerate(space.vocabulary):
        before = table[token_id]
        after = turned_table[token_id]
        length_before = float(np.linalg.norm(before))
        length_after = float(np.linalg.norm(after))
        worst_length = max(worst_length, abs(length_before - length_after))
        largest_move = max(largest_move, abs(float(before[0]) - float(after[0])))
        rows.append(
            TurnedRow(
                word=word,
                before=[float(value) for value in before],
                after=[float(value) for value in after],
                length_before=length_before,
                length_after=length_after,
            )
        )

    worst_cosine = 0.0
    unchanged = 0
    tops_unchanged = 0
    for word in space.vocabulary:
        if float(np.linalg.norm(_vector_of(space, word))) == 0.0:
            continue
        before_order = space.most_similar(word, n_results=space.n_words).words
        after_order = turned.most_similar(word, n_results=space.n_words).words
        if before_order == after_order:
            unchanged += 1
        if before_order[:1] == after_order[:1]:
            tops_unchanged += 1
    for first, second in itertools.combinations(space.vocabulary, 2):
        if 0.0 in (
            float(np.linalg.norm(_vector_of(space, first))),
            float(np.linalg.norm(_vector_of(space, second))),
        ):
            continue
        worst_cosine = max(
            worst_cosine,
            abs(space.similarity(first, second) - turned.similarity(first, second)),
        )

    watched = next(row for row in rows if row.word == request.word)
    return RotationResponse(
        angle_degrees=request.angle_degrees,
        rows=rows,
        watched=watched,
        neighbours_before=_neighbour_rows(space, request.word, 4),
        neighbours_after=_neighbour_rows(turned, request.word, 4),
        worst_cosine_change=worst_cosine,
        worst_length_change=worst_length,
        largest_coordinate_move=largest_move,
        n_orders_unchanged=unchanged,
        n_top_unchanged=tops_unchanged,
        n_words=space.n_words,
    )


@router.get("/senses", response_model=SensesResponse)
def senses() -> SensesResponse:
    """One spelling used for two different things, against the two it replaced."""
    merged = _merged_space()
    plain = _documents_space()

    neighbour_rows = _neighbour_rows(merged, AMBIGUOUS_WORD, 6)
    separate = [
        SenseRow(
            word=word,
            nearest=plain.most_similar(word, n_results=1)[0].word,
            cosine=plain.most_similar(word, n_results=1)[0].similarity,
        )
        for word in AMBIGUOUS_SOURCES
    ]

    cooking = [word for word in COOKING_WORDS if word in merged]
    sailing = [word for word in SAILING_WORDS if word in merged]
    to_cooking = float(
        np.mean([merged.similarity(AMBIGUOUS_WORD, word) for word in cooking])
    )
    to_sailing = float(
        np.mean([merged.similarity(AMBIGUOUS_WORD, word) for word in sailing])
    )
    comparison_cooking = float(
        np.mean(
            [
                merged.similarity(UNAMBIGUOUS_COMPARISON, word)
                for word in cooking
                if word != UNAMBIGUOUS_COMPARISON
            ]
        )
    )
    comparison_sailing = float(
        np.mean([merged.similarity(UNAMBIGUOUS_COMPARISON, word) for word in sailing])
    )

    bests = [
        SenseRow(
            word=word,
            nearest=merged.most_similar(word, n_results=1)[0].word,
            cosine=merged.most_similar(word, n_results=1)[0].similarity,
        )
        for word in merged.vocabulary
    ]
    lowest = min(bests, key=lambda row: row.cosine)

    return SensesResponse(
        merged_word=AMBIGUOUS_WORD,
        replaced=AMBIGUOUS_SOURCES,
        merged_documents=[COOKING_DOCUMENTS[5], SAILING_DOCUMENTS[3]],
        neighbours=neighbour_rows,
        best_cosine=neighbour_rows[0].cosine,
        separate=separate,
        mean_to_cooking=to_cooking,
        mean_to_sailing=to_sailing,
        comparison_word=UNAMBIGUOUS_COMPARISON,
        comparison_to_cooking=comparison_cooking,
        comparison_to_sailing=comparison_sailing,
        lowest_best_cosine=lowest,
    )


def _as_coordinates(space: WordEmbeddings, word: str) -> str:
    """One word's position written out, never its repr, which names a class."""
    return ", ".join(f"{value:.4f}" for value in _vector_of(space, word))


def _probe(edge: str, attempt: Callable[[], str]) -> ContractProbe:
    """Run one attempt and record whether it was answered or refused."""
    try:
        result = attempt()
    except MLLibError as refusal:
        return ContractProbe(
            edge=edge, outcome="refused", result=None, reason=_in_plain_words(refusal)
        )
    return ContractProbe(edge=edge, outcome="answered", result=result, reason=None)


@router.get("/contracts", response_model=ContractsResponse)
def contracts() -> ContractsResponse:
    """Try each edge of the definition and report what came back."""
    documents = _documents_space()
    drawing = _sketch_space()

    return ContractsResponse(
        probes=[
            _probe(
                f"asking for the position of {MISSING_WORD!r}",
                lambda: _as_coordinates(documents, MISSING_WORD),
            ),
            _probe(
                "how alike a word with no direction is to another",
                lambda: f"{drawing.similarity('king', 'silent'):.4f}",
            ),
            _probe(
                "the nearest words to a word with no direction",
                lambda: ", ".join(drawing.most_similar("silent").words),
            ),
            _probe(
                "an analogy whose two words cancel exactly",
                lambda: ", ".join(drawing.analogy(["king"], ["queen"]).words),
            ),
            _probe(
                "an analogy with nothing added",
                lambda: ", ".join(drawing.analogy([], ["king"]).words),
            ),
            _probe(
                "asking for no nearest words at all",
                lambda: ", ".join(documents.most_similar("sail", n_results=0).words),
            ),
            _probe(
                "asking for more neighbours than the vocabulary holds",
                lambda: str(len(documents.most_similar("sail", n_results=50))),
            ),
            _probe(
                "a word with no direction sitting in a neighbour list",
                lambda: ", ".join(drawing.most_similar("king", n_results=10).words),
            ),
        ]
    )


TWO_FIT_DIMENSION = 8
TWO_FIT_EPOCHS = 20
TWO_FIT_LEARNING_RATE = 0.05
TWO_FIT_SEEDS = (0, 1)
TWO_FIT_NEIGHBOURS = 3
"""The second technique's settings, fixed so the two fits differ only in seed."""


class OneFit(BaseModel):
    """What one fit of the same texts came out with.

    ``within_topic`` and ``across_topic`` are the mean cosine between two words
    of one topic and between two words of different topics, which is what says
    whether the fit found anything at all.
    """

    seed: int
    within_topic: float
    across_topic: float
    watched_vector: list[float]
    watched_neighbours: list[NeighbourRow]


class TwoFitsResponse(BaseModel):
    """Two fits of one corpus, and how far they agree.

    ``self_cosine`` is the cosine between the watched word's vector in one fit
    and its vector in the other, ``mean_first_coordinate_gap`` how far apart the
    two fits put a word's first coordinate on average, and
    ``shared_neighbour_places`` how many of the two fits' nearest-word places
    hold the same word.
    """

    watched: str
    fits: list[OneFit]
    self_cosine: float
    mean_first_coordinate_gap: float
    largest_first_coordinate_gap: float
    shared_neighbour_places: int
    neighbour_places: int


@cache
def _two_fits() -> TwoFitsResponse:
    spaces = {
        seed: Word2Vec(
            dimension=TWO_FIT_DIMENSION,
            window=WINDOW,
            epochs=TWO_FIT_EPOCHS,
            learning_rate=TWO_FIT_LEARNING_RATE,
            random_seed=seed,
        )
        .fit(TWO_TOPIC_DOCUMENTS)
        .embeddings
        for seed in TWO_FIT_SEEDS
    }
    first, second = (spaces[seed] for seed in TWO_FIT_SEEDS)
    words = list(first.vocabulary)

    fits: list[OneFit] = []
    for seed in TWO_FIT_SEEDS:
        space = spaces[seed]
        within: list[float] = []
        across: list[float] = []
        for left, right in itertools.combinations(COOKING_WORDS + SAILING_WORDS, 2):
            same = (left in COOKING_WORDS) == (right in COOKING_WORDS)
            (within if same else across).append(space.similarity(left, right))
        fits.append(
            OneFit(
                seed=seed,
                within_topic=float(np.mean(within)),
                across_topic=float(np.mean(across)),
                watched_vector=[float(value) for value in _vector_of(space, HOME_WORD)],
                watched_neighbours=_neighbour_rows(
                    space, HOME_WORD, TWO_FIT_NEIGHBOURS
                ),
            )
        )

    gaps = [
        abs(float(_vector_of(first, word)[0]) - float(_vector_of(second, word)[0]))
        for word in words
    ]
    shared = sum(
        len(
            set(first.most_similar(word, n_results=TWO_FIT_NEIGHBOURS).words)
            & set(second.most_similar(word, n_results=TWO_FIT_NEIGHBOURS).words)
        )
        for word in words
    )
    return TwoFitsResponse(
        watched=HOME_WORD,
        fits=fits,
        self_cosine=cosine_similarity(
            _vector_of(first, HOME_WORD), _vector_of(second, HOME_WORD)
        ),
        mean_first_coordinate_gap=float(np.mean(gaps)),
        largest_first_coordinate_gap=float(np.max(gaps)),
        shared_neighbour_places=shared,
        neighbour_places=TWO_FIT_NEIGHBOURS * len(words),
    )


@router.get("/two-fits", response_model=TwoFitsResponse)
def two_fits() -> TwoFitsResponse:
    """Fit the same texts twice from different starts and compare the two."""
    return _two_fits()


class ContextCount(BaseModel):
    """One word that kept company with another, and how much."""

    word: str
    weight: float


class CompanyResponse(BaseModel):
    """What two words were each seen beside, and how much of it they share.

    Weights are the ones the fit itself counted with, so a word one position
    away counts one and a word two positions away counts a half. ``between`` is
    how much company the two words kept with each other directly, which can be
    zero for a pair the fit nonetheless places together.
    """

    window: int
    first: str
    second: str
    first_company: list[ContextCount]
    second_company: list[ContextCount]
    shared: list[str]
    between: float
    cosine: float


@cache
def _cooccurrence() -> CooccurrenceMatrix:
    """The same counts the documents fit was built from."""
    embedder = PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=DIMENSION)
    tokenised = TokenisedCorpus.from_texts(TWO_TOPIC_DOCUMENTS, embedder.pre_tokenizer)
    vocabulary = tokenised.vocabulary(embedder.minimum_count)
    return CooccurrenceMatrix.from_id_sequences(
        vocabulary,
        tokenised.id_sequences(vocabulary),
        embedder.window,
        embedder.weighting,
    )


class CompanyRequest(BaseModel):
    """Two words of the documents, to be compared by the company they keep."""

    first: str = Field(min_length=1, max_length=40)
    second: str = Field(min_length=1, max_length=40)

    model_config = {"extra": "forbid"}


def _company(matrix: CooccurrenceMatrix, word: str) -> list[ContextCount]:
    counts = [
        ContextCount(word=other, weight=matrix.count_between(word, other))
        for other in matrix.vocabulary
        if matrix.count_between(word, other) > 0.0
    ]
    return sorted(counts, key=lambda entry: (-entry.weight, entry.word))


@router.post("/company", response_model=CompanyResponse)
def company(request: CompanyRequest) -> CompanyResponse:
    """What each of two words was seen beside, and how much of that they share."""
    matrix = _cooccurrence()
    space = _documents_space()
    first_company = _company(matrix, request.first)
    second_company = _company(matrix, request.second)
    shared = sorted(
        {entry.word for entry in first_company}
        & {entry.word for entry in second_company}
    )
    return CompanyResponse(
        window=matrix.window,
        first=request.first,
        second=request.second,
        first_company=first_company,
        second_company=second_company,
        shared=shared,
        between=matrix.count_between(request.first, request.second),
        cosine=space.similarity(request.first, request.second),
    )


# --------------------------------------------------------------------------
# The space as a sphere.
# --------------------------------------------------------------------------

SPHERE_COORDINATE_NAMES = ("first", "second", "third", "fourth")
"""Names for the table's columns, so the projection can be fitted on Features."""


class SphereWord(BaseModel):
    """One word placed on the unit sphere, with what it is coloured by."""

    word: str
    token_id: int
    count: int
    x: float
    y: float
    z: float
    topic: str
    length: float


class SphereResponse(BaseModel):
    """Every word as a direction, projected to three dimensions.

    ``kept_share`` is how much of the spread the three drawn directions hold,
    so the picture can say what it is discarding rather than implying it drew
    everything. ``within_topic`` and ``across_topic`` are mean cosines in the
    **full** space, not in the projection, so the clustering claim is about the
    space itself and not about the drawing.
    """

    words: list[SphereWord]
    dimension: int
    n_drawn: int
    kept_share: float
    within_topic: float
    across_topic: float


def _topic_of(word: str) -> str:
    if word in COOKING_WORDS:
        return "cooking"
    if word in SAILING_WORDS:
        return "sailing"
    return "shared"


@router.get("/sphere", response_model=SphereResponse)
def sphere() -> SphereResponse:
    """Every word as a unit direction, projected onto a drawable sphere.

    Cosine reads a direction and ignores a length, so the honest picture of
    what it compares is the unit sphere: every word is put on it first, and
    only then projected, which is why two words that a cosine calls alike sit
    near one another here.

    The projection is the library's own principal components, fitted on the
    already-normalised directions, and the three coordinates are pushed back
    onto the sphere afterwards so nothing sits inside it.
    """
    space = _documents_space()
    counts = _counts()
    words = [word for word in space.vocabulary if word in counts]

    directions = np.asarray(
        [
            _vector_of(space, word) / float(np.linalg.norm(_vector_of(space, word)))
            for word in words
        ],
        dtype=np.float64,
    )

    columns = [
        Feature(name, directions[:, position])
        for position, name in enumerate(SPHERE_COORDINATE_NAMES[: space.dimension])
    ]
    analysis = PrincipalComponentAnalysis(n_components=3).fit(columns)
    placed = np.column_stack(
        [
            np.asarray(component.values, dtype=np.float64)
            for component in analysis.transform(columns)
        ]
    )
    lengths = np.linalg.norm(placed, axis=1, keepdims=True)
    placed = placed / np.where(lengths == 0.0, 1.0, lengths)

    within: list[float] = []
    across: list[float] = []
    for first_position, first in enumerate(words):
        for second in words[first_position + 1 :]:
            first_topic, second_topic = _topic_of(first), _topic_of(second)
            if "shared" in (first_topic, second_topic):
                continue
            pair = space.similarity(first, second)
            (within if first_topic == second_topic else across).append(pair)

    return SphereResponse(
        words=[
            SphereWord(
                word=word,
                token_id=space.vocabulary.id_of(word),
                count=counts[word],
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                topic=_topic_of(word),
                length=float(np.linalg.norm(_vector_of(space, word))),
            )
            for word, point in zip(words, placed, strict=True)
        ],
        dimension=space.dimension,
        n_drawn=len(words),
        kept_share=float(sum(analysis.components.variance_shares)),
        within_topic=float(np.mean(within)),
        across_topic=float(np.mean(across)),
    )
