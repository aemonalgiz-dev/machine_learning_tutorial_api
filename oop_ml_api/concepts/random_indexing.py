"""Random Indexing.

Every endpoint here is registered under ``/concepts/random-indexing``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about RandomIndexing, IndexVectors.

Two corpora, both of them the library's own spec fixtures, reproduced here
rather than imported because ``test`` is not part of the installed package; a
test asserts the two agree text for text. The three-sentence one is small
enough to print every index vector and every context vector of, and its
co-occurrence counts are whole numbers, so the accumulation can be checked with
a pencil. The twenty-four document one is the cyclic two-topic design the
count-based specs share, built so that a topic word's company is mostly its own
topic, which is what makes the separation numbers mean anything.

Every number the page quotes is computed by the library. Four things are
computed here rather than by a model, and each of them is a comparison the
method is measured against rather than part of the method. The cosine between
two *raw* co-occurrence rows, which is the answer the projection approximates
and which no embedder on this site returns. The accumulation of one word's
position one occurrence at a time, which is the definition the one-pass form
implements and is walked here so a reader can watch it. The geometry of the
drawn index vectors, which is arithmetic on a table the library drew. And the
comparison against a weighted reading of the same counts, which is a second
library model fitted to the same corpus.

Nothing a widget renders may name a class, a parameter or a module, so the two
weightings cross the wire as the page's own words, ``uniform`` and
``harmonic``, and the corpora as ``two-topic`` and ``three-sentences``. A
refusal is rewritten by ``_in_plain_words`` before it can reach a widget, and a
test asserts nothing identifier-shaped survives that rewriting.
"""

from __future__ import annotations

import re
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.counts.pointwise_mutual_information import (
    PointwiseMutualInformationEmbeddings,
)
from oop_ml.core.natural_language_processing.embeddings.counts.random_indexing import (
    IndexVectors,
    RandomIndexing,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/random-indexing", tags=["random-indexing"])


# --- The two corpora ---------------------------------------------------------

THREE_SENTENCES: tuple[str, ...] = ("the cat sat", "the dog sat", "the cat ran")
"""Five words over three sentences, small enough to print whole."""

COOKING_DOCUMENTS: tuple[str, ...] = (
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
"""Twelve documents built from ten cooking words, five consecutive at a time."""

SAILING_DOCUMENTS: tuple[str, ...] = (
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
"""Twelve more, built the same way from ten sailing words."""

TWENTY_FOUR_DOCUMENTS: list[str] = list(COOKING_DOCUMENTS) + list(SAILING_DOCUMENTS)
"""The corpus every fixed measurement on this page is made on."""

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
SHARED_WORDS: tuple[str, ...] = ("and", "the", "we")
"""The three words both halves use, which are the only overlap there is."""

TOPIC_WORDS: tuple[str, ...] = COOKING_WORDS + SAILING_WORDS

CorpusName = Literal["two-topic", "three-sentences"]
WeightingName = Literal["uniform", "harmonic"]

_CORPORA: dict[CorpusName, list[str]] = {
    "two-topic": TWENTY_FOUR_DOCUMENTS,
    "three-sentences": list(THREE_SENTENCES),
}

_WEIGHTINGS: dict[WeightingName, ContextWeighting] = {
    "uniform": ContextWeighting.UNIFORM,
    "harmonic": ContextWeighting.HARMONIC,
}

_GROUP_OF: dict[CorpusName, dict[str, str]] = {
    "two-topic": {
        **{word: "cooking" for word in COOKING_WORDS},
        **{word: "sailing" for word in SAILING_WORDS},
        **{word: "shared" for word in SHARED_WORDS},
    },
    "three-sentences": {
        "the": "shared",
        "cat": "cat",
        "sat": "cat",
        "ran": "cat",
        "dog": "dog",
    },
}
"""Which half of a corpus each word belongs to, for the widgets that colour by it."""


# --- What a request may ask for ----------------------------------------------

PAGE_DIMENSION = 16
"""How wide the page's headline fit is.

Sixteen, because it is narrower than the twenty-three words the corpus has, so
the fit really is a squeeze rather than a relabelling, and because sixteen
squares of a strip fit on a screen next to a word.
"""

PAGE_NONZEROS = 4
"""How many non-zero entries a drawn direction carries by default."""

PAGE_WINDOW = 5
"""How many positions on each side of a word count as its company."""

MIN_DIMENSION = 2
MAX_DIMENSION = 2048
"""The widest fit a request may ask for.

Two thousand and forty-eight, because that is where a draw first stops any two
of these twenty-three words sharing a position at all, which is a measurement
the page makes and would otherwise be out of reach.
"""

MAX_NONZEROS = 32
MAX_WINDOW = 8
MAX_SEED = 999
MAX_RESULTS = 12

PERPENDICULARITY_WIDTHS: tuple[int, ...] = (8, 16, 32, 50, 100, 200, 500, 1000)
"""The widths the drawn directions are measured at."""

PERPENDICULARITY_NONZEROS: tuple[int, ...] = (2, 4, 8, 16, 50)
"""How full a direction is made, at one fixed width."""

PERPENDICULARITY_VECTORS = 200
"""How many directions are drawn for the geometry measurement.

Two hundred, which is the number the library's own spec draws, so the two
measurements are of the same thing.
"""

COST_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 529, 1024, 2048)
"""The widths the projection is measured against the whole table at."""

DRAW_WIDTHS: tuple[int, ...] = (8, 16, 50, 200)
"""The widths the spread over draws is measured at."""

N_DRAWS = 20
"""How many draws the spread is measured over."""

REPEAT_COUNTS: tuple[int, ...] = (1, 2, 4, 8)
"""How many copies of the corpus the invariance is checked on."""


# --- Shared pieces -----------------------------------------------------------


@cache
def _tokenised(corpus: CorpusName) -> TokenisedCorpus:
    """One corpus split into words, once."""
    return TokenisedCorpus.from_texts(_CORPORA[corpus], WhitespacePreTokenizer())


@cache
def _fitted(
    corpus: CorpusName,
    dimension: int,
    n_nonzero: int,
    window: int,
    weighting: WeightingName,
    random_seed: int,
) -> RandomIndexing:
    """One fit, kept, since several widgets ask for the same one."""
    return RandomIndexing(
        dimension=dimension,
        n_nonzero=n_nonzero,
        window=window,
        weighting=_WEIGHTINGS[weighting],
        random_seed=random_seed,
    ).fit(_CORPORA[corpus])


@cache
def _words(corpus: CorpusName) -> tuple[str, ...]:
    """The vocabulary of one corpus, commonest first."""
    return tuple(_tokenised(corpus).vocabulary(1))


def _counts(model: RandomIndexing) -> FloatArray:
    """The table of how often each word appeared near each other word."""
    return np.asarray(model.cooccurrence.counts)


def _unit_rows(table: FloatArray) -> FloatArray:
    """Every row scaled to length one, leaving a row of zeros alone."""
    lengths = np.linalg.norm(table, axis=1)
    lengths[lengths == 0.0] = 1.0
    return table / lengths[:, None]


def _pairwise_cosines(table: FloatArray) -> FloatArray:
    """The cosine between every pair of rows."""
    unit = _unit_rows(table)
    return unit @ unit.T


def _mean_over_pairs(
    cosines: FloatArray,
    words: tuple[str, ...],
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> float:
    """The mean cosine over every pair with one word from each group."""
    positions = {word: place for place, word in enumerate(words)}
    values = [
        cosines[positions[one], positions[other]]
        for one in first
        for other in second
        if one != other
    ]
    return float(np.mean(values)) if values else 0.0


def _within_and_across(
    cosines: FloatArray, words: tuple[str, ...]
) -> tuple[float, float]:
    """Mean cosine inside a half, and mean cosine between the two halves.

    Only the topic words count, since the three words both halves use belong to
    neither and comparing them to anything says nothing about whether the halves
    came apart.
    """
    positions = {word: place for place, word in enumerate(words)}
    inside = [
        cosines[positions[one], positions[other]]
        for group in (COOKING_WORDS, SAILING_WORDS)
        for place, one in enumerate(group)
        for other in group[place + 1 :]
    ]
    between = [
        cosines[positions[one], positions[other]]
        for one in COOKING_WORDS
        for other in SAILING_WORDS
    ]
    return float(np.mean(inside)), float(np.mean(between))


def _topic_pairs(words: tuple[str, ...]) -> list[tuple[int, int]]:
    """Every pair of two topic words, as positions in the vocabulary."""
    positions = [words.index(word) for word in TOPIC_WORDS if word in words]
    return [
        (one, other)
        for place, one in enumerate(positions)
        for other in positions[place + 1 :]
    ]


def _gap_against_the_table(
    projected: FloatArray, full: FloatArray, pairs: list[tuple[int, int]]
) -> tuple[float, float]:
    """Mean and largest change in the cosine of a pair of topic words."""
    differences = [
        abs(projected[one, other] - full[one, other]) for one, other in pairs
    ]
    return float(np.mean(differences)), float(np.max(differences))


def _summed_one_occurrence_at_a_time(
    corpus: CorpusName,
    index_vectors: IndexVectors,
    window: int,
    weighting: WeightingName,
) -> FloatArray:
    """The definition: walk every sentence and add each neighbour's direction.

    This is the one-pass form the matrix product is checked against, and it is
    written here from the definition rather than taken from anything, since an
    oracle that shares an implementation with what it checks asserts nothing.
    """
    table = np.zeros((index_vectors.n_words, index_vectors.dimension))
    harmonic = _WEIGHTINGS[weighting] is ContextWeighting.HARMONIC
    for sentence in _tokenised(corpus).sentences:
        for position, word in enumerate(sentence):
            for other_position, neighbour in enumerate(sentence):
                distance = abs(other_position - position)
                if distance == 0 or distance > window:
                    continue
                weight = 1.0 / distance if harmonic else 1.0
                table[index_vectors.vocabulary.id_of(word)] += weight * np.asarray(
                    index_vectors.vector_of(neighbour).values
                )
    return table


NAMED_IDENTIFIER = re.compile(r"\b(?:n_nonzero|dimension|minimum_count|window)=")
"""The parameter names the refusals reachable from this page can carry."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with any identifier taken out of it.

    The messages are written to be read, and the ones reachable from this page
    name the setting they refused by its parameter name. A test asserts nothing
    identifier-shaped survives, so a message that grows one later fails there
    rather than reaching a reader.
    """
    return NAMED_IDENTIFIER.sub("", str(refusal)).replace("  ", " ").strip()


def _checked_settings(dimension: int, n_nonzero: int) -> None:
    """Refuse a draw that cannot exist, in the page's own words."""
    if n_nonzero % 2 != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a direction holds as many entries of +1 as of −1, so it cannot "
                f"hold {n_nonzero} of them in all"
            ),
        )
    if n_nonzero > dimension:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{n_nonzero} non-zero entries do not fit in a direction "
                f"{dimension} numbers long"
            ),
        )


def _checked_word(corpus: CorpusName, word: str) -> str:
    """Refuse a word the corpus never used, naming what it does hold."""
    words = _words(corpus)
    if word not in words:
        raise HTTPException(
            status_code=422,
            detail=(
                f"“{word}” is not one of the {len(words)} words this collection "
                f"used, so nothing was ever counted beside it and it has no position"
            ),
        )
    return word


# --- One fit -----------------------------------------------------------------


class DrawnDirection(BaseModel):
    """One word's fixed random direction, as its entries."""

    word: str
    group: str
    entries: list[int]
    positions: list[int]


class WordPosition(BaseModel):
    """One word's position, summed from the directions of its company."""

    word: str
    group: str
    coordinates: list[float]
    length: float
    occurrences: float


class Neighbour(BaseModel):
    """One word near another, and how near, in both readings."""

    word: str
    group: str
    similarity: float
    table_similarity: float
    table_rank: int


class FitRequest(BaseModel):
    """One fit, and the word to ask about."""

    corpus: CorpusName = "two-topic"
    dimension: int = Field(default=PAGE_DIMENSION, ge=MIN_DIMENSION, le=MAX_DIMENSION)
    n_nonzero: int = Field(default=PAGE_NONZEROS, ge=2, le=MAX_NONZEROS)
    window: int = Field(default=PAGE_WINDOW, ge=1, le=MAX_WINDOW)
    weighting: WeightingName = "uniform"
    random_seed: int = Field(default=0, ge=0, le=MAX_SEED)
    word: str = Field(default="flour", min_length=1, max_length=32)
    n_results: int = Field(default=8, ge=1, le=MAX_RESULTS)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything one fit has to say, and what the whole table says beside it.

    ``identity_gap`` is the largest disagreement between the positions read off
    one accumulating pass and the same positions read off the table multiplied
    by the directions, which is the whole argument that this is a projection and
    not a rule of thumb. ``table_gap_mean`` is how far the projected cosines
    moved from the ones the whole table gives, averaged over every pair of two
    topic words.
    """

    corpus: CorpusName
    dimension: int
    n_nonzero: int
    window: int
    weighting: WeightingName
    random_seed: int
    words: list[str]
    n_words: int
    directions: list[DrawnDirection]
    positions: list[WordPosition]
    word: str
    nearest: list[Neighbour]
    table_nearest: list[Neighbour]
    against_every_word: list[Neighbour]
    within_similarity: float
    across_similarity: float
    table_within_similarity: float
    table_across_similarity: float
    shared_to_topic: float
    table_shared_to_topic: float
    identity_gap: float
    largest_coordinate: float
    table_gap_mean: float
    table_gap_largest: float
    direction_mean_absolute_cosine: float
    direction_predicted_cosine: float
    direction_exactly_perpendicular: float
    colliding_pairs: int
    n_direction_pairs: int
    table_cells: int
    position_cells: int
    occurrences: int
    neighbour_visits: int
    slot_updates: int
    product_multiplications: int
    seconds: float


def _neighbours(
    words: tuple[str, ...],
    corpus: CorpusName,
    word: str,
    projected: FloatArray,
    full: FloatArray,
    n_results: int | None,
    ordered_by: Literal["projection", "table", "vocabulary"],
) -> list[Neighbour]:
    """The words near ``word`` under one ordering, carrying both numbers."""
    positions = {other: place for place, other in enumerate(words)}
    here = positions[word]
    table_order = list(np.argsort(-full[here]))
    table_rank = {
        int(place): rank
        for rank, place in enumerate(
            [candidate for candidate in table_order if int(candidate) != here], start=1
        )
    }
    if ordered_by == "vocabulary":
        order = [place for place in range(len(words)) if place != here]
    else:
        ranked = full[here] if ordered_by == "table" else projected[here]
        order = [int(place) for place in np.argsort(-ranked) if int(place) != here]
    order = order[:n_results] if n_results is not None else order
    return [
        Neighbour(
            word=words[place],
            group=_GROUP_OF[corpus].get(words[place], "shared"),
            similarity=float(projected[here, place]),
            table_similarity=float(full[here, place]),
            table_rank=table_rank[place],
        )
        for place in order
    ]


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Draw a direction for every word, add them up, and read the positions."""
    _checked_settings(request.dimension, request.n_nonzero)
    word = _checked_word(request.corpus, request.word)

    started = time.perf_counter()
    try:
        model = _fitted(
            request.corpus,
            request.dimension,
            request.n_nonzero,
            request.window,
            request.weighting,
            request.random_seed,
        )
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal
    seconds = time.perf_counter() - started

    words = _words(request.corpus)
    counts = _counts(model)
    table = np.asarray(model.embeddings.table)
    directions = np.asarray(model.index_vectors.table)
    projected = _pairwise_cosines(table)
    full = _pairwise_cosines(counts)

    unit_directions = _unit_rows(directions)
    direction_cosines = (unit_directions @ unit_directions.T)[
        np.triu_indices(len(words), k=1)
    ]

    one_pass = _summed_one_occurrence_at_a_time(
        request.corpus, model.index_vectors, request.window, request.weighting
    )

    sentences = _tokenised(request.corpus).sentences
    occurrences = sum(len(sentence) for sentence in sentences)
    neighbour_visits = sum(
        1
        for sentence in sentences
        for position in range(len(sentence))
        for other in range(len(sentence))
        if other != position and abs(other - position) <= request.window
    )

    topic_here = tuple(one for one in TOPIC_WORDS if one in words)
    shared_here = tuple(one for one in SHARED_WORDS if one in words)
    if topic_here:
        within, across = _within_and_across(projected, words)
        table_within, table_across = _within_and_across(full, words)
        shared_to_topic = _mean_over_pairs(projected, words, shared_here, topic_here)
        table_shared_to_topic = _mean_over_pairs(full, words, shared_here, topic_here)
        pairs = _topic_pairs(words)
        gap_mean, gap_largest = _gap_against_the_table(projected, full, pairs)
    else:
        within = across = table_within = table_across = 0.0
        shared_to_topic = table_shared_to_topic = 0.0
        gap_mean = gap_largest = 0.0

    return FitResponse(
        corpus=request.corpus,
        dimension=request.dimension,
        n_nonzero=request.n_nonzero,
        window=request.window,
        weighting=request.weighting,
        random_seed=request.random_seed,
        words=list(words),
        n_words=len(words),
        directions=[
            DrawnDirection(
                word=other,
                group=_GROUP_OF[request.corpus].get(other, "shared"),
                entries=[int(value) for value in directions[place]],
                positions=[int(slot) for slot in np.flatnonzero(directions[place])],
            )
            for place, other in enumerate(words)
        ],
        positions=[
            WordPosition(
                word=other,
                group=_GROUP_OF[request.corpus].get(other, "shared"),
                coordinates=[float(value) for value in table[place]],
                length=float(np.linalg.norm(table[place])),
                occurrences=float(counts[place].sum()),
            )
            for place, other in enumerate(words)
        ],
        word=word,
        nearest=_neighbours(
            words,
            request.corpus,
            word,
            projected,
            full,
            request.n_results,
            "projection",
        ),
        table_nearest=_neighbours(
            words, request.corpus, word, projected, full, request.n_results, "table"
        ),
        against_every_word=_neighbours(
            words, request.corpus, word, projected, full, None, "vocabulary"
        ),
        within_similarity=within,
        across_similarity=across,
        table_within_similarity=table_within,
        table_across_similarity=table_across,
        shared_to_topic=shared_to_topic,
        table_shared_to_topic=table_shared_to_topic,
        identity_gap=float(np.abs(one_pass - table).max()),
        largest_coordinate=float(np.abs(table).max()),
        table_gap_mean=gap_mean,
        table_gap_largest=gap_largest,
        direction_mean_absolute_cosine=float(np.abs(direction_cosines).mean()),
        direction_predicted_cosine=request.n_nonzero / request.dimension,
        direction_exactly_perpendicular=float((direction_cosines == 0.0).mean()),
        colliding_pairs=int((direction_cosines != 0.0).sum()),
        n_direction_pairs=int(direction_cosines.size),
        table_cells=int(counts.size),
        position_cells=int(table.size),
        occurrences=occurrences,
        neighbour_visits=neighbour_visits,
        slot_updates=neighbour_visits * request.n_nonzero,
        product_multiplications=int(counts.size * request.dimension),
        seconds=seconds,
    )


# --- One word's position, built up ------------------------------------------


class AccumulationStep(BaseModel):
    """One neighbour added, and where the running total stood afterwards."""

    step: int
    sentence: int
    sentence_text: str
    position: int
    neighbour: str
    neighbour_group: str
    distance: int
    weight: float
    added: list[float]
    running: list[float]


class AccumulationRequest(BaseModel):
    """Which word to follow through the corpus, and under which draw."""

    corpus: CorpusName = "three-sentences"
    word: str = Field(default="cat", min_length=1, max_length=32)
    dimension: int = Field(default=8, ge=MIN_DIMENSION, le=128)
    n_nonzero: int = Field(default=PAGE_NONZEROS, ge=2, le=MAX_NONZEROS)
    window: int = Field(default=2, ge=1, le=MAX_WINDOW)
    weighting: WeightingName = "uniform"
    random_seed: int = Field(default=0, ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class AccumulationResponse(BaseModel):
    """Every addition that built one word's position, in corpus order.

    ``final`` is what the fit reports for the word, and it is the running total
    after the last step, which is the claim the steps are here to let a reader
    check.
    """

    corpus: CorpusName
    word: str
    group: str
    dimension: int
    n_nonzero: int
    window: int
    weighting: WeightingName
    random_seed: int
    sentences: list[str]
    words: list[str]
    directions: list[DrawnDirection]
    counts: list[list[float]]
    steps: list[AccumulationStep]
    final: list[float]
    n_occurrences: int
    row_total: float
    matches_the_fit: bool


@router.post("/worked", response_model=AccumulationResponse)
def worked(request: AccumulationRequest) -> AccumulationResponse:
    """Follow one word through the corpus, adding a direction at a time."""
    _checked_settings(request.dimension, request.n_nonzero)
    word = _checked_word(request.corpus, request.word)

    try:
        model = _fitted(
            request.corpus,
            request.dimension,
            request.n_nonzero,
            request.window,
            request.weighting,
            request.random_seed,
        )
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    words = _words(request.corpus)
    directions = np.asarray(model.index_vectors.table)
    harmonic = _WEIGHTINGS[request.weighting] is ContextWeighting.HARMONIC
    sentences = _tokenised(request.corpus).sentences

    running = np.zeros(request.dimension)
    steps: list[AccumulationStep] = []
    occurrences = 0
    for place, sentence in enumerate(sentences):
        for position, here in enumerate(sentence):
            if here != word:
                continue
            occurrences += 1
            for other_position, neighbour in enumerate(sentence):
                distance = abs(other_position - position)
                if distance == 0 or distance > request.window:
                    continue
                weight = 1.0 / distance if harmonic else 1.0
                added = weight * directions[words.index(neighbour)]
                running = running + added
                steps.append(
                    AccumulationStep(
                        step=len(steps) + 1,
                        sentence=place,
                        sentence_text=_CORPORA[request.corpus][place],
                        position=position,
                        neighbour=neighbour,
                        neighbour_group=_GROUP_OF[request.corpus].get(
                            neighbour, "shared"
                        ),
                        distance=distance,
                        weight=weight,
                        added=[float(value) for value in added],
                        running=[float(value) for value in running],
                    )
                )

    fitted_row = np.asarray(model.vector_of(word).values)
    counts = _counts(model)

    return AccumulationResponse(
        corpus=request.corpus,
        word=word,
        group=_GROUP_OF[request.corpus].get(word, "shared"),
        dimension=request.dimension,
        n_nonzero=request.n_nonzero,
        window=request.window,
        weighting=request.weighting,
        random_seed=request.random_seed,
        sentences=list(_CORPORA[request.corpus]),
        words=list(words),
        directions=[
            DrawnDirection(
                word=other,
                group=_GROUP_OF[request.corpus].get(other, "shared"),
                entries=[int(value) for value in directions[place]],
                positions=[int(slot) for slot in np.flatnonzero(directions[place])],
            )
            for place, other in enumerate(words)
        ],
        counts=[[float(value) for value in row] for row in counts],
        steps=steps,
        final=[float(value) for value in fitted_row],
        n_occurrences=occurrences,
        row_total=float(counts[words.index(word)].sum()),
        matches_the_fit=bool(np.allclose(running, fitted_row)),
    )


# --- How nearly perpendicular the drawn directions are -----------------------


class WidthReading(BaseModel):
    """The geometry of a fixed number of drawn directions, at one width."""

    dimension: int
    n_nonzero: int
    predicted: float
    measured: float
    ratio: float
    exactly_perpendicular: float
    largest_absolute: float


class PerpendicularityResponse(BaseModel):
    """How nearly perpendicular random directions are, measured two ways.

    ``widths`` holds one reading per width at four non-zeros, and ``fullness``
    holds one reading per number of non-zeros at a single width, which is where
    the simple rule stops predicting anything at all.
    """

    n_vectors: int
    n_pairs: int
    widths: list[WidthReading]
    fullness: list[WidthReading]
    fullness_dimension: int


def _geometry(dimension: int, n_nonzero: int) -> WidthReading:
    """Draw the directions and measure how far from perpendicular they came."""
    vocabulary = Vocabulary(
        [f"word{place}" for place in range(PERPENDICULARITY_VECTORS)]
    )
    drawn = IndexVectors.drawn(
        vocabulary, dimension, n_nonzero, np.random.default_rng(0)
    )
    table = np.asarray(drawn.table)
    unit = table / np.linalg.norm(table, axis=1)[:, None]
    cosines = (unit @ unit.T)[np.triu_indices(PERPENDICULARITY_VECTORS, k=1)]
    measured = float(np.abs(cosines).mean())
    predicted = n_nonzero / dimension
    return WidthReading(
        dimension=dimension,
        n_nonzero=n_nonzero,
        predicted=predicted,
        measured=measured,
        ratio=measured / predicted,
        exactly_perpendicular=float((cosines == 0.0).mean()),
        largest_absolute=float(np.abs(cosines).max()),
    )


@cache
def _perpendicularity() -> PerpendicularityResponse:
    """The whole geometry measurement, computed once."""
    return PerpendicularityResponse(
        n_vectors=PERPENDICULARITY_VECTORS,
        n_pairs=PERPENDICULARITY_VECTORS * (PERPENDICULARITY_VECTORS - 1) // 2,
        widths=[
            _geometry(dimension, PAGE_NONZEROS) for dimension in PERPENDICULARITY_WIDTHS
        ],
        fullness=[_geometry(50, n_nonzero) for n_nonzero in PERPENDICULARITY_NONZEROS],
        fullness_dimension=50,
    )


@router.get("/perpendicularity", response_model=PerpendicularityResponse)
def perpendicularity() -> PerpendicularityResponse:
    """How close to perpendicular two hundred drawn directions actually come."""
    return _perpendicularity()


# --- What the shortcut costs -------------------------------------------------


class CostRow(BaseModel):
    """One width, and how far its answer sits from the whole table's."""

    dimension: int
    gap_mean: float
    gap_largest: float
    within_similarity: float
    across_similarity: float
    separation: float
    colliding_pairs: int
    shared_neighbours: int
    position_cells: int


class DrawRow(BaseModel):
    """How much one width's answer moves when only the draw changes."""

    dimension: int
    n_draws: int
    smallest_pair: float
    largest_pair: float
    pair_spread: float
    smallest_separation: float
    largest_separation: float
    mean_gap: float


class RepeatRow(BaseModel):
    """The same corpus, written out several times over."""

    repeats: int
    occurrences: int
    gap_mean: float
    pair_similarity: float


class SparsityRow(BaseModel):
    """How full a direction is, against how much it buys.

    ``mean_gap`` is averaged over several draws and ``one_draw_gap`` is the
    single draw the rest of the page uses, which put the settings in an order
    the average does not.
    """

    dimension: int
    n_nonzero: int
    n_draws: int
    mean_gap: float
    one_draw_gap: float


class NeverMet(BaseModel):
    """A pair of words the collection never once put beside each other."""

    first: str
    second: str
    count: float
    table_similarity: float
    projected_similarity: float


class CostsResponse(BaseModel):
    """What is lost by never building the table, measured three ways.

    ``widths`` is the trade the width buys, ``draws`` is how much of the answer
    is the draw rather than the corpus, and ``repeats`` is the one that says the
    error is not a sampling error, since writing the collection out eight times
    leaves every number where it was. ``sparsity`` is the one that says how full
    a direction is buys almost nothing, which is what makes choosing the cheap
    option free rather than a compromise.
    """

    n_never_met: int
    n_topic_pairs: int
    never_met: NeverMet
    sparsity: list[SparsityRow]
    sparsity_draws: int
    reference_word: str
    table_neighbours: list[str]
    table_within_similarity: float
    table_across_similarity: float
    table_separation: float
    table_cells: int
    n_words: int
    n_top: int
    widths: list[CostRow]
    draws: list[DrawRow]
    repeats: list[RepeatRow]
    exact_dimension: int
    exact_seed: int
    exact_gap: float
    exact_colliding: int
    inexact_seed: int
    inexact_gap: float
    inexact_colliding: int


COST_WORD = "flour"
COST_TOP = 6

SPARSITY_WIDTHS: tuple[int, ...] = (16, 64, 256)
"""The widths the fullness of a direction is measured against."""

SPARSITY_NONZEROS: tuple[int, ...] = (2, 4, 8, 16, 32)
"""How many non-zero entries a direction is given, at each of those widths."""

SPARSITY_DRAWS = 12
"""How many draws each of those readings is averaged over.

Averaged, because a single draw puts the options in an order the next draw
reverses, which is the finding rather than a nuisance.
"""


@cache
def _costs() -> CostsResponse:
    """The whole cost measurement, computed once."""
    corpus: CorpusName = "two-topic"
    words = _words(corpus)
    reference = _fitted(
        corpus, PAGE_DIMENSION, PAGE_NONZEROS, PAGE_WINDOW, "uniform", 0
    )
    counts = _counts(reference)
    full = _pairwise_cosines(counts)
    pairs = _topic_pairs(words)
    here = words.index(COST_WORD)
    table_neighbours = [
        words[int(place)] for place in np.argsort(-full[here]) if int(place) != here
    ][:COST_TOP]
    table_within, table_across = _within_and_across(full, words)

    rows: list[CostRow] = []
    for dimension in COST_WIDTHS:
        model = _fitted(corpus, dimension, PAGE_NONZEROS, PAGE_WINDOW, "uniform", 0)
        table = np.asarray(model.embeddings.table)
        projected = _pairwise_cosines(table)
        gap_mean, gap_largest = _gap_against_the_table(projected, full, pairs)
        within, across = _within_and_across(projected, words)
        directions = _unit_rows(np.asarray(model.index_vectors.table))
        direction_cosines = (directions @ directions.T)[
            np.triu_indices(len(words), k=1)
        ]
        nearest = [
            words[int(place)]
            for place in np.argsort(-projected[here])
            if int(place) != here
        ][:COST_TOP]
        rows.append(
            CostRow(
                dimension=dimension,
                gap_mean=gap_mean,
                gap_largest=gap_largest,
                within_similarity=within,
                across_similarity=across,
                separation=within - across,
                colliding_pairs=int((direction_cosines != 0.0).sum()),
                shared_neighbours=len(set(nearest) & set(table_neighbours)),
                position_cells=int(table.size),
            )
        )

    draws: list[DrawRow] = []
    for dimension in DRAW_WIDTHS:
        pair_values: list[float] = []
        separations: list[float] = []
        gaps: list[float] = []
        for seed in range(N_DRAWS):
            model = _fitted(
                corpus, dimension, PAGE_NONZEROS, PAGE_WINDOW, "uniform", seed
            )
            table = np.asarray(model.embeddings.table)
            projected = _pairwise_cosines(table)
            within, across = _within_and_across(projected, words)
            pair_values.append(float(projected[here, words.index("sugar")]))
            separations.append(within - across)
            gaps.append(_gap_against_the_table(projected, full, pairs)[0])
        draws.append(
            DrawRow(
                dimension=dimension,
                n_draws=N_DRAWS,
                smallest_pair=min(pair_values),
                largest_pair=max(pair_values),
                pair_spread=max(pair_values) - min(pair_values),
                smallest_separation=min(separations),
                largest_separation=max(separations),
                mean_gap=float(np.mean(gaps)),
            )
        )

    repeats: list[RepeatRow] = []
    for count in REPEAT_COUNTS:
        model = RandomIndexing(
            dimension=PAGE_DIMENSION,
            n_nonzero=PAGE_NONZEROS,
            window=PAGE_WINDOW,
            random_seed=0,
        ).fit(TWENTY_FOUR_DOCUMENTS * count)
        projected = _pairwise_cosines(np.asarray(model.embeddings.table))
        repeats.append(
            RepeatRow(
                repeats=count,
                occurrences=144 * count,
                gap_mean=_gap_against_the_table(projected, full, pairs)[0],
                pair_similarity=float(projected[here, words.index("sugar")]),
            )
        )

    sparsity: list[SparsityRow] = []
    for dimension in SPARSITY_WIDTHS:
        for n_nonzero in SPARSITY_NONZEROS:
            if n_nonzero > dimension:
                continue
            measured = [
                _gap_against_the_table(
                    _pairwise_cosines(
                        np.asarray(
                            _fitted(
                                corpus,
                                dimension,
                                n_nonzero,
                                PAGE_WINDOW,
                                "uniform",
                                seed,
                            ).embeddings.table
                        )
                    ),
                    full,
                    pairs,
                )[0]
                for seed in range(SPARSITY_DRAWS)
            ]
            sparsity.append(
                SparsityRow(
                    dimension=dimension,
                    n_nonzero=n_nonzero,
                    n_draws=SPARSITY_DRAWS,
                    mean_gap=float(np.mean(measured)),
                    one_draw_gap=measured[0],
                )
            )

    projected_here = _pairwise_cosines(np.asarray(reference.embeddings.table))
    strangers = [
        (one, other)
        for place, one in enumerate(TOPIC_WORDS)
        for other in TOPIC_WORDS[place + 1 :]
        if reference.cooccurrence.count_between(one, other) == 0.0
    ]
    n_never_met = len(strangers)
    best = max(
        [pair for pair in strangers if COST_WORD in pair],
        key=lambda pair: full[words.index(pair[0]), words.index(pair[1])],
    )
    never_met = NeverMet(
        first=best[0],
        second=best[1],
        count=reference.cooccurrence.count_between(best[0], best[1]),
        table_similarity=float(full[words.index(best[0]), words.index(best[1])]),
        projected_similarity=float(
            projected_here[words.index(best[0]), words.index(best[1])]
        ),
    )

    exact = _fitted(corpus, MAX_DIMENSION, PAGE_NONZEROS, PAGE_WINDOW, "uniform", 1)
    inexact = _fitted(corpus, MAX_DIMENSION, PAGE_NONZEROS, PAGE_WINDOW, "uniform", 0)

    def collisions(model: RandomIndexing) -> int:
        directions = _unit_rows(np.asarray(model.index_vectors.table))
        cosines = (directions @ directions.T)[np.triu_indices(len(words), k=1)]
        return int((cosines != 0.0).sum())

    return CostsResponse(
        n_never_met=n_never_met,
        n_topic_pairs=len(pairs),
        never_met=never_met,
        sparsity=sparsity,
        sparsity_draws=SPARSITY_DRAWS,
        reference_word=COST_WORD,
        table_neighbours=table_neighbours,
        table_within_similarity=table_within,
        table_across_similarity=table_across,
        table_separation=table_within - table_across,
        table_cells=int(counts.size),
        n_words=len(words),
        n_top=COST_TOP,
        widths=rows,
        draws=draws,
        repeats=repeats,
        exact_dimension=MAX_DIMENSION,
        exact_seed=1,
        exact_gap=_gap_against_the_table(
            _pairwise_cosines(np.asarray(exact.embeddings.table)), full, pairs
        )[0],
        exact_colliding=collisions(exact),
        inexact_seed=0,
        inexact_gap=_gap_against_the_table(
            _pairwise_cosines(np.asarray(inexact.embeddings.table)), full, pairs
        )[0],
        inexact_colliding=collisions(inexact),
    )


@router.get("/costs", response_model=CostsResponse)
def costs() -> CostsResponse:
    """What the projection costs, against the table it never builds."""
    return _costs()


# --- Against a reading that weights the counts -------------------------------


class ReadingRow(BaseModel):
    """One word's nearest neighbour under each of the three readings."""

    word: str
    group: str
    by_the_table: str
    by_the_projection: str
    by_the_weighted_table: str


class AgainstTheTableResponse(BaseModel):
    """Where adding up raw counts is beaten by reading them properly.

    Every number here is over the same twenty-four documents. The three words
    both halves use have the largest row totals of anything in the collection,
    so a reading that leaves the counts alone carries that loudness into every
    position, and the projection inherits it exactly.
    """

    words: list[str]
    row_totals: list[float]
    shared_words: list[str]
    shared_row_total: float
    largest_topic_row_total: float
    shared_share_of_the_table: float
    table_shared_to_topic: float
    projected_shared_to_topic: float
    weighted_shared_to_topic: float
    table_within: float
    table_across: float
    projected_within: float
    projected_across: float
    weighted_within: float
    weighted_across: float
    shared_pairs: list[str]
    shared_pair_projected: list[float]
    shared_pair_table: list[float]
    rows: list[ReadingRow]


@cache
def _against_the_table() -> AgainstTheTableResponse:
    """The three readings of one collection, computed once."""
    corpus: CorpusName = "two-topic"
    words = _words(corpus)
    model = _fitted(corpus, PAGE_DIMENSION, PAGE_NONZEROS, PAGE_WINDOW, "uniform", 0)
    counts = _counts(model)
    full = _pairwise_cosines(counts)
    projected = _pairwise_cosines(np.asarray(model.embeddings.table))
    weighted = PointwiseMutualInformationEmbeddings(
        dimension=PAGE_DIMENSION, window=PAGE_WINDOW
    ).fit(TWENTY_FOUR_DOCUMENTS)
    weighted_cosines = _pairwise_cosines(np.asarray(weighted.embeddings.table))

    table_within, table_across = _within_and_across(full, words)
    projected_within, projected_across = _within_and_across(projected, words)
    weighted_within, weighted_across = _within_and_across(weighted_cosines, words)

    pairs = [
        (one, other)
        for place, one in enumerate(SHARED_WORDS)
        for other in SHARED_WORDS[place + 1 :]
    ]
    positions = {word: place for place, word in enumerate(words)}

    def nearest_to(cosines: FloatArray, here: int) -> str:
        """The single nearest word to one, under one reading."""
        return words[
            next(
                int(place) for place in np.argsort(-cosines[here]) if int(place) != here
            )
        ]

    rows: list[ReadingRow] = []
    for word in COOKING_WORDS[:6]:
        here = positions[word]
        rows.append(
            ReadingRow(
                word=word,
                group="cooking",
                by_the_table=nearest_to(full, here),
                by_the_projection=nearest_to(projected, here),
                by_the_weighted_table=nearest_to(weighted_cosines, here),
            )
        )

    return AgainstTheTableResponse(
        words=list(words),
        row_totals=[float(counts[place].sum()) for place in range(len(words))],
        shared_words=list(SHARED_WORDS),
        shared_row_total=float(counts[positions["the"]].sum()),
        largest_topic_row_total=float(
            max(counts[positions[word]].sum() for word in TOPIC_WORDS)
        ),
        shared_share_of_the_table=float(
            sum(counts[positions[word]].sum() for word in SHARED_WORDS) / counts.sum()
        ),
        table_shared_to_topic=_mean_over_pairs(full, words, SHARED_WORDS, TOPIC_WORDS),
        projected_shared_to_topic=_mean_over_pairs(
            projected, words, SHARED_WORDS, TOPIC_WORDS
        ),
        weighted_shared_to_topic=_mean_over_pairs(
            weighted_cosines, words, SHARED_WORDS, TOPIC_WORDS
        ),
        table_within=table_within,
        table_across=table_across,
        projected_within=projected_within,
        projected_across=projected_across,
        weighted_within=weighted_within,
        weighted_across=weighted_across,
        shared_pairs=[f"{one} and {other}" for one, other in pairs],
        shared_pair_projected=[
            float(projected[positions[one], positions[other]]) for one, other in pairs
        ],
        shared_pair_table=[
            float(full[positions[one], positions[other]]) for one, other in pairs
        ],
        rows=rows,
    )


@router.get("/against-the-table", response_model=AgainstTheTableResponse)
def against_the_table() -> AgainstTheTableResponse:
    """The same counts read three ways, and what the loudest rows do."""
    return _against_the_table()
