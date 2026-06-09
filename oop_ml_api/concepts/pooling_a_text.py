"""Pooling a Text.

Every endpoint here is registered under ``/concepts/pooling-a-text``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about BagOfWords, MeanPooling, SmoothInverseFrequency.

Two collections, both of them the library's own spec fixtures, reproduced here
rather than imported because ``test`` is not part of the installed package; a
test asserts the two agree text for text. The larger is the twenty-four
documents the count-based embedding specs use, twelve about cooking and twelve
about sailing, whose word vectors are the same four-number positions the sibling
page on word vectors fits. The smaller is the pooling spec's own hand-built
table, seven words in four dimensions with ``the`` twice as long as any of them,
which exists so that a plain mean can be worked with a pencil and so that the
failure the weightings repair is visible in one document.

Six methods cross the wire under the page's own names. Two of them count words
and never look at a word's position; four of them average positions, under
uniform weights, inverse document frequency weights, the smoothed weights of
Arora, Liang and Ma, and those weights with the collection's leading direction
projected out.

Contributions are the one calculation here the library does not expose, since a
pooled vector is a sum and the library returns only the sum. Each token's own
term is rebuilt, the method's linear post-step applied to it, and its projection
on the pooled direction reported as a share; the shares sum to one by
construction, and ``contribution_gap`` in every response is the largest
disagreement between those rebuilt terms summed and the vector the library
answered, which a test pins at machine precision so the rebuilding cannot drift
from the thing it describes.

Nothing a widget renders may name a class, a parameter or a module. A refusal is
rewritten by ``_in_plain_words`` before it can reach a widget, and a test asserts
nothing identifier-shaped survives that rewriting.
"""

from __future__ import annotations

import re
from functools import cache
from itertools import combinations
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings import (
    PointwiseMutualInformationEmbeddings,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.documents.bag_of_words import (
    BagOfWords,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.documents.pooling import (
    MeanPooling,
    PoolingWeighting,
    SmoothInverseFrequency,
    smooth_inverse_frequency_weights,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/pooling-a-text", tags=["pooling-a-text"])


# --- The two collections -----------------------------------------------------

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
"""The collection every fixed measurement on this page is made on."""

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
"""The three words carrying no subject, one in each document."""

SKETCH_WORDS: tuple[str, ...] = (
    "the",
    "cat",
    "dog",
    "pet",
    "stock",
    "bond",
    "market",
)
"""The pooling spec's hand-built table, in its own order."""

SKETCH_TABLE: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 2.0, 0.0),
    (1.0, 0.0, 0.0, 0.1),
    (1.0, 0.0, 0.0, -0.1),
    (0.9, 0.1, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.1),
    (0.0, 1.0, 0.0, -0.1),
    (0.1, 0.9, 0.0, 0.0),
)
"""Animals along the first axis, finance along the second, ``the`` along the
third and twice as long as any of them."""

SKETCH_DOCUMENTS: tuple[str, ...] = (
    "the cat the dog",
    "the pet the cat",
    "the dog the pet",
    "the stock the bond",
    "the market the stock",
    "the bond the market",
)
"""Six two-word documents, each carrying ``the`` twice."""

THREE_TEXTS: tuple[str, ...] = (
    "the crew and the boat and we sail",
    "the flour and the dough and we bake",
    "the wind and the tide and we anchor",
)
"""Two about sailing and one about cooking, sharing no word that carries a
subject; the first and third share nothing at all but ``and``, ``the`` and
``we``."""

REVERSED_PAIR: tuple[str, ...] = ("crew sail the boat", "boat sail the crew")
"""One says the crew sails the boat and the other says the boat sails the crew."""

MIXED_TEXT = "the crew and the boat and the flour and the dough"
"""Two sailing words and two cooking words in one text."""

PURE_SAILING = THREE_TEXTS[0]
PURE_COOKING = THREE_TEXTS[1]

UNKNOWN_TEXT = "zebra kettle telescope"
"""Three words the collection never used, so no method has anything to read."""


# --- What the page may ask for -----------------------------------------------

WINDOW = 2
"""How far either side of a word its company is counted, fitting the positions."""

DIMENSION = 4
"""How many numbers a word's position holds, which is also the pooled width."""

MAX_TEXTS = 6
"""The most texts one request may pool, which is more than any widget shows."""

MAX_TEXT_LENGTH = 120
"""How long one text may be, in characters."""

TOP_WORDS = 6
"""How many nearest words a pooled vector is asked for when measuring what it
gives back, chosen because every document of the larger collection is six words
long."""

PAPER_SMOOTHING = 1e-3
"""The value of the smoothing the 2017 paper uses throughout."""

COMMON_PROBABILITY = 0.5
"""A word filling half of every text, for the illustration of the weights."""

RARE_PROBABILITY = 1e-6
"""A word appearing about once in a million, for the same illustration."""

CorpusName = Literal["documents", "sketch"]
MethodName = Literal[
    "counts",
    "weighted-counts",
    "average",
    "weighted-average",
    "smooth-weights",
    "smooth-inverse-frequency",
]

METHOD_LABELS: dict[MethodName, str] = {
    "counts": "counting",
    "weighted-counts": "counting, weighted by rarity",
    "average": "the plain average",
    "weighted-average": "the average, weighted by rarity",
    "smooth-weights": "the smooth weights alone",
    "smooth-inverse-frequency": "smooth inverse frequency",
}

COUNTING_METHODS: tuple[MethodName, ...] = ("counts", "weighted-counts")
POOLING_METHODS: tuple[MethodName, ...] = (
    "average",
    "weighted-average",
    "smooth-weights",
    "smooth-inverse-frequency",
)
ALL_METHODS: tuple[MethodName, ...] = COUNTING_METHODS + POOLING_METHODS


PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("minimum_count", "the smallest number of uses a word needs"),
    ("n_results", "the number of neighbours asked for"),
    ("smoothing", "the smoothing"),
    ("n_documents", "the number of documents"),
)
"""Identifiers a reader has no reason to meet, and what to say instead."""

IDENTIFIER_SHAPED = re.compile(r"[A-Za-z]+_[A-Za-z_]+|\b[a-z]+[A-Z][A-Za-z]*\b")
"""Anything spelled the way a name in code is spelled."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the ones that name a setting and leaves the rest alone. A test
    asserts that no identifier survives, so a message that grows one later fails
    there rather than reaching a reader.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _refuse(refusal: MLLibError) -> HTTPException:
    return HTTPException(status_code=422, detail=_in_plain_words(refusal))


# --- The fitted pieces, each computed once -----------------------------------


def group_of(word: str) -> str:
    """Which half of a collection a word belongs to, or neither."""
    if word in COOKING_WORDS or word in ("cat", "dog", "pet"):
        return "cooking"
    if word in SAILING_WORDS or word in ("stock", "bond", "market"):
        return "sailing"
    return "shared"


def documents_of(corpus: CorpusName) -> list[str]:
    """The texts a collection is fitted on."""
    if corpus == "documents":
        return TWENTY_FOUR_DOCUMENTS
    return list(SKETCH_DOCUMENTS)


def halves_of(corpus: CorpusName) -> tuple[range, range]:
    """The two topic halves of a collection, as positions."""
    if corpus == "documents":
        return range(12), range(12, 24)
    return range(3), range(3, 6)


@cache
def _positions(corpus: CorpusName) -> WordEmbeddings:
    """One position per word: fitted for the larger collection, given for the
    smaller."""
    if corpus == "sketch":
        return WordEmbeddings(
            Vocabulary(list(SKETCH_WORDS)), np.asarray(SKETCH_TABLE, dtype=np.float64)
        )
    return (
        PointwiseMutualInformationEmbeddings(window=WINDOW, dimension=DIMENSION)
        .fit(TWENTY_FOUR_DOCUMENTS)
        .embeddings
    )


@cache
def _model(
    corpus: CorpusName, method: MethodName
) -> BagOfWords | MeanPooling | SmoothInverseFrequency:
    """The fitted embedder for one collection and one method."""
    texts = documents_of(corpus)
    if method == "counts":
        return BagOfWords(normalise=True).fit(texts)
    if method == "weighted-counts":
        return BagOfWords(
            weighting=TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
            normalise=True,
        ).fit(texts)
    positions = _positions(corpus)
    if method == "average":
        return MeanPooling(embeddings=positions).fit(texts)
    if method == "weighted-average":
        return MeanPooling(
            embeddings=positions,
            weighting=PoolingWeighting.INVERSE_DOCUMENT_FREQUENCY,
        ).fit(texts)
    if method == "smooth-weights":
        return SmoothInverseFrequency(
            embeddings=positions,
            smoothing=PAPER_SMOOTHING,
            remove_first_component=False,
        ).fit(texts)
    return SmoothInverseFrequency(
        embeddings=positions, smoothing=PAPER_SMOOTHING, remove_first_component=True
    ).fit(texts)


@cache
def _component(corpus: CorpusName) -> FloatArray:
    """The one direction the collection's documents share."""
    model = _model(corpus, "smooth-inverse-frequency")
    assert isinstance(model, SmoothInverseFrequency)
    return np.asarray(model.first_component, dtype=np.float64)


def _vocabulary_words(corpus: CorpusName) -> list[str]:
    model = _model(corpus, "counts")
    assert isinstance(model, BagOfWords)
    return list(model.vocabulary)


# --- Rebuilding one token's share --------------------------------------------


def _token_terms(
    corpus: CorpusName, method: MethodName, words: list[str]
) -> FloatArray:
    """``(n_tokens, dimension)``: what each token of one text put into its vector.

    Every method here is a weighted sum of one term per token followed by a
    linear step applied to the sum, so the same step applied to each term
    separately still adds up to the answer. That is what makes a share of the
    result meaningful, and the endpoints report the largest disagreement between
    these terms summed and what the library returned.
    """
    model = _model(corpus, method)
    if method in COUNTING_METHODS:
        assert isinstance(model, BagOfWords)
        vocabulary = model.vocabulary
        width = vocabulary.n_tokens
        terms = np.zeros((len(words), width))
        weights = np.asarray(model.inverse_document_frequencies, dtype=np.float64)
        for row, word in enumerate(words):
            if word in vocabulary:
                position = vocabulary.id_of(word)
                terms[row, position] = (
                    weights[position] if method == "weighted-counts" else 1.0
                )
        total = float(np.linalg.norm(terms.sum(axis=0)))
        return terms / total if total > 0.0 else terms

    positions = _positions(corpus)
    vocabulary = positions.vocabulary
    known = [word for word in words if word in vocabulary]
    terms = np.zeros((len(words), positions.dimension))
    if not known:
        return terms

    if method == "average":
        for row, word in enumerate(words):
            if word in vocabulary:
                terms[row] = positions.vector_of(word).values / len(known)
        return terms

    if method == "weighted-average":
        assert isinstance(model, MeanPooling)
        weights = np.asarray(model.inverse_document_frequencies, dtype=np.float64)
        total = float(sum(weights[vocabulary.id_of(word)] for word in known))
        for row, word in enumerate(words):
            if word in vocabulary:
                weight = weights[vocabulary.id_of(word)]
                terms[row] = weight * positions.vector_of(word).values / total
        return terms

    assert isinstance(model, SmoothInverseFrequency)
    weights = np.asarray(model.word_weights, dtype=np.float64)
    for row, word in enumerate(words):
        if word in vocabulary:
            weight = weights[vocabulary.id_of(word)]
            terms[row] = weight * positions.vector_of(word).values / len(known)
    if method == "smooth-inverse-frequency":
        component = _component(corpus)
        terms = terms - np.outer(terms @ component, component)
    return terms


def _shares(terms: FloatArray, pooled: FloatArray) -> list[float]:
    """Each token's share of the pooled vector's length along its own direction."""
    length = float(np.linalg.norm(pooled))
    if length == 0.0:
        return [0.0] * terms.shape[0]
    return [float(value) for value in (terms @ pooled) / (length * length)]


def _words_of(text: str) -> list[str]:
    return text.split()


# --- The shape a pooled text comes back in -----------------------------------


class WordShare(BaseModel):
    """One token of one text, and how much of that text's vector it supplied."""

    word: str
    group: str
    share: float


class PooledText(BaseModel):
    """One text, its vector, and where the vector came from."""

    text: str
    words: list[str]
    known_words: int
    vector: list[float]
    length: float
    shares: list[WordShare]
    by_word: list[WordShare]


class TextPair(BaseModel):
    """Two texts of one request and the angle between their vectors."""

    first: int
    second: int
    similarity: float | None
    undefined_because: str | None


class PoolRequest(BaseModel):
    """Some texts, one collection, and one way of turning each into a vector."""

    corpus: CorpusName = "documents"
    method: MethodName = "average"
    texts: list[str] = Field(
        default_factory=lambda: list(THREE_TEXTS), min_length=1, max_length=MAX_TEXTS
    )

    model_config = {"extra": "forbid"}


class PoolResponse(BaseModel):
    """Every text of the request as a vector, with the pairwise angles.

    ``contribution_gap`` is the largest disagreement between each text's shares
    rebuilt from its tokens and the vector answered for it, and is at machine
    precision whenever the rebuilding is right.
    """

    corpus: CorpusName
    method: MethodName
    method_label: str
    dimension: int
    reads_positions: bool
    texts: list[PooledText]
    pairs: list[TextPair]
    smallest_similarity: float | None
    largest_similarity: float | None
    contribution_gap: float


def _pooled(
    corpus: CorpusName, method: MethodName, texts: list[str]
) -> DocumentVectors:
    try:
        return _model(corpus, method).transform(texts)
    except MLLibError as refusal:
        raise _refuse(refusal) from refusal


@router.post("/pool", response_model=PoolResponse)
def pool(request: PoolRequest) -> PoolResponse:
    """Turn each text into one vector, and report where the vector came from."""
    for text in request.texts:
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"a text here may be {MAX_TEXT_LENGTH} characters long at most, "
                    f"and this one is {len(text)}"
                ),
            )

    table = _pooled(request.corpus, request.method, request.texts)
    vocabulary = _model(request.corpus, "counts").vocabulary  # type: ignore[union-attr]
    gap = 0.0
    pooled_texts: list[PooledText] = []
    for position, text in enumerate(request.texts):
        words = _words_of(text)
        vector = np.asarray(table.vector_of(position), dtype=np.float64)
        terms = _token_terms(request.corpus, request.method, words)
        gap = max(gap, float(np.max(np.abs(terms.sum(axis=0) - vector))))
        shares = _shares(terms, vector)
        totals: dict[str, float] = {}
        for word, share in zip(words, shares):
            totals[word] = totals.get(word, 0.0) + share
        pooled_texts.append(
            PooledText(
                text=text,
                words=words,
                known_words=sum(1 for word in words if word in vocabulary),
                vector=[float(value) for value in vector],
                length=float(np.linalg.norm(vector)),
                shares=[
                    WordShare(word=word, group=group_of(word), share=share)
                    for word, share in zip(words, shares)
                ],
                by_word=[
                    WordShare(word=word, group=group_of(word), share=share)
                    for word, share in sorted(totals.items(), key=lambda pair: -pair[1])
                ],
            )
        )

    pairs: list[TextPair] = []
    for first, second in combinations(range(len(request.texts)), 2):
        try:
            pairs.append(
                TextPair(
                    first=first,
                    second=second,
                    similarity=float(table.similarity(first, second)),
                    undefined_because=None,
                )
            )
        except MLLibError as refusal:
            pairs.append(
                TextPair(
                    first=first,
                    second=second,
                    similarity=None,
                    undefined_because=_in_plain_words(refusal),
                )
            )

    measured = [pair.similarity for pair in pairs if pair.similarity is not None]
    return PoolResponse(
        corpus=request.corpus,
        method=request.method,
        method_label=METHOD_LABELS[request.method],
        dimension=table.dimension,
        reads_positions=request.method in POOLING_METHODS,
        texts=pooled_texts,
        pairs=pairs,
        smallest_similarity=min(measured) if measured else None,
        largest_similarity=max(measured) if measured else None,
        contribution_gap=gap,
    )


# --- How well each method keeps the two halves apart -------------------------


class SeparationRow(BaseModel):
    """One method on one collection, scored on how far apart it puts the halves."""

    corpus: CorpusName
    method: MethodName
    method_label: str
    reads_positions: bool
    width: int
    within_mean: float
    within_smallest: float
    between_mean: float
    between_largest: float
    margin: float
    mean_gap: float
    words_given_back: float | None


class ThreeTextRow(BaseModel):
    """The three probe texts under one method."""

    method: MethodName
    method_label: str
    sailing_to_cooking: float
    sailing_to_sailing: float
    cooking_to_sailing: float
    margin: float


class SeparationResponse(BaseModel):
    """Every method on both collections, and the three probe texts."""

    rows: list[SeparationRow]
    three_texts: list[ThreeTextRow]
    texts: list[str]
    text_groups: list[str]
    shared_between_texts: list[str]
    n_documents: int
    n_words: int
    position_width: int
    counted_width: int


def _within_and_between(
    corpus: CorpusName, table: FloatArray
) -> tuple[list[float], list[float]]:
    first_half, second_half = halves_of(corpus)
    within = [
        cosine_similarity(table[first], table[second])
        for half in (first_half, second_half)
        for first, second in combinations(half, 2)
    ]
    between = [
        cosine_similarity(table[first], table[second])
        for first in first_half
        for second in second_half
    ]
    return within, between


def _words_given_back(corpus: CorpusName, method: MethodName) -> float:
    """How many of a document's own words are among the words nearest its vector."""
    positions = _positions(corpus)
    table = np.asarray(_pooled(corpus, method, documents_of(corpus)))
    found: list[int] = []
    for position, document in enumerate(documents_of(corpus)):
        vector = table[position]
        if float(np.linalg.norm(vector)) == 0.0:
            found.append(0)
            continue
        ranked = sorted(
            positions.vocabulary,
            key=lambda word: (
                -cosine_similarity(positions.vector_of(word).values, vector)
            ),
        )
        found.append(len(set(ranked[:TOP_WORDS]) & set(_words_of(document))))
    return float(np.mean(found))


@cache
def _separation() -> SeparationResponse:
    rows: list[SeparationRow] = []
    for corpus in ("documents", "sketch"):
        for method in ALL_METHODS:
            table = np.asarray(_pooled(corpus, method, documents_of(corpus)))
            within, between = _within_and_between(corpus, table)
            rows.append(
                SeparationRow(
                    corpus=corpus,
                    method=method,
                    method_label=METHOD_LABELS[method],
                    reads_positions=method in POOLING_METHODS,
                    width=int(table.shape[1]),
                    within_mean=float(np.mean(within)),
                    within_smallest=float(min(within)),
                    between_mean=float(np.mean(between)),
                    between_largest=float(max(between)),
                    margin=float(min(within) - max(between)),
                    mean_gap=float(np.mean(within) - np.mean(between)),
                    words_given_back=(
                        _words_given_back(corpus, method)
                        if method in POOLING_METHODS
                        else None
                    ),
                )
            )

    three: list[ThreeTextRow] = []
    for method in ALL_METHODS:
        table = _pooled("documents", method, list(THREE_TEXTS))
        sailing_cooking = float(table.similarity(0, 1))
        sailing_sailing = float(table.similarity(0, 2))
        cooking_sailing = float(table.similarity(1, 2))
        three.append(
            ThreeTextRow(
                method=method,
                method_label=METHOD_LABELS[method],
                sailing_to_cooking=sailing_cooking,
                sailing_to_sailing=sailing_sailing,
                cooking_to_sailing=cooking_sailing,
                margin=sailing_sailing - max(sailing_cooking, cooking_sailing),
            )
        )

    shared = set(_words_of(THREE_TEXTS[0]))
    for text in THREE_TEXTS[1:]:
        shared &= set(_words_of(text))

    return SeparationResponse(
        rows=rows,
        three_texts=three,
        texts=list(THREE_TEXTS),
        text_groups=["sailing", "cooking", "sailing"],
        shared_between_texts=sorted(shared),
        n_documents=len(TWENTY_FOUR_DOCUMENTS),
        n_words=len(_vocabulary_words("documents")),
        position_width=_positions("documents").dimension,
        counted_width=len(_vocabulary_words("documents")),
    )


@router.get("/separation", response_model=SeparationResponse)
def separation() -> SeparationResponse:
    """Score every method on how far apart it puts the two halves."""
    return _separation()


# --- The direction every text shares -----------------------------------------


class WordAgainstDirection(BaseModel):
    """One word, how far it leans along the shared direction, and how long it is."""

    word: str
    group: str
    cosine: float
    length: float
    use_count: int
    weight: float


class DirectionResponse(BaseModel):
    """The leading direction of a collection, and what turns out to lie along it."""

    corpus: CorpusName
    component: list[float]
    words: list[WordAgainstDirection]
    shared_words: list[str]
    largest_content_cosine: float
    smallest_shared_cosine: float
    along_mean: float
    along_smallest: float
    along_largest: float
    shared_share_mean: float
    shared_share_equal: float
    smallest_weight: float
    largest_weight: float
    weight_ratio: float
    weight_of_a_common_word: float
    weight_of_a_rare_word: float
    illustrated_ratio: float
    smoothing: float


@cache
def _direction(corpus: CorpusName) -> DirectionResponse:
    positions = _positions(corpus)
    model = _model(corpus, "smooth-inverse-frequency")
    assert isinstance(model, SmoothInverseFrequency)
    component = _component(corpus)
    counts = _model(corpus, "counts")
    assert isinstance(counts, BagOfWords)
    totals = np.asarray(counts.term_document_matrix.values).sum(axis=1)

    words: list[WordAgainstDirection] = []
    for word in positions.vocabulary:
        vector = positions.vector_of(word).values
        words.append(
            WordAgainstDirection(
                word=word,
                group=group_of(word),
                cosine=cosine_similarity(vector, component),
                length=float(np.linalg.norm(vector)),
                use_count=int(totals[counts.vocabulary.id_of(word)]),
                weight=float(model.weight_of(word)),
            )
        )
    words.sort(key=lambda entry: -entry.cosine)

    shared = [entry for entry in words if entry.group == "shared"]
    content = [entry for entry in words if entry.group != "shared"]

    plain = np.asarray(_pooled(corpus, "average", documents_of(corpus)))
    along = np.abs(plain @ component) / np.linalg.norm(plain, axis=1)
    shared_shares: list[float] = []
    equal_shares: list[float] = []
    for position, document in enumerate(documents_of(corpus)):
        document_words = _words_of(document)
        terms = _token_terms(corpus, "average", document_words)
        shares = _shares(terms, plain[position])
        shared_shares.append(
            sum(
                share
                for word, share in zip(document_words, shares)
                if group_of(word) == "shared"
            )
        )
        equal_shares.append(
            sum(
                1.0 / len(document_words)
                for word in document_words
                if group_of(word) == "shared"
            )
        )
    weights = np.asarray(model.word_weights, dtype=np.float64)
    illustrated = smooth_inverse_frequency_weights(
        PAPER_SMOOTHING, np.asarray([COMMON_PROBABILITY, RARE_PROBABILITY])
    )

    return DirectionResponse(
        corpus=corpus,
        component=[float(value) for value in component],
        words=words,
        shared_words=[entry.word for entry in shared],
        largest_content_cosine=max(entry.cosine for entry in content),
        smallest_shared_cosine=min(entry.cosine for entry in shared),
        along_mean=float(np.mean(along)),
        along_smallest=float(np.min(along)),
        along_largest=float(np.max(along)),
        shared_share_mean=float(np.mean(shared_shares)),
        shared_share_equal=float(np.mean(equal_shares)),
        smallest_weight=float(weights.min()),
        largest_weight=float(weights.max()),
        weight_ratio=float(weights.max() / weights.min()),
        weight_of_a_common_word=float(illustrated[0]),
        weight_of_a_rare_word=float(illustrated[1]),
        illustrated_ratio=float(illustrated[1] / illustrated[0]),
        smoothing=PAPER_SMOOTHING,
    )


class DirectionRequest(BaseModel):
    """Which collection's leading direction to read."""

    corpus: CorpusName = "documents"

    model_config = {"extra": "forbid"}


@router.post("/direction", response_model=DirectionResponse)
def direction(request: DirectionRequest) -> DirectionResponse:
    """Report the leading direction of a collection and what lies along it."""
    return _direction(request.corpus)


# --- What reordering a text does ---------------------------------------------


class OrderRow(BaseModel):
    """One method's reading of a text and of the same words in another order."""

    method: MethodName
    method_label: str
    first_vector: list[float]
    second_vector: list[float]
    largest_gap: float
    similarity: float
    identical: bool


class OrderResponse(BaseModel):
    """Two texts with the same words in different orders, under every method."""

    first: str
    second: str
    rows: list[OrderRow]
    n_orderings: int


@cache
def _order() -> OrderResponse:
    rows: list[OrderRow] = []
    for method in ALL_METHODS:
        table = _pooled("documents", method, list(REVERSED_PAIR))
        first = np.asarray(table.vector_of(0), dtype=np.float64)
        second = np.asarray(table.vector_of(1), dtype=np.float64)
        gap = float(np.max(np.abs(first - second)))
        rows.append(
            OrderRow(
                method=method,
                method_label=METHOD_LABELS[method],
                first_vector=[float(value) for value in first],
                second_vector=[float(value) for value in second],
                largest_gap=gap,
                similarity=float(table.similarity(0, 1)),
                identical=gap == 0.0,
            )
        )
    orderings = 1
    for factor in range(1, len(_words_of(REVERSED_PAIR[0])) + 1):
        orderings *= factor
    return OrderResponse(
        first=REVERSED_PAIR[0],
        second=REVERSED_PAIR[1],
        rows=rows,
        n_orderings=orderings,
    )


@router.get("/order", response_model=OrderResponse)
def order() -> OrderResponse:
    """Read one text and the same words in another order, under every method."""
    return _order()


# --- The same text in different company --------------------------------------


class CompanyWord(BaseModel):
    """One word's lean along each of two collections' leading directions."""

    word: str
    group: str
    in_the_whole: float
    in_the_half: float


class CompanyResponse(BaseModel):
    """One text scored twice, against the whole collection and against one half."""

    text: str
    whole_vector: list[float]
    half_vector: list[float]
    similarity_between_readings: float
    whole_component: list[float]
    half_component: list[float]
    component_similarity: float
    words: list[CompanyWord]
    n_whole: int
    n_half: int


@cache
def _company() -> CompanyResponse:
    positions = _positions("documents")
    whole = _model("documents", "smooth-inverse-frequency")
    assert isinstance(whole, SmoothInverseFrequency)
    half = SmoothInverseFrequency(embeddings=positions, smoothing=PAPER_SMOOTHING).fit(
        list(SAILING_DOCUMENTS)
    )

    whole_vector = np.asarray(whole.transform([PURE_SAILING]).vector_of(0))
    half_vector = np.asarray(half.transform([PURE_SAILING]).vector_of(0))
    whole_component = np.asarray(whole.first_component, dtype=np.float64)
    half_component = np.asarray(half.first_component, dtype=np.float64)

    words = [
        CompanyWord(
            word=word,
            group=group_of(word),
            in_the_whole=cosine_similarity(
                positions.vector_of(word).values, whole_component
            ),
            in_the_half=cosine_similarity(
                positions.vector_of(word).values, half_component
            ),
        )
        for word in _words_of(PURE_SAILING)
        if word in positions.vocabulary
    ]
    seen: dict[str, CompanyWord] = {}
    for entry in words:
        seen.setdefault(entry.word, entry)

    return CompanyResponse(
        text=PURE_SAILING,
        whole_vector=[float(value) for value in whole_vector],
        half_vector=[float(value) for value in half_vector],
        similarity_between_readings=cosine_similarity(whole_vector, half_vector),
        whole_component=[float(value) for value in whole_component],
        half_component=[float(value) for value in half_component],
        component_similarity=cosine_similarity(whole_component, half_component),
        words=list(seen.values()),
        n_whole=len(TWENTY_FOUR_DOCUMENTS),
        n_half=len(SAILING_DOCUMENTS),
    )


@router.get("/company", response_model=CompanyResponse)
def company() -> CompanyResponse:
    """Score one text against the whole collection and against one half of it."""
    return _company()


# --- The cases with nothing to compute ---------------------------------------


class MixedRow(BaseModel):
    """A text about both subjects, scored against a text about each."""

    method: MethodName
    method_label: str
    to_sailing: float
    to_cooking: float
    sailing_to_cooking: float


class EdgeResponse(BaseModel):
    """The measurements the closing part of the page rests on."""

    mixed_text: str
    pure_sailing: str
    pure_cooking: str
    mixed: list[MixedRow]
    mixed_nearest: list[str]
    mixed_nearest_groups: list[str]
    unknown_text: str
    unknown_vector_length: float
    unknown_refusal: str
    single_residue_length: float
    single_document: str
    counting_smallest_within: float
    smooth_smallest_within: float
    average_smallest_within: float


@cache
def _edges() -> EdgeResponse:
    positions = _positions("documents")
    probe = [MIXED_TEXT, PURE_SAILING, PURE_COOKING]
    mixed: list[MixedRow] = []
    for method in ALL_METHODS:
        table = _pooled("documents", method, probe)
        mixed.append(
            MixedRow(
                method=method,
                method_label=METHOD_LABELS[method],
                to_sailing=float(table.similarity(0, 1)),
                to_cooking=float(table.similarity(0, 2)),
                sailing_to_cooking=float(table.similarity(1, 2)),
            )
        )

    smooth = _model("documents", "smooth-inverse-frequency")
    trained = smooth.transform(TWENTY_FOUR_DOCUMENTS)
    query = np.asarray(smooth.transform([MIXED_TEXT]).vector_of(0))
    nearest = trained.similar_to_vector(query, n_results=4)

    unknown_table = _pooled("documents", "average", [UNKNOWN_TEXT, PURE_SAILING])
    try:
        unknown_table.similarity(0, 1)
        refusal_sentence = ""
    except MLLibError as refusal:
        refusal_sentence = _in_plain_words(refusal)

    single = SmoothInverseFrequency(
        embeddings=positions, smoothing=PAPER_SMOOTHING
    ).fit([PURE_SAILING])
    residue = np.asarray(single.transform([PURE_SAILING]).vector_of(0))

    smallest: dict[MethodName, float] = {}
    for method in ("counts", "average", "smooth-inverse-frequency"):
        table = np.asarray(_pooled("documents", method, TWENTY_FOUR_DOCUMENTS))
        within, _ = _within_and_between("documents", table)
        smallest[method] = float(min(within))

    return EdgeResponse(
        mixed_text=MIXED_TEXT,
        pure_sailing=PURE_SAILING,
        pure_cooking=PURE_COOKING,
        mixed=mixed,
        mixed_nearest=[TWENTY_FOUR_DOCUMENTS[position] for position in nearest],
        mixed_nearest_groups=[
            "cooking" if position < 12 else "sailing" for position in nearest
        ],
        unknown_text=UNKNOWN_TEXT,
        unknown_vector_length=float(
            np.linalg.norm(np.asarray(unknown_table.vector_of(0)))
        ),
        unknown_refusal=refusal_sentence,
        single_residue_length=float(np.linalg.norm(residue)),
        single_document=PURE_SAILING,
        counting_smallest_within=smallest["counts"],
        smooth_smallest_within=smallest["smooth-inverse-frequency"],
        average_smallest_within=smallest["average"],
    )


@router.get("/edges", response_model=EdgeResponse)
def edges() -> EdgeResponse:
    """The measurements behind the closing part of the page."""
    return _edges()
