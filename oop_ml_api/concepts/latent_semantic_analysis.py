"""Latent semantic analysis.

Every endpoint here is registered under ``/concepts/latent-semantic-analysis``,
and the router is created and registered before any endpoint exists, so the
pages of this section can be built in parallel without many edits to one shared
file.

The page is about LatentSemanticAnalysis, TermDocumentMatrix.

Two corpora, both of them the library's own spec fixtures, reproduced here
rather than imported because ``test`` is not part of the installed package; a
test asserts the two agree text for text. The four-document one is small enough
to print a whole term-document table of, and its singular values are exactly
``sqrt(7)``, ``sqrt(3)``, 1 and 1. The twenty-four document one is a cyclic
design over two topics, built so that the difference between the topics carries
more spread than anything inside a topic, which is what makes the claim about
the second component testable rather than lucky.

Every number the page quotes is computed by the library. The three things
computed here are the cosine between two *raw* rows or columns of the matrix,
which is the comparison the method is an alternative to and so has no model of
its own; the same decomposition applied to the row-centred matrix, which is
what the other decomposition on this site does and is here only for the
contrast; and the corpus in which one spelling carries two senses, which is a
substitution on the twenty-four documents. All three go through the library's
own cosine, its own term-document matrix and its own decomposition.

Nothing a widget renders may name a class, a parameter or a module, so the two
weightings cross the wire as the page's own words, ``counts`` and ``weighted``,
and ``_WEIGHTINGS`` is the only place a weighting meets the library's own name.
A refusal is rewritten by ``_in_plain_words`` before it can reach a widget, and
a test asserts nothing identifier-shaped survives that rewriting.
"""

from __future__ import annotations

import re
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (
    LatentSemanticAnalysis,
    TruncatedSingularValueDecomposition,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermDocumentMatrix,
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.embeddings.vectors import cosine_similarity
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/latent-semantic-analysis", tags=["latent-semantic-analysis"]
)


# --- The two corpora ---------------------------------------------------------

FOUR_DOCUMENTS: tuple[str, ...] = (
    "the cat sat",
    "the cat ran",
    "the boat sailed",
    "the boat sank",
)
"""Seven words over four documents, whose whole table fits on a page."""

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

N_COOKING_DOCUMENTS = len(COOKING_DOCUMENTS)

CorpusName = Literal["four", "twenty-four"]
WeightingName = Literal["counts", "weighted"]

_CORPORA: dict[CorpusName, list[str]] = {
    "four": list(FOUR_DOCUMENTS),
    "twenty-four": TWENTY_FOUR_DOCUMENTS,
}

_WEIGHTINGS: dict[WeightingName, TermWeighting] = {
    "counts": TermWeighting.COUNT,
    "weighted": TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
}

_GROUP_OF: dict[CorpusName, dict[str, str]] = {
    "four": {
        "the": "shared",
        "cat": "cat",
        "sat": "cat",
        "ran": "cat",
        "boat": "boat",
        "sailed": "boat",
        "sank": "boat",
    },
    "twenty-four": {
        **{word: "cooking" for word in COOKING_WORDS},
        **{word: "sailing" for word in SAILING_WORDS},
        **{word: "shared" for word in SHARED_WORDS},
    },
}
"""Which half of a corpus each word belongs to, for the widgets that colour by it."""


def _document_group(corpus: CorpusName, position: int) -> str:
    """Which half a document belongs to, by where it sits in the corpus."""
    if corpus == "four":
        return "cat" if position < 2 else "boat"
    return "cooking" if position < N_COOKING_DOCUMENTS else "sailing"


# --- What a request may ask for ----------------------------------------------

PAGE_DIMENSION = 2
"""How many directions the page's headline measurements keep.

Two, because two is what can be drawn and because the first two are where the
claim about the first one lives. Several sections deliberately use six instead,
and say so.
"""

WIDER_DIMENSION = 6
"""The width at which the coordinates start telling two words of one half apart."""

MAX_DIMENSION = 23
"""The most directions any corpus here has, which is the twenty-four documents'
twenty-three terms."""

MAX_NEW_TEXT = 200
"""How long a text a reader may hand the fitted space, in characters."""


# --- Shared pieces -----------------------------------------------------------


@cache
def _tokenised(corpus: CorpusName) -> TokenisedCorpus:
    """One corpus split into words, once."""
    return TokenisedCorpus.from_texts(_CORPORA[corpus], WhitespacePreTokenizer())


@cache
def _vocabulary(corpus: CorpusName) -> Vocabulary:
    """The words of one corpus, commonest first."""
    return _tokenised(corpus).vocabulary(1)


@cache
def _matrix(corpus: CorpusName, weighting: WeightingName) -> TermDocumentMatrix:
    """One row per word, one column per document, weighted as asked."""
    return TermDocumentMatrix.from_tokenised(
        _tokenised(corpus), _vocabulary(corpus), _WEIGHTINGS[weighting]
    )


@cache
def _fitted(
    corpus: CorpusName, weighting: WeightingName, dimension: int
) -> LatentSemanticAnalysis:
    """One fit, kept, since several widgets ask for the same one."""
    return LatentSemanticAnalysis(
        dimension=dimension, weighting=_WEIGHTINGS[weighting]
    ).fit(_CORPORA[corpus])


@cache
def _largest_rank(corpus: CorpusName) -> int:
    """The most directions a corpus's table can be asked for."""
    matrix = _matrix(corpus, "counts")
    return min(matrix.n_terms, matrix.n_documents)


NAMED_IDENTIFIER = re.compile(r"\bdimension=")
"""The one parameter name the refusals here can carry."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifier taken out of it.

    The messages are written to be read, and the only one reachable from this
    page names the setting it refused by its parameter name. A test asserts
    nothing identifier-shaped survives, so a message that grows one later fails
    there rather than reaching a reader.
    """
    return NAMED_IDENTIFIER.sub("keeping ", str(refusal))


def _checked_dimension(corpus: CorpusName, dimension: int) -> int:
    """Refuse a width the corpus has no directions for, in the page's words."""
    largest = _largest_rank(corpus)
    if dimension > largest:
        raise HTTPException(
            status_code=422,
            detail=(
                f"this corpus of {_matrix(corpus, 'counts').n_terms} words and "
                f"{_matrix(corpus, 'counts').n_documents} documents has "
                f"{largest} directions in all, so {dimension} of them cannot be "
                f"kept"
            ),
        )
    return dimension


def _cosine(first: FloatArray, second: FloatArray) -> float:
    """The angle between two vectors, or zero where one of them has no length.

    A raw row or column of a term-document matrix is never zero, since every
    word appears somewhere and every document here holds a word, so the guard
    is for the rebuilt tables, where a rounded row can be.
    """
    if float(np.linalg.norm(first)) == 0.0 or float(np.linalg.norm(second)) == 0.0:
        return 0.0
    return cosine_similarity(np.asarray(first), np.asarray(second))


def _within_and_across(model: LatentSemanticAnalysis) -> tuple[float, float]:
    """Mean cosine inside a half, and mean cosine between the two halves.

    Only the topic words count, since the three words both halves use belong to
    neither and comparing them to anything says nothing about whether the halves
    came apart.
    """
    inside = [
        model.similarity(first, second)
        for words in (COOKING_WORDS, SAILING_WORDS)
        for position, first in enumerate(words)
        for second in words[position + 1 :]
    ]
    between = [
        model.similarity(first, second)
        for first in COOKING_WORDS
        for second in SAILING_WORDS
    ]
    return float(np.mean(inside)), float(np.mean(between))


# --- The table itself --------------------------------------------------------


class TermRow(BaseModel):
    """One word's row, counted and weighted, with how it earned its weight."""

    word: str
    group: str
    document_frequency: int
    weight: float
    counts: list[float]
    weighted: list[float]
    total_count: float


class TableRequest(BaseModel):
    """Which corpus to lay out as a table."""

    corpus: CorpusName = "four"

    model_config = {"extra": "forbid"}


class TableResponse(BaseModel):
    """The whole table, both ways, with what the weighting moved.

    ``everywhere_words`` are the words that appear in every document, which are
    the ones the raw counts cannot use and the weighting is aimed at.
    """

    corpus: CorpusName
    documents: list[str]
    document_groups: list[str]
    words: list[str]
    rows: list[TermRow]
    n_terms: int
    n_documents: int
    n_cells: int
    n_nonzero: int
    zero_share: float
    everywhere_words: list[str]
    commonest_word: str
    count_share_of_a_column: float
    weighted_share_of_a_column: float
    smallest_weight: float
    largest_weight: float
    unsmoothed_weight_of_the_commonest: float


@router.post("/table", response_model=TableResponse)
def table(request: TableRequest) -> TableResponse:
    """Count every word in every document, then weight the counts."""
    corpus = request.corpus
    texts = _CORPORA[corpus]
    counted = _matrix(corpus, "counts")
    weighted = _matrix(corpus, "weighted")
    words = list(_vocabulary(corpus))
    frequencies = np.asarray(counted.document_frequencies)
    weights = np.asarray(counted.inverse_document_frequencies)
    counts = np.asarray(counted.values)
    weighted_values = np.asarray(weighted.values)

    commonest = words[0]
    first_column_counts = counts[:, 0]
    first_column_weighted = weighted_values[:, 0]
    n_documents = counted.n_documents

    return TableResponse(
        corpus=corpus,
        documents=texts,
        document_groups=[
            _document_group(corpus, position) for position in range(len(texts))
        ],
        words=words,
        rows=[
            TermRow(
                word=word,
                group=_GROUP_OF[corpus].get(word, "shared"),
                document_frequency=int(frequencies[position]),
                weight=float(weights[position]),
                counts=[float(value) for value in counts[position]],
                weighted=[float(value) for value in weighted_values[position]],
                total_count=float(counts[position].sum()),
            )
            for position, word in enumerate(words)
        ],
        n_terms=counted.n_terms,
        n_documents=n_documents,
        n_cells=int(counts.size),
        n_nonzero=int((counts > 0.0).sum()),
        zero_share=float((counts == 0.0).mean()),
        everywhere_words=[
            word
            for position, word in enumerate(words)
            if int(frequencies[position]) == n_documents
        ],
        commonest_word=commonest,
        count_share_of_a_column=float(
            first_column_counts[0] / first_column_counts.sum()
        ),
        weighted_share_of_a_column=float(
            first_column_weighted[0] / first_column_weighted.sum()
        ),
        smallest_weight=float(weights.min()),
        largest_weight=float(weights.max()),
        unsmoothed_weight_of_the_commonest=float(np.log(n_documents / frequencies[0])),
    )


# --- One fit -----------------------------------------------------------------


class WordPoint(BaseModel):
    """One word's position in the fitted space."""

    word: str
    group: str
    coordinates: list[float]


class DocumentPoint(BaseModel):
    """One document's position in the same space."""

    position: int
    text: str
    group: str
    coordinates: list[float]


class NeighbourWord(BaseModel):
    """One word near another, and how near."""

    word: str
    group: str
    similarity: float


class NeighbourDocument(BaseModel):
    """One document near another, and how near."""

    position: int
    text: str
    group: str
    similarity: float


class ComponentReport(BaseModel):
    """One direction, what it accounts for, and how the two halves sit on it.

    ``group_difference`` is the first half's mean coordinate minus the second's,
    which is zero to rounding on the leading direction and large on the one
    after it.
    """

    position: int
    singular_value: float
    share: float
    cumulative: float
    first_group_mean: float
    second_group_mean: float
    group_difference: float
    all_one_sign: bool


class FitRequest(BaseModel):
    """One fit, and the word, the document and the new text to ask about."""

    corpus: CorpusName = "twenty-four"
    weighting: WeightingName = "weighted"
    dimension: int = Field(default=PAGE_DIMENSION, ge=1, le=MAX_DIMENSION)
    word: str = Field(default="flour", min_length=1, max_length=32)
    document: int = Field(default=0, ge=0, le=23)
    n_results: int = Field(default=8, ge=1, le=12)
    new_text: str = Field(default="", max_length=MAX_NEW_TEXT)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything one fit has to say, from its spectrum to its neighbours.

    ``components`` covers every direction the table has, not only the kept
    ones, so the shares a truncated fit reports can be read against the whole.
    ``placed`` is filled only when a new text was handed over.
    """

    corpus: CorpusName
    weighting: WeightingName
    dimension: int
    n_terms: int
    n_documents: int
    rank: int
    components: list[ComponentReport]
    kept_share: float
    word_points: list[WordPoint]
    document_points: list[DocumentPoint]
    nearest_words: list[NeighbourWord]
    nearest_documents: list[NeighbourDocument]
    within_similarity: float | None
    across_similarity: float | None
    numbers_in_the_table: int
    numbers_kept: int
    reconstruction_error: float
    discarded_energy: float
    seconds: float
    placed: list[float] | None
    placed_group: str | None


@cache
def _components(
    corpus: CorpusName, weighting: WeightingName
) -> tuple[ComponentReport, ...]:
    """Every direction the table has, with how the two halves sit on each.

    The full-rank fit is what knows the shares, since a truncated one has
    already discarded the values it would need; the coordinates come from the
    same fit, and a truncated fit's leading coordinates agree with it.
    """
    largest = _largest_rank(corpus)
    full = _fitted(corpus, weighting, largest)
    documents = np.asarray(full.document_vectors.vectors)
    shares = np.asarray(full.variance_shares)
    cumulative = np.cumsum(shares)
    split = 2 if corpus == "four" else N_COOKING_DOCUMENTS

    reports: list[ComponentReport] = []
    for position in range(largest):
        column = documents[:, position]
        first_mean = float(column[:split].mean())
        second_mean = float(column[split:].mean())
        reports.append(
            ComponentReport(
                position=position + 1,
                singular_value=float(full.singular_values[position]),
                share=float(shares[position]),
                cumulative=float(cumulative[position]),
                first_group_mean=first_mean,
                second_group_mean=second_mean,
                group_difference=first_mean - second_mean,
                all_one_sign=bool(np.all(column > 0.0) or np.all(column < 0.0)),
            )
        )
    return tuple(reports)


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Decompose one corpus's table and report what the coordinates say."""
    corpus = request.corpus
    dimension = _checked_dimension(corpus, request.dimension)
    texts = _CORPORA[corpus]
    if request.document >= len(texts):
        raise HTTPException(
            status_code=422,
            detail=(
                f"this corpus holds {len(texts)} documents, so there is no "
                f"document {request.document}"
            ),
        )

    started = time.perf_counter()
    try:
        model = _fitted(corpus, request.weighting, dimension)
        seconds = time.perf_counter() - started
        words = list(model.vocabulary)
        if request.word not in words:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the word {request.word} does not appear in this corpus, "
                    f"so it has no position"
                ),
            )
        nearest_words = model.most_similar(request.word, request.n_results)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    documents = model.document_vectors
    matrix = np.asarray(model.term_document_matrix.values)
    rebuilt = model.decomposition.reconstruction()
    discarded = model.decomposition.total_squared_singular_values - float(
        np.sum(np.asarray(model.singular_values) ** 2)
    )

    nearest_documents = [
        NeighbourDocument(
            position=int(position),
            text=texts[int(position)],
            group=_document_group(corpus, int(position)),
            similarity=documents.similarity(request.document, int(position)),
        )
        for position in documents.most_similar(request.document, request.n_results)
    ]

    within: float | None = None
    across: float | None = None
    if corpus == "twenty-four":
        within, across = _within_and_across(model)

    placed: list[float] | None = None
    placed_group: str | None = None
    if request.new_text.strip():
        try:
            folded = model.transform([request.new_text])
        except MLLibError as refusal:
            raise HTTPException(
                status_code=422, detail=_in_plain_words(refusal)
            ) from refusal
        coordinates = np.asarray(folded.vector_of(0))
        placed = [float(value) for value in coordinates]
        if dimension >= 2 and float(np.linalg.norm(coordinates)) > 0.0:
            first_document_side = float(documents.vectors[0, 1])
            same_side = np.sign(placed[1]) == np.sign(first_document_side)
            placed_group = _document_group(corpus, 0 if same_side else len(texts) - 1)

    return FitResponse(
        corpus=corpus,
        weighting=request.weighting,
        dimension=dimension,
        n_terms=model.term_document_matrix.n_terms,
        n_documents=model.term_document_matrix.n_documents,
        rank=int(np.linalg.matrix_rank(matrix)),
        components=list(_components(corpus, request.weighting)),
        kept_share=float(sum(model.variance_shares)),
        word_points=[
            WordPoint(
                word=word,
                group=_GROUP_OF[corpus].get(word, "shared"),
                coordinates=[
                    float(value)
                    for value in np.asarray(model.embeddings.vector_of(word).values)
                ],
            )
            for word in words
        ],
        document_points=[
            DocumentPoint(
                position=position,
                text=text,
                group=_document_group(corpus, position),
                coordinates=[
                    float(value) for value in np.asarray(documents.vector_of(position))
                ],
            )
            for position, text in enumerate(texts)
        ],
        nearest_words=[
            NeighbourWord(
                word=similar.word,
                group=_GROUP_OF[corpus].get(similar.word, "shared"),
                similarity=float(similar.similarity),
            )
            for similar in nearest_words
        ],
        nearest_documents=nearest_documents,
        within_similarity=within,
        across_similarity=across,
        numbers_in_the_table=int(matrix.size),
        numbers_kept=int(
            model.embeddings.table.size + documents.vectors.size + dimension
        ),
        reconstruction_error=float(np.sum((rebuilt - matrix) ** 2)),
        discarded_energy=float(discarded),
        seconds=seconds,
        placed=placed,
        placed_group=placed_group,
    )


# --- What the squeeze fills in -----------------------------------------------


class ReconstructionRequest(BaseModel):
    """Which corpus to rebuild at which width, and whose row to print."""

    corpus: CorpusName = "twenty-four"
    weighting: WeightingName = "weighted"
    dimension: int = Field(default=PAGE_DIMENSION, ge=1, le=MAX_DIMENSION)
    word: str = Field(default="flour", min_length=1, max_length=32)

    model_config = {"extra": "forbid"}


class ReconstructionResponse(BaseModel):
    """The table before and after, and what the empty cells became.

    ``empty_cell_mean`` is the mean of the rebuilt table over exactly the cells
    the corpus left at zero, which is where the method puts a word it never saw
    in a document.
    """

    corpus: CorpusName
    dimension: int
    words: list[str]
    documents: list[str]
    document_groups: list[str]
    original: list[list[float]]
    rebuilt: list[list[float]]
    squared_error: float
    discarded_energy: float
    kept_share: float
    n_negative: int
    most_negative: float
    empty_cell_mean: float
    empty_cell_largest: float
    row_word: str
    row_original: list[float]
    row_rebuilt: list[float]
    row_seen_in: int
    row_filled_in: int


@router.post("/reconstruction", response_model=ReconstructionResponse)
def reconstruction(request: ReconstructionRequest) -> ReconstructionResponse:
    """Rebuild the table from the directions kept and compare it with the original."""
    corpus = request.corpus
    dimension = _checked_dimension(corpus, request.dimension)
    model = _fitted(corpus, request.weighting, dimension)
    words = list(model.vocabulary)
    if request.word not in words:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the word {request.word} does not appear in this corpus, so "
                f"it has no row"
            ),
        )

    matrix = np.asarray(model.term_document_matrix.values)
    rebuilt = model.decomposition.reconstruction()
    empty = matrix == 0.0
    row = words.index(request.word)
    discarded = model.decomposition.total_squared_singular_values - float(
        np.sum(np.asarray(model.singular_values) ** 2)
    )

    return ReconstructionResponse(
        corpus=corpus,
        dimension=dimension,
        words=words,
        documents=_CORPORA[corpus],
        document_groups=[
            _document_group(corpus, position)
            for position in range(len(_CORPORA[corpus]))
        ],
        original=[[float(value) for value in line] for line in matrix],
        rebuilt=[[float(value) for value in line] for line in rebuilt],
        squared_error=float(np.sum((rebuilt - matrix) ** 2)),
        discarded_energy=float(discarded),
        kept_share=float(sum(model.variance_shares)),
        n_negative=int((rebuilt < -1e-9).sum()),
        most_negative=float(rebuilt.min()),
        empty_cell_mean=float(rebuilt[empty].mean()),
        empty_cell_largest=float(rebuilt[empty].max()),
        row_word=request.word,
        row_original=[float(value) for value in matrix[row]],
        row_rebuilt=[float(value) for value in rebuilt[row]],
        row_seen_in=int((matrix[row] > 0.0).sum()),
        row_filled_in=int((rebuilt[row] > 0.1 * float(rebuilt[row].max())).sum()),
    )


# --- What the raw table cannot say -------------------------------------------


class PairReading(BaseModel):
    """One pair, read off the raw table and off two fits."""

    first: str
    second: str
    raw: float
    narrow: float
    wide: float


class WidthRow(BaseModel):
    """One width, and how far apart it holds the two halves.

    ``within_smallest`` is the smallest cosine between two words of one half,
    which is what a hard squeeze destroys and a raw table keeps.
    """

    label: str
    dimension: int | None
    kept_share: float | None
    within_mean: float
    within_smallest: float
    within_largest: float
    across_mean: float
    gap: float


class SharpnessResponse(BaseModel):
    """Where the raw counts fail, and what the squeeze costs to repair it."""

    document_pairs: list[PairReading]
    crossing_pair: PairReading
    word_pairs: list[PairReading]
    n_document_pairs: int
    n_document_pairs_at_zero: int
    n_across_document_pairs: int
    n_across_document_pairs_at_zero: int
    n_word_pairs: int
    n_word_pairs_at_zero: int
    widths: list[WidthRow]
    narrow_dimension: int
    wide_dimension: int


@cache
def _sharpness() -> SharpnessResponse:
    matrix = np.asarray(_matrix("twenty-four", "weighted").values)
    words = list(_vocabulary("twenty-four"))
    texts = TWENTY_FOUR_DOCUMENTS
    narrow = _fitted("twenty-four", "weighted", PAGE_DIMENSION)
    wide = _fitted("twenty-four", "weighted", WIDER_DIMENSION)

    document_pairs = [
        PairReading(
            first=texts[first],
            second=texts[second],
            raw=_cosine(matrix[:, first], matrix[:, second]),
            narrow=narrow.document_vectors.similarity(first, second),
            wide=wide.document_vectors.similarity(first, second),
        )
        for first, second in ((0, 5), (1, 6), (2, 7))
    ]
    crossing = PairReading(
        first=texts[0],
        second=texts[N_COOKING_DOCUMENTS],
        raw=_cosine(matrix[:, 0], matrix[:, N_COOKING_DOCUMENTS]),
        narrow=narrow.document_vectors.similarity(0, N_COOKING_DOCUMENTS),
        wide=wide.document_vectors.similarity(0, N_COOKING_DOCUMENTS),
    )

    word_pairs: list[PairReading] = []
    for position, first in enumerate(COOKING_WORDS):
        for second in COOKING_WORDS[position + 1 :]:
            raw = _cosine(matrix[words.index(first)], matrix[words.index(second)])
            if raw == 0.0:
                word_pairs.append(
                    PairReading(
                        first=first,
                        second=second,
                        raw=raw,
                        narrow=narrow.similarity(first, second),
                        wide=wide.similarity(first, second),
                    )
                )
    word_pairs.append(
        PairReading(
            first="flour",
            second="anchor",
            raw=_cosine(matrix[words.index("flour")], matrix[words.index("anchor")]),
            narrow=narrow.similarity("flour", "anchor"),
            wide=wide.similarity("flour", "anchor"),
        )
    )

    within_documents = [
        _cosine(matrix[:, first], matrix[:, second])
        for group in (range(N_COOKING_DOCUMENTS), range(N_COOKING_DOCUMENTS, 24))
        for position, first in enumerate(group)
        for second in list(group)[position + 1 :]
    ]
    across_documents = [
        _cosine(matrix[:, first], matrix[:, second])
        for first in range(N_COOKING_DOCUMENTS)
        for second in range(N_COOKING_DOCUMENTS, 24)
    ]
    within_words = [
        _cosine(matrix[words.index(first)], matrix[words.index(second)])
        for group in (COOKING_WORDS, SAILING_WORDS)
        for position, first in enumerate(group)
        for second in group[position + 1 :]
    ]
    across_words = [
        _cosine(matrix[words.index(first)], matrix[words.index(second)])
        for first in COOKING_WORDS
        for second in SAILING_WORDS
    ]

    widths = [
        WidthRow(
            label="the table itself",
            dimension=None,
            kept_share=None,
            within_mean=float(np.mean(within_words)),
            within_smallest=float(np.min(within_words)),
            within_largest=float(np.max(within_words)),
            across_mean=float(np.mean(across_words)),
            gap=float(np.mean(within_words)) - float(np.mean(across_words)),
        )
    ]
    for dimension in (2, 3, 4, 6, 12, MAX_DIMENSION):
        model = _fitted("twenty-four", "weighted", dimension)
        inside = [
            model.similarity(first, second)
            for group in (COOKING_WORDS, SAILING_WORDS)
            for position, first in enumerate(group)
            for second in group[position + 1 :]
        ]
        between = [
            model.similarity(first, second)
            for first in COOKING_WORDS
            for second in SAILING_WORDS
        ]
        widths.append(
            WidthRow(
                label=f"{dimension} directions kept",
                dimension=dimension,
                kept_share=float(sum(model.variance_shares)),
                within_mean=float(np.mean(inside)),
                within_smallest=float(np.min(inside)),
                within_largest=float(np.max(inside)),
                across_mean=float(np.mean(between)),
                gap=float(np.mean(inside)) - float(np.mean(between)),
            )
        )

    return SharpnessResponse(
        document_pairs=document_pairs,
        crossing_pair=crossing,
        word_pairs=word_pairs,
        n_document_pairs=len(within_documents),
        n_document_pairs_at_zero=sum(1 for value in within_documents if value == 0.0),
        n_across_document_pairs=len(across_documents),
        n_across_document_pairs_at_zero=sum(
            1 for value in across_documents if value == 0.0
        ),
        n_word_pairs=len(within_words),
        n_word_pairs_at_zero=sum(1 for value in within_words if value == 0.0),
        widths=widths,
        narrow_dimension=PAGE_DIMENSION,
        wide_dimension=WIDER_DIMENSION,
    )


@router.get("/sharpness", response_model=SharpnessResponse)
def sharpness() -> SharpnessResponse:
    """What two texts about one thing in different words score, before and after."""
    return _sharpness()


# --- The leading direction, with the shift and without -----------------------


class DirectionResponse(BaseModel):
    """Every document's coordinate on the leading direction, two ways.

    ``as_counted`` decomposes the table as it stands, which is the method;
    ``after_shifting`` subtracts each word's mean across the documents first,
    which is what the decomposition on the neighbouring page does to its own
    table and is here only for the contrast.
    """

    documents: list[str]
    document_groups: list[str]
    as_counted_first: list[float]
    as_counted_second: list[float]
    after_shifting_first: list[float]
    as_counted_first_means: list[float]
    as_counted_first_difference: float
    as_counted_second_means: list[float]
    as_counted_second_difference: float
    after_shifting_means: list[float]
    after_shifting_difference: float
    as_counted_shares: list[float]
    after_shifting_shares: list[float]
    shared_words: list[str]
    shared_word_second_coordinates: list[float]
    smallest_first_coordinate: float
    largest_first_coordinate: float


@cache
def _directions() -> DirectionResponse:
    model = _fitted("twenty-four", "weighted", 3)
    documents = np.asarray(model.document_vectors.vectors)
    matrix = np.asarray(_matrix("twenty-four", "weighted").values)
    shifted = matrix - matrix.mean(axis=1, keepdims=True)
    decomposition = TruncatedSingularValueDecomposition.of(shifted, 3)
    shifted_documents = np.asarray(
        decomposition.right_vectors * decomposition.singular_values
    )
    split = N_COOKING_DOCUMENTS

    return DirectionResponse(
        documents=TWENTY_FOUR_DOCUMENTS,
        document_groups=[
            _document_group("twenty-four", position) for position in range(24)
        ],
        as_counted_first=[float(value) for value in documents[:, 0]],
        as_counted_second=[float(value) for value in documents[:, 1]],
        after_shifting_first=[float(value) for value in shifted_documents[:, 0]],
        as_counted_first_means=[
            float(documents[:split, 0].mean()),
            float(documents[split:, 0].mean()),
        ],
        as_counted_first_difference=float(
            documents[:split, 0].mean() - documents[split:, 0].mean()
        ),
        as_counted_second_means=[
            float(documents[:split, 1].mean()),
            float(documents[split:, 1].mean()),
        ],
        as_counted_second_difference=float(
            documents[:split, 1].mean() - documents[split:, 1].mean()
        ),
        after_shifting_means=[
            float(shifted_documents[:split, 0].mean()),
            float(shifted_documents[split:, 0].mean()),
        ],
        after_shifting_difference=float(
            shifted_documents[:split, 0].mean() - shifted_documents[split:, 0].mean()
        ),
        as_counted_shares=[float(share) for share in model.variance_shares],
        after_shifting_shares=[float(share) for share in decomposition.variance_shares],
        shared_words=list(SHARED_WORDS),
        shared_word_second_coordinates=[
            float(np.asarray(model.embeddings.vector_of(word).values)[1])
            for word in SHARED_WORDS
        ],
        smallest_first_coordinate=float(documents[:, 0].min()),
        largest_first_coordinate=float(documents[:, 0].max()),
    )


@router.get("/directions", response_model=DirectionResponse)
def directions() -> DirectionResponse:
    """Where the documents fall on the leading direction, shifted and not."""
    return _directions()


# --- One spelling, two senses ------------------------------------------------

TWO_SENSE_WORD = "roll"
"""The spelling given two senses: a bread roll, and the roll of a boat."""

REPLACED_IN_COOKING = "dough"
REPLACED_IN_SAILING = "rope"
"""The two words it stands in for, one in each half."""


class SenseResponse(BaseModel):
    """Where a spelling used two ways ends up, against the words it stands for.

    ``coordinates`` are on the two kept directions, so the second number is the
    one that says which half a word belongs to.
    """

    word: str
    coordinates: list[float]
    second_coordinate: float
    nearest: list[NeighbourWord]
    similarity_to_cooking_word: float
    similarity_to_sailing_word: float
    similarity_to_a_shared_word: float
    cooking_example: str
    sailing_example: str
    shared_example: str
    cooking_word_second_coordinate: float
    sailing_word_second_coordinate: float
    documents: list[str]


@cache
def _senses() -> SenseResponse:
    cooking = [
        text.replace(REPLACED_IN_COOKING, TWO_SENSE_WORD) for text in COOKING_DOCUMENTS
    ]
    sailing = [
        text.replace(REPLACED_IN_SAILING, TWO_SENSE_WORD) for text in SAILING_DOCUMENTS
    ]
    corpus = cooking + sailing
    model = LatentSemanticAnalysis(
        dimension=PAGE_DIMENSION,
        weighting=TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
    ).fit(corpus)
    coordinates = np.asarray(model.embeddings.vector_of(TWO_SENSE_WORD).values)

    return SenseResponse(
        word=TWO_SENSE_WORD,
        coordinates=[float(value) for value in coordinates],
        second_coordinate=float(coordinates[1]),
        nearest=[
            NeighbourWord(
                word=similar.word,
                group=_GROUP_OF["twenty-four"].get(similar.word, "shared"),
                similarity=float(similar.similarity),
            )
            for similar in model.most_similar(TWO_SENSE_WORD, 6)
        ],
        similarity_to_cooking_word=model.similarity(TWO_SENSE_WORD, "flour"),
        similarity_to_sailing_word=model.similarity(TWO_SENSE_WORD, "sail"),
        similarity_to_a_shared_word=model.similarity(TWO_SENSE_WORD, "the"),
        cooking_example="flour",
        sailing_example="sail",
        shared_example="the",
        cooking_word_second_coordinate=float(
            np.asarray(model.embeddings.vector_of("flour").values)[1]
        ),
        sailing_word_second_coordinate=float(
            np.asarray(model.embeddings.vector_of("sail").values)[1]
        ),
        documents=corpus,
    )


@router.get("/senses", response_model=SenseResponse)
def senses() -> SenseResponse:
    """One spelling standing for two things, and where the fit puts it."""
    return _senses()
