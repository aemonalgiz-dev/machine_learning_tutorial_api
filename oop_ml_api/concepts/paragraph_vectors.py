"""Paragraph Vectors.

Every endpoint here is registered under ``/concepts/paragraph-vectors``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about ParagraphVectors.

The running collection is the one the library's own spec fits: thirty
eight-word documents, fifteen drawn from a list of ten cooking words and
fifteen from a list of ten sailing words, with no word in both lists, under a
fixed seed. It is reproduced here rather than imported, because ``test`` is not
part of the installed package, and a test asserts the two agree document for
document.

Two quantities are computed here rather than in the library and both are
definitions rather than fits. The mean cosine inside a topic and between the
topics is arithmetic over the pairs of a table the library produced, and the
comparison against averaging a document's word vectors runs the library's own
pooling on the library's own word table. Everything that is a fit, a draw or a
descent is the library's.

Nothing a widget renders may name a class, a parameter or a module, so the two
architectures cross the wire under Le and Mikolov's published names, shortened
to ``distributed-memory`` and ``bag-of-words``, and ``_ARCHITECTURES`` is the
only place those meet the library's own. A refusal is rewritten by
``_in_plain_words`` before it can reach a widget, and a test asserts no
identifier survives that rewriting.
"""

from __future__ import annotations

import statistics
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings.documents.pooling import (
    MeanPooling,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.paragraph_vectors import (
    ParagraphArchitecture,
    ParagraphVectors,
    averaged_hidden,
    shared_input_step,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import cosine_similarity
from oop_ml.core.types import FloatArray
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/paragraph-vectors", tags=["paragraph-vectors"])


# --- The collection ----------------------------------------------------------

COOKING_WORDS: tuple[str, ...] = (
    "simmer",
    "onion",
    "garlic",
    "butter",
    "saucepan",
    "whisk",
    "flour",
    "season",
    "ladle",
    "broth",
)
"""One list: ten words about cooking."""

SAILING_WORDS: tuple[str, ...] = (
    "mainsail",
    "rudder",
    "harbour",
    "anchor",
    "keel",
    "tide",
    "starboard",
    "rigging",
    "mooring",
    "gust",
)
"""The other list: ten words about sailing, sharing nothing with the first."""

N_PER_TOPIC = 15
"""How many documents each list produces."""

WORDS_PER_DOCUMENT = 8
"""How long each document is."""

CORPUS_SEED = 7
"""The seed the documents are drawn under, so the collection is a function of
nothing but this code."""


def _topic_documents(
    words: tuple[str, ...], generator: np.random.Generator
) -> list[str]:
    """Fifteen eight-word documents drawn with replacement from one list."""
    return [
        " ".join(str(word) for word in generator.choice(words, WORDS_PER_DOCUMENT))
        for _ in range(N_PER_TOPIC)
    ]


def _two_topic_collection() -> list[str]:
    """The cooking documents, then the sailing ones, in the spec's own order."""
    generator = np.random.default_rng(CORPUS_SEED)
    cooking = _topic_documents(COOKING_WORDS, generator)
    sailing = _topic_documents(SAILING_WORDS, generator)
    return [*cooking, *sailing]


TWO_TOPICS: list[str] = _two_topic_collection()
"""Thirty documents: the first fifteen about cooking, the last fifteen sailing."""

HELD_OUT_COOKING = "butter garlic onion simmer broth whisk flour ladle"
"""A cooking text the fit never saw, and the one every inference figure uses."""

HELD_OUT_SAILING = "anchor tide rudder harbour keel gust mooring mainsail"
"""The same, about the other subject."""

SHUFFLED_COOKING = "flour simmer whisk butter ladle onion broth garlic"
"""The held-out cooking text's own eight words in a different order."""


# --- The settings every fixed measurement uses -------------------------------

DIMENSION = 8
"""How many numbers a vector holds, word and document alike."""

WINDOW = 3
"""How many positions on each side of a word count as its context."""

EPOCHS = 30
"""How many passes over the collection a fit makes."""

LEARNING_RATE = 0.1
"""The step size a fit starts from."""

FIT_SEED = 0
"""The seed the tables are drawn under."""

PUBLISHED_INFERENCE_PASSES = 20
"""The number of passes the published work gives a new text."""

MAX_INFERENCE_PASSES = 400
"""The most passes one request may ask for.

Each pass is a Python loop over the text's positions, and four hundred passes
of an eight-word text already takes about half a second, which is at the edge
of what a reader waits for. The ceiling keeps a crafted request from turning a
teaching endpoint into a compute job.
"""

MAX_TEXT_LENGTH = 200
"""The longest text a reader may hand in, in characters."""

INFERENCE_PASS_COUNTS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 400)
"""The sweep the inference curve is drawn over."""

TIMING_RUNS = 3
"""How many runs a reported time is the median of.

Three rather than one, so a single slow run does not become the number, and
three rather than more, because every run here is a whole fit and the endpoint
that reports the times is the one a reader waits on.
"""


# --- Translating the page's words into the library's -------------------------

ArchitectureName = Literal["distributed-memory", "bag-of-words"]

_ARCHITECTURES: dict[ArchitectureName, ParagraphArchitecture] = {
    "distributed-memory": ParagraphArchitecture.DISTRIBUTED_MEMORY,
    "bag-of-words": ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS,
}

ARCHITECTURE_NAMES: tuple[ArchitectureName, ...] = (
    "distributed-memory",
    "bag-of-words",
)

ARCHITECTURE_LABELS: dict[ArchitectureName, str] = {
    "distributed-memory": "distributed memory",
    "bag-of-words": "distributed bag of words",
}

COOKING = "cooking"
SAILING = "sailing"


def topic_of_position(position: int) -> str:
    """Which list the document at this position was drawn from."""
    return COOKING if position < N_PER_TOPIC else SAILING


# --- Rewriting a refusal in the page's own words ------------------------------

_PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("n_results", "the number of neighbours asked for"),
    ("inference_epochs", "the number of passes"),
    ("random_seed", "the starting draw"),
    ("minimum_count", "the count a word must reach"),
    ("n_averaged", "the number of vectors averaged"),
)


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the few that name a parameter and leaves the rest alone. A test
    asserts that no identifier survives, so a message that grows one later fails
    there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in _PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _refuse(refusal: MLLibError) -> HTTPException:
    return HTTPException(status_code=422, detail=_in_plain_words(refusal))


# --- Shared pieces ------------------------------------------------------------


@cache
def _fitted(
    architecture: ArchitectureName,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    train_words: bool = False,
) -> ParagraphVectors:
    """One fit of the running collection, with the page's words translated."""
    return ParagraphVectors(
        dimension=DIMENSION,
        window=WINDOW,
        epochs=epochs,
        learning_rate=learning_rate,
        inference_epochs=PUBLISHED_INFERENCE_PASSES,
        random_seed=FIT_SEED,
        architecture=_ARCHITECTURES[architecture],
        train_words=train_words,
        minimum_count=1,
    ).fit(TWO_TOPICS)


@cache
def _fit_seconds(architecture: ArchitectureName) -> float:
    """The median of three fits, so one slow run does not become the number."""
    times: list[float] = []
    for _ in range(TIMING_RUNS):
        started = time.perf_counter()
        ParagraphVectors(
            dimension=DIMENSION,
            window=WINDOW,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            random_seed=FIT_SEED,
            architecture=_ARCHITECTURES[architecture],
            minimum_count=1,
        ).fit(TWO_TOPICS)
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def _given_passes(model: ParagraphVectors, passes: int) -> ParagraphVectors:
    """The same fitted model, asked to run a different number of passes.

    A copy carrying the same frozen tables, so nothing is refitted and the
    original is untouched. The number of passes is the one field that changes,
    and the sweep endpoint's own test pins that a one-pass answer and a
    two-hundred-pass answer differ, which is what would fail if the copy ever
    stopped taking the new value.
    """
    return model.model_copy(update={"inference_epochs": passes})


def _document_table(model: ParagraphVectors) -> FloatArray:
    return np.asarray(model.document_vectors.vectors)


def _pair_summary(vectors: FloatArray) -> dict[str, float]:
    """Mean cosine inside each topic, between them, and the two worst cases."""
    cooking = [
        float(cosine_similarity(vectors[first], vectors[second]))
        for first in range(N_PER_TOPIC)
        for second in range(first + 1, N_PER_TOPIC)
    ]
    sailing = [
        float(cosine_similarity(vectors[first], vectors[second]))
        for first in range(N_PER_TOPIC, 2 * N_PER_TOPIC)
        for second in range(first + 1, 2 * N_PER_TOPIC)
    ]
    across = [
        float(cosine_similarity(vectors[first], vectors[second]))
        for first in range(N_PER_TOPIC)
        for second in range(N_PER_TOPIC, 2 * N_PER_TOPIC)
    ]
    within = cooking + sailing
    return {
        "cooking": float(np.mean(cooking)),
        "sailing": float(np.mean(sailing)),
        "within": float(np.mean(within)),
        "across": float(np.mean(across)),
        "worst_within": float(np.min(within)),
        "best_across": float(np.max(across)),
        "margin": float(np.min(within) - np.max(across)),
    }


def _topic_means(vector: FloatArray, table: FloatArray) -> tuple[float, float]:
    """A vector's mean cosine to the cooking documents and to the sailing ones."""
    cooking = float(
        np.mean(
            [
                cosine_similarity(vector, table[position])
                for position in range(N_PER_TOPIC)
            ]
        )
    )
    sailing = float(
        np.mean(
            [
                cosine_similarity(vector, table[position])
                for position in range(N_PER_TOPIC, 2 * N_PER_TOPIC)
            ]
        )
    )
    return cooking, sailing


class TopicScores(BaseModel):
    """A position's mean cosine to each of the two halves of the collection."""

    cooking: float
    sailing: float
    leans_towards: str


def _topic_scores(vector: FloatArray, table: FloatArray) -> TopicScores:
    cooking, sailing = _topic_means(vector, table)
    return TopicScores(
        cooking=cooking,
        sailing=sailing,
        leans_towards=COOKING if cooking >= sailing else SAILING,
    )


# --- The two fits, and what they produced ------------------------------------


class DocumentEntry(BaseModel):
    """One document of the collection, in the order it was fitted."""

    position: int
    text: str
    topic: str


class EpochReport(BaseModel):
    """What one pass over the collection cost."""

    epoch: int
    mean_loss: float
    n_pairs: int


class RowLengths(BaseModel):
    """How long a family of rows ended up, which is how far they were stepped."""

    smallest: float
    largest: float
    mean: float


class ArchitectureReport(BaseModel):
    """One architecture's fit, and everything the page quotes from it.

    ``similarities`` is the whole thirty by thirty table of cosines between
    fitted documents, so a widget can draw it without asking again.
    ``fourteen_nearest_all_cooking`` says whether the fourteen documents nearest
    the first cooking one are the other fourteen cooking documents.
    """

    architecture: ArchitectureName
    label: str
    epochs: list[EpochReport]
    summary: dict[str, float]
    similarities: list[list[float]]
    fourteen_nearest_all_cooking: bool
    word_lengths: RowLengths
    document_lengths: RowLengths
    word_numbers: int
    document_numbers: int
    output_numbers: int
    word_vectors_moved: bool
    seconds: float
    near_pair: float
    far_pair: float


class ReferenceDefaults(BaseModel):
    """The same collection under the published rate and pass count.

    The two topics do not separate at all there, which is why every other
    number on this page comes from a longer fit at a larger rate.
    """

    epochs: int
    learning_rate: float
    positions: int
    first_loss: float
    last_loss: float
    summary: dict[str, float]


class FitsResponse(BaseModel):
    """Both architectures fitted, the collection they were fitted to, and the
    weaker fit the published settings produce."""

    documents: list[DocumentEntry]
    n_documents: int
    n_words: int
    n_occurrences: int
    dimension: int
    window: int
    epochs: int
    learning_rate: float
    untrained_cost: float
    architectures: list[ArchitectureReport]
    reference_defaults: ReferenceDefaults
    bag_of_words_with_word_pairs_near: float
    bag_of_words_with_word_pairs_far: float


def _epoch_reports(model: ParagraphVectors) -> list[EpochReport]:
    return [
        EpochReport(
            epoch=record.epoch, mean_loss=record.mean_loss, n_pairs=record.n_pairs
        )
        for record in model.history
    ]


def _row_lengths(table: FloatArray) -> RowLengths:
    norms = np.linalg.norm(table, axis=1)
    return RowLengths(
        smallest=float(norms.min()),
        largest=float(norms.max()),
        mean=float(norms.mean()),
    )


def _architecture_report(architecture: ArchitectureName) -> ArchitectureReport:
    model = _fitted(architecture)
    table = _document_table(model)
    words = np.asarray(model.embeddings.table)
    similarities = [
        [
            1.0
            if first == second
            else float(cosine_similarity(table[first], table[second]))
            for second in range(len(table))
        ]
        for first in range(len(table))
    ]
    nearest = model.most_similar_documents(0, 14)
    return ArchitectureReport(
        architecture=architecture,
        label=ARCHITECTURE_LABELS[architecture],
        epochs=_epoch_reports(model),
        summary=_pair_summary(table),
        similarities=similarities,
        fourteen_nearest_all_cooking=all(
            position < N_PER_TOPIC for position in nearest
        ),
        word_lengths=_row_lengths(words),
        document_lengths=_row_lengths(table),
        word_numbers=int(words.size),
        document_numbers=int(table.size),
        output_numbers=int(np.asarray(model.output_vectors).size),
        word_vectors_moved=bool(model.word_vectors_were_trained),
        seconds=_fit_seconds(architecture),
        near_pair=float(model.embeddings.similarity("butter", "garlic")),
        far_pair=float(model.embeddings.similarity("butter", "anchor")),
    )


@cache
def _fits_response() -> FitsResponse:
    weak = _fitted("distributed-memory", epochs=10, learning_rate=0.025)
    weak_losses = list(weak.history.mean_losses)
    with_word_pairs = _fitted(
        "bag-of-words", epochs=10, learning_rate=LEARNING_RATE, train_words=True
    )
    reference = _fitted("distributed-memory")
    return FitsResponse(
        documents=[
            DocumentEntry(
                position=position, text=text, topic=topic_of_position(position)
            )
            for position, text in enumerate(TWO_TOPICS)
        ],
        n_documents=reference.n_documents,
        n_words=reference.embeddings.n_words,
        n_occurrences=len(TWO_TOPICS) * WORDS_PER_DOCUMENT,
        dimension=DIMENSION,
        window=WINDOW,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        untrained_cost=float(6.0 * np.log(2.0)),
        architectures=[
            _architecture_report(architecture) for architecture in ARCHITECTURE_NAMES
        ],
        reference_defaults=ReferenceDefaults(
            epochs=10,
            learning_rate=0.025,
            positions=sum(record.n_pairs for record in weak.history),
            first_loss=weak_losses[0],
            last_loss=weak_losses[-1],
            summary=_pair_summary(_document_table(weak)),
        ),
        bag_of_words_with_word_pairs_near=float(
            with_word_pairs.embeddings.similarity("butter", "garlic")
        ),
        bag_of_words_with_word_pairs_far=float(
            with_word_pairs.embeddings.similarity("butter", "anchor")
        ),
    )


@router.get("/fits", response_model=FitsResponse)
def fits() -> FitsResponse:
    """Both architectures fitted to the running collection, cached."""
    return _fits_response()


# --- One text, inferred on request -------------------------------------------


class InferRequest(BaseModel):
    """A text the fit never saw, and how many passes to give it."""

    architecture: ArchitectureName = "distributed-memory"
    text: str = Field(default=HELD_OUT_COOKING, max_length=MAX_TEXT_LENGTH)
    passes: int = Field(
        default=PUBLISHED_INFERENCE_PASSES, ge=1, le=MAX_INFERENCE_PASSES
    )
    random_seed: int = Field(default=0, ge=0, le=999)
    n_nearest: int = Field(default=5, ge=1, le=10)

    model_config = {"extra": "forbid"}


class NearDocument(BaseModel):
    """One fitted document near an inferred position."""

    position: int
    text: str
    topic: str
    similarity: float


class InferResponse(BaseModel):
    """Where the descent put a held-out text, and how far it had to walk.

    ``known_words`` are the text's words the collection used, and
    ``unknown_words`` the ones it did not, which contribute nothing because
    there is no row to read for them. ``nothing_to_descend_on`` is true when
    none of the words were known, in which case the position is the random
    start it began at.
    """

    architecture: ArchitectureName
    label: str
    passes: int
    known_words: list[str]
    unknown_words: list[str]
    nothing_to_descend_on: bool
    scores: TopicScores
    length: float
    nearest: list[NearDocument]
    seconds: float
    document_lengths: RowLengths


@router.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    """Give one held-out text a position by descent against the frozen tables."""
    model = _given_passes(_fitted(request.architecture), request.passes)
    vocabulary = model.embeddings.vocabulary
    words = list(model.pre_tokenizer.split(request.text).texts)
    known = [word for word in words if word in vocabulary]
    unknown = [word for word in words if word not in vocabulary]

    try:
        started = time.perf_counter()
        vector = model.infer_vector(request.text, random_seed=request.random_seed)
        seconds = time.perf_counter() - started
        table = _document_table(model)
        nearest = model.document_vectors.similar_to_vector(vector, request.n_nearest)
        neighbours = [
            NearDocument(
                position=position,
                text=TWO_TOPICS[position],
                topic=topic_of_position(position),
                similarity=float(cosine_similarity(vector, table[position])),
            )
            for position in nearest
        ]
    except MLLibError as refusal:
        raise _refuse(refusal) from refusal

    return InferResponse(
        architecture=request.architecture,
        label=ARCHITECTURE_LABELS[request.architecture],
        passes=request.passes,
        known_words=known,
        unknown_words=unknown,
        nothing_to_descend_on=not known,
        scores=_topic_scores(vector, table),
        length=float(np.linalg.norm(vector)),
        nearest=neighbours,
        seconds=seconds,
        document_lengths=_row_lengths(table),
    )


# --- How the answer moves with the number of passes ---------------------------


class CurvePoint(BaseModel):
    """One held-out text inferred at one number of passes."""

    passes: int
    cooking: float
    sailing: float
    length: float
    seconds: float
    leans_towards: str


class CurveSeries(BaseModel):
    """The whole sweep under one architecture."""

    architecture: ArchitectureName
    label: str
    points: list[CurvePoint]
    document_length_mean: float
    word_length_mean: float


class CurveResponse(BaseModel):
    """The sweep under both architectures, and the published pass count."""

    text: str
    published_passes: int
    series: list[CurveSeries]


@cache
def _curve_series(architecture: ArchitectureName) -> CurveSeries:
    fitted_model = _fitted(architecture)
    table = _document_table(fitted_model)
    points: list[CurvePoint] = []
    for passes in INFERENCE_PASS_COUNTS:
        model = _given_passes(fitted_model, passes)
        started = time.perf_counter()
        vector = model.infer_vector(HELD_OUT_COOKING)
        seconds = time.perf_counter() - started
        scores = _topic_scores(vector, table)
        points.append(
            CurvePoint(
                passes=passes,
                cooking=scores.cooking,
                sailing=scores.sailing,
                length=float(np.linalg.norm(vector)),
                seconds=seconds,
                leans_towards=scores.leans_towards,
            )
        )
    return CurveSeries(
        architecture=architecture,
        label=ARCHITECTURE_LABELS[architecture],
        points=points,
        document_length_mean=float(np.linalg.norm(table, axis=1).mean()),
        word_length_mean=float(
            np.linalg.norm(np.asarray(fitted_model.embeddings.table), axis=1).mean()
        ),
    )


@router.get("/inference-curve", response_model=CurveResponse)
def inference_curve() -> CurveResponse:
    """The held-out cooking text's position against the number of passes."""
    return CurveResponse(
        text=HELD_OUT_COOKING,
        published_passes=PUBLISHED_INFERENCE_PASSES,
        series=[_curve_series(name) for name in ARCHITECTURE_NAMES],
    )


# --- Against simply averaging the words --------------------------------------


class RuleReport(BaseModel):
    """One way of giving a document a position, scored the same way.

    ``seconds_to_prepare`` is what has to happen before a single text can be
    scored at all, and ``seconds_per_text`` what one held-out text then costs.
    """

    rule: str
    summary: dict[str, float]
    held_out: TopicScores
    seconds_to_prepare: float
    seconds_per_text: float
    numbers_per_document: int


class AveragingResponse(BaseModel):
    """The two architectures and the average of a document's word vectors."""

    rules: list[RuleReport]


@cache
def _averaging_response() -> AveragingResponse:
    rules: list[RuleReport] = []
    for architecture in ARCHITECTURE_NAMES:
        model = _fitted(architecture)
        table = _document_table(model)
        passes = 200 if architecture == "distributed-memory" else 20
        walking = _given_passes(model, passes)
        vector = walking.infer_vector(HELD_OUT_COOKING)
        times: list[float] = []
        for _ in range(TIMING_RUNS):
            started = time.perf_counter()
            walking.infer_vector(HELD_OUT_COOKING)
            times.append(time.perf_counter() - started)
        rules.append(
            RuleReport(
                rule=f"{ARCHITECTURE_LABELS[architecture]}, {passes} passes",
                summary=_pair_summary(table),
                held_out=_topic_scores(vector, table),
                seconds_to_prepare=_fit_seconds(architecture),
                seconds_per_text=statistics.median(times),
                numbers_per_document=DIMENSION,
            )
        )

    word_table_owner = _fitted("distributed-memory")
    started = time.perf_counter()
    pooling = MeanPooling(embeddings=word_table_owner.embeddings).fit(TWO_TOPICS)
    prepare_seconds = time.perf_counter() - started
    pooled = np.asarray(pooling.transform(TWO_TOPICS).vectors)
    held = np.asarray(pooling.transform([HELD_OUT_COOKING]).vectors)[0]
    times = []
    for _ in range(TIMING_RUNS):
        started = time.perf_counter()
        pooling.transform([HELD_OUT_COOKING])
        times.append(time.perf_counter() - started)
    rules.append(
        RuleReport(
            rule="the average of a document's word positions",
            summary=_pair_summary(pooled),
            held_out=_topic_scores(held, pooled),
            seconds_to_prepare=prepare_seconds,
            seconds_per_text=statistics.median(times),
            numbers_per_document=DIMENSION,
        )
    )
    return AveragingResponse(rules=rules)


@router.get("/against-averaging", response_model=AveragingResponse)
def against_averaging() -> AveragingResponse:
    """The learned positions and the averaged ones, scored the same way."""
    return _averaging_response()


# --- Does any of this read the order the words came in ------------------------


class OrderRow(BaseModel):
    """One rule, asked the same eight words twice in two orders.

    ``two_orders`` is the cosine between the answers for the two orderings from
    one starting draw, and the two ``two_starts`` figures are the cosine between
    two starting draws of the same ordering, which is the control: a rule reads
    order only where the first number is below the other two.
    """

    rule: str
    passes: int
    two_orders: float
    two_starts_first: float
    two_starts_second: float
    reads_order: bool


class OrderResponse(BaseModel):
    """Both orderings, both architectures, and the averaging rule."""

    text: str
    shuffled: str
    rows: list[OrderRow]


@cache
def _order_response() -> OrderResponse:
    rows: list[OrderRow] = []
    for architecture in ARCHITECTURE_NAMES:
        for passes in (PUBLISHED_INFERENCE_PASSES, 200):
            model = _given_passes(_fitted(architecture), passes)
            first_order = model.infer_vector(HELD_OUT_COOKING, random_seed=1)
            other_order = model.infer_vector(SHUFFLED_COOKING, random_seed=1)
            same_a = model.infer_vector(HELD_OUT_COOKING, random_seed=2)
            same_b = model.infer_vector(HELD_OUT_COOKING, random_seed=3)
            two_orders = float(cosine_similarity(first_order, other_order))
            starts_first = float(cosine_similarity(first_order, same_a))
            starts_second = float(cosine_similarity(same_a, same_b))
            rows.append(
                OrderRow(
                    rule=ARCHITECTURE_LABELS[architecture],
                    passes=passes,
                    two_orders=two_orders,
                    two_starts_first=starts_first,
                    two_starts_second=starts_second,
                    reads_order=two_orders < min(starts_first, starts_second),
                )
            )

    pooling = MeanPooling(embeddings=_fitted("distributed-memory").embeddings).fit(
        TWO_TOPICS
    )
    pooled = np.asarray(pooling.transform([HELD_OUT_COOKING, SHUFFLED_COOKING]).vectors)
    rows.append(
        OrderRow(
            rule="the average of a document's word positions",
            passes=0,
            two_orders=float(cosine_similarity(pooled[0], pooled[1])),
            two_starts_first=1.0,
            two_starts_second=1.0,
            reads_order=False,
        )
    )
    return OrderResponse(text=HELD_OUT_COOKING, shuffled=SHUFFLED_COOKING, rows=rows)


@router.get("/order", response_model=OrderResponse)
def order() -> OrderResponse:
    """Two orderings of one text against two starting draws of one ordering."""
    return _order_response()


# --- One position, small enough to check by hand ------------------------------


class HandWorkedResponse(BaseModel):
    """One position of one document, at a width of three.

    Every number here is the library's, so the arithmetic a reader checks with
    a pencil is the arithmetic the fit runs.
    """

    document_vector: list[float]
    context_vector: list[float]
    hidden: list[float]
    score: float
    probability: float
    hidden_gradient: list[float]
    learning_rate: float
    n_averaged: int
    step: list[float]


@router.get("/one-position", response_model=HandWorkedResponse)
def one_position() -> HandWorkedResponse:
    """The gradient of a single distributed-memory position, worked out."""
    document_vector = np.array([0.2, -0.1, 0.4])
    context = np.array([[0.0, 0.3, -0.2]])
    hidden = averaged_hidden(document_vector, context)
    target_row = np.array([1.0, 0.0, 0.0])
    negative_row = np.array([0.0, 1.0, 0.0])
    score = float(hidden @ target_row)
    probability = float(1.0 / (1.0 + np.exp(-score)))
    hidden_gradient = (probability - 1.0) * target_row + (
        probability - 0.0
    ) * negative_row
    learning_rate = 0.025
    step = shared_input_step(hidden_gradient, learning_rate, 2)
    return HandWorkedResponse(
        document_vector=[float(value) for value in document_vector],
        context_vector=[float(value) for value in context[0]],
        hidden=[float(value) for value in hidden],
        score=score,
        probability=probability,
        hidden_gradient=[float(value) for value in hidden_gradient],
        learning_rate=learning_rate,
        n_averaged=2,
        step=[float(value) for value in step],
    )
