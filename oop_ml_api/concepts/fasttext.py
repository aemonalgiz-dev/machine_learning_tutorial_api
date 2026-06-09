"""FastText.

Every endpoint here is registered under ``/concepts/fasttext``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about FastText.

The running corpus is the one the library's own FastText spec fits, two word
lists that never share a sentence: twenty verb forms and eleven money words,
drawn into two hundred sentences of five to eight words under a fixed seed. The
verb list deliberately leaves out ``playing`` while holding four other forms of
the same verb, so the page has a word that is absent from every sentence and
whose pieces were all seen elsewhere. It is reproduced here rather than
imported, because ``test`` is not part of the installed package, and a test
asserts the two agree word for word.

Nothing with any mathematics in it is computed here. The pieces, the buckets,
the composition, the fits and the refusals are all the library's; this module
picks a corpus, runs the probes, and counts. The one thing it does itself is
inverting the piece-to-word map so a widget can say which corpus words own a
piece, which is bookkeeping over the library's own answers.

Nothing a widget renders may name a class, a parameter or a module, so the
architectures cross the wire as the page's own words and every refusal a widget
shows goes through ``_in_plain_words``, with a test asserting no identifier
survives it.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from functools import cache

import numpy as np
from fastapi import APIRouter
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.embeddings.prediction.fasttext import (
    FastText,
    fowler_noll_vo_hash,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    Word2Vec,
    Word2VecArchitecture,
    Word2VecObjective,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field, model_validator

router = APIRouter(prefix="/concepts/fasttext", tags=["fasttext"])


# --- The corpus ------------------------------------------------------------

VERB_WORDS: tuple[str, ...] = (
    "play",
    "played",
    "plays",
    "player",
    "stay",
    "stayed",
    "stays",
    "staying",
    "say",
    "said",
    "says",
    "saying",
    "walk",
    "walked",
    "walks",
    "walking",
    "talk",
    "talked",
    "talks",
    "talking",
)
"""One list: twenty forms of five verbs, and ``playing`` is not among them."""

MONEY_WORDS: tuple[str, ...] = (
    "stock",
    "bond",
    "market",
    "price",
    "trade",
    "fund",
    "bank",
    "cash",
    "debt",
    "yield",
    "share",
)
"""The other list: eleven words about money, none sharing a piece of ``playing``."""


def _two_topic_sentences(n_per_list: int = 100, seed: int = 11) -> list[str]:
    """Sentences of five to eight words, each drawn from one list only.

    The lists alternate so neither sits at one end of the corpus, and the draw
    is seeded, so the corpus is a function of nothing but this code.
    """
    draw = random.Random(seed)
    sentences: list[str] = []
    for _ in range(n_per_list):
        for words in (VERB_WORDS, MONEY_WORDS):
            length = draw.randint(5, 8)
            sentences.append(" ".join(draw.choice(words) for _ in range(length)))
    return sentences


TWO_LIST_CORPUS: list[str] = _two_topic_sentences()
"""Two hundred sentences, a hundred drawn from each list."""

TOPIC_OF: dict[str, str] = {
    **{word: "verb" for word in VERB_WORDS},
    **{word: "money" for word in MONEY_WORDS},
}
"""Which list each word came from, for the widgets that colour by it."""

MISSING_FORM = "playing"
"""The form of a corpus verb that no sentence holds."""

HELD_OUT_WORD = "walking"
"""A word the corpus does hold, removed from it to make a second missing one."""

SHARELESS_WORD = "zap"
"""A word whose pieces land where nothing the fit saw ever landed."""


# --- The settings every fixed measurement uses -------------------------------

DEFAULT_DIMENSION = 12
"""The width every fixed measurement on the page uses."""

DEFAULT_WINDOW = 3
"""The reach every fixed measurement on the page uses."""

DEFAULT_RATE = 0.025
"""The published first step, which is what the fixed measurements use."""

DOUBLED_RATE = 0.05
"""Twice that, for the one comparison that varies it."""

DEFAULT_EPOCHS = 5
"""Passes over the corpus for every fixed measurement."""

DEFAULT_MINIMUM_LENGTH = 3
DEFAULT_MAXIMUM_LENGTH = 6
"""The published piece lengths."""

DEFAULT_BUCKETS = 2000
"""How many rows the pieces share here.

The published value is two million, which on a corpus of thirty one words would
never let two pieces meet; two thousand is small enough that the sharing is
visible and large enough that most pieces still get a row to themselves.
"""

NO_PIECE_LENGTH = 50
"""Longer than any wrapped word here, so a fit at this length has no piece."""

MAX_EPOCHS = 6
"""The most passes one interactive fit may make.

A pass is a Python loop over every position of two hundred sentences and costs
about half a second, so six is already longer than a reader waits.
"""

MAX_BUCKETS = 50_000
"""The largest table an interactive fit may allocate.

Rows cost nothing to hold and everything above a few thousand behaves
identically here, since there are 279 pieces to place; the bound stops a
crafted request turning a teaching endpoint into a memory job.
"""

MAX_LENGTH = 10
"""The longest piece a request may ask for, which is longer than any word here."""


# --- Shared pieces -----------------------------------------------------------


@cache
def _corpus_vocabulary() -> Vocabulary:
    """The corpus split into words, in the order a fit numbers them."""
    return TokenisedCorpus.from_texts(
        TWO_LIST_CORPUS, WhitespacePreTokenizer()
    ).vocabulary(1)


@cache
def _n_occurrences() -> int:
    return TokenisedCorpus.from_texts(TWO_LIST_CORPUS, WhitespacePreTokenizer()).n_words


@cache
def _owners(minimum_length: int, maximum_length: int) -> dict[str, tuple[str, ...]]:
    """Which corpus words own each piece, at one pair of lengths.

    The library answers a word's pieces; this inverts that map, so a widget can
    show a piece beside the words it came from.
    """
    shape = FastText(
        minimum_n_gram_length=minimum_length, maximum_n_gram_length=maximum_length
    )
    owners: dict[str, list[str]] = {}
    for word in _corpus_vocabulary():
        for piece in shape.n_grams_of(word):
            owners.setdefault(piece, []).append(word)
    return {piece: tuple(words) for piece, words in owners.items()}


def _fit_fasttext(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_RATE,
    minimum_length: int = DEFAULT_MINIMUM_LENGTH,
    maximum_length: int = DEFAULT_MAXIMUM_LENGTH,
    n_buckets: int = DEFAULT_BUCKETS,
    window: int = DEFAULT_WINDOW,
    architecture: Word2VecArchitecture = Word2VecArchitecture.SKIP_GRAM,
    objective: Word2VecObjective = Word2VecObjective.NEGATIVE_SAMPLING,
) -> FastText:
    return FastText(
        dimension=DEFAULT_DIMENSION,
        window=window,
        epochs=epochs,
        learning_rate=learning_rate,
        random_seed=0,
        architecture=architecture,
        objective=objective,
        minimum_n_gram_length=minimum_length,
        maximum_n_gram_length=maximum_length,
        n_buckets=n_buckets,
    ).fit(TWO_LIST_CORPUS)


def _fit_plain(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_RATE,
    architecture: Word2VecArchitecture = Word2VecArchitecture.SKIP_GRAM,
    objective: Word2VecObjective = Word2VecObjective.NEGATIVE_SAMPLING,
) -> Word2Vec:
    return Word2Vec(
        dimension=DEFAULT_DIMENSION,
        window=DEFAULT_WINDOW,
        epochs=epochs,
        learning_rate=learning_rate,
        random_seed=0,
        architecture=architecture,
        objective=objective,
    ).fit(TWO_LIST_CORPUS)


@cache
def _reference_fasttext() -> FastText:
    """The one fit every fixed measurement on the page reads."""
    return _fit_fasttext()


@cache
def _reference_plain() -> Word2Vec:
    """The same settings without the pieces, for every side-by-side reading."""
    return _fit_plain()


def _within_and_across(embeddings: WordEmbeddings) -> tuple[float, float]:
    """Mean cosine inside a list, and mean cosine between the two lists."""
    inside: list[float] = []
    for words in (VERB_WORDS, MONEY_WORDS):
        for position, first in enumerate(words):
            for second in words[position + 1 :]:
                inside.append(embeddings.similarity(first, second))
    between = [
        embeddings.similarity(first, second)
        for first in VERB_WORDS
        for second in MONEY_WORDS
    ]
    return float(np.mean(inside)), float(np.mean(between))


def _similarities_to(model: FastText, word: str, words: tuple[str, ...]) -> list[float]:
    vector = model.vector_of(word).values
    return [
        float(cosine_similarity(vector, model.vector_of(other).values))
        for other in words
    ]


PLAIN_WORDS: tuple[tuple[str, str], ...] = (("token", "word"),)
"""The substitutions that take the code's vocabulary out of a refusal."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with nothing in it a reader has to decode.

    The messages are written to be read and mostly already are, so this
    substitutes the one word that belongs to the code rather than to the page
    and leaves the rest alone. A test asserts no identifier survives, so a
    message that grows one later fails there rather than reaching a reader.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _refusal_of(attempt: Callable[[], object]) -> str | None:
    """The plain words of whatever the attempt refused with, or nothing."""
    try:
        attempt()
    except MLLibError as refusal:
        return _in_plain_words(refusal)
    return None


class Neighbour(BaseModel):
    """One nearby word and how near it is."""

    word: str
    similarity: float
    topic: str


def _neighbours_of_vector(
    model: FastText, word: str, n_results: int
) -> list[Neighbour]:
    return [
        Neighbour(
            word=similar.word,
            similarity=float(similar.similarity),
            topic=TOPIC_OF[similar.word],
        )
        for similar in model.most_similar(word, n_results)
    ]


class EpochReport(BaseModel):
    """What one pass over the corpus cost."""

    epoch: int
    mean_loss: float
    n_pairs: int
    learning_rate: float


def _epoch_reports(model: Word2Vec) -> list[EpochReport]:
    return [
        EpochReport(
            epoch=record.epoch,
            mean_loss=record.mean_loss,
            n_pairs=record.n_pairs,
            learning_rate=record.learning_rate,
        )
        for record in model.history
    ]


# --- The pieces of one word, which need no fit -------------------------------


class PieceReport(BaseModel):
    """One piece of a word, where it is kept, and who else owns it."""

    piece: str
    length: int
    bucket: int
    owners: list[str]


class PiecesRequest(BaseModel):
    """A word, and the lengths of the pieces to cut it into."""

    word: str = Field(default="playing", min_length=1, max_length=24)
    minimum_length: int = Field(default=DEFAULT_MINIMUM_LENGTH, ge=1, le=MAX_LENGTH)
    maximum_length: int = Field(default=DEFAULT_MAXIMUM_LENGTH, ge=1, le=MAX_LENGTH)
    n_buckets: int = Field(default=DEFAULT_BUCKETS, ge=1, le=MAX_BUCKETS)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_the_lengths_are_ordered(self) -> PiecesRequest:
        if self.maximum_length < self.minimum_length:
            raise ValueError("the longest piece cannot be shorter than the shortest")
        return self


class PiecesResponse(BaseModel):
    """Every piece of one wrapped word, in the order the cutting produces them.

    ``shared_buckets`` counts the pieces whose row some corpus word also writes
    to, which is how much of this word's answer the corpus can have taught, and
    ``owners`` on a piece names the corpus words that own it, empty for a piece
    no sentence ever contained.
    """

    word: str
    wrapped: str
    seen: bool
    pieces: list[PieceReport]
    n_pieces: int
    n_pieces_owned: int
    shared_buckets: int
    n_buckets: int


@router.post("/pieces", response_model=PiecesResponse)
def pieces(request: PiecesRequest) -> PiecesResponse:
    """Cut one word into its pieces and say where each of them is kept."""
    shape = FastText(
        minimum_n_gram_length=request.minimum_length,
        maximum_n_gram_length=request.maximum_length,
        n_buckets=request.n_buckets,
    )
    owners = _owners(request.minimum_length, request.maximum_length)
    owned_buckets = {fowler_noll_vo_hash(piece) % request.n_buckets for piece in owners}
    cut = shape.n_grams_of(request.word)
    reports = [
        PieceReport(
            piece=piece,
            length=len(piece),
            bucket=fowler_noll_vo_hash(piece) % request.n_buckets,
            owners=list(owners.get(piece, ())),
        )
        for piece in cut
    ]
    return PiecesResponse(
        word=request.word,
        wrapped=f"<{request.word}>",
        seen=request.word in _corpus_vocabulary(),
        pieces=reports,
        n_pieces=len(reports),
        n_pieces_owned=sum(1 for report in reports if report.owners),
        shared_buckets=len({report.bucket for report in reports} & owned_buckets),
        n_buckets=request.n_buckets,
    )


# --- One fit, on request -----------------------------------------------------


class FitRequest(BaseModel):
    """One fit of the running corpus, and the word to look around afterwards."""

    minimum_length: int = Field(default=DEFAULT_MINIMUM_LENGTH, ge=1, le=MAX_LENGTH)
    maximum_length: int = Field(default=DEFAULT_MAXIMUM_LENGTH, ge=1, le=MAX_LENGTH)
    n_buckets: int = Field(default=DEFAULT_BUCKETS, ge=1, le=MAX_BUCKETS)
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(default=DEFAULT_RATE, gt=0.0, le=0.5)
    window: int = Field(default=DEFAULT_WINDOW, ge=1, le=5)
    word: str = Field(default=MISSING_FORM, min_length=1, max_length=24)
    n_neighbours: int = Field(default=6, ge=1, le=12)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_the_lengths_are_ordered(self) -> FitRequest:
        if self.maximum_length < self.minimum_length:
            raise ValueError("the longest piece cannot be shorter than the shortest")
        return self


class FitResponse(BaseModel):
    """A fit, the objective falling, and where one word ended up.

    ``word_refusal`` carries the sentence a reader gets when the chosen word
    shares no row with anything the fit saw, in which case its vector is all
    zeros and has no direction to compare; the neighbours are then empty rather
    than misleading.
    """

    n_sentences: int
    n_occurrences: int
    n_words: int
    epochs: list[EpochReport]
    total_pairs: int
    within_topic: float
    across_topic: float
    word: str
    word_seen: bool
    word_pieces: list[str]
    n_word_pieces: int
    shared_buckets: int
    mean_to_verbs: float
    mean_to_money: float
    neighbours: list[Neighbour]
    word_refusal: str | None
    n_distinct_pieces: int
    n_distinct_rows: int
    n_shared_pieces: int
    word_row_numbers: int
    bucket_row_numbers: int
    rows_that_moved: int
    seconds: float


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Fit the running corpus once and look up one word, seen or not."""
    started = time.perf_counter()
    model = _fit_fasttext(
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        minimum_length=request.minimum_length,
        maximum_length=request.maximum_length,
        n_buckets=request.n_buckets,
        window=request.window,
    )
    seconds = time.perf_counter() - started

    within, across = _within_and_across(model.embeddings)
    corpus_pieces = {
        piece for word in model.vocabulary for piece in model.n_grams_of(word)
    }
    corpus_rows = {
        bucket for word in model.vocabulary for bucket in model.n_gram_ids_of(word)
    }
    word_pieces = model.n_grams_of(request.word)
    word_rows = set(model.n_gram_ids_of(request.word))

    refusal = _refusal_of(lambda: model.most_similar(request.word, 1))
    neighbours: list[Neighbour] = []
    to_verbs = 0.0
    to_money = 0.0
    if refusal is None:
        neighbours = _neighbours_of_vector(model, request.word, request.n_neighbours)
        to_verbs = float(np.mean(_similarities_to(model, request.word, VERB_WORDS)))
        to_money = float(np.mean(_similarities_to(model, request.word, MONEY_WORDS)))

    return FitResponse(
        n_sentences=len(TWO_LIST_CORPUS),
        n_occurrences=_n_occurrences(),
        n_words=model.embeddings.n_words,
        epochs=_epoch_reports(model),
        total_pairs=sum(record.n_pairs for record in model.history),
        within_topic=within,
        across_topic=across,
        word=request.word,
        word_seen=request.word in model.vocabulary,
        word_pieces=list(word_pieces),
        n_word_pieces=len(word_pieces),
        shared_buckets=len(word_rows & corpus_rows),
        mean_to_verbs=to_verbs,
        mean_to_money=to_money,
        neighbours=neighbours,
        word_refusal=refusal,
        n_distinct_pieces=len(corpus_pieces),
        n_distinct_rows=len(corpus_rows),
        n_shared_pieces=len(corpus_pieces) - len(corpus_rows),
        word_row_numbers=int(model.word_row_vectors.size),
        bucket_row_numbers=int(model.bucket_vectors.size),
        rows_that_moved=int(np.sum(np.any(model.bucket_vectors != 0.0, axis=1))),
        seconds=seconds,
    )


# --- A word the corpus never held --------------------------------------------


class SharedRow(BaseModel):
    """One row a word writes to that some corpus piece writes to as well.

    ``piece`` is the probe word's own piece and ``corpus_pieces`` the pieces of
    corpus words that landed in the same row, which may be the same spelling or
    an unrelated one that happened to hash there.
    """

    bucket: int
    piece: str
    corpus_pieces: list[str]
    owners: list[str]


class UnseenProbe(BaseModel):
    """One word absent from the corpus, and where its spelling puts it."""

    word: str
    n_pieces: int
    shared_buckets: int
    length: float
    mean_to_verbs: float
    mean_to_money: float
    nearest: list[Neighbour]
    shared_rows: list[SharedRow]
    refusal: str | None


class SharerReport(BaseModel):
    """A corpus word that shares pieces with the missing form, and which."""

    word: str
    pieces: list[str]


class SpellingShare(BaseModel):
    """How much of a word the corpus did hold is carried by its spelling.

    ``cosine`` compares the word's whole vector against the sum of its pieces
    alone, and ``length_share`` is how long that sum is beside the whole.
    """

    word: str
    n_pieces: int
    cosine: float
    length_share: float


class UnseenResponse(BaseModel):
    """The missing verb form answered from its spelling, and the plain refusal.

    ``plain_refusal`` is what a fit with one row per word says when the same
    word is asked for, and ``smallest_to_verbs`` and ``largest_to_money`` are
    the worst cases behind the two means, so a reader can see the two ranges
    do not overlap rather than take an average on trust.
    """

    word: str
    n_pieces: int
    n_pieces_owned: int
    shared_buckets: int
    sharers: list[SharerReport]
    mean_to_verbs: float
    smallest_to_verbs: float
    mean_to_money: float
    largest_to_money: float
    nearest: list[Neighbour]
    plain_refusal: str
    probes: list[UnseenProbe]
    spelling_shares: list[SpellingShare]


PROBE_WORDS: tuple[str, ...] = (
    MISSING_FORM,
    "walkway",
    "banking",
    "stocking",
    "traded",
    "sharing",
    "priced",
    "queue",
    SHARELESS_WORD,
)
"""Words no sentence holds, ordered so the two that mislead come last."""

SPELLING_SHARE_WORDS: tuple[str, ...] = ("played", "walking", "market", "bond")
"""Words the corpus did hold, for the reading that drops their own row."""


@cache
def _unseen() -> UnseenResponse:
    model = _reference_fasttext()
    plain = _reference_plain()
    owners = _owners(DEFAULT_MINIMUM_LENGTH, DEFAULT_MAXIMUM_LENGTH)
    corpus_rows = {
        bucket for word in model.vocabulary for bucket in model.n_gram_ids_of(word)
    }
    landed: dict[int, list[str]] = {}
    for piece in owners:
        landed.setdefault(fowler_noll_vo_hash(piece) % DEFAULT_BUCKETS, []).append(
            piece
        )

    missing_pieces = set(model.n_grams_of(MISSING_FORM))
    sharers = [
        SharerReport(
            word=word,
            pieces=sorted(missing_pieces & set(model.n_grams_of(word))),
        )
        for word in model.vocabulary
        if missing_pieces & set(model.n_grams_of(word))
    ]
    sharers.sort(key=lambda entry: (-len(entry.pieces), entry.word))

    to_verbs = _similarities_to(model, MISSING_FORM, VERB_WORDS)
    to_money = _similarities_to(model, MISSING_FORM, MONEY_WORDS)

    probes: list[UnseenProbe] = []
    for word in PROBE_WORDS:
        refusal = _refusal_of(lambda word=word: model.most_similar(word, 1))
        shared_rows: list[SharedRow] = []
        for piece in model.n_grams_of(word):
            bucket = fowler_noll_vo_hash(piece) % DEFAULT_BUCKETS
            if bucket not in landed:
                continue
            shared_rows.append(
                SharedRow(
                    bucket=bucket,
                    piece=piece,
                    corpus_pieces=sorted(landed[bucket]),
                    owners=sorted(
                        {
                            owner
                            for corpus_piece in landed[bucket]
                            for owner in owners[corpus_piece]
                        }
                    ),
                )
            )
        probes.append(
            UnseenProbe(
                word=word,
                n_pieces=len(model.n_grams_of(word)),
                shared_buckets=len(set(model.n_gram_ids_of(word)) & corpus_rows),
                length=float(np.linalg.norm(model.vector_of(word).values)),
                mean_to_verbs=(
                    0.0
                    if refusal
                    else float(np.mean(_similarities_to(model, word, VERB_WORDS)))
                ),
                mean_to_money=(
                    0.0
                    if refusal
                    else float(np.mean(_similarities_to(model, word, MONEY_WORDS)))
                ),
                nearest=([] if refusal else _neighbours_of_vector(model, word, 3)),
                shared_rows=shared_rows,
                refusal=refusal,
            )
        )

    shares: list[SpellingShare] = []
    for word in SPELLING_SHARE_WORDS:
        whole = np.asarray(model.vector_of(word).values)
        spelling = np.asarray(model.vector_of_unseen(word).values)
        shares.append(
            SpellingShare(
                word=word,
                n_pieces=len(model.n_grams_of(word)),
                cosine=float(cosine_similarity(whole, spelling)),
                length_share=float(np.linalg.norm(spelling) / np.linalg.norm(whole)),
            )
        )

    plain_refusal = _refusal_of(lambda: plain.embeddings.most_similar(MISSING_FORM, 1))
    return UnseenResponse(
        word=MISSING_FORM,
        n_pieces=len(missing_pieces),
        n_pieces_owned=sum(1 for piece in missing_pieces if piece in owners),
        shared_buckets=len(set(model.n_gram_ids_of(MISSING_FORM)) & corpus_rows),
        sharers=sharers,
        mean_to_verbs=float(np.mean(to_verbs)),
        smallest_to_verbs=float(np.min(to_verbs)),
        mean_to_money=float(np.mean(to_money)),
        largest_to_money=float(np.max(to_money)),
        nearest=_neighbours_of_vector(model, MISSING_FORM, 6),
        plain_refusal=plain_refusal if plain_refusal else "",
        probes=probes,
        spelling_shares=shares,
    )


@router.get("/unseen", response_model=UnseenResponse)
def unseen() -> UnseenResponse:
    """The missing verb form, answered from its spelling alone."""
    return _unseen()


# --- The fit with no piece short enough to exist -----------------------------


class ReductionRow(BaseModel):
    """One pairing of the two choices, and the largest gap anywhere in it."""

    architecture: str
    objective: str
    table_gap: float
    output_gap: float
    loss_gap: float
    largest_bucket_value: float


class ReductionResponse(BaseModel):
    """What a fit whose pieces cannot exist agrees with, and by how much.

    ``losses_with_pieces`` and ``losses_without`` are the same two passes of
    the first pairing, so a reader can read the equality off two columns rather
    than take the maximum gap on trust.
    """

    minimum_length: int
    longest_wrapped_word: int
    epochs: int
    rows: list[ReductionRow]
    losses_with_pieces: list[float]
    losses_without: list[float]


ARCHITECTURE_WORDS: dict[Word2VecArchitecture, str] = {
    Word2VecArchitecture.SKIP_GRAM: "the centre predicting each neighbour",
    Word2VecArchitecture.CONTINUOUS_BAG_OF_WORDS: "the neighbours predicting the centre",
}
OBJECTIVE_WORDS: dict[Word2VecObjective, str] = {
    Word2VecObjective.NEGATIVE_SAMPLING: "drawn wrong answers",
    Word2VecObjective.HIERARCHICAL_SOFTMAX: "a tree",
}
"""The page's words for the two choices, so no enum value crosses the wire."""

REDUCTION_EPOCHS = 2
"""Two passes, which is enough for the two walks to part if they were going to."""


@cache
def _reduction() -> ReductionResponse:
    rows: list[ReductionRow] = []
    with_pieces: list[float] = []
    without: list[float] = []
    for architecture in Word2VecArchitecture:
        for objective in Word2VecObjective:
            reduced = _fit_fasttext(
                epochs=REDUCTION_EPOCHS,
                minimum_length=NO_PIECE_LENGTH,
                maximum_length=NO_PIECE_LENGTH,
                architecture=architecture,
                objective=objective,
            )
            plain = _fit_plain(
                epochs=REDUCTION_EPOCHS,
                architecture=architecture,
                objective=objective,
            )
            reduced_table = np.asarray(reduced.embeddings.table)
            plain_table = np.asarray(plain.embeddings.table)
            reduced_losses = [record.mean_loss for record in reduced.history]
            plain_losses = [record.mean_loss for record in plain.history]
            if not with_pieces:
                with_pieces = reduced_losses
                without = plain_losses
            rows.append(
                ReductionRow(
                    architecture=ARCHITECTURE_WORDS[architecture],
                    objective=OBJECTIVE_WORDS[objective],
                    table_gap=float(np.max(np.abs(reduced_table - plain_table))),
                    output_gap=float(
                        np.max(
                            np.abs(
                                np.asarray(reduced.output_vectors)
                                - np.asarray(plain.output_vectors)
                            )
                        )
                    ),
                    loss_gap=float(
                        max(
                            abs(first - second)
                            for first, second in zip(reduced_losses, plain_losses)
                        )
                    ),
                    largest_bucket_value=float(np.max(np.abs(reduced.bucket_vectors))),
                )
            )
    return ReductionResponse(
        minimum_length=NO_PIECE_LENGTH,
        longest_wrapped_word=max(len(word) for word in _corpus_vocabulary()) + 2,
        epochs=REDUCTION_EPOCHS,
        rows=rows,
        losses_with_pieces=with_pieces,
        losses_without=without,
    )


@router.get("/reduction", response_model=ReductionResponse)
def reduction() -> ReductionResponse:
    """What happens when the pieces are asked to be longer than any word."""
    return _reduction()


# --- How many rows the pieces have to share ----------------------------------


class SharingRow(BaseModel):
    """One table size, and how much of it the pieces have to themselves."""

    n_buckets: int
    n_rows_used: int
    n_shared: int


class CollidingRow(BaseModel):
    """One row two or more pieces of the corpus write to."""

    bucket: int
    pieces: list[str]
    owners: list[list[str]]


class BucketFitRow(BaseModel):
    """A fit at one table size, and what the sharing did to it."""

    n_buckets: int
    n_rows_used: int
    n_shared: int
    within_topic: float
    across_topic: float
    gap: float
    missing_to_verbs: float
    missing_to_money: float


class LengthRow(BaseModel):
    """One pair of piece lengths, and what cutting that way produced."""

    minimum_length: int
    maximum_length: int
    n_distinct_pieces: int
    pieces_per_word: float
    within_topic: float
    across_topic: float
    gap: float
    missing_to_verbs: float
    missing_to_money: float


class BucketsResponse(BaseModel):
    """How 279 pieces fit into a fixed number of rows, and what it costs.

    ``sharing`` is arithmetic on the hash alone and needs no fit; ``fits`` runs
    the corpus at four of those sizes, so the reader can see a table too small
    doing damage rather than be told it would; ``lengths`` does the same for the
    other choice with no principle behind it, how short and how long a piece may
    be.
    """

    n_words: int
    n_distinct_pieces: int
    pieces_per_word: float
    fewest_pieces: int
    most_pieces: int
    sharing: list[SharingRow]
    colliding: list[CollidingRow]
    fits: list[BucketFitRow]
    lengths: list[LengthRow]
    published_buckets: int


SHARING_SIZES: tuple[int, ...] = (50, 100, 250, 500, 1000, 2000, 20_000, 2_000_000)
"""Table sizes to count rows for, ending at the published one."""

FITTED_SIZES: tuple[int, ...] = (10, 50, 200, 2000)
"""Table sizes to actually fit at, small enough that the damage shows."""

FITTED_LENGTHS: tuple[tuple[int, int], ...] = ((3, 3), (3, 6), (4, 6))
"""Piece lengths to fit at, with the published pair among them."""

PUBLISHED_BUCKETS = 2_000_000
"""The table size the published work uses."""


@cache
def _buckets() -> BucketsResponse:
    shape = FastText()
    words = list(_corpus_vocabulary())
    pieces = sorted({piece for word in words for piece in shape.n_grams_of(word)})
    owners = _owners(DEFAULT_MINIMUM_LENGTH, DEFAULT_MAXIMUM_LENGTH)
    per_word = [len(shape.n_grams_of(word)) for word in words]

    sharing = [
        SharingRow(
            n_buckets=size,
            n_rows_used=len({fowler_noll_vo_hash(piece) % size for piece in pieces}),
            n_shared=len(pieces)
            - len({fowler_noll_vo_hash(piece) % size for piece in pieces}),
        )
        for size in SHARING_SIZES
    ]

    landed: dict[int, list[str]] = {}
    for piece in pieces:
        landed.setdefault(fowler_noll_vo_hash(piece) % DEFAULT_BUCKETS, []).append(
            piece
        )
    colliding = [
        CollidingRow(
            bucket=bucket,
            pieces=sharing_pieces,
            owners=[list(owners[piece]) for piece in sharing_pieces],
        )
        for bucket, sharing_pieces in sorted(landed.items())
        if len(sharing_pieces) > 1
    ]

    fits: list[BucketFitRow] = []
    for size in FITTED_SIZES:
        model = (
            _reference_fasttext()
            if size == DEFAULT_BUCKETS
            else _fit_fasttext(n_buckets=size)
        )
        within, across = _within_and_across(model.embeddings)
        used = len({fowler_noll_vo_hash(piece) % size for piece in pieces})
        fits.append(
            BucketFitRow(
                n_buckets=size,
                n_rows_used=used,
                n_shared=len(pieces) - used,
                within_topic=within,
                across_topic=across,
                gap=within - across,
                missing_to_verbs=float(
                    np.mean(_similarities_to(model, MISSING_FORM, VERB_WORDS))
                ),
                missing_to_money=float(
                    np.mean(_similarities_to(model, MISSING_FORM, MONEY_WORDS))
                ),
            )
        )

    lengths: list[LengthRow] = []
    for shortest, longest in FITTED_LENGTHS:
        published = (shortest, longest) == (
            DEFAULT_MINIMUM_LENGTH,
            DEFAULT_MAXIMUM_LENGTH,
        )
        model = (
            _reference_fasttext()
            if published
            else _fit_fasttext(minimum_length=shortest, maximum_length=longest)
        )
        within, across = _within_and_across(model.embeddings)
        cut = FastText(minimum_n_gram_length=shortest, maximum_n_gram_length=longest)
        distinct = {piece for word in words for piece in cut.n_grams_of(word)}
        lengths.append(
            LengthRow(
                minimum_length=shortest,
                maximum_length=longest,
                n_distinct_pieces=len(distinct),
                pieces_per_word=float(
                    np.mean([len(cut.n_grams_of(word)) for word in words])
                ),
                within_topic=within,
                across_topic=across,
                gap=within - across,
                missing_to_verbs=float(
                    np.mean(_similarities_to(model, MISSING_FORM, VERB_WORDS))
                ),
                missing_to_money=float(
                    np.mean(_similarities_to(model, MISSING_FORM, MONEY_WORDS))
                ),
            )
        )

    return BucketsResponse(
        n_words=len(words),
        n_distinct_pieces=len(pieces),
        pieces_per_word=float(np.mean(per_word)),
        fewest_pieces=int(min(per_word)),
        most_pieces=int(max(per_word)),
        sharing=sharing,
        colliding=colliding,
        fits=fits,
        lengths=lengths,
        published_buckets=PUBLISHED_BUCKETS,
    )


@router.get("/buckets", response_model=BucketsResponse)
def buckets() -> BucketsResponse:
    """How many rows the corpus's pieces need, and which of them share one."""
    return _buckets()


# --- The two fits side by side ------------------------------------------------


class WordComparison(BaseModel):
    """One word, and the nearest words to it under each of the two fits."""

    word: str
    topic: str
    with_pieces: list[Neighbour]
    without_pieces: list[Neighbour]
    shared: list[str]


class SeparationRow(BaseModel):
    """What one fit did to the two lists, at one first step."""

    method: str
    learning_rate: float
    within_topic: float
    across_topic: float
    gap: float


class ComparisonResponse(BaseModel):
    """The two fits read side by side, in neighbours, separation and bill.

    ``parts_per_word`` is one for the word's own row plus one for each of its
    pieces, which is how many rows a single correction is added to, so it is
    also how much further the word's composed vector moves than a plain one
    would for the same gradient.
    """

    words: list[WordComparison]
    separation: list[SeparationRow]
    numbers_with_pieces: int
    numbers_without_pieces: int
    rows_that_moved: int
    composed_table_numbers: int
    parts_per_word: float
    fewest_parts: int
    most_parts: int
    losses_with_pieces: list[float]
    losses_without: list[float]


COMPARISON_WORDS: tuple[str, ...] = (
    "walked",
    "player",
    "saying",
    "market",
    "share",
)
"""Words in the corpus whose neighbour lists the page prints for both fits."""


@cache
def _comparison() -> ComparisonResponse:
    model = _reference_fasttext()
    plain = _reference_plain()

    words: list[WordComparison] = []
    for word in COMPARISON_WORDS:
        with_pieces = _neighbours_of_vector(model, word, 5)
        without = [
            Neighbour(
                word=similar.word,
                similarity=float(similar.similarity),
                topic=TOPIC_OF[similar.word],
            )
            for similar in plain.embeddings.most_similar(word, 5)
        ]
        words.append(
            WordComparison(
                word=word,
                topic=TOPIC_OF[word],
                with_pieces=with_pieces,
                without_pieces=without,
                shared=sorted(
                    {entry.word for entry in with_pieces}
                    & {entry.word for entry in without}
                ),
            )
        )

    separation: list[SeparationRow] = []
    for rate in (DEFAULT_RATE, DOUBLED_RATE):
        pieced = model if rate == DEFAULT_RATE else _fit_fasttext(learning_rate=rate)
        bare = plain if rate == DEFAULT_RATE else _fit_plain(learning_rate=rate)
        for method, fitted in (("pieces", pieced), ("words", bare)):
            within, across = _within_and_across(fitted.embeddings)
            separation.append(
                SeparationRow(
                    method=method,
                    learning_rate=rate,
                    within_topic=within,
                    across_topic=across,
                    gap=within - across,
                )
            )

    parts = [len(model.n_grams_of(word)) + 1 for word in model.vocabulary]
    return ComparisonResponse(
        words=words,
        separation=separation,
        numbers_with_pieces=int(
            model.word_row_vectors.size + model.bucket_vectors.size
        ),
        numbers_without_pieces=int(np.asarray(plain.embeddings.table).size),
        rows_that_moved=int(np.sum(np.any(model.bucket_vectors != 0.0, axis=1))),
        composed_table_numbers=int(np.asarray(model.embeddings.table).size),
        parts_per_word=float(np.mean(parts)),
        fewest_parts=int(min(parts)),
        most_parts=int(max(parts)),
        losses_with_pieces=[record.mean_loss for record in model.history],
        losses_without=[record.mean_loss for record in plain.history],
    )


@router.get("/comparison", response_model=ComparisonResponse)
def comparison() -> ComparisonResponse:
    """The same corpus fitted with the pieces and without them."""
    return _comparison()


# --- Taking a word out of the corpus and asking for it anyway ------------------


class HeldOutResponse(BaseModel):
    """A word the corpus did hold, removed from it and then asked for.

    The counts describe the corpus as it stands, so a reader can see it has no
    rare words at all and therefore cannot show the missing-word case by
    itself; everything else is the case made deliberately, by dropping every
    sentence that held one word.
    """

    word: str
    fewest_occurrences: int
    most_occurrences: int
    words_seen_once: int
    n_sentences_dropped: int
    n_occurrences_dropped: int
    n_words_left: int
    n_pieces: int
    shared_buckets: int
    mean_to_verbs: float
    mean_to_money: float
    nearest: list[Neighbour]
    plain_refusal: str


@cache
def _held_out() -> HeldOutResponse:
    kept = [
        sentence
        for sentence in TWO_LIST_CORPUS
        if HELD_OUT_WORD not in sentence.split()
    ]
    counts = TokenisedCorpus.from_texts(
        TWO_LIST_CORPUS, WhitespacePreTokenizer()
    ).word_counts
    occurrences = [counts[word] for word in _corpus_vocabulary()]

    model = FastText(
        dimension=DEFAULT_DIMENSION,
        window=DEFAULT_WINDOW,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_RATE,
        random_seed=0,
    ).fit(kept)
    plain = Word2Vec(
        dimension=DEFAULT_DIMENSION,
        window=DEFAULT_WINDOW,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_RATE,
        random_seed=0,
    ).fit(kept)

    corpus_rows = {
        bucket for word in model.vocabulary for bucket in model.n_gram_ids_of(word)
    }
    remaining_verbs = tuple(word for word in VERB_WORDS if word != HELD_OUT_WORD)
    refusal = _refusal_of(lambda: plain.embeddings.most_similar(HELD_OUT_WORD, 1))
    return HeldOutResponse(
        word=HELD_OUT_WORD,
        fewest_occurrences=int(min(occurrences)),
        most_occurrences=int(max(occurrences)),
        words_seen_once=sum(1 for count in occurrences if count == 1),
        n_sentences_dropped=len(TWO_LIST_CORPUS) - len(kept),
        n_occurrences_dropped=counts[HELD_OUT_WORD],
        n_words_left=model.embeddings.n_words,
        n_pieces=len(model.n_grams_of(HELD_OUT_WORD)),
        shared_buckets=len(set(model.n_gram_ids_of(HELD_OUT_WORD)) & corpus_rows),
        mean_to_verbs=float(
            np.mean(_similarities_to(model, HELD_OUT_WORD, remaining_verbs))
        ),
        mean_to_money=float(
            np.mean(_similarities_to(model, HELD_OUT_WORD, MONEY_WORDS))
        ),
        nearest=_neighbours_of_vector(model, HELD_OUT_WORD, 5),
        plain_refusal=refusal if refusal else "",
    )


@router.get("/held-out", response_model=HeldOutResponse)
def held_out() -> HeldOutResponse:
    """Drop every sentence holding one word, then ask for that word."""
    return _held_out()
