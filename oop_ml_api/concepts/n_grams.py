"""N-grams, and a model built from counting runs of words.

Every endpoint here is registered under ``/concepts/n-grams``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about the n-gram language model, its smoothing rules and what a
held-out text scores under it.

Five static endpoints and one playground, grouped by the family of widgets each
serves rather than by the number each returns. ``/grams`` is everything about
the windows themselves: the running sentence framed and cut at four orders, how
many distinct runs the training text produced against how many the vocabulary
allows, how much of a held-out text is made of runs that never occurred, and
what happens to all three as the corpus grows. ``/counts`` is the arithmetic
small enough to check with a pencil: a three-sentence corpus with every count
listed, the two estimates of a handful of conditional probabilities, and then
the same walk taken word by word through the running sentence on the larger
corpus. ``/smoothing`` is everything about the additive constant: the sweep on a
tuning text with the winner also scored on a test text, what the constant does
to a context the counts were certain about, the two limits, and four next-word
distributions drawn at two strengths. ``/scores`` is judgement: perplexity
against order under both rules and on both texts, the same test text with its
words reversed, and a dependency deliberately placed further back than the
window reaches. ``/generation`` is the model writing rather than scoring.

The one thing computed here rather than by the fitted model is the count of how
many runs of a held-out text never occurred in training, which is a question
about the pair of texts rather than about either one; every probability,
perplexity and drawn sentence is a fit.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from functools import cache

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.n_grams.grams import (
    SENTENCE_END,
    SENTENCE_START,
    UNKNOWN_WORD,
    padded,
    word_n_grams,
)
from oop_ml.core.natural_language_processing.n_grams.language_model import (
    NGramLanguageModel,
)
from oop_ml.core.natural_language_processing.n_grams.smoothing.base import (
    AdditiveSmoothing,
    MaximumLikelihood,
    Smoothing,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/n-grams", tags=["n-grams"])


MAX_ORDER = 6
"""The largest window a request may ask for.

The training corpus holds 326 word occurrences, so a window of six already sees
almost every sentence whole; nothing above it would teach anything, and the
bound keeps a crafted request from asking for a very deep table.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask a model to read."""

MAX_SMOOTHING_STRENGTH = 100.0
"""The largest additive constant a request may ask for.

Far past the point where every word of the vocabulary is equally likely, which
the page shows happening by a strength of a thousand on its own corpus, so this
is a bound on the request rather than on the mathematics.
"""

MAX_GENERATED_WORDS = 30
"""The most words one drawn sentence may run to before it is cut off."""

DEFAULT_STRENGTH = 0.005
"""The additive constant the page uses wherever it wants a well chosen one.

Chosen on the tuning texts by ``/smoothing``, not on the test texts, so a number
quoted beside it is not a number it was picked to flatter.
"""


# --------------------------------------------------------------------------
# The texts the page works with.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

SMALL_CORPUS: list[str] = [
    "the report was late",
    "the report was short",
    "a summary was late",
]
"""Three sentences, chosen so every count on the page can be checked by hand.

Nine words that can be predicted once the end marker and the stand-in for an
unseen word are counted, twelve predicted positions in all, and one word,
``summary``, that never follows ``the``, which is the hole the whole page is
about. Written in the same world as the larger corpus so a reader meets one
subject rather than two.
"""

TRAINING_CORPUS: list[str] = [
    "The report was expected on Monday.",
    "The report was expected to cost less.",
    "The first report was late.",
    "The second report was late as well.",
    "Dr. Alvarez wrote the report on Monday.",
    "Dr. Alvarez wrote the report on Tuesday.",
    "Dr. Alvarez read the report and asked for a re-analysis.",
    "The team asked for a re-analysis of the samples.",
    "The team asked for a low-cost re-analysis.",
    "A low-cost re-analysis was expected.",
    "A low-cost re-analysis of the samples was expected.",
    "The cost of the analysis was the reason.",
    "The cost of the re-analysis was the reason.",
    "The cost of the report was the reason for the delay.",
    "The analysis of the samples was expected on Monday.",
    "The analysis of the samples was expected to agree.",
    "The analysis of the first batch was late.",
    "The analysis of the second batch was late.",
    "We reviewed the analysis and rewrote the summary.",
    "We reviewed the report and rewrote the summary.",
    "We reviewed the samples and wrote the summary.",
    "The summary of the analysis was short.",
    "The summary of the report was short.",
    "The samples were expected on Monday.",
    "The samples were expected on Tuesday.",
    "The samples were late.",
    "Nobody expected the samples to arrive early.",
    "Nobody expected the report to arrive early.",
    "Nobody expected the delay.",
    "The team reran the tests on Monday.",
    "The team reran the tests on the second batch.",
    "The tests on the first batch were late.",
    "The estimate was lower than the cost.",
    "The estimate was lower than the first estimate.",
    "The revised estimate was lower again.",
    "The revised estimate of the cost was short.",
    "Dr. Alvarez asked for the analysis of the samples.",
    "Dr. Alvarez asked for the cost of the analysis.",
    "Dr. Alvarez expected the analysis to agree.",
    "Dr. Alvarez expected the report to agree.",
    "The analysis and the re-analysis agree.",
    "The report and the summary agree.",
    "The delay was the reason for the revised estimate.",
    "The delay was the reason for the low-cost re-analysis.",
]
"""Forty-four sentences from the world the running sentence belongs to.

Written so that a few phrases recur often enough for a window of three to have
something to stand on, while the vocabulary stays small enough that a table of
every distinct run is a thing a reader can imagine holding. None of them is the
running sentence, and none of them contains the words ``didn't`` or ``expect``,
which is what makes the running sentence carry two words the model has never
met.
"""

TUNING_TEXTS: list[str] = [
    "The cost of the samples was the reason for the delay.",
    "The team asked for the analysis of the second batch.",
    "We reviewed the summary and rewrote the analysis.",
    "The revised report was expected on Monday.",
    "Nobody expected the analysis to arrive early.",
    "The estimate of the re-analysis was short.",
]
"""Six sentences held back, on which the smoothing strength is chosen."""

TEST_TEXTS: list[str] = [
    RUNNING_SENTENCE,
    "The cost of the re-analysis was lower than the estimate.",
    "The team expected the samples on Tuesday.",
    "We reviewed the tests and rewrote the analysis.",
    "Nobody expected the second batch to be late.",
    "The summary of the re-analysis was short.",
]
"""Six more, never used to choose anything, on which every number is reported."""

STRENGTH_SWEEP: tuple[float, ...] = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
)
"""The additive constants swept, spread evenly on a logarithmic scale."""

LIMIT_STRENGTHS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 100000.0)
"""Constants far above one, where every context turns into the flat answer."""

KNOWING_NOTHING_STRENGTH = 100000.0
"""The constant at which a model has been pretended into knowing nothing.

Every word of the vocabulary comes out equally likely whatever stands before
it, so the perplexity is the size of the vocabulary, which is what makes it the
right yardstick for a model of that vocabulary and useless as a comparison
between two models with different ones.
"""

CERTAIN_STRENGTHS: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01, 0.005, 0.001)
"""Constants at which the page reads one context the counts were certain about."""

SWEEP_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The window widths every sweep on the page runs over."""

FRAMING_ORDERS: tuple[int, ...] = (1, 2, 3, 4)
"""The window widths the running sentence is cut at."""

GENERATION_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
"""The window widths the page draws sentences at."""

GENERATION_SEEDS: tuple[int, ...] = (0, 1, 2)
"""The three draws shown at each width, so a reader sees a range rather than one."""

N_GENERATION_DRAWS = 50
"""How many sentences are drawn when counting how often training text comes back."""

TEMPERATURES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
"""How flat or peaked each draw is made before a word is taken from it."""

CERTAIN_CONTEXT = "Dr."
"""A word the training text always followed with the same next word."""

CERTAIN_WORD = "Alvarez"
"""What always followed it."""

UNSEEN_CONTEXT = "didn't"
"""A word the training text never contains, so nothing ever followed it."""

DISTRIBUTION_CONTEXTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("before the first word", (SENTENCE_START,)),
    ("after “the”", ("the",)),
    ("after “Dr.”", (CERTAIN_CONTEXT,)),
    ("after a word never seen", (UNSEEN_CONTEXT,)),
)
"""The four contexts the page draws a whole next-word answer for."""

DISTRIBUTION_STRENGTHS: tuple[float, ...] = (1.0, DEFAULT_STRENGTH)
"""The two strengths each of those contexts is drawn at."""

N_DISTRIBUTION_WORDS = 10
"""How many words of a next-word answer the page lists, likeliest first."""

GROWTH_SHARES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
"""The shares of the training text the growth table refits on."""

HAND_CHECKED_PAIRS: tuple[tuple[str, str], ...] = (
    (SENTENCE_START, "the"),
    ("the", "report"),
    ("report", "was"),
    ("was", "late"),
    ("late", SENTENCE_END),
    ("the", "summary"),
    ("was", "short"),
)
"""The seven conditional probabilities the small corpus is worked through."""

HAND_CHECKED_SENTENCES: tuple[str, ...] = (
    "the report was late",
    "the summary was late",
    "a report was short",
)
"""One sentence the small corpus contains and two it does not."""

LONG_DEPENDENCY_CORPUS: list[str] = [
    "The samples that Dr. Alvarez read were late.",
    "The sample that Dr. Alvarez read was late.",
    "The samples that the team read were late.",
    "The sample that the team read was late.",
    "The samples that we read were late.",
    "The sample that we read was late.",
]
"""Six sentences in which the word that settles the verb stands five words back.

Every sentence agrees: a plural subject takes ``were`` and a singular one takes
``was``, and the corpus is otherwise identical between the pair, so a model that
can see the subject answers with certainty and one that cannot answers with a
half.
"""

LONG_DEPENDENCY_CONTEXT: tuple[str, ...] = (
    "The",
    "samples",
    "that",
    "Dr.",
    "Alvarez",
    "read",
)
"""The words standing before the verb, the deciding one first."""

LONG_DEPENDENCY_WORDS: tuple[str, str] = ("were", "was")
"""The two continuations, of which exactly one agrees with the subject."""


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("every text in the corpus is blank", "there are no words here to read"),
    ("a corpus needs at least one text", "there are no words here to read"),
    ("max_words", "how many words to draw"),
    ("minimum_count", "how often a word must occur to be kept"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("n_results", "how many answers to list"),
    ("NGramLanguageModel", "this model"),
    ("AdditiveSmoothing", "this rule"),
    ("MaximumLikelihood", "this rule"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the phrases that name a parameter and leaves the rest alone. A
    test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# Small helpers shared by the endpoints.
# --------------------------------------------------------------------------

SPLITTER = WhitespacePreTokenizer()
"""What decides where the words are, before any run of them is counted."""


def _words_of(text: str) -> list[str]:
    return list(SPLITTER.split(text).texts)


def _finite(value: float) -> float | None:
    """``value``, or ``None`` where it is infinite.

    A perplexity of infinity is a real answer and the one the unsmoothed rule
    gives on almost any new text, but there is no such number in the transport,
    so the endpoints carry it as an absence beside a boolean that names it.
    """
    return None if math.isinf(value) else float(value)


def _fitted(
    order: int, strength: float | None, corpus: Sequence[str] | None = None
) -> NGramLanguageModel:
    """A model of the given width, smoothed by ``strength`` or not at all."""
    rule: Smoothing = (
        MaximumLikelihood()
        if strength is None
        else AdditiveSmoothing(pretended_count=strength)
    )
    return NGramLanguageModel(order=order, smoothing=rule).fit(
        list(TRAINING_CORPUS if corpus is None else corpus)
    )


def _known_words(model: NGramLanguageModel, text: str) -> list[str]:
    """The words of ``text``, each replaced by the stand-in if it is unknown."""
    return [
        word if word in model.vocabulary else UNKNOWN_WORD for word in _words_of(text)
    ]


def _unseen_runs(model: NGramLanguageModel, texts: Sequence[str]) -> tuple[int, int]:
    """How many runs of ``texts`` never occurred in training, and how many there are.

    Written here rather than asked of the model because it is a question about
    the pair of texts, not about either one: the model answers how probable a
    run is, and this counts how many of them the counts had never met.
    """
    order = model.order
    unseen = total = 0
    for text in texts:
        framed = padded(_known_words(model, text), order)
        for position in range(order - 1, len(framed)):
            total += 1
            if model.counts.count_of(framed[position - order + 1 : position + 1]) == 0:
                unseen += 1
    return unseen, total


# --------------------------------------------------------------------------
# /grams: the windows, how many of them there could be, and how few there are.
# --------------------------------------------------------------------------


class Framing(BaseModel):
    """One sentence framed for a window of ``order`` words, and cut into them."""

    order: int
    framed: list[str]
    windows: list[list[str]]
    n_windows: int
    n_start_markers: int


class SparsityRow(BaseModel):
    """How many distinct runs of one width occurred, against how many could."""

    order: int
    n_seen: int
    n_possible: int
    share_seen: float
    n_seen_once: int
    share_seen_once: float
    n_occurrences: int


class NoveltyRow(BaseModel):
    """How much of a text the model was not fitted on is made of new runs."""

    order: int
    n_runs: int
    n_unseen: int
    share_unseen: float


class GrowthRow(BaseModel):
    """What one share of the training text produced, and what it then scored."""

    n_sentences: int
    n_word_occurrences: int
    vocabulary_size: int
    n_unigram_types: int
    n_bigram_types: int
    n_trigram_types: int
    test_perplexity: float
    n_unknown: int
    unknown_share: float
    knowing_nothing_perplexity: float


class GramsResponse(BaseModel):
    """The windows themselves, and the counting problem they set up."""

    sentence: str
    words: list[str]
    n_words: int
    sentence_start: str
    sentence_end: str
    framings: list[Framing]
    n_training_sentences: int
    n_word_occurrences: int
    n_distinct_words: int
    vocabulary_size: int
    sparsity: list[SparsityRow]
    novelty: list[NoveltyRow]
    growth: list[GrowthRow]


@cache
def _grams() -> GramsResponse:
    words = _words_of(RUNNING_SENTENCE)
    framings = [
        Framing(
            order=order,
            framed=list(padded(words, order)),
            windows=[
                list(gram.words) for gram in word_n_grams(padded(words, order), order)
            ],
            n_windows=len(word_n_grams(padded(words, order), order)),
            n_start_markers=order - 1,
        )
        for order in FRAMING_ORDERS
    ]

    training_words = [word for text in TRAINING_CORPUS for word in _words_of(text)]
    sparsity: list[SparsityRow] = []
    novelty: list[NoveltyRow] = []
    for order in SWEEP_ORDERS:
        model = _fitted(order, DEFAULT_STRENGTH)
        vocabulary_size = model.vocabulary.n_tokens
        table = model.counts.frequency_of_frequencies(order)
        n_seen = model.counts.n_types(order)
        sparsity.append(
            SparsityRow(
                order=order,
                n_seen=n_seen,
                n_possible=vocabulary_size**order,
                share_seen=n_seen / vocabulary_size**order,
                n_seen_once=table.n_types_with_count(1),
                share_seen_once=table.n_types_with_count(1) / table.n_types,
                n_occurrences=table.n_tokens,
            )
        )
        unseen, total = _unseen_runs(model, TEST_TEXTS)
        novelty.append(
            NoveltyRow(
                order=order, n_runs=total, n_unseen=unseen, share_unseen=unseen / total
            )
        )

    growth: list[GrowthRow] = []
    for share in GROWTH_SHARES:
        part = TRAINING_CORPUS[: round(len(TRAINING_CORPUS) * share)]
        bigram = _fitted(2, DEFAULT_STRENGTH, part)
        read = bigram.evaluate(TEST_TEXTS)
        growth.append(
            GrowthRow(
                n_sentences=len(part),
                n_word_occurrences=sum(len(_words_of(text)) for text in part),
                vocabulary_size=bigram.vocabulary.n_tokens,
                n_unigram_types=bigram.counts.n_types(1),
                n_bigram_types=bigram.counts.n_types(2),
                n_trigram_types=_fitted(3, DEFAULT_STRENGTH, part).counts.n_types(3),
                test_perplexity=read.perplexity,
                n_unknown=read.n_unknown,
                unknown_share=read.unknown_share,
                knowing_nothing_perplexity=_fitted(2, KNOWING_NOTHING_STRENGTH, part)
                .evaluate(TEST_TEXTS)
                .perplexity,
            )
        )

    model = _fitted(2, DEFAULT_STRENGTH)
    return GramsResponse(
        sentence=RUNNING_SENTENCE,
        words=words,
        n_words=len(words),
        sentence_start=SENTENCE_START,
        sentence_end=SENTENCE_END,
        framings=framings,
        n_training_sentences=len(TRAINING_CORPUS),
        n_word_occurrences=len(training_words),
        n_distinct_words=len(set(training_words)),
        vocabulary_size=model.vocabulary.n_tokens,
        sparsity=sparsity,
        novelty=novelty,
        growth=growth,
    )


@router.get("/grams", response_model=GramsResponse)
def grams() -> GramsResponse:
    """The running sentence as runs of words, and how sparse the counts are."""
    return _grams()


# --------------------------------------------------------------------------
# /counts: the arithmetic small enough to check with a pencil.
# --------------------------------------------------------------------------


class UnigramRow(BaseModel):
    """One word, how often it occurred, and how often it stood before another."""

    word: str
    count: int
    n_times_followed: int


class BigramRow(BaseModel):
    """One pair of adjacent words and how often it occurred."""

    context: str
    word: str
    count: int


class ProbabilityRow(BaseModel):
    """One conditional probability under both rules, with the counts behind it."""

    context: str
    word: str
    count: int
    context_total: int
    unsmoothed: float
    smoothed: float


class SentenceScore(BaseModel):
    """One sentence scored under both rules, with the positions counted."""

    text: str
    n_predicted: int
    unsmoothed_probability: float | None
    unsmoothed_perplexity: float | None
    unsmoothed_is_finite: bool
    smoothed_log_probability: float
    smoothed_perplexity: float
    in_corpus: bool


class WalkStep(BaseModel):
    """One predicted position of the running sentence, with what it cost."""

    context: str
    word: str
    count: int
    context_total: int
    probability: float
    is_unknown: bool


class CountsResponse(BaseModel):
    """The small corpus in full, and the running sentence walked through."""

    corpus: list[str]
    vocabulary: list[str]
    vocabulary_size: int
    unigrams: list[UnigramRow]
    bigrams: list[BigramRow]
    n_predicted_positions: int
    probabilities: list[ProbabilityRow]
    sentences: list[SentenceScore]
    walk_strength: float
    walk: list[WalkStep]
    walk_perplexity: float
    walk_cross_entropy: float
    walk_n_unknown: int
    walk_n_predicted: int


@cache
def _counts() -> CountsResponse:
    unsmoothed = NGramLanguageModel(order=2, smoothing=MaximumLikelihood()).fit(
        SMALL_CORPUS
    )
    smoothed = NGramLanguageModel(
        order=2, smoothing=AdditiveSmoothing(pretended_count=1.0)
    ).fit(SMALL_CORPUS)
    table = unsmoothed.counts

    unigrams = [
        UnigramRow(
            word=gram.words[0],
            count=table.count_of(gram.words),
            n_times_followed=table.context_total(gram.words),
        )
        for gram in table.grams(1)
    ]
    bigrams = [
        BigramRow(
            context=gram.words[0], word=gram.words[1], count=table.count_of(gram.words)
        )
        for gram in table.grams(2)
    ]
    probabilities = [
        ProbabilityRow(
            context=context,
            word=word,
            count=table.count_of((context, word)),
            context_total=table.context_total((context,)),
            unsmoothed=unsmoothed.probability_of(word, [context]),
            smoothed=smoothed.probability_of(word, [context]),
        )
        for context, word in HAND_CHECKED_PAIRS
    ]

    sentences: list[SentenceScore] = []
    for text in HAND_CHECKED_SENTENCES:
        without = unsmoothed.evaluate([text])
        with_rule = smoothed.evaluate([text])
        sentences.append(
            SentenceScore(
                text=text,
                n_predicted=without.n_words,
                unsmoothed_probability=(
                    math.exp(without.log_probability) if without.is_finite else None
                ),
                unsmoothed_perplexity=_finite(without.perplexity),
                unsmoothed_is_finite=without.is_finite,
                smoothed_log_probability=with_rule.log_probability,
                smoothed_perplexity=with_rule.perplexity,
                in_corpus=text in SMALL_CORPUS,
            )
        )

    walker = _fitted(2, DEFAULT_STRENGTH)
    framed = padded(_known_words(walker, RUNNING_SENTENCE), 2)
    walk = [
        WalkStep(
            context=framed[position - 1],
            word=framed[position],
            count=walker.counts.count_of(framed[position - 1 : position + 1]),
            context_total=walker.counts.context_total((framed[position - 1],)),
            probability=walker.probability_of(framed[position], [framed[position - 1]]),
            is_unknown=framed[position] == UNKNOWN_WORD,
        )
        for position in range(1, len(framed))
    ]
    walked = walker.evaluate([RUNNING_SENTENCE])

    return CountsResponse(
        corpus=list(SMALL_CORPUS),
        vocabulary=list(unsmoothed.vocabulary),
        vocabulary_size=unsmoothed.vocabulary.n_tokens,
        unigrams=unigrams,
        bigrams=bigrams,
        n_predicted_positions=table.total(1),
        probabilities=probabilities,
        sentences=sentences,
        walk_strength=DEFAULT_STRENGTH,
        walk=walk,
        walk_perplexity=walked.perplexity,
        walk_cross_entropy=walked.cross_entropy,
        walk_n_unknown=walked.n_unknown,
        walk_n_predicted=walked.n_words,
    )


@router.get("/counts", response_model=CountsResponse)
def counts() -> CountsResponse:
    """Three sentences with every count listed, and one sentence walked through."""
    return _counts()


# --------------------------------------------------------------------------
# /smoothing: how much to take from what was seen.
# --------------------------------------------------------------------------


class StrengthRow(BaseModel):
    """One additive constant, scored on the text it was chosen on and one it was not."""

    strength: float
    tuning_perplexity: float
    test_perplexity: float


class StrengthSweep(BaseModel):
    """One window width swept over every constant, with the winner named."""

    order: int
    rows: list[StrengthRow]
    best_strength: float
    best_tuning_perplexity: float
    best_test_perplexity: float
    laplace_tuning_perplexity: float
    laplace_test_perplexity: float


class CertainRow(BaseModel):
    """What one constant leaves of a context the counts were certain about."""

    strength: float
    probability: float
    entropy: float


class LimitRow(BaseModel):
    """What a very large constant does to the whole test text."""

    strength: float
    test_perplexity: float


class WordShare(BaseModel):
    """One word and what the model gives it."""

    word: str
    probability: float


class ContextAnswer(BaseModel):
    """A whole next-word answer for one context at one strength."""

    label: str
    context: list[str]
    strength: float
    entropy: float
    n_words: int
    context_total: int
    n_followers: int
    top: list[WordShare]


class SmoothingResponse(BaseModel):
    """Everything the page shows about the additive constant."""

    strengths: list[float]
    sweeps: list[StrengthSweep]
    certain_context: str
    certain_word: str
    certain_count: int
    certain_context_total: int
    certain_rows: list[CertainRow]
    certain_unsmoothed_probability: float
    certain_unsmoothed_entropy: float
    limits: list[LimitRow]
    vocabulary_size: int
    flat_probability: float
    flat_entropy: float
    answers: list[ContextAnswer]


@cache
def _smoothing() -> SmoothingResponse:
    sweeps: list[StrengthSweep] = []
    for order in SWEEP_ORDERS:
        rows: list[StrengthRow] = []
        for strength in STRENGTH_SWEEP:
            model = _fitted(order, strength)
            rows.append(
                StrengthRow(
                    strength=strength,
                    tuning_perplexity=model.evaluate(TUNING_TEXTS).perplexity,
                    test_perplexity=model.evaluate(TEST_TEXTS).perplexity,
                )
            )
        best = min(rows, key=lambda row: row.tuning_perplexity)
        laplace = next(row for row in rows if row.strength == 1.0)
        sweeps.append(
            StrengthSweep(
                order=order,
                rows=rows,
                best_strength=best.strength,
                best_tuning_perplexity=best.tuning_perplexity,
                best_test_perplexity=best.test_perplexity,
                laplace_tuning_perplexity=laplace.tuning_perplexity,
                laplace_test_perplexity=laplace.test_perplexity,
            )
        )

    certain_rows = [
        CertainRow(
            strength=strength,
            probability=_fitted(2, strength).probability_of(
                CERTAIN_WORD, [CERTAIN_CONTEXT]
            ),
            entropy=_fitted(2, strength)
            .next_word_distribution([CERTAIN_CONTEXT])
            .entropy,
        )
        for strength in CERTAIN_STRENGTHS
    ]
    unsmoothed = _fitted(2, None)
    limits = [
        LimitRow(
            strength=strength,
            test_perplexity=_fitted(2, strength).evaluate(TEST_TEXTS).perplexity,
        )
        for strength in LIMIT_STRENGTHS
    ]

    answers: list[ContextAnswer] = []
    for label, context in DISTRIBUTION_CONTEXTS:
        for strength in DISTRIBUTION_STRENGTHS:
            model = _fitted(2, strength)
            answer = model.next_word_distribution(list(context))
            answers.append(
                ContextAnswer(
                    label=label,
                    context=list(context),
                    strength=strength,
                    entropy=answer.entropy,
                    n_words=answer.n_words,
                    context_total=model.counts.context_total(context),
                    n_followers=model.counts.n_followers(context),
                    top=[
                        WordShare(word=entry.word, probability=entry.probability)
                        for entry in answer.most_likely(N_DISTRIBUTION_WORDS)
                    ],
                )
            )

    vocabulary_size = unsmoothed.vocabulary.n_tokens
    return SmoothingResponse(
        strengths=list(STRENGTH_SWEEP),
        sweeps=sweeps,
        certain_context=CERTAIN_CONTEXT,
        certain_word=CERTAIN_WORD,
        certain_count=unsmoothed.counts.count_of((CERTAIN_CONTEXT, CERTAIN_WORD)),
        certain_context_total=unsmoothed.counts.context_total((CERTAIN_CONTEXT,)),
        certain_rows=certain_rows,
        certain_unsmoothed_probability=unsmoothed.probability_of(
            CERTAIN_WORD, [CERTAIN_CONTEXT]
        ),
        certain_unsmoothed_entropy=unsmoothed.next_word_distribution(
            [CERTAIN_CONTEXT]
        ).entropy,
        limits=limits,
        vocabulary_size=vocabulary_size,
        flat_probability=1.0 / vocabulary_size,
        flat_entropy=math.log2(vocabulary_size),
        answers=answers,
    )


@router.get("/smoothing", response_model=SmoothingResponse)
def smoothing() -> SmoothingResponse:
    """The additive constant swept, and what each setting leaves of the counts."""
    return _smoothing()


# --------------------------------------------------------------------------
# /scores: judging a model on text it was not fitted on.
# --------------------------------------------------------------------------


class OrderRow(BaseModel):
    """One window width, judged four ways on two texts."""

    order: int
    n_types: int
    unsmoothed_training_perplexity: float
    unsmoothed_test_perplexity: float | None
    unsmoothed_test_is_finite: bool
    laplace_training_perplexity: float
    laplace_test_perplexity: float
    laplace_test_cross_entropy: float
    tuned_strength: float
    tuned_training_perplexity: float
    tuned_test_perplexity: float
    unknown_share: float


class ReversedRow(BaseModel):
    """The same words read forwards and backwards, under one width."""

    order: int
    forwards_perplexity: float
    reversed_perplexity: float


class DependencyRow(BaseModel):
    """What one width makes of a verb whose subject stands five words back."""

    order: int
    n_context_words: int
    agreeing_probability: float
    disagreeing_probability: float


class RunningRow(BaseModel):
    """The running sentence read at one width and one strength."""

    order: int
    strength: float | None
    perplexity: float | None
    cross_entropy: float | None
    is_finite: bool


class ScoresResponse(BaseModel):
    """Perplexity against width, and the two things width cannot buy."""

    n_test_texts: int
    n_predicted_positions: int
    orders: list[OrderRow]
    reversed_rows: list[ReversedRow]
    reversed_example: str
    dependency_corpus: list[str]
    dependency_context: list[str]
    agreeing_word: str
    disagreeing_word: str
    dependency_rows: list[DependencyRow]
    running_rows: list[RunningRow]


@cache
def _scores() -> ScoresResponse:
    sweeps = {sweep.order: sweep for sweep in _smoothing().sweeps}
    orders: list[OrderRow] = []
    for order in SWEEP_ORDERS:
        unsmoothed = _fitted(order, None)
        laplace = _fitted(order, 1.0)
        tuned_strength = sweeps[order].best_strength
        tuned = _fitted(order, tuned_strength)
        unsmoothed_test = unsmoothed.evaluate(TEST_TEXTS)
        laplace_test = laplace.evaluate(TEST_TEXTS)
        orders.append(
            OrderRow(
                order=order,
                n_types=laplace.counts.n_types(order),
                unsmoothed_training_perplexity=unsmoothed.evaluate(
                    TRAINING_CORPUS
                ).perplexity,
                unsmoothed_test_perplexity=_finite(unsmoothed_test.perplexity),
                unsmoothed_test_is_finite=unsmoothed_test.is_finite,
                laplace_training_perplexity=laplace.evaluate(
                    TRAINING_CORPUS
                ).perplexity,
                laplace_test_perplexity=laplace_test.perplexity,
                laplace_test_cross_entropy=laplace_test.cross_entropy,
                tuned_strength=tuned_strength,
                tuned_training_perplexity=tuned.evaluate(TRAINING_CORPUS).perplexity,
                tuned_test_perplexity=tuned.evaluate(TEST_TEXTS).perplexity,
                unknown_share=laplace_test.unknown_share,
            )
        )

    backwards = [" ".join(reversed(_words_of(text))) for text in TEST_TEXTS]
    reversed_rows = [
        ReversedRow(
            order=order,
            forwards_perplexity=_fitted(order, DEFAULT_STRENGTH)
            .evaluate(TEST_TEXTS)
            .perplexity,
            reversed_perplexity=_fitted(order, DEFAULT_STRENGTH)
            .evaluate(backwards)
            .perplexity,
        )
        for order in (1, 2, 3)
    ]

    dependency_rows: list[DependencyRow] = []
    for order in (2, 3, 4, 5, 6):
        model = NGramLanguageModel(order=order, smoothing=MaximumLikelihood()).fit(
            LONG_DEPENDENCY_CORPUS
        )
        dependency_rows.append(
            DependencyRow(
                order=order,
                n_context_words=order - 1,
                agreeing_probability=model.probability_of(
                    LONG_DEPENDENCY_WORDS[0], list(LONG_DEPENDENCY_CONTEXT)
                ),
                disagreeing_probability=model.probability_of(
                    LONG_DEPENDENCY_WORDS[1], list(LONG_DEPENDENCY_CONTEXT)
                ),
            )
        )

    running_rows: list[RunningRow] = []
    for order in (1, 2, 3):
        for strength in (None, 1.0, DEFAULT_STRENGTH):
            read = _fitted(order, strength).evaluate([RUNNING_SENTENCE])
            running_rows.append(
                RunningRow(
                    order=order,
                    strength=strength,
                    perplexity=_finite(read.perplexity),
                    cross_entropy=_finite(read.cross_entropy),
                    is_finite=read.is_finite,
                )
            )

    return ScoresResponse(
        n_test_texts=len(TEST_TEXTS),
        n_predicted_positions=_fitted(2, 1.0).evaluate(TEST_TEXTS).n_words,
        orders=orders,
        reversed_rows=reversed_rows,
        reversed_example=backwards[0],
        dependency_corpus=list(LONG_DEPENDENCY_CORPUS),
        dependency_context=list(LONG_DEPENDENCY_CONTEXT),
        agreeing_word=LONG_DEPENDENCY_WORDS[0],
        disagreeing_word=LONG_DEPENDENCY_WORDS[1],
        dependency_rows=dependency_rows,
        running_rows=running_rows,
    )


@router.get("/scores", response_model=ScoresResponse)
def scores() -> ScoresResponse:
    """Perplexity against window width, and two things a wider window cannot buy."""
    return _scores()


# --------------------------------------------------------------------------
# /generation: the same distribution drawn from rather than read off.
# --------------------------------------------------------------------------


class DrawnByOrder(BaseModel):
    """Three sentences drawn at one width, and how often the training text returns."""

    order: int
    samples: list[str]
    n_draws: int
    n_exact_copies: int
    share_exact_copies: float


class DrawnByTemperature(BaseModel):
    """Two sentences drawn at one flattening of the answer."""

    temperature: float
    samples: list[str]


class GenerationResponse(BaseModel):
    """What the model writes, by width and by how peaked each draw is."""

    by_order: list[DrawnByOrder]
    temperature_order: int
    temperature_strength: float
    by_temperature: list[DrawnByTemperature]
    n_training_sentences: int


@cache
def _generation() -> GenerationResponse:
    written = {" ".join(_words_of(text)) for text in TRAINING_CORPUS}
    by_order: list[DrawnByOrder] = []
    for order in GENERATION_ORDERS:
        model = _fitted(order, None)
        copies = sum(
            1
            for seed in range(N_GENERATION_DRAWS)
            if " ".join(model.generate(max_words=MAX_GENERATED_WORDS, random_seed=seed))
            in written
        )
        by_order.append(
            DrawnByOrder(
                order=order,
                samples=[
                    " ".join(
                        model.generate(max_words=MAX_GENERATED_WORDS, random_seed=seed)
                    )
                    for seed in GENERATION_SEEDS
                ],
                n_draws=N_GENERATION_DRAWS,
                n_exact_copies=copies,
                share_exact_copies=copies / N_GENERATION_DRAWS,
            )
        )

    peaked = _fitted(3, DEFAULT_STRENGTH)
    by_temperature = [
        DrawnByTemperature(
            temperature=temperature,
            samples=[
                " ".join(
                    peaked.generate(
                        max_words=MAX_GENERATED_WORDS,
                        random_seed=seed,
                        temperature=temperature,
                    )
                )
                for seed in (0, 1)
            ],
        )
        for temperature in TEMPERATURES
    ]

    return GenerationResponse(
        by_order=by_order,
        temperature_order=3,
        temperature_strength=DEFAULT_STRENGTH,
        by_temperature=by_temperature,
        n_training_sentences=len(TRAINING_CORPUS),
    )


@router.get("/generation", response_model=GenerationResponse)
def generation() -> GenerationResponse:
    """Sentences drawn from the model, by window width and by flattening."""
    return _generation()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class RuleChoice(StrEnum):
    """Whether the counts are divided as they stand or padded first."""

    COUNTS_AS_THEY_STAND = "counts_as_they_stand"
    ADD_A_CONSTANT = "add_a_constant"


class FitRequest(BaseModel):
    """A window width, a way of turning counts into probabilities, and a text."""

    order: int = Field(default=2, ge=1, le=MAX_ORDER)
    rule: RuleChoice = RuleChoice.ADD_A_CONSTANT
    strength: float = Field(default=DEFAULT_STRENGTH, gt=0.0, le=MAX_SMOOTHING_STRENGTH)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)
    random_seed: int = Field(default=0, ge=0, le=9999)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything the playground shows for one setting on one text."""

    order: int
    rule: RuleChoice
    strength: float | None
    text: str
    words: list[str]
    framed: list[str]
    vocabulary_size: int
    n_types: int
    steps: list[WalkStep]
    n_predicted: int
    n_unknown: int
    unknown_words: list[str]
    n_unseen_runs: int
    log_probability: float | None
    perplexity: float | None
    cross_entropy: float | None
    is_finite: bool
    generated: list[str]
    next_context: list[str]
    next_is_defined: bool
    next_entropy: float | None
    next_top: list[WordShare]


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Read one text at one setting, and draw one sentence at the same setting."""
    strength = (
        None if request.rule is RuleChoice.COUNTS_AS_THEY_STAND else request.strength
    )
    model = _fitted(request.order, strength)
    try:
        known = _known_words(model, request.text)
    except MLLibError:
        known = []
    if not known:
        raise HTTPException(status_code=422, detail="There are no words here to read.")

    try:
        framed = padded(known, request.order)
        steps = [
            WalkStep(
                context=" ".join(framed[position - request.order + 1 : position]),
                word=framed[position],
                count=model.counts.count_of(
                    framed[position - request.order + 1 : position + 1]
                ),
                context_total=model.counts.context_total(
                    framed[position - request.order + 1 : position]
                ),
                probability=model.probability_of(
                    framed[position],
                    list(framed[position - request.order + 1 : position]),
                ),
                is_unknown=framed[position] == UNKNOWN_WORD,
            )
            for position in range(request.order - 1, len(framed))
        ]
        read = model.evaluate([request.text])
        drawn = model.generate(
            max_words=MAX_GENERATED_WORDS, random_seed=request.random_seed
        )
        next_context = list(framed[len(framed) - request.order : len(framed) - 1])
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    # A context nothing ever followed has no answer at all under the unsmoothed
    # rule, since every word of it would be zero, and that is the page's whole
    # subject rather than a failure worth refusing the request over.
    try:
        answer = model.next_word_distribution(next_context)
        next_entropy: float | None = answer.entropy
        next_top = [
            WordShare(word=entry.word, probability=entry.probability)
            for entry in answer.most_likely(N_DISTRIBUTION_WORDS)
        ]
    except MLLibError:
        next_entropy, next_top = None, []

    return FitResponse(
        order=request.order,
        rule=request.rule,
        strength=strength,
        text=request.text,
        words=_words_of(request.text),
        framed=list(framed),
        vocabulary_size=model.vocabulary.n_tokens,
        n_types=model.counts.n_types(request.order),
        steps=steps,
        n_predicted=read.n_words,
        n_unknown=read.n_unknown,
        unknown_words=[
            word for word in _words_of(request.text) if word not in model.vocabulary
        ],
        n_unseen_runs=sum(1 for step in steps if step.count == 0),
        log_probability=_finite(read.log_probability),
        perplexity=_finite(read.perplexity),
        cross_entropy=_finite(read.cross_entropy),
        is_finite=read.is_finite,
        generated=list(drawn),
        next_context=next_context,
        next_is_defined=next_entropy is not None,
        next_entropy=next_entropy,
        next_top=next_top,
    )
