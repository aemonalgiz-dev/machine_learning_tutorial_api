"""The WordPiece page's endpoints.

Four of them, one per family of widgets rather than one per number.
``/scoring`` is everything about the score that replaces the pair count: the
four-word corpus's symbol counts and every candidate pair under both rules, the
tie the two top candidates reach, the two merge orders side by side on both
corpora, the corpus length each order reaches at a given number of merges, and
the same pair scores on the eighteen sentences for the scatter. ``/spelling``
is everything about marking a piece that continues a word: the two alphabets,
the vocabulary-size sweep beside a count-scored fit of the same size, the
running sentence cut at four sizes, and the words each spelling can and cannot
reach. ``/limits`` carries the closing part's three measurements. ``/fit`` is
the playground.

Two things here are the page's own rather than the library's, and both are
comparisons the library has no reason to make. The first is scoring by pair
count over the *word-initial* spelling, which is byte pair encoding's rule
applied to WordPiece's symbols, so that a difference between the two orders is
a difference of score and not of spelling. The second is the reading of the
minimum pair count as a test on the winner that ends the walk, which is the
mistake the shared trainer documents having avoided; showing what it would have
learned means running it. Everything else is a fit.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from itertools import product
from math import log

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    PairScoring,
    SpelledWord,
    alphabet_of,
    best_pair,
    learn_merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.word_piece import (
    WORD_INITIAL_MARKER,
    WordPiece,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/wordpiece", tags=["wordpiece"])


MAX_VOCABULARY_SIZE = 600
"""The largest vocabulary a request may ask for.

Both page corpora run out of pairs worth merging long before this, so the bound
never truncates anything the page shows; it exists so a crafted request cannot
turn a teaching endpoint into a long fit.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to encode."""

CONTINUATION = "##"
"""The mark this page writes on a piece that continues a word."""

END_OF_WORD = "</w>"
"""The mark the count-scored fits on this page write on a word's last piece."""

UNKNOWN = "[UNK]"
"""What either fit answers when it cannot spell something."""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every merge is checkable.

The corpus the original subword paper worked its example on, and the corpus the
from-scratch trainer's own spec is pinned against, so every number the page
quotes from it is a number a test already holds.
"""

FOUR_WORD_CORPUS: list[str] = [
    " ".join(word for word, count in FOUR_WORD_COUNTS.items() for _ in range(count))
]

UNSEEN_WORD = "lowest"
"""A word the four-word corpus never contains, spellable from what it taught."""

SENTENCE_CORPUS: list[str] = [
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
]
"""Eighteen sentences of ordinary English, none of them the running sentence.

Written so that every character of the running sentence appears somewhere,
including the apostrophe, the hyphen and both capitals, and so that the words
around it are plausible neighbours of the ones it uses. The same eighteen
sentences the sibling page on count-scored merging carries, kept here rather
than imported so the two pages cannot move each other's numbers.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

NEIGHBOUR_TEXT = "The dial was set and the diode tested."
"""A sentence in the corpus's own register, for the pieces-per-word comparison."""

SWEEP_SIZES = (52, 60, 74, 90, 110, 130, 150, 200)
"""The vocabulary sizes the size table is drawn at.

Starts at 52 because that is the floor for the count-scored fit as well, whose
alphabet holds two symbols more, so both columns of the table exist at every row.
"""

CUT_SIZES = (52, 90, 130, 200)
"""The four sizes the page prints the running sentence's pieces at."""

FOUR_WORD_BUDGETS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
SENTENCE_BUDGETS = (0, 5, 10, 20, 40, 60, 80, 100, 108)
"""How many merges the length comparison is drawn at, on each corpus."""

COMPARED_WORDS = ("Alvarez", "old", "you", "zebra", "hour")
"""Words that separate the two markings, both directions represented."""

FOUR_WORD_COMPARED = ("lowest", "lox", "old")
"""The same on the small corpus, where the whole vocabulary fits on a screen."""

BOUND_REPEATS = (2, 3, 5, 9)
"""How often the bound pair occurs in the closing part's constructed corpus."""

GREEDY_LONGEST_WORD = 4
"""How long a word the exhaustive cut check runs to.

Ten letters to the fourth is 11,110 words and takes under a second; the fifth
power is ten times as many for the same answer.
"""

MINIMUM_UNDER_TEST = 3
"""The minimum pair count the two readings are compared at."""


class CorpusChoice(StrEnum):
    """Which of the page's two corpora a request wants fitted."""

    FOUR_WORDS = "four_words"
    SENTENCES = "sentences"


CORPORA: dict[CorpusChoice, list[str]] = {
    CorpusChoice.FOUR_WORDS: FOUR_WORD_CORPUS,
    CorpusChoice.SENTENCES: SENTENCE_CORPUS,
}

CORPUS_LABELS: dict[CorpusChoice, str] = {
    CorpusChoice.FOUR_WORDS: "Four words, repeated",
    CorpusChoice.SENTENCES: "Eighteen sentences",
}


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("vocabulary_size=", "a vocabulary of "),
    ("minimum_pair_frequency", "the fewest times a pair may be seen"),
    ("max_word_characters", "the longest word that may be looked up"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("WhitespacePreTokenizer", "the rule that finds the words"),
    ("the continuation prefix", "the mark that says a piece continues a word"),
    ("a different prefix", "a different mark"),
    ("the unknown token", "the one token for anything unspellable"),
    ("WordPiece reserves", "this scheme reserves"),
    ("WordPiece", "this scheme"),
    ("BytePairEncoding", "this scheme"),
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


def _marked_symbols(word: str) -> tuple[str, ...]:
    """A word with its first symbol marked, which is the spelling merges run over."""
    return (WORD_INITIAL_MARKER + word[0], *word[1:])


def _ended_symbols(word: str) -> tuple[str, ...]:
    """A word with its last symbol marked, which is the count-scored spelling."""
    return (*word[:-1], word[-1] + END_OF_WORD)


def _piece_of(symbol: str) -> str:
    """One internal symbol as a reader sees it, with the continuation mark on it."""
    if symbol.startswith(WORD_INITIAL_MARKER):
        return symbol[len(WORD_INITIAL_MARKER) :]
    return CONTINUATION + symbol


def _word_counts(corpus: Sequence[str]) -> dict[str, int]:
    counts = Corpus.of(list(corpus)).word_counts(WhitespacePreTokenizer())
    return {word_count.word: word_count.count for word_count in counts}


def _spelled(counts: dict[str, int]) -> list[SpelledWord]:
    return [SpelledWord(_marked_symbols(word), count) for word, count in counts.items()]


def _symbol_counts(spelled: Sequence[SpelledWord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for word in spelled:
        for symbol in word:
            counter[symbol] += word.count
    return counter


def _pair_counts(spelled: Sequence[SpelledWord]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for word in spelled:
        for left, right in zip(word.symbols, word.symbols[1:], strict=False):
            counter[(left, right)] += word.count
    return counter


def _length_after(counts: dict[str, int], merges: Merges) -> int:
    """How many symbols the whole corpus holds once these merges are applied."""
    current = _spelled(counts)
    for merge in merges:
        current = [word.with_pair_merged(merge) for word in current]
    return sum(len(word) * word.count for word in current)


# --------------------------------------------------------------------------
# /scoring: the count against the ratio.
# --------------------------------------------------------------------------


class SpelledEntry(BaseModel):
    """One distinct word, how often it occurs, and its pieces before any merge."""

    word: str
    count: int
    pieces: list[str]


class SymbolCount(BaseModel):
    """One symbol of the alphabet and how often the corpus holds it."""

    piece: str
    count: int


class PairScore(BaseModel):
    """One candidate pair under both rules.

    ``count`` is how often the two are adjacent, weighted by how often each word
    occurs; ``left_count`` and ``right_count`` are how often each half occurs
    anywhere; ``score`` is the count divided by their product.

    ``total_gain`` is what the whole corpus's log likelihood would gain from the
    merge, which is the count multiplied by the log of the score times the
    corpus's symbol total. The rule compares ``score``, which is the gain one
    occurrence makes, so the two orderings need not agree and on the small
    corpus they do not.
    """

    left: str
    right: str
    count: int
    left_count: int
    right_count: int
    score: float
    total_gain: float


class MergeRow(BaseModel):
    """One merge, with the count and the score it carried when it was chosen."""

    rank: int
    left: str
    right: str
    merged: str
    count: int
    score: float


class OrderPair(BaseModel):
    """The same corpus merged under both rules, and where the two orders part.

    ``first_difference`` is the earliest rank at which the two chose different
    pairs, and is ``-1`` if they never do.
    """

    corpus: str
    corpus_label: str
    by_count: list[MergeRow]
    by_score: list[MergeRow]
    first_difference: int


class LengthRow(BaseModel):
    """How long the corpus is after a given number of merges, under each rule."""

    asked: int
    by_count_learned: int
    by_count_length: int
    by_score_learned: int
    by_score_length: int


class LengthGroup(BaseModel):
    """One corpus's length comparison, both rules over the same spelling."""

    corpus: str
    corpus_label: str
    start_length: int
    rows: list[LengthRow]


class TieView(BaseModel):
    """Two candidates the score cannot separate, and the character that does.

    The pairs are compared as text in the spelling the merges are learned in,
    where a word's first piece carries a mark of its own, so the comparison ends
    up resting on that mark's position in the character table.
    """

    tied: list[PairScore]
    taken: str
    rejected: str
    word_start_mark: str
    word_start_mark_codepoint: int
    rival_character: str
    rival_codepoint: int


class ScoringResponse(BaseModel):
    """Everything the page says about how a pair is chosen."""

    four_word_corpus: list[str]
    four_word_occurrences: int
    four_word_symbol_total: int
    spellings: list[SpelledEntry]
    symbols: list[SymbolCount]
    pairs: list[PairScore]
    four_word_best_total_gain: PairScore
    tie: TieView
    orders: list[OrderPair]
    lengths: list[LengthGroup]
    sentence_pairs: list[PairScore]
    sentence_largest_count: PairScore
    sentence_best_score: PairScore
    sentence_largest_count_rank: int
    sentence_n_candidates: int


def _pair_scores(spelled: Sequence[SpelledWord], minimum: int) -> list[PairScore]:
    symbols = _symbol_counts(spelled)
    pairs = _pair_counts(spelled)
    total = sum(symbols.values())
    scored = [
        PairScore(
            left=_piece_of(left),
            right=_piece_of(right),
            count=count,
            left_count=symbols[left],
            right_count=symbols[right],
            score=count / (symbols[left] * symbols[right]),
            total_gain=count * log(count * total / (symbols[left] * symbols[right])),
        )
        for (left, right), count in pairs.items()
        if count >= minimum
    ]
    return sorted(scored, key=lambda pair: (-pair.score, pair.left, pair.right))


def _merge_rows(
    counts: dict[str, int], budget: int, minimum: int, scoring: PairScoring
) -> list[MergeRow]:
    current = _spelled(counts)
    rows: list[MergeRow] = []
    for rank in range(budget):
        symbols = _symbol_counts(current)
        merge = best_pair(current, scoring, minimum)
        if merge is None:
            break
        rows.append(
            MergeRow(
                rank=rank,
                left=_piece_of(merge.left),
                right=_piece_of(merge.right),
                merged=_piece_of(merge.merged),
                count=int(merge.score),
                score=int(merge.score) / (symbols[merge.left] * symbols[merge.right]),
            )
        )
        current = [word.with_pair_merged(merge) for word in current]
    return rows


def _order_pair(
    choice: CorpusChoice, counts: dict[str, int], budget: int, minimum: int
) -> OrderPair:
    by_count = _merge_rows(counts, budget, minimum, PairScoring.FREQUENCY)
    by_score = _merge_rows(counts, budget, minimum, PairScoring.LIKELIHOOD)
    difference = -1
    for rank in range(min(len(by_count), len(by_score))):
        if (by_count[rank].left, by_count[rank].right) != (
            by_score[rank].left,
            by_score[rank].right,
        ):
            difference = rank
            break
    return OrderPair(
        corpus=choice.value,
        corpus_label=CORPUS_LABELS[choice],
        by_count=by_count,
        by_score=by_score,
        first_difference=difference,
    )


def _length_group(
    choice: CorpusChoice,
    counts: dict[str, int],
    budgets: Sequence[int],
    minimum: int,
) -> LengthGroup:
    start = _spelled(counts)
    rows: list[LengthRow] = []
    for asked in budgets:
        by_count = learn_merges(start, asked, minimum, PairScoring.FREQUENCY)
        by_score = learn_merges(start, asked, minimum, PairScoring.LIKELIHOOD)
        rows.append(
            LengthRow(
                asked=asked,
                by_count_learned=by_count.n_merges,
                by_count_length=_length_after(counts, by_count),
                by_score_learned=by_score.n_merges,
                by_score_length=_length_after(counts, by_score),
            )
        )
    return LengthGroup(
        corpus=choice.value,
        corpus_label=CORPUS_LABELS[choice],
        start_length=_length_after(counts, Merges([])),
        rows=rows,
    )


def _first_difference(left: str, right: str) -> tuple[str, str]:
    """The first pair of characters at which two strings differ."""
    for one, other in zip(left, right, strict=False):
        if one != other:
            return (one, other)
    return (left[len(right) :] or " ", right[len(left) :] or " ")


def _tie_view(spelled: Sequence[SpelledWord]) -> TieView:
    """The candidates at the top of the score, and the character separating them.

    Two pairs reaching the same score are compared as text in the spelling the
    merges run over, so what actually decides is the first character at which
    the two comparisons differ, which here is a word's start mark against an
    ordinary letter.
    """
    symbols = _symbol_counts(spelled)
    pairs = _pair_counts(spelled)
    best = max(
        count / (symbols[left] * symbols[right])
        for (left, right), count in pairs.items()
    )
    tied = sorted(
        pair
        for pair, count in pairs.items()
        if count / (symbols[pair[0]] * symbols[pair[1]]) == best
    )
    taken, runner_up = tied[0], tied[1]
    winning, losing = _first_difference("".join(taken), "".join(runner_up))
    return TieView(
        tied=[
            pair
            for pair in _pair_scores(spelled, minimum=1)
            if (pair.left, pair.right)
            in {(_piece_of(left), _piece_of(right)) for left, right in tied}
        ],
        taken=f"{_piece_of(taken[0])} + {_piece_of(taken[1])}",
        rejected=f"{_piece_of(runner_up[0])} + {_piece_of(runner_up[1])}",
        word_start_mark=losing,
        word_start_mark_codepoint=ord(losing),
        rival_character=winning,
        rival_codepoint=ord(winning),
    )


@cache
def _scoring() -> ScoringResponse:
    four = FOUR_WORD_COUNTS
    spelled = _spelled(four)
    sentences = _word_counts(SENTENCE_CORPUS)
    sentence_pairs = _pair_scores(_spelled(sentences), minimum=2)
    largest = max(sentence_pairs, key=lambda pair: (pair.count, -pair.score))
    ranked = [pair for pair in sentence_pairs].index(largest) + 1

    return ScoringResponse(
        four_word_corpus=FOUR_WORD_CORPUS,
        four_word_occurrences=sum(four.values()),
        four_word_symbol_total=sum(_symbol_counts(spelled).values()),
        spellings=[
            SpelledEntry(
                word=word,
                count=count,
                pieces=[_piece_of(symbol) for symbol in _marked_symbols(word)],
            )
            for word, count in four.items()
        ],
        symbols=[
            SymbolCount(piece=_piece_of(symbol), count=count)
            for symbol, count in sorted(_symbol_counts(spelled).items())
        ],
        pairs=_pair_scores(spelled, minimum=1),
        four_word_best_total_gain=max(
            _pair_scores(spelled, minimum=1), key=lambda pair: pair.total_gain
        ),
        tie=_tie_view(spelled),
        orders=[
            _order_pair(CorpusChoice.FOUR_WORDS, four, budget=12, minimum=1),
            _order_pair(CorpusChoice.SENTENCES, sentences, budget=15, minimum=2),
        ],
        lengths=[
            _length_group(CorpusChoice.FOUR_WORDS, four, FOUR_WORD_BUDGETS, 1),
            _length_group(CorpusChoice.SENTENCES, sentences, SENTENCE_BUDGETS, 2),
        ],
        sentence_pairs=sentence_pairs,
        sentence_largest_count=largest,
        sentence_best_score=sentence_pairs[0],
        sentence_largest_count_rank=ranked,
        sentence_n_candidates=len(sentence_pairs),
    )


@router.get("/scoring", response_model=ScoringResponse)
def scoring() -> ScoringResponse:
    """The pair count against the ratio, on both of the page's corpora."""
    return _scoring()


# --------------------------------------------------------------------------
# /spelling: what marking the continuation changes.
# --------------------------------------------------------------------------


class SizeRow(BaseModel):
    """One vocabulary size, and what the corpus and the sentence cost at it.

    ``asked`` is what the fit was asked for and ``learned`` what it managed; the
    two part once the corpus has no pair left that occurs twice.
    """

    asked: int
    learned: int
    n_merges: int
    corpus_pieces: int
    sentence_pieces: int
    n_unknown: int
    round_trip_exact: bool


class SentenceCut(BaseModel):
    """The running sentence as one fit cut it, and what came back."""

    vocabulary_size: int
    learned: int
    n_merges: int
    pieces: list[str]
    n_pieces: int
    decoded: str
    round_trip_exact: bool


class WordPair(BaseModel):
    """One word cut by both spellings, so the two refusals can be compared."""

    word: str
    marked: list[str]
    marked_unknown: bool
    ended: list[str]
    ended_unknown: bool


class SpellingResponse(BaseModel):
    """The two spellings, the sweep, the sentence, and who can spell what."""

    sentence: str
    corpus: list[str]
    n_distinct_words: int
    n_word_occurrences: int
    marked_alphabet: list[str]
    ended_alphabet_size: int
    four_word_marked_alphabet: list[str]
    four_word_ended_alphabet: list[str]
    character_pieces: int
    rows: list[SizeRow]
    count_rows: list[SizeRow]
    cuts: list[SentenceCut]
    distinct_characters: int
    never_opens: list[str]
    never_ends: list[str]
    comparisons: list[WordPair]
    four_word_comparisons: list[WordPair]


def _size_row(asked: int, marked: bool) -> SizeRow:
    if marked:
        model: WordPiece | BytePairEncoding = WordPiece(vocabulary_size=asked).fit(
            SENTENCE_CORPUS
        )
    else:
        model = BytePairEncoding(byte_fallback=False, vocabulary_size=asked).fit(
            SENTENCE_CORPUS
        )
    encoding = model.encode(RUNNING_SENTENCE)
    return SizeRow(
        asked=asked,
        learned=model.vocabulary.n_tokens,
        n_merges=model.n_merges,
        corpus_pieces=sum(model.encode(text).n_tokens for text in SENTENCE_CORPUS),
        sentence_pieces=encoding.n_tokens,
        n_unknown=sum(1 for piece in encoding.texts if piece == UNKNOWN),
        round_trip_exact=model.decode(encoding.ids) == RUNNING_SENTENCE,
    )


def _word_pair(word: str, marked: WordPiece, ended: BytePairEncoding) -> WordPair:
    marked_pieces = list(marked.encode(word).texts)
    ended_pieces = list(ended.encode(word).texts)
    return WordPair(
        word=word,
        marked=marked_pieces,
        marked_unknown=UNKNOWN in marked_pieces,
        ended=ended_pieces,
        ended_unknown=UNKNOWN in ended_pieces,
    )


@cache
def _spelling() -> SpellingResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    spelled = _spelled(counts)
    ended = alphabet_of(
        [SpelledWord(_ended_symbols(word), count) for word, count in counts.items()]
    )
    characters = {character for word in counts for character in word}
    opens = {word[0] for word in counts}
    ends = {word[-1] for word in counts}

    marked_model = WordPiece(vocabulary_size=MAX_VOCABULARY_SIZE).fit(SENTENCE_CORPUS)
    ended_model = BytePairEncoding(
        byte_fallback=False, vocabulary_size=MAX_VOCABULARY_SIZE
    ).fit(SENTENCE_CORPUS)
    four_marked = WordPiece(vocabulary_size=22).fit(FOUR_WORD_CORPUS)
    four_ended = BytePairEncoding(byte_fallback=False, vocabulary_size=22).fit(
        FOUR_WORD_CORPUS
    )

    cuts: list[SentenceCut] = []
    for asked in CUT_SIZES:
        model = WordPiece(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        encoding = model.encode(RUNNING_SENTENCE)
        decoded = model.decode(encoding.ids)
        cuts.append(
            SentenceCut(
                vocabulary_size=asked,
                learned=model.vocabulary.n_tokens,
                n_merges=model.n_merges,
                pieces=list(encoding.texts),
                n_pieces=encoding.n_tokens,
                decoded=decoded,
                round_trip_exact=decoded == RUNNING_SENTENCE,
            )
        )

    four_counts = FOUR_WORD_COUNTS
    return SpellingResponse(
        sentence=RUNNING_SENTENCE,
        corpus=SENTENCE_CORPUS,
        n_distinct_words=len(counts),
        n_word_occurrences=sum(counts.values()),
        marked_alphabet=[_piece_of(symbol) for symbol in alphabet_of(spelled)],
        ended_alphabet_size=len(ended),
        four_word_marked_alphabet=[
            _piece_of(symbol) for symbol in alphabet_of(_spelled(four_counts))
        ],
        four_word_ended_alphabet=list(
            alphabet_of(
                [
                    SpelledWord(_ended_symbols(word), count)
                    for word, count in four_counts.items()
                ]
            )
        ),
        character_pieces=_length_after(counts, Merges([])),
        rows=[_size_row(asked, marked=True) for asked in SWEEP_SIZES],
        count_rows=[_size_row(asked, marked=False) for asked in SWEEP_SIZES],
        cuts=cuts,
        distinct_characters=len(characters),
        never_opens=sorted(characters - opens),
        never_ends=sorted(characters - ends),
        comparisons=[
            _word_pair(word, marked_model, ended_model) for word in COMPARED_WORDS
        ],
        four_word_comparisons=[
            _word_pair(word, four_marked, four_ended) for word in FOUR_WORD_COMPARED
        ],
    )


@router.get("/spelling", response_model=SpellingResponse)
def spelling() -> SpellingResponse:
    """The two markings, the size sweep, and the running sentence cut."""
    return _spelling()


# --------------------------------------------------------------------------
# /limits: the closing part's three measurements.
# --------------------------------------------------------------------------


class BoundRow(BaseModel):
    """A pair whose halves never occur apart, against a rarer one of the same kind.

    Both pairs are perfectly bound, so both scores are one over the pair's own
    count, and the rarer pair therefore scores higher.
    """

    repeats: int
    score: float
    rival_repeats: int
    rival_score: float
    taken: str


class GreedyCheck(BaseModel):
    """Every short word over the small corpus's letters, cut or refused.

    ``n_refused_with_a_cut_available`` is the number the greedy left-to-right cut
    refused although some other division into vocabulary pieces existed, which is
    the failure the closing part asks about.
    """

    longest_word: int
    n_letters: int
    n_words: int
    n_cut: int
    n_refused: int
    n_refused_at_the_first_letter: int
    n_refused_later: int
    n_refused_with_a_cut_available: int
    opening_pieces: list[str]
    continuation_letters: list[str]
    missing_continuations: list[str]


class MinimumReading(BaseModel):
    """The fewest-times rule read as a filter and read as a stopping test."""

    minimum: int
    as_filter: list[str]
    as_stopping_test: list[str]
    stopped_on: str
    stopped_on_count: int
    left_unmerged: list[str]


class CarriedCorpus(BaseModel):
    """What the finished vocabulary costs on its own corpus and on other text."""

    learned: int
    n_merges: int
    corpus_pieces: int
    corpus_words: int
    corpus_per_word: float
    sentence_pieces: int
    sentence_words: int
    sentence_per_word: float
    neighbour_text: str
    neighbour_pieces: int
    neighbour_words: int
    neighbour_per_word: float
    whole_word_pieces: list[str]


class LimitsResponse(BaseModel):
    """The facts the closing part rests on, every one of them measured."""

    bound_rows: list[BoundRow]
    greedy: GreedyCheck
    minimum: MinimumReading
    carried: CarriedCorpus


def _bound_rows() -> list[BoundRow]:
    """Two pairs, each of which never occurs except beside its partner."""
    rows: list[BoundRow] = []
    for repeats in BOUND_REPEATS:
        learned = learn_merges(
            _spelled({"qx": repeats, "ab": 2}),
            n_merges=1,
            minimum_pair_frequency=1,
            scoring=PairScoring.LIKELIHOOD,
        )
        rows.append(
            BoundRow(
                repeats=repeats,
                score=1 / repeats,
                rival_repeats=2,
                rival_score=1 / 2,
                taken=f"{_piece_of(learned[0].left)} + {_piece_of(learned[0].right)}",
            )
        )
    return rows


def _greedy_check() -> GreedyCheck:
    model = WordPiece(vocabulary_size=22).fit(FOUR_WORD_CORPUS)
    vocabulary = set(model.vocabulary)
    letters = sorted({character for word in FOUR_WORD_COUNTS for character in word})
    openings = sorted(
        piece
        for piece in vocabulary
        if piece != UNKNOWN and not piece.startswith(CONTINUATION)
    )
    continuations = sorted(
        piece[len(CONTINUATION) :]
        for piece in vocabulary
        if piece.startswith(CONTINUATION) and len(piece) == len(CONTINUATION) + 1
    )

    def any_cut(word: str) -> bool:
        """Whether any division of the word into vocabulary pieces exists."""
        reachable = {0}
        for begin in range(len(word)):
            if begin not in reachable:
                continue
            mark = "" if begin == 0 else CONTINUATION
            for end in range(begin + 1, len(word) + 1):
                if mark + word[begin:end] in vocabulary:
                    reachable.add(end)
        return len(word) in reachable

    cut = refused = first_letter = later = available = words = 0
    for length in range(1, GREEDY_LONGEST_WORD + 1):
        for letters_of in product(letters, repeat=length):
            word = "".join(letters_of)
            words += 1
            if model.encode(word).texts != (UNKNOWN,):
                cut += 1
                continue
            refused += 1
            if not any(word.startswith(piece) for piece in openings):
                first_letter += 1
            else:
                later += 1
            if any_cut(word):
                available += 1

    return GreedyCheck(
        longest_word=GREEDY_LONGEST_WORD,
        n_letters=len(letters),
        n_words=words,
        n_cut=cut,
        n_refused=refused,
        n_refused_at_the_first_letter=first_letter,
        n_refused_later=later,
        n_refused_with_a_cut_available=available,
        opening_pieces=openings,
        continuation_letters=continuations,
        missing_continuations=[
            letter for letter in letters if letter not in continuations
        ],
    )


def _minimum_reading() -> MinimumReading:
    """The filter reading beside the stopping-test reading, at the same minimum."""
    filtered = WordPiece(
        vocabulary_size=MAX_VOCABULARY_SIZE, minimum_pair_frequency=MINIMUM_UNDER_TEST
    ).fit(FOUR_WORD_CORPUS)
    n_alphabet = len(alphabet_of(_spelled(FOUR_WORD_COUNTS))) + 1
    as_filter = list(filtered.vocabulary)[n_alphabet:]

    current = _spelled(FOUR_WORD_COUNTS)
    stopped: list[Merge] = []
    quit_on: Merge | None = None
    while True:
        merge = best_pair(current, PairScoring.LIKELIHOOD, 1)
        if merge is None:
            break
        if merge.score < MINIMUM_UNDER_TEST:
            quit_on = merge
            break
        current = [word.with_pair_merged(merge) for word in current]
        stopped.append(merge)
    as_stopping = [_piece_of(merge.merged) for merge in stopped]

    return MinimumReading(
        minimum=MINIMUM_UNDER_TEST,
        as_filter=as_filter,
        as_stopping_test=as_stopping,
        stopped_on=(
            f"{_piece_of(quit_on.left)} + {_piece_of(quit_on.right)}"
            if quit_on is not None
            else ""
        ),
        stopped_on_count=int(quit_on.score) if quit_on is not None else 0,
        left_unmerged=[piece for piece in as_filter if piece not in as_stopping],
    )


def _carried_corpus() -> CarriedCorpus:
    model = WordPiece(vocabulary_size=MAX_VOCABULARY_SIZE).fit(SENTENCE_CORPUS)
    counts = _word_counts(SENTENCE_CORPUS)
    corpus_pieces = sum(model.encode(text).n_tokens for text in SENTENCE_CORPUS)
    sentence_words = len(WhitespacePreTokenizer().split(RUNNING_SENTENCE).texts)
    sentence_pieces = model.encode(RUNNING_SENTENCE).n_tokens
    neighbour_words = len(WhitespacePreTokenizer().split(NEIGHBOUR_TEXT).texts)
    neighbour_pieces = model.encode(NEIGHBOUR_TEXT).n_tokens
    corpus_words = sum(counts.values())
    return CarriedCorpus(
        learned=model.vocabulary.n_tokens,
        n_merges=model.n_merges,
        corpus_pieces=corpus_pieces,
        corpus_words=corpus_words,
        corpus_per_word=corpus_pieces / corpus_words,
        sentence_pieces=sentence_pieces,
        sentence_words=sentence_words,
        sentence_per_word=sentence_pieces / sentence_words,
        neighbour_text=NEIGHBOUR_TEXT,
        neighbour_pieces=neighbour_pieces,
        neighbour_words=neighbour_words,
        neighbour_per_word=neighbour_pieces / neighbour_words,
        whole_word_pieces=[piece for piece in model.vocabulary if piece in counts],
    )


@cache
def _limits() -> LimitsResponse:
    return LimitsResponse(
        bound_rows=_bound_rows(),
        greedy=_greedy_check(),
        minimum=_minimum_reading(),
        carried=_carried_corpus(),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The bound pair, the exhaustive cut check, the minimum, and the corpus."""
    return _limits()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to learn from, a size to stop at, and a text to read back."""

    corpus: CorpusChoice = CorpusChoice.SENTENCES
    vocabulary_size: int = Field(ge=2, le=MAX_VOCABULARY_SIZE)
    minimum_pair_frequency: int = Field(default=2, ge=1, le=20)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything the playground shows for one fit.

    ``count_pieces`` is the same text under a count-scored fit of the same
    requested size, so the contrast the page is about is always on screen; it is
    absent where that fit cannot be made at the size asked for.
    """

    corpus: list[str]
    corpus_label: str
    alphabet: list[str]
    smallest_size: int
    asked: int
    learned: int
    n_merges: int
    merges: list[MergeRow]
    tokens: list[str]
    corpus_pieces: int
    corpus_character_pieces: int
    text: str
    pieces: list[str]
    n_pieces: int
    n_unknown: int
    decoded: str
    round_trip_exact: bool
    count_pieces: list[str] | None
    count_n_pieces: int | None
    count_corpus_pieces: int | None


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn a vocabulary of the requested size and read one text with it."""
    corpus = CORPORA[request.corpus]
    counts = _word_counts(corpus)
    alphabet = alphabet_of(_spelled(counts))
    try:
        model = WordPiece(
            vocabulary_size=request.vocabulary_size,
            minimum_pair_frequency=request.minimum_pair_frequency,
        ).fit(corpus)
        encoding = model.encode(request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    rival_pieces: list[str] | None = None
    rival_n_pieces: int | None = None
    rival_corpus: int | None = None
    try:
        rival = BytePairEncoding(
            byte_fallback=False,
            vocabulary_size=request.vocabulary_size,
            minimum_pair_frequency=request.minimum_pair_frequency,
        ).fit(corpus)
    except MLLibError:
        rival = None
    if rival is not None:
        rival_encoding = rival.encode(request.text)
        rival_pieces = list(rival_encoding.texts)
        rival_n_pieces = rival_encoding.n_tokens
        rival_corpus = sum(rival.encode(text).n_tokens for text in corpus)

    decoded = model.decode(encoding.ids)
    return FitResponse(
        corpus=corpus,
        corpus_label=CORPUS_LABELS[request.corpus],
        alphabet=[_piece_of(symbol) for symbol in alphabet],
        smallest_size=len(alphabet) + 1,
        asked=request.vocabulary_size,
        learned=model.vocabulary.n_tokens,
        n_merges=model.n_merges,
        merges=_merge_rows(
            counts,
            model.n_merges,
            request.minimum_pair_frequency,
            PairScoring.LIKELIHOOD,
        ),
        tokens=list(model.vocabulary),
        corpus_pieces=sum(model.encode(text).n_tokens for text in corpus),
        corpus_character_pieces=_length_after(counts, Merges([])),
        text=request.text,
        pieces=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        n_unknown=sum(1 for piece in encoding.texts if piece == UNKNOWN),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
        count_pieces=rival_pieces,
        count_n_pieces=rival_n_pieces,
        count_corpus_pieces=rival_corpus,
    )
