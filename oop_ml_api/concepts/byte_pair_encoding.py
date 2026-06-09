"""The byte pair encoding page's endpoints.

Six of them, one per family of widgets rather than one per number. ``/learning``
walks the four-word corpus one merge at a time, with the tied candidates at each
step, so the page can show the vocabulary assembling and can show that the ties
are real. ``/sizes`` sweeps the vocabulary size over the eighteen-sentence
corpus and reports what the corpus and the running sentence cost at each,
together with the whole-word and single-character baselines the page opens on.
``/fallback`` reads three texts twice, once under the method exactly as
published and once with a spelling for every character, which is the pair the
page's repair section rests on. ``/variants`` fits the three variants beside the
plain method on the same corpus. ``/limits`` carries the two facts the closing
part rests on, the tie whose branches the counts cannot separate and an
exhaustive search that beats greedy merging. ``/fit`` is the playground.

Two fits here deliberately switch the byte fallback off. ``/learning`` does,
because it walks the original paper's ten-merge example and a table of
twenty-two rows is the thing a reader can check by hand; the 257 structural rows
would bury it. ``/limits`` does for its lost-boundary reading, because that
section is about where the method as published stops being defined. Everywhere
else the fits are the ordinary ones, with every character spellable.

Rows are reported in two numbers throughout. ``learned`` counts only what the
corpus taught, which is the unknown token, the alphabet and the merges, and is
what the requested size bounds; ``total_rows`` adds the 257 structural rows that
sit underneath every fit and belong to no corpus.

The exhaustive search in ``/limits`` is this module's own, because the library
learns merges greedily by definition and has no reason to enumerate merge
sequences; everything else is a fit, and every fit is the library's.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from itertools import islice

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.morphology.constrained_merges import (
    MorphemeConstrainedBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_level import (
    ByteLevelBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BYTE_VALUE_OF,
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    apply_merges,
    learn_merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.super_byte_pair_encoding import (
    SuperBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.pattern import (
    PatternPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/byte-pair-encoding", tags=["byte-pair-encoding"])


MAX_VOCABULARY_SIZE = 600
"""The largest vocabulary a request may ask for.

Both page corpora run out of pairs worth merging long before this, so the bound
never truncates anything the page shows; it exists so a crafted request cannot
turn a teaching endpoint into a long fit.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to encode."""

END_OF_WORD = "</w>"
"""The marker every fit on this page uses, written out once."""

N_BYTE_VALUES = 256
"""How many byte values there are, which is the byte variant's whole alphabet."""

N_FLOOR_ROWS = N_BYTE_VALUES + 1
"""The structural rows every ordinary fit here carries.

One row per byte value, plus one standalone end-of-word marker, which is what a
word whose last character had to be spelled in bytes needs in order to say that
a word ended there. They sit outside the requested size, so the page reports
them separately rather than folding them into what the corpus taught.
"""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every merge is checkable.

This is the corpus the original subword paper worked its example on, and it is
the corpus the from-scratch trainer's own spec is pinned against, so every
number the page quotes from it is a number a test already holds.
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
around it are plausible neighbours of the ones it uses. It still fails to spell
one *symbol* of the sentence, since no word in it ends in ``z``, which is the
finding the page's tenth section rests on.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

FOREIGN_TEXT = "The café rejected the re-analysis."
"""The same shape of sentence with one character no corpus here contains."""

FOREIGN_WORD = "café"

GREEK_TEXT = "Η ανάλυση ήταν φθηνότερη."
"""The same kind of sentence in a script the eighteen sentences never use.

Every letter of it is absent from the corpus, so it is the case where a
character alphabet has nothing at all to offer and the difference between
losing the text and keeping it is at its widest.
"""

ACCOUNTING_SIZES = (52, 54, 100, 137, 300)
"""The sizes the page prints the row accounting at, floor to ceiling."""

MORPH_WORDS = ("undo", "untie", "redo", "retie")
MORPH_CORPUS: list[str] = [" ".join(word for word in MORPH_WORDS for _ in range(10))]
"""Two prefixes across two roots, ten times each: the corpus where a frequent
pair sits exactly on the joint between a prefix and a root."""

MORPH_PATTERN = "un|re|do|tie"
"""The cuts the constrained fit must respect, stated rather than learned."""

GREEDY_TRAP_COUNTS: dict[str, int] = {"report": 2, "taste": 8, "state": 10}
"""Three words on which greedy merging is beaten by eight pieces at two merges.

Small enough that every sequence of up to four merges can be enumerated, which
is what makes the comparison a proof rather than a search that gave up.
"""

GREEDY_TRAP_BUDGETS = (1, 2, 3, 4)

SWEEP_SIZES = (52, 56, 60, 66, 72, 80, 90, 100, 110, 120, 130, 137, 150, 200, 300)
"""The vocabulary sizes the size curve is drawn at."""

CUT_SIZES = (52, 66, 80, 137)
"""The four sizes the page prints the running sentence's pieces at."""


class CorpusChoice(StrEnum):
    """Which of the page's two corpora a request wants fitted."""

    FOUR_WORDS = "four_words"
    SENTENCES = "sentences"


CORPORA: dict[CorpusChoice, list[str]] = {
    CorpusChoice.FOUR_WORDS: FOUR_WORD_CORPUS,
    CorpusChoice.SENTENCES: SENTENCE_CORPUS,
}


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("vocabulary_size=", "a vocabulary of "),
    ("transition_size=", "a transition at "),
    ("minimum_pair_frequency", "the fewest times a pair may be seen"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("the unknown token", "one token for anything unspellable"),
    ("BytePairEncoding", "this scheme"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    The messages are written to be read and most of them already are, so this
    substitutes the phrases that name a parameter and leaves the rest alone.
    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# Small helpers shared by the endpoints.
# --------------------------------------------------------------------------


def _symbols_of(word: str) -> tuple[str, ...]:
    """A word as characters, the last one carrying the end-of-word marker."""
    return (*word[:-1], word[-1] + END_OF_WORD)


def _word_counts(corpus: Sequence[str]) -> dict[str, int]:
    counts = Corpus.of(list(corpus)).word_counts(WhitespacePreTokenizer())
    return {word_count.word: word_count.count for word_count in counts}


def _spelled(counts: dict[str, int]) -> list[SpelledWord]:
    return [SpelledWord(_symbols_of(word), count) for word, count in counts.items()]


def _pair_counts(spelled: Sequence[SpelledWord]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for word in spelled:
        for left, right in zip(word.symbols, word.symbols[1:], strict=False):
            counter[(left, right)] += word.count
    return counter


def _corpus_pieces(counts: dict[str, int], merges: Merges) -> int:
    """How many pieces the whole corpus becomes under these merges."""
    return sum(
        len(apply_merges(_symbols_of(word), merges)) * count
        for word, count in counts.items()
    )


def _alphabet(corpus: Sequence[str]) -> list[str]:
    return sorted(
        {symbol for word in _spelled(_word_counts(corpus)) for symbol in word}
    )


def _merge_labels(merges: Merges) -> list[str]:
    return [f"{merge.left} + {merge.right}" for merge in merges]


def _taught_rows(model: BytePairEncoding) -> int:
    """How many rows of ``model`` the corpus taught.

    The unknown token, the alphabet and the merges. This is the number the
    requested size bounds, and it is the whole table only when the structural
    rows are switched off.
    """
    floor = N_FLOOR_ROWS if model.byte_fallback else 0
    return model.vocabulary.n_tokens - floor


def _byte_rows_in(pieces: Sequence[str]) -> int:
    """How many of ``pieces`` are a byte row rather than something learned."""
    return sum(1 for piece in pieces if piece in BYTE_VALUE_OF)


# --------------------------------------------------------------------------
# /learning: the four-word corpus, one merge at a time.
# --------------------------------------------------------------------------


class SpelledEntry(BaseModel):
    """One distinct word, how often it occurs, and its spelling right now."""

    word: str
    count: int
    symbols: list[str]


class LadderStep(BaseModel):
    """The corpus before a merge, the merge chosen, and what it cost.

    ``step`` is zero before any merge has happened, in which case ``left``,
    ``right`` and ``merged`` are absent. ``tied`` names every pair that reached
    the winning count, so a reader can see when the count alone did not decide.
    """

    step: int
    left: str | None
    right: str | None
    merged: str | None
    count: int | None
    tied: list[str]
    spellings: list[SpelledEntry]
    corpus_pieces: int
    unseen_pieces: list[str]


class LearningResponse(BaseModel):
    """The four-word corpus walked from characters to a finished vocabulary.

    Fitted as the original paper's method, with no structural rows underneath
    it, so that ``tokens`` is twenty-two rows a reader can check one at a time.
    """

    corpus: list[str]
    word_counts: list[SpelledEntry]
    alphabet: list[str]
    total_occurrences: int
    character_pieces: int
    unseen_word: str
    steps: list[LadderStep]
    tokens: list[str]


@cache
def _learning() -> LearningResponse:
    counts = FOUR_WORD_COUNTS
    alphabet = _alphabet(FOUR_WORD_CORPUS)
    model = BytePairEncoding(
        vocabulary_size=len(alphabet) + 1 + 10, byte_fallback=False
    ).fit(FOUR_WORD_CORPUS)

    current = _spelled(counts)
    chosen: list[Merge] = []
    steps: list[LadderStep] = []
    for step in range(model.n_merges + 1):
        merge = model.merges[step] if step < model.n_merges else None
        pair_counts = _pair_counts(current)
        best = max(pair_counts.values(), default=0)
        steps.append(
            LadderStep(
                step=step,
                left=merge.left if merge else None,
                right=merge.right if merge else None,
                merged=merge.merged if merge else None,
                count=int(merge.score) if merge else None,
                tied=sorted(
                    f"{left} + {right}"
                    for (left, right), count in pair_counts.items()
                    if merge is not None and count == best
                ),
                spellings=[
                    SpelledEntry(
                        word=word, count=counts[word], symbols=list(spelled.symbols)
                    )
                    for word, spelled in zip(counts, current, strict=True)
                ],
                corpus_pieces=_corpus_pieces(counts, Merges(chosen)),
                unseen_pieces=list(
                    apply_merges(_symbols_of(UNSEEN_WORD), Merges(chosen))
                ),
            )
        )
        if merge is None:
            break
        chosen.append(merge)
        current = [word.with_pair_merged(merge) for word in current]

    return LearningResponse(
        corpus=FOUR_WORD_CORPUS,
        word_counts=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in counts.items()
        ],
        alphabet=alphabet,
        total_occurrences=sum(counts.values()),
        character_pieces=_corpus_pieces(counts, Merges([])),
        unseen_word=UNSEEN_WORD,
        steps=steps,
        tokens=list(model.vocabulary),
    )


@router.get("/learning", response_model=LearningResponse)
def learning() -> LearningResponse:
    """The four-word corpus from single characters to ten merges.

    The original paper's method exactly, so the twenty-two rows are the whole
    table rather than the taught part of a larger one.
    """
    return _learning()


# --------------------------------------------------------------------------
# /sizes: what a vocabulary size buys on the eighteen sentences.
# --------------------------------------------------------------------------


class SizeRow(BaseModel):
    """One vocabulary size, and what the corpus and the sentence cost at it.

    ``asked`` is what the fit was asked for and ``learned`` what it managed;
    the two part once the corpus has no pair left that occurs twice.
    ``total_rows`` adds the structural rows, which no size ever changes.
    """

    asked: int
    learned: int
    total_rows: int
    n_merges: int
    corpus_pieces: int
    sentence_pieces: int
    round_trip_exact: bool


class SentenceCut(BaseModel):
    """The running sentence as one fit cut it, and what came back."""

    vocabulary_size: int
    learned: int
    total_rows: int
    n_merges: int
    pieces: list[str]
    n_pieces: int
    decoded: str
    round_trip_exact: bool


NEIGHBOUR_WORDS = (
    "analysis",
    "analysis.",
    "expected",
    "expect",
    "cost",
    "low-cost",
)
"""Six words the page reads one at a time, chosen so that each pair differs by
one thing: a full stop, a suffix, a hyphen. The point they make is that a row
carries an end-of-word marker, so a word the corpus held is one piece and the
same letters ending differently are several."""


class WordReading(BaseModel):
    """One word on its own, and the pieces the finished vocabulary cuts it into."""

    word: str
    pieces: list[str]
    n_pieces: int


class BaselineCounts(BaseModel):
    """The two things a vocabulary of pieces sits between.

    ``sentence_words_unseen`` is how many of the running sentence's words the
    corpus never contained, which is what a vocabulary of whole words could not
    read; ``sentence_character_pieces`` is what the same sentence costs when
    every character is its own piece.

    ``sentence_symbols_unspellable`` is the part of that second claim which is
    usually stated too strongly. A character alphabet holds a word-final letter
    apart from the same letter inside a word, so a sentence can be written
    entirely in characters the corpus used and still contain a symbol it never
    learned.
    """

    n_distinct_words: int
    n_word_occurrences: int
    alphabet_size: int
    corpus_character_pieces: int
    sentence_words: list[str]
    sentence_words_unseen: list[str]
    sentence_characters: int
    sentence_character_pieces: int
    sentence_characters_unseen: list[str]
    sentence_symbols_unspellable: list[str]


class SizesResponse(BaseModel):
    """The sweep, the baselines, and the sentence cut at four sizes."""

    corpus: list[str]
    sentence: str
    baselines: BaselineCounts
    smallest_size: int
    largest_useful_size: int
    floor_rows: int
    rows: list[SizeRow]
    cuts: list[SentenceCut]
    neighbour_readings: list[WordReading]


@cache
def _sizes() -> SizesResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    alphabet = _alphabet(SENTENCE_CORPUS)
    exhausted = BytePairEncoding(vocabulary_size=MAX_VOCABULARY_SIZE).fit(
        SENTENCE_CORPUS
    )
    words = WhitespacePreTokenizer().split(RUNNING_SENTENCE).texts

    rows: list[SizeRow] = []
    for asked in SWEEP_SIZES:
        model = BytePairEncoding(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        encoding = model.encode(RUNNING_SENTENCE)
        rows.append(
            SizeRow(
                asked=asked,
                learned=_taught_rows(model),
                total_rows=model.vocabulary.n_tokens,
                n_merges=model.n_merges,
                corpus_pieces=_corpus_pieces(counts, model.merges),
                sentence_pieces=encoding.n_tokens,
                round_trip_exact=model.decode(encoding.ids) == RUNNING_SENTENCE,
            )
        )

    cuts: list[SentenceCut] = []
    for asked in CUT_SIZES:
        model = BytePairEncoding(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        encoding = model.encode(RUNNING_SENTENCE)
        decoded = model.decode(encoding.ids)
        cuts.append(
            SentenceCut(
                vocabulary_size=asked,
                learned=_taught_rows(model),
                total_rows=model.vocabulary.n_tokens,
                n_merges=model.n_merges,
                pieces=list(encoding.texts),
                n_pieces=encoding.n_tokens,
                decoded=decoded,
                round_trip_exact=decoded == RUNNING_SENTENCE,
            )
        )

    return SizesResponse(
        corpus=SENTENCE_CORPUS,
        sentence=RUNNING_SENTENCE,
        baselines=BaselineCounts(
            n_distinct_words=len(counts),
            n_word_occurrences=sum(counts.values()),
            alphabet_size=len(alphabet),
            corpus_character_pieces=_corpus_pieces(counts, Merges([])),
            sentence_words=list(words),
            sentence_words_unseen=[word for word in words if word not in counts],
            sentence_characters=len(RUNNING_SENTENCE),
            sentence_character_pieces=sum(len(_symbols_of(word)) for word in words),
            sentence_characters_unseen=sorted(
                set(RUNNING_SENTENCE)
                - {character for text in SENTENCE_CORPUS for character in text}
            ),
            sentence_symbols_unspellable=[
                symbol
                for word in words
                for symbol in _symbols_of(word)
                if symbol not in set(alphabet)
            ],
        ),
        smallest_size=len(alphabet) + 1,
        largest_useful_size=_taught_rows(exhausted),
        floor_rows=N_FLOOR_ROWS,
        rows=rows,
        cuts=cuts,
        neighbour_readings=[
            WordReading(
                word=word,
                pieces=list(exhausted.encode(word).texts),
                n_pieces=exhausted.encode(word).n_tokens,
            )
            for word in NEIGHBOUR_WORDS
        ],
    )


@router.get("/sizes", response_model=SizesResponse)
def sizes() -> SizesResponse:
    """The vocabulary-size sweep on the eighteen sentences, with the baselines."""
    return _sizes()


# --------------------------------------------------------------------------
# /fallback: the same three texts read twice, with and without a spelling for
# every character.
# --------------------------------------------------------------------------


class FallbackReading(BaseModel):
    """One scheme reading one text, with what came back when it was glued.

    ``n_byte_rows`` and ``n_bare_markers`` say which part of the length is the
    spelling machinery rather than anything the corpus taught, which is the
    number the page needs in order to say what the repair costs.
    """

    key: str
    label: str
    pieces: list[str]
    n_pieces: int
    n_byte_rows: int
    n_bare_markers: int
    n_stand_ins: int
    decoded: str
    round_trip_exact: bool


class FallbackText(BaseModel):
    """One text, its characters the corpus never held, and both readings."""

    key: str
    label: str
    text: str
    n_characters: int
    characters_unseen: list[str]
    readings: list[FallbackReading]


class AccountingRow(BaseModel):
    """One requested size against the table that actually results from it."""

    asked: int
    n_merges: int
    learned: int
    floor_rows: int
    total_rows: int


class FallbackResponse(BaseModel):
    """The repair, measured on three texts and on the whole corpus.

    ``unspellable_symbol`` is the one symbol of the running sentence the
    eighteen sentences never taught, and the two spellings beside it are what
    each scheme puts in its place.
    """

    alphabet_size: int
    floor_rows: int
    n_byte_rows: int
    unspellable_symbol: str
    unspellable_character: str
    stand_in: str
    repaired_spelling: list[str]
    byte_spelled_character: str
    byte_spelling: list[str]
    corpus_pieces: int
    corpus_pieces_published: int
    texts: list[FallbackText]
    accounting: list[AccountingRow]


PUBLISHED_LABEL = "As published"
FALLBACK_LABEL = "With every character spellable"

FALLBACK_TEXTS: tuple[tuple[str, str, str], ...] = (
    ("sentence", "The running sentence", RUNNING_SENTENCE),
    ("accented", "A sentence with an accent in it", FOREIGN_TEXT),
    ("greek", "A sentence in another script", GREEK_TEXT),
)


def _fallback_reading(
    model: BytePairEncoding, key: str, label: str, text: str
) -> FallbackReading:
    encoding = model.encode(text)
    decoded = model.decode(encoding.ids)
    pieces = list(encoding.texts)
    return FallbackReading(
        key=key,
        label=label,
        pieces=pieces,
        n_pieces=encoding.n_tokens,
        n_byte_rows=_byte_rows_in(pieces),
        n_bare_markers=sum(1 for piece in pieces if piece == END_OF_WORD),
        n_stand_ins=sum(
            1 for piece in pieces if piece == model.vocabulary.unknown_token
        ),
        decoded=decoded,
        round_trip_exact=decoded == text,
    )


@cache
def _fallback() -> FallbackResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    alphabet = _alphabet(SENTENCE_CORPUS)
    published = BytePairEncoding(
        vocabulary_size=MAX_VOCABULARY_SIZE, byte_fallback=False
    ).fit(SENTENCE_CORPUS)
    repaired = BytePairEncoding(vocabulary_size=MAX_VOCABULARY_SIZE).fit(
        SENTENCE_CORPUS
    )

    seen = {character for text in SENTENCE_CORPUS for character in text}
    words = WhitespacePreTokenizer().split(RUNNING_SENTENCE).texts
    unspellable = next(
        symbol
        for word in words
        for symbol in _symbols_of(word)
        if symbol not in set(alphabet)
    )

    accounting: list[AccountingRow] = []
    for asked in ACCOUNTING_SIZES:
        model = BytePairEncoding(vocabulary_size=asked).fit(SENTENCE_CORPUS)
        accounting.append(
            AccountingRow(
                asked=asked,
                n_merges=model.n_merges,
                learned=_taught_rows(model),
                floor_rows=N_FLOOR_ROWS,
                total_rows=model.vocabulary.n_tokens,
            )
        )

    return FallbackResponse(
        alphabet_size=len(alphabet),
        floor_rows=N_FLOOR_ROWS,
        n_byte_rows=N_BYTE_VALUES,
        unspellable_symbol=unspellable,
        unspellable_character=unspellable[: -len(END_OF_WORD)],
        stand_in=published.vocabulary.unknown_token,
        repaired_spelling=[unspellable[: -len(END_OF_WORD)], END_OF_WORD],
        byte_spelled_character=FOREIGN_WORD[-1],
        byte_spelling=[
            piece
            for piece in repaired.encode(FOREIGN_WORD).texts
            if piece in BYTE_VALUE_OF
        ],
        corpus_pieces=_corpus_pieces(counts, repaired.merges),
        corpus_pieces_published=_corpus_pieces(counts, published.merges),
        texts=[
            FallbackText(
                key=key,
                label=label,
                text=text,
                n_characters=len(text),
                characters_unseen=sorted(set(text) - seen),
                readings=[
                    _fallback_reading(published, "published", PUBLISHED_LABEL, text),
                    _fallback_reading(repaired, "repaired", FALLBACK_LABEL, text),
                ],
            )
            for key, label, text in FALLBACK_TEXTS
        ],
        accounting=accounting,
    )


@router.get("/fallback", response_model=FallbackResponse)
def fallback() -> FallbackResponse:
    """Three texts read by the published method and by the repaired one."""
    return _fallback()


# --------------------------------------------------------------------------
# /variants: the three things one can change.
# --------------------------------------------------------------------------


class VariantReading(BaseModel):
    """One fit's reading of one text."""

    pieces: list[str]
    n_pieces: int
    decoded: str
    round_trip_exact: bool


class VariantSummary(BaseModel):
    """One alphabet fitted on the eighteen sentences and read three times.

    ``alphabet_size`` is what the fit merges over, and ``floor_rows`` the rows
    it carries underneath that for the characters its alphabet lacks, which is
    zero once the alphabet is the byte values themselves.
    """

    key: str
    label: str
    asked: int
    learned: int
    n_merges: int
    alphabet_size: int
    floor_rows: int
    corpus_pieces: int
    sentence: VariantReading
    foreign: VariantReading
    greek: VariantReading


class SuperwordSummary(BaseModel):
    """What lifting the word boundary learns, on one corpus."""

    corpus_label: str
    n_word_merges: int
    n_superword_merges: int
    superword_merges: list[str]
    superword_tokens: list[str]
    example_text: str
    example_pieces: list[str]


class MorphComparison(BaseModel):
    """Plain and constrained merging on the same prefix-and-root corpus."""

    corpus: list[str]
    words: list[str]
    morphs: list[list[str]]
    plain_merges: list[str]
    plain_tokens: list[str]
    plain_cut: list[str]
    plain_corpus_pieces: int
    constrained_merges: list[str]
    constrained_tokens: list[str]
    constrained_cut: list[str]
    constrained_corpus_pieces: int


class VariantsResponse(BaseModel):
    """The plain method beside the three variants, on the page's own texts.

    ``corpus_distinct_characters`` counts the corpus's characters rather than
    its symbols, since a character alphabet holds a word-final letter apart
    from the same letter inside a word and a byte alphabet does not, and the
    comparison the page draws is between characters and byte values.
    """

    sentence: str
    foreign: str
    greek: str
    fits: list[VariantSummary]
    corpus_distinct_characters: int
    foreign_word: str
    foreign_character: str
    foreign_character_bytes: list[int]
    published_foreign_word: list[str]
    plain_foreign_word: list[str]
    byte_foreign_word: list[str]
    superwords: list[SuperwordSummary]
    morphology: MorphComparison


def _reading(model: Tokenizer, text: str) -> VariantReading:
    encoding = model.encode(text)
    decoded = model.decode(encoding.ids)
    return VariantReading(
        pieces=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        decoded=decoded,
        round_trip_exact=decoded == text,
    )


def _superword_summary(
    label: str,
    corpus: list[str],
    transition_size: int,
    vocabulary_size: int,
    example_text: str,
) -> SuperwordSummary:
    model = SuperBytePairEncoding(
        vocabulary_size=vocabulary_size, transition_size=transition_size
    ).fit(corpus)
    return SuperwordSummary(
        corpus_label=label,
        n_word_merges=model.n_word_merges,
        n_superword_merges=model.n_superword_merges,
        superword_merges=_merge_labels(model.superword_merges),
        superword_tokens=list(model.superword_tokens),
        example_text=example_text,
        example_pieces=list(model.encode(example_text).texts),
    )


def _morphology() -> MorphComparison:
    splitter = PatternPreTokenizer(pattern=MORPH_PATTERN)
    plain = BytePairEncoding(vocabulary_size=30).fit(MORPH_CORPUS)
    constrained = MorphemeConstrainedBytePairEncoding(
        vocabulary_size=30, morph_splitter=splitter
    ).fit(MORPH_CORPUS)
    return MorphComparison(
        corpus=MORPH_CORPUS,
        words=list(MORPH_WORDS),
        morphs=[list(constrained.morphs_of(word)) for word in MORPH_WORDS],
        plain_merges=_merge_labels(plain.merges),
        plain_tokens=list(plain.vocabulary),
        plain_cut=list(plain.encode("undo").texts),
        plain_corpus_pieces=sum(plain.encode(text).n_tokens for text in MORPH_CORPUS),
        constrained_merges=_merge_labels(constrained.merges),
        constrained_tokens=list(constrained.vocabulary),
        constrained_cut=list(constrained.encode("undo").texts),
        constrained_corpus_pieces=sum(
            constrained.encode(text).n_tokens for text in MORPH_CORPUS
        ),
    )


@cache
def _variants() -> VariantsResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    alphabet = _alphabet(SENTENCE_CORPUS)

    plain = BytePairEncoding(vocabulary_size=MAX_VOCABULARY_SIZE).fit(SENTENCE_CORPUS)
    published = BytePairEncoding(
        vocabulary_size=MAX_VOCABULARY_SIZE, byte_fallback=False
    ).fit(SENTENCE_CORPUS)
    byte = ByteLevelBytePairEncoding(
        vocabulary_size=N_BYTE_VALUES + plain.n_merges
    ).fit(SENTENCE_CORPUS)

    fits = [
        VariantSummary(
            key="plain",
            label="Characters",
            asked=MAX_VOCABULARY_SIZE,
            learned=_taught_rows(plain),
            n_merges=plain.n_merges,
            alphabet_size=len(alphabet),
            floor_rows=N_FLOOR_ROWS,
            corpus_pieces=_corpus_pieces(counts, plain.merges),
            sentence=_reading(plain, RUNNING_SENTENCE),
            foreign=_reading(plain, FOREIGN_TEXT),
            greek=_reading(plain, GREEK_TEXT),
        ),
        VariantSummary(
            key="bytes",
            label="Bytes",
            asked=N_BYTE_VALUES + plain.n_merges,
            learned=byte.vocabulary.n_tokens,
            n_merges=byte.n_merges,
            alphabet_size=N_BYTE_VALUES,
            floor_rows=0,
            corpus_pieces=sum(byte.encode(text).n_tokens for text in SENTENCE_CORPUS),
            sentence=_reading(byte, RUNNING_SENTENCE),
            foreign=_reading(byte, FOREIGN_TEXT),
            greek=_reading(byte, GREEK_TEXT),
        ),
    ]

    return VariantsResponse(
        sentence=RUNNING_SENTENCE,
        foreign=FOREIGN_TEXT,
        greek=GREEK_TEXT,
        fits=fits,
        corpus_distinct_characters=len(
            {character for text in SENTENCE_CORPUS for character in text}
        ),
        foreign_word=FOREIGN_WORD,
        foreign_character=FOREIGN_WORD[-1],
        foreign_character_bytes=list(FOREIGN_WORD[-1].encode()),
        published_foreign_word=list(published.encode(FOREIGN_WORD).texts),
        plain_foreign_word=list(plain.encode(FOREIGN_WORD).texts),
        byte_foreign_word=list(byte.encode(FOREIGN_WORD).texts),
        superwords=[
            _superword_summary(
                "four words",
                FOUR_WORD_CORPUS,
                transition_size=22,
                vocabulary_size=25,
                example_text="low lowest",
            ),
            _superword_summary(
                "eighteen sentences",
                SENTENCE_CORPUS,
                transition_size=137,
                vocabulary_size=145,
                example_text="The report was the reason.",
            ),
        ],
        morphology=_morphology(),
    )


@router.get("/variants", response_model=VariantsResponse)
def variants() -> VariantsResponse:
    """The plain method beside the byte, superword and morph-constrained ones."""
    return _variants()


# --------------------------------------------------------------------------
# /limits: the tie, and greedy against exhaustive.
# --------------------------------------------------------------------------


class TieBranch(BaseModel):
    """One of the tied pairs taken, and what the rest of the fit became."""

    pair: str
    chosen_by_the_rule: bool
    tokens: list[str]
    corpus_pieces: int
    unseen_pieces: list[str]


class TieView(BaseModel):
    """A merge step where several pairs reached the same count."""

    step: int
    count: int
    unseen_word: str
    branches: list[TieBranch]


class GreedyRow(BaseModel):
    """One merge budget, greedy against the best sequence that exists."""

    budget: int
    greedy_pieces: int
    best_pieces: int
    greedy_merges: list[str]
    best_merges: list[str]


class LimitsResponse(BaseModel):
    """The facts the closing part rests on, every one of them measured.

    The lost-boundary fields are the method exactly as published, since that
    is the case the closing part is about; the repaired pair beside them is
    the same two words read once every character has a spelling.
    """

    ties: list[TieView]
    trap_words: list[SpelledEntry]
    trap_start_pieces: int
    trap_first_tie: list[str]
    trap_first_tie_count: int
    greedy_rows: list[GreedyRow]
    lost_boundary_text: str
    lost_boundary_pieces: list[str]
    lost_boundary_decoded: str
    lost_boundary_repaired_pieces: list[str]
    lost_boundary_repaired_decoded: str


def _tie_at(step: int, n_merges: int) -> TieView:
    counts = FOUR_WORD_COUNTS
    alphabet = _alphabet(FOUR_WORD_CORPUS)
    fitted = BytePairEncoding(vocabulary_size=len(alphabet) + 1 + n_merges).fit(
        FOUR_WORD_CORPUS
    )
    prefix = list(islice(fitted.merges, step))

    current = _spelled(counts)
    for merge in prefix:
        current = [word.with_pair_merged(merge) for word in current]
    pair_counts = _pair_counts(current)
    best = max(pair_counts.values())
    tied = sorted(pair for pair, count in pair_counts.items() if count == best)

    branches: list[TieBranch] = []
    for left, right in tied:
        first = Merge(left, right, best)
        walked = [word.with_pair_merged(first) for word in current]
        rest = learn_merges(
            walked, n_merges=n_merges - step - 1, minimum_pair_frequency=2
        )
        whole = Merges([*prefix, first, *rest])
        branches.append(
            TieBranch(
                pair=f"{left} + {right}",
                chosen_by_the_rule=(left, right)
                == (fitted.merges[step].left, fitted.merges[step].right),
                tokens=[merge.merged for merge in whole],
                corpus_pieces=_corpus_pieces(counts, whole),
                unseen_pieces=list(apply_merges(_symbols_of(UNSEEN_WORD), whole)),
            )
        )
    return TieView(
        step=step, count=int(best), unseen_word=UNSEEN_WORD, branches=branches
    )


def _best_sequence(counts: dict[str, int], budget: int) -> tuple[int, list[str]]:
    """The shortest corpus any ``budget`` merges can reach, by enumeration.

    Written here rather than in the library because greedy choice *is* the
    method; enumerating what a non-greedy trainer could have reached is the
    page's question, not the trainer's.
    """
    best: list[tuple[int, list[str]]] = []

    def record(chosen: list[Merge]) -> None:
        total = _corpus_pieces(counts, Merges(chosen))
        if not best or total < best[0][0]:
            best.clear()
            best.append((total, _merge_labels(Merges(chosen))))

    def walk(current: list[SpelledWord], chosen: list[Merge], left: int) -> None:
        pair_counts = _pair_counts(current)
        candidates = [pair for pair, count in pair_counts.items() if count >= 2]
        if left == 0 or not candidates:
            record(chosen)
            return
        for pair in candidates:
            merge = Merge(pair[0], pair[1], pair_counts[pair])
            walk(
                [word.with_pair_merged(merge) for word in current],
                [*chosen, merge],
                left - 1,
            )

    walk(_spelled(counts), [], budget)
    return best[0]


@cache
def _limits() -> LimitsResponse:
    counts = GREEDY_TRAP_COUNTS
    start_counts = _pair_counts(_spelled(counts))
    best_count = max(start_counts.values())

    greedy_rows: list[GreedyRow] = []
    for budget in GREEDY_TRAP_BUDGETS:
        greedy = learn_merges(
            _spelled(counts), n_merges=budget, minimum_pair_frequency=2
        )
        best_pieces, best_merges = _best_sequence(counts, budget)
        greedy_rows.append(
            GreedyRow(
                budget=budget,
                greedy_pieces=_corpus_pieces(counts, greedy),
                best_pieces=best_pieces,
                greedy_merges=_merge_labels(greedy),
                best_merges=best_merges,
            )
        )

    lost_text = "Alvarez didn't"
    lost = BytePairEncoding(
        vocabulary_size=MAX_VOCABULARY_SIZE, byte_fallback=False
    ).fit(SENTENCE_CORPUS)
    lost_encoding = lost.encode(lost_text)
    repaired = BytePairEncoding(vocabulary_size=MAX_VOCABULARY_SIZE).fit(
        SENTENCE_CORPUS
    )
    repaired_encoding = repaired.encode(lost_text)

    return LimitsResponse(
        ties=[_tie_at(0, 10), _tie_at(3, 10)],
        trap_words=[
            SpelledEntry(word=word, count=count, symbols=list(_symbols_of(word)))
            for word, count in counts.items()
        ],
        trap_start_pieces=_corpus_pieces(counts, Merges([])),
        trap_first_tie=sorted(
            f"{left} + {right}"
            for (left, right), count in start_counts.items()
            if count == best_count
        ),
        trap_first_tie_count=int(best_count),
        greedy_rows=greedy_rows,
        lost_boundary_text=lost_text,
        lost_boundary_pieces=list(lost_encoding.texts),
        lost_boundary_decoded=lost.decode(lost_encoding.ids),
        lost_boundary_repaired_pieces=list(repaired_encoding.texts),
        lost_boundary_repaired_decoded=repaired.decode(repaired_encoding.ids),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The tied merges and the exhaustive comparison against greedy merging."""
    return _limits()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to learn from, a size to stop at, and a text to read back.

    ``byte_fallback`` switches between the ordinary fit, where every character
    has a spelling, and the method exactly as published, where a character the
    corpus never used has none.
    """

    corpus: CorpusChoice = CorpusChoice.SENTENCES
    vocabulary_size: int = Field(ge=2, le=MAX_VOCABULARY_SIZE)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)
    byte_fallback: bool = True

    model_config = {"extra": "forbid"}


class MergeRow(BaseModel):
    """One merge, with the count that chose it."""

    rank: int
    left: str
    right: str
    merged: str
    count: int


class FitResponse(BaseModel):
    """Everything the playground shows for one fit.

    ``learned`` is what the corpus taught and is what the requested size
    bounds; ``total_rows`` is the whole table, which adds the structural rows
    when there are any.
    """

    corpus: list[str]
    alphabet: list[str]
    smallest_size: int
    asked: int
    learned: int
    floor_rows: int
    total_rows: int
    n_merges: int
    merges: list[MergeRow]
    tokens: list[str]
    corpus_pieces: int
    corpus_character_pieces: int
    text: str
    pieces: list[str]
    n_pieces: int
    unknown_pieces: int
    decoded: str
    round_trip_exact: bool


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn a vocabulary of the requested size and read one text with it."""
    corpus = CORPORA[request.corpus]
    counts = _word_counts(corpus)
    alphabet = _alphabet(corpus)
    try:
        model = BytePairEncoding(
            vocabulary_size=request.vocabulary_size,
            byte_fallback=request.byte_fallback,
        ).fit(corpus)
        encoding = model.encode(request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    unknown = model.vocabulary.unknown_token
    decoded = model.decode(encoding.ids)
    return FitResponse(
        corpus=corpus,
        alphabet=alphabet,
        smallest_size=len(alphabet) + 1,
        asked=request.vocabulary_size,
        learned=_taught_rows(model),
        floor_rows=N_FLOOR_ROWS if request.byte_fallback else 0,
        total_rows=model.vocabulary.n_tokens,
        n_merges=model.n_merges,
        merges=[
            MergeRow(
                rank=rank,
                left=merge.left,
                right=merge.right,
                merged=merge.merged,
                count=int(merge.score),
            )
            for rank, merge in enumerate(model.merges)
        ],
        # The structural rows go last, so a prefix is exactly what the corpus
        # taught and is the only part a reader has any use for.
        tokens=list(model.vocabulary)[: _taught_rows(model)],
        corpus_pieces=_corpus_pieces(counts, model.merges),
        corpus_character_pieces=_corpus_pieces(counts, Merges([])),
        text=request.text,
        pieces=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        unknown_pieces=sum(1 for piece in encoding.texts if piece == unknown),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
    )
