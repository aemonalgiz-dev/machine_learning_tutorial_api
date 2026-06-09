"""The Word Lattice.

Every endpoint here is registered under ``/concepts/the-word-lattice``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about DictionaryLatticeSegmenter, Lattice.

What this module adds to the library
------------------------------------
The segmenter builds every candidate and answers the best path; nothing here
re-implements the recurrence as a way of answering. What is assembled around it
is the bookkeeping the page quotes, and there are four pieces of it.

First, how many complete paths a lattice holds, which is a count over the same
edges the segmenter built and is the number the search does not enumerate.
Second, the ranked readings, since a page that shows only the winner cannot show
a margin; the ranking is a k-best dynamic program over the segmenter's own edges,
and its first entry is asserted in the tests to be the path the segmenter itself
chose, which is what makes it evidence rather than a second opinion. Third, the
greedy answers from the previous page, so the repair can be measured rather than
described. Fourth, the counts of candidate substrings each method asks the word
list about, which is what the two cost.

The frequency tables are the ones the segmenter's own specs are written against
wherever such a table exists. Two do not exist there and are written here, the
counts for the wildlife park words and the six-word table in which one compound
is rarer than either of its characters; both are declared as chosen rather than
counted, and the page says so.

The long English word list is imported from the previous page's module rather
than copied, because the comparison between the two methods is only fair if both
are handed the identical list.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The only free text returned is the labels written here, the
words of the lists, and the reader's own text; a test asserts that no identifier
survives into any of them, including into a refusal.
"""

from __future__ import annotations

import math
from enum import StrEnum
from functools import cache
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    DictionaryEntry,
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.lattice import (
    DictionaryLatticeSegmenter,
    Lattice,
    LatticeEdge,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.maximum_matching import (
    MatchingDirection,
    MaximumMatchingSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.maximum_matching import ORDINARY_ENGLISH

router = APIRouter(prefix="/concepts/the-word-lattice", tags=["the-word-lattice"])


# --------------------------------------------------------------------------
# The texts and the frequency tables the page carries.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page of this section carries, spaces and all."""

SPACELESS_SENTENCE = RUNNING_SENTENCE.replace(" ", "")
"""The same sentence with its spaces taken out."""

WATER_TEXT = "thewaterunderthetable"
"""Five ordinary English words run together, where the greedy scan is led astray.

The longest entry beginning at the fourteenth character is a five-letter Greek
letter, and taking it leaves three characters no entry covers.
"""

TABLE_TEXT = "thetabledownthere"
"""Four ordinary English words run together, where two readings score identically."""

GARDEN_TEXT = "thetabledownthereinthegarden"
"""Seven ordinary English words run together, long enough for the readings to multiply."""

PARK_SENTENCE = "我们在野生动物园玩"
"""We play at the wildlife park, the sentence the greedy scan is classically shown on."""

PARK_WORDS: tuple[str, ...] = (
    "我们",
    "在",
    "在野",
    "野生",
    "生动",
    "动物园",
    "物",
    "园",
    "玩",
)
"""Nine words, six of which the sentence contains and three of which it does not."""

PARK_COUNTS: dict[str, int] = {
    "我们": 50,
    "在": 80,
    "在野": 3,
    "野生": 12,
    "生动": 5,
    "动物园": 8,
    "物": 20,
    "园": 6,
    "玩": 15,
}
"""Counts for the same nine words, written by hand rather than counted from a corpus.

Nothing on this site counts Chinese text, so these are a plausible ordering
rather than a measurement: a preposition ahead of a pronoun ahead of the two
compounds the sentence uses, and the two compounds it does not use behind them.
The page says as much where it shows them.
"""

RESEARCH_SENTENCE = "研究生命起源"
"""Research into the origin of life, or a graduate student's fate and origin."""

LIFE_FAVOURING: dict[str, int] = {
    "研究": 10,
    "生命": 8,
    "起源": 5,
    "研究生": 6,
    "命": 4,
}
"""The frequency table under which the research reading wins, from the segmenter's spec."""

STUDENT_FAVOURING: dict[str, int] = {
    "研究": 10,
    "生命": 8,
    "起源": 5,
    "研究生": 20,
    "命": 12,
}
"""The same five words with the graduate student reading favoured, also from the spec."""

SPLIT_FAVOURING: dict[str, int] = {
    "研究": 6,
    "研究生": 1,
    "生命": 3,
    "生": 10,
    "命": 10,
    "起源": 2,
}
"""A sixth word added, and a table in which a compound is rarer than its characters.

Written here rather than taken from the spec. The two characters of the compound
meaning life are both very common words in their own right and the compound is
not, which is ordinary in Chinese, and it is the arrangement under which the
total the frequencies are divided by decides whether the compound survives.
"""

UNRELATED_WORD = "很"
"""A word the research sentence does not contain, added to watch the total move.

It shares no character with the sentence, so nothing about the sentence's own
candidates changes when it joins the list.
"""

NAME_SENTENCE = "阿尔瓦雷斯研究生命起源"
"""The research sentence behind a transliterated surname no word list holds."""

NAME_READING: tuple[str, ...] = ("阿尔瓦雷斯", "研究", "生命", "起源")
"""What a reader answers for that sentence, which is four words rather than eight."""

PARK_READING: tuple[str, ...] = ("我们", "在", "野生", "动物园", "玩")
"""What a reader answers for the wildlife park sentence."""

WATER_READING: tuple[str, ...] = ("the", "water", "under", "the", "table")
TABLE_READING: tuple[str, ...] = ("the", "table", "down", "there")
GARDEN_READING: tuple[str, ...] = (
    "the",
    "table",
    "down",
    "there",
    "in",
    "the",
    "garden",
)
SENTENCE_READING: tuple[str, ...] = (
    "Dr.",
    "Alvarez",
    "didn't",
    "expect",
    "the",
    "low-cost",
    "re-analysis.",
)


class WordListKey(StrEnum):
    """Which list of words, with which counts, a request is scored against.

    Attributes
    ----------
    ENGLISH:
        The common English words of the previous page, every count one.
    PARK_PLAIN:
        The nine wildlife park words, every count one.
    PARK_COUNTED:
        The same nine with the hand-written counts.
    RESEARCH_LIFE:
        The five words of the research sentence, research favoured.
    RESEARCH_STUDENT:
        The same five, the graduate student favoured.
    RESEARCH_SPLIT:
        Six words, with the compound rarer than either of its characters.
    """

    ENGLISH = "english"
    PARK_PLAIN = "park plain"
    PARK_COUNTED = "park counted"
    RESEARCH_LIFE = "research life"
    RESEARCH_STUDENT = "research student"
    RESEARCH_SPLIT = "research split"


WORD_LISTS: dict[WordListKey, dict[str, int]] = {
    WordListKey.ENGLISH: {word: 1 for word in ORDINARY_ENGLISH},
    WordListKey.PARK_PLAIN: {word: 1 for word in PARK_WORDS},
    WordListKey.PARK_COUNTED: dict(PARK_COUNTS),
    WordListKey.RESEARCH_LIFE: dict(LIFE_FAVOURING),
    WordListKey.RESEARCH_STUDENT: dict(STUDENT_FAVOURING),
    WordListKey.RESEARCH_SPLIT: dict(SPLIT_FAVOURING),
}

WORD_LIST_LABELS: dict[WordListKey, str] = {
    WordListKey.ENGLISH: "common English words, every count one",
    WordListKey.PARK_PLAIN: "the wildlife park words, every count one",
    WordListKey.PARK_COUNTED: "the wildlife park words with counts",
    WordListKey.RESEARCH_LIFE: "the research words, research favoured",
    WordListKey.RESEARCH_STUDENT: "the research words, the graduate student favoured",
    WordListKey.RESEARCH_SPLIT: "six words, the compound rarer than its characters",
}

SHOWN_WORD_LISTS: frozenset[WordListKey] = frozenset(
    key for key in WordListKey if key is not WordListKey.ENGLISH
)
"""The lists short enough to send whole. Eight hundred words is not something to show."""


MAX_TEXT_CHARACTERS = 60
"""How long a text a request may carry, a little longer than the running sentence.

The bound keeps the count of readings honest as well as fast, since that count is
returned as digits and grows with the length of the text.
"""

MAX_REMOVED_WORDS = 20
"""How many entries one request may strike out of the list it chose."""

MAX_UNRELATED_FREQUENCY = 500
"""The largest count of other words a request may add to a list.

It is what raises the total every score is divided by, so it is the control that
shows a cut moving because of text the sentence does not contain.
"""

MAX_LISTED_READINGS = 6
"""How many readings are ranked, however many the lattice turns out to hold."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]
BoundedWord = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("a word dictionary needs at least one entry", "a word list cannot be empty"),
    ("dictionary word", "word of the list"),
    ("dictionary", "word list"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# The shapes every endpoint answers in.
# --------------------------------------------------------------------------


class Entry(BaseModel):
    """One word of the list, how often it was counted, and what it scores."""

    word: str
    frequency: int
    log_score: float


class Edge(BaseModel):
    """One candidate word at one position, with the span it covers and its score.

    ``frequency`` is zero exactly when the list does not hold the candidate, which
    happens only for a lone character standing in for a word nobody wrote down.
    """

    word: str
    start: int
    end: int
    length: int
    frequency: int
    log_score: float
    on_best_path: bool


class Reading(BaseModel):
    """One complete path through the lattice, its words and its total.

    ``share_of_best`` is how likely this reading is against the best one, which
    is the exponential of the difference between the two totals, so a value of
    0.3 says the reading is three tenths as likely as the winner under these
    counts.
    """

    words: list[str]
    n_words: int
    total_log_score: float
    share_of_best: float
    matches_reader: bool


class GreedyAnswer(BaseModel):
    """What the previous page's rule answers for the same text and the same list."""

    scan: str
    words: list[str]
    n_words: int
    n_lookups: int
    matches_reader: bool


class Analysis(BaseModel):
    """One text against one list of words, laid out and scored every way the page quotes.

    ``n_cuts`` is how many ways the characters could be cut in principle and
    ``n_readings`` how many of those the list permits, both as digits because both
    outgrow a number. ``n_edges`` is how many candidates the lattice holds, which
    is also how many steps the search takes, and ``n_lookups`` is how many
    substrings the list was asked about while the candidates were found.
    """

    key: str
    label: str
    text: str
    n_characters: int
    word_list_label: str
    n_words_in_list: int
    total_frequency: int
    longest_entry: int
    entries: list[Entry]
    edges: list[Edge]
    n_edges: int
    n_cuts: str
    n_readings: str
    n_lookups: int
    readings: list[Reading]
    best: Reading
    runner_up: Reading | None
    margin: float
    greedy: list[GreedyAnswer]
    reader_reading: list[str] | None


class ScenariosResponse(BaseModel):
    """Every fixed text of the page, in the order the page meets them."""

    scenarios: list[Analysis]


class SetupResponse(BaseModel):
    """What the greedy scan left behind, restated in the numbers this page repairs.

    Each figure here is the previous page's answer for one of this page's texts,
    kept so the repair can be shown as a change in a number rather than asserted.
    """

    sentence: str
    spaceless_sentence: str
    n_spaceless_characters: int
    n_words_with_spaces: int
    n_english_words: int
    longest_english_entry: int
    park_sentence: str
    park_greedy_left: list[str]
    park_greedy_right: list[str]
    park_best_path: list[str]
    water_text: str
    water_greedy_left: list[str]
    water_best_path: list[str]
    water_reader_reading: list[str]


class ScaleStep(BaseModel):
    """One prefix of a text, and what it costs to find the best reading of it."""

    n_characters: int
    n_cuts: str
    n_readings: str
    n_edges: int
    n_lookups: int


class ScaleResponse(BaseModel):
    """How the readings and the work grow as a text gets longer.

    The readings grow with the length of the text and the work does not, which is
    the whole of why the best path is affordable.
    """

    text: str
    word_list_label: str
    steps: list[ScaleStep]


class GrowthStep(BaseModel):
    """The same sentence scored against a list that has grown by an unrelated word.

    ``whole_score`` reads the compound as one word and ``split_score`` reads it as
    its two characters, both from the definition, so the crossing can be seen as
    two lines meeting rather than as an answer changing.
    """

    added_frequency: int
    total_frequency: int
    best: list[str]
    best_score: float
    whole_score: float
    split_score: float
    whole_wins: bool


class ScalingStep(BaseModel):
    """Every count multiplied through by one factor, and the answer that does not move.

    The control for the growth above. Counting the same text several times over
    raises the total and raises every count with it, and the comparison between
    two readings is unchanged by that, so what moves a cut is specifically the
    counts of words the text does not use.
    """

    factor: int
    total_frequency: int
    best: list[str]
    best_score: float


class GrowthResponse(BaseModel):
    """Adding a word the text does not contain, and watching the cut move.

    ``crossing_total`` is the first total at which the compound survives, and
    ``exact_threshold`` is the total the two readings would score identically at,
    which is a product of two counts divided by a third. ``scalings`` is the
    control in which the total rises and nothing happens.
    """

    text: str
    added_word: str
    base_total: int
    crossing_total: int
    exact_threshold: float
    steps: list[GrowthStep]
    scalings: list[ScalingStep]


class SegmentRequest(BaseModel):
    """A text, a list of words with counts, entries struck out, and counts added."""

    text: BoundedText
    word_list: WordListKey = WordListKey.RESEARCH_LIFE
    removed: list[BoundedWord] = Field(
        default_factory=list, max_length=MAX_REMOVED_WORDS
    )
    unrelated_frequency: int = Field(default=0, ge=0, le=MAX_UNRELATED_FREQUENCY)

    model_config = {"extra": "forbid"}


class SegmentResponse(BaseModel):
    """The lattice for the requested text, ranked, with what was struck out of the list.

    ``words_in_list`` is the whole of a short list in the order it was written,
    struck entries included, so a caller can show what was taken out beside what
    was left. The long English list is not sent.
    """

    analysis: Analysis
    words_in_list: list[Entry]
    removed: list[str]
    unrelated_frequency: int


# --------------------------------------------------------------------------
# The work.
# --------------------------------------------------------------------------


def _dictionary(frequencies: dict[str, int]) -> WordDictionary:
    """The list of words with their counts, as something a lattice can be built from."""
    return WordDictionary(
        [DictionaryEntry(word, frequency) for word, frequency in frequencies.items()]
    )


def _n_cuts(text: str) -> int:
    """How many ways the characters of a text could be cut, ignoring any word list.

    A run of ``n`` characters has ``n - 1`` places a boundary could go and each is
    free, so a text is the product of two to that power over its runs.
    """
    total = 1
    for match in NON_WHITESPACE_RUN.finditer(text):
        total *= 2 ** (len(match.group()) - 1)
    return total


def _n_lookups(text: str, dictionary: WordDictionary) -> int:
    """Every candidate substring the list is asked about while the lattice is built.

    Each position offers every length up to the longest entry, and unlike the
    greedy scan none of them is skipped, because a candidate that loses at one
    position may still be on the best path.
    """
    longest = dictionary.longest_word_length
    return sum(
        min(longest, len(match.group()) - position)
        for match in NON_WHITESPACE_RUN.finditer(text)
        for position in range(len(match.group()))
    )


def _greedy_lookups(text: str, dictionary: WordDictionary, forward: bool) -> int:
    """How many substrings the greedy scan asks about, counted as the previous page does.

    It offers the longest candidate first and stops at the first entry it finds,
    so a position where a long word matches costs one question and a position
    where nothing matches costs as many as the longest entry is long.
    """
    asked = 0
    for match in NON_WHITESPACE_RUN.finditer(text):
        run = match.group()
        if forward:
            position = 0
            while position < len(run):
                longest = min(dictionary.longest_word_length, len(run) - position)
                taken = run[position]
                for length in range(longest, 1, -1):
                    asked += 1
                    candidate = run[position : position + length]
                    if candidate in dictionary:
                        taken = candidate
                        break
                position += len(taken)
        else:
            end = len(run)
            while end > 0:
                longest = min(dictionary.longest_word_length, end)
                taken = run[end - 1]
                for length in range(longest, 1, -1):
                    asked += 1
                    candidate = run[end - length : end]
                    if candidate in dictionary:
                        taken = candidate
                        break
                end -= len(taken)
    return asked


def _n_readings(lattice: Lattice) -> int:
    """How many complete paths run from one end of the text to the other.

    Counted over the segmenter's own candidates, from the right, by the same
    recurrence the best path uses with a sum where that one takes a maximum. A
    position with no candidate is whitespace and is stepped over.
    """
    n_positions = lattice.n_positions
    ways = [0] * (n_positions + 1)
    ways[n_positions] = 1
    for position in range(n_positions - 1, -1, -1):
        edges = lattice.edges_from(position)
        if not edges:
            ways[position] = ways[position + 1]
            continue
        ways[position] = sum(ways[edge.end] for edge in edges)
    return ways[0]


class RankedPath:
    """One complete path while the ranking is being built: its total and its words.

    A small holder rather than a pair, because the two travel together through
    the whole of the ranking and a caller reading position zero of a tuple is
    exactly the confusion this avoids.
    """

    __slots__ = ("first_length", "total", "words")

    def __init__(self, total: float, words: tuple[str, ...], first_length: int) -> None:
        self.total = total
        self.words = words
        self.first_length = first_length


def _ranked_paths(lattice: Lattice, wanted: int) -> list[RankedPath]:
    """The highest-scoring complete paths, best first.

    The same recurrence as the best path, keeping the leading few answers at each
    position rather than only the winner. Ties are ordered the way the segmenter
    orders them, by the longer leading word and then by the order the candidates
    were listed in, so the first entry here is the path the segmenter itself
    answers with; the tests hold it to that on every text the page carries.
    """
    n_positions = lattice.n_positions
    table: list[list[RankedPath]] = [[] for _ in range(n_positions + 1)]
    table[n_positions] = [RankedPath(0.0, (), 0)]

    for position in range(n_positions - 1, -1, -1):
        edges = lattice.edges_from(position)
        if not edges:
            table[position] = list(table[position + 1])
            continue
        found: list[tuple[float, int, int, RankedPath, LatticeEdge]] = []
        for order, edge in enumerate(edges):
            for tail in table[edge.end]:
                found.append(
                    (
                        -(edge.log_score + tail.total),
                        -edge.length,
                        order,
                        tail,
                        edge,
                    )
                )
        found.sort(key=lambda item: (item[0], item[1], item[2]))
        table[position] = [
            RankedPath(-negated_total, (edge.word,) + tail.words, edge.length)
            for negated_total, _, _, tail, edge in found[:wanted]
        ]
    return table[0]


def _reading(path: RankedPath, best_total: float, reader: tuple[str, ...] | None):
    """One ranked path as the page shows it, against the winner and against a reader."""
    return Reading(
        words=list(path.words),
        n_words=len(path.words),
        total_log_score=path.total,
        share_of_best=math.exp(path.total - best_total),
        matches_reader=reader is not None and tuple(path.words) == reader,
    )


def _greedy(
    text: str, dictionary: WordDictionary, reader: tuple[str, ...] | None
) -> list[GreedyAnswer]:
    """What the previous page's rule answers from each end, for the same list."""
    answers: list[GreedyAnswer] = []
    for label, direction, forward in (
        ("scanned from the left", MatchingDirection.FORWARD, True),
        ("scanned from the right", MatchingDirection.BACKWARD, False),
    ):
        words = MaximumMatchingSegmenter(
            dictionary=dictionary, direction=direction
        ).split(text)
        answers.append(
            GreedyAnswer(
                scan=label,
                words=list(words.texts),
                n_words=words.n_words,
                n_lookups=_greedy_lookups(text, dictionary, forward),
                matches_reader=reader is not None and tuple(words.texts) == reader,
            )
        )
    return answers


def _analysis(
    key: str,
    label: str,
    text: str,
    frequencies: dict[str, int],
    word_list_label: str,
    reader: tuple[str, ...] | None,
    listed: bool = True,
) -> Analysis:
    """One text laid out as a lattice, ranked, and measured against the greedy scan."""
    dictionary = _dictionary(frequencies)
    lattice = DictionaryLatticeSegmenter(dictionary=dictionary).lattice_of(text)
    ranked = _ranked_paths(lattice, MAX_LISTED_READINGS)
    best_total = ranked[0].total if ranked else 0.0
    on_best = {
        (edge.start, edge.end)
        for edge in DictionaryLatticeSegmenter(dictionary=dictionary)
        .lattice_of(text)
        .best_path()
    }

    readings = [_reading(path, best_total, reader) for path in ranked]

    return Analysis(
        key=key,
        label=label,
        text=text,
        n_characters=len(text),
        word_list_label=word_list_label,
        n_words_in_list=dictionary.n_words,
        total_frequency=dictionary.total_frequency,
        longest_entry=dictionary.longest_word_length,
        entries=(
            [
                Entry(
                    word=entry.word,
                    frequency=entry.frequency,
                    log_score=math.log(entry.frequency / dictionary.total_frequency),
                )
                for entry in dictionary
            ]
            if listed
            else []
        ),
        edges=[
            Edge(
                word=edge.word,
                start=edge.start,
                end=edge.end,
                length=edge.length,
                frequency=dictionary.frequency_of(edge.word),
                log_score=edge.log_score,
                on_best_path=(edge.start, edge.end) in on_best,
            )
            for edge in lattice
        ],
        n_edges=lattice.n_edges,
        n_cuts=str(_n_cuts(text)),
        n_readings=str(_n_readings(lattice)),
        n_lookups=_n_lookups(text, dictionary),
        readings=readings,
        best=readings[0],
        runner_up=readings[1] if len(readings) > 1 else None,
        margin=(
            readings[0].total_log_score - readings[1].total_log_score
            if len(readings) > 1
            else 0.0
        ),
        greedy=_greedy(text, dictionary, reader),
        reader_reading=list(reader) if reader is not None else None,
    )


ENGLISH_FREQUENCIES: dict[str, int] = {word: 1 for word in ORDINARY_ENGLISH}


@cache
def _scenarios() -> ScenariosResponse:
    """The page's fixed texts, each laid out and ranked, computed once."""
    english_label = WORD_LIST_LABELS[WordListKey.ENGLISH]
    return ScenariosResponse(
        scenarios=[
            _analysis(
                key="research",
                label="the sentence with two defensible readings",
                text=RESEARCH_SENTENCE,
                frequencies=LIFE_FAVOURING,
                word_list_label=WORD_LIST_LABELS[WordListKey.RESEARCH_LIFE],
                reader=None,
            ),
            _analysis(
                key="research-student",
                label="the same sentence, the other set of counts",
                text=RESEARCH_SENTENCE,
                frequencies=STUDENT_FAVOURING,
                word_list_label=WORD_LIST_LABELS[WordListKey.RESEARCH_STUDENT],
                reader=None,
            ),
            _analysis(
                key="water",
                label="five English words run together",
                text=WATER_TEXT,
                frequencies=ENGLISH_FREQUENCIES,
                word_list_label=english_label,
                reader=WATER_READING,
                listed=False,
            ),
            _analysis(
                key="table",
                label="four English words run together",
                text=TABLE_TEXT,
                frequencies=ENGLISH_FREQUENCIES,
                word_list_label=english_label,
                reader=TABLE_READING,
                listed=False,
            ),
            _analysis(
                key="park-plain",
                label="we play at the wildlife park, on a bare list",
                text=PARK_SENTENCE,
                frequencies={word: 1 for word in PARK_WORDS},
                word_list_label=WORD_LIST_LABELS[WordListKey.PARK_PLAIN],
                reader=PARK_READING,
            ),
            _analysis(
                key="park-counted",
                label="the same sentence, the same words, with counts",
                text=PARK_SENTENCE,
                frequencies=PARK_COUNTS,
                word_list_label=WORD_LIST_LABELS[WordListKey.PARK_COUNTED],
                reader=PARK_READING,
            ),
            _analysis(
                key="name",
                label="the research sentence behind a surname",
                text=NAME_SENTENCE,
                frequencies=LIFE_FAVOURING,
                word_list_label=WORD_LIST_LABELS[WordListKey.RESEARCH_LIFE],
                reader=NAME_READING,
            ),
            _analysis(
                key="spaceless",
                label="our sentence with its spaces taken out",
                text=SPACELESS_SENTENCE,
                frequencies=ENGLISH_FREQUENCIES,
                word_list_label=english_label,
                reader=SENTENCE_READING,
                listed=False,
            ),
            _analysis(
                key="garden",
                label="seven English words run together",
                text=GARDEN_TEXT,
                frequencies=ENGLISH_FREQUENCIES,
                word_list_label=english_label,
                reader=GARDEN_READING,
                listed=False,
            ),
        ]
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    """The page's texts, each laid out as a lattice and ranked by whole-path score."""
    return _scenarios()


@cache
def _setup() -> SetupResponse:
    """What the greedy scan answered for this page's texts, computed once."""
    park_plain = _dictionary({word: 1 for word in PARK_WORDS})
    english = _dictionary(ENGLISH_FREQUENCIES)

    def greedy(text: str, dictionary: WordDictionary, forward: bool) -> list[str]:
        direction = MatchingDirection.FORWARD if forward else MatchingDirection.BACKWARD
        return list(
            MaximumMatchingSegmenter(dictionary=dictionary, direction=direction)
            .split(text)
            .texts
        )

    def best(text: str, dictionary: WordDictionary) -> list[str]:
        return list(
            DictionaryLatticeSegmenter(dictionary=dictionary).best_path(text).words
        )

    return SetupResponse(
        sentence=RUNNING_SENTENCE,
        spaceless_sentence=SPACELESS_SENTENCE,
        n_spaceless_characters=len(SPACELESS_SENTENCE),
        n_words_with_spaces=len(SENTENCE_READING),
        n_english_words=english.n_words,
        longest_english_entry=english.longest_word_length,
        park_sentence=PARK_SENTENCE,
        park_greedy_left=greedy(PARK_SENTENCE, park_plain, True),
        park_greedy_right=greedy(PARK_SENTENCE, park_plain, False),
        park_best_path=best(PARK_SENTENCE, park_plain),
        water_text=WATER_TEXT,
        water_greedy_left=greedy(WATER_TEXT, english, True),
        water_best_path=best(WATER_TEXT, english),
        water_reader_reading=list(WATER_READING),
    )


@router.get("/setup", response_model=SetupResponse)
def setup() -> SetupResponse:
    """The two failures of the greedy scan this page repairs, with what it answers now."""
    return _setup()


@cache
def _scale() -> ScaleResponse:
    """How the readings and the work grow along the prefixes of one text."""
    dictionary = _dictionary(ENGLISH_FREQUENCIES)
    steps: list[ScaleStep] = []
    for length in range(4, len(GARDEN_TEXT) + 1, 4):
        prefix = GARDEN_TEXT[:length]
        lattice = DictionaryLatticeSegmenter(dictionary=dictionary).lattice_of(prefix)
        steps.append(
            ScaleStep(
                n_characters=length,
                n_cuts=str(_n_cuts(prefix)),
                n_readings=str(_n_readings(lattice)),
                n_edges=lattice.n_edges,
                n_lookups=_n_lookups(prefix, dictionary),
            )
        )
    return ScaleResponse(
        text=GARDEN_TEXT,
        word_list_label=WORD_LIST_LABELS[WordListKey.ENGLISH],
        steps=steps,
    )


@router.get("/scale", response_model=ScaleResponse)
def scale() -> ScaleResponse:
    """The readings grow with the text and the work does not, measured along one text."""
    return _scale()


WHOLE_READING: tuple[str, ...] = ("研究", "生命", "起源")
SPLIT_READING: tuple[str, ...] = ("研究", "生", "命", "起源")
GROWTH_ADDITIONS: tuple[int, ...] = (0, 1, 2, 3, 5, 10, 20, 50)
SCALING_FACTORS: tuple[int, ...] = (1, 2, 5, 20, 100)


def _score_of(reading: tuple[str, ...], dictionary: WordDictionary) -> float:
    """A reading's total from the definition, one count standing in for a missing word."""
    return sum(
        math.log((dictionary.frequency_of(word) or 1) / dictionary.total_frequency)
        for word in reading
    )


@cache
def _growth() -> GrowthResponse:
    """Adding one unrelated word to the list, and the cut of the sentence moving."""
    steps: list[GrowthStep] = []
    crossing = 0
    for added in GROWTH_ADDITIONS:
        frequencies = dict(SPLIT_FAVOURING)
        if added:
            frequencies[UNRELATED_WORD] = added
        dictionary = _dictionary(frequencies)
        best = DictionaryLatticeSegmenter(dictionary=dictionary).best_path(
            RESEARCH_SENTENCE
        )
        whole = _score_of(WHOLE_READING, dictionary)
        split = _score_of(SPLIT_READING, dictionary)
        whole_wins = tuple(best.words) == WHOLE_READING
        if whole_wins and crossing == 0:
            crossing = dictionary.total_frequency
        steps.append(
            GrowthStep(
                added_frequency=added,
                total_frequency=dictionary.total_frequency,
                best=list(best.words),
                best_score=best.total_log_score,
                whole_score=whole,
                split_score=split,
                whole_wins=whole_wins,
            )
        )

    scalings: list[ScalingStep] = []
    for factor in SCALING_FACTORS:
        multiplied = _dictionary(
            {word: count * factor for word, count in SPLIT_FAVOURING.items()}
        )
        path = DictionaryLatticeSegmenter(dictionary=multiplied).best_path(
            RESEARCH_SENTENCE
        )
        scalings.append(
            ScalingStep(
                factor=factor,
                total_frequency=multiplied.total_frequency,
                best=list(path.words),
                best_score=path.total_log_score,
            )
        )

    base = _dictionary(SPLIT_FAVOURING)
    threshold = (
        base.frequency_of("生") * base.frequency_of("命") / base.frequency_of("生命")
    )
    return GrowthResponse(
        text=RESEARCH_SENTENCE,
        added_word=UNRELATED_WORD,
        base_total=base.total_frequency,
        crossing_total=crossing,
        exact_threshold=threshold,
        steps=steps,
        scalings=scalings,
    )


@router.get("/growth", response_model=GrowthResponse)
def growth() -> GrowthResponse:
    """One word the sentence does not contain, added, and the sentence cut differently."""
    return _growth()


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest) -> SegmentResponse:
    """Lay the requested text out against the chosen list, less what was struck out."""
    struck = set(request.removed)
    whole = WORD_LISTS[request.word_list]
    frequencies = {
        word: frequency for word, frequency in whole.items() if word not in struck
    }
    if request.unrelated_frequency:
        frequencies = dict(frequencies)
        frequencies[UNRELATED_WORD] = request.unrelated_frequency

    try:
        analysis = _analysis(
            key="requested",
            label="your text",
            text=request.text,
            frequencies=frequencies,
            word_list_label=WORD_LIST_LABELS[request.word_list],
            reader=None,
            listed=request.word_list in SHOWN_WORD_LISTS,
        )
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    total = sum(frequencies.values())
    listed = (
        [
            Entry(
                word=word,
                frequency=frequency,
                log_score=math.log(frequency / total),
            )
            for word, frequency in whole.items()
        ]
        if request.word_list in SHOWN_WORD_LISTS
        else []
    )

    return SegmentResponse(
        analysis=analysis,
        words_in_list=listed,
        removed=[word for word in whole if word in struck],
        unrelated_frequency=request.unrelated_frequency,
    )
