"""Finite-State Morphology.

Every endpoint here is registered under ``/concepts/finite-state-morphology``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about FiniteStateAnalyzer.

What this module adds to the library
------------------------------------
Three grammars and four measurements. The analyser itself produces every
analysis, every morph and every offset on the page; nothing here reimplements
the search as a way of answering.

The grammars are data. The small one is the toy English morphology the
library's own spec carries, in both its versions, the one that cannot read
``baked`` and the one that can, with its states renamed into ordinary words so
that a reader meets a state called a verb ending rather than an identifier. The
larger one is written here, thirty-six verbs and forty nouns and three
prefixes, assembled from word lists rather than typed out one line at a time,
which is what a real lexicon compiler is handed.

The measurements are the bookkeeping the page quotes and the analyser has no
reason to carry. First, the language a grammar accepts: every string a path
through it spells, found by walking the states with no word in hand, which is
possible only because every step consumes a character and no state can be
reached from itself. Second, the walk itself, the arcs tried at each position
and which of them led anywhere, which is a second implementation of the search
and is treated as one -- the tests assert that the paths it says succeeded are
exactly the analyses the analyser returns. Third, coverage, which is the share
of a text the grammar can read at all. Fourth, the comparison against a learned
vocabulary fitted on the grammar's own language, which is the most generous
corpus a learned method could be handed here.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The states are named in ordinary words for that reason, and
a refusal is rewritten on the way out; a test asserts that nothing
identifier-shaped survives into any string any endpoint returns.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from functools import cache
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.morphology.finite_state import (
    END,
    Analyses,
    Analysis,
    FiniteStateAnalyzer,
    Lexicon,
    LexiconEntry,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/finite-state-morphology", tags=["finite-state-morphology"]
)


# --------------------------------------------------------------------------
# The texts the page carries.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page of this section carries."""

PARAGRAPH = (
    "The market opens early and the first hour is the busiest. Traders set out "
    "their goods, call to the people passing, and change their prices as the "
    "morning goes on. By noon the crowd has thinned, the best of the produce "
    "has gone, and what is left is sold at whatever it will fetch. A "
    "stallholder who has watched the same square for twenty years can tell you "
    "within a pound what a Saturday will bring."
)
"""Seventy-six words of ordinary English, written before the grammar was measured on it.

Nothing in it was chosen to suit or to defeat the grammar below, which is the
only way a coverage figure means anything. It is prose about an ordinary
subject, with the mixture of function words, irregular verbs, proper names and
compounds that ordinary prose has.
"""


# --------------------------------------------------------------------------
# The states, named in the page's own words.
# --------------------------------------------------------------------------

WORD = "word"
"""Where every walk starts."""

STEM = "stem"
"""Where a walk arrives after taking a prefix."""

VERB_ENDING = "verb ending"
NOUN_ENDING = "noun ending"
CONSONANT_ENDING = "ending that starts with a consonant"
VOWEL_ENDING = "ending that starts with a vowel"

END_OF_WORD = "the end of the word"
"""What the page calls the accepting state, wherever one is shown."""


# --------------------------------------------------------------------------
# The small grammar, which is the library spec's toy English morphology.
# --------------------------------------------------------------------------

VERB_ENDINGS = Lexicon(
    VERB_ENDING,
    [
        LexiconEntry("s", "+3SG", END),
        LexiconEntry("ed", "+PAST", END),
        LexiconEntry("ing", "+PROG", END),
    ],
)

NOUN_ENDINGS = Lexicon(NOUN_ENDING, [LexiconEntry("s", "+PL", END)])

SMALL_BEFORE = FiniteStateAnalyzer(
    root=WORD,
    lexicons=(
        Lexicon(
            WORD,
            [
                LexiconEntry("walk", "walk+V", VERB_ENDING),
                LexiconEntry("walk", "walk+V", END),
                LexiconEntry("talk", "talk+V", VERB_ENDING),
                LexiconEntry("talk", "talk+V", END),
                LexiconEntry("bake", "bake+V", VERB_ENDING),
                LexiconEntry("bake", "bake+V", END),
                LexiconEntry("walk", "walk+N", NOUN_ENDING),
                LexiconEntry("walk", "walk+N", END),
            ],
        ),
        VERB_ENDINGS,
        NOUN_ENDINGS,
    ),
)
"""Three verbs and their endings, before the spelling change is written down.

Every verb reaches every ending, so this grammar cannot read ``baked`` and
happily reads ``bakeed``, which is the pair the page uses.
"""

SMALL = FiniteStateAnalyzer(
    root=WORD,
    lexicons=(
        Lexicon(
            WORD,
            [
                LexiconEntry("walk", "walk+V", VERB_ENDING),
                LexiconEntry("walk", "walk+V", END),
                LexiconEntry("talk", "talk+V", VERB_ENDING),
                LexiconEntry("talk", "talk+V", END),
                LexiconEntry("bake", "bake+V", CONSONANT_ENDING),
                LexiconEntry("bake", "bake+V", END),
                LexiconEntry("bak", "bake+V", VOWEL_ENDING),
                LexiconEntry("walk", "walk+N", NOUN_ENDING),
                LexiconEntry("walk", "walk+N", END),
            ],
        ),
        VERB_ENDINGS,
        Lexicon(CONSONANT_ENDING, [LexiconEntry("s", "+3SG", END)]),
        Lexicon(
            VOWEL_ENDING,
            [LexiconEntry("ed", "+PAST", END), LexiconEntry("ing", "+PROG", END)],
        ),
        NOUN_ENDINGS,
    ),
)
"""The same three verbs with the dropped ``e`` written out as a second spelling."""


# --------------------------------------------------------------------------
# The larger grammar, assembled from word lists.
# --------------------------------------------------------------------------

CONSONANT_VERBS: tuple[str, ...] = (
    "walk",
    "talk",
    "work",
    "look",
    "call",
    "help",
    "open",
    "order",
    "form",
    "cover",
    "recover",
    "play",
    "want",
    "need",
    "start",
    "turn",
    "point",
    "answer",
    "follow",
    "offer",
    "report",
    "record",
    "learn",
    "expect",
)
"""Twenty-four verbs whose stem is unchanged by any ending the grammar knows.

Every one of them takes ``s``, ``ed``, ``ing`` and ``er`` with nothing else
happening to the spelling, which is the case a morphology can state in two
lines. Verbs whose final consonant doubles, and verbs whose ``y`` becomes an
``i``, are deliberately absent, since neither spelling change is written down
here and the page says so rather than hiding it.
"""

SILENT_E_VERBS: tuple[str, ...] = (
    "bake",
    "move",
    "use",
    "close",
    "name",
    "place",
    "note",
    "care",
    "change",
    "produce",
    "measure",
    "continue",
)
"""Twelve verbs that drop a final ``e`` before an ending that starts with a vowel."""

NOUNS: tuple[str, ...] = (
    "walk",
    "work",
    "cost",
    "form",
    "order",
    "name",
    "part",
    "number",
    "hand",
    "word",
    "point",
    "cover",
    "question",
    "answer",
    "result",
    "method",
    "reason",
    "value",
    "place",
    "note",
    "care",
    "car",
    "book",
    "letter",
    "paper",
    "group",
    "list",
    "line",
    "page",
    "term",
    "report",
    "record",
    "view",
    "sign",
    "market",
    "price",
    "year",
    "time",
    "way",
    "thing",
)
"""Forty nouns whose plural is the stem with an ``s`` on it.

Ten of them are also in one of the verb lists, which is where most of the
grammar's ambiguity comes from.
"""

PREFIXES: tuple[tuple[str, str], ...] = (("re", "+RE"), ("un", "+UN"), ("pre", "+PRE"))
"""Three prefixes, each of which may stand in front of any stem."""

FUNCTION_WORDS: tuple[str, ...] = (
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "as",
    "by",
    "from",
    "is",
    "was",
    "are",
    "were",
    "be",
    "it",
    "this",
    "that",
    "what",
    "who",
    "has",
    "had",
    "have",
    "will",
    "not",
    "no",
    "so",
    "if",
    "out",
    "up",
    "all",
    "can",
    "you",
    "we",
    "they",
    "he",
    "she",
    "i",
    "their",
    "its",
    "his",
    "her",
    "there",
    "then",
)
"""Fifty words with no internal structure, each of which needs its own line.

They are the commonest words in any English text and a morphology has nothing
to say about them, so listing them is not analysis; it is a list. The page
measures what the fifty lines buy.
"""


def _stem_lines() -> list[LexiconEntry]:
    """One line per stem per part of speech, plus a second for the bare form."""
    lines: list[LexiconEntry] = []
    for stem in CONSONANT_VERBS:
        lines.append(LexiconEntry(stem, f"{stem}+V", VERB_ENDING))
        lines.append(LexiconEntry(stem, f"{stem}+V", END))
    for stem in SILENT_E_VERBS:
        lines.append(LexiconEntry(stem, f"{stem}+V", CONSONANT_ENDING))
        lines.append(LexiconEntry(stem, f"{stem}+V", END))
        lines.append(LexiconEntry(stem[:-1], f"{stem}+V", VOWEL_ENDING))
    for stem in NOUNS:
        lines.append(LexiconEntry(stem, f"{stem}+N", NOUN_ENDING))
        lines.append(LexiconEntry(stem, f"{stem}+N", END))
    return lines


STEM_LINES = _stem_lines()
PREFIX_LINES = [LexiconEntry(surface, tag, STEM) for surface, tag in PREFIXES]
FUNCTION_LINES = [LexiconEntry(word, f"{word}+FUNC", END) for word in FUNCTION_WORDS]

LARGER_ENDINGS: tuple[Lexicon, ...] = (
    Lexicon(
        VERB_ENDING,
        [
            LexiconEntry("s", "+3SG", END),
            LexiconEntry("ed", "+PAST", END),
            LexiconEntry("ing", "+PROG", END),
            LexiconEntry("er", "+AGENT", END),
            LexiconEntry("er", "+AGENT", NOUN_ENDING),
        ],
    ),
    Lexicon(CONSONANT_ENDING, [LexiconEntry("s", "+3SG", END)]),
    Lexicon(
        VOWEL_ENDING,
        [
            LexiconEntry("ed", "+PAST", END),
            LexiconEntry("ing", "+PROG", END),
            LexiconEntry("er", "+AGENT", END),
            LexiconEntry("er", "+AGENT", NOUN_ENDING),
        ],
    ),
    Lexicon(NOUN_ENDING, [LexiconEntry("s", "+PL", END)]),
)


def _larger(extra: list[LexiconEntry]) -> FiniteStateAnalyzer:
    """The larger grammar, with whatever extra lines a caller has added to it."""
    return FiniteStateAnalyzer(
        root=WORD,
        lexicons=(
            Lexicon(WORD, PREFIX_LINES + STEM_LINES + extra),
            Lexicon(STEM, STEM_LINES),
            *LARGER_ENDINGS,
        ),
    )


LARGER = _larger([])
"""Thirty-six verbs, forty nouns and three prefixes."""

LARGER_WITH_FUNCTION_WORDS = _larger(FUNCTION_LINES)
"""The same, with fifty structureless words listed one at a time."""

REPEATING = FiniteStateAnalyzer(
    root=WORD,
    lexicons=(
        Lexicon(
            WORD,
            [
                LexiconEntry("a", "one", WORD),
                LexiconEntry("a", "one", END),
                LexiconEntry("aa", "two", WORD),
                LexiconEntry("aa", "two", END),
            ],
        ),
    ),
)
"""One state, four lines, and as many readings of a long word as anyone likes.

A piece of one character and a piece of two, either of which may be followed by
another, so a word of n characters has as many readings as there are ordered
ways of writing n as ones and twos. The page uses it for the growth alone.
"""


# --------------------------------------------------------------------------
# How many lines a person wrote, which is not how many arcs the machine has.
# --------------------------------------------------------------------------


def _arcs(analyzer: FiniteStateAnalyzer) -> int:
    """How many steps the machine has once the copies are made."""
    return sum(len(lexicon) for lexicon in analyzer.lexicons)


def _written_lines(analyzer: FiniteStateAnalyzer) -> int:
    """How many lines a person types to produce the grammar.

    The larger grammar repeats its whole stem list in a second state, because a
    prefix has to lead somewhere and every step of this machine consumes a
    character, so there is no step that takes a prefix and leaves the walk where
    the stems are. A person writes the stems once and names them; the copy is
    what a compiler makes of that. Every other line is typed where it appears,
    including the three endings that two states both list, so those are counted
    twice, which is how many times they are written.
    """
    by_name = {lexicon.name: lexicon for lexicon in analyzer.lexicons}
    copied = by_name.get(STEM)
    if copied is None:
        return _arcs(analyzer)

    start = set(by_name[analyzer.root])
    return _arcs(analyzer) - sum(1 for entry in copied if entry in start)


def _accepted(analyzer: FiniteStateAnalyzer) -> Counter[str]:
    """Every string a path through the grammar spells, and how many paths spell it.

    Walking the states with no word in hand terminates because every step
    consumes at least one character and no state of these grammars can be
    reached from itself, so the set of accepted strings is finite. The repeating
    grammar is the exception and is never passed here.
    """
    by_name = {lexicon.name: lexicon for lexicon in analyzer.lexicons}
    found: Counter[str] = Counter()

    def walk(state: str, so_far: str) -> None:
        if state == END:
            found[so_far] += 1
            return
        for entry in by_name[state]:
            walk(entry.continuation, so_far + entry.surface)

    walk(analyzer.root, "")
    return found


# --------------------------------------------------------------------------
# Saying what a tag means, in words.
# --------------------------------------------------------------------------

MEANINGS: dict[str, str] = {
    "+3SG": "he or she does it",
    "+PAST": "it already happened",
    "+PROG": "it is going on",
    "+PL": "more than one",
    "+AGENT": "the one who does it",
    "+RE": "again",
    "+UN": "the reverse of it",
    "+PRE": "before",
    "+PREFIX": "put in front of a stem",
}
"""What each ending or prefix is saying, for a reader who has not met the notation."""


def _meaning(tag: str) -> str:
    """One tag turned into ordinary words."""
    if tag.endswith("+V"):
        return f"the verb {tag[:-2]}"
    if tag.endswith("+N"):
        return f"the noun {tag[:-2]}"
    if tag.endswith("+FUNC"):
        return f"the word {tag[:-5]}, which has no parts"
    return MEANINGS.get(tag, tag)


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

_ENTRY_SHAPED = re.compile(r"LexiconEntry\('([^']*)', '([^']*)', '([^']*)'\)")

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    (
        (
            "a lexicon entry's surface must hold at least one character; a bare "
            "form is written as a second entry continuing to END"
        ),
        (
            "a line of a grammar needs at least one character of spelling, since "
            "every step has to read something; a bare form is written as a second "
            "line that ends the word"
        ),
    ),
    (
        "which would analyse every word through it twice",
        "which would read every word through it twice",
    ),
    ("the lexicon", "the state"),
    ("a lexicon", "a state"),
    ("lexicon", "state"),
    ("cannot analyse an empty word", "there is no word here to read"),
    (
        "analyses are of a word of at least one character",
        "there is no word here to read",
    ),
    (
        "neither a state nor END",
        "neither a state of this grammar nor the end of a word",
    ),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: Exception) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page. A refusal made
    while the machine is being checked for closure arrives wrapped in several
    lines of report, and the sentence worth showing is the one after the
    marker, so that is the line taken.
    """
    sentence = str(refusal)
    if "Value error, " in sentence:
        sentence = sentence.split("Value error, ", 1)[1].split("\n")[0].strip()
        sentence = sentence.split(" [type=")[0]
    sentence = _ENTRY_SHAPED.sub(r"the line spelled \1", sentence)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence.replace(f"{END!r}", "the end of the word")


# --------------------------------------------------------------------------
# The shapes every endpoint answers in.
# --------------------------------------------------------------------------


class GrammarKey(StrEnum):
    """Which of the page's grammars a request is asking about.

    Attributes
    ----------
    SMALL:
        Three verbs and their endings, small enough to print whole.
    SMALL_BEFORE:
        The same three verbs before the dropped ``e`` was written down.
    LARGER:
        Thirty-six verbs, forty nouns and three prefixes.
    """

    SMALL = "small"
    SMALL_BEFORE = "small before"
    LARGER = "larger"


GRAMMARS: dict[GrammarKey, FiniteStateAnalyzer] = {
    GrammarKey.SMALL: SMALL,
    GrammarKey.SMALL_BEFORE: SMALL_BEFORE,
    GrammarKey.LARGER: LARGER,
}

GRAMMAR_LABELS: dict[GrammarKey, str] = {
    GrammarKey.SMALL: "three verbs and their endings",
    GrammarKey.SMALL_BEFORE: "the same three verbs, before the spelling change",
    GrammarKey.LARGER: "thirty-six verbs, forty nouns and three prefixes",
}


class EntryKind(StrEnum):
    """What a reader is adding to a grammar, in the page's words."""

    VERB = "a verb"
    SILENT_E_VERB = "a verb ending in e"
    NOUN = "a noun"
    PREFIX = "a prefix"
    WHOLE_WORD = "a word with no parts"


class Line(BaseModel):
    """One line of a grammar: a spelling, what it means, and what may follow it.

    ``goes_to`` is the state a walk moves into after taking this line, or the
    page's phrase for the end of the word.
    """

    surface: str
    label: str
    meaning: str
    goes_to: str
    ends_the_word: bool


class State(BaseModel):
    """One state of a grammar and every line that leaves it."""

    name: str
    lines: list[Line]
    n_lines: int
    is_start: bool


class Morph(BaseModel):
    """One piece of a word, where it sits, and what it was identified as."""

    text: str
    start: int
    end: int
    label: str
    meaning: str


class AnalysisView(BaseModel):
    """One reading of one word: its pieces, its labels, and its tag string."""

    morphs: list[Morph]
    tags: str
    n_morphemes: int


class TraceStep(BaseModel):
    """One line tried at one position of the walk, and what came of it.

    ``depth`` is how many lines had already been taken when this one was tried,
    so a reader can indent the walk and see the search backing out of a dead
    end. ``leads_anywhere`` is false exactly when no continuation of this line
    reaches the end of the word.
    """

    depth: int
    at: int
    state: str
    surface: str
    label: str
    goes_to: str
    outcome: str
    leads_anywhere: bool


class CharacterState(BaseModel):
    """One character of a word and the state it was read in."""

    character: str
    at: int
    state: str
    morph_index: int


class WordView(BaseModel):
    """Everything the page shows about one word under one grammar.

    ``cut`` is the word cut by the reading with the fewest pieces, which is what
    a word is handed on as; a word with no reading at all is handed on whole,
    and ``is_readable`` is how a caller tells the two apart.
    """

    word: str
    grammar_label: str
    is_readable: bool
    n_analyses: int
    analyses: list[AnalysisView]
    trace: list[TraceStep]
    n_lines_tried: int
    states_by_character: list[CharacterState]
    cut: list[str]


class GrammarView(BaseModel):
    """One grammar written out, with the size of the language it accepts.

    ``n_written_lines`` counts the distinct lines a person writes and
    ``n_arcs`` counts the steps the machine has once a prefix state's copy of
    the stems is made. ``n_forms`` is how many different words the grammar
    accepts and ``n_paths`` how many readings those words have between them, so
    the difference between the two is the ambiguity in the grammar's own
    language.
    """

    key: str
    label: str
    n_states: int
    n_written_lines: int
    n_arcs: int
    n_forms: int
    n_paths: int
    n_ambiguous_forms: int
    mean_morphs: float
    states: list[State]
    sample_forms: list[str]


class MachineResponse(BaseModel):
    """The grammars the page shows, small one first."""

    grammars: list[GrammarView]


class LearnedCut(BaseModel):
    """One word cut two ways: by counting, and by the written grammar.

    ``learned`` is what a vocabulary fitted on the grammar's own language
    answers, with the marker that says a word ended stripped off so the two
    columns can be read side by side.
    """

    word: str
    learned: list[str]
    written: list[str]
    agree: bool


class Budget(BaseModel):
    """A learned vocabulary of a stated size, measured on the grammar's language."""

    n_pieces: int
    mean_pieces: float
    n_whole_words: int
    cuts: list[LearnedCut]


class CostResponse(BaseModel):
    """What a written grammar costs, and what the same budget buys by counting.

    ``n_written_lines`` and ``n_forms`` are the grammar's; the budgets are
    learned vocabularies of the same size and of twice it, fitted on a corpus
    holding every one of those forms once, which is the most helpful corpus a
    learned method could be given for this comparison and is not a corpus
    anybody has.
    """

    n_written_lines: int
    n_states: int
    n_forms: int
    n_paths: int
    mean_morphs: float
    max_morphs: int
    budgets: list[Budget]


class Growth(BaseModel):
    """How many readings a word of a given length has under the repeating grammar."""

    length: int
    n_readings: int


class AmbiguityResponse(BaseModel):
    """Where more than one reading is the right answer, and how fast that grows.

    ``forms`` are the words of the larger grammar's own language that it reads
    more than one way; ``growth`` is the four-line grammar whose readings count
    the ways of writing a number as ones and twos.
    """

    n_forms: int
    n_ambiguous_forms: int
    share_ambiguous: float
    most_read_words: list[WordView]
    growth: list[Growth]


class Reading(BaseModel):
    """One word of a text and whether the grammar could read it."""

    text: str
    is_readable: bool
    cut: list[str]


class CoverageView(BaseModel):
    """What share of one text one grammar can read at all.

    ``n_types`` counts distinct words and ``n_tokens`` counts occurrences, and
    the two differ because the words a grammar misses are not the words a text
    repeats.
    """

    text_label: str
    grammar_label: str
    n_written_lines: int
    n_tokens: int
    n_readable_tokens: int
    token_share: float
    n_types: int
    n_readable_types: int
    type_share: float
    unreadable_types: list[str]


class SplitterView(BaseModel):
    """What one way of finding words hands the grammar, and how much of it reads."""

    rule: str
    pieces: list[Reading]
    n_readable: int


class CoverageResponse(BaseModel):
    """Coverage on the section's sentence and on a paragraph of ordinary prose."""

    sentence: str
    splitters: list[SplitterView]
    paragraph: str
    coverages: list[CoverageView]
    function_words: list[str]


class ScenariosResponse(BaseModel):
    """The fixed words the page works through, each read by a named grammar."""

    walked: WordView
    walks: WordView
    bakes: WordView
    baked_before: WordView
    baked_after: WordView
    bakeed_before: WordView
    bakeed_after: WordView
    recovers: WordView
    reopeners: WordView
    unreadable: list[WordView]


class AddedLine(BaseModel):
    """A line a reader is adding to the grammar before it reads a word."""

    surface: Annotated[str, Field(max_length=24)]
    kind: EntryKind

    model_config = {"extra": "forbid"}


class ReadRequest(BaseModel):
    """A word, a grammar to read it with, and any lines the reader has added."""

    word: Annotated[str, Field(max_length=24)]
    grammar: GrammarKey = GrammarKey.LARGER
    added: list[AddedLine] = Field(default_factory=list, max_length=8)

    model_config = {"extra": "forbid"}


class ReadResponse(BaseModel):
    """One word read, and how large the grammar was that read it."""

    view: WordView
    n_written_lines: int
    added_lines: list[Line]


# --------------------------------------------------------------------------
# The work.
# --------------------------------------------------------------------------

_WHITESPACE = WhitespacePreTokenizer()
_TREEBANK = PennTreebankPreTokenizer()
_PUNCTUATION = ".,;:!?\"'()"


def _goes_to(entry: LexiconEntry) -> str:
    """The state a line leads into, in the page's words."""
    return END_OF_WORD if entry.leads_to_end else entry.continuation


def _line(entry: LexiconEntry) -> Line:
    """One line of a grammar, as the page shows it."""
    return Line(
        surface=entry.surface,
        label=entry.analysis,
        meaning=_meaning(entry.analysis),
        goes_to=_goes_to(entry),
        ends_the_word=entry.leads_to_end,
    )


def _states(analyzer: FiniteStateAnalyzer) -> list[State]:
    """Every state of a grammar, the start one first."""
    ordered = sorted(
        analyzer.lexicons, key=lambda lexicon: lexicon.name != analyzer.root
    )
    return [
        State(
            name=lexicon.name,
            lines=[_line(entry) for entry in lexicon],
            n_lines=len(lexicon),
            is_start=lexicon.name == analyzer.root,
        )
        for lexicon in ordered
    ]


def _analysis_view(analysis: Analysis) -> AnalysisView:
    """One reading, with each piece placed in the word."""
    morphs: list[Morph] = []
    position = 0
    for entry in analysis:
        morphs.append(
            Morph(
                text=entry.surface,
                start=position,
                end=position + len(entry.surface),
                label=entry.analysis,
                meaning=_meaning(entry.analysis),
            )
        )
        position += len(entry.surface)
    return AnalysisView(
        morphs=morphs, tags=analysis.tags, n_morphemes=analysis.n_morphemes
    )


def _trace(analyzer: FiniteStateAnalyzer, word: str) -> list[TraceStep]:
    """Every line tried, in the order the search tries them, with what came of it.

    This walks the states a second time rather than reading anything back off
    the analyser, so the tests assert that the paths it marks as leading
    somewhere spell exactly the readings the analyser returns.
    """
    by_name = {lexicon.name: lexicon for lexicon in analyzer.lexicons}
    steps: list[TraceStep] = []

    def walk(position: int, state: str, depth: int) -> bool:
        if state == END:
            return position == len(word)
        if position >= len(word):
            return False

        anywhere = False
        for entry in by_name[state]:
            if not word.startswith(entry.surface, position):
                continue
            index = len(steps)
            steps.append(
                TraceStep(
                    depth=depth,
                    at=position,
                    state=state,
                    surface=entry.surface,
                    label=entry.analysis,
                    goes_to=_goes_to(entry),
                    outcome="",
                    leads_anywhere=False,
                )
            )
            after = position + len(entry.surface)
            reached = walk(after, entry.continuation, depth + 1)
            if entry.leads_to_end:
                outcome = (
                    "the word ends here"
                    if reached
                    else "this would end the word, and the word does not end here"
                )
            elif reached:
                outcome = f"carries on in the {entry.continuation}"
            else:
                outcome = f"nothing in the {entry.continuation} finishes the word"
            steps[index] = steps[index].model_copy(
                update={"outcome": outcome, "leads_anywhere": reached}
            )
            anywhere = anywhere or reached
        return anywhere

    walk(0, analyzer.root, 0)
    return steps


def _states_by_character(analyses: Analyses, root: str) -> list[CharacterState]:
    """Each character of the word paired with the state it was read in."""
    if not analyses.is_analysable:
        return []

    rows: list[CharacterState] = []
    position = 0
    state = root
    for morph_index, entry in enumerate(analyses.shortest):
        for offset, character in enumerate(entry.surface):
            rows.append(
                CharacterState(
                    character=character,
                    at=position + offset,
                    state=state,
                    morph_index=morph_index,
                )
            )
        position += len(entry.surface)
        state = entry.continuation if not entry.leads_to_end else END_OF_WORD
    return rows


def _word_view(analyzer: FiniteStateAnalyzer, word: str, label: str) -> WordView:
    """One word read by one grammar, with the walk that read it."""
    analyses = analyzer.analyse(word)
    trace = _trace(analyzer, word)
    return WordView(
        word=word,
        grammar_label=label,
        is_readable=analyses.is_analysable,
        n_analyses=analyses.n_analyses,
        analyses=[_analysis_view(analysis) for analysis in analyses],
        trace=trace,
        n_lines_tried=len(trace),
        states_by_character=_states_by_character(analyses, analyzer.root),
        cut=[piece.text for piece in analyzer.segment(word)],
    )


def _grammar_view(key: GrammarKey) -> GrammarView:
    """One grammar written out and measured."""
    analyzer = GRAMMARS[key]
    accepted = _accepted(analyzer)
    morphs = [analyzer.analyse(form).shortest.n_morphemes for form in sorted(accepted)]
    return GrammarView(
        key=key.value,
        label=GRAMMAR_LABELS[key],
        n_states=analyzer.n_lexicons,
        n_written_lines=_written_lines(analyzer),
        n_arcs=_arcs(analyzer),
        n_forms=len(accepted),
        n_paths=sum(accepted.values()),
        n_ambiguous_forms=sum(1 for count in accepted.values() if count > 1),
        mean_morphs=sum(morphs) / len(morphs),
        states=_states(analyzer),
        sample_forms=sorted(accepted)[:24],
    )


LEARNED_BUDGET_MULTIPLES: tuple[int, ...] = (1, 2)
"""The learned vocabularies the page fits: the grammar's size, and twice it."""

COMPARED_WORDS: tuple[str, ...] = (
    "walked",
    "walking",
    "recovers",
    "unnamed",
    "bakes",
    "measured",
)
"""The words the page shows cut both ways."""


@cache
def _cost() -> CostResponse:
    """The grammar's size against a learned vocabulary of the same size."""
    accepted = _accepted(LARGER)
    forms = sorted(accepted)
    lines = _written_lines(LARGER)
    morphs = [LARGER.analyse(form).shortest.n_morphemes for form in forms]
    corpus = [" ".join(forms)]

    budgets: list[Budget] = []
    for multiple in LEARNED_BUDGET_MULTIPLES:
        size = lines * multiple
        learned = BytePairEncoding(
            byte_fallback=False, vocabulary_size=size, minimum_pair_frequency=1
        ).fit(corpus)
        counts = [learned.encode(form).n_tokens for form in forms]
        cuts: list[LearnedCut] = []
        for word in COMPARED_WORDS:
            pieces = [
                piece.removesuffix("</w>") for piece in learned.encode(word).texts
            ]
            written = list(LARGER.analyse(word).shortest.morphs)
            cuts.append(
                LearnedCut(
                    word=word,
                    learned=pieces,
                    written=written,
                    agree=pieces == written,
                )
            )
        budgets.append(
            Budget(
                n_pieces=learned.vocabulary.n_tokens,
                mean_pieces=sum(counts) / len(counts),
                n_whole_words=sum(1 for count in counts if count == 1),
                cuts=cuts,
            )
        )

    return CostResponse(
        n_written_lines=lines,
        n_states=LARGER.n_lexicons,
        n_forms=len(forms),
        n_paths=sum(accepted.values()),
        mean_morphs=sum(morphs) / len(morphs),
        max_morphs=max(morphs),
        budgets=budgets,
    )


def _cleaned(text: str) -> list[str]:
    """A text as the lower-case words a morphology could be handed.

    Whitespace finds the runs, the punctuation that ends or opens a run is
    taken off, a hyphen is treated as a join between two words, and what is
    left is lower-cased, because a grammar that lists ``walk`` does not list
    ``Walk``. Every one of those is a decision made outside the grammar and
    each of them changes the coverage figure.
    """
    return [
        piece
        for word in _WHITESPACE.split(text)
        for piece in word.text.strip(_PUNCTUATION).lower().split("-")
        if piece
    ]


def _coverage(
    analyzer: FiniteStateAnalyzer, text: str, text_label: str, grammar_label: str
) -> CoverageView:
    """What share of a text one grammar reads, by occurrence and by distinct word."""
    tokens = _cleaned(text)
    types = sorted(set(tokens))
    readable = {word for word in types if analyzer.analyse(word).is_analysable}
    readable_tokens = [word for word in tokens if word in readable]
    return CoverageView(
        text_label=text_label,
        grammar_label=grammar_label,
        n_written_lines=_written_lines(analyzer),
        n_tokens=len(tokens),
        n_readable_tokens=len(readable_tokens),
        token_share=len(readable_tokens) / len(tokens),
        n_types=len(types),
        n_readable_types=len(readable),
        type_share=len(readable) / len(types),
        unreadable_types=[word for word in types if word not in readable],
    )


def _splitter_view(rule: str, pieces: list[str]) -> SplitterView:
    """What one way of finding words hands the grammar, and how much of it reads."""
    readings = [
        Reading(
            text=piece,
            is_readable=LARGER.analyse(piece.lower()).is_analysable
            if piece.strip()
            else False,
            cut=[part.text for part in LARGER.segment(piece.lower())]
            if piece.strip()
            else [piece],
        )
        for piece in pieces
    ]
    return SplitterView(
        rule=rule,
        pieces=readings,
        n_readable=sum(1 for reading in readings if reading.is_readable),
    )


def _added_lines(added: list[AddedLine], stem_state: str) -> list[LexiconEntry]:
    """The reader's lines, in the form the grammar takes them.

    A verb ending in ``e`` becomes three lines rather than one, since the
    spelling change has to be written out; that is the point of the page's
    section on spelling and is worth a reader seeing happen. ``stem_state`` is
    where a prefix leads, which differs between the two grammars because only
    the larger one keeps its stems in a state of their own.
    """
    lines: list[LexiconEntry] = []
    for line in added:
        surface = line.surface
        if line.kind is EntryKind.VERB:
            lines.append(LexiconEntry(surface, f"{surface}+V", VERB_ENDING))
            lines.append(LexiconEntry(surface, f"{surface}+V", END))
        elif line.kind is EntryKind.SILENT_E_VERB:
            lines.append(LexiconEntry(surface, f"{surface}+V", CONSONANT_ENDING))
            lines.append(LexiconEntry(surface, f"{surface}+V", END))
            lines.append(LexiconEntry(surface[:-1], f"{surface}+V", VOWEL_ENDING))
        elif line.kind is EntryKind.NOUN:
            lines.append(LexiconEntry(surface, f"{surface}+N", NOUN_ENDING))
            lines.append(LexiconEntry(surface, f"{surface}+N", END))
        elif line.kind is EntryKind.PREFIX:
            lines.append(LexiconEntry(surface, "+PREFIX", stem_state))
        else:
            lines.append(LexiconEntry(surface, f"{surface}+FUNC", END))
    return lines


# --------------------------------------------------------------------------
# The endpoints.
# --------------------------------------------------------------------------


@cache
def _machine() -> MachineResponse:
    """The three grammars written out, computed once because they take no input."""
    return MachineResponse(
        grammars=[
            _grammar_view(GrammarKey.SMALL),
            _grammar_view(GrammarKey.SMALL_BEFORE),
            _grammar_view(GrammarKey.LARGER),
        ]
    )


@router.get("/machine", response_model=MachineResponse)
def machine() -> MachineResponse:
    """Every state and every line of the page's grammars, with the language each accepts."""
    return _machine()


@cache
def _scenarios() -> ScenariosResponse:
    """The fixed words the page works through, computed once."""
    small = GRAMMAR_LABELS[GrammarKey.SMALL]
    before = GRAMMAR_LABELS[GrammarKey.SMALL_BEFORE]
    larger = GRAMMAR_LABELS[GrammarKey.LARGER]
    return ScenariosResponse(
        walked=_word_view(SMALL, "walked", small),
        walks=_word_view(SMALL, "walks", small),
        bakes=_word_view(SMALL, "bakes", small),
        baked_before=_word_view(SMALL_BEFORE, "baked", before),
        baked_after=_word_view(SMALL, "baked", small),
        bakeed_before=_word_view(SMALL_BEFORE, "bakeed", before),
        bakeed_after=_word_view(SMALL, "bakeed", small),
        recovers=_word_view(LARGER, "recovers", larger),
        reopeners=_word_view(LARGER, "reopeners", larger),
        unreadable=[
            _word_view(LARGER, word, larger)
            for word in ("alvarez", "analysis", "busiest", "thinned")
        ],
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    """The words the page walks through, each with the search that read it."""
    return _scenarios()


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """What the written lines buy, against a learned vocabulary of the same size."""
    return _cost()


GROWTH_LENGTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12)
"""The word lengths the growth table reports, which is enough to see the shape."""


@cache
def _ambiguity() -> AmbiguityResponse:
    """The larger grammar's own ambiguity, and how fast readings can multiply."""
    accepted = _accepted(LARGER)
    ambiguous = [form for form, count in accepted.items() if count > 1]
    label = GRAMMAR_LABELS[GrammarKey.LARGER]
    most = sorted(ambiguous, key=lambda form: (-accepted[form], form))[:3]
    return AmbiguityResponse(
        n_forms=len(accepted),
        n_ambiguous_forms=len(ambiguous),
        share_ambiguous=len(ambiguous) / len(accepted),
        most_read_words=[_word_view(LARGER, form, label) for form in most],
        growth=[
            Growth(length=length, n_readings=REPEATING.analyse("a" * length).n_analyses)
            for length in GROWTH_LENGTHS
        ],
    )


@router.get("/ambiguity", response_model=AmbiguityResponse)
def ambiguity() -> AmbiguityResponse:
    """Where a grammar reads a word more than one way, and how fast that grows."""
    return _ambiguity()


@cache
def _coverage_response() -> CoverageResponse:
    """Coverage on the section's sentence and on a paragraph of plain prose."""
    larger_label = GRAMMAR_LABELS[GrammarKey.LARGER]
    with_function = "the same grammar, with fifty structureless words listed"
    return CoverageResponse(
        sentence=RUNNING_SENTENCE,
        splitters=[
            _splitter_view(
                "a run of characters with a space either side",
                [word.text for word in _WHITESPACE.split(RUNNING_SENTENCE)],
            ),
            _splitter_view(
                "the annotation rules for English",
                [word.text for word in _TREEBANK.split(RUNNING_SENTENCE)],
            ),
        ],
        paragraph=PARAGRAPH,
        coverages=[
            _coverage(LARGER, RUNNING_SENTENCE, "our sentence", larger_label),
            _coverage(LARGER, PARAGRAPH, "a paragraph of plain prose", larger_label),
            _coverage(
                LARGER_WITH_FUNCTION_WORDS,
                PARAGRAPH,
                "a paragraph of plain prose",
                with_function,
            ),
        ],
        function_words=list(FUNCTION_WORDS),
    )


@router.get("/coverage", response_model=CoverageResponse)
def coverage() -> CoverageResponse:
    """What share of two texts the larger grammar can read at all."""
    return _coverage_response()


def _with_added(grammar: GrammarKey, added: list[LexiconEntry]) -> FiniteStateAnalyzer:
    """The chosen grammar with the reader's lines in its starting state."""
    if not added:
        return GRAMMARS[grammar]
    if grammar is GrammarKey.LARGER:
        return _larger(added)

    base = GRAMMARS[grammar]
    return FiniteStateAnalyzer(
        root=base.root,
        lexicons=tuple(
            Lexicon(lexicon.name, list(lexicon) + added)
            if lexicon.name == base.root
            else lexicon
            for lexicon in base.lexicons
        ),
    )


@router.post("/read", response_model=ReadResponse)
def read(request: ReadRequest) -> ReadResponse:
    """Read one word with the chosen grammar, plus whatever lines were added to it."""
    stem_state = STEM if request.grammar is GrammarKey.LARGER else WORD
    try:
        added = _added_lines(request.added, stem_state)
        analyzer = _with_added(request.grammar, added)
        view = _word_view(analyzer, request.word, GRAMMAR_LABELS[request.grammar])
    except (MLLibError, ValueError) as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    return ReadResponse(
        view=view,
        n_written_lines=_written_lines(analyzer),
        added_lines=[_line(entry) for entry in added],
    )
