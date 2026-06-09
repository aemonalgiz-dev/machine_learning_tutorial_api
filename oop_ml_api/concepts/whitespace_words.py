"""Splitting on Spaces.

Every endpoint here is registered under ``/concepts/splitting-on-spaces``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about WhitespacePreTokenizer.

What this module adds to the library
------------------------------------
Almost nothing, which is the point of the page. The splitter is the library's
own ``WhitespacePreTokenizer`` and every piece and every span comes straight out
of it. What is assembled here is the bookkeeping a page wants and a splitter has
no reason to carry: how much of a text the spans cover, what lies in the gaps
between them, what comes back when the pieces are glued with single spaces
against what comes back when the spans are used instead, and the same corpus put
to the Unicode word boundary rule so the two can be compared on the same
sentences.

The comparison rule is the library's ``UnicodeWordPreTokenizer``, imported for
one contrast only. The page is about splitting on spaces and says so.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The only free text returned is the labels written here and
the words of the corpus itself, and a test asserts that no identifier survives
into any of them.
"""

from __future__ import annotations

import string
from functools import cache
from typing import Annotated

from fastapi import APIRouter
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/splitting-on-spaces", tags=["splitting-on-spaces"])

SPLITTER = WhitespacePreTokenizer()
"""One splitter, since it holds no state and every call would build the same one."""

BOUNDARY_RULE = UnicodeWordPreTokenizer()
"""The rule the page contrasts this one with, once, in the section on cost."""

NOTEBOOK: tuple[str, ...] = (
    "Dr. Alvarez didn't expect the low-cost re-analysis.",
    "The re-analysis used the low-cost sensors from the store room.",
    "Alvarez had run the first analysis in 2019, before the sensors arrived.",
    "Her notes say the low-cost readings drifted every afternoon.",
    "The drift was small, though it was larger than the effect she expected.",
    "She re-ran the analysis twice and the drift did not go away.",
)
"""The six sentences the tokenization pages share, so counts agree between them.

The first is the running sentence of the whole section, chosen because it breaks
things: a full stop that ends an abbreviation, a contraction, a hyphenated
compound and a prefix.
"""

RUNNING_SENTENCE: str = NOTEBOOK[0]
"""The sentence the page carries from beginning to end."""

ODD_SPACING = "Dr.  Alvarez didn't expect\nthe low-cost re-analysis."
"""The running sentence with a doubled space and a line break in place of a space."""

MAX_TEXTS = 6
"""How many texts one request may split. The widgets ask for at most three."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]

EDGE_PUNCTUATION = string.punctuation
"""The marks stripped from the ends of a piece when counting duplicated entries."""


class SeparatorCandidate:
    """One character, its official name, and the text that puts it between two letters.

    A plain class rather than a response model because it is input to the probe
    rather than output from it, and because the character itself never leaves
    this module. Only its name and its code point are sent, since several of
    these characters would arrive at a reader as an empty box.
    """

    __slots__ = ("code", "name")

    def __init__(self, code: int, name: str) -> None:
        self.code = code
        self.name = name

    @property
    def character(self) -> str:
        """The character itself, built from its number so the source stays readable."""
        return chr(self.code)

    @property
    def code_point(self) -> str:
        """The character written the way a character table writes it."""
        return f"U+{self.code:04X}"

    @property
    def probe(self) -> str:
        """The character with one letter either side of it."""
        return f"a{self.character}b"


SEPARATOR_CANDIDATES: tuple[SeparatorCandidate, ...] = (
    SeparatorCandidate(0x0020, "space"),
    SeparatorCandidate(0x0009, "character tabulation"),
    SeparatorCandidate(0x000A, "line feed"),
    SeparatorCandidate(0x00A0, "no-break space"),
    SeparatorCandidate(0x2009, "thin space"),
    SeparatorCandidate(0x3000, "ideographic space"),
    SeparatorCandidate(0x2029, "paragraph separator"),
    SeparatorCandidate(0x200B, "zero width space"),
    SeparatorCandidate(0x00AD, "soft hyphen"),
    SeparatorCandidate(0x002D, "hyphen-minus"),
)
"""Ten characters put between two letters, seven of which end a word and three of which do not.

Chosen so the last three are the interesting ones. One carries the word space in
its name and ends no word, and one exists to offer a line break and ends none
either, because the rule asks whether a character *is* white space and never
what it is called or what it is for.
"""


class Breakage:
    """One short text and the label the page gives the difficulty it shows."""

    __slots__ = ("label", "look_at", "text")

    def __init__(self, label: str, text: str, look_at: str) -> None:
        self.label = label
        self.text = text
        self.look_at = look_at


BREAKAGES: tuple[Breakage, ...] = (
    Breakage(
        "a full stop that closes a sentence",
        "The drift did not go away.",
        "away.",
    ),
    Breakage(
        "a comma inside a sentence",
        "The drift was small, though it was larger.",
        "small,",
    ),
    Breakage(
        "quotation marks and brackets",
        'She wrote "low-cost" (twice) in her notes.',
        '"low-cost"',
    ),
    Breakage(
        "a full stop that closes nothing",
        "Dr. Alvarez ran the re-analysis.",
        "Dr.",
    ),
    Breakage(
        "an apostrophe standing for two missing letters",
        "Alvarez didn't expect the drift.",
        "didn't",
    ),
    Breakage(
        "a hyphen joining two words into one",
        "the low-cost re-analysis",
        "low-cost",
    ),
    Breakage(
        "a script written without spaces",
        "我来到北京清华大学",
        "我来到北京清华大学",
    ),
)
"""The kinds of writing the rule handles by doing nothing, and the piece to look at.

Six of them are English and the seventh is not, because the last one is the
failure the next section of the site exists for and it cannot be shown in a
script that writes spaces.
"""


class Piece(BaseModel):
    """One piece of a text, and the half-open span of the source it came from."""

    text: str
    start: int
    end: int


class SplitView(BaseModel):
    """One text put to the rule, and everything the page asks about the answer.

    ``glued`` is the pieces joined with one space each, which is the only glue
    available to anything holding the pieces alone. ``rebuilt`` uses the spans
    instead, taking the source back between one piece and the next, so it is
    exact whenever the pieces themselves were cut rather than rewritten.
    """

    source: str
    n_characters: int
    pieces: list[Piece]
    n_pieces: int
    n_characters_in_pieces: int
    share_covered: float
    gaps: list[str]
    glued: str
    glue_is_exact: bool
    rebuilt: str
    rebuild_is_exact: bool


class SplitRequest(BaseModel):
    """Texts to split. The rule takes no settings, so there is nothing else to send."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)

    model_config = {"extra": "forbid"}


class SplitResponse(BaseModel):
    """Every text of the request, seen through the rule."""

    texts: list[SplitView]


def _split_view(text: str, rule: PreTokenizer = SPLITTER) -> SplitView:
    """Split one text and record what the spans cover and what glue recovers."""
    words = rule.split(text)
    pieces = [Piece(text=word.text, start=word.start, end=word.end) for word in words]

    gaps: list[str] = []
    rebuilt = ""
    previous = 0
    for word in words:
        gaps.append(text[previous : word.start])
        rebuilt += text[previous : word.start] + word.text
        previous = word.end
    gaps.append(text[previous:])
    rebuilt += text[previous:]

    covered = sum(word.end - word.start for word in words)
    glued = " ".join(words.texts)

    return SplitView(
        source=text,
        n_characters=len(text),
        pieces=pieces,
        n_pieces=words.n_words,
        n_characters_in_pieces=covered,
        share_covered=covered / len(text) if text else 0.0,
        gaps=gaps,
        glued=glued,
        glue_is_exact=glued == text,
        rebuilt=rebuilt,
        rebuild_is_exact=rebuilt == text,
    )


@router.post("/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    """Split every text of the request and report the spans, the coverage and the glue."""
    return SplitResponse(texts=[_split_view(text) for text in request.texts])


class SeparatorProbe(BaseModel):
    """One character put between two letters, and whether a word ended there."""

    name: str
    code_point: str
    n_pieces: int
    ends_a_word: bool


class BreakageProbe(BaseModel):
    """One text the rule handles by doing nothing, and what came out.

    ``look_at`` is one of ``pieces``, named here so the page and the widget
    agree on which piece carries the difficulty rather than each choosing.
    """

    label: str
    text: str
    n_pieces: int
    pieces: list[str]
    look_at: str


class SurveyResponse(BaseModel):
    """What counts as a space, and the kinds of writing the rule leaves alone."""

    separators: list[SeparatorProbe]
    n_separators_that_end_a_word: int
    breakages: list[BreakageProbe]


@cache
def _survey() -> SurveyResponse:
    """The two fixed probes, computed once because neither takes an input."""
    separators = [
        SeparatorProbe(
            name=candidate.name,
            code_point=candidate.code_point,
            n_pieces=SPLITTER.split(candidate.probe).n_words,
            ends_a_word=SPLITTER.split(candidate.probe).n_words > 1,
        )
        for candidate in SEPARATOR_CANDIDATES
    ]

    breakages = [
        BreakageProbe(
            label=breakage.label,
            text=breakage.text,
            n_pieces=SPLITTER.split(breakage.text).n_words,
            pieces=list(SPLITTER.split(breakage.text).texts),
            look_at=breakage.look_at,
        )
        for breakage in BREAKAGES
    ]

    return SurveyResponse(
        separators=separators,
        n_separators_that_end_a_word=sum(
            1 for probe in separators if probe.ends_a_word
        ),
        breakages=breakages,
    )


@router.get("/survey", response_model=SurveyResponse)
def survey() -> SurveyResponse:
    """Which characters end a word, and the writing the rule cuts nowhere."""
    return _survey()


class RuleCost(BaseModel):
    """What one rule spends over the six sentences.

    ``n_distinct`` is how many different pieces the six sentences produced, so
    it is how many entries a table over them would need; ``n_pieces`` is how
    many pieces they produced in all, so it is what a model reading them pays
    per corpus. ``running_pieces`` is the running sentence alone, and
    ``running_covered`` is how many of its 51 characters ended up inside a
    piece.
    """

    n_distinct: int
    n_pieces: int
    running_pieces: list[str]
    running_covered: int
    running_glued: str
    running_glue_is_exact: bool


class CostResponse(BaseModel):
    """The two rules over the same six sentences, and what the attached marks cost.

    ``n_entries_with_edge_punctuation`` counts the entries of the
    space-delimited list that open or close on a punctuation mark, and
    ``n_distinct_after_stripping`` is what the list would come to if those marks
    were taken off, so the difference between the two is the number of entries
    that are a word this list already holds wearing a mark.
    """

    corpus: list[str]
    n_characters: int
    on_spaces: RuleCost
    on_word_boundaries: RuleCost
    n_shared_entries: int
    only_on_spaces: list[str]
    only_on_word_boundaries: list[str]
    n_entries_with_edge_punctuation: int
    entries_ending_in_a_full_stop: list[str]
    n_distinct_after_stripping: int
    duplicated_entries: list[str]


def _rule_cost(rule: PreTokenizer) -> RuleCost:
    """How one rule reads the six sentences, and how it reads the first of them."""
    counts = Corpus.of(list(NOTEBOOK)).word_counts(rule)
    running = _split_view(RUNNING_SENTENCE, rule)

    return RuleCost(
        n_distinct=len(list(counts)),
        n_pieces=sum(rule.split(text).n_words for text in NOTEBOOK),
        running_pieces=[piece.text for piece in running.pieces],
        running_covered=running.n_characters_in_pieces,
        running_glued=running.glued,
        running_glue_is_exact=running.glue_is_exact,
    )


@cache
def _cost() -> CostResponse:
    """The corpus comparison, computed once because it takes no input."""
    on_spaces = sorted(
        count.word for count in Corpus.of(list(NOTEBOOK)).word_counts(SPLITTER)
    )
    on_boundaries = sorted(
        count.word for count in Corpus.of(list(NOTEBOOK)).word_counts(BOUNDARY_RULE)
    )

    stripped = {word.strip(EDGE_PUNCTUATION) for word in on_spaces}
    duplicated = sorted(
        word
        for word in on_spaces
        if word.strip(EDGE_PUNCTUATION) != word
        and word.strip(EDGE_PUNCTUATION) in set(on_spaces)
    )

    return CostResponse(
        corpus=list(NOTEBOOK),
        n_characters=sum(len(text) for text in NOTEBOOK),
        on_spaces=_rule_cost(SPLITTER),
        on_word_boundaries=_rule_cost(BOUNDARY_RULE),
        n_shared_entries=len(set(on_spaces) & set(on_boundaries)),
        only_on_spaces=sorted(set(on_spaces) - set(on_boundaries)),
        only_on_word_boundaries=sorted(set(on_boundaries) - set(on_spaces)),
        n_entries_with_edge_punctuation=sum(
            1 for word in on_spaces if word.strip(EDGE_PUNCTUATION) != word
        ),
        entries_ending_in_a_full_stop=[
            word for word in on_spaces if word.endswith(".")
        ],
        n_distinct_after_stripping=len(stripped),
        duplicated_entries=duplicated,
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """The six sentences read by both rules, and what each spends on the other's work."""
    return _cost()
