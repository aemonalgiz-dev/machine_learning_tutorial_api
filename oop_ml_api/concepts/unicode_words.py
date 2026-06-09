"""Unicode Word Boundaries.

Every endpoint here is registered under ``/concepts/unicode-word-boundaries``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about UnicodeWordPreTokenizer.

What this module adds to the library
------------------------------------
Nothing about the rules themselves. Every boundary, every segment and every
character's class comes out of the library's ``UnicodeWordPreTokenizer`` and its
``word_break_class_of``, and the contrast throughout is the library's
``WhitespacePreTokenizer`` on the same text. What is assembled here is the
bookkeeping a page wants and a splitter has no reason to carry: how much of a
text ended up inside a word under each rule, what the segments look like before
the non-words are dropped, what one mark between two letters does against what
the same mark between two digits does, how often each class turns up in six
sentences of English, and the same six sentences read by both rules.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings. In particular the annex's own property
values arrive here as snake_case enum members and leave as plain descriptions,
through ``CLASS_DESCRIPTIONS``; a test asserts that no identifier survives into
any string any endpoint returns.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from typing import Annotated

from fastapi import APIRouter
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
    WordBreakClass,
    word_break_class_of,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/unicode-word-boundaries", tags=["unicode-word-boundaries"]
)

BOUNDARY_RULE = UnicodeWordPreTokenizer()
"""The subject of the page, keeping only the segments that hold a letter or a digit."""

KEEPING_PUNCTUATION = UnicodeWordPreTokenizer(keep_punctuation=True)
"""The same rules, keeping every segment that does not begin with white space."""

SPACE_RULE = WhitespacePreTokenizer()
"""The rule the page is measured against, throughout."""

NOTEBOOK: tuple[str, ...] = (
    "Dr. Alvarez didn't expect the low-cost re-analysis.",
    "The re-analysis used the low-cost sensors from the store room.",
    "Alvarez had run the first analysis in 2019, before the sensors arrived.",
    "Her notes say the low-cost readings drifted every afternoon.",
    "The drift was small, though it was larger than the effect she expected.",
    "She re-ran the analysis twice and the drift did not go away.",
)
"""The six sentences the tokenization pages share, so counts agree between them."""

RUNNING_SENTENCE: str = NOTEBOOK[0]
"""The sentence the page carries from beginning to end."""

SHORT_TEXT = "Dr. Alvarez didn't"
"""Eighteen characters, small enough to show one class per character."""

MAX_TEXTS = 6
"""How many texts one request may split. The widgets ask for one at a time."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]

CLASS_DESCRIPTIONS: dict[WordBreakClass, str] = {
    WordBreakClass.CARRIAGE_RETURN: "the end of a line",
    WordBreakClass.LINE_FEED: "the end of a line",
    WordBreakClass.NEWLINE: "the end of a line",
    WordBreakClass.EXTEND: "belongs to the character before it",
    WordBreakClass.FORMAT: "invisible formatting",
    WordBreakClass.KATAKANA: "katakana",
    WordBreakClass.HEBREW_LETTER: "a Hebrew letter",
    WordBreakClass.ALPHABETIC_LETTER: "a letter",
    WordBreakClass.SINGLE_QUOTE: "an apostrophe",
    WordBreakClass.DOUBLE_QUOTE: "a quotation mark",
    WordBreakClass.MID_LETTER: "may stand between two letters",
    WordBreakClass.MID_NUMBER: "may stand between two digits",
    WordBreakClass.MID_NUMBER_LETTER: "may stand between two letters or two digits",
    WordBreakClass.NUMERIC: "a digit",
    WordBreakClass.EXTEND_NUMBER_LETTER: "joins whatever it finds on both sides",
    WordBreakClass.REGIONAL_INDICATOR: "half of a flag",
    WordBreakClass.WORD_SEGMENT_SPACE: "a space",
    WordBreakClass.OTHER: "none of the above",
}
"""Each class of the standard's table, said in words a reader has already met.

The library spells these as enum members and the standard spells them as short
property values, and neither spelling belongs on a page. The description is what
the class *does* in the rules, which is the only thing the page asks of it.
"""


class MarkCandidate:
    """One character, its published name, and the two texts that surround it.

    A plain class rather than a response model because it is input to the probe
    and not output from it. Only the name and the code point are sent, since
    several of these would arrive at a reader as a box.
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
    def between_letters(self) -> str:
        """The character with one letter either side of it."""
        return f"a{self.character}b"

    @property
    def between_digits(self) -> str:
        """The character with one digit either side of it."""
        return f"1{self.character}2"


MARK_CANDIDATES: tuple[MarkCandidate, ...] = (
    MarkCandidate(0x0027, "apostrophe"),
    MarkCandidate(0x2019, "right single quotation mark"),
    MarkCandidate(0x002E, "full stop"),
    MarkCandidate(0x003A, "colon"),
    MarkCandidate(0x00B7, "middle dot"),
    MarkCandidate(0x002C, "comma"),
    MarkCandidate(0x005F, "low line"),
    MarkCandidate(0x202F, "narrow no-break space"),
    MarkCandidate(0x002D, "hyphen-minus"),
    MarkCandidate(0x0022, "quotation mark"),
    MarkCandidate(0x002F, "solidus"),
    MarkCandidate(0x0020, "space"),
)
"""Twelve marks put between two letters and then between two digits.

Chosen so the two columns disagree. The colon holds two letters together and
separates two digits; the comma does the reverse; the low line and the narrow
no-break space hold both pairs together, one of which is a space by every other
measure the site has used.
"""


class ScriptCase:
    """One text, the label the page gives it, and what to look at in the answer."""

    __slots__ = ("label", "look_at", "text")

    def __init__(self, label: str, text: str, look_at: str) -> None:
        self.label = label
        self.text = text
        self.look_at = look_at


SCRIPT_CASES: tuple[ScriptCase, ...] = (
    ScriptCase(
        "a Japanese sentence, mixing three scripts",
        "アルバレス博士は驚いた",
        "the katakana run at the front is one word and everything after it is one word per character",
    ),
    ScriptCase(
        "nine characters of Chinese",
        "我来到北京清华大学",
        "nine words of one character each, where a reader of Chinese sees four",
    ),
    ScriptCase(
        "seven characters of Thai",
        "ภาษาไทย",
        "one word per character again, and the standard says in as many words that it will not do better here",
    ),
    ScriptCase(
        "Korean, which is written with spaces",
        "한국어 단어",
        "two words, the same two the space rule finds, because Korean syllables are letters to the rules",
    ),
    ScriptCase(
        "a French thousands group",
        "1 000 sensors",
        "the space inside the number does not end the word, though every rule that reads white space says it does",
    ),
    ScriptCase(
        "two flags",
        "\U0001f1eb\U0001f1f7\U0001f1e9\U0001f1ea",
        "two segments, one per flag, and neither is a word because neither holds a letter",
    ),
    ScriptCase(
        "an address",
        "alvarez@example.com",
        "two words, and the character between them is gone",
    ),
    ScriptCase(
        "a file name",
        "re-analysis_2019.csv",
        "three words, though the low line inside the middle one held it together",
    ),
    ScriptCase(
        "a single approving hand",
        "\U0001f44d",
        "no words at all, because there is no letter and no digit anywhere in it",
    ),
)
"""Writing that separates the two rules, in both directions.

The first three are the case the standard was written for and the case it
declines; the fourth is the script it deliberately does not decline. The fifth
and sixth are marks a space rule cannot read. The last three are where a space
rule gives the better answer, which the page reports rather than hides.
"""


class Piece(BaseModel):
    """One piece of a text, and the half-open span of the source it came from."""

    text: str
    start: int
    end: int


class Segment(BaseModel):
    """One stretch between two boundaries, whether or not it is a word."""

    text: str
    start: int
    end: int
    is_word: bool


class CharacterClass(BaseModel):
    """One character, the class the rules give it, and whether a word ended before it."""

    character: str
    code_point: str
    description: str
    breaks_before: bool


class RuleView(BaseModel):
    """One rule's reading of one text, and the two things the page asks of it.

    ``glued`` is the words joined with one space each, which is the only glue
    available to anything holding the words alone, and ``glue_is_exact`` says
    whether that reproduced the text.
    """

    pieces: list[Piece]
    n_pieces: int
    n_characters_in_pieces: int
    share_covered: float
    glued: str
    glue_is_exact: bool


class SplitView(BaseModel):
    """One text read by the boundary rules, with the space rule beside it."""

    source: str
    n_characters: int
    characters: list[CharacterClass]
    segments: list[Segment]
    n_segments: int
    segments_tile_the_text: bool
    on_boundaries: RuleView
    on_spaces: RuleView
    keeping_punctuation: list[str]
    n_keeping_punctuation: int


class SplitRequest(BaseModel):
    """Texts to read. The rules take no settings, so there is nothing else to send."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)

    model_config = {"extra": "forbid"}


class SplitResponse(BaseModel):
    """Every text of the request, seen through both rules."""

    texts: list[SplitView]


def _rule_view(text: str, rule: PreTokenizer) -> RuleView:
    """How one rule reads one text, and what gluing its words back gives."""
    words = rule.split(text)
    covered = sum(word.end - word.start for word in words)
    glued = " ".join(words.texts)

    return RuleView(
        pieces=[
            Piece(text=word.text, start=word.start, end=word.end) for word in words
        ],
        n_pieces=words.n_words,
        n_characters_in_pieces=covered,
        share_covered=covered / len(text) if text else 0.0,
        glued=glued,
        glue_is_exact=glued == text,
    )


def _split_view(text: str) -> SplitView:
    """One text taken all the way through: classes, segments, words, and the contrast."""
    segments = BOUNDARY_RULE.segments_of(text)
    words = BOUNDARY_RULE.split(text)
    word_spans = {(word.start, word.end) for word in words}
    starts = {segment.start for segment in segments}

    characters = [
        CharacterClass(
            character=character,
            code_point=f"U+{ord(character):04X}",
            description=CLASS_DESCRIPTIONS[word_break_class_of(character)],
            breaks_before=position in starts and position > 0,
        )
        for position, character in enumerate(text)
    ]

    return SplitView(
        source=text,
        n_characters=len(text),
        characters=characters,
        segments=[
            Segment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                is_word=(segment.start, segment.end) in word_spans,
            )
            for segment in segments
        ],
        n_segments=segments.n_words,
        segments_tile_the_text="".join(segments.texts) == text,
        on_boundaries=_rule_view(text, BOUNDARY_RULE),
        on_spaces=_rule_view(text, SPACE_RULE),
        keeping_punctuation=list(KEEPING_PUNCTUATION.split(text).texts),
        n_keeping_punctuation=KEEPING_PUNCTUATION.split(text).n_words,
    )


@router.post("/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    """Read every text of the request by both rules and report everything between."""
    return SplitResponse(texts=[_split_view(text) for text in request.texts])


class MarkProbe(BaseModel):
    """One mark put between two letters and then between two digits."""

    name: str
    code_point: str
    description: str
    between_letters: list[str]
    joins_letters: bool
    between_digits: list[str]
    joins_digits: bool


class ClassCount(BaseModel):
    """One class, and how many characters of the six sentences fell into it."""

    description: str
    count: int


class ClassesResponse(BaseModel):
    """The table of classes, counted over the six sentences, and the marks probe.

    ``n_classes`` is how many classes the table holds in all and ``n_classes_used``
    how many of them six sentences of English reach, so the difference is how much
    of the table exists for writing this corpus does not contain.
    """

    n_classes: int
    n_classes_used: int
    census: list[ClassCount]
    n_characters: int
    marks: list[MarkProbe]
    n_marks_joining_letters: int
    n_marks_joining_digits: int
    n_marks_disagreeing: int


@cache
def _classes() -> ClassesResponse:
    """The class census and the marks probe, computed once because neither takes input."""
    tally: Counter[WordBreakClass] = Counter()
    for text in NOTEBOOK:
        for character in text:
            tally[word_break_class_of(character)] += 1

    census = [
        ClassCount(description=CLASS_DESCRIPTIONS[found], count=count)
        for found, count in tally.most_common()
    ]

    marks: list[MarkProbe] = []
    for candidate in MARK_CANDIDATES:
        letters = BOUNDARY_RULE.split(candidate.between_letters)
        digits = BOUNDARY_RULE.split(candidate.between_digits)
        marks.append(
            MarkProbe(
                name=candidate.name,
                code_point=candidate.code_point,
                description=CLASS_DESCRIPTIONS[
                    word_break_class_of(candidate.character)
                ],
                between_letters=list(letters.texts),
                joins_letters=letters.n_words == 1,
                between_digits=list(digits.texts),
                joins_digits=digits.n_words == 1,
            )
        )

    return ClassesResponse(
        n_classes=len(WordBreakClass),
        n_classes_used=len(tally),
        census=census,
        n_characters=sum(len(text) for text in NOTEBOOK),
        marks=marks,
        n_marks_joining_letters=sum(1 for probe in marks if probe.joins_letters),
        n_marks_joining_digits=sum(1 for probe in marks if probe.joins_digits),
        n_marks_disagreeing=sum(
            1 for probe in marks if probe.joins_letters != probe.joins_digits
        ),
    )


@router.get("/classes", response_model=ClassesResponse)
def classes() -> ClassesResponse:
    """How often each class turns up, and what one mark does on each side of it."""
    return _classes()


class ScriptProbe(BaseModel):
    """One text read by both rules, with the difference the page points at."""

    label: str
    text: str
    n_characters: int
    boundary_pieces: list[str]
    n_boundary_pieces: int
    boundary_covered: int
    space_pieces: list[str]
    n_space_pieces: int
    space_covered: int
    look_at: str


class ScriptsResponse(BaseModel):
    """The writing that separates the two rules, in both directions."""

    probes: list[ScriptProbe]


@cache
def _scripts() -> ScriptsResponse:
    """The fixed script probes, computed once because none of them takes an input."""
    probes: list[ScriptProbe] = []
    for case in SCRIPT_CASES:
        on_boundaries = _rule_view(case.text, BOUNDARY_RULE)
        on_spaces = _rule_view(case.text, SPACE_RULE)
        probes.append(
            ScriptProbe(
                label=case.label,
                text=case.text,
                n_characters=len(case.text),
                boundary_pieces=[piece.text for piece in on_boundaries.pieces],
                n_boundary_pieces=on_boundaries.n_pieces,
                boundary_covered=on_boundaries.n_characters_in_pieces,
                space_pieces=[piece.text for piece in on_spaces.pieces],
                n_space_pieces=on_spaces.n_pieces,
                space_covered=on_spaces.n_characters_in_pieces,
                look_at=case.look_at,
            )
        )
    return ScriptsResponse(probes=probes)


@router.get("/scripts", response_model=ScriptsResponse)
def scripts() -> ScriptsResponse:
    """Writing where the two rules part company, in both directions."""
    return _scripts()


class CorpusCost(BaseModel):
    """What one rule spends over the six sentences.

    ``n_distinct`` is how many different words the six sentences produced, so it
    is how many entries a table over them would need, and ``n_pieces`` is how
    many words they produced in all, so it is what a model reading them pays per
    corpus. ``n_characters_covered`` is how many of the 375 characters ended up
    inside a word.
    """

    n_distinct: int
    n_pieces: int
    n_characters_covered: int
    share_covered: float
    running_pieces: list[str]
    running_covered: int
    running_glued: str
    running_glue_is_exact: bool


class CostResponse(BaseModel):
    """The two rules over the same six sentences, and what the difference is made of."""

    corpus: list[str]
    n_characters: int
    on_boundaries: CorpusCost
    on_spaces: CorpusCost
    n_shared_entries: int
    only_on_boundaries: list[str]
    only_on_spaces: list[str]
    n_segments: int
    n_pieces_keeping_punctuation: int
    n_distinct_keeping_punctuation: int


def _corpus_cost(rule: PreTokenizer) -> CorpusCost:
    """How one rule reads the six sentences, and how it reads the first of them."""
    counts = Corpus.of(list(NOTEBOOK)).word_counts(rule)
    running = _rule_view(RUNNING_SENTENCE, rule)
    covered = sum(
        sum(word.end - word.start for word in rule.split(text)) for text in NOTEBOOK
    )
    characters = sum(len(text) for text in NOTEBOOK)

    return CorpusCost(
        n_distinct=len(list(counts)),
        n_pieces=sum(rule.split(text).n_words for text in NOTEBOOK),
        n_characters_covered=covered,
        share_covered=covered / characters,
        running_pieces=[piece.text for piece in running.pieces],
        running_covered=running.n_characters_in_pieces,
        running_glued=running.glued,
        running_glue_is_exact=running.glue_is_exact,
    )


@cache
def _cost() -> CostResponse:
    """The corpus comparison, computed once because it takes no input."""
    on_boundaries = sorted(
        count.word for count in Corpus.of(list(NOTEBOOK)).word_counts(BOUNDARY_RULE)
    )
    on_spaces = sorted(
        count.word for count in Corpus.of(list(NOTEBOOK)).word_counts(SPACE_RULE)
    )
    keeping = Corpus.of(list(NOTEBOOK)).word_counts(KEEPING_PUNCTUATION)

    return CostResponse(
        corpus=list(NOTEBOOK),
        n_characters=sum(len(text) for text in NOTEBOOK),
        on_boundaries=_corpus_cost(BOUNDARY_RULE),
        on_spaces=_corpus_cost(SPACE_RULE),
        n_shared_entries=len(set(on_boundaries) & set(on_spaces)),
        only_on_boundaries=sorted(set(on_boundaries) - set(on_spaces)),
        only_on_spaces=sorted(set(on_spaces) - set(on_boundaries)),
        n_segments=sum(BOUNDARY_RULE.segments_of(text).n_words for text in NOTEBOOK),
        n_pieces_keeping_punctuation=sum(
            KEEPING_PUNCTUATION.split(text).n_words for text in NOTEBOOK
        ),
        n_distinct_keeping_punctuation=len(list(keeping)),
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """The six sentences read by both rules, and what each spends on the other's work."""
    return _cost()
