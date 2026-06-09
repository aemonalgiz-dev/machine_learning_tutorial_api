"""The Pattern Language Models Use.

Every endpoint here is registered under
``/concepts/the-pattern-language-models-use``, and the router is created and
registered before any endpoint exists, so the pages of this section can be
built in parallel without many edits to one shared file.

The page is about PatternPreTokenizer, GPT2_PATTERN.

What this module adds to the library
------------------------------------
Nothing about the splitting. Every piece and every span comes out of the
library's ``PatternPreTokenizer``, driven by ``GPT2_PATTERN`` and by six other
patterns built from it by deleting one branch at a time, and the four rules it
is measured against are ``WhitespacePreTokenizer``,
``UnicodeWordPreTokenizer``, ``PennTreebankPreTokenizer`` and
``MosesPreTokenizer`` on the same texts, so the numbers line up with the four
sibling pages. What is assembled here is the bookkeeping a page wants and a
splitter has no reason to carry: which branch of the pattern claimed each
piece, whether the pieces concatenate back to the whole text including its
spacing, how many pieces carry a leading space and how many words the corpus
therefore spells twice, and what the published pattern's own character classes
do on texts where they part from the ones ``re`` can express.

The published classes are reconstructed here rather than imported, because
``re`` has no ``\\p{L}``: ``_category_ranges`` walks every codepoint once and
builds an explicit character class out of Unicode's own general categories, so
the published reading and the reading here are both run through the same
``PatternPreTokenizer`` and the only difference between them is the two classes.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings. A test asserts that no identifier
survives into any string any endpoint returns, with the patterns themselves and
the writing the rules were given excluded, since those are the notation the
page is about.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from functools import cache
from typing import Annotated, Literal

from fastapi import APIRouter
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.moses import (
    MosesPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.pattern import (
    GPT2_PATTERN,
    WORD_OR_PUNCTUATION_PATTERN,
    PatternPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(
    prefix="/concepts/the-pattern-language-models-use",
    tags=["the-pattern-language-models-use"],
)

SPACE_RULE = WhitespacePreTokenizer()
"""The crudest of the four earlier rules, so all five pages count over one corpus."""

BOUNDARY_RULE = UnicodeWordPreTokenizer()
"""The standard word boundary rules, at their default of words only."""

ANNOTATION_RULE = PennTreebankPreTokenizer()
"""The annotation rule list of the third sibling page."""

TRANSLATION_RULE = MosesPreTokenizer()
"""The translation rule list of the fourth sibling page."""

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

SPACED_SENTENCE = "Dr.  Alvarez  didn't expect"
"""The running sentence with two spaces where it had one, twice over.

Single spacing cannot show what the second spacing branch is for, since a lone
space is claimed by the word after it and that branch is never reached.
"""

WITHOUT_THE_CONTRACTIONS = r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
"""The pattern with its seven leading branches deleted."""

WITHOUT_THE_OPTIONAL_SPACE = r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d+|(?:[^\s\w]|_)+|\s+"
"""The pattern with the optional leading space deleted from all three runs."""

WITHOUT_THE_HELD_SPACE = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+"
)
"""The pattern with the branch that hands the last space to the next word deleted."""

RUNS_WITH_NO_SPACES = r"\S+"
"""Splitting on spaces, written as a pattern, which is what it always was."""

LETTERS_ONLY = r"[^\W\d_]+"
"""A pattern that names only the letters, so every other character falls out."""

PatternKey = Literal[
    "the-pattern",
    "without-the-contractions",
    "without-the-optional-space",
    "without-the-held-back-space",
    "words-and-marks",
    "runs-with-no-spaces",
    "letters-only",
]
"""The patterns a request may ask for, as a closed choice rather than free text.

A caller naming a pattern of its own would be a caller able to name one whose
matching cost grows exponentially with the length of the text, so the choice is
closed and every member of it is a variation this page is about.
"""


class PatternChoice:
    """One offered pattern, its source and what it says in ordinary words."""

    __slots__ = ("label", "note", "source")

    def __init__(self, source: str, label: str, note: str) -> None:
        self.source = source
        self.label = label
        self.note = note


PATTERNS: dict[str, PatternChoice] = {
    "the-pattern": PatternChoice(
        GPT2_PATTERN,
        "the pattern itself",
        "seven contractions, then letters, digits and marks each with an "
        "optional space in front, then the spacing",
    ),
    "without-the-contractions": PatternChoice(
        WITHOUT_THE_CONTRACTIONS,
        "with the seven contraction branches deleted",
        "an apostrophe is then just another mark, so a contraction breaks into "
        "three pieces instead of two",
    ),
    "without-the-optional-space": PatternChoice(
        WITHOUT_THE_OPTIONAL_SPACE,
        "with the optional space deleted from the three runs",
        "every space then stands alone as a piece of its own, which is the "
        "arrangement the leading space was chosen over",
    ),
    "without-the-held-back-space": PatternChoice(
        WITHOUT_THE_HELD_SPACE,
        "with the branch that holds back the last space deleted",
        "a run of spacing is then taken whole, so the word after it loses the "
        "leading space it would otherwise carry",
    ),
    "words-and-marks": PatternChoice(
        WORD_OR_PUNCTUATION_PATTERN,
        "a run of word characters, or a run of marks",
        "the rule a dozen tools ship under a dozen names, which keeps no "
        "spacing at all",
    ),
    "runs-with-no-spaces": PatternChoice(
        RUNS_WITH_NO_SPACES,
        "every run of characters that is not spacing",
        "splitting on spaces, written as a pattern",
    ),
    "letters-only": PatternChoice(
        LETTERS_ONLY,
        "every run of letters",
        "a pattern that names some characters and not others, so the rest fall "
        "out of the answer entirely",
    ),
}
"""The seven patterns the page offers, each a change to one part of the first."""

MAX_TEXTS = 4
"""How many texts one request may split. The widgets ask for one at a time."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


class Branch:
    """One alternative of the pattern, and what it says in ordinary words."""

    __slots__ = ("family", "reads", "source")

    def __init__(self, source: str, reads: str, family: str) -> None:
        self.source = source
        self.reads = reads
        self.family = family


BRANCHES: tuple[Branch, ...] = (
    Branch("'s", "an apostrophe and the letter s", "contractions"),
    Branch("'t", "an apostrophe and the letter t", "contractions"),
    Branch("'re", "an apostrophe and the letters r and e", "contractions"),
    Branch("'ve", "an apostrophe and the letters v and e", "contractions"),
    Branch("'m", "an apostrophe and the letter m", "contractions"),
    Branch("'ll", "an apostrophe and two letters l", "contractions"),
    Branch("'d", "an apostrophe and the letter d", "contractions"),
    Branch(
        r" ?[^\W\d_]+",
        "an optional space, then a run of letters",
        "runs",
    ),
    Branch(r" ?\d+", "an optional space, then a run of digits", "runs"),
    Branch(
        r" ?(?:[^\s\w]|_)+",
        "an optional space, then a run of marks, which is everything that is "
        "not spacing, a letter or a digit",
        "runs",
    ),
    Branch(
        r"\s+(?!\S)",
        "a run of spacing that does not reach the next piece of writing, which "
        "leaves its last character behind for that piece to claim",
        "spacing",
    ),
    Branch(r"\s+", "a run of spacing", "spacing"),
)
"""The twelve alternatives, in the order the engine tries them.

The order is the rule. A piece is the match of the first branch that succeeds
at its starting position, so the contractions being written first is what makes
an apostrophe and a letter t one piece rather than two.
"""

COMPILED_BRANCHES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(branch.source) for branch in BRANCHES
)
"""Each branch on its own, so a piece can be credited to the one that made it."""


def _rule_for(key: str) -> PatternPreTokenizer:
    """The splitter for one of the offered patterns."""
    return PatternPreTokenizer(pattern=PATTERNS[key].source)


THE_RULE = _rule_for("the-pattern")
"""The subject of the page."""


def _branch_of(text: str, start: int, end: int) -> int | None:
    """Which branch produced the piece at this span, or nothing if unattributable.

    A branch is tried at the piece's own starting position in the whole text
    rather than against the piece's characters alone, because two of the twelve
    read what comes after the match before deciding, and cutting the text at the
    piece would hide it from them.
    """
    for position, compiled in enumerate(COMPILED_BRANCHES):
        found = compiled.match(text, start)
        if found is not None and found.end() == end:
            return position
    return None


class Piece(BaseModel):
    """One piece, where it came from, and which branch claimed it.

    ``branch`` is the position in the list of twelve, and is absent whenever the
    pattern in force is not the one the page is about, since the twelve are not
    that pattern's branches then.
    """

    text: str
    start: int
    end: int
    has_a_leading_space: bool
    branch: int | None


class Reading(BaseModel):
    """One rule's answer for one text, and the two ways of putting it back.

    ``joins_back_exactly`` concatenates the pieces with nothing between them and
    asks for the whole text back, spacing included, which is the question this
    page turns on. ``joins_back_with_one_space`` puts one space between every
    pair instead, which is what a rule that discarded the spacing has to fall
    back on.
    """

    label: str
    pieces: list[str]
    n_pieces: int
    n_characters_in_pieces: int
    joins_back_exactly: bool
    joins_back_with_one_space: bool


class PatternReading(BaseModel):
    """One text read by one pattern, with the four earlier rules beside it."""

    source: str
    n_characters: int
    n_non_space_characters: int
    pattern_key: str
    pattern_source: str
    pattern_label: str
    pattern_note: str
    pieces: list[Piece]
    n_pieces: int
    n_with_a_leading_space: int
    n_characters_in_pieces: int
    joins_back_exactly: bool
    joins_back_with_one_space: bool
    on_spaces: Reading
    on_boundaries: Reading
    on_annotation_rules: Reading
    on_translation_rules: Reading


class SplitRequest(BaseModel):
    """Texts to read, and which of the seven patterns to read them with."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)
    pattern: PatternKey = "the-pattern"

    model_config = {"extra": "forbid"}


class SplitResponse(BaseModel):
    """Every text of the request, read by the chosen pattern and by four rules."""

    texts: list[PatternReading]


def _reading(text: str, rule: PreTokenizer, label: str) -> Reading:
    """How one of the earlier rules reads one text, and what putting it back gives."""
    pieces = [word.text for word in rule.split(text)]
    return Reading(
        label=label,
        pieces=pieces,
        n_pieces=len(pieces),
        n_characters_in_pieces=sum(len(piece) for piece in pieces),
        joins_back_exactly="".join(pieces) == text,
        joins_back_with_one_space=" ".join(pieces) == text,
    )


def _pattern_reading(text: str, key: str) -> PatternReading:
    """One text taken all the way through, by one pattern and by the four rules."""
    choice = PATTERNS[key]
    attribute = key == "the-pattern"
    words = list(_rule_for(key).split(text))
    pieces = [
        Piece(
            text=word.text,
            start=word.start,
            end=word.end,
            has_a_leading_space=word.text.startswith(" "),
            branch=_branch_of(text, word.start, word.end) if attribute else None,
        )
        for word in words
    ]
    joined = "".join(piece.text for piece in pieces)

    return PatternReading(
        source=text,
        n_characters=len(text),
        n_non_space_characters=sum(1 for character in text if not character.isspace()),
        pattern_key=key,
        pattern_source=choice.source,
        pattern_label=choice.label,
        pattern_note=choice.note,
        pieces=pieces,
        n_pieces=len(pieces),
        n_with_a_leading_space=sum(1 for piece in pieces if piece.has_a_leading_space),
        n_characters_in_pieces=sum(len(piece.text) for piece in pieces),
        joins_back_exactly=joined == text,
        joins_back_with_one_space=" ".join(piece.text for piece in pieces) == text,
        on_spaces=_reading(text, SPACE_RULE, "every run of characters with no spacing"),
        on_boundaries=_reading(text, BOUNDARY_RULE, "the standard boundary rules"),
        on_annotation_rules=_reading(text, ANNOTATION_RULE, "the annotation rules"),
        on_translation_rules=_reading(text, TRANSLATION_RULE, "the translation rules"),
    )


@router.post("/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    """Read every text of the request by the chosen pattern and by four earlier rules."""
    return SplitResponse(
        texts=[_pattern_reading(text, request.pattern) for text in request.texts]
    )


class BranchView(BaseModel):
    """One branch, what it reads, and how much of a corpus it claimed."""

    position: int
    source: str
    reads: str
    family: str
    n_claimed: int
    share_claimed: float
    examples: list[str]


class HeldSpaceView(BaseModel):
    """One text with doubled spacing, read with and without the eleventh branch."""

    text: str
    with_the_branch: list[str]
    without_the_branch: list[str]
    n_with: int
    n_without: int


class RefusedPattern(BaseModel):
    """A pattern that names no rule at all, and why not."""

    source: str
    reason: str


class RandomTexts(BaseModel):
    """How many drawn texts came back from their own pieces.

    The claim that the pieces tile any text is a property of the branches
    rather than an observation, so this attacks it rather than illustrating it.
    """

    n_texts: int
    n_joining_back: int
    n_codepoints_drawn_from: int
    longest_text: int


class BranchesResponse(BaseModel):
    """The twelve branches, what each claimed, and the three things they show."""

    pattern_source: str
    branches: list[BranchView]
    n_branches: int
    n_branches_used: int
    n_pieces: int
    n_texts: int
    running: PatternReading
    held_space: HeldSpaceView
    random_texts: RandomTexts
    refusals: list[RefusedPattern]


RANDOM_SEED = 11
"""One seed, fixed, so the drawn texts are the same texts on every request."""

N_RANDOM_TEXTS = 4000
"""How many texts to draw. Four thousand runs in a few hundredths of a second."""

LONGEST_RANDOM_TEXT = 30
"""The longest drawn text, in characters."""

AWKWARD_CHARACTERS = " \t\n ½Ⅻ²₂_"
"""Characters the drawing would otherwise almost never produce.

Spacing of four kinds, the fraction and the Roman numeral the two readings of
the character classes disagree about, a superscript, a subscript, and the
underscore that falls between two of the branches.
"""


@cache
def _random_texts() -> RandomTexts:
    """Draw texts and ask each time whether its pieces are the text again."""
    alphabet = "".join(chr(code) for code in range(32, 2000)) + AWKWARD_CHARACTERS
    draw = random.Random(RANDOM_SEED)
    joining = 0
    for _ in range(N_RANDOM_TEXTS):
        text = "".join(
            draw.choice(alphabet) for _ in range(draw.randint(0, LONGEST_RANDOM_TEXT))
        )
        if "".join(THE_RULE.split(text).texts) == text:
            joining += 1

    return RandomTexts(
        n_texts=N_RANDOM_TEXTS,
        n_joining_back=joining,
        n_codepoints_drawn_from=len(set(alphabet)),
        longest_text=LONGEST_RANDOM_TEXT,
    )


MALFORMED_PATTERNS: tuple[str, ...] = (
    r"'s|'t|( ?[^\W\d_]+",
    r"* ?[^\W\d_]+",
    r" ?[^\W\d_+",
)
"""Three patterns that do not name a rule, each a plausible mistyping of one branch.

Not offered to a caller, since the choice of pattern is closed. They are here so
the page can say what happens to a pattern that is not well formed, which is
that it is refused before it ever reads a text.
"""


def _refused(source: str) -> RefusedPattern:
    """One malformed pattern, with the refusal written in the page's own words.

    The refusal is rewritten here rather than passed through, because a widget
    renders this sentence and a reader has no reason to meet an identifier. A
    test asserts that none survives.
    """
    try:
        PatternPreTokenizer(pattern=source)
    except ValidationError:
        return RefusedPattern(
            source=source,
            reason=(
                "refused before it read anything, because a run of characters "
                "that is not a well formed pattern names no set of matches and "
                "so names no rule"
            ),
        )
    return RefusedPattern(source=source, reason="accepted, and it names a rule")


@cache
def _branches() -> BranchesResponse:
    """The branch tally over the notebook, computed once because it takes no input."""
    tally: Counter[int] = Counter()
    examples: dict[int, list[str]] = {}
    total = 0
    for text in NOTEBOOK:
        for word in THE_RULE.split(text):
            total += 1
            position = _branch_of(text, word.start, word.end)
            if position is None:
                continue
            tally[position] += 1
            shown = examples.setdefault(position, [])
            if word.text not in shown and len(shown) < 4:
                shown.append(word.text)

    with_branch = [word.text for word in THE_RULE.split(SPACED_SENTENCE)]
    without_branch = [
        word.text
        for word in _rule_for("without-the-held-back-space").split(SPACED_SENTENCE)
    ]

    return BranchesResponse(
        pattern_source=GPT2_PATTERN,
        branches=[
            BranchView(
                position=position,
                source=branch.source,
                reads=branch.reads,
                family=branch.family,
                n_claimed=tally[position],
                share_claimed=tally[position] / total if total else 0.0,
                examples=examples.get(position, []),
            )
            for position, branch in enumerate(BRANCHES)
        ],
        n_branches=len(BRANCHES),
        n_branches_used=sum(1 for position in range(len(BRANCHES)) if tally[position]),
        n_pieces=total,
        n_texts=len(NOTEBOOK),
        running=_pattern_reading(RUNNING_SENTENCE, "the-pattern"),
        held_space=HeldSpaceView(
            text=SPACED_SENTENCE,
            with_the_branch=with_branch,
            without_the_branch=without_branch,
            n_with=len(with_branch),
            n_without=len(without_branch),
        ),
        random_texts=_random_texts(),
        refusals=[_refused(source) for source in MALFORMED_PATTERNS],
    )


@router.get("/branches", response_model=BranchesResponse)
def branches() -> BranchesResponse:
    """The twelve branches read one at a time, and what six sentences gave each."""
    return _branches()


LATER_CONTRACTIONS = r"(?i:'s)|(?i:'t)|(?i:'re)|(?i:'ve)|(?i:'m)|(?i:'ll)|(?i:'d)"
"""The seven spellings again, each allowed to be written in capitals."""

LATER_PATTERN = (
    rf"{LATER_CONTRACTIONS}| ?[^\W\d_]+| ?\d{{1,3}}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)
"""The two changes the tokenizers of the models that followed made, and nothing else.

The seven spellings stop being case sensitive, and a run of digits is capped at
three characters. Everything else is held fixed so that what the page measures
is those two changes rather than a different pattern.
"""

LATER_RULE = PatternPreTokenizer(pattern=LATER_PATTERN)
"""The same rule with those two changes in it."""


class ContractionProbe(BaseModel):
    """One text, what the pattern made of it, and what two other rules made."""

    text: str
    pieces: list[str]
    on_annotation_rules: list[str]
    on_the_later_pattern: list[str]
    n_pieces: int
    a_contraction_branch_fired: bool
    look_at: str


class LaterChange(BaseModel):
    """One of the two later changes, with a text that shows it."""

    change: str
    was: str
    became: str
    text: str
    before: list[str]
    after: list[str]


CONTRACTION_CASES: tuple[tuple[str, str], ...] = (
    (
        "she didn't go",
        (
            "two pieces where a reader sees one word, and the cut falls at the "
            "apostrophe rather than in front of the negation"
        ),
    ),
    (
        "it's here",
        "the first of the seven, and the only one of them that is not a verb ending",
    ),
    (
        "they're here",
        (
            "three characters listed as one piece, which is what makes the list a "
            "list rather than a rule"
        ),
    ),
    (
        "IT'S HERE",
        (
            "the same word in capitals, and the branch does not fire, because the "
            "seven spellings are written in small letters"
        ),
    ),
    (
        "she didn’t go",
        (
            "the same word typed with the curved apostrophe a word processor "
            "inserts, and the branch does not fire, because it names the straight "
            "one"
        ),
    ),
    (
        "y'all came",
        (
            "an apostrophe followed by a letter that is not one of the seven, so "
            "the word comes apart into three"
        ),
    ),
    (
        "o'clock struck",
        (
            "the same again, on a word where the apostrophe is standing for a "
            "whole word rather than for letters"
        ),
    ),
    (
        "the students' notes",
        "an apostrophe with no letter after it, which is a mark and nothing else",
    ),
    (
        "'twas cold",
        (
            "the branch fires on a word it was not written for, and the piece it "
            "makes stands for the beginning of it rather than for a negation"
        ),
    ),
    (
        "he can't won't shan't",
        (
            "three negations, cut in the same place every time, where the "
            "annotation rules cut each of them in front of the negation"
        ),
    ),
    (
        "l'analyse est prête",
        (
            "a French article, where the apostrophe belongs with the letter in "
            "front of it and no branch says so, so it stands alone"
        ),
    ),
    (
        "she'd've gone",
        "two of the seven in one word, matched one after the other",
    ),
)
"""Twelve texts for the seven branches, chosen so half of them are misses."""


class ContractionsResponse(BaseModel):
    """The seven listed spellings, what they catch, and what they do not."""

    clitics: list[str]
    n_clitics: int
    probes: list[ContractionProbe]
    n_probes: int
    n_firing: int
    later_changes: list[LaterChange]


def _contraction_probe(text: str, look_at: str) -> ContractionProbe:
    """One text put to the pattern, the annotation rules and the later pattern."""
    words = list(THE_RULE.split(text))
    fired = any(
        (position := _branch_of(text, word.start, word.end)) is not None
        and position < 7
        for word in words
    )
    return ContractionProbe(
        text=text,
        pieces=[word.text for word in words],
        on_annotation_rules=list(ANNOTATION_RULE.split(text).texts),
        on_the_later_pattern=list(LATER_RULE.split(text).texts),
        n_pieces=len(words),
        a_contraction_branch_fired=fired,
        look_at=look_at,
    )


@cache
def _contractions() -> ContractionsResponse:
    """The contraction probes, computed once because none takes an input."""
    probes = [_contraction_probe(text, note) for text, note in CONTRACTION_CASES]
    capitals = "THE DRIFT DIDN'T GO AWAY"
    figures = "in 2019 the drift was 1000000 counts"

    return ContractionsResponse(
        clitics=[branch.source for branch in BRANCHES[:7]],
        n_clitics=7,
        probes=probes,
        n_probes=len(probes),
        n_firing=sum(1 for probe in probes if probe.a_contraction_branch_fired),
        later_changes=[
            LaterChange(
                change=(
                    "the seven spellings stop being case sensitive, so a "
                    "sentence written in capitals is cut the same way as one "
                    "written in small letters"
                ),
                was="'s|'t|'re|'ve|'m|'ll|'d",
                became="(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                text=capitals,
                before=list(THE_RULE.split(capitals).texts),
                after=list(LATER_RULE.split(capitals).texts),
            ),
            LaterChange(
                change=(
                    "a run of digits is capped at three characters, so a long "
                    "figure is cut into groups instead of standing as one piece "
                    "the model has probably never seen"
                ),
                was=r" ?\d+",
                became=r" ?\d{1,3}",
                text=figures,
                before=list(THE_RULE.split(figures).texts),
                after=list(LATER_RULE.split(figures).texts),
            ),
        ],
    )


@router.get("/contractions", response_model=ContractionsResponse)
def contractions() -> ContractionsResponse:
    """The seven listed spellings, the words they miss, and the two later changes."""
    return _contractions()


class CorpusReading(BaseModel):
    """What one rule spends over one corpus.

    ``n_distinct`` is how many different pieces the corpus produced, so it is
    how many rows a table over it would need, and ``n_pieces`` is what anything
    reading the corpus pays per pass.
    """

    label: str
    n_pieces: int
    n_distinct: int
    n_characters_in_pieces: int
    share_of_characters: float
    n_joining_back_exactly: int
    n_joining_back_with_one_space: int


class DoubledRow(BaseModel):
    """One word the corpus spells twice, once with its space and once without."""

    word: str
    n_with_the_space: int
    n_without_the_space: int


class TellApartPair(BaseModel):
    """Two texts that differ only in their spacing, and which rules notice."""

    first: str
    second: str
    first_here: list[str]
    second_here: list[str]
    told_apart_here: bool
    told_apart_on_spaces: bool
    told_apart_on_boundaries: bool
    told_apart_on_annotation_rules: bool
    told_apart_on_translation_rules: bool


SPACING_PAIRS: tuple[tuple[str, str], ...] = (
    ("one two", "one  two"),
    ("The drift was small", "The drift was  small"),
    ("Dr. Alvarez", "Dr.\tAlvarez"),
    ("the notes", "the\nnotes"),
)
"""Four pairs differing only in the spacing between two words."""

WITHOUT_SPACES_IN_THE_WRITING = "低コストの再分析"
"""The running sentence's subject in a script that writes no spaces at all."""


class CorpusResponse(BaseModel):
    """The six sentences under five rules, and what attaching the space costs."""

    n_texts: int
    n_characters: int
    n_non_space_characters: int
    readings: list[CorpusReading]
    without_the_optional_space: CorpusReading
    n_distinct_with_a_leading_space: int
    n_distinct: int
    doubled: list[DoubledRow]
    told_apart: list[TellApartPair]
    scriptless_text: str
    scriptless_pieces: list[str]
    scriptless_on_boundaries: list[str]
    notebook: list[str]
    running: PatternReading


def _corpus_reading(
    texts: tuple[str, ...], rule: PreTokenizer, label: str
) -> CorpusReading:
    """How one rule reads the whole corpus."""
    pieces = [word.text for text in texts for word in rule.split(text)]
    characters = sum(len(text) for text in texts)

    return CorpusReading(
        label=label,
        n_pieces=len(pieces),
        n_distinct=len(set(pieces)),
        n_characters_in_pieces=sum(len(piece) for piece in pieces),
        share_of_characters=(
            sum(len(piece) for piece in pieces) / characters if characters else 0.0
        ),
        n_joining_back_exactly=sum(
            1 for text in texts if "".join(rule.split(text).texts) == text
        ),
        n_joining_back_with_one_space=sum(
            1 for text in texts if " ".join(rule.split(text).texts) == text
        ),
    )


def _tell_apart(first: str, second: str) -> TellApartPair:
    """Whether each rule gives the two texts different answers."""

    def differs(rule: PreTokenizer) -> bool:
        return list(rule.split(first).texts) != list(rule.split(second).texts)

    return TellApartPair(
        first=first,
        second=second,
        first_here=list(THE_RULE.split(first).texts),
        second_here=list(THE_RULE.split(second).texts),
        told_apart_here=differs(THE_RULE),
        told_apart_on_spaces=differs(SPACE_RULE),
        told_apart_on_boundaries=differs(BOUNDARY_RULE),
        told_apart_on_annotation_rules=differs(ANNOTATION_RULE),
        told_apart_on_translation_rules=differs(TRANSLATION_RULE),
    )


@cache
def _corpus() -> CorpusResponse:
    """The corpus counts, computed once because they take no input."""
    pieces = [word.text for text in NOTEBOOK for word in THE_RULE.split(text)]
    counts = Counter(pieces)
    distinct = set(pieces)
    doubled = sorted(
        piece[1:]
        for piece in distinct
        if piece.startswith(" ") and piece[1:] in distinct and piece[1:].strip()
    )

    return CorpusResponse(
        n_texts=len(NOTEBOOK),
        n_characters=sum(len(text) for text in NOTEBOOK),
        n_non_space_characters=sum(
            1 for text in NOTEBOOK for character in text if not character.isspace()
        ),
        readings=[
            _corpus_reading(
                NOTEBOOK, SPACE_RULE, "every run of characters with no spacing"
            ),
            _corpus_reading(NOTEBOOK, BOUNDARY_RULE, "the standard boundary rules"),
            _corpus_reading(NOTEBOOK, ANNOTATION_RULE, "the annotation rules"),
            _corpus_reading(NOTEBOOK, TRANSLATION_RULE, "the translation rules"),
            _corpus_reading(NOTEBOOK, THE_RULE, "the pattern"),
        ],
        without_the_optional_space=_corpus_reading(
            NOTEBOOK,
            _rule_for("without-the-optional-space"),
            "the pattern with every space standing alone",
        ),
        n_distinct_with_a_leading_space=sum(
            1 for piece in distinct if piece.startswith(" ")
        ),
        n_distinct=len(distinct),
        doubled=[
            DoubledRow(
                word=word,
                n_with_the_space=counts[" " + word],
                n_without_the_space=counts[word],
            )
            for word in doubled
        ],
        told_apart=[_tell_apart(first, second) for first, second in SPACING_PAIRS],
        scriptless_text=WITHOUT_SPACES_IN_THE_WRITING,
        scriptless_pieces=list(THE_RULE.split(WITHOUT_SPACES_IN_THE_WRITING).texts),
        scriptless_on_boundaries=list(
            BOUNDARY_RULE.split(WITHOUT_SPACES_IN_THE_WRITING).texts
        ),
        notebook=list(NOTEBOOK),
        running=_pattern_reading(RUNNING_SENTENCE, "the-pattern"),
    )


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    """Six sentences under five rules, and what a leading space costs in rows."""
    return _corpus()


@cache
def _category_ranges(wanted: str) -> str:
    """A character class body holding every codepoint of these general categories.

    The published pattern names its two classes by Unicode general category, and
    ``re`` cannot. Walking all 1,114,112 codepoints once and writing the answer
    out as explicit ranges gives a class ``re`` can compile whose members are
    exactly the published one's, so both readings run through the same splitter
    and the classes are the only thing that differs between them.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    previous = 0
    for code in range(0x110000):
        inside = unicodedata.category(chr(code)).startswith(wanted)
        if inside and start is None:
            start = code
        if not inside and start is not None:
            spans.append((start, previous))
            start = None
        if inside:
            previous = code
    if start is not None:
        spans.append((start, 0x10FFFF))

    return "".join(
        f"\\U{low:08x}" if low == high else f"\\U{low:08x}-\\U{high:08x}"
        for low, high in spans
    )


@cache
def _as_published() -> PatternPreTokenizer:
    """The published pattern, with its two classes written out codepoint by codepoint."""
    letters = _category_ranges("L")
    numbers = _category_ranges("N")
    return PatternPreTokenizer(
        pattern=(
            r"'s|'t|'re|'ve|'m|'ll|'d"
            rf"| ?[{letters}]+| ?[{numbers}]+| ?[^\s{letters}{numbers}]+"
            r"|\s+(?!\S)|\s+"
        )
    )


class DialectProbe(BaseModel):
    """One text, read by the published classes and by the ones available here."""

    text: str
    as_published: list[str]
    here: list[str]
    agree: bool
    look_at: str


class ClassCounts(BaseModel):
    """How many codepoints the two readings put in different classes.

    ``n_read_as_letters_here`` is the whole of the disagreement: a codepoint
    Unicode files under a number category that is not a decimal digit is a
    number to the published classes and a letter to the ones here.
    """

    n_number_codepoints: int
    n_read_as_digits_here: int
    n_read_as_letters_here: int
    n_letter_codepoints: int
    n_letters_agreeing: int
    examples: list[str]


class AdjacencyRow(BaseModel):
    """One arrangement, and how many of the disputed characters split differently in it."""

    context: str
    shape: str
    n_differing: int
    n_characters: int


class UnderscoreView(BaseModel):
    """What the third branch does with the underscore, with and without its repair."""

    text: str
    with_the_repair: list[str]
    without_the_repair: list[str]
    joins_back_with_the_repair: bool
    joins_back_without_the_repair: bool
    n_characters_lost: int


class DialectResponse(BaseModel):
    """Where the published pattern and the one that can be written here part."""

    published_letters: str
    published_numbers: str
    letters_here: str
    numbers_here: str
    marks_here: str
    probes: list[DialectProbe]
    n_probes: int
    n_agreeing: int
    counts: ClassCounts
    adjacency: list[AdjacencyRow]
    underscore: UnderscoreView
    n_notebook_texts_differing: int


DIALECT_CASES: tuple[tuple[str, str], ...] = (
    (
        RUNNING_SENTENCE,
        (
            "identical, and so is every other sentence of the six, because nothing "
            "in ordinary English writing reaches the disputed characters"
        ),
    ),
    (
        "the drift was 1½ per cent",
        (
            "the published classes read the figure and the fraction as one run; "
            "here the fraction is a letter and the figure is cut away from it"
        ),
    ),
    (
        "page aⅫ of the report",
        (
            "the published classes cut the Roman numeral away from the letter "
            "before it; here it is a letter too, so the two stay joined"
        ),
    ),
    (
        "chapter Ⅻ of the agreement",
        (
            "the same numeral with a space on each side, and the two readings "
            "agree, because nothing is beside it to be joined to or cut from"
        ),
    ),
    (
        "a² and x₂",
        (
            "a superscript and a subscript, each cut off by the published classes "
            "and each kept by the ones here"
        ),
    ),
    (
        "٣٤ counts",
        (
            "Arabic-Indic figures, which are decimal digits, so both readings call "
            "them a number"
        ),
    ),
    (
        "Дрейф был мал",
        (
            "an alphabetic script that is not Latin, where the two readings agree "
            "character for character"
        ),
    ),
)
"""Seven texts, three of which split differently under the two readings."""

ADJACENCY_SHAPES: tuple[tuple[str, str], ...] = (
    ("standing alone between two spaces", "a {} b"),
    ("directly after a letter", "a{} b"),
    ("directly after a digit", "1{} b"),
    ("directly before a letter", "a {}b"),
)
"""Four arrangements of one disputed character, to find where the two part."""


@cache
def _dialect() -> DialectResponse:
    """The two readings compared, computed once because none of it takes an input."""
    published = _as_published()
    letter_here = re.compile(r"[^\W\d_]")
    digit_here = re.compile(r"\d")

    disputed: list[str] = []
    n_numbers = 0
    n_digits_here = 0
    for code in range(0x110000):
        character = chr(code)
        if not unicodedata.category(character).startswith("N"):
            continue
        n_numbers += 1
        if digit_here.fullmatch(character):
            n_digits_here += 1
        elif letter_here.fullmatch(character):
            disputed.append(character)

    n_letters = 0
    n_letters_agreeing = 0
    for code in range(0x110000):
        character = chr(code)
        if unicodedata.category(character).startswith("L"):
            n_letters += 1
            if letter_here.fullmatch(character):
                n_letters_agreeing += 1

    probes = [
        DialectProbe(
            text=text,
            as_published=list(published.split(text).texts),
            here=list(THE_RULE.split(text).texts),
            agree=list(published.split(text).texts) == list(THE_RULE.split(text).texts),
            look_at=note,
        )
        for text, note in DIALECT_CASES
    ]

    without_repair = PatternPreTokenizer(
        pattern=r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
    )
    underscored = "re-analysis_2019.csv"
    kept = list(THE_RULE.split(underscored).texts)
    dropped = list(without_repair.split(underscored).texts)

    return DialectResponse(
        published_letters=r"\p{L}",
        published_numbers=r"\p{N}",
        letters_here=r"[^\W\d_]",
        numbers_here=r"\d",
        marks_here=r"[^\s\w]|_",
        probes=probes,
        n_probes=len(probes),
        n_agreeing=sum(1 for probe in probes if probe.agree),
        counts=ClassCounts(
            n_number_codepoints=n_numbers,
            n_read_as_digits_here=n_digits_here,
            n_read_as_letters_here=len(disputed),
            n_letter_codepoints=n_letters,
            n_letters_agreeing=n_letters_agreeing,
            examples=disputed[:8],
        ),
        adjacency=[
            AdjacencyRow(
                context=context,
                shape=shape.format("½"),
                n_differing=sum(
                    1
                    for character in disputed
                    if list(published.split(shape.format(character)).texts)
                    != list(THE_RULE.split(shape.format(character)).texts)
                ),
                n_characters=len(disputed),
            )
            for context, shape in ADJACENCY_SHAPES
        ],
        underscore=UnderscoreView(
            text=underscored,
            with_the_repair=kept,
            without_the_repair=dropped,
            joins_back_with_the_repair="".join(kept) == underscored,
            joins_back_without_the_repair="".join(dropped) == underscored,
            n_characters_lost=len(underscored) - sum(len(piece) for piece in dropped),
        ),
        n_notebook_texts_differing=sum(
            1
            for text in NOTEBOOK
            if list(published.split(text).texts) != list(THE_RULE.split(text).texts)
        ),
    )


@router.get("/dialect", response_model=DialectResponse)
def dialect() -> DialectResponse:
    """The published character classes against the ones that can be written here."""
    return _dialect()
