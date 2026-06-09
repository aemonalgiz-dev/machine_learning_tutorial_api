"""Moses Rules.

Every endpoint here is registered under ``/concepts/moses-rules``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about MosesPreTokenizer, RuleAndExceptionPreTokenizer.

What this module adds to the library
------------------------------------
Nothing about the rules themselves. Every piece and every span comes out of the
library's ``MosesPreTokenizer``, and the three rules it is measured against are
``PennTreebankPreTokenizer``, ``WhitespacePreTokenizer`` and the English
configuration of ``RuleAndExceptionPreTokenizer``, on the same texts, so the
numbers here line up with the three sibling pages. What is assembled here is
the bookkeeping a page wants and a splitter has no reason to carry: whether the
pieces of a text concatenate back to everything in it that is not spacing,
whether undoing the rewriting recovers it when the rewriting is switched on,
which full stops each rule keeps against a stated answer, how many entries of
the nonbreaking-prefix list a corpus of the right register actually reaches,
and what the four rules cost in pieces over the same two corpora.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings. A test asserts that no identifier
survives into any string any endpoint returns.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache
from typing import Annotated, Literal

from fastapi import APIRouter
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.moses import (
    HYPHEN_PLACEHOLDER,
    NONBREAKING_PREFIXES,
    NUMERIC_ONLY_PREFIXES,
    UNESCAPES,
    MosesPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.rule_based import (
    ENGLISH_EXCEPTIONS,
    RuleAndExceptionPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/concepts/moses-rules", tags=["moses-rules"])

MOSES = MosesPreTokenizer()
"""The subject of the page at its defaults: English, no placeholder, no rewriting."""

ANNOTATION = PennTreebankPreTokenizer()
"""The rules of the sibling page, which the whole page is measured against."""

SPACE_RULE = WhitespacePreTokenizer()
"""The crudest of the four, so all four sibling pages count over one corpus."""

TABLE_RULE = RuleAndExceptionPreTokenizer.english()
"""The same shape written out in full, a rule with a table of exceptions."""

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

QUOTED_SENTENCE: str = (
    '"Dr. Alvarez didn\'t expect the low-cost re-analysis," she wrote.'
)
"""The same sentence in quotation marks, which is where the two rules part."""

MINUTES: tuple[str, ...] = (
    "Dr. Alvarez didn't expect the low-cost re-analysis.",
    "Mr. Okafor asked whether the sensors had been calibrated in Nov. 2019.",
    "Mrs. Lindqvist replied that Art. 3 of the agreement covers calibration.",
    "The contract was signed by Alvarez Instruments Ltd. and by Okafor Co.",
    "Prof. Nakamura said the drift was 5,300 parts per million, not 1,000.",
    "Gov. Reyes had asked for the same figures in Jan.",
    "Sen. Duarte objected that Fig. 4 was drawn from the wrong column.",
    "The delegation met at St. Andrew's, near Mt. Pleasant.",
    "Ms. Haddad read out the reading, the drift and the residual, etc.",
    '"We cannot sign this," said Mr. Okafor, "until No. 5 is corrected."',
    "The reading of 3.14 was checked against the low-cost sensor vs. the reference one.",
    "J. Alvarez had run the first analysis in 2019.",
    "The re-analysis cost 1,000 euros and took until Dec.",
    "It's the store room readings, not the field ones, that drifted.",
    "The committee's own well-known preference is for a re-analysis.",
    "Inc. and Corp. are written differently in the two translations.",
    "Rev. Santos asked whether approx. 40 per cent was the right share.",
    "The drift was small. It did not go away.",
)
"""Eighteen sentences in the register the rules were written for.

The six-sentence notebook holds one abbreviation, which is too few to say
anything about a list of fifty-five entries, so the page needs a second corpus
of the kind of writing a translation system was trained on: minutes of a
meeting, with titles, dates, article numbers, company suffixes and figures in
them. Written here rather than borrowed, and every count over it is reported as
being over these eighteen sentences.
"""

MAX_TEXTS = 4
"""How many texts one request may split. The widgets ask for one at a time."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]

Language = Literal["en", "fr", "it"]
"""The three settings the apostrophe clause distinguishes, as a closed choice.

A closed choice rather than free text, so a request cannot name a setting the
rules have no clause for and there is no refusal for a reader to meet.
"""

UNDO_PATTERN = re.compile(
    "|".join(re.escape(rewritten) for rewritten in (*UNESCAPES, HYPHEN_PLACEHOLDER))
)
"""Every spelling the rules can write that is not the source it stands for."""

UNDO: dict[str, str] = {**UNESCAPES, HYPHEN_PLACEHOLDER: "-"}
"""What each of those spellings stands for, which is what makes the rewriting undoable."""


def _undone(text: str) -> str:
    """One piece's text with every rewriting the rules can do put back."""
    return UNDO_PATTERN.sub(lambda found: UNDO[found.group(0)], text)


def _configured(
    language: Language, mark_hyphens: bool, rewrite_marks: bool
) -> MosesPreTokenizer:
    """The rules at one setting of the three switches the page exposes."""
    return MosesPreTokenizer(
        language=language,
        aggressive_hyphen_splitting=mark_hyphens,
        escape_special_characters=rewrite_marks,
    )


class Piece(BaseModel):
    """One piece, the span it claims, and whether its text is still that slice.

    ``source`` is what the span actually covers in the text. ``is_rewritten``
    says the two differ, and ``undone`` is what the text becomes once every
    rewriting is put back, which for these rules is the source again.
    """

    text: str
    start: int
    end: int
    source: str
    is_rewritten: bool
    undone: str


class RuleView(BaseModel):
    """One rule's reading of one text, and the four things the page asks of it.

    ``glued`` is the pieces joined with one space each, which is what a reader
    sees if nothing restores the spacing. ``pieces_join_to_the_source`` is the
    stronger question, whether the pieces concatenated with nothing between
    them give back everything in the text that is not spacing, and
    ``undoing_recovers_the_source`` asks the same after putting every rewriting
    back.
    """

    pieces: list[Piece]
    n_pieces: int
    n_characters_in_pieces: int
    share_covered: float
    glued: str
    glue_is_exact: bool
    n_rewritten: int
    pieces_join_to_the_source: bool
    undoing_recovers_the_source: bool


class SplitView(BaseModel):
    """One text read by these rules, with the three others beside it."""

    source: str
    n_characters: int
    n_non_space_characters: int
    on_these_rules: RuleView
    on_annotation_rules: RuleView
    on_spaces: RuleView
    on_the_table: RuleView


class SplitRequest(BaseModel):
    """Texts to read, and the three settings the rules take."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)
    language: Language = "en"
    mark_hyphens: bool = False
    rewrite_marks: bool = False

    model_config = {"extra": "forbid"}


class SplitResponse(BaseModel):
    """Every text of the request, seen through all four rules."""

    texts: list[SplitView]


def _pieces_of(text: str, rule: PreTokenizer) -> list[Piece]:
    """Every piece one rule found, paired with the slice its span names."""
    return [
        Piece(
            text=word.text,
            start=word.start,
            end=word.end,
            source=text[word.start : word.end],
            is_rewritten=text[word.start : word.end] != word.text,
            undone=_undone(word.text),
        )
        for word in rule.split(text)
    ]


def _rule_view(text: str, rule: PreTokenizer) -> RuleView:
    """How one rule reads one text, and what each way back gives."""
    pieces = _pieces_of(text, rule)
    covered = sum(piece.end - piece.start for piece in pieces)
    glued = " ".join(piece.text for piece in pieces)
    without_space = "".join(character for character in text if not character.isspace())

    return RuleView(
        pieces=pieces,
        n_pieces=len(pieces),
        n_characters_in_pieces=covered,
        share_covered=covered / len(text) if text else 0.0,
        glued=glued,
        glue_is_exact=glued == text,
        n_rewritten=sum(1 for piece in pieces if piece.is_rewritten),
        pieces_join_to_the_source=(
            "".join(piece.text for piece in pieces) == without_space
        ),
        undoing_recovers_the_source=(
            "".join(piece.undone for piece in pieces) == without_space
        ),
    )


def _split_view(text: str, rule: MosesPreTokenizer) -> SplitView:
    """One text taken all the way through, by the four rules the page uses."""
    return SplitView(
        source=text,
        n_characters=len(text),
        n_non_space_characters=sum(1 for character in text if not character.isspace()),
        on_these_rules=_rule_view(text, rule),
        on_annotation_rules=_rule_view(text, ANNOTATION),
        on_spaces=_rule_view(text, SPACE_RULE),
        on_the_table=_rule_view(text, TABLE_RULE),
    )


@router.post("/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    """Read every text of the request by all four rules and report everything between."""
    rule = _configured(request.language, request.mark_hyphens, request.rewrite_marks)
    return SplitResponse(texts=[_split_view(text, rule) for text in request.texts])


class StopCase:
    """One text, the word a full stop follows, and where that stop belongs."""

    __slots__ = ("belongs_to_the_word", "label", "look_at", "text", "word")

    def __init__(
        self,
        label: str,
        text: str,
        word: str,
        belongs_to_the_word: bool,
        look_at: str,
    ) -> None:
        self.label = label
        self.text = text
        self.word = word
        self.belongs_to_the_word = belongs_to_the_word
        self.look_at = look_at


DECIDABLE_STOPS: tuple[StopCase, ...] = (
    StopCase(
        "a title, on the list",
        "Dr. Alvarez signed the re-analysis.",
        "Dr",
        True,
        "every rule here keeps it, and only one of them consulted anything to do so",
    ),
    StopCase(
        "a company suffix, on the list",
        "Okafor Co. signed the re-analysis.",
        "Co",
        True,
        "the same again, and the stop is not the last character of the text",
    ),
    StopCase(
        "a numbering word with a figure after it",
        "No. 5 shows the drift.",
        "No",
        True,
        "kept, and the clause that keeps it also reads the first character of the next word",
    ),
    StopCase(
        "the same word, with a sentence after it",
        "No. Nothing shows the drift.",
        "No",
        False,
        "split, from the same clause reading a letter where it wanted a figure",
    ),
    StopCase(
        "an abbreviation nobody listed, with a figure after it",
        "The reading was approx. 40 per cent.",
        "approx",
        True,
        "split, and it is an abbreviation, so this one is wrong",
    ),
    StopCase(
        "the same abbreviation, with a lowercase word after it",
        "The reading was approx. forty per cent.",
        "approx",
        True,
        "kept, and nothing about the abbreviation decided it; the next word being lowercase did",
    ),
    StopCase(
        "a Latin abbreviation, with a lowercase word after it",
        "Alvarez et al. ran the analysis.",
        "al",
        True,
        "kept, by the same clause and for the same reason",
    ),
    StopCase(
        "the same abbreviation, with a capital after it",
        "Alvarez et al. Ran it again.",
        "al",
        True,
        "split, because a capital is what a new sentence starts with",
    ),
    StopCase(
        "a title nobody listed",
        "Gov. Reyes asked for the figures.",
        "Gov",
        True,
        "split, and there is no clause anywhere that could have saved it",
    ),
    StopCase(
        "an ordinary word ending a sentence",
        "The drift was small. It did not go away.",
        "small",
        False,
        "the one the whole clause exists for, and the text holds two sentences",
    ),
)
"""Ten full stops whose answer can be stated, since none of them ends its text.

Every one is a stop somewhere in the middle of a text, so a rule that read
English perfectly would keep eight of them and split two, and the two rules can
therefore be scored. The paired entries are the point: the same abbreviation
twice, decided differently by what follows it.
"""

UNDECIDABLE_STOPS: tuple[StopCase, ...] = (
    StopCase(
        "a title at the very end",
        "The re-analysis was signed by Dr.",
        "Dr",
        True,
        "one character is closing the abbreviation and the sentence at once",
    ),
    StopCase(
        "a company suffix at the very end",
        "The re-analysis was signed by Okafor Co.",
        "Co",
        True,
        "the same, and the two rules answer differently with neither being right",
    ),
    StopCase(
        "an abbreviation at the very end",
        "The figures were requested in Jan.",
        "Jan",
        True,
        "the writing holds one stop where the reading holds two",
    ),
)
"""The three stops with no correct answer, kept apart from the score.

A sentence ending in an abbreviation is written with one full stop doing both
jobs, and no division of the characters gives that character to two pieces, so
scoring a rule on these would be scoring it on a question with no answer.
"""


class StopProbe(BaseModel):
    """One text, what each rule did to the focal stop, and whether it was right."""

    label: str
    text: str
    word: str
    belongs_to_the_word: bool
    on_these_rules: list[str]
    on_annotation_rules: list[str]
    on_the_table: list[str]
    kept_here: bool
    kept_by_annotation: bool
    kept_by_the_table: bool
    right_here: bool
    right_by_annotation: bool
    right_by_the_table: bool
    look_at: str


class Scoreboard(BaseModel):
    """How many of the decidable stops each rule placed where it belongs.

    ``n_wanting_kept`` is there because a rule that answers ``kept`` every time
    scores exactly that number without distinguishing anything, which is what
    one of the three does.
    """

    n_cases: int
    n_wanting_kept: int
    n_right_here: int
    n_right_by_annotation: int
    n_right_by_the_table: int
    n_answers_here: int
    n_answers_by_annotation: int
    n_answers_by_the_table: int


class FullStopsResponse(BaseModel):
    """The decidable stops, scored, and the ones with no answer to score against."""

    probes: list[StopProbe]
    undecidable: list[StopProbe]
    scoreboard: Scoreboard


def _stop_probe(case: StopCase) -> StopProbe:
    """One case put to all three rules, with the stated answer beside them."""
    here = [piece.text for piece in _pieces_of(case.text, MOSES)]
    there = [piece.text for piece in _pieces_of(case.text, ANNOTATION)]
    table = [piece.text for piece in _pieces_of(case.text, TABLE_RULE)]
    attached = case.word + "."

    kept_here = attached in here
    kept_there = attached in there
    kept_table = attached in table

    return StopProbe(
        label=case.label,
        text=case.text,
        word=case.word,
        belongs_to_the_word=case.belongs_to_the_word,
        on_these_rules=here,
        on_annotation_rules=there,
        on_the_table=table,
        kept_here=kept_here,
        kept_by_annotation=kept_there,
        kept_by_the_table=kept_table,
        right_here=kept_here == case.belongs_to_the_word,
        right_by_annotation=kept_there == case.belongs_to_the_word,
        right_by_the_table=kept_table == case.belongs_to_the_word,
        look_at=case.look_at,
    )


@cache
def _full_stops() -> FullStopsResponse:
    """The stop probes and the score, computed once because none takes an input."""
    probes = [_stop_probe(case) for case in DECIDABLE_STOPS]

    return FullStopsResponse(
        probes=probes,
        undecidable=[_stop_probe(case) for case in UNDECIDABLE_STOPS],
        scoreboard=Scoreboard(
            n_cases=len(probes),
            n_wanting_kept=sum(1 for probe in probes if probe.belongs_to_the_word),
            n_right_here=sum(1 for probe in probes if probe.right_here),
            n_right_by_annotation=sum(
                1 for probe in probes if probe.right_by_annotation
            ),
            n_right_by_the_table=sum(1 for probe in probes if probe.right_by_the_table),
            n_answers_here=sum(1 for probe in probes if probe.kept_here),
            n_answers_by_annotation=sum(
                1 for probe in probes if probe.kept_by_annotation
            ),
            n_answers_by_the_table=sum(
                1 for probe in probes if probe.kept_by_the_table
            ),
        ),
    )


@router.get("/full-stops", response_model=FullStopsResponse)
def full_stops() -> FullStopsResponse:
    """Where each rule puts a full stop, against an answer stated in advance."""
    return _full_stops()


class ClauseTally(BaseModel):
    """One of the four clauses, and how many stops it was what kept."""

    clause: str
    n_kept: int
    examples: list[str]


def _clause_that_kept(before_the_stop: str, following_first_character: str) -> str:
    """Which of the four clauses is the first to keep this stop, in the page's words.

    The order is the order the clauses are tried in, so a stop that two clauses
    would have kept is credited to the first of them, which is what makes the
    tally a count of what was needed rather than of what would have sufficed.
    """
    if "." in before_the_stop and any(
        character.isalpha() for character in before_the_stop
    ):
        return "an interior stop and a letter inside the word"
    if before_the_stop in NONBREAKING_PREFIXES:
        return "the named list"
    if following_first_character.islower():
        return "a lowercase word after it"
    if (
        before_the_stop in NUMERIC_ONLY_PREFIXES
        and following_first_character.isnumeric()
    ):
        return "the figures list, with a figure after it"
    return "nothing, so the stop came off"


CLAUSES: tuple[str, ...] = (
    "an interior stop and a letter inside the word",
    "the named list",
    "a lowercase word after it",
    "the figures list, with a figure after it",
)
"""The four clauses in the order they are tried, so a tally can report all four."""


class ListCoverage(BaseModel):
    """How much of the named list a corpus of the right register reaches.

    ``n_words_ending_in_one_stop`` is how often the question was asked at all,
    ``reached`` is the entries that answered it, and ``never_reached`` is how
    many entries the corpus never had occasion to consult.
    """

    n_entries: int
    n_named_entries: int
    n_single_letter_entries: int
    n_figure_only_entries: int
    n_texts: int
    n_words_ending_in_one_stop: int
    reached: list[str]
    n_reached: int
    n_never_reached: int
    share_reached: float
    kept_whole: list[str]
    lost_the_stop: list[str]
    texts_ending_in_an_abbreviation: list[str]
    n_texts_ending_in_an_abbreviation: int
    by_clause: list[ClauseTally]
    n_kept_in_all: int


class ProposedEntry(BaseModel):
    """One proposed table entry, and whether a table of this shape may hold it."""

    word: str
    pieces: list[str]
    accepted: bool
    verdict: str


class ListResponse(BaseModel):
    """The named list, what a corpus reaches of it, and what an entry may say."""

    coverage: ListCoverage
    named_entries: list[str]
    figure_only_entries: list[str]
    n_table_entries: int
    n_table_entries_kept_whole: int
    n_table_entries_cut: int
    n_table_entries_reached: int
    table_entries_reached: list[str]
    proposals: list[ProposedEntry]
    minutes: list[str]


PROPOSED_ENTRIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("don't", ("do", "n't")),
    ("don't", ("do", "not")),
    ("Mr.", ("Mr.",)),
    ("Mr.", ("Mr",)),
    ("cannot", ("can", "not")),
    ("New York", ("New", "York")),
)
"""Six entries somebody might want, three of which a table of this shape refuses.

The pairs are the argument. The same word listed as a cut is accepted and
listed as a rewriting is refused, and that single constraint is what keeps
every piece a slice of the writing it came from.
"""


def _verdict(word: str, pieces: tuple[str, ...]) -> ProposedEntry:
    """Whether a table may hold one proposed entry, said in the page's own words.

    The refusal is rewritten here rather than passed through, because a widget
    renders this sentence and a reader has no reason to meet an identifier. A
    test asserts that none survives.
    """
    try:
        RuleAndExceptionPreTokenizer(exceptions={word: pieces})
    except ValidationError:
        if any(character.isspace() for character in word) or not word:
            verdict = (
                "refused, because an entry names one run of characters with no "
                "spacing in it, and the spacing has already decided this one"
            )
        elif not pieces or any(not piece for piece in pieces):
            verdict = "refused, because an entry that lists no pieces cuts nothing"
        else:
            joined = "".join(pieces)
            verdict = (
                f"refused, because the pieces join to {joined} rather than to "
                f"{word}, and an entry says where a word is cut and never what "
                f"it is turned into"
            )
        return ProposedEntry(
            word=word, pieces=list(pieces), accepted=False, verdict=verdict
        )

    return ProposedEntry(
        word=word,
        pieces=list(pieces),
        accepted=True,
        verdict="accepted, and the pieces join back to exactly the word listed",
    )


def _words_ending_in_one_stop(texts: tuple[str, ...]) -> list[str]:
    """Every whitespace-free run that ends in exactly one full stop, less that stop."""
    return [
        word.text[:-1]
        for text in texts
        for word in SPACE_RULE.split(text)
        if len(word.text) >= 2
        and word.text.endswith(".")
        and not word.text.endswith("..")
    ]


@cache
def _the_list() -> ListResponse:
    """The list, its coverage and the proposals, computed once."""
    asked = _words_ending_in_one_stop(MINUTES)
    named = sorted(entry for entry in NONBREAKING_PREFIXES if len(entry) > 1)
    letters = sorted(entry for entry in NONBREAKING_PREFIXES if len(entry) == 1)
    figures = sorted(NUMERIC_ONLY_PREFIXES)
    reached = sorted(
        {word for word in asked if word in NONBREAKING_PREFIXES | NUMERIC_ONLY_PREFIXES}
    )
    n_entries = len(NONBREAKING_PREFIXES) + len(NUMERIC_ONLY_PREFIXES)

    kept: list[str] = []
    lost: list[str] = []
    credited: dict[str, list[str]] = {clause: [] for clause in CLAUSES}
    for text in MINUTES:
        found = _pieces_of(text, MOSES)
        pieces = [piece.text for piece in found]
        for position, piece in enumerate(pieces):
            if len(piece) >= 2 and piece.endswith(".") and not piece.endswith(".."):
                kept.append(piece)
                following = (
                    text[found[position + 1].start] if position + 1 < len(found) else ""
                )
                credited[_clause_that_kept(piece[:-1], following)].append(piece)
            if piece == "." and position > 0:
                lost.append(pieces[position - 1])

    table_reached = sorted(
        {
            word.text
            for text in MINUTES
            for word in SPACE_RULE.split(text)
            if word.text in ENGLISH_EXCEPTIONS
        }
    )

    # A text whose last piece is a word with a stop still on it is a text
    # ending in an abbreviation, which is the case with no correct answer.
    ending_in_an_abbreviation = [
        text
        for text in MINUTES
        if (last := [piece.text for piece in _pieces_of(text, MOSES)][-1]).endswith(".")
        and last != "."
    ]

    return ListResponse(
        coverage=ListCoverage(
            n_entries=n_entries,
            n_named_entries=len(named),
            n_single_letter_entries=len(letters),
            n_figure_only_entries=len(figures),
            n_texts=len(MINUTES),
            n_words_ending_in_one_stop=len(asked),
            reached=reached,
            n_reached=len(reached),
            n_never_reached=n_entries - len(reached),
            share_reached=len(reached) / n_entries,
            kept_whole=[entry for entry, _ in Counter(kept).most_common()],
            lost_the_stop=[entry for entry, _ in Counter(lost).most_common()],
            texts_ending_in_an_abbreviation=ending_in_an_abbreviation,
            n_texts_ending_in_an_abbreviation=len(ending_in_an_abbreviation),
            by_clause=[
                ClauseTally(
                    clause=clause,
                    n_kept=len(credited[clause]),
                    examples=sorted(set(credited[clause])),
                )
                for clause in CLAUSES
            ],
            n_kept_in_all=len(kept),
        ),
        named_entries=named,
        figure_only_entries=figures,
        n_table_entries=len(ENGLISH_EXCEPTIONS),
        n_table_entries_kept_whole=sum(
            1 for pieces in ENGLISH_EXCEPTIONS.values() if len(pieces) == 1
        ),
        n_table_entries_cut=sum(
            1 for pieces in ENGLISH_EXCEPTIONS.values() if len(pieces) > 1
        ),
        n_table_entries_reached=len(table_reached),
        table_entries_reached=table_reached,
        proposals=[_verdict(word, pieces) for word, pieces in PROPOSED_ENTRIES],
        minutes=list(MINUTES),
    )


@router.get("/the-list", response_model=ListResponse)
def the_list() -> ListResponse:
    """The named list, how much of it eighteen sentences reach, and what an entry may say."""
    return _the_list()


class AmbiguityPair(BaseModel):
    """Two different texts, and whether one rule tells them apart.

    The pair is chosen so that one of them is written with the characters the
    other rule produces, which is the whole of why a rewriting cannot be undone.
    """

    first: str
    second: str
    first_on_these_rules: list[str]
    second_on_these_rules: list[str]
    first_on_annotation_rules: list[str]
    second_on_annotation_rules: list[str]
    told_apart_here: bool
    told_apart_by_annotation: bool


class ReversibilityResponse(BaseModel):
    """The round trip, the pair that defeats it, and what the rewriting costs."""

    quoted: SplitView
    quoted_with_marks_rewritten: SplitView
    running_with_hyphens_marked: SplitView
    ambiguity: AmbiguityPair
    n_texts: int
    n_joining_here: int
    n_joining_by_annotation: int
    n_glue_exact_here: int
    n_glue_exact_on_spaces: int


AMBIGUOUS_FIRST = 'she said "wait" and left'
"""A text with a plain quotation mark in it."""

AMBIGUOUS_SECOND = "she said ``wait'' and left"
"""A text already written in the spelling the annotation rules produce."""


@cache
def _reversibility() -> ReversibilityResponse:
    """The round-trip measurements, computed once because none takes an input."""
    marks = _configured("en", mark_hyphens=False, rewrite_marks=True)
    hyphens = _configured("en", mark_hyphens=True, rewrite_marks=False)

    return ReversibilityResponse(
        quoted=_split_view(QUOTED_SENTENCE, MOSES),
        quoted_with_marks_rewritten=_split_view(QUOTED_SENTENCE, marks),
        running_with_hyphens_marked=_split_view(RUNNING_SENTENCE, hyphens),
        ambiguity=AmbiguityPair(
            first=AMBIGUOUS_FIRST,
            second=AMBIGUOUS_SECOND,
            first_on_these_rules=list(MOSES.split(AMBIGUOUS_FIRST).texts),
            second_on_these_rules=list(MOSES.split(AMBIGUOUS_SECOND).texts),
            first_on_annotation_rules=list(ANNOTATION.split(AMBIGUOUS_FIRST).texts),
            second_on_annotation_rules=list(ANNOTATION.split(AMBIGUOUS_SECOND).texts),
            told_apart_here=(
                list(MOSES.split(AMBIGUOUS_FIRST).texts)
                != list(MOSES.split(AMBIGUOUS_SECOND).texts)
            ),
            told_apart_by_annotation=(
                list(ANNOTATION.split(AMBIGUOUS_FIRST).texts)
                != list(ANNOTATION.split(AMBIGUOUS_SECOND).texts)
            ),
        ),
        n_texts=len(MINUTES),
        n_joining_here=sum(
            1 for text in MINUTES if _rule_view(text, MOSES).pieces_join_to_the_source
        ),
        n_joining_by_annotation=sum(
            1
            for text in MINUTES
            if _rule_view(text, ANNOTATION).pieces_join_to_the_source
        ),
        n_glue_exact_here=sum(
            1 for text in MINUTES if _rule_view(text, MOSES).glue_is_exact
        ),
        n_glue_exact_on_spaces=sum(
            1 for text in MINUTES if _rule_view(text, SPACE_RULE).glue_is_exact
        ),
    )


@router.get("/reversibility", response_model=ReversibilityResponse)
def reversibility() -> ReversibilityResponse:
    """What survives being taken apart, and what the two rewritings cost."""
    return _reversibility()


class CorpusReading(BaseModel):
    """What one rule spends over one corpus.

    ``n_distinct`` is how many different pieces the corpus produced, so it is
    how many rows a table over it would need, and ``n_pieces`` is what anything
    reading the corpus pays per pass.
    """

    n_pieces: int
    n_distinct: int
    n_characters_covered: int
    share_covered: float


class ScatteredStem(BaseModel):
    """One word the crudest rule spells more than one way, and its spellings."""

    stem: str
    spellings: list[str]


class CorpusView(BaseModel):
    """One corpus read by all four rules, with what separates them."""

    n_texts: int
    n_characters: int
    n_non_space_characters: int
    on_these_rules: CorpusReading
    on_annotation_rules: CorpusReading
    on_spaces: CorpusReading
    on_the_table: CorpusReading
    only_on_these_rules: list[str]
    only_on_annotation_rules: list[str]
    scattered: list[ScatteredStem]
    n_pieces_with_hyphens_marked: int
    n_pieces_without_letter_or_digit: int


class CorpusResponse(BaseModel):
    """Both corpora, all four rules, and what the settings cost in length."""

    notebook: CorpusView
    notebook_texts: list[str]
    minutes: CorpusView
    minutes_texts: list[str]
    running: SplitView


def _corpus_reading(texts: tuple[str, ...], rule: PreTokenizer) -> CorpusReading:
    """How one rule reads one corpus."""
    pieces = [piece for text in texts for piece in _pieces_of(text, rule)]
    characters = sum(len(text) for text in texts)
    covered = sum(piece.end - piece.start for piece in pieces)

    return CorpusReading(
        n_pieces=len(pieces),
        n_distinct=len({piece.text for piece in pieces}),
        n_characters_covered=covered,
        share_covered=covered / characters if characters else 0.0,
    )


def _scattered(texts: tuple[str, ...]) -> list[ScatteredStem]:
    """Words the crudest rule spells more than one way, because a mark is stuck on.

    This is the failure the whole method exists to repair, counted directly: a
    run of characters with a comma on it and the same run without one are two
    different rows of any table built over the corpus.
    """
    stems: dict[str, set[str]] = {}
    for text in texts:
        for word in SPACE_RULE.split(text):
            stems.setdefault(word.text.strip(".,;:!?\"'()[]"), set()).add(word.text)
    return [
        ScatteredStem(stem=stem, spellings=sorted(spellings))
        for stem, spellings in sorted(stems.items())
        if len(spellings) > 1
    ]


def _corpus_view(texts: tuple[str, ...]) -> CorpusView:
    """One corpus read by all four rules, and the entries that separate two of them."""
    here = {piece for text in texts for piece in MOSES.split(text).texts}
    there = {piece for text in texts for piece in ANNOTATION.split(text).texts}
    hyphens = _configured("en", mark_hyphens=True, rewrite_marks=False)

    return CorpusView(
        n_texts=len(texts),
        n_characters=sum(len(text) for text in texts),
        n_non_space_characters=sum(
            1 for text in texts for character in text if not character.isspace()
        ),
        on_these_rules=_corpus_reading(texts, MOSES),
        on_annotation_rules=_corpus_reading(texts, ANNOTATION),
        on_spaces=_corpus_reading(texts, SPACE_RULE),
        on_the_table=_corpus_reading(texts, TABLE_RULE),
        only_on_these_rules=sorted(here - there),
        only_on_annotation_rules=sorted(there - here),
        scattered=_scattered(texts),
        n_pieces_with_hyphens_marked=sum(
            len(list(hyphens.split(text))) for text in texts
        ),
        n_pieces_without_letter_or_digit=sum(
            1
            for text in texts
            for piece in MOSES.split(text).texts
            if not any(character.isalnum() for character in piece)
        ),
    )


@cache
def _corpus() -> CorpusResponse:
    """Both corpora, computed once because neither takes an input."""
    return CorpusResponse(
        notebook=_corpus_view(NOTEBOOK),
        notebook_texts=list(NOTEBOOK),
        minutes=_corpus_view(MINUTES),
        minutes_texts=list(MINUTES),
        running=_split_view(RUNNING_SENTENCE, MOSES),
    )


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    """The two corpora read by all four rules, and what each setting costs in length."""
    return _corpus()
