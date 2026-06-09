"""Penn Treebank Rules.

Every endpoint here is registered under ``/concepts/penn-treebank-rules``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about PennTreebankPreTokenizer.

What this module adds to the library
------------------------------------
Nothing about the rules themselves. Every piece and every span comes out of the
library's ``PennTreebankPreTokenizer``, and the two rules it is measured against
are the library's ``WhitespacePreTokenizer`` and ``UnicodeWordPreTokenizer`` on
the same text, so the numbers here line up with the two sibling pages. What is
assembled here is the bookkeeping a page wants and a splitter has no reason to
carry: whether each piece's text is still the slice of source its span names,
how many pieces over a corpus are not, what a handful of contractions become,
what the final-period rule does when the text it is given holds more than one
sentence, and the three rules' counts over the same two corpora.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings. A test asserts that no identifier
survives into any string any endpoint returns.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from typing import Annotated

from fastapi import APIRouter
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    CLITIC_SUFFIXES,
    CONTRACTIONS,
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/penn-treebank-rules", tags=["penn-treebank-rules"])

TREEBANK = PennTreebankPreTokenizer()
"""The subject of the page, leaving a separated bracket as the character it was."""

TREEBANK_NAMING_BRACKETS = PennTreebankPreTokenizer(convert_brackets=True)
"""The same rules with the treebank's own bracket spellings switched on."""

SPACE_RULE = WhitespacePreTokenizer()
"""One of the two rules the page is measured against, throughout."""

BOUNDARY_RULE = UnicodeWordPreTokenizer()
"""The other, so the three sibling pages count over one corpus."""

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
"""The same sentence inside quotation marks, which is where the rewriting shows."""

ANNOTATED_CORPUS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "The dog didn't mind; it was asleep.",
    "Foxes, unlike dogs, are rarely asleep at noon.",
    '"Hello," she said. "It\'s 3.14, isn\'t it?"',
    "I can't believe it's not butter.",
    "They're gonna wanna see this -- trust me!",
    "The U.S. Navy (est. 1775) bought 1,000 ships at $5,000,000 each.",
    '"Don\'t," he said; "you cannot win."',
    "It costs 50% more'n I'd pay: about 3.14 dollars, give or take...",
    "Gimme a break, lemme think, d'ye mind?",
    "AT&T's shares [NYSE: T] fell 2.5%; investors weren't pleased!",
    "'Tis the season, 'twas the night.",
    "The dogs' bowls were empty (again).",
    "Well-known authors' books sell; unknown ones' don't.",
    "``Quoted'' with typewriter quotes, and \"quoted\" with straight ones.",
    "Is it 3:30 yet?!",
    'He whispered: "wait..." and then "go!"',
    "What?? No!! Really?!",
    "Prices: $1, $2.50 and $3,000.",
    "I'LL DO IT, I'M SURE, THEY'VE SAID SO.",
    "e-mail me at a@b.com #now",
    "She said (quietly) that it's fine.",
)
"""The sentences the rules are checked against, one clause of the rules each.

Chosen so that every rule on the page is exercised at least once, which is what
makes the rewriting census here a count over writing the rules were meant for
rather than over writing picked to flatter them.
"""

CLITIC_PIECES = frozenset(
    {"n't", "'s", "'m", "'d", "'ll", "'re", "'ve", "'", "not", "na", "me", "ta"}
)
"""Pieces that exist only because a host was cut, in either case.

Counted so the page can say how much of the extra length is the contraction rule
rather than the punctuation rule. ``not``, ``na``, ``me`` and ``ta`` are the
halves the fixed list produces and are also ordinary words, so the count is an
upper bound and the page says so.
"""

MAX_TEXTS = 4
"""How many texts one request may split. The widgets ask for one at a time."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


class Piece(BaseModel):
    """One piece, the span it claims, and whether its text is still that slice.

    ``source`` is what the span actually covers in the text. ``is_rewritten``
    says the two differ, which happens for a quotation mark and, when they are
    switched on, for a bracket.
    """

    text: str
    start: int
    end: int
    source: str
    is_rewritten: bool


class RuleView(BaseModel):
    """One rule's reading of one text, and the three things the page asks of it.

    ``glued`` is the pieces joined with one space each, which is the only glue
    available to anything holding the pieces alone. ``spans_recover_the_source``
    says whether reading the spans back out of the text, rather than the pieces,
    reproduces everything in it that is not white space.
    """

    pieces: list[Piece]
    n_pieces: int
    n_characters_in_pieces: int
    share_covered: float
    glued: str
    glue_is_exact: bool
    n_rewritten: int
    spans_recover_the_source: bool


class SplitView(BaseModel):
    """One text read by the treebank rules, with the two other rules beside it."""

    source: str
    n_characters: int
    n_non_space_characters: int
    on_treebank: RuleView
    on_treebank_naming_brackets: RuleView
    on_spaces: RuleView
    on_boundaries: RuleView


class SplitRequest(BaseModel):
    """Texts to read. The rules take one setting, and it is asked for separately."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)

    model_config = {"extra": "forbid"}


class SplitResponse(BaseModel):
    """Every text of the request, seen through all three rules."""

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
        )
        for word in rule.split(text)
    ]


def _rule_view(text: str, rule: PreTokenizer) -> RuleView:
    """How one rule reads one text, and what the two ways back give."""
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
        spans_recover_the_source=(
            "".join(piece.source for piece in pieces) == without_space
        ),
    )


def _split_view(text: str) -> SplitView:
    """One text taken all the way through, by the three rules the section uses."""
    return SplitView(
        source=text,
        n_characters=len(text),
        n_non_space_characters=sum(1 for character in text if not character.isspace()),
        on_treebank=_rule_view(text, TREEBANK),
        on_treebank_naming_brackets=_rule_view(text, TREEBANK_NAMING_BRACKETS),
        on_spaces=_rule_view(text, SPACE_RULE),
        on_boundaries=_rule_view(text, BOUNDARY_RULE),
    )


@router.post("/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    """Read every text of the request by all three rules and report everything between."""
    return SplitResponse(texts=[_split_view(text) for text in request.texts])


class ContractionCase:
    """One short text, the label the page gives it, and what to look at."""

    __slots__ = ("label", "look_at", "text")

    def __init__(self, label: str, text: str, look_at: str) -> None:
        self.label = label
        self.text = text
        self.look_at = look_at


CONTRACTION_CASES: tuple[ContractionCase, ...] = (
    ContractionCase(
        "the ordinary case",
        "didn't",
        "two pieces, and both of them are still exactly the letters they came from",
    ),
    ContractionCase(
        "a host that is not a word",
        "can't",
        "the negative morpheme is three characters, so what is left of the host is two",
    ),
    ContractionCase(
        "another of the same",
        "won't",
        "the same cut, and the host is even further from the word it stands for",
    ),
    ContractionCase(
        "and a third",
        "shan't",
        "there is no rule here that knows what the host was meant to be",
    ),
    ContractionCase(
        "a possessive",
        "the dogs' bowls",
        "a lone apostrophe with a host in front of it comes off as a piece of its own",
    ),
    ContractionCase(
        "no apostrophe at all",
        "cannot",
        "cut although nothing in the writing marks the join",
    ),
    ContractionCase(
        "a spoken form on the fixed list",
        "gonna",
        "the same treatment, and the two halves are not words either",
    ),
    ContractionCase(
        "cut at the front rather than the end",
        "'twas the night",
        "the fixed list holds two forms that begin a piece rather than ending it",
    ),
    ContractionCase(
        "inside quotation marks",
        '"It\'s fine,"',
        "the two quotation marks are the only pieces here whose text is not the source",
    ),
    ContractionCase(
        "the same contraction, typed with a curved apostrophe",
        "didn’t",
        "one piece, because the rules were written when there was one apostrophe",
    ),
    ContractionCase(
        "a French article",
        "l'analyse",
        "one piece, where a reader of French would want two",
    ),
    ContractionCase(
        "an Italian article",
        "dell'analisi",
        "one piece again, for the same reason",
    ),
    ContractionCase(
        "a spoken form nobody added",
        "gotcha",
        "one piece, and nothing anywhere reports that the list was consulted",
    ),
    ContractionCase(
        "and another",
        "y'all",
        "one piece, though the apostrophe is doing exactly what it does elsewhere",
    ),
    ContractionCase(
        "an apostrophe that is not a join at all",
        "o'clock",
        "one piece, which is the right answer reached by the same silence",
    ),
)
"""Contractions the rules cut, contractions they leave whole, and the boundary.

The order runs from the case the rules were built for to the cases nobody wrote
a clause for, since the page's argument is that the second set is silent.
"""


class ContractionProbe(BaseModel):
    """One short text, its pieces, and whether any piece was rewritten."""

    label: str
    text: str
    pieces: list[Piece]
    n_pieces: int
    is_cut: bool
    n_rewritten: int
    look_at: str


class ContractionsResponse(BaseModel):
    """The contraction probes, and how many of them the rules cut at all."""

    probes: list[ContractionProbe]
    n_probes: int
    n_cut: int
    n_left_whole: int
    n_fixed_forms: int
    n_clitic_endings: int


N_FIXED_FORMS = len(CONTRACTIONS)
"""How many whole-word forms the rules carry by name, counted from the rules."""

N_CLITIC_ENDINGS = len({suffix.lower() for suffix in CLITIC_SUFFIXES})
"""How many different endings come off the end of a run, counted once per case."""


@cache
def _contractions() -> ContractionsResponse:
    """The contraction probes, computed once because none of them takes an input."""
    probes: list[ContractionProbe] = []
    for case in CONTRACTION_CASES:
        pieces = _pieces_of(case.text, TREEBANK)
        probes.append(
            ContractionProbe(
                label=case.label,
                text=case.text,
                pieces=pieces,
                n_pieces=len(pieces),
                is_cut=len(pieces) > len(case.text.split()),
                n_rewritten=sum(1 for piece in pieces if piece.is_rewritten),
                look_at=case.look_at,
            )
        )

    return ContractionsResponse(
        probes=probes,
        n_probes=len(probes),
        n_cut=sum(1 for probe in probes if probe.is_cut),
        n_left_whole=sum(1 for probe in probes if not probe.is_cut),
        n_fixed_forms=N_FIXED_FORMS,
        n_clitic_endings=N_CLITIC_ENDINGS,
    )


@router.get("/contractions", response_model=ContractionsResponse)
def contractions() -> ContractionsResponse:
    """What the rules do to a contraction, and which contractions they never see."""
    return _contractions()


FULL_STOP_CASES: tuple[ContractionCase, ...] = (
    ContractionCase(
        "one sentence, an abbreviation inside it",
        RUNNING_SENTENCE,
        "the abbreviation keeps its stop and the sentence loses its own",
    ),
    ContractionCase(
        "an abbreviation written with two stops",
        "The U.S. Navy arrived.",
        "both interior stops stay, because neither of them ends the text",
    ),
    ContractionCase(
        "two sentences in one text",
        "The drift was small. It did not go away.",
        "the first sentence keeps its stop and becomes an entry of its own",
    ),
    ContractionCase(
        "a sentence that ends in an abbreviation",
        "She consulted Dr.",
        "the abbreviation loses the stop that belongs to it",
    ),
    ContractionCase(
        "and the same with a longer abbreviation",
        "The analysis was run by Alvarez et al.",
        "the same loss, and nothing here could tell the two jobs apart",
    ),
    ContractionCase(
        "a stop inside a number",
        "It came to 3.14 dollars.",
        "the interior stop stays and the final one comes off",
    ),
    ContractionCase(
        "a comma inside a number",
        "It cost 1,000 and took 3:30.",
        "neither the comma nor the colon comes off, because a digit follows each",
    ),
    ContractionCase(
        "a comma and a colon with a letter after them",
        "note: yes, no",
        "both come off, from the same rule reading the next character",
    ),
)
"""The final-period rule, and the two ways a text can defeat it.

The rule is a statement about position rather than about vocabulary, so it is
right exactly when the text it was handed is one sentence.
"""


class FullStopProbe(BaseModel):
    """One text, its pieces, and which of them still carry an attached stop."""

    label: str
    text: str
    pieces: list[Piece]
    n_pieces: int
    attached_stops: list[str]
    has_own_stop_piece: bool
    look_at: str


class SentenceCountView(BaseModel):
    """The same six sentences counted two ways, one at a time and as one text.

    ``separately`` is what the rules give when each sentence arrives on its own,
    which is the way the rules were run. ``together`` is what they give when the
    six arrive joined, and the difference is the whole of the rule's assumption.
    """

    separately: int
    together: int
    attached_stops_separately: list[str]
    attached_stops_together: list[str]
    stop_pieces_separately: int
    stop_pieces_together: int


class FullStopsResponse(BaseModel):
    """The final-period rule, probed and then counted over the six sentences."""

    probes: list[FullStopProbe]
    sentences: SentenceCountView


def _attached_stops(pieces: list[Piece]) -> list[str]:
    """Pieces that end in a full stop and are more than a full stop."""
    return [
        piece.text
        for piece in pieces
        if piece.text.endswith(".") and piece.text.rstrip(".") != ""
    ]


@cache
def _full_stops() -> FullStopsResponse:
    """The period probes and the six-sentence count, computed once."""
    probes: list[FullStopProbe] = []
    for case in FULL_STOP_CASES:
        pieces = _pieces_of(case.text, TREEBANK)
        probes.append(
            FullStopProbe(
                label=case.label,
                text=case.text,
                pieces=pieces,
                n_pieces=len(pieces),
                attached_stops=_attached_stops(pieces),
                has_own_stop_piece=any(piece.text == "." for piece in pieces),
                look_at=case.look_at,
            )
        )

    joined = " ".join(NOTEBOOK)
    together = _pieces_of(joined, TREEBANK)
    separately = [piece for text in NOTEBOOK for piece in _pieces_of(text, TREEBANK)]

    return FullStopsResponse(
        probes=probes,
        sentences=SentenceCountView(
            separately=len(separately),
            together=len(together),
            attached_stops_separately=_attached_stops(separately),
            attached_stops_together=_attached_stops(together),
            stop_pieces_separately=sum(1 for piece in separately if piece.text == "."),
            stop_pieces_together=sum(1 for piece in together if piece.text == "."),
        ),
    )


@router.get("/full-stops", response_model=FullStopsResponse)
def full_stops() -> FullStopsResponse:
    """Where the final stop goes, and what happens when the text is not one sentence."""
    return _full_stops()


class CorpusReading(BaseModel):
    """What one rule spends over one corpus.

    ``n_distinct`` is how many different pieces the corpus produced, so it is how
    many entries a table over it would need, and ``n_pieces`` is how many pieces
    it produced in all, so it is what anything reading the corpus pays per pass.
    """

    n_pieces: int
    n_distinct: int
    n_characters_covered: int
    share_covered: float


class CorpusView(BaseModel):
    """One corpus read by all three rules, with what separates them."""

    n_texts: int
    n_characters: int
    n_non_space_characters: int
    on_treebank: CorpusReading
    on_spaces: CorpusReading
    on_boundaries: CorpusReading
    only_on_treebank_against_spaces: list[str]
    only_on_spaces: list[str]
    shared_with_spaces: int
    only_on_treebank_against_boundaries: list[str]
    only_on_boundaries: list[str]
    shared_with_boundaries: int


class RewritingCensus(BaseModel):
    """How many pieces of a corpus no longer hold the text their span names.

    ``by_piece`` lists each rewritten spelling with how often it turned up and
    what one source character it stands for.
    """

    n_pieces: int
    n_rewritten: int
    share_rewritten: float
    by_piece: list[str]
    n_rewritten_naming_brackets: int
    n_pieces_without_letter_or_digit: int
    n_pieces_from_a_cut_host: int


class CorpusResponse(BaseModel):
    """Both corpora, all three rules, and the cost of the rewriting."""

    notebook: CorpusView
    notebook_texts: list[str]
    annotated: CorpusView
    census: RewritingCensus
    running: SplitView
    quoted: SplitView


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


def _entries(texts: tuple[str, ...], rule: PreTokenizer) -> set[str]:
    """The different pieces one rule finds over one corpus."""
    return {piece for text in texts for piece in rule.split(text).texts}


def _corpus_view(texts: tuple[str, ...]) -> CorpusView:
    """One corpus read by all three rules, and the entries that separate them."""
    treebank = _entries(texts, TREEBANK)
    spaces = _entries(texts, SPACE_RULE)
    boundaries = _entries(texts, BOUNDARY_RULE)

    return CorpusView(
        n_texts=len(texts),
        n_characters=sum(len(text) for text in texts),
        n_non_space_characters=sum(
            1 for text in texts for character in text if not character.isspace()
        ),
        on_treebank=_corpus_reading(texts, TREEBANK),
        on_spaces=_corpus_reading(texts, SPACE_RULE),
        on_boundaries=_corpus_reading(texts, BOUNDARY_RULE),
        only_on_treebank_against_spaces=sorted(treebank - spaces),
        only_on_spaces=sorted(spaces - treebank),
        shared_with_spaces=len(treebank & spaces),
        only_on_treebank_against_boundaries=sorted(treebank - boundaries),
        only_on_boundaries=sorted(boundaries - treebank),
        shared_with_boundaries=len(treebank & boundaries),
    )


@cache
def _corpus() -> CorpusResponse:
    """Both corpora and the rewriting census, computed once because none takes input."""
    pieces = [
        piece for text in ANNOTATED_CORPUS for piece in _pieces_of(text, TREEBANK)
    ]
    rewritten = Counter(
        (piece.text, piece.source) for piece in pieces if piece.is_rewritten
    )
    naming_brackets = [
        piece
        for text in ANNOTATED_CORPUS
        for piece in _pieces_of(text, TREEBANK_NAMING_BRACKETS)
    ]

    return CorpusResponse(
        notebook=_corpus_view(NOTEBOOK),
        notebook_texts=list(NOTEBOOK),
        annotated=_corpus_view(ANNOTATED_CORPUS),
        census=RewritingCensus(
            n_pieces=len(pieces),
            n_rewritten=sum(1 for piece in pieces if piece.is_rewritten),
            share_rewritten=(
                sum(1 for piece in pieces if piece.is_rewritten) / len(pieces)
                if pieces
                else 0.0
            ),
            by_piece=[
                f"{spelling} stands for {source}, {count} times"
                for (spelling, source), count in rewritten.most_common()
            ],
            n_rewritten_naming_brackets=sum(
                1 for piece in naming_brackets if piece.is_rewritten
            ),
            n_pieces_without_letter_or_digit=sum(
                1
                for piece in pieces
                if not any(character.isalnum() for character in piece.text)
            ),
            n_pieces_from_a_cut_host=sum(
                1 for piece in pieces if piece.text.lower() in CLITIC_PIECES
            ),
        ),
        running=_split_view(RUNNING_SENTENCE),
        quoted=_split_view(QUOTED_SENTENCE),
    )


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    """The two corpora read by all three rules, and what the rewriting costs."""
    return _corpus()
