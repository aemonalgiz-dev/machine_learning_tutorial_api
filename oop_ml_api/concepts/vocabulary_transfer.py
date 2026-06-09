"""Moving a Vocabulary.

Every endpoint here is registered under ``/concepts/moving-a-vocabulary``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about VocabularyTransfer.

Three endpoints, one per family of widgets. ``/worked`` carries the four-word
corpus the subword pages share, where two vocabularies of different sizes over
the same words make every mapping checkable by hand, and it also carries the two
crossings that break the copy rule: a WordPiece target read by a byte pair
source, and a byte-level target whose whitespace symbols name no text at all.
``/domains`` carries the pair of eighteen-sentence corpora the page's argument
rests on, with the survival counts, the sequence lengths in both directions, the
parameter counts and the sweep over target vocabulary size. ``/transfer`` is the
playground.

Everything here is a fit or a mapping the library computed. The one arithmetic
this module does for itself is multiplying a row count by an embedding width,
since the model half of the method belongs with the model that owns the table
and is deliberately absent from the library.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from enum import StrEnum
from functools import cache

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (
    BYTE_SYMBOLS,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_level import (
    ByteLevelBytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.vocabulary_transfer import (
    TokenMappings,
    VocabularyTransfer,
)
from oop_ml.core.natural_language_processing.tokenization.subword.word_piece import (
    WordPiece,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/moving-a-vocabulary", tags=["moving-a-vocabulary"])


MAX_VOCABULARY_SIZE = 300
"""The largest vocabulary a request may ask for.

Both page corpora run out of pairs worth merging well below this, so the bound
never truncates anything the page shows; it exists so a crafted request cannot
turn a teaching endpoint into a long fit.
"""

MIN_VOCABULARY_SIZE = 12
"""Below this no fit on these corpora can hold its own alphabet."""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to read."""

END_OF_WORD = "</w>"
"""The marker every byte pair fit on this page uses, written out once."""

EMBEDDING_WIDTH = 256
"""How many numbers this page gives each row of a table, for counting purposes.

Nothing here learns such a table; the width is stated so that "one row per
token" can be quoted as a count of numbers rather than as an adjective, and the
page says plainly that it chose the width.
"""

LARGE_TABLE_ROWS = 32_000
LARGE_TABLE_WIDTH = 768
"""An ordinary size for a table in a small language model, for the same purpose."""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every mapping is checkable.

The corpus the original subword paper worked its example on, and the corpus the
from-scratch transfer's own spec is pinned against, so the mappings this page
prints from it are mappings a test already holds.
"""

FOUR_WORD_CORPUS: list[str] = [
    " ".join(word for word, count in FOUR_WORD_COUNTS.items() for _ in range(count))
]

FOUR_WORD_ALPHABET = 11
"""How many symbols those four words are spelled in, with the marker attached."""

REPORT_CORPUS: list[str] = [
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
"""The old domain: eighteen sentences about reports, costs and analyses.

The same eighteen the byte pair page learns its running vocabulary from, so a
reader arriving here already knows what this vocabulary holds.
"""

KITCHEN_CORPUS: list[str] = [
    "The dough should rest for an hour.",
    "Warm the milk until it steams.",
    "Add the butter to the flour.",
    "The oven must be hot before the bread goes in.",
    "Brush the loaf with a beaten egg.",
    "The crust browns in the last ten minutes.",
    "Mix the salt through the flour first.",
    "The starter bubbles when it is ready.",
    "Turn the dough onto a board and knead it.",
    "The butter must be cold for the pastry.",
    "The cake batter thickens as it beats.",
    "Bake the loaf until it sounds hollow.",
    "The bread cools on a wire rack.",
    "Weigh the flour before the water goes in.",
    "The dough doubles in an hour in a warm kitchen.",
    "Beat the eggs and the sugar together.",
    "The kitchen smells of warm bread.",
    "Draw the pastry over the fruit and seal it.",
]
"""The new domain: eighteen sentences about bread and pastry.

Written so that every character it uses appears in the report corpus too, since
a character the old corpus never held would be unspellable for a reason that has
nothing to do with transfer and would drown the effect the page is measuring.
The two corpora share their alphabet and share almost none of their words.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries. It belongs to the old domain."""

KITCHEN_SENTENCE = "The warm dough doubled before the butter was folded in."
"""One sentence of the new domain, held out of the corpus it belongs to."""


class CorpusChoice(StrEnum):
    """Which corpus a request wants a vocabulary learned from."""

    REPORTS = "reports"
    KITCHEN = "kitchen"
    FOUR_WORDS = "four_words"


CORPORA: dict[CorpusChoice, list[str]] = {
    CorpusChoice.REPORTS: REPORT_CORPUS,
    CorpusChoice.KITCHEN: KITCHEN_CORPUS,
    CorpusChoice.FOUR_WORDS: FOUR_WORD_CORPUS,
}

SWEEP_SIZES = (52, 60, 70, 80, 90, 100, 110, 125, 137)
"""The target vocabulary sizes the survival sweep is drawn at."""

EXHAUSTING_SIZE = 137
"""A size neither eighteen-sentence corpus can fill, so both fits run out."""

WATCHED_PIECES = ("lo", "an</w>", "st</w>", "ar", "te", "in")
"""Six pieces both eighteen-sentence vocabularies hold, for the closing part.

Each survives the move by spelling; the page prints the words each one turns up
inside on either side, which is the thing spelling cannot see.
"""


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("vocabulary_size=", "a vocabulary of "),
    ("vocabulary_size", "the size asked for"),
    ("minimum_pair_frequency", "the fewest times a pair may be seen"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("the unknown token", "one token for anything unspellable"),
    ("BytePairEncoding", "this scheme"),
    ("VocabularyTransfer", "the move"),
    ("TokenMappings", "the move"),
    ("TokenMapping", "one mapped token"),
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


def _fit(corpus: list[str], vocabulary_size: int) -> BytePairEncoding:
    return BytePairEncoding(byte_fallback=False, vocabulary_size=vocabulary_size).fit(
        corpus
    )


def _corpus_pieces(model: Tokenizer, corpus: list[str]) -> int:
    """How many pieces one vocabulary reads a whole corpus in."""
    return sum(model.encode(text).n_tokens for text in corpus)


class MappingRow(BaseModel):
    """One token of the new vocabulary and the old rows that seed it.

    ``route`` is which of the three rules answered: ``stand_in`` for the token
    meaning "something else", ``copied`` for a spelling both
    vocabularies hold, ``rebuilt`` for a token that had to be decoded to text
    and read again, and ``nothing`` for a token the old vocabulary found no
    text in at all. ``gained_a_marker`` says the rebuilt spelling now claims a
    word ended where the original piece claimed nothing of the kind.
    """

    target_token: str
    target_id: int
    route: str
    decoded: str
    source_tokens: list[str]
    source_ids: list[int]
    n_source_ids: int
    uses_stand_in: bool
    gained_a_marker: bool


def _route_of(source: Tokenizer, target: Tokenizer, token: str, mapped: bool) -> str:
    if not mapped:
        return "nothing"
    if (
        token == target.vocabulary.unknown_token
        and source.vocabulary.unknown_id is not None
    ):
        return "stand_in"
    if token in source.vocabulary:
        return "copied"
    return "rebuilt"


def _rows_of(
    source: Tokenizer, target: Tokenizer, mappings: TokenMappings
) -> list[MappingRow]:
    """Every mapping, with the rule that answered it and what it lost."""
    stand_in = source.vocabulary.unknown_id
    rows: list[MappingRow] = []
    for mapping in mappings:
        source_tokens = list(source.vocabulary.tokens_of(mapping.source_ids))
        route = _route_of(source, target, mapping.target_token, mapping.is_mapped)
        rows.append(
            MappingRow(
                target_token=mapping.target_token,
                target_id=mapping.target_id,
                route=route,
                decoded=target.decode([mapping.target_id]),
                source_tokens=source_tokens,
                source_ids=list(mapping.source_ids),
                n_source_ids=mapping.n_source_ids,
                uses_stand_in=stand_in is not None and stand_in in mapping.source_ids,
                gained_a_marker=(
                    route == "rebuilt"
                    and not mapping.target_token.endswith(END_OF_WORD)
                    and bool(source_tokens)
                    and source_tokens[-1].endswith(END_OF_WORD)
                ),
            )
        )
    return rows


class MoveSummary(BaseModel):
    """How a whole vocabulary fared, counted by which rule answered each token.

    Every count from ``n_using_stand_in`` down is over the rebuilt tokens only,
    since a copied token is one row by construction and the token meaning
    "something else" is meant to map to its opposite number.
    ``n_stand_in_only`` counts the rebuilt tokens whose entire seed is that row,
    which carries nothing about the token it is standing in for, and
    ``n_inner_rebuilt`` counts the rebuilt tokens that never ended a word.
    """

    n_target_tokens: int
    n_source_tokens: int
    n_copied: int
    n_copied_multi_character: int
    n_rebuilt: int
    n_stand_in_route: int
    n_nothing: int
    n_using_stand_in: int
    n_stand_in_only: int
    n_inner_rebuilt: int
    n_gained_a_marker: int
    mean_pieces_when_rebuilt: float


def _summary_of(
    source: Tokenizer, target: Tokenizer, rows: list[MappingRow]
) -> MoveSummary:
    rebuilt = [row for row in rows if row.route == "rebuilt"]
    return MoveSummary(
        n_target_tokens=len(rows),
        n_source_tokens=source.vocabulary.n_tokens,
        n_copied=sum(1 for row in rows if row.route == "copied"),
        n_copied_multi_character=sum(
            1 for row in rows if row.route == "copied" and len(row.decoded) > 1
        ),
        n_rebuilt=len(rebuilt),
        n_stand_in_route=sum(1 for row in rows if row.route == "stand_in"),
        n_nothing=sum(1 for row in rows if row.route == "nothing"),
        n_using_stand_in=sum(1 for row in rebuilt if row.uses_stand_in),
        n_stand_in_only=sum(
            1 for row in rebuilt if row.uses_stand_in and row.n_source_ids == 1
        ),
        n_inner_rebuilt=sum(
            1 for row in rebuilt if not row.target_token.endswith(END_OF_WORD)
        ),
        n_gained_a_marker=sum(1 for row in rows if row.gained_a_marker),
        mean_pieces_when_rebuilt=(
            sum(row.n_source_ids for row in rebuilt) / len(rebuilt) if rebuilt else 0.0
        ),
    )


class Move(BaseModel):
    """One whole move: every token's mapping, and the counts over them."""

    rows: list[MappingRow]
    summary: MoveSummary


def _moved(source: Tokenizer, target: Tokenizer) -> Move:
    rows = _rows_of(source, target, VocabularyTransfer.between(source, target))
    return Move(rows=rows, summary=_summary_of(source, target, rows))


# --------------------------------------------------------------------------
# /worked: two vocabularies over four words, and two crossings.
# --------------------------------------------------------------------------


class Reading(BaseModel):
    """One text as one vocabulary reads it, under a label naming that vocabulary."""

    label: str
    pieces: list[str]
    n_pieces: int


class CrossingSummary(BaseModel):
    """A move whose two vocabularies do not share a convention."""

    label: str
    source_tokens: list[str]
    target_tokens: list[str]
    summary: MoveSummary
    rows: list[MappingRow]


class WorkedResponse(BaseModel):
    """The four-word corpus moved three ways, plus the two crossings.

    ``smaller_to_larger`` is the move a growing vocabulary asks for, where the
    old vocabulary is a prefix of the new one and nothing is lost; ``narrow``
    repeats it from a source two merges short, which is where a piece the source
    cannot spell at all appears.
    """

    corpus: list[str]
    word_counts: list[list[str]]
    small_tokens: list[str]
    large_tokens: list[str]
    narrow_tokens: list[str]
    larger_to_smaller: list[MappingRow]
    larger_to_smaller_summary: MoveSummary
    smaller_to_larger: list[MappingRow]
    smaller_to_larger_summary: MoveSummary
    narrow: list[MappingRow]
    narrow_summary: MoveSummary
    word_piece: CrossingSummary
    byte_level_n_tokens: int
    byte_level_n_nothing: int
    byte_level_nothing: list[str]
    byte_level_space_symbol: str
    byte_level_mapped_example: MappingRow


@cache
def _worked() -> WorkedResponse:
    small = _fit(FOUR_WORD_CORPUS, FOUR_WORD_ALPHABET + 1 + 3)
    large = _fit(FOUR_WORD_CORPUS, FOUR_WORD_ALPHABET + 1 + 10)
    narrow = _fit(FOUR_WORD_CORPUS, FOUR_WORD_ALPHABET + 1 + 2)

    down = _moved(large, small)
    up = _moved(small, large)
    narrowed = _moved(narrow, large)

    word_piece = WordPiece(vocabulary_size=22).fit(FOUR_WORD_CORPUS)
    crossing = _moved(large, word_piece)

    byte_level = ByteLevelBytePairEncoding(vocabulary_size=266).fit(FOUR_WORD_CORPUS)
    byte_rows = _moved(large, byte_level).rows
    mapped_example = next(row for row in byte_rows if row.target_token == "low")

    return WorkedResponse(
        corpus=FOUR_WORD_CORPUS,
        word_counts=[[word, str(count)] for word, count in FOUR_WORD_COUNTS.items()],
        small_tokens=list(small.vocabulary),
        large_tokens=list(large.vocabulary),
        narrow_tokens=list(narrow.vocabulary),
        larger_to_smaller=down.rows,
        larger_to_smaller_summary=down.summary,
        smaller_to_larger=up.rows,
        smaller_to_larger_summary=up.summary,
        narrow=narrowed.rows,
        narrow_summary=narrowed.summary,
        word_piece=CrossingSummary(
            label="A vocabulary that marks continuations rather than endings",
            source_tokens=list(large.vocabulary),
            target_tokens=list(word_piece.vocabulary),
            summary=crossing.summary,
            rows=crossing.rows,
        ),
        byte_level_n_tokens=len(byte_rows),
        byte_level_n_nothing=sum(1 for row in byte_rows if row.route == "nothing"),
        byte_level_nothing=[
            row.target_token for row in byte_rows if row.route == "nothing"
        ],
        byte_level_space_symbol=BYTE_SYMBOLS[ord(" ")],
        byte_level_mapped_example=mapped_example,
    )


@router.get("/worked", response_model=WorkedResponse)
def worked() -> WorkedResponse:
    """Two vocabularies over four words, moved in both directions."""
    return _worked()


# --------------------------------------------------------------------------
# /domains: the old vocabulary, the new corpus, and what the move costs.
# --------------------------------------------------------------------------


class LengthRow(BaseModel):
    """One text read by the old vocabulary and by the new one."""

    label: str
    old_pieces: int
    new_pieces: int


class TableCost(BaseModel):
    """A table of rows, and how many numbers that is at a stated width."""

    label: str
    rows: int
    width: int
    n_numbers: int


class SweepRow(BaseModel):
    """One target vocabulary size, and what survives a move into it."""

    asked: int
    learned: int
    n_copied: int
    n_rebuilt: int
    n_nothing: int
    new_corpus_pieces: int


class SharedPiece(BaseModel):
    """A piece both vocabularies hold, and the words it turns up inside.

    Counted with each corpus read by its own vocabulary, since that is the
    reading each model would actually see.
    """

    piece: str
    old_words: list[str]
    new_words: list[str]
    old_count: int
    new_count: int
    shared_words: list[str]


class CollidingId(BaseModel):
    """One position, and the two different tokens the two vocabularies put there."""

    token_id: int
    old_token: str
    new_token: str


class DomainsResponse(BaseModel):
    """The whole domain move, measured.

    ``old_reading`` and ``new_reading`` are the running sentence and the new
    domain's own sentence under each vocabulary, so the page can show the move
    helping on one and hurting on the other.
    """

    old_corpus: list[str]
    new_corpus: list[str]
    old_tokens: int
    new_tokens: int
    old_merges: int
    new_merges: int
    summary: MoveSummary
    rows: list[MappingRow]
    stand_in_rows: list[MappingRow]
    marker_rows: list[MappingRow]
    lengths: list[LengthRow]
    old_sentence_readings: list[Reading]
    new_sentence_readings: list[Reading]
    tables: list[TableCost]
    seeded_numbers: int
    stand_in_only_numbers: int
    sweep: list[SweepRow]
    shared_pieces: list[SharedPiece]
    colliding_ids: list[CollidingId]


def _words_holding(model: Tokenizer, corpus: list[str]) -> dict[str, Counter[str]]:
    holders: dict[str, Counter[str]] = defaultdict(Counter)
    for text in corpus:
        for word in SPLITTER.split(text).texts:
            for piece in model.encode(word).texts:
                holders[piece][word] += 1
    return holders


def _reading(model: Tokenizer, label: str, text: str) -> Reading:
    encoding = model.encode(text)
    return Reading(label=label, pieces=list(encoding.texts), n_pieces=encoding.n_tokens)


@cache
def _domains() -> DomainsResponse:
    source = _fit(REPORT_CORPUS, EXHAUSTING_SIZE)
    target = _fit(KITCHEN_CORPUS, EXHAUSTING_SIZE)
    move = _moved(source, target)

    old_holders = _words_holding(source, REPORT_CORPUS)
    new_holders = _words_holding(target, KITCHEN_CORPUS)

    sweep: list[SweepRow] = []
    for asked in SWEEP_SIZES:
        fitted = _fit(KITCHEN_CORPUS, asked)
        swept = _moved(source, fitted).summary
        sweep.append(
            SweepRow(
                asked=asked,
                learned=fitted.vocabulary.n_tokens,
                n_copied=swept.n_copied,
                n_rebuilt=swept.n_rebuilt,
                n_nothing=swept.n_nothing,
                new_corpus_pieces=_corpus_pieces(fitted, KITCHEN_CORPUS),
            )
        )

    seeded_rows = move.summary.n_target_tokens - move.summary.n_nothing
    return DomainsResponse(
        old_corpus=REPORT_CORPUS,
        new_corpus=KITCHEN_CORPUS,
        old_tokens=source.vocabulary.n_tokens,
        new_tokens=target.vocabulary.n_tokens,
        old_merges=source.n_merges,
        new_merges=target.n_merges,
        summary=move.summary,
        rows=move.rows,
        stand_in_rows=[
            row for row in move.rows if row.uses_stand_in and row.route == "rebuilt"
        ],
        marker_rows=[row for row in move.rows if row.gained_a_marker],
        lengths=[
            LengthRow(
                label="the new corpus",
                old_pieces=_corpus_pieces(source, KITCHEN_CORPUS),
                new_pieces=_corpus_pieces(target, KITCHEN_CORPUS),
            ),
            LengthRow(
                label="the old corpus",
                old_pieces=_corpus_pieces(source, REPORT_CORPUS),
                new_pieces=_corpus_pieces(target, REPORT_CORPUS),
            ),
            LengthRow(
                label="one new sentence",
                old_pieces=source.encode(KITCHEN_SENTENCE).n_tokens,
                new_pieces=target.encode(KITCHEN_SENTENCE).n_tokens,
            ),
            LengthRow(
                label="one old sentence",
                old_pieces=source.encode(RUNNING_SENTENCE).n_tokens,
                new_pieces=target.encode(RUNNING_SENTENCE).n_tokens,
            ),
        ],
        old_sentence_readings=[
            _reading(source, "the old vocabulary", RUNNING_SENTENCE),
            _reading(target, "the new vocabulary", RUNNING_SENTENCE),
        ],
        new_sentence_readings=[
            _reading(source, "the old vocabulary", KITCHEN_SENTENCE),
            _reading(target, "the new vocabulary", KITCHEN_SENTENCE),
        ],
        tables=[
            TableCost(
                label="the old table",
                rows=source.vocabulary.n_tokens,
                width=EMBEDDING_WIDTH,
                n_numbers=source.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="the new table",
                rows=target.vocabulary.n_tokens,
                width=EMBEDDING_WIDTH,
                n_numbers=target.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="a small language model",
                rows=LARGE_TABLE_ROWS,
                width=LARGE_TABLE_WIDTH,
                n_numbers=LARGE_TABLE_ROWS * LARGE_TABLE_WIDTH,
            ),
        ],
        seeded_numbers=seeded_rows * EMBEDDING_WIDTH,
        stand_in_only_numbers=move.summary.n_stand_in_only * EMBEDDING_WIDTH,
        sweep=sweep,
        shared_pieces=[
            SharedPiece(
                piece=piece,
                old_words=sorted(old_holders[piece]),
                new_words=sorted(new_holders[piece]),
                old_count=sum(old_holders[piece].values()),
                new_count=sum(new_holders[piece].values()),
                shared_words=sorted(set(old_holders[piece]) & set(new_holders[piece])),
            )
            for piece in WATCHED_PIECES
        ],
        colliding_ids=[
            CollidingId(
                token_id=token_id,
                old_token=source.vocabulary.token_of(token_id),
                new_token=target.vocabulary.token_of(token_id),
            )
            for token_id in (0, 40, 60, 80, 100, 120, 124)
        ],
    )


@router.get("/domains", response_model=DomainsResponse)
def domains() -> DomainsResponse:
    """The old vocabulary carried to the new domain, and what it cost."""
    return _domains()


# --------------------------------------------------------------------------
# /transfer: the playground.
# --------------------------------------------------------------------------


class TransferRequest(BaseModel):
    """Two corpora to learn vocabularies from, a size for each, and a text."""

    source_corpus: CorpusChoice = CorpusChoice.REPORTS
    target_corpus: CorpusChoice = CorpusChoice.KITCHEN
    source_size: int = Field(
        default=137, ge=MIN_VOCABULARY_SIZE, le=MAX_VOCABULARY_SIZE
    )
    target_size: int = Field(
        default=137, ge=MIN_VOCABULARY_SIZE, le=MAX_VOCABULARY_SIZE
    )
    text: str = Field(default=KITCHEN_SENTENCE, max_length=MAX_TEXT_LENGTH)

    model_config = {"extra": "forbid"}


class TransferResponse(BaseModel):
    """One move, with both readings of one text and every mapped token."""

    source_learned: int
    target_learned: int
    summary: MoveSummary
    rows: list[MappingRow]
    source_reading: Reading
    target_reading: Reading
    source_corpus_pieces: int
    target_corpus_pieces: int


@router.post("/transfer", response_model=TransferResponse)
def transfer(request: TransferRequest) -> TransferResponse:
    """Learn two vocabularies, move one into the other, and read one text."""
    try:
        source = _fit(CORPORA[request.source_corpus], request.source_size)
        target = _fit(CORPORA[request.target_corpus], request.target_size)
        move = _moved(source, target)
        source_reading = _reading(source, "the old vocabulary", request.text)
        target_reading = _reading(target, "the new vocabulary", request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    new_corpus = CORPORA[request.target_corpus]
    return TransferResponse(
        source_learned=source.vocabulary.n_tokens,
        target_learned=target.vocabulary.n_tokens,
        summary=move.summary,
        rows=move.rows,
        source_reading=source_reading,
        target_reading=target_reading,
        source_corpus_pieces=_corpus_pieces(source, new_corpus),
        target_corpus_pieces=_corpus_pieces(target, new_corpus),
    )
