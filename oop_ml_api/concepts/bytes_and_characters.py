"""Bytes and Characters.

Every endpoint here is registered under ``/concepts/bytes-and-characters``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about ByteTokenizer, CharacterTokenizer.

What this module computes
-------------------------
Five endpoints, one per family of widgets rather than one per number.
``/readings`` puts a text to all three readings the page compares, the 256 byte
values, the characters an English corpus used, and a vocabulary learned by
merging over that same corpus, and records what each of them made of it and what
came back when the numbers were read. ``/scripts`` reads one short sentence
written in twelve languages, which is where the two readings stop agreeing about
length. ``/alphabet-growth`` lets those twelve arrive one at a time so the
character table can be watched growing beside a byte table that cannot.
``/byte-runs`` measures how narrow validity is for a run of byte numbers, which
is what a model emitting numbers one at a time walks into. ``/limits`` carries
the figures the closing part rests on.

The learned vocabulary is the library's byte pair encoder fitted on the same
eighteen sentences the byte pair encoding page uses, so the two pages' numbers
can be laid beside each other.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a refusal needs,
and a test asserts that no identifier survives it.
"""

from __future__ import annotations

import random
import unicodedata
from functools import cache
from typing import Annotated, Literal

from fastapi import APIRouter
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.characters.byte import (
    ByteTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.characters.character import (
    CharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/bytes-and-characters", tags=["bytes-and-characters"]
)


# --------------------------------------------------------------------------
# The texts every endpoint works from.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

ACCENTED_TEXT = "Dr. Alvarez didn't expect the low-cost re-analysis in Ångström units."
"""The running sentence with two letters no English corpus here contains."""

GREEK_TEXT = "Η έκθεση έφτασε αργά."
"""One short sentence in a script none of the eighteen sentences uses."""

CORPUS: tuple[str, ...] = (
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
)
"""Eighteen sentences of ordinary English, none of them the running sentence.

The same corpus the byte pair encoding page is built on, so that the learned
vocabulary this page compares against is the one that page describes and the
two sets of numbers can be read together.
"""

LANGUAGES: tuple[tuple[str, str], ...] = (
    ("English", "The report arrived late."),
    ("French", "Le rapport est arrivé en retard."),
    ("German", "Der Bericht kam zu spät."),
    ("Spanish", "El informe llegó tarde."),
    ("Polish", "Raport dotarł późno."),
    ("Greek", "Η έκθεση έφτασε αργά."),
    ("Russian", "Отчёт пришёл поздно."),
    ("Arabic", "وصل التقرير متأخرا."),
    ("Hindi", "रिपोर्ट देर से आई।"),
    ("Korean", "보고서가 늦게 도착했다."),
    ("Japanese", "報告書は遅れて届いた。"),
    ("Chinese", "报告迟到了。"),
)
"""One short sentence with one meaning, written in twelve languages.

The ratio between the two readings is a fact about the writing system rather
than about the wording, which is why the same sentence is used throughout: the
only thing that changes down the list is the script it is written in.
"""

WELSH_FLAG = "\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f"
"""A black flag followed by six invisible tag characters spelling gbwls."""

REPLACEMENT = "�"
"""What a run of bytes that is not text decodes to, one per broken position."""

EMBEDDING_WIDTH = 768
"""An ordinary width for the row a model gives each entry of a table."""

UNICODE_CODEPOINTS = 0x110000
"""How many codepoints Unicode defines room for."""

MAX_TEXTS = 6
"""How many texts one request may read. The widgets ask for at most three."""

MAX_TEXT_CHARACTERS = 300
"""How long any one text may be, a few times the running sentence."""

RANDOM_RUN_LENGTHS: tuple[int, ...] = (2, 4, 8, 16)
"""The run lengths the validity measurement is taken at."""

RANDOM_RUN_TRIALS = 2000
"""How many runs are drawn at each length. Seeded, so the answer is repeatable."""

BoundedText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)]

Reading = Literal["bytes", "characters", "learned"]

READING_LABELS: dict[Reading, str] = {
    "bytes": "every byte value",
    "characters": "the characters the corpus used",
    "learned": "pieces merged from the corpus",
}
"""What each reading is called on the page. No identifier appears in one."""


# --------------------------------------------------------------------------
# The three readings, built once each.
# --------------------------------------------------------------------------


@cache
def _byte_reading() -> ByteTokenizer:
    """The reading whose table is the 256 byte values and needs no corpus."""
    return ByteTokenizer()


@cache
def _character_reading(corpus: tuple[str, ...] = CORPUS) -> CharacterTokenizer:
    """The reading whose table is every character ``corpus`` used."""
    return CharacterTokenizer().fit(list(corpus))


@cache
def _learned_reading() -> BytePairEncoding:
    """The reading whose pieces were merged out of the eighteen sentences."""
    return BytePairEncoding(byte_fallback=False, vocabulary_size=300).fit(list(CORPUS))


def _tokenizer(reading: Reading) -> Tokenizer:
    """Whichever of the three readings was asked for."""
    if reading == "bytes":
        return _byte_reading()
    if reading == "characters":
        return _character_reading()
    return _learned_reading()


PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("token", "entry"),
    ("vocabulary", "table"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the vocabulary of the code taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _has_a_byte_spelling(text: str) -> bool:
    """Whether ``text`` can be written as bytes at all.

    A surrogate codepoint has no encoding, so a string carrying one is not text
    and there is nothing for either reading to read. Requests never reach that
    case, since a body carrying a lone surrogate is refused as it is parsed, but
    it is a fact about the method the closing part of the page quotes.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


# --------------------------------------------------------------------------
# Reading one text three ways.
# --------------------------------------------------------------------------


class PieceView(BaseModel):
    """One piece of a read text, with the number it goes in as."""

    text: str
    token_id: int
    known: bool


class ReadingView(BaseModel):
    """What one reading made of one text, and what came back.

    ``exact`` compares the decoded string against the source character for
    character, so it is false whenever anything at all was lost.
    """

    reading: Reading
    label: str
    table_size: int
    n_tokens: int
    n_unseen: int
    pieces: list[PieceView]
    ids: list[int]
    decoded: str
    exact: bool


class TextReadings(BaseModel):
    """One text seen through all three readings at once."""

    source: str
    n_characters: int
    n_bytes: int
    readings: list[ReadingView]


class ReadingsRequest(BaseModel):
    """Texts to read every way."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)

    model_config = {"extra": "forbid"}


class ReadingsResponse(BaseModel):
    """Every text under every reading, with the three table sizes."""

    corpus: list[str]
    byte_table: int
    character_table: int
    learned_table: int
    n_distinct_characters: int
    n_merges: int
    texts: list[TextReadings]


def _read(reading: Reading, text: str) -> ReadingView:
    """Read one text one way, then read the numbers straight back again."""
    tokenizer = _tokenizer(reading)
    vocabulary = tokenizer.vocabulary
    stand_in = vocabulary.unknown_token
    encoding = tokenizer.encode(text)
    decoded = tokenizer.decode(encoding.ids)
    return ReadingView(
        reading=reading,
        label=READING_LABELS[reading],
        table_size=vocabulary.n_tokens,
        n_tokens=encoding.n_tokens,
        n_unseen=sum(1 for piece in encoding if piece.text == stand_in),
        pieces=[
            PieceView(
                text=piece.text,
                token_id=piece.token_id,
                known=piece.text != stand_in,
            )
            for piece in encoding
        ],
        ids=list(encoding.ids),
        decoded=decoded,
        exact=decoded == text,
    )


def _text_readings(text: str) -> TextReadings:
    """One text under all three readings."""
    return TextReadings(
        source=text,
        n_characters=len(text),
        n_bytes=len(text.encode("utf-8")),
        readings=[
            _read(reading, text) for reading in ("bytes", "characters", "learned")
        ],
    )


@router.post("/readings", response_model=ReadingsResponse)
def readings(request: ReadingsRequest) -> ReadingsResponse:
    """Read every text three ways and read the numbers back again."""
    characters = _character_reading()
    learned = _learned_reading()
    return ReadingsResponse(
        corpus=list(CORPUS),
        byte_table=_byte_reading().vocabulary.n_tokens,
        character_table=characters.vocabulary.n_tokens,
        learned_table=learned.vocabulary.n_tokens,
        n_distinct_characters=characters.n_characters,
        n_merges=learned.n_merges,
        texts=[_text_readings(text) for text in request.texts],
    )


# --------------------------------------------------------------------------
# One sentence in twelve languages.
# --------------------------------------------------------------------------


class ScriptRow(BaseModel):
    """One language's sentence, measured under each reading.

    ``bytes_per_character`` is the ratio the two readings disagree by, which is
    a property of the writing system. ``n_bytes`` is what the byte reading
    actually charges for this sentence, which is a different question and does
    not follow from the ratio.
    """

    language: str
    text: str
    n_characters: int
    n_bytes: int
    bytes_per_character: float
    byte_exact: bool
    character_tokens: int
    character_unseen: int
    learned_tokens: int
    learned_unseen: int


class ScriptsResponse(BaseModel):
    """The same sentence in twelve languages, read every way."""

    rows: list[ScriptRow]
    byte_table: int
    character_table: int
    learned_table: int


@cache
def _script_rows() -> tuple[ScriptRow, ...]:
    """The twelve sentences measured, computed once."""
    byte_reading = _byte_reading()
    characters = _character_reading()
    learned = _learned_reading()
    rows: list[ScriptRow] = []
    for language, text in LANGUAGES:
        byte_encoding = byte_reading.encode(text)
        character_encoding = characters.encode(text)
        learned_encoding = learned.encode(text)
        rows.append(
            ScriptRow(
                language=language,
                text=text,
                n_characters=len(text),
                n_bytes=byte_encoding.n_tokens,
                bytes_per_character=byte_encoding.n_tokens / len(text),
                byte_exact=byte_reading.decode(byte_encoding.ids) == text,
                character_tokens=character_encoding.n_tokens,
                character_unseen=sum(
                    1
                    for piece in character_encoding
                    if piece.text == characters.vocabulary.unknown_token
                ),
                learned_tokens=learned_encoding.n_tokens,
                learned_unseen=sum(
                    1
                    for piece in learned_encoding
                    if piece.text == learned.vocabulary.unknown_token
                ),
            )
        )
    return tuple(rows)


@router.get("/scripts", response_model=ScriptsResponse)
def scripts() -> ScriptsResponse:
    """One sentence in twelve languages, and what each reading charges for it."""
    return ScriptsResponse(
        rows=list(_script_rows()),
        byte_table=_byte_reading().vocabulary.n_tokens,
        character_table=_character_reading().vocabulary.n_tokens,
        learned_table=_learned_reading().vocabulary.n_tokens,
    )


# --------------------------------------------------------------------------
# The table that grows and the table that cannot.
# --------------------------------------------------------------------------


class GrowthStep(BaseModel):
    """The two tables after one more language has arrived.

    ``byte_values_used`` is how many of the 256 rows any text so far has
    touched, which is not the same as how many rows the byte table has.
    """

    label: str
    n_texts: int
    character_table: int
    byte_table: int
    byte_values_used: int


class GrowthResponse(BaseModel):
    """Twelve languages arriving one at a time, with both tables after each."""

    steps: list[GrowthStep]


@router.get("/alphabet-growth", response_model=GrowthResponse)
def alphabet_growth() -> GrowthResponse:
    """Grow a character table one language at a time beside a fixed byte table."""
    byte_table = _byte_reading().vocabulary.n_tokens
    texts = list(CORPUS)
    used: set[int] = set()
    for text in texts:
        used.update(text.encode("utf-8"))

    steps = [
        GrowthStep(
            label="eighteen English sentences",
            n_texts=len(texts),
            character_table=_character_reading(tuple(texts)).n_characters,
            byte_table=byte_table,
            byte_values_used=len(used),
        )
    ]
    for language, sentence in LANGUAGES:
        texts.append(sentence)
        used.update(sentence.encode("utf-8"))
        steps.append(
            GrowthStep(
                label=language,
                n_texts=len(texts),
                character_table=_character_reading(tuple(texts)).n_characters,
                byte_table=byte_table,
                byte_values_used=len(used),
            )
        )
    return GrowthResponse(steps=steps)


# --------------------------------------------------------------------------
# A run of byte numbers that is not text.
# --------------------------------------------------------------------------


class PrefixStep(BaseModel):
    """The first ``n_ids`` numbers of one reading, and what they decode to."""

    n_ids: int
    decoded: str
    is_text: bool


class RunView(BaseModel):
    """One sentence's byte reading, stopped at every position in turn."""

    language: str
    text: str
    n_ids: int
    n_broken: int
    prefixes: list[PrefixStep]


class RandomRun(BaseModel):
    """How many drawn runs of ``length`` numbers turned out to be text."""

    length: int
    n_text: int
    n_trials: int


class ByteRunsResponse(BaseModel):
    """How narrow validity is for a run of byte numbers.

    ``single_ids_that_are_text`` counts the numbers that stand for a character
    on their own; the rest are continuation numbers, which mean nothing without
    the number before them.
    """

    single_ids_that_are_text: int
    single_ids_that_are_not: int
    table_size: int
    accented_ids: list[int]
    accented_whole: str
    accented_first_alone: str
    accented_second_alone: str
    random_runs: list[RandomRun]
    runs: list[RunView]


def _run_view(language: str, text: str) -> RunView:
    """One text's byte numbers, cut at every position."""
    byte_reading = _byte_reading()
    ids = list(byte_reading.encode(text).ids)
    prefixes: list[PrefixStep] = []
    for cut in range(1, len(ids) + 1):
        decoded = byte_reading.decode(ids[:cut])
        prefixes.append(
            PrefixStep(n_ids=cut, decoded=decoded, is_text=REPLACEMENT not in decoded)
        )
    return RunView(
        language=language,
        text=text,
        n_ids=len(ids),
        n_broken=sum(1 for prefix in prefixes if not prefix.is_text),
        prefixes=prefixes,
    )


@cache
def _random_runs() -> tuple[RandomRun, ...]:
    """Drawn runs of numbers, counted for how many are text. Seeded once."""
    byte_reading = _byte_reading()
    drawn: list[RandomRun] = []
    for length in RANDOM_RUN_LENGTHS:
        generator = random.Random(0)
        clean = 0
        for _ in range(RANDOM_RUN_TRIALS):
            ids = [generator.randrange(256) for _ in range(length)]
            if REPLACEMENT not in byte_reading.decode(ids):
                clean += 1
        drawn.append(RandomRun(length=length, n_text=clean, n_trials=RANDOM_RUN_TRIALS))
    return tuple(drawn)


@router.get("/byte-runs", response_model=ByteRunsResponse)
def byte_runs() -> ByteRunsResponse:
    """Where a run of byte numbers stops standing for text at all."""
    byte_reading = _byte_reading()
    table_size = byte_reading.vocabulary.n_tokens
    alone = sum(
        1
        for value in range(table_size)
        if REPLACEMENT not in byte_reading.decode([value])
    )
    accented = list(byte_reading.encode("é").ids)
    return ByteRunsResponse(
        single_ids_that_are_text=alone,
        single_ids_that_are_not=table_size - alone,
        table_size=table_size,
        accented_ids=accented,
        accented_whole=byte_reading.decode(accented),
        accented_first_alone=byte_reading.decode(accented[:1]),
        accented_second_alone=byte_reading.decode(accented[1:]),
        random_runs=list(_random_runs()),
        runs=[
            _run_view("Greek", GREEK_TEXT),
            _run_view("English", LANGUAGES[0][1]),
        ],
    )


# --------------------------------------------------------------------------
# The figures the closing part rests on.
# --------------------------------------------------------------------------


class TableCost(BaseModel):
    """One table, its rows, and what those rows cost at an ordinary width."""

    label: str
    rows: int
    numbers: int


class WidthSample(BaseModel):
    """One character and how many bytes it takes."""

    label: str
    text: str
    n_bytes: int


class ClusterSample(BaseModel):
    """One thing a reader would call a single mark, counted both ways."""

    label: str
    text: str
    n_characters: int
    n_bytes: int
    byte_exact: bool


class SpellingSample(BaseModel):
    """One word spelled two ways that look identical on the page."""

    label: str
    text: str
    codepoints: list[str]
    n_characters: int
    n_bytes: int
    character_unseen: int
    byte_exact: bool


class EmptyCase(BaseModel):
    """A text with little or nothing in it, read both ways."""

    label: str
    byte_tokens: int
    character_tokens: int


class LimitsResponse(BaseModel):
    """Everything the closing part quotes, computed rather than asserted."""

    codepoints_in_unicode: int
    embedding_width: int
    bytes_that_keep_their_own_shape: int
    bytes_given_a_borrowed_shape: int
    tables: list[TableCost]
    widths: list[WidthSample]
    clusters: list[ClusterSample]
    spellings: list[SpellingSample]
    spellings_equal: bool
    empty_cases: list[EmptyCase]
    out_of_range_refusal: str
    lone_surrogate_has_a_byte_spelling: bool
    corpus_byte_values_used: int
    all_texts_byte_values_used: int
    corpus_bytes: int
    corpus_characters: int
    corpus_learned: int


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The measured facts the closing part of the page rests on."""
    byte_reading = _byte_reading()
    characters = _character_reading()
    learned = _learned_reading()
    stand_in = characters.vocabulary.unknown_token

    try:
        byte_reading.decode([byte_reading.vocabulary.n_tokens])
        refusal_sentence = ""
    except MLLibError as refusal:
        refusal_sentence = _in_plain_words(refusal)

    corpus_used: set[int] = set()
    for text in CORPUS:
        corpus_used.update(text.encode("utf-8"))
    all_used = set(corpus_used)
    for _, sentence in LANGUAGES:
        all_used.update(sentence.encode("utf-8"))

    spellings: list[SpellingSample] = []
    for label, text in (
        ("one entry for the accented letter", "café"),
        ("a letter and a mark on it", unicodedata.normalize("NFD", "café")),
    ):
        encoding = characters.encode(text)
        spellings.append(
            SpellingSample(
                label=label,
                text=text,
                codepoints=[f"U+{ord(character):04X}" for character in text],
                n_characters=len(text),
                n_bytes=len(text.encode("utf-8")),
                character_unseen=sum(1 for piece in encoding if piece.text == stand_in),
                byte_exact=byte_reading.decode(byte_reading.encode(text).ids) == text,
            )
        )

    clusters = [
        ClusterSample(
            label=label,
            text=text,
            n_characters=len(text),
            n_bytes=len(text.encode("utf-8")),
            byte_exact=byte_reading.decode(byte_reading.encode(text).ids) == text,
        )
        for label, text in (
            ("a raised thumb", "\U0001f44d"),
            ("the same thumb, tinted", "\U0001f44d\U0001f3fd"),
            ("a family", "\U0001f468‍\U0001f469‍\U0001f467"),
            ("the flag of Wales", WELSH_FLAG),
        )
    ]

    keeps_its_shape = sum(
        1
        for value in range(byte_reading.vocabulary.n_tokens)
        if byte_reading.vocabulary.token_of(value) == chr(value)
    )

    return LimitsResponse(
        codepoints_in_unicode=UNICODE_CODEPOINTS,
        embedding_width=EMBEDDING_WIDTH,
        bytes_that_keep_their_own_shape=keeps_its_shape,
        bytes_given_a_borrowed_shape=(
            byte_reading.vocabulary.n_tokens - keeps_its_shape
        ),
        tables=[
            TableCost(
                label="every byte value",
                rows=byte_reading.vocabulary.n_tokens,
                numbers=byte_reading.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="the characters the corpus used",
                rows=characters.vocabulary.n_tokens,
                numbers=characters.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="pieces merged from the corpus",
                rows=learned.vocabulary.n_tokens,
                numbers=learned.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="one row per codepoint",
                rows=UNICODE_CODEPOINTS,
                numbers=UNICODE_CODEPOINTS * EMBEDDING_WIDTH,
            ),
        ],
        widths=[
            WidthSample(label=label, text=text, n_bytes=len(text.encode("utf-8")))
            for label, text in (
                ("a Latin letter", "r"),
                ("an accented Latin letter", "é"),
                ("a Greek letter", "θ"),
                ("a Han character", "报"),
                ("a raised thumb", "\U0001f44d"),
            )
        ],
        clusters=clusters,
        spellings=spellings,
        spellings_equal=spellings[0].text == spellings[1].text,
        empty_cases=[
            EmptyCase(
                label=label,
                byte_tokens=byte_reading.encode(text).n_tokens,
                character_tokens=characters.encode(text).n_tokens,
            )
            for label, text in (
                ("nothing at all", ""),
                ("one space", " "),
                ("three spaces", "   "),
            )
        ],
        out_of_range_refusal=refusal_sentence,
        lone_surrogate_has_a_byte_spelling=_has_a_byte_spelling("\ud800"),
        corpus_byte_values_used=len(corpus_used),
        all_texts_byte_values_used=len(all_used),
        corpus_bytes=sum(byte_reading.encode(text).n_tokens for text in CORPUS),
        corpus_characters=sum(characters.encode(text).n_tokens for text in CORPUS),
        corpus_learned=sum(learned.encode(text).n_tokens for text in CORPUS),
    )
