"""What a Token Is.

Every endpoint here is registered under ``/concepts/what-a-token-is``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about Vocabulary, Encoding, Tokenizer.

What this module adds to the library
------------------------------------
Two of the three schemes the page shows are the library's own classes,
``CharacterTokenizer`` and ``ByteTokenizer``. The third, one token per run of
non-whitespace characters, has no class in the library, because a whole-word
vocabulary is the baseline every other family exists to improve on rather than
a method anybody ships. It is assembled here out of the two pieces the library
does provide, a ``Vocabulary`` and a ``WhitespacePreTokenizer``, through the
``Tokenizer`` base, so that the lookups, the substitution of the unknown token
and the refusal of a closed table are all the library's template and none of
them is written again here.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a library refusal
needs, and a test asserts that no identifier survives it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/what-a-token-is", tags=["what-a-token-is"])

NOTEBOOK: tuple[str, ...] = (
    "Dr. Alvarez didn't expect the low-cost re-analysis.",
    "The re-analysis used the low-cost sensors from the store room.",
    "Alvarez had run the first analysis in 2019, before the sensors arrived.",
    "Her notes say the low-cost readings drifted every afternoon.",
    "The drift was small, though it was larger than the effect she expected.",
    "She re-ran the analysis twice and the drift did not go away.",
)
"""The six sentences every table on the page is built from.

The first is the sentence the tokenization pages share, chosen because it
breaks things: a full stop that ends an abbreviation, a contraction, a
hyphenated compound and a prefix. The other five continue it, so that one
corpus serves the growth curve, the encodings and the round trips, and a reader
never has to hold two sets of texts in mind at once.
"""

RUNNING_SENTENCE: str = NOTEBOOK[0]
"""The sentence the page carries from beginning to end."""

STAND_IN = "[UNK]"
"""The token an unknown piece falls back to, spelled the way the field spells it."""

UNSEEN_TEXT = "The calibration drift was larger than expected."
"""One word outside the six sentences, in a sentence of words that are inside them."""

TWIN_TEXT = "The temperature drift was larger than expected."
"""The same sentence with a different unseen word, which encodes identically."""

ACCENTED_TEXT = "Dr. Alvarez didn't expect the low-cost re-analysis in Ångström units."
"""A text carrying two characters the six sentences never used."""

AWKWARD_TEXT = "Dr.  Alvarez didn't expect\nthe low-cost re-analysis."
"""The running sentence with a doubled space and a line break, for the round trip."""

MAX_TEXTS = 8
"""How many texts one request may encode. The widgets ask for at most three."""

MAX_TEXT_CHARACTERS = 400
"""How long any one text may be, a few times the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]

Scheme = Literal["character", "word"]

WHITESPACE = WhitespacePreTokenizer()
"""One splitter, since it holds no state and every call would build the same one."""


class WholeWordTokenizer(Tokenizer):
    """Every run of non-whitespace characters is one token.

    Assembled from the library's own splitter and its own table, so that
    ``encode`` and ``decode`` are the library's templates and only the cut and
    the glue are supplied here. The glue is a single space, which is why this
    scheme cannot reproduce a text whose spacing was anything else.
    """

    table: Vocabulary

    @property
    def vocabulary(self) -> Vocabulary:
        """The table this tokenizer was built around."""
        return self.table

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        return WHITESPACE.split(text).texts

    def _text_from(self, pieces: Sequence[str]) -> str:
        return " ".join(pieces)


def word_table(texts: Sequence[str], *, stand_in: str | None = STAND_IN) -> Vocabulary:
    """Every distinct whitespace-delimited word of ``texts``, in alphabetical order.

    Alphabetical rather than first-seen, so the order is a fact about the words
    rather than about which sentence happened to arrive first, which is the
    argument the character scheme makes for codepoint order.
    """
    counts = Corpus.of(list(texts)).word_counts(WHITESPACE)
    words = sorted(counts.words)
    if stand_in is None:
        return Vocabulary(words)
    return Vocabulary([stand_in, *words], unknown_token=stand_in)


@cache
def _character_tokenizer(corpus: tuple[str, ...]) -> CharacterTokenizer:
    """The character scheme fitted on ``corpus``, built once per corpus."""
    return CharacterTokenizer(unknown_token=STAND_IN).fit(list(corpus))


@cache
def _word_tokenizer(corpus: tuple[str, ...]) -> WholeWordTokenizer:
    """The whole-word scheme over ``corpus``, built once per corpus."""
    return WholeWordTokenizer(table=word_table(corpus))


def _tokenizer(scheme: Scheme, corpus: tuple[str, ...] = NOTEBOOK) -> Tokenizer:
    """Whichever of the two schemes was asked for, fitted on ``corpus``."""
    if scheme == "character":
        return _character_tokenizer(corpus)
    return _word_tokenizer(corpus)


PRODUCER_PATTERN = re.compile(r"\b[A-Z][A-Za-z]*(?= produced the piece)")
"""A refusal that opens on the class name of whatever cut the text up."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifier taken out of it.

    The messages are written to be read and all but one of them already are, so
    this substitutes the single opening that names a class and leaves the rest
    alone. A test asserts that no identifier survives, so a message that grows
    one later fails there rather than reaching the page.
    """
    return PRODUCER_PATTERN.sub("the scheme", str(refusal))


class TokenView(BaseModel):
    """One piece of an encoded text, with the number it goes in as."""

    text: str
    token_id: int
    known: bool


class EncodedText(BaseModel):
    """One text, what it became, and what came back when the numbers were read.

    ``round_trip_exact`` compares ``decoded`` against ``source`` character for
    character, so it is false whenever anything at all was lost, whether that
    was an unseen piece or the spacing between the pieces.
    """

    source: str
    tokens: list[TokenView]
    ids: list[int]
    n_tokens: int
    n_unknown: int
    decoded: str
    round_trip_exact: bool


class EncodeRequest(BaseModel):
    """Texts to encode, and which of the two schemes to encode them under."""

    texts: list[BoundedText] = Field(min_length=1, max_length=MAX_TEXTS)
    scheme: Scheme

    model_config = {"extra": "forbid"}


class EncodeResponse(BaseModel):
    """The table, and every text seen through it.

    ``vocabulary`` is the whole table in number order, which is short enough to
    send because both schemes over six sentences run to a few dozen entries.
    ``n_pieces_learned`` is the table without its stand-in, which is the number
    that answers "how many different pieces did the six sentences use".
    """

    scheme: Scheme
    corpus: list[str]
    vocabulary_size: int
    n_pieces_learned: int
    stand_in: str | None
    vocabulary: list[str]
    encodings: list[EncodedText]


def _encoded(tokenizer: Tokenizer, text: str) -> EncodedText:
    """Encode one text, decode it again, and record what survived."""
    encoding = tokenizer.encode(text)
    stand_in = tokenizer.vocabulary.unknown_token
    decoded = tokenizer.decode(encoding.ids)
    return EncodedText(
        source=text,
        tokens=[
            TokenView(
                text=token.text,
                token_id=token.token_id,
                known=token.text != stand_in,
            )
            for token in encoding
        ],
        ids=list(encoding.ids),
        n_tokens=encoding.n_tokens,
        n_unknown=sum(1 for token in encoding if token.text == stand_in),
        decoded=decoded,
        round_trip_exact=decoded == text,
    )


@router.post("/encode", response_model=EncodeResponse)
def encode(request: EncodeRequest) -> EncodeResponse:
    """Encode every text under one scheme and read the numbers back again."""
    tokenizer = _tokenizer(request.scheme)
    vocabulary = tokenizer.vocabulary
    return EncodeResponse(
        scheme=request.scheme,
        corpus=list(NOTEBOOK),
        vocabulary_size=vocabulary.n_tokens,
        n_pieces_learned=vocabulary.n_tokens - (1 if vocabulary.has_unknown else 0),
        stand_in=vocabulary.unknown_token,
        vocabulary=list(vocabulary),
        encodings=[_encoded(tokenizer, text) for text in request.texts],
    )


class GrowthStep(BaseModel):
    """The two tables after the first ``n_texts`` sentences, and what the next costs.

    The ``next_`` figures are the sentence after those, encoded against tables
    that have not met it, so they say how much of a genuinely new sentence each
    scheme can already spell. They are absent on the last step, where there is
    no next sentence.
    """

    n_texts: int
    characters_seen: int
    words_seen: int
    character_table: int
    word_table: int
    next_text: str | None
    next_character_tokens: int | None
    next_unknown_characters: int | None
    next_word_tokens: int | None
    next_unknown_words: int | None


class GrowthResponse(BaseModel):
    """The six sentences arriving one at a time, with both tables measured after each."""

    corpus: list[str]
    steps: list[GrowthStep]


@router.get("/vocabulary-growth", response_model=GrowthResponse)
def vocabulary_growth() -> GrowthResponse:
    """Grow both tables one sentence at a time and measure them after each."""
    steps: list[GrowthStep] = []
    for size in range(1, len(NOTEBOOK) + 1):
        prefix = NOTEBOOK[:size]
        characters = _character_tokenizer(prefix)
        words = _word_tokenizer(prefix)

        next_text = NOTEBOOK[size] if size < len(NOTEBOOK) else None
        character_view = None if next_text is None else _encoded(characters, next_text)
        word_view = None if next_text is None else _encoded(words, next_text)

        steps.append(
            GrowthStep(
                n_texts=size,
                characters_seen=sum(len(text) for text in prefix),
                words_seen=sum(len(WHITESPACE.split(text)) for text in prefix),
                character_table=characters.vocabulary.n_tokens,
                word_table=words.vocabulary.n_tokens,
                next_text=next_text,
                next_character_tokens=(
                    None if character_view is None else character_view.n_tokens
                ),
                next_unknown_characters=(
                    None if character_view is None else character_view.n_unknown
                ),
                next_word_tokens=None if word_view is None else word_view.n_tokens,
                next_unknown_words=None if word_view is None else word_view.n_unknown,
            )
        )

    return GrowthResponse(corpus=list(NOTEBOOK), steps=steps)


Answer = Literal["substituted", "refused", "nothing was unknown"]


class UnknownProbe(BaseModel):
    """One table meeting one text, and what it did about the pieces it lacked.

    ``reason`` is present only where the table refused, and is the refusal's
    own sentence with the identifier taken out of it.
    """

    table: str
    text: str
    answer: Answer
    n_tokens: int | None
    ids: list[int] | None
    pieces: list[str] | None
    decoded: str | None
    round_trip_exact: bool | None
    reason: str | None


class UnknownAnswersResponse(BaseModel):
    """Every answer a table can give at a piece it has never seen.

    ``table_size`` is the whole-word table with its stand-in, ``closed_size``
    the same words without one, and ``byte_size`` the table that cannot meet an
    unknown piece at all because it already holds every byte value.
    """

    table_size: int
    closed_size: int
    byte_size: int
    probes: list[UnknownProbe]


def _probe(table: str, tokenizer: Tokenizer, text: str) -> UnknownProbe:
    """Encode one text against one table and record what happened."""
    try:
        view = _encoded(tokenizer, text)
    except MLLibError as refusal:
        return UnknownProbe(
            table=table,
            text=text,
            answer="refused",
            n_tokens=None,
            ids=None,
            pieces=None,
            decoded=None,
            round_trip_exact=None,
            reason=_in_plain_words(refusal),
        )

    return UnknownProbe(
        table=table,
        text=text,
        answer="substituted" if view.n_unknown else "nothing was unknown",
        n_tokens=view.n_tokens,
        ids=view.ids,
        pieces=[token.text for token in view.tokens],
        decoded=view.decoded,
        round_trip_exact=view.round_trip_exact,
        reason=None,
    )


@router.get("/unknown-answers", response_model=UnknownAnswersResponse)
def unknown_answers() -> UnknownAnswersResponse:
    """Four tables meeting a piece they have never seen, and what each answers."""
    substituting = _word_tokenizer(NOTEBOOK)
    closed = WholeWordTokenizer(table=word_table(NOTEBOOK, stand_in=None))
    characters = _character_tokenizer(NOTEBOOK)
    bytes_scheme = ByteTokenizer()

    return UnknownAnswersResponse(
        table_size=substituting.vocabulary.n_tokens,
        closed_size=closed.vocabulary.n_tokens,
        byte_size=bytes_scheme.vocabulary.n_tokens,
        probes=[
            _probe("whole words, with a stand-in", substituting, UNSEEN_TEXT),
            _probe("whole words, with a stand-in", substituting, TWIN_TEXT),
            _probe("whole words, no stand-in", closed, UNSEEN_TEXT),
            _probe("characters, with a stand-in", characters, ACCENTED_TEXT),
            _probe("every byte value", bytes_scheme, ACCENTED_TEXT),
        ],
    )
