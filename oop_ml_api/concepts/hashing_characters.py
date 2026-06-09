"""Hashing Characters.

Every endpoint here is registered under ``/concepts/hashing-characters``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about HashedCharacterTokenizer, TrigramHashTokenizer.

What this module computes
-------------------------
Five endpoints, one per family of widgets rather than one per number.
``/hash-text`` puts a text through the multiplicative hash at a chosen bucket
count and hash count and reports every character's buckets, the table those
buckets imply, and any two characters of the text that came out identical.
``/collisions`` answers the other half of that question, taking one character and
finding every codepoint that shares its whole set of buckets. ``/separation``
counts how many kinds of character the arrangement can tell apart, over the
whole codepoint space, at several bucket counts and several numbers of hashes,
which is the measurement the fourth part of the page is built on.
``/fingerprints`` is the trigram half, a word becoming the set of buckets its
overlapping three-character pieces light. ``/limits`` carries the fixed figures
the closing part rests on.

The counting in ``/separation`` uses one identity rather than a sweep over all
1,114,112 codepoints. Every hash here is ``((codepoint + 1) * prime) mod
n_buckets``, which reads the codepoint only through ``(codepoint + 1) mod
n_buckets``, so counting over one full run of residues counts over every
codepoint. A test checks that shortcut against the sweep it replaces.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a refusal needs,
and a test asserts that no identifier survives it.
"""

from __future__ import annotations

import itertools
import unicodedata
from functools import cache
from math import gcd, lcm
from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.characters.byte import (
    ByteTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.characters.character import (
    CharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.hashing.hashed_characters import (
    HASH_PRIMES,
    HashedCharacterTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.hashing.trigram_hashing import (
    HASH_MULTIPLIERS,
    TrigramHashTokenizer,
    overlap,
    polynomial_hash,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/hashing-characters", tags=["hashing-characters"])


# --------------------------------------------------------------------------
# The texts and the settings every endpoint works from.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

GREEK_TEXT = "Η έκθεση έφτασε αργά."
"""One short sentence in a script the English corpus below never uses."""

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
"""The eighteen English sentences the neighbouring pages of this section use.

Carried here only so the table a corpus produces can be laid beside the table
arithmetic produces, and so the two pages' figures can be read together.
"""

LANGUAGES: tuple[tuple[str, str], ...] = (
    ("English", "The report arrived late."),
    ("Greek", "Η έκθεση έφτασε αργά."),
    ("Russian", "Отчёт пришёл поздно."),
    ("Hindi", "रिपोर्ट देर से आई।"),
    ("Japanese", "報告書は遅れて届いた。"),
    ("Chinese", "报告迟到了。"),
)
"""One short sentence with one meaning, in six writing systems."""

UNICODE_CODEPOINTS = 0x110000
"""How many codepoints Unicode defines room for."""

EMBEDDING_WIDTH = 768
"""An ordinary width for the row a model gives each entry of a table."""

PUBLISHED_BUCKETS = 16384
"""The bucket count the published arrangement uses."""

PUBLISHED_HASHES = 8
"""How many hashes the published arrangement applies to each character."""

TRIGRAM_BUCKETS = 8192
"""The bucket count the trigram examples on the page use."""

MAX_TEXT_CHARACTERS = 200
"""How long a text one request may hash, a few times the running sentence."""

MAX_BUCKETS = 262144
"""The widest table a request may ask for, sixteen times the published one."""

MAX_WORDS = 8
"""How many words one fingerprint request may carry."""

MAX_WORD_CHARACTERS = 40
"""How long any one word may be."""

MAX_PARTNERS_LISTED = 12
"""How many of a character's sharers a response spells out."""

SWEEP_BUCKETS: tuple[int, ...] = (93, 256, 961, 1333, 2048, 3999, 16384)
"""The bucket counts the separation chart sweeps.

Three of them are multiples of the first multiplier and four are not, which is
the distinction the chart exists to show.
"""

SWEEP_HASHES: tuple[int, ...] = (1, 2, 4, 8)
"""The numbers of hashes the separation chart compares."""

SWEEP_LIMIT = 2000
"""The largest bucket count the exhaustive count over every width runs to."""

TRIGRAM_ALPHABET = "abcdefghijklmnopqrstuvwxyz_"
"""Twenty-six lowercase letters and the boundary marker, for the spread count."""

SPREAD_BUCKETS: tuple[int, ...] = (1024, 4096, 8192, 16384, 32768)
"""The widths the trigram spread is counted at."""

NARROW_BUCKETS = 512
"""A deliberately narrow trigram width, where whole words do share a set."""

BoundedText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)]
BoundedWord = Annotated[str, Field(min_length=1, max_length=MAX_WORD_CHARACTERS)]


PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("n_hash_functions", "the number of hashes"),
    ("n_buckets", "the number of buckets"),
    ("codepoint", "character number"),
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


def _hasher(n_buckets: int, n_hash_functions: int) -> HashedCharacterTokenizer:
    """The arrangement asked for, or a typed refusal if it cannot be built."""
    try:
        return HashedCharacterTokenizer(
            n_buckets=n_buckets, n_hash_functions=n_hash_functions
        )
    except MLLibError as refusal:  # pragma: no cover - bounds catch this first
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal


def _display(character: str) -> str:
    """A character with something visible standing in for a space."""
    return "␣" if character == " " else character


def _codepoint_label(codepoint: int) -> str:
    """The conventional way of writing a character's number down."""
    return f"U+{codepoint:04X}"


def _unicode_name(codepoint: int) -> str:
    """Unicode's own name for a character, or an empty string where it has none."""
    return unicodedata.name(chr(codepoint), "")


@cache
def _sharing_step(n_buckets: int, n_hash_functions: int) -> int:
    """How far apart two character numbers must be to share every bucket.

    Hash ``k`` sends two numbers to one bucket exactly when their difference is
    a multiple of ``n_buckets`` divided by whatever that hash's multiplier
    shares with it, so sharing every bucket at once means being a multiple of
    the least common multiple of those.
    """
    return lcm(
        *(
            n_buckets // gcd(prime, n_buckets)
            for prime in HASH_PRIMES[:n_hash_functions]
        )
    )


def _count_sharers(codepoint: int, step: int) -> int:
    """How many other character numbers sit at a whole number of steps from this one.

    Every one of them lands in the same bucket under every hash, so this is the
    size of the crowd one bucket set stands for, less the character itself.
    """
    first = codepoint % step
    return (UNICODE_CODEPOINTS - 1 - first) // step


# --------------------------------------------------------------------------
# One text through the hash.
# --------------------------------------------------------------------------


class HashedCharacterView(BaseModel):
    """One character of a text, its number, and the bucket each hash chose."""

    character: str
    display: str
    codepoint: int
    codepoint_label: str
    bucket_ids: list[int]
    n_sharers: int


class ClashGroup(BaseModel):
    """Two or more characters of this text that came out with one set of buckets."""

    characters: list[str]
    bucket_ids: list[int]


class HashTextRequest(BaseModel):
    """A text to hash, and the arrangement to hash it under."""

    text: BoundedText
    n_buckets: int = Field(default=PUBLISHED_BUCKETS, ge=2, le=MAX_BUCKETS)
    n_hash_functions: int = Field(default=PUBLISHED_HASHES, ge=1, le=len(HASH_PRIMES))

    model_config = {"extra": "forbid"}


class HashTextResponse(BaseModel):
    """Every character's buckets, what they cost, and what they lost.

    ``n_ids`` is how many numbers the whole text becomes, which is one per
    character per hash. ``table_numbers`` is what the rows cost a model at an
    ordinary width, and it does not depend on how many hashes were applied,
    since each hash reads a slice of the same total width.
    """

    text: str
    n_characters: int
    n_bytes: int
    byte_tokens: int
    n_buckets: int
    n_hash_functions: int
    multipliers: list[int]
    characters: list[HashedCharacterView]
    n_ids: int
    table_rows: int
    table_numbers: int
    slice_width: int
    sharing_step: int
    characters_per_bucket_set: float
    clashes: list[ClashGroup]
    rebuilt: str
    rebuilt_exactly: bool
    n_distinct_characters: int
    n_distinct_bucket_sets: int


@router.post("/hash-text", response_model=HashTextResponse)
def hash_text(request: HashTextRequest) -> HashTextResponse:
    """Hash every character of a text and report what the arrangement cost."""
    hasher = _hasher(request.n_buckets, request.n_hash_functions)
    step = _sharing_step(request.n_buckets, request.n_hash_functions)

    try:
        hashed = hasher.encode(request.text)
    except MLLibError as refusal:  # pragma: no cover - bounds catch this first
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    seen: dict[tuple[int, ...], list[str]] = {}
    for character in hashed:
        group = seen.setdefault(character.bucket_ids, [])
        if character.character not in group:
            group.append(character.character)

    return HashTextResponse(
        text=request.text,
        n_characters=hashed.n_characters,
        n_bytes=len(request.text.encode("utf-8")),
        byte_tokens=_byte_reading().encode(request.text).n_tokens,
        n_buckets=request.n_buckets,
        n_hash_functions=request.n_hash_functions,
        multipliers=list(hasher.primes),
        characters=[
            HashedCharacterView(
                character=character.character,
                display=_display(character.character),
                codepoint=character.codepoint,
                codepoint_label=_codepoint_label(character.codepoint),
                bucket_ids=list(character.bucket_ids),
                n_sharers=_count_sharers(character.codepoint, step),
            )
            for character in hashed
        ],
        n_ids=hashed.n_characters * request.n_hash_functions,
        table_rows=request.n_buckets,
        table_numbers=request.n_buckets * EMBEDDING_WIDTH,
        slice_width=EMBEDDING_WIDTH // request.n_hash_functions,
        sharing_step=step,
        characters_per_bucket_set=UNICODE_CODEPOINTS / step,
        clashes=[
            ClashGroup(
                characters=[_display(one) for one in group], bucket_ids=list(key)
            )
            for key, group in seen.items()
            if len(group) > 1
        ],
        rebuilt=hashed.text,
        rebuilt_exactly=hashed.text == request.text,
        n_distinct_characters=len(set(request.text)),
        n_distinct_bucket_sets=len(seen),
    )


# --------------------------------------------------------------------------
# Everything one character cannot be told apart from.
# --------------------------------------------------------------------------


class SharerView(BaseModel):
    """One other character that came out with the same set of buckets."""

    character: str
    codepoint: int
    codepoint_label: str
    unicode_name: str


class CollisionsRequest(BaseModel):
    """One character, and the arrangement to look at it under."""

    character: BoundedWord = Field(default="A")
    n_buckets: int = Field(default=PUBLISHED_BUCKETS, ge=2, le=MAX_BUCKETS)
    n_hash_functions: int = Field(default=PUBLISHED_HASHES, ge=1, le=len(HASH_PRIMES))

    model_config = {"extra": "forbid"}


class CollisionsResponse(BaseModel):
    """What a character's buckets are, and who else has them.

    ``n_sharers`` counts every other character number the arrangement cannot
    separate from this one; ``named`` is the first few of those Unicode has a
    name for, which is a smaller number, since most of the codepoint space is
    unassigned.
    """

    character: str
    display: str
    codepoint: int
    codepoint_label: str
    unicode_name: str
    bucket_ids: list[int]
    n_buckets: int
    n_hash_functions: int
    multipliers: list[int]
    sharing_step: int
    n_sharers: int
    n_named_sharers: int
    named: list[SharerView]
    every_sharer_agrees: bool


@router.post("/collisions", response_model=CollisionsResponse)
def collisions(request: CollisionsRequest) -> CollisionsResponse:
    """Find every character number that shares one character's whole set."""
    if len(request.character) != 1:
        raise HTTPException(
            status_code=422,
            detail="Give one character at a time, since a bucket is chosen from one.",
        )

    hasher = _hasher(request.n_buckets, request.n_hash_functions)
    codepoint = ord(request.character)
    try:
        buckets = hasher.bucket_ids_of(codepoint)
    except MLLibError as refusal:  # pragma: no cover - a string holds no such value
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    step = _sharing_step(request.n_buckets, request.n_hash_functions)
    sharers = [
        other
        for other in range(codepoint % step, UNICODE_CODEPOINTS, step)
        if other != codepoint
    ]
    named = [other for other in sharers if _unicode_name(other)]

    return CollisionsResponse(
        character=request.character,
        display=_display(request.character),
        codepoint=codepoint,
        codepoint_label=_codepoint_label(codepoint),
        unicode_name=_unicode_name(codepoint),
        bucket_ids=list(buckets),
        n_buckets=request.n_buckets,
        n_hash_functions=request.n_hash_functions,
        multipliers=list(hasher.primes),
        sharing_step=step,
        n_sharers=len(sharers),
        n_named_sharers=len(named),
        named=[
            SharerView(
                character=chr(other),
                codepoint=other,
                codepoint_label=_codepoint_label(other),
                unicode_name=_unicode_name(other),
            )
            for other in named[:MAX_PARTNERS_LISTED]
        ],
        every_sharer_agrees=all(
            hasher.bucket_ids_of(other) == buckets for other in sharers
        ),
    )


# --------------------------------------------------------------------------
# How many kinds of character the arrangement can tell apart at all.
# --------------------------------------------------------------------------


@cache
def _distinct_bucket_sets(n_buckets: int, n_hash_functions: int) -> int:
    """How many different sets of buckets the whole codepoint space produces.

    Each hash multiplies ``codepoint + 1`` and reduces modulo the bucket count,
    so it reads the character only through that residue. Counting over one run
    of residues therefore counts over every codepoint, and the test module
    checks this against the sweep over all 1,114,112 of them.
    """
    residues = np.arange(min(n_buckets, UNICODE_CODEPOINTS), dtype=np.int64)
    multipliers = np.array(HASH_PRIMES[:n_hash_functions], dtype=np.int64)
    sets = (residues[:, None] * multipliers[None, :]) % n_buckets
    return len(np.unique(sets, axis=0))


class SeparationCount(BaseModel):
    """How many kinds of character one number of hashes can tell apart."""

    n_hash_functions: int
    distinct_sets: int
    reaches_the_ceiling: bool


class SeparationRow(BaseModel):
    """One bucket count, swept over several numbers of hashes.

    ``ceiling`` is the most kinds any number of hashes could tell apart at this
    width, and ``shares_a_factor`` says whether the first multiplier and the
    bucket count have a common divisor, which is the condition that decides
    whether the second hash has anything to add.
    """

    n_buckets: int
    ceiling: int
    shares_a_factor: bool
    counts: list[SeparationCount]


class SeparationResponse(BaseModel):
    """The separation sweep, and how often an extra hash is worth anything.

    ``widths_where_one_falls_short`` counts the bucket counts from 2 to the
    limit at which a single hash cannot reach the ceiling;
    ``widths_where_two_fall_short`` counts those at which two cannot.
    """

    codepoints: int
    multipliers: list[int]
    rows: list[SeparationRow]
    widths_tried: int
    widths_where_one_falls_short: int
    widths_where_two_fall_short: int
    first_ten_short_widths: list[int]
    every_short_width_is_a_multiple: bool
    first_multiplier: int
    multiples_of_the_first_multiplier: int


@cache
def _separation() -> SeparationResponse:
    """The sweep and the exhaustive count, computed once."""
    rows = [
        SeparationRow(
            n_buckets=n_buckets,
            ceiling=min(n_buckets, UNICODE_CODEPOINTS),
            shares_a_factor=gcd(HASH_PRIMES[0], n_buckets) > 1,
            counts=[
                SeparationCount(
                    n_hash_functions=n_hashes,
                    distinct_sets=_distinct_bucket_sets(n_buckets, n_hashes),
                    reaches_the_ceiling=(
                        _distinct_bucket_sets(n_buckets, n_hashes)
                        == min(n_buckets, UNICODE_CODEPOINTS)
                    ),
                )
                for n_hashes in SWEEP_HASHES
            ],
        )
        for n_buckets in SWEEP_BUCKETS
    ]

    # Two numbers land in one bucket set exactly when they differ by a multiple
    # of the sharing step, and the residues run over the whole width, so the
    # count of distinct sets is that step. Used here rather than the count above
    # only because this loop runs two thousand times; a test asserts the two
    # agree at every width the chart shows.
    short_at_one: list[int] = []
    short_at_two = 0
    for n_buckets in range(2, SWEEP_LIMIT + 1):
        ceiling = min(n_buckets, UNICODE_CODEPOINTS)
        if _sharing_step(n_buckets, 1) < ceiling:
            short_at_one.append(n_buckets)
        if _sharing_step(n_buckets, 2) < ceiling:
            short_at_two += 1

    return SeparationResponse(
        codepoints=UNICODE_CODEPOINTS,
        multipliers=list(HASH_PRIMES[:PUBLISHED_HASHES]),
        rows=rows,
        widths_tried=SWEEP_LIMIT - 1,
        widths_where_one_falls_short=len(short_at_one),
        widths_where_two_fall_short=short_at_two,
        first_ten_short_widths=short_at_one[:10],
        every_short_width_is_a_multiple=all(
            width % HASH_PRIMES[0] == 0 for width in short_at_one
        ),
        first_multiplier=HASH_PRIMES[0],
        multiples_of_the_first_multiplier=sum(
            1 for width in range(2, SWEEP_LIMIT + 1) if width % HASH_PRIMES[0] == 0
        ),
    )


@router.get("/separation", response_model=SeparationResponse)
def separation() -> SeparationResponse:
    """How many kinds of character each arrangement can tell apart."""
    return _separation()


# --------------------------------------------------------------------------
# The same idea on a piece of a word.
# --------------------------------------------------------------------------


class WordFingerprint(BaseModel):
    """One word, its overlapping three-character pieces, and the buckets they lit."""

    word: str
    trigrams: list[str]
    bucket_ids: list[int]
    n_trigrams: int
    n_active: int


class PairOverlap(BaseModel):
    """Two words and how many buckets they both light."""

    first: str
    second: str
    shared: int
    first_active: int
    second_active: int
    identical: bool


class FingerprintsRequest(BaseModel):
    """Words to fingerprint, and the width to fingerprint them at."""

    words: list[BoundedWord] = Field(min_length=1, max_length=MAX_WORDS)
    n_buckets: int = Field(default=TRIGRAM_BUCKETS, ge=2, le=MAX_BUCKETS)
    n_hash_functions: int = Field(default=1, ge=1, le=len(HASH_MULTIPLIERS))

    model_config = {"extra": "forbid"}


class FingerprintsResponse(BaseModel):
    """Every word's fingerprint, and every pair's overlap."""

    n_buckets: int
    n_hash_functions: int
    multipliers: list[int]
    boundary_marker: str
    words: list[WordFingerprint]
    pairs: list[PairOverlap]


@router.post("/fingerprints", response_model=FingerprintsResponse)
def fingerprints(request: FingerprintsRequest) -> FingerprintsResponse:
    """Turn each word into the set of buckets its three-character pieces light."""
    try:
        tokenizer = TrigramHashTokenizer(
            n_buckets=request.n_buckets, n_hash_functions=request.n_hash_functions
        )
        activations = [tokenizer.activations_of(word) for word in request.words]
    except MLLibError as refusal:  # pragma: no cover - bounds catch this first
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    return FingerprintsResponse(
        n_buckets=request.n_buckets,
        n_hash_functions=request.n_hash_functions,
        multipliers=list(tokenizer.multipliers),
        boundary_marker=tokenizer.boundary_marker,
        words=[
            WordFingerprint(
                word=one.word,
                trigrams=list(tokenizer.trigrams_of(one.word)),
                bucket_ids=list(one.bucket_ids),
                n_trigrams=len(tokenizer.trigrams_of(one.word)),
                n_active=one.n_active,
            )
            for one in activations
        ],
        pairs=[
            PairOverlap(
                first=first.word,
                second=second.word,
                shared=overlap(first, second),
                first_active=first.n_active,
                second_active=second.n_active,
                identical=first.bucket_ids == second.bucket_ids,
            )
            for position, first in enumerate(activations)
            for second in activations[position + 1 :]
        ],
    )


# --------------------------------------------------------------------------
# The fixed figures the page rests on.
# --------------------------------------------------------------------------


@cache
def _byte_reading() -> ByteTokenizer:
    """The reading whose table is the 256 byte values and needs no corpus."""
    return ByteTokenizer()


@cache
def _character_reading() -> CharacterTokenizer:
    """The reading whose table is every character the eighteen sentences used."""
    return CharacterTokenizer().fit(list(CORPUS))


@cache
def _learned_reading() -> BytePairEncoding:
    """The reading whose pieces were merged out of the eighteen sentences."""
    return BytePairEncoding(byte_fallback=False, vocabulary_size=300).fit(list(CORPUS))


class TableCost(BaseModel):
    """One table, its rows, and what those rows cost at an ordinary width."""

    label: str
    rows: int
    numbers: int


class LengthRow(BaseModel):
    """One sentence measured under each way of cutting it up."""

    language: str
    text: str
    n_characters: int
    n_bytes: int
    hashed_positions: int
    hashed_ids: int
    merged_tokens: int
    merged_stand_ins: int
    character_stand_ins: int


class WidthTrade(BaseModel):
    """One bucket count, what it costs and how badly it crowds the characters."""

    n_buckets: int
    numbers: int
    characters_per_bucket_set: float


class SpreadRow(BaseModel):
    """How the three-character pieces of a small alphabet fall into buckets."""

    n_buckets: int
    buckets_used: int
    heaviest_bucket: int
    mean_load: float


class TrigramClash(BaseModel):
    """Several three-character pieces that land in one bucket."""

    bucket_id: int
    trigrams: list[str]


class WordClash(BaseModel):
    """Two whole words the fingerprint cannot tell apart."""

    first: str
    second: str


class RepeatedLetterCase(BaseModel):
    """Two words differing by a repeat, which a set records only sometimes.

    ``identical`` is true where the repeat produced a piece the shorter word
    already had, and false where it produced a new one, which is where the
    boundary of the effect lies.
    """

    first: str
    second: str
    first_trigrams: list[str]
    second_trigrams: list[str]
    identical: bool


class LimitsResponse(BaseModel):
    """Everything the closing part quotes, computed rather than asserted."""

    codepoints: int
    embedding_width: int
    published_buckets: int
    published_hashes: int
    tables: list[TableCost]
    codepoint_table_over_hashed: float
    hashed_over_byte_table: float
    lengths: list[LengthRow]
    width_trades: list[WidthTrade]
    trigram_alphabet_size: int
    trigrams_counted: int
    spread: list[SpreadRow]
    trigram_clash: TrigramClash
    heaviest_trigram_bucket: TrigramClash
    narrow_buckets: int
    narrow_words_tried: int
    narrow_distinct_fingerprints: int
    narrow_clashes: int
    narrow_examples: list[WordClash]
    repeated_letters: list[RepeatedLetterCase]
    sentence_words: list[str]
    sentence_positions: int
    sentence_active_buckets: int
    sentence_merged_tokens: int
    sentence_characters: int


@cache
def _spread() -> tuple[
    tuple[SpreadRow, ...],
    TrigramClash,
    TrigramClash,
    int,
    int,
    int,
    tuple[WordClash, ...],
]:
    """How the three-character pieces spread, and where whole words collide."""
    rows: list[SpreadRow] = []
    at_published: dict[int, list[str]] = {}
    trigrams = [
        "".join(pieces) for pieces in itertools.product(TRIGRAM_ALPHABET, repeat=3)
    ]

    for n_buckets in SPREAD_BUCKETS:
        loads: dict[int, list[str]] = {}
        for trigram in trigrams:
            bucket = polynomial_hash(trigram, HASH_MULTIPLIERS[0]) % n_buckets
            loads.setdefault(bucket, []).append(trigram)
        if n_buckets == TRIGRAM_BUCKETS:
            at_published = loads
        rows.append(
            SpreadRow(
                n_buckets=n_buckets,
                buckets_used=len(loads),
                heaviest_bucket=max(len(group) for group in loads.values()),
                mean_load=round(len(trigrams) / len(loads), 3),
            )
        )

    crowded = sorted(
        (item for item in at_published.items() if len(item[1]) > 1),
        key=lambda item: (-len(item[1]), item[0]),
    )
    heaviest = TrigramClash(bucket_id=crowded[0][0], trigrams=sorted(crowded[0][1]))
    with_nal = next(
        (bucket, group) for bucket, group in at_published.items() if "nal" in group
    )
    named = TrigramClash(bucket_id=with_nal[0], trigrams=sorted(with_nal[1]))

    narrow = TrigramHashTokenizer(n_buckets=NARROW_BUCKETS)
    fingerprints_seen: dict[tuple[int, ...], list[str]] = {}
    letters = TRIGRAM_ALPHABET[:26]
    for pieces in itertools.product(letters, repeat=3):
        word = "".join(pieces)
        fingerprints_seen.setdefault(narrow.activations_of(word).bucket_ids, []).append(
            word
        )
    clashes = [group for group in fingerprints_seen.values() if len(group) > 1]

    return (
        tuple(rows),
        named,
        heaviest,
        26**3,
        len(fingerprints_seen),
        len(clashes),
        tuple(WordClash(first=group[0], second=group[1]) for group in clashes[:3]),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The measured facts the closing part of the page rests on."""
    hasher = HashedCharacterTokenizer()
    byte_reading = _byte_reading()
    characters = _character_reading()
    learned = _learned_reading()
    stand_in = characters.vocabulary.unknown_token
    learned_stand_in = learned.vocabulary.unknown_token

    hashed_numbers = PUBLISHED_BUCKETS * EMBEDDING_WIDTH
    (
        spread,
        trigram_clash,
        heaviest,
        narrow_tried,
        narrow_distinct,
        narrow_clashes,
        narrow_examples,
    ) = _spread()

    trigrams = TrigramHashTokenizer(n_buckets=TRIGRAM_BUCKETS)
    sentence = trigrams.encode(RUNNING_SENTENCE)
    wide = TrigramHashTokenizer(n_buckets=MAX_BUCKETS)

    return LimitsResponse(
        codepoints=UNICODE_CODEPOINTS,
        embedding_width=EMBEDDING_WIDTH,
        published_buckets=PUBLISHED_BUCKETS,
        published_hashes=PUBLISHED_HASHES,
        tables=[
            TableCost(
                label="one row per character Unicode has room for",
                rows=UNICODE_CODEPOINTS,
                numbers=UNICODE_CODEPOINTS * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="one row per bucket, at the published width",
                rows=PUBLISHED_BUCKETS,
                numbers=hashed_numbers,
            ),
            TableCost(
                label="one row per byte value",
                rows=byte_reading.vocabulary.n_tokens,
                numbers=byte_reading.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="one row per piece merged from eighteen sentences",
                rows=learned.vocabulary.n_tokens,
                numbers=learned.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
            TableCost(
                label="one row per character those sentences used",
                rows=characters.vocabulary.n_tokens,
                numbers=characters.vocabulary.n_tokens * EMBEDDING_WIDTH,
            ),
        ],
        codepoint_table_over_hashed=(UNICODE_CODEPOINTS * EMBEDDING_WIDTH)
        / hashed_numbers,
        hashed_over_byte_table=hashed_numbers
        / (byte_reading.vocabulary.n_tokens * EMBEDDING_WIDTH),
        lengths=[
            LengthRow(
                language=language,
                text=text,
                n_characters=len(text),
                n_bytes=byte_reading.encode(text).n_tokens,
                hashed_positions=hasher.encode(text).n_characters,
                hashed_ids=hasher.encode(text).n_characters * PUBLISHED_HASHES,
                merged_tokens=learned.encode(text).n_tokens,
                merged_stand_ins=sum(
                    1
                    for piece in learned.encode(text)
                    if piece.text == learned_stand_in
                ),
                character_stand_ins=sum(
                    1 for piece in characters.encode(text) if piece.text == stand_in
                ),
            )
            for language, text in (
                ("the running sentence", RUNNING_SENTENCE),
                *LANGUAGES,
            )
        ],
        width_trades=[
            WidthTrade(
                n_buckets=n_buckets,
                numbers=n_buckets * EMBEDDING_WIDTH,
                characters_per_bucket_set=UNICODE_CODEPOINTS
                / _sharing_step(n_buckets, PUBLISHED_HASHES),
            )
            for n_buckets in (256, 1024, 4096, 16384, 65536, 262144)
        ],
        trigram_alphabet_size=len(TRIGRAM_ALPHABET),
        trigrams_counted=len(TRIGRAM_ALPHABET) ** 3,
        spread=list(spread),
        trigram_clash=trigram_clash,
        heaviest_trigram_bucket=heaviest,
        narrow_buckets=NARROW_BUCKETS,
        narrow_words_tried=narrow_tried,
        narrow_distinct_fingerprints=narrow_distinct,
        narrow_clashes=narrow_clashes,
        narrow_examples=list(narrow_examples),
        repeated_letters=[
            RepeatedLetterCase(
                first=first,
                second=second,
                first_trigrams=list(wide.trigrams_of(first)),
                second_trigrams=list(wide.trigrams_of(second)),
                identical=(
                    wide.activations_of(first).bucket_ids
                    == wide.activations_of(second).bucket_ids
                ),
            )
            for first, second in (
                ("aaa", "aaaa"),
                ("aaa", "aaaaa"),
                ("baa", "baaa"),
            )
        ],
        sentence_words=list(sentence.words),
        sentence_positions=sentence.n_words,
        sentence_active_buckets=sum(word.n_active for word in sentence),
        sentence_merged_tokens=learned.encode(RUNNING_SENTENCE).n_tokens,
        sentence_characters=len(RUNNING_SENTENCE),
    )
