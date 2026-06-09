"""Patching Without a Vocabulary.

Every endpoint here is registered under ``/concepts/patching-without-a-vocabulary``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about EntropyPatcher, FixedSizePatcher.

What this module computes
-------------------------
Four endpoints, one per family of widgets rather than one per number.
``/patch`` cuts one text both ways at once, into blocks of a fixed size and at
the positions a byte model found hard to predict, and returns the per-byte
uncertainty the second of those rests on, so the playground and the profile
chart read one response. ``/cut-landing`` sweeps both methods across the
eighteen-sentence corpus and counts where every cut fell, against the share of
all positions that fall inside a word, which is what an uninformed cut would
give. ``/corpus-condition`` is the finding the page is built around: the same
sentence read by models fitted on more and more copies of it, at two levels of
smoothing, so a reader can see the point at which the uncertainties separate at
all. ``/limits`` carries the lengths, the model dependence and the small worked
example the closing parts quote.

The byte model behind the entropy cuts is a byte n-gram counted over the same
eighteen English sentences the byte pair encoding and bytes-and-characters pages
use, so the three sets of numbers can be laid beside each other. Fitting it costs
about five milliseconds and cutting one sentence about four, so the request
endpoint refits per call and only the sweeps are cached.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. ``_in_plain_words`` does the one rewriting a refusal needs,
and a test asserts that no identifier survives it.
"""

from __future__ import annotations

from functools import cache
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.characters.patching import (
    EntropyPatcher,
    FixedSizePatcher,
    Patches,
    PatchingRule,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from pydantic import BaseModel, Field

router = APIRouter(
    prefix="/concepts/patching-without-a-vocabulary",
    tags=["patching-without-a-vocabulary"],
)


# --------------------------------------------------------------------------
# The texts and the settings every endpoint works from.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries, and no corpus here holds."""

ACCENTED_TEXT = "Dr. Ångström expected the re-analysis."
"""A sentence with two letters that take two bytes each, so a cut can split one."""

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

The same corpus the byte pair encoding and bytes-and-characters pages are built
on, so the sequence lengths this page quotes can be read against theirs.
"""

SEEN_SENTENCE = CORPUS[0]
"""One sentence the byte model did read, for the case where the method works."""

FIXTURE_SENTENCE = "the cat sat on the mat"
"""The sentence whose repeated copies are the one corpus small enough to check.

Six words, two of which repeat, and one word that is only ever followed by one
other, which is what makes the boundary before it stay uncut.
"""

FIXTURE_WORDS: tuple[str, ...] = ("the", "cat", "sat", "on", "the", "mat")
"""Its words, so a widget can mark where a boundary would fall at a word start."""

DEFAULT_COPIES = 20
"""How many copies of a corpus the byte model behind the page is counted over."""

DEFAULT_SMOOTHING = 0.001
"""Added to every byte's count in every context. Far below the usual one."""

DEFAULT_ORDER = 4
"""How many preceding bytes the next-byte guess is conditioned on."""

DEFAULT_THRESHOLD = 1.0
"""In bits. A position above this begins a new block."""

DEFAULT_PATCH_SIZE = 4
"""The block size the fixed rule is shown at throughout the page."""

MAX_TEXT_CHARACTERS = 200
"""How long any one text may be, a few times the running sentence."""

MAX_ORDER = 6
"""The longest context a request may ask for. Beyond this nothing is ever seen."""

MAX_PATCH_SIZE = 32
"""The largest fixed block a request may ask for."""

MAX_BITS = 8.0
"""No uncertainty over 256 outcomes can exceed this, so no threshold may."""

MAX_COPIES = 40
"""The most copies of a corpus a request may have counted."""

FIXED_SIZES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
"""The block sizes the landing sweep is taken at."""

SWEEP_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
"""The thresholds the landing sweep is taken at."""

SWEEP_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The context lengths the landing sweep is taken at."""

CONDITION_COPIES: tuple[int, ...] = (1, 2, 4, 6, 7, 8, 10, 20)
"""How many copies of one sentence the corpus condition is measured over."""

CONDITION_SMOOTHINGS: tuple[float, ...] = (1.0, DEFAULT_SMOOTHING)
"""Add-one, and the far smaller value the separation actually needs."""

INSIDE_A_WORD_POSITION = 2
"""The third byte of the fixture sentence, which is inside the word ``the``."""

WORD_START_POSITION = 4
"""Its fifth byte, which is the first letter of the word after ``the``."""

REPLACEMENT = "�"
"""What a block that stops part way through a character gives back."""

SPACE_MARK = "␣"
"""A visible stand-in for a space, so a block boundary can be seen beside one."""

CONTINUATION_MARK = "·"
"""A visible stand-in for a byte that only ever continues a character."""

BoundedText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)]

Source = Literal["reports", "one sentence"]

SOURCE_LABELS: dict[Source, str] = {
    "reports": "eighteen sentences about reports and costs",
    "one sentence": "copies of one six-word sentence",
}
"""What each corpus is called on the page. No identifier appears in one."""

Rule = Literal["global_threshold", "relative_increase"]

RULE_LABELS: dict[Rule, str] = {
    "global_threshold": "wherever the uncertainty is above the threshold",
    "relative_increase": "wherever it rises by more than the threshold",
}
"""What each of the two rules is called on the page."""

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("patch_size", "the block size"),
    ("byte_ids", "byte values"),
    ("patch", "block"),
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


def _texts_of(source: Source, n_copies: int) -> list[str]:
    """The corpus a byte model is counted over, repeated ``n_copies`` times."""
    if source == "reports":
        return list(CORPUS) * n_copies
    return [FIXTURE_SENTENCE] * n_copies


def _byte_model(
    source: Source,
    order: int,
    threshold: float,
    smoothing: float,
    rule: Rule,
    n_copies: int,
) -> EntropyPatcher:
    """A byte model counted over ``source``, ready to cut a text."""
    try:
        return EntropyPatcher(
            order=order,
            threshold=threshold,
            smoothing=smoothing,
            rule=PatchingRule(rule),
        ).fit(_texts_of(source, n_copies))
    except MLLibError as refusal:  # pragma: no cover - bounds catch this first
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal


@cache
def _learned_reading() -> BytePairEncoding:
    """The vocabulary merged out of the eighteen sentences, for the length table."""
    return BytePairEncoding(byte_fallback=False, vocabulary_size=300).fit(list(CORPUS))


# --------------------------------------------------------------------------
# Where a cut falls, which is the whole comparison between the two rules.
# --------------------------------------------------------------------------

WORD_START = "at the start of a word"
INSIDE_A_WORD = "inside a word"
ELSEWHERE = "beside a mark or a space"
"""The three places a cut can land, in the page's own words."""


def _where_the_cut_falls(byte_values: bytes, position: int) -> str:
    """Which of the three places the boundary at ``position`` landed in."""
    before = chr(byte_values[position - 1])
    here = chr(byte_values[position])
    if before == " " and here != " ":
        return WORD_START
    if before.isalnum() and here.isalnum():
        return INSIDE_A_WORD
    return ELSEWHERE


def _display_of(byte_value: int) -> str:
    """What one byte is drawn as, since most of them have nothing to show."""
    if byte_value == 0x20:
        return SPACE_MARK
    if 0x21 <= byte_value <= 0x7E:
        return chr(byte_value)
    return CONTINUATION_MARK


# --------------------------------------------------------------------------
# One text, cut both ways.
# --------------------------------------------------------------------------


class PatchPiece(BaseModel):
    """One block of consecutive bytes, and where it sits in the text."""

    text: str
    start: int
    end: int
    n_bytes: int
    displays: list[str]


class PatchView(BaseModel):
    """What one rule made of one text, and where every cut it made landed.

    ``n_broken_characters`` counts the positions that came back as the mark
    meaning a character stood here and could not be read, which happens when a
    block stops part way through a character of more than one byte.
    """

    label: str
    n_patches: int
    boundaries: list[int]
    pieces: list[PatchPiece]
    at_a_word_start: int
    inside_a_word: int
    elsewhere: int
    n_broken_characters: int
    longest: int
    shortest: int


class BytePosition(BaseModel):
    """One byte of a text, what it draws as, and how hard it was to guess."""

    position: int
    byte_id: int
    display: str
    entropy: float
    begins_a_patch: bool
    at_a_word_start: bool


class PatchRequest(BaseModel):
    """A text to cut, and the settings to cut it under."""

    text: BoundedText
    source: Source = "reports"
    n_copies: int = Field(default=DEFAULT_COPIES, ge=1, le=MAX_COPIES)
    order: int = Field(default=DEFAULT_ORDER, ge=1, le=MAX_ORDER)
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0.0, le=MAX_BITS)
    smoothing: float = Field(default=DEFAULT_SMOOTHING, gt=0.0, le=1.0)
    rule: Rule = "global_threshold"
    patch_size: int = Field(default=DEFAULT_PATCH_SIZE, ge=1, le=MAX_PATCH_SIZE)

    model_config = {"extra": "forbid"}


class PatchResponse(BaseModel):
    """One text under both rules, with the uncertainty the second one reads."""

    text: str
    n_characters: int
    n_bytes: int
    source_label: str
    n_texts_counted: int
    n_contexts: int
    order: int
    threshold: float
    smoothing: float
    rule_label: str
    patch_size: int
    fixed: PatchView
    entropy: PatchView
    positions: list[BytePosition]
    highest_entropy: float
    lowest_entropy: float


def _patch_view(label: str, text: str, patches: Patches) -> PatchView:
    """One set of blocks, with its cuts sorted into the three places they fall."""
    byte_values = text.encode("utf-8")
    landed = [
        _where_the_cut_falls(byte_values, start)
        for start in patches.boundaries
        if start != 0
    ]
    sizes = list(patches.sizes) or [0]
    return PatchView(
        label=label,
        n_patches=patches.n_patches,
        boundaries=list(patches.boundaries),
        pieces=[
            PatchPiece(
                text=piece.text,
                start=piece.start,
                end=piece.end,
                n_bytes=piece.n_bytes,
                displays=[_display_of(value) for value in piece],
            )
            for piece in patches
        ],
        at_a_word_start=landed.count(WORD_START),
        inside_a_word=landed.count(INSIDE_A_WORD),
        elsewhere=landed.count(ELSEWHERE),
        n_broken_characters=sum(piece.text.count(REPLACEMENT) for piece in patches),
        longest=max(sizes),
        shortest=min(sizes),
    )


@router.post("/patch", response_model=PatchResponse)
def patch(request: PatchRequest) -> PatchResponse:
    """Cut one text into fixed blocks and into blocks a byte model chose."""
    model = _byte_model(
        request.source,
        request.order,
        request.threshold,
        request.smoothing,
        request.rule,
        request.n_copies,
    )
    try:
        entropies = model.entropies_of(request.text)
        by_model = model.patch(request.text)
        by_size = FixedSizePatcher(patch_size=request.patch_size).patch(request.text)
    except MLLibError as refusal:  # pragma: no cover - bounds catch this first
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    byte_values = request.text.encode("utf-8")
    starts = set(by_model.boundaries)
    return PatchResponse(
        text=request.text,
        n_characters=len(request.text),
        n_bytes=len(byte_values),
        source_label=SOURCE_LABELS[request.source],
        n_texts_counted=len(_texts_of(request.source, request.n_copies)),
        n_contexts=model.n_contexts,
        order=request.order,
        threshold=request.threshold,
        smoothing=request.smoothing,
        rule_label=RULE_LABELS[request.rule],
        patch_size=request.patch_size,
        fixed=_patch_view(f"every {request.patch_size} bytes", request.text, by_size),
        entropy=_patch_view(RULE_LABELS[request.rule], request.text, by_model),
        positions=[
            BytePosition(
                position=position,
                byte_id=byte_values[position],
                display=_display_of(byte_values[position]),
                entropy=entropies[position],
                begins_a_patch=position in starts,
                at_a_word_start=position > 0
                and _where_the_cut_falls(byte_values, position) == WORD_START,
            )
            for position in range(len(byte_values))
        ],
        highest_entropy=max(entropies, default=0.0),
        lowest_entropy=min(entropies, default=0.0),
    )


# --------------------------------------------------------------------------
# Where the cuts land, across the whole corpus.
# --------------------------------------------------------------------------


class LandingRow(BaseModel):
    """One setting of one rule, with every cut it made across the corpus sorted.

    ``share_inside_a_word`` is the number to read against the share of all
    positions that fall inside a word, since a rule that knows nothing about the
    text can do no better than that.
    """

    setting: float
    n_patches: int
    at_a_word_start: int
    inside_a_word: int
    elsewhere: int
    share_inside_a_word: float
    bytes_per_patch: float


class CutLandingResponse(BaseModel):
    """Both rules swept across the eighteen sentences, and the uninformed rate."""

    n_texts: int
    corpus_bytes: int
    word_starts: int
    positions_inside_a_word: int
    positions_between_bytes: int
    share_of_positions_inside_a_word: float
    order: int
    threshold: float
    fixed: list[LandingRow]
    thresholds: list[LandingRow]
    orders: list[LandingRow]


def _landing_row(setting: float, every_patches: list[Patches]) -> LandingRow:
    """One row of the sweep, totalled over the eighteen sentences."""
    at_a_word_start = 0
    inside_a_word = 0
    elsewhere = 0
    n_patches = 0
    n_bytes = 0
    for text, patches in zip(CORPUS, every_patches):
        byte_values = text.encode("utf-8")
        n_bytes += len(byte_values)
        n_patches += patches.n_patches
        for start in patches.boundaries:
            if start == 0:
                continue
            where = _where_the_cut_falls(byte_values, start)
            if where == WORD_START:
                at_a_word_start += 1
            elif where == INSIDE_A_WORD:
                inside_a_word += 1
            else:
                elsewhere += 1
    cuts = at_a_word_start + inside_a_word + elsewhere
    return LandingRow(
        setting=setting,
        n_patches=n_patches,
        at_a_word_start=at_a_word_start,
        inside_a_word=inside_a_word,
        elsewhere=elsewhere,
        share_inside_a_word=inside_a_word / cuts if cuts else 0.0,
        bytes_per_patch=n_bytes / n_patches if n_patches else 0.0,
    )


@cache
def _cut_landing() -> CutLandingResponse:
    """The whole sweep, computed once because it is the same for every reader."""
    inside_a_word = 0
    word_starts = 0
    between = 0
    for text in CORPUS:
        byte_values = text.encode("utf-8")
        for position in range(1, len(byte_values)):
            between += 1
            where = _where_the_cut_falls(byte_values, position)
            if where == INSIDE_A_WORD:
                inside_a_word += 1
            elif where == WORD_START:
                word_starts += 1

    fixed = [
        _landing_row(
            float(size),
            [FixedSizePatcher(patch_size=size).patch(text) for text in CORPUS],
        )
        for size in FIXED_SIZES
    ]
    thresholds = []
    for threshold in SWEEP_THRESHOLDS:
        model = _byte_model(
            "reports",
            DEFAULT_ORDER,
            threshold,
            DEFAULT_SMOOTHING,
            "global_threshold",
            DEFAULT_COPIES,
        )
        thresholds.append(
            _landing_row(threshold, [model.patch(text) for text in CORPUS])
        )
    orders = []
    for order in SWEEP_ORDERS:
        model = _byte_model(
            "reports",
            order,
            DEFAULT_THRESHOLD,
            DEFAULT_SMOOTHING,
            "global_threshold",
            DEFAULT_COPIES,
        )
        orders.append(
            _landing_row(float(order), [model.patch(text) for text in CORPUS])
        )

    return CutLandingResponse(
        n_texts=len(CORPUS),
        corpus_bytes=sum(len(text.encode("utf-8")) for text in CORPUS),
        word_starts=word_starts,
        positions_inside_a_word=inside_a_word,
        positions_between_bytes=between,
        share_of_positions_inside_a_word=inside_a_word / between,
        order=DEFAULT_ORDER,
        threshold=DEFAULT_THRESHOLD,
        fixed=fixed,
        thresholds=thresholds,
        orders=orders,
    )


@router.get("/cut-landing", response_model=CutLandingResponse)
def cut_landing() -> CutLandingResponse:
    """Where every cut each rule makes across the eighteen sentences lands."""
    return _cut_landing()


# --------------------------------------------------------------------------
# What the corpus has to do before any of this means anything.
# --------------------------------------------------------------------------


class ConditionRow(BaseModel):
    """One corpus size at one smoothing, and whether the words came out of it.

    ``inside_a_word`` and ``at_a_word_start`` are two positions of the same
    sentence, the third byte and the fifth, so the pair can be compared directly
    rather than through an average over positions that are not alike.
    """

    n_copies: int
    smoothing: float
    inside_a_word: float
    at_a_word_start: float
    gap: float
    lowest: float
    highest: float
    n_patches: int
    patches: list[str]
    words_recovered: bool


class OnePhraseView(BaseModel):
    """A corpus of one repeated phrase, where every position is equally sure."""

    phrase: str
    n_copies: int
    entropies: list[float]
    displays: list[str]
    patches: list[str]
    every_position_agrees: bool


class EquivalenceView(BaseModel):
    """Many copies at one smoothing against one copy at a smaller smoothing."""

    n_copies: int
    smoothing: float
    single_copy_smoothing: float
    largest_gap: float
    patches_agree: bool


class CorpusConditionResponse(BaseModel):
    """The sentence read by models counted over more and more copies of it."""

    sentence: str
    n_bytes: int
    words: list[str]
    order: int
    threshold: float
    displays: list[str]
    rows: list[ConditionRow]
    one_phrase: OnePhraseView
    equivalence: EquivalenceView


def _condition_row(n_copies: int, smoothing: float) -> ConditionRow:
    """One corpus size at one smoothing, measured on the sentence itself."""
    model = _byte_model("one sentence", 2, 0.5, smoothing, "global_threshold", n_copies)
    entropies = model.entropies_of(FIXTURE_SENTENCE)
    patches = model.patch(FIXTURE_SENTENCE)
    texts = [piece.text for piece in patches]
    return ConditionRow(
        n_copies=n_copies,
        smoothing=smoothing,
        inside_a_word=entropies[INSIDE_A_WORD_POSITION],
        at_a_word_start=entropies[WORD_START_POSITION],
        gap=entropies[WORD_START_POSITION] - entropies[INSIDE_A_WORD_POSITION],
        lowest=min(entropies),
        highest=max(entropies),
        n_patches=patches.n_patches,
        patches=texts,
        words_recovered=[text.strip() for text in texts]
        == ["the", "cat", "sat", "on the", "mat"],
    )


@cache
def _corpus_condition() -> CorpusConditionResponse:
    """The corpus-size finding, computed once."""
    rows = [
        _condition_row(n_copies, smoothing)
        for smoothing in CONDITION_SMOOTHINGS
        for n_copies in CONDITION_COPIES
    ]

    phrase = "the cat sat"
    repeated = EntropyPatcher(order=2, threshold=0.5, smoothing=DEFAULT_SMOOTHING).fit(
        [phrase] * DEFAULT_COPIES
    )
    phrase_entropies = list(repeated.entropies_of(phrase))

    many = _byte_model(
        "one sentence", 2, 0.5, DEFAULT_SMOOTHING, "global_threshold", DEFAULT_COPIES
    )
    single_smoothing = DEFAULT_SMOOTHING / DEFAULT_COPIES
    one = _byte_model("one sentence", 2, 0.5, single_smoothing, "global_threshold", 1)
    many_entropies = many.entropies_of(FIXTURE_SENTENCE)
    one_entropies = one.entropies_of(FIXTURE_SENTENCE)

    return CorpusConditionResponse(
        sentence=FIXTURE_SENTENCE,
        n_bytes=len(FIXTURE_SENTENCE.encode("utf-8")),
        words=list(FIXTURE_WORDS),
        order=2,
        threshold=0.5,
        displays=[_display_of(value) for value in FIXTURE_SENTENCE.encode("utf-8")],
        rows=rows,
        one_phrase=OnePhraseView(
            phrase=phrase,
            n_copies=DEFAULT_COPIES,
            entropies=phrase_entropies,
            displays=[_display_of(value) for value in phrase.encode("utf-8")],
            patches=[piece.text for piece in repeated.patch(phrase)],
            every_position_agrees=max(phrase_entropies) - min(phrase_entropies) < 1e-12,
        ),
        equivalence=EquivalenceView(
            n_copies=DEFAULT_COPIES,
            smoothing=DEFAULT_SMOOTHING,
            single_copy_smoothing=single_smoothing,
            largest_gap=max(
                abs(left - right) for left, right in zip(many_entropies, one_entropies)
            ),
            patches_agree=many.patch(FIXTURE_SENTENCE) == one.patch(FIXTURE_SENTENCE),
        ),
    )


@router.get("/corpus-condition", response_model=CorpusConditionResponse)
def corpus_condition() -> CorpusConditionResponse:
    """What a byte model needs before its uncertainties say anything at all."""
    return _corpus_condition()


# --------------------------------------------------------------------------
# The lengths, the model dependence, and the small worked example.
# --------------------------------------------------------------------------


class LengthRow(BaseModel):
    """One way of reading a text, and how many pieces it comes to."""

    label: str
    sentence: int
    corpus: int
    table_entries: int | None


class SmallExample(BaseModel):
    """Four bytes read by a model counted over three, checkable by hand.

    The corpus is the single text ``aab`` at a context of one byte and the
    ordinary add-one smoothing, which is small enough that every number here has
    a closed form a reader can evaluate.
    """

    corpus: str
    text: str
    displays: list[str]
    entropies: list[float]
    n_contexts: int
    patches_at_a_high_threshold: list[str]
    threshold: float


class ModelRow(BaseModel):
    """One byte model, and what it made of a sentence every model here reads."""

    label: str
    n_contexts: int
    patches: list[str]
    n_patches: int
    held_out_patches: int


class RuleRow(BaseModel):
    """One of the two rules for high, on a sentence the model read and one it did not."""

    label: str
    seen: list[str]
    unseen: list[str]
    corpus_patches: int


class RepeatedPiece(BaseModel):
    """One spelling that turned up more than once as a block of its own."""

    text: str
    n_occurrences: int


class BrokenCharacter(BaseModel):
    """A text whose fixed blocks stop part way through a character."""

    text: str
    n_characters: int
    n_bytes: int
    patch_size: int
    pieces: list[str]
    n_broken_characters: int


class LimitsResponse(BaseModel):
    """Everything the closing parts of the page quote."""

    sentence: str
    seen_sentence: str
    order: int
    threshold: float
    smoothing: float
    n_copies: int
    lengths: list[LengthRow]
    small_example: SmallExample
    models: list[ModelRow]
    rules: list[RuleRow]
    distinct_patch_spellings: int
    total_patches: int
    repeated_pieces: list[RepeatedPiece]
    broken: BrokenCharacter
    unseen_bytes_at_the_top: int
    sentence_bytes_at_the_top: int


@cache
def _limits() -> LimitsResponse:
    """Every closing figure, computed once."""
    model = _byte_model(
        "reports",
        DEFAULT_ORDER,
        DEFAULT_THRESHOLD,
        DEFAULT_SMOOTHING,
        "global_threshold",
        DEFAULT_COPIES,
    )
    learned = _learned_reading()
    corpus_patches = [model.patch(text) for text in CORPUS]

    spellings: dict[str, int] = {}
    for patches in corpus_patches:
        for piece in patches:
            spellings[piece.text] = spellings.get(piece.text, 0) + 1

    small = EntropyPatcher(order=1, threshold=1.0, smoothing=1.0).fit(["aab"])
    small_text = "aabb"
    high = EntropyPatcher(order=1, threshold=7.996, smoothing=1.0).fit(["aab"])

    models: list[ModelRow] = []
    for label, order, texts in (
        ("two bytes back", 2, list(CORPUS)),
        ("three bytes back", 3, list(CORPUS)),
        ("four bytes back", 4, list(CORPUS)),
        ("four bytes back, half the sentences", 4, list(CORPUS[:9])),
    ):
        each = EntropyPatcher(
            order=order, threshold=DEFAULT_THRESHOLD, smoothing=DEFAULT_SMOOTHING
        ).fit(texts * DEFAULT_COPIES)
        models.append(
            ModelRow(
                label=label,
                n_contexts=each.n_contexts,
                patches=[piece.text for piece in each.patch(SEEN_SENTENCE)],
                n_patches=each.patch(SEEN_SENTENCE).n_patches,
                held_out_patches=each.patch(RUNNING_SENTENCE).n_patches,
            )
        )

    rules: list[RuleRow] = []
    for rule in ("global_threshold", "relative_increase"):
        each = _byte_model(
            "reports",
            DEFAULT_ORDER,
            DEFAULT_THRESHOLD,
            DEFAULT_SMOOTHING,
            rule,
            DEFAULT_COPIES,
        )
        rules.append(
            RuleRow(
                label=RULE_LABELS[rule],
                seen=[piece.text for piece in each.patch(SEEN_SENTENCE)],
                unseen=[piece.text for piece in each.patch(RUNNING_SENTENCE)],
                corpus_patches=sum(each.patch(text).n_patches for text in CORPUS),
            )
        )

    broken_patches = FixedSizePatcher(patch_size=DEFAULT_PATCH_SIZE).patch(
        ACCENTED_TEXT
    )

    entropies = model.entropies_of(RUNNING_SENTENCE)
    return LimitsResponse(
        sentence=RUNNING_SENTENCE,
        seen_sentence=SEEN_SENTENCE,
        order=DEFAULT_ORDER,
        threshold=DEFAULT_THRESHOLD,
        smoothing=DEFAULT_SMOOTHING,
        n_copies=DEFAULT_COPIES,
        lengths=[
            LengthRow(
                label="one number per byte",
                sentence=len(RUNNING_SENTENCE.encode("utf-8")),
                corpus=sum(len(text.encode("utf-8")) for text in CORPUS),
                table_entries=256,
            ),
            LengthRow(
                label=f"blocks of {DEFAULT_PATCH_SIZE} bytes",
                sentence=FixedSizePatcher(patch_size=DEFAULT_PATCH_SIZE)
                .patch(RUNNING_SENTENCE)
                .n_patches,
                corpus=sum(
                    FixedSizePatcher(patch_size=DEFAULT_PATCH_SIZE)
                    .patch(text)
                    .n_patches
                    for text in CORPUS
                ),
                table_entries=None,
            ),
            LengthRow(
                label="blocks cut where the next byte was hard to guess",
                sentence=model.patch(RUNNING_SENTENCE).n_patches,
                corpus=sum(patches.n_patches for patches in corpus_patches),
                table_entries=None,
            ),
            LengthRow(
                label="pieces merged from the eighteen sentences",
                sentence=learned.encode(RUNNING_SENTENCE).n_tokens,
                corpus=sum(learned.encode(text).n_tokens for text in CORPUS),
                table_entries=learned.vocabulary.n_tokens,
            ),
        ],
        small_example=SmallExample(
            corpus="aab",
            text=small_text,
            displays=[_display_of(value) for value in small_text.encode("utf-8")],
            entropies=list(small.entropies_of(small_text)),
            n_contexts=small.n_contexts,
            patches_at_a_high_threshold=[
                piece.text for piece in high.patch(small_text)
            ],
            threshold=7.996,
        ),
        models=models,
        rules=rules,
        distinct_patch_spellings=len(spellings),
        total_patches=sum(spellings.values()),
        repeated_pieces=sorted(
            (
                RepeatedPiece(text=text, n_occurrences=count)
                for text, count in spellings.items()
                if count > 1
            ),
            key=lambda piece: (-piece.n_occurrences, piece.text),
        )[:8],
        broken=BrokenCharacter(
            text=ACCENTED_TEXT,
            n_characters=len(ACCENTED_TEXT),
            n_bytes=len(ACCENTED_TEXT.encode("utf-8")),
            patch_size=DEFAULT_PATCH_SIZE,
            pieces=[piece.text for piece in broken_patches],
            n_broken_characters=sum(
                piece.text.count(REPLACEMENT) for piece in broken_patches
            ),
        ),
        unseen_bytes_at_the_top=sum(
            1 for entropy in entropies if entropy >= MAX_BITS - 1e-9
        ),
        sentence_bytes_at_the_top=len(RUNNING_SENTENCE.encode("utf-8")),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """The lengths, the model dependence and the worked example the page quotes."""
    return _limits()
