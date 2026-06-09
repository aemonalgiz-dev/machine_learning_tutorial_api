"""The Morfessor page's endpoints.

Five of them, one per family of widgets rather than one per number.
``/counts`` is everything about where a count-driven vocabulary cuts: the merge
ladder on the twelve inflected forms, piece by learned piece, and the same two
methods set against each other on eighteen sentences of ordinary English.
``/cost`` is the two-part code itself, the two-word example worked small enough
to check by hand and the three lexicons of the twelve forms priced beside each
other. ``/search`` is the greedy walk, epoch by epoch on both corpora, with the
Viterbi cut of a handful of words the corpus never held. ``/stall`` is the
finding this page is built around, that the same corpus repeated more times is
segmented worse, and it carries the repeat sweep, every first split at two
repeat counts, and the weight that lets the search through. ``/fit`` is the
playground.

Two things here are the page's own rather than the library's. The first is the
enumeration of every binary first split of every word from the whole-word
start, which no fit needs, since the search only ever asks for the best one and
throws the rest away; showing that a search stalled means listing what it had
to choose from. The second is the epoch trace, built by fitting the same corpus
under a rising epoch bound, since a fit reports where it stopped and not what
it held on the way. Everything else is a fit.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from functools import cache, lru_cache

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.morphology.morfessor import (
    DescriptionLength,
    MorfessorBaseline,
    Morph,
    Morphs,
    description_length,
)
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/morfessor", tags=["morfessor"])


END_OF_WORD = "</w>"
"""The mark every fit on this page writes on a word's last piece."""

UNKNOWN = "[UNK]"
"""What a fit answers when no piece it holds covers a character."""

SEED = 0
"""The seed every fit on this page uses, so the page's numbers do not move."""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to read."""

MAX_WORD_LENGTH = 40
"""The longest single word the playground may ask to cut."""

MAX_REPEATS = 8
"""How many times over the playground may repeat its small corpus."""

MIN_WEIGHT = 0.05
MAX_WEIGHT = 4.0
"""What the playground's dial on the corpus half of the cost may be set to."""


# --------------------------------------------------------------------------
# The corpora the page works with.
# --------------------------------------------------------------------------

INFLECTED_FORMS: tuple[str, ...] = (
    "walk",
    "walks",
    "walked",
    "walking",
    "talk",
    "talks",
    "talked",
    "talking",
    "play",
    "plays",
    "played",
    "playing",
)
"""Three verbs in four forms each, which is this method's own spec fixture.

Chosen because the joints are not in dispute: every reader agrees that walking
is walk followed by ing, so a vocabulary that cuts it anywhere else can be shown
to be wrong without appealing to a linguist.
"""

DEFAULT_REPEATS = 3
"""How many times each form occurs in the corpus the page works on.

Three, because the search succeeds there and fails at five, and the page is
about the difference.
"""

STALL_REPEATS = 5
"""The repeat count at which the same twelve forms are left whole."""

PREFIXED_WORDS: tuple[str, ...] = ("undo", "untie", "redo", "retie")
"""Two prefixes on two roots, every combination, ten times each.

The corpus on which the principle itself declines the morphemes rather than the
search failing to find them, which is the closing part's example.
"""

PREFIX_REPEATS = 10

SENTENCE_CORPUS: list[str] = [
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
"""Eighteen sentences of ordinary English, none of them the running sentence.

The same eighteen the sibling pages on merging carry, kept here rather than
imported so the pages cannot move each other's numbers.
"""

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

SENTENCE_WEIGHT = 0.5
"""The setting at which the eighteen sentences yield stems and endings."""

READABLE_WEIGHT = 0.25
"""The setting at which the same eighteen sentences keep their characters."""

MERGED_SENTENCE_SIZE = 135
"""The merged vocabulary set beside the sentence fit, matched row for row."""

LADDER_SIZES: tuple[int, ...] = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30)
"""The merged vocabulary sizes the twelve forms are read at."""

LADDER_WORDS: tuple[str, ...] = ("walking", "plays", "talked")
"""The three forms whose cut is shown at every rung of the ladder."""

COMPARED_WORDS: tuple[str, ...] = (
    "lower",
    "lowest",
    "costing",
    "costs",
    "analysts",
    "reviewers",
    "reran",
    "readings",
    "expected",
    "rewrote",
)
"""Words of the eighteen sentences both methods are asked to cut."""

UNSEEN_WORDS: tuple[str, ...] = (
    "walkings",
    "stalking",
    "splay",
    "playful",
    "walkxed",
    "jumping",
)
"""Words the twelve forms never held, cut by what they taught."""

REPEAT_SWEEP: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
"""The repeat counts the twelve forms are fitted at."""

SPLIT_REPEATS: tuple[int, ...] = (DEFAULT_REPEATS, STALL_REPEATS)
"""The two repeat counts whose every first split the page draws."""

WEIGHT_SWEEP: tuple[float, ...] = (
    0.25,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.83,
    0.9,
    1.0,
    1.5,
    2.0,
)
"""The settings of the dial the five-repeat corpus is fitted at."""

SENTENCE_EPOCH_BOUND = 9
"""How far the epoch trace on the eighteen sentences runs."""


class CorpusChoice(StrEnum):
    """Which of the page's three corpora a request wants fitted."""

    INFLECTIONS = "inflections"
    SENTENCES = "sentences"
    PREFIXES = "prefixes"


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("corpus_weight", "the weight on the text half of the cost"),
    ("convergence_threshold", "how small a change ends the search"),
    ("max_epochs", "how many passes the search may make"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("end_of_word_marker", "the mark that records a word ending"),
    ("unknown_token", "one piece for anything unspellable"),
    ("random_seed", "the seed the visiting order is drawn from"),
    ("best_segmentation", "asking for the cheapest cut"),
    ("MorfessorBaseline", "this method"),
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


def _inflection_corpus(repeats: int) -> list[str]:
    """The twelve forms, each ``repeats`` times, as one line of text."""
    return [" ".join(word for word in INFLECTED_FORMS for _ in range(repeats))]


def _prefix_corpus() -> list[str]:
    return [" ".join(word for word in PREFIXED_WORDS for _ in range(PREFIX_REPEATS))]


def _corpus_for(choice: CorpusChoice, repeats: int) -> list[str]:
    if choice is CorpusChoice.SENTENCES:
        return SENTENCE_CORPUS
    if choice is CorpusChoice.PREFIXES:
        return _prefix_corpus()
    return _inflection_corpus(repeats)


def _word_counts(corpus: Sequence[str]) -> dict[str, int]:
    counts = Corpus.of(list(corpus)).word_counts(WhitespacePreTokenizer())
    return {word_count.word: word_count.count for word_count in counts}


def _whole_word_lexicon(words: Sequence[str], repeats: int) -> Morphs:
    """Every word its own piece, which is where every search here starts."""
    return Morphs([Morph(word, repeats) for word in words])


def _character_lexicon(words: Sequence[str], repeats: int) -> Morphs:
    """Every character its own piece, which is the other extreme."""
    characters: Counter[str] = Counter()
    for word in words:
        for character in word:
            characters[character] += repeats
    return Morphs([Morph(text, count) for text, count in characters.items()])


def _stem_and_suffix_lexicon(repeats: int) -> Morphs:
    """The six morphs of the twelve forms, at whatever repeat count."""
    return Morphs(
        [
            Morph("walk", 4 * repeats),
            Morph("talk", 4 * repeats),
            Morph("play", 4 * repeats),
            Morph("s", 3 * repeats),
            Morph("ed", 3 * repeats),
            Morph("ing", 3 * repeats),
        ]
    )


def _symbols_of(words: Sequence[str]) -> set[str]:
    """The symbols a merging fit starts from, a word ending kept apart.

    A merge-grown vocabulary marks the last character of a word, so the w that
    ends one word and the w inside another are two rows rather than one, and
    the floor of that vocabulary is this set plus one row for the unspellable.
    """
    symbols: set[str] = set()
    for word in words:
        symbols.update(word[:-1])
        symbols.add(word[-1] + END_OF_WORD)
    return symbols


def _lexicon_characters(morphs: Morphs) -> int:
    """Characters across the types, plus one end symbol per type.

    The count the spelling half of the code is paid on, worth returning because
    the page walks through the formula and a reader wants to check the number
    that goes into it.
    """
    return sum(len(morph.text) for morph in morphs) + morphs.n_morphs


@lru_cache(maxsize=64)
def _fit(
    choice: CorpusChoice, repeats: int, weight: float, epochs: int
) -> MorfessorBaseline:
    return MorfessorBaseline(
        random_seed=SEED, corpus_weight=weight, max_epochs=epochs
    ).fit(_corpus_for(choice, repeats))


@lru_cache(maxsize=32)
def _merged(choice: CorpusChoice, repeats: int, size: int) -> BytePairEncoding:
    return BytePairEncoding(byte_fallback=False, vocabulary_size=size).fit(
        _corpus_for(choice, repeats)
    )


def _corpus_pieces(model: MorfessorBaseline, counts: dict[str, int]) -> int:
    return sum(model.encode(word).n_tokens * count for word, count in counts.items())


def _merged_pieces(model: BytePairEncoding, counts: dict[str, int]) -> int:
    return sum(model.encode(word).n_tokens * count for word, count in counts.items())


# --------------------------------------------------------------------------
# The shapes several endpoints answer with.
# --------------------------------------------------------------------------


class CostBreakdown(BaseModel):
    """The three parts of one description, and what they add to.

    ``spelling_cost`` writes the list of pieces out, ``count_cost`` says how
    often each is used, and ``corpus_cost`` writes the text as a run of
    references. Every figure is in nats, since the logarithms are natural.
    """

    n_types: int
    n_tokens: int
    lexicon_characters: int
    spelling_cost: float
    count_cost: float
    lexicon_cost: float
    corpus_cost: float
    total: float


class MorphRow(BaseModel):
    """One piece of a lexicon and how many times the text uses it."""

    text: str
    count: int


class WordCut(BaseModel):
    """One word and the pieces it was read as."""

    word: str
    pieces: list[str]
    n_pieces: int


class SegmentationRow(BaseModel):
    """One word cut by a fitted lexicon, with what that cut costs in nats."""

    word: str
    pieces: list[str]
    n_pieces: int
    cost: float
    n_unknown: int


def _breakdown(cost: DescriptionLength, morphs: Morphs) -> CostBreakdown:
    return CostBreakdown(
        n_types=morphs.n_morphs,
        n_tokens=morphs.total,
        lexicon_characters=_lexicon_characters(morphs),
        spelling_cost=cost.spelling_cost,
        count_cost=cost.count_cost,
        lexicon_cost=cost.lexicon_cost,
        corpus_cost=cost.corpus_cost,
        total=cost.total,
    )


def _cut(model: BytePairEncoding | MorfessorBaseline, word: str) -> WordCut:
    pieces = list(model.encode(word).texts)
    return WordCut(word=word, pieces=pieces, n_pieces=len(pieces))


# --------------------------------------------------------------------------
# /counts: where a vocabulary chosen by counts puts its cuts.
# --------------------------------------------------------------------------


class MergeRung(BaseModel):
    """One size of a merge-grown vocabulary, and what it reads the corpus as.

    ``learned`` is the piece this rung bought over the one below it, and is null
    at the floor, where the vocabulary is the alphabet and nothing has been
    merged yet.
    """

    size: int
    learned: str | None
    corpus_pieces: int
    cuts: list[WordCut]


class MethodPair(BaseModel):
    """One word read by both methods, so the two cuts sit side by side."""

    word: str
    morph_pieces: list[str]
    merged_pieces: list[str]


class SentenceReading(BaseModel):
    """One fitted table asked to read the running sentence.

    ``method`` is ``shortest_description`` where the pieces were chosen to
    shorten a two-part code and ``commonest_pair`` where they were grown by
    joining the commonest adjacent pair, and ``weight`` is null for the second,
    which has no such dial. ``characters_held`` counts how many of the corpus's
    own characters the table can spell on their own, which is what decides
    whether an unfamiliar word survives.
    """

    method: str
    weight: float | None
    n_rows: int
    corpus_pieces: int
    pieces: list[str]
    n_pieces: int
    n_unknown: int
    round_trip: bool
    characters_held: int
    characters_missing: list[str]
    n_characters: int


class CountsResponse(BaseModel):
    """What merging by counts learns, on both of the page's corpora."""

    words: list[str]
    repeats: int
    n_occurrences: int
    n_characters: int
    n_symbols: int
    smallest_merged_size: int
    ladder: list[MergeRung]
    morph_rows: int
    morph_types: int
    morph_pieces: int
    morph_cuts: list[WordCut]
    matched_size: int
    matched_learned: list[str]
    sentence: str
    n_distinct_words: int
    n_word_occurrences: int
    merged_size: int
    merged_sentence_pieces: list[str]
    merged_sentence_n: int
    pairs: list[MethodPair]
    readings: list[SentenceReading]


@cache
def _counts_view() -> CountsResponse:
    counts = _word_counts(_inflection_corpus(DEFAULT_REPEATS))
    characters = {character for word in counts for character in word}
    symbols = _symbols_of(list(counts))

    ladder: list[MergeRung] = []
    previous: set[str] = set()
    for size in LADDER_SIZES:
        merged = _merged(CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, size)
        tokens = set(merged.vocabulary)
        gained = sorted(tokens - previous) if previous else []
        previous = tokens
        ladder.append(
            MergeRung(
                size=merged.vocabulary.n_tokens,
                learned=gained[0] if len(gained) == 1 else None,
                corpus_pieces=_merged_pieces(merged, counts),
                cuts=[_cut(merged, word) for word in LADDER_WORDS],
            )
        )

    model = _fit(CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, 1.0, 10)
    morph_pieces = _corpus_pieces(model, counts)

    matched_size = LADDER_SIZES[0]
    for size in range(LADDER_SIZES[0], 64):
        merged = _merged(CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, size)
        matched_size = merged.vocabulary.n_tokens
        if _merged_pieces(merged, counts) <= morph_pieces:
            break
    matched = _merged(CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, matched_size)

    sentence_counts = _word_counts(SENTENCE_CORPUS)
    sentence_merged = _merged(
        CorpusChoice.SENTENCES, DEFAULT_REPEATS, MERGED_SENTENCE_SIZE
    )
    sentence_model = _fit(CorpusChoice.SENTENCES, DEFAULT_REPEATS, SENTENCE_WEIGHT, 10)
    encoded = sentence_merged.encode(RUNNING_SENTENCE)

    sentence_characters = {character for word in sentence_counts for character in word}
    readings: list[SentenceReading] = []
    for weight in (SENTENCE_WEIGHT, READABLE_WEIGHT):
        fitted = _fit(CorpusChoice.SENTENCES, DEFAULT_REPEATS, weight, 10)
        reading = fitted.encode(RUNNING_SENTENCE)
        readings.append(
            SentenceReading(
                method="shortest_description",
                weight=weight,
                n_rows=fitted.vocabulary.n_tokens,
                corpus_pieces=_corpus_pieces(fitted, sentence_counts),
                pieces=list(reading.texts),
                n_pieces=reading.n_tokens,
                n_unknown=sum(1 for piece in reading.texts if piece == UNKNOWN),
                round_trip=fitted.decode(reading.ids) == RUNNING_SENTENCE,
                characters_held=len(sentence_characters & set(fitted.morphs.texts)),
                characters_missing=sorted(
                    sentence_characters - set(fitted.morphs.texts)
                ),
                n_characters=len(sentence_characters),
            )
        )
    readings.append(
        SentenceReading(
            method="commonest_pair",
            weight=None,
            n_rows=sentence_merged.vocabulary.n_tokens,
            corpus_pieces=_merged_pieces(sentence_merged, sentence_counts),
            pieces=list(encoded.texts),
            n_pieces=encoded.n_tokens,
            n_unknown=sum(1 for piece in encoded.texts if piece == UNKNOWN),
            round_trip=sentence_merged.decode(encoded.ids) == RUNNING_SENTENCE,
            characters_held=len(sentence_characters & set(sentence_merged.vocabulary)),
            characters_missing=sorted(
                sentence_characters - set(sentence_merged.vocabulary)
            ),
            n_characters=len(sentence_characters),
        )
    )

    return CountsResponse(
        words=list(INFLECTED_FORMS),
        repeats=DEFAULT_REPEATS,
        n_occurrences=sum(counts.values()),
        n_characters=len(characters),
        n_symbols=len(symbols),
        smallest_merged_size=len(symbols) + 1,
        ladder=ladder,
        morph_rows=model.vocabulary.n_tokens,
        morph_types=model.morphs.n_morphs,
        morph_pieces=morph_pieces,
        morph_cuts=[
            WordCut(
                word=segmentation.word,
                pieces=list(segmentation.morphs),
                n_pieces=segmentation.n_morphs,
            )
            for segmentation in model.segmentations
        ],
        matched_size=matched.vocabulary.n_tokens,
        matched_learned=[
            token
            for token in matched.vocabulary
            if token != UNKNOWN and len(token.replace(END_OF_WORD, "")) > 1
        ],
        sentence=RUNNING_SENTENCE,
        n_distinct_words=len(sentence_counts),
        n_word_occurrences=sum(sentence_counts.values()),
        merged_size=sentence_merged.vocabulary.n_tokens,
        merged_sentence_pieces=list(encoded.texts),
        merged_sentence_n=encoded.n_tokens,
        pairs=[
            MethodPair(
                word=word,
                morph_pieces=list(sentence_model.encode(word).texts),
                merged_pieces=list(sentence_merged.encode(word).texts),
            )
            for word in COMPARED_WORDS
        ],
        readings=readings,
    )


@router.get("/counts", response_model=CountsResponse)
def counts() -> CountsResponse:
    """The merge ladder on twelve inflected forms, and the same words compared."""
    return _counts_view()


# --------------------------------------------------------------------------
# /cost: the two-part code, priced.
# --------------------------------------------------------------------------


class NamedLexicon(BaseModel):
    """One choice of pieces for one corpus, with what describing it costs."""

    label: str
    morphs: list[MorphRow]
    breakdown: CostBreakdown


class CostResponse(BaseModel):
    """The two-word example worked by hand, and the three lexicons priced."""

    two_word_corpus: list[str]
    two_word_whole: NamedLexicon
    two_word_split: NamedLexicon
    two_word_saving: float
    words: list[str]
    repeats: int
    lexicons: list[NamedLexicon]
    nat_in_bits: float
    prefixed_words: list[str]
    prefix_repeats: int
    prefix_whole: NamedLexicon
    prefix_split: NamedLexicon
    prefix_penalty: float
    prefix_found: list[MorphRow]


@cache
def _cost_view() -> CostResponse:
    whole_pair = Morphs([Morph("walk", 1), Morph("walks", 1)])
    split_pair = Morphs([Morph("walk", 2), Morph("s", 1)])
    whole_cost = description_length(whole_pair)
    split_cost = description_length(split_pair)

    def named(label: str, morphs: Morphs) -> NamedLexicon:
        return NamedLexicon(
            label=label,
            morphs=[MorphRow(text=morph.text, count=morph.count) for morph in morphs],
            breakdown=_breakdown(description_length(morphs), morphs),
        )

    prefix_whole = _whole_word_lexicon(PREFIXED_WORDS, PREFIX_REPEATS)
    prefix_split = Morphs(
        [
            Morph("un", 2 * PREFIX_REPEATS),
            Morph("re", 2 * PREFIX_REPEATS),
            Morph("do", 2 * PREFIX_REPEATS),
            Morph("tie", 2 * PREFIX_REPEATS),
        ]
    )
    prefix_model = _fit(CorpusChoice.PREFIXES, PREFIX_REPEATS, 1.0, 10)

    return CostResponse(
        two_word_corpus=["walk", "walks"],
        two_word_whole=named("both words whole", whole_pair),
        two_word_split=named("a shared stem and an ending", split_pair),
        two_word_saving=whole_cost.total - split_cost.total,
        words=list(INFLECTED_FORMS),
        repeats=DEFAULT_REPEATS,
        lexicons=[
            named(
                "one piece per word",
                _whole_word_lexicon(INFLECTED_FORMS, DEFAULT_REPEATS),
            ),
            named(
                "three stems and three endings",
                _stem_and_suffix_lexicon(DEFAULT_REPEATS),
            ),
            named(
                "one piece per character",
                _character_lexicon(INFLECTED_FORMS, DEFAULT_REPEATS),
            ),
        ],
        nat_in_bits=1.0 / math.log(2.0),
        prefixed_words=list(PREFIXED_WORDS),
        prefix_repeats=PREFIX_REPEATS,
        prefix_whole=named("every word whole", prefix_whole),
        prefix_split=named("the prefixes and the roots", prefix_split),
        prefix_penalty=description_length(prefix_split).total
        - description_length(prefix_whole).total,
        prefix_found=[
            MorphRow(text=morph.text, count=morph.count)
            for morph in prefix_model.morphs
        ],
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """Two words worked by hand, and three ways of describing twelve."""
    return _cost_view()


# --------------------------------------------------------------------------
# /search: the greedy walk, epoch by epoch.
# --------------------------------------------------------------------------


class EpochRow(BaseModel):
    """Where the search stood after a stated number of passes.

    Built by fitting under a rising bound on the passes, since a finished fit
    reports where it stopped rather than what it held on the way.
    """

    epoch: int
    n_morphs: int
    cost: float
    converged: bool
    gained: list[str]
    lost: list[str]


class SearchResponse(BaseModel):
    """The walk on both corpora, and what the finished lexicon does to new words."""

    words: list[str]
    repeats: int
    start_cost: float
    final_cost: float
    epochs_run: int
    converged: bool
    morphs: list[MorphRow]
    segmentations: list[WordCut]
    n_tokens: int
    inflection_epochs: list[EpochRow]
    sentence_weight: float
    sentence_epochs: list[EpochRow]
    sentence_final_morphs: int
    unseen: list[SegmentationRow]


def _epoch_rows(
    choice: CorpusChoice, repeats: int, weight: float, bound: int
) -> list[EpochRow]:
    rows: list[EpochRow] = []
    previous: set[str] = set()
    for epoch in range(1, bound + 1):
        model = _fit(choice, repeats, weight, epoch)
        held = {morph.text for morph in model.morphs}
        rows.append(
            EpochRow(
                epoch=model.epochs_run,
                n_morphs=len(held),
                cost=model.cost,
                converged=model.converged,
                gained=sorted(held - previous),
                lost=sorted(previous - held),
            )
        )
        previous = held
        if model.converged:
            break
    return rows


@cache
def _search_view() -> SearchResponse:
    model = _fit(CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, 1.0, 10)
    start = description_length(_whole_word_lexicon(INFLECTED_FORMS, DEFAULT_REPEATS))
    sentence_model = _fit(CorpusChoice.SENTENCES, DEFAULT_REPEATS, READABLE_WEIGHT, 10)

    unseen: list[SegmentationRow] = []
    for word in UNSEEN_WORDS:
        segmentation = model.best_segmentation(word)
        unseen.append(
            SegmentationRow(
                word=word,
                pieces=list(segmentation.morphs),
                n_pieces=segmentation.n_morphs,
                cost=segmentation.cost,
                n_unknown=sum(
                    1 for piece in segmentation.morphs if piece not in model.morphs
                ),
            )
        )

    return SearchResponse(
        words=list(INFLECTED_FORMS),
        repeats=DEFAULT_REPEATS,
        start_cost=start.total,
        final_cost=model.cost,
        epochs_run=model.epochs_run,
        converged=model.converged,
        morphs=[MorphRow(text=m.text, count=m.count) for m in model.morphs],
        segmentations=[
            WordCut(
                word=segmentation.word,
                pieces=list(segmentation.morphs),
                n_pieces=segmentation.n_morphs,
            )
            for segmentation in model.segmentations
        ],
        n_tokens=model.morphs.total,
        inflection_epochs=_epoch_rows(
            CorpusChoice.INFLECTIONS, DEFAULT_REPEATS, 1.0, 4
        ),
        sentence_weight=READABLE_WEIGHT,
        sentence_epochs=_epoch_rows(
            CorpusChoice.SENTENCES,
            DEFAULT_REPEATS,
            READABLE_WEIGHT,
            SENTENCE_EPOCH_BOUND,
        ),
        sentence_final_morphs=sentence_model.morphs.n_morphs,
        unseen=unseen,
    )


@router.get("/search", response_model=SearchResponse)
def search() -> SearchResponse:
    """The walk on twelve forms and on eighteen sentences, pass by pass."""
    return _search_view()


# --------------------------------------------------------------------------
# /stall: the same corpus, more times over, segmented worse.
# --------------------------------------------------------------------------


class SplitRow(BaseModel):
    """One binary first split of one word, and what it does to the total.

    ``delta`` is the description length with that one word cut in two, minus
    the description length with every word whole, so a negative figure is a
    move the search can take and a positive one is a move it cannot.
    """

    word: str
    left: str
    right: str
    delta: float


class SplitSet(BaseModel):
    """Every first split available at one repeat count, cheapest first."""

    repeats: int
    whole_cost: float
    n_splits: int
    n_downhill: int
    rows: list[SplitRow]


class RepeatRow(BaseModel):
    """One repeat count, what the search found there, and its best first move."""

    repeats: int
    n_morphs: int
    corpus_pieces: int
    cost: float
    epochs_run: int
    best_delta: float
    best_word: str
    best_left: str
    best_right: str


class WeightRow(BaseModel):
    """One setting of the dial, and the lexicon the five-repeat corpus yields."""

    weight: float
    n_morphs: int
    corpus_pieces: int
    cost: float
    epochs_run: int
    found_the_morphs: bool


class StallResponse(BaseModel):
    """The finding this page is built around, measured at both repeat counts."""

    words: list[str]
    repeats_that_work: int
    repeats_that_stall: int
    repeat_rows: list[RepeatRow]
    split_sets: list[SplitSet]
    stalled_cost: float
    stem_and_suffix_cost: float
    gap: float
    weights: list[WeightRow]
    threshold_below: float
    threshold_above: float
    escape_weight: float
    escape_cost: float


def _split_rows(repeats: int) -> SplitSet:
    """Every way of cutting exactly one word in two, priced against the start."""
    base = description_length(_whole_word_lexicon(INFLECTED_FORMS, repeats))
    rows: list[SplitRow] = []
    for word in INFLECTED_FORMS:
        for position in range(1, len(word)):
            counts = {each: repeats for each in INFLECTED_FORMS}
            del counts[word]
            for piece in (word[:position], word[position:]):
                counts[piece] = counts.get(piece, 0) + repeats
            after = description_length(
                Morphs([Morph(text, count) for text, count in counts.items()])
            )
            rows.append(
                SplitRow(
                    word=word,
                    left=word[:position],
                    right=word[position:],
                    delta=after.total - base.total,
                )
            )
    rows.sort(key=lambda row: (row.delta, row.word, row.left))
    return SplitSet(
        repeats=repeats,
        whole_cost=base.total,
        n_splits=len(rows),
        n_downhill=sum(1 for row in rows if row.delta < 0.0),
        rows=rows,
    )


@cache
def _stall_view() -> StallResponse:
    repeat_rows: list[RepeatRow] = []
    for repeats in REPEAT_SWEEP:
        model = _fit(CorpusChoice.INFLECTIONS, repeats, 1.0, 10)
        best = min(_split_rows(repeats).rows, key=lambda row: row.delta)
        repeat_rows.append(
            RepeatRow(
                repeats=repeats,
                n_morphs=model.morphs.n_morphs,
                corpus_pieces=model.morphs.total,
                cost=model.cost,
                epochs_run=model.epochs_run,
                best_delta=best.delta,
                best_word=best.word,
                best_left=best.left,
                best_right=best.right,
            )
        )

    stalled = _fit(CorpusChoice.INFLECTIONS, STALL_REPEATS, 1.0, 10)
    stems = description_length(_stem_and_suffix_lexicon(STALL_REPEATS))

    weights: list[WeightRow] = []
    for weight in WEIGHT_SWEEP:
        model = _fit(CorpusChoice.INFLECTIONS, STALL_REPEATS, weight, 10)
        weights.append(
            WeightRow(
                weight=weight,
                n_morphs=model.morphs.n_morphs,
                corpus_pieces=model.morphs.total,
                cost=model.cost,
                epochs_run=model.epochs_run,
                found_the_morphs=model.morphs
                == _stem_and_suffix_lexicon(STALL_REPEATS),
            )
        )

    below, above = 0.5, 1.0
    for _ in range(16):
        middle = (below + above) / 2.0
        model = _fit(CorpusChoice.INFLECTIONS, STALL_REPEATS, middle, 10)
        if model.morphs == _stem_and_suffix_lexicon(STALL_REPEATS):
            below = middle
        else:
            above = middle

    escape = _fit(CorpusChoice.INFLECTIONS, STALL_REPEATS, SENTENCE_WEIGHT, 10)
    return StallResponse(
        words=list(INFLECTED_FORMS),
        repeats_that_work=DEFAULT_REPEATS,
        repeats_that_stall=STALL_REPEATS,
        repeat_rows=repeat_rows,
        split_sets=[_split_rows(repeats) for repeats in SPLIT_REPEATS],
        stalled_cost=stalled.cost,
        stem_and_suffix_cost=stems.total,
        gap=stalled.cost - stems.total,
        weights=weights,
        threshold_below=below,
        threshold_above=above,
        escape_weight=SENTENCE_WEIGHT,
        escape_cost=escape.cost,
    )


@router.get("/stall", response_model=StallResponse)
def stall() -> StallResponse:
    """The same twelve words repeated more often, and the move that goes uphill."""
    return _stall_view()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to learn from, a weight to learn under, and text to read back."""

    corpus: CorpusChoice = CorpusChoice.INFLECTIONS
    repeats: int = Field(default=DEFAULT_REPEATS, ge=1, le=MAX_REPEATS)
    corpus_weight: float = Field(default=1.0, ge=MIN_WEIGHT, le=MAX_WEIGHT)
    text: str = Field(default="walking and playing", max_length=MAX_TEXT_LENGTH)
    word: str = Field(default="walkings", max_length=MAX_WORD_LENGTH)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything the playground shows for one fit."""

    corpus: list[str]
    n_distinct_words: int
    n_word_occurrences: int
    alphabet_size: int
    alphabet_covered: int
    corpus_weight: float
    n_morphs: int
    n_rows: int
    morphs: list[MorphRow]
    breakdown: CostBreakdown
    start_total: float
    epochs_run: int
    converged: bool
    corpus_pieces: int
    whole_word_pieces: int
    segmentations: list[WordCut]
    text: str
    cut: list[str]
    n_pieces: int
    n_unknown: int
    decoded: str
    round_trip_exact: bool
    word: str
    word_pieces: list[str]
    word_cost: float


MAX_SEGMENTATIONS_SHOWN = 24
"""How many of a fit's segmented words the playground lists."""

MAX_MORPHS_SHOWN = 60
"""How many of a fit's pieces the playground lists."""


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn a lexicon under one weight and read one text and one word with it."""
    corpus = _corpus_for(request.corpus, request.repeats)
    counts = _word_counts(corpus)
    alphabet = {character for word in counts for character in word}
    try:
        model = _fit(request.corpus, request.repeats, request.corpus_weight, 10)
        encoding = model.encode(request.text)
        segmentation = model.best_segmentation(request.word)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    morphs = model.morphs
    breakdown = _breakdown(description_length(morphs, request.corpus_weight), morphs)
    start = description_length(
        Morphs([Morph(word, count) for word, count in counts.items()]),
        request.corpus_weight,
    )
    decoded = model.decode(encoding.ids)
    return FitResponse(
        corpus=corpus,
        n_distinct_words=len(counts),
        n_word_occurrences=sum(counts.values()),
        alphabet_size=len(alphabet),
        alphabet_covered=len(alphabet & set(morphs.texts)),
        corpus_weight=request.corpus_weight,
        n_morphs=morphs.n_morphs,
        n_rows=model.vocabulary.n_tokens,
        morphs=[
            MorphRow(text=morph.text, count=morph.count)
            for morph in sorted(morphs, key=lambda each: (-each.count, each.text))[
                :MAX_MORPHS_SHOWN
            ]
        ],
        breakdown=breakdown,
        start_total=start.total,
        epochs_run=model.epochs_run,
        converged=model.converged,
        corpus_pieces=_corpus_pieces(model, counts),
        whole_word_pieces=sum(counts.values()),
        segmentations=[
            WordCut(word=each.word, pieces=list(each.morphs), n_pieces=each.n_morphs)
            for each in sorted(model.segmentations, key=lambda each: each.word)[
                :MAX_SEGMENTATIONS_SHOWN
            ]
        ],
        text=request.text,
        cut=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        n_unknown=sum(1 for piece in encoding.texts if piece == UNKNOWN),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
        word=request.word,
        word_pieces=list(segmentation.morphs),
        word_cost=segmentation.cost,
    )
