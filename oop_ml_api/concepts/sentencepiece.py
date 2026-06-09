"""The SentencePiece page's endpoints.

Five of them, one per family of widgets rather than one per number.
``/marking`` carries the two ways of recording a word boundary, the alphabet
each leaves behind and the round trip each manages on the running sentence.
``/learning`` walks the four-word corpus one merge at a time with the tied
candidates at every step, which is where the mark's codepoint decides an
outcome, and reports what the eighteen sentences learn. ``/costs`` sweeps the
vocabulary size for both schemes and adds the corpus written without spaces,
where one of the two has nothing to split on. ``/limits`` carries the facts the
closing part rests on, merging with nothing cut and the same fit under a
different mark. ``/fit`` is the playground.

Two calculations here are this module's own rather than the library's. Merging
a corpus with nothing cut at the marks is not something any tokenizer here
does, since every one of them cuts first; it is the page's question about what
the method would do unrestrained, and it is asked with the same merging
arithmetic the trainers use. Counting how many rows of a vocabulary differ from
another row only by a leading mark is likewise a question about a finished
vocabulary rather than a step in producing one. Everything else is a fit.

Every refusal is rewritten by ``_in_plain_words`` before it leaves here, since
the page must never show a parameter name, and the test module asserts that no
identifier survives the rewriting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from functools import cache

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding import (
    BytePairEncoding,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    learn_merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.sentence_piece import (
    WHITESPACE_MARKER,
    SentencePiece,
    SubwordAlgorithm,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/sentencepiece", tags=["sentencepiece"])


MAX_VOCABULARY_SIZE = 400
"""The largest vocabulary a request may ask for.

Every corpus this page carries runs out of pairs worth merging long before
this, so the bound never truncates anything shown; it exists so that a crafted
request cannot turn a teaching endpoint into a long fit.
"""

MAX_TEXT_LENGTH = 200
"""The longest text the playground may ask to read."""

MARK = WHITESPACE_MARKER
"""The stand-in for a space, taken from the from-scratch model rather than
retyped, so the page and the fit can never disagree about which character it is."""

END_OF_WORD = "</w>"
"""The other scheme's marker, written out once."""


# --------------------------------------------------------------------------
# The texts the page works with.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page in this section carries."""

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

Shared with the other pages of this section so that a reader meeting the same
corpus twice can compare what two methods made of it. Every character the
running sentence uses appears somewhere in these, which matters here because
one of the two boundary schemes still cannot spell the sentence and the other
can.
"""

FOUR_WORD_COUNTS: dict[str, int] = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
"""Four words with their counts, small enough that every merge is checkable.

The corpus the original subword paper worked its example on, and the corpus the
from-scratch trainer's own spec is pinned against.
"""

FOUR_WORD_CORPUS: list[str] = [
    " ".join(word for word, count in FOUR_WORD_COUNTS.items() for _ in range(count))
]

UNSEEN_WORD = "lowest"
"""A word the four-word corpus never contains, spellable from what it taught."""

UNSPACED_CORPUS: list[str] = [
    "研究生命起源",
    "学生很多",
    "生命很好",
    "起源研究",
    "学生研究生命",
]
"""Five short sentences of Chinese, written the way Chinese is written.

Taken from the fixtures the from-scratch segmenters are pinned against, so the
words inside them are known: 研究 (research), 生命 (life), 起源 (origin), 学生
(student), 很 (very), 多 (many), 好 (good). Nothing separates them in the
writing, which is the situation a whitespace rule has nothing to say about.
"""

UNSPACED_TEXT = "学生研究生命起源"
"""A sentence the five never contain, made of words they do."""

UNSPACED_GLOSSES: list[tuple[str, str]] = [
    ("研究", "research"),
    ("生命", "life"),
    ("起源", "origin"),
    ("学生", "student"),
    ("很", "very"),
    ("多", "many"),
    ("好", "good"),
]

NEIGHBOUR_WORDS = ("analysis", "re-analysis.", "expect", "expected", "Alvarez")
"""Five words read one at a time by both schemes, chosen so that each shows a
different part of the trade: a corpus word, the same word inside something
longer, a stem the corpus lacks, the inflection it has, and a name ending in a
letter no corpus word ends in."""

SWEEP_SIZES = (52, 60, 70, 80, 90, 100, 110, 120, 130, 137, 200)
"""The sizes both schemes are fitted at. The floor of the stricter of the two
sets the first entry, since below it only one of the pair exists."""

FRONT_ONLY_SIZES = (37, 40, 45)
"""Sizes at which only the front-marked scheme can be fitted at all."""

NORMALISED_TEXTS = (
    "low  lower",
    " low lower ",
    "low\tlower",
    f"low {MARK} lower",
)
"""Four texts whose round trip is not exact, each losing a different thing."""


class CorpusChoice(StrEnum):
    """Which of the page's three corpora a request wants fitted."""

    FOUR_WORDS = "four_words"
    SENTENCES = "sentences"
    UNSPACED = "unspaced"


CORPORA: dict[CorpusChoice, list[str]] = {
    CorpusChoice.FOUR_WORDS: FOUR_WORD_CORPUS,
    CorpusChoice.SENTENCES: SENTENCE_CORPUS,
    CorpusChoice.UNSPACED: UNSPACED_CORPUS,
}


class Learner(StrEnum):
    """Which of the two learners the framework wraps a request wants."""

    MERGING = "merging"
    SHRINKING = "shrinking"


LEARNERS: dict[Learner, SubwordAlgorithm] = {
    Learner.MERGING: SubwordAlgorithm.BYTE_PAIR,
    Learner.SHRINKING: SubwordAlgorithm.UNIGRAM,
}


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("vocabulary_size=", "a vocabulary of "),
    ("whitespace_marker", "the mark that stands for a space"),
    ("minimum_pair_frequency", "the fewest times a pair may be seen"),
    ("pre_tokenizer", "the rule that finds the words"),
    ("the unknown token", "one row for anything unspellable"),
    ("SentencePiece", "this scheme"),
    ("BytePairEncoding", "this scheme"),
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


def _word_counts(corpus: Sequence[str]) -> dict[str, int]:
    counts = Corpus.of(list(corpus)).word_counts(WhitespacePreTokenizer())
    return {word_count.word: word_count.count for word_count in counts}


def _front_marked(corpus: Sequence[str], size: int) -> SentencePiece:
    return SentencePiece(
        vocabulary_size=size, algorithm=SubwordAlgorithm.BYTE_PAIR
    ).fit(list(corpus))


def _end_marked(corpus: Sequence[str], size: int) -> BytePairEncoding:
    return BytePairEncoding(byte_fallback=False, vocabulary_size=size).fit(list(corpus))


def _corpus_pieces(model: Tokenizer, corpus: Sequence[str]) -> int:
    return sum(model.encode(text).n_tokens for text in corpus)


def _front_alphabet(corpus: Sequence[str]) -> list[str]:
    return sorted(
        {character for word in _word_counts(corpus) for character in word} | {MARK}
    )


def _end_alphabet(corpus: Sequence[str]) -> list[str]:
    symbols: set[str] = set()
    for word in _word_counts(corpus):
        symbols.update(word[:-1])
        symbols.add(word[-1] + END_OF_WORD)
    return sorted(symbols)


def _n_merges(model: Tokenizer) -> int:
    """How many merges a fitted model learned, whichever of the two it is."""
    if isinstance(model, BytePairEncoding):
        return model.n_merges
    if isinstance(model, SentencePiece) and model.merges is not None:
        return len(list(model.merges))
    return 0


def _merge_labels(merges: Merges) -> list[str]:
    return [f"{merge.left} + {merge.right}" for merge in merges]


def _pair_counts(spelled: Sequence[SpelledWord]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for word in spelled:
        for left, right in zip(word.symbols, word.symbols[1:], strict=False):
            counter[(left, right)] += word.count
    return counter


def _front_spelled(counts: dict[str, int]) -> list[SpelledWord]:
    return [SpelledWord((MARK, *word), count) for word, count in counts.items()]


# --------------------------------------------------------------------------
# /marking: two ways to record a boundary, and what each leaves behind.
# --------------------------------------------------------------------------


class Reading(BaseModel):
    """One fit's reading of one text, out and back again."""

    pieces: list[str]
    n_pieces: int
    decoded: str
    round_trip_exact: bool
    n_unspellable: int


class SchemeSummary(BaseModel):
    """One way of recording where a word ends, fitted on the eighteen sentences.

    ``alphabet`` is what the corpus turns out to be spelled in once the boundary
    has been recorded, and ``floor`` is the smallest vocabulary that can hold it
    together with the row that stands in for anything unspellable.
    """

    key: str
    label: str
    alphabet: list[str]
    alphabet_size: int
    floor: int
    learned: int
    n_merges: int
    corpus_pieces: int
    sentence: Reading


class WordPair(BaseModel):
    """One word as each scheme reads it."""

    word: str
    front_pieces: list[str]
    end_pieces: list[str]


class LossRow(BaseModel):
    """A text whose round trip is not exact, and what came back instead."""

    text: str
    pieces: list[str]
    decoded: str
    what_was_lost: str


class MarkingResponse(BaseModel):
    """The two boundary schemes side by side on the page's own corpus."""

    sentence: str
    corpus: list[str]
    n_distinct_words: int
    n_word_occurrences: int
    distinct_characters: int
    mark: str
    mark_codepoint: int
    marked_sentence: str
    marked_units: list[str]
    schemes: list[SchemeSummary]
    word_pairs: list[WordPair]
    losses: list[LossRow]
    unspaced_corpus: list[str]
    unspaced_words_found: int


LOSS_DESCRIPTIONS: dict[str, str] = {
    "low  lower": "a run of two spaces comes back as one",
    " low lower ": "the space at each end is gone",
    "low\tlower": "a tab comes back as a space",
}


@cache
def _marking() -> MarkingResponse:
    counts = _word_counts(SENTENCE_CORPUS)
    front = _front_marked(SENTENCE_CORPUS, MAX_VOCABULARY_SIZE)
    end = _end_marked(SENTENCE_CORPUS, MAX_VOCABULARY_SIZE)

    losses: list[LossRow] = []
    for text in NORMALISED_TEXTS:
        encoding = front.encode(text)
        decoded = front.decode(encoding.ids)
        losses.append(
            LossRow(
                text=text,
                pieces=list(encoding.texts),
                decoded=decoded,
                what_was_lost=LOSS_DESCRIPTIONS.get(
                    text, "a mark written in the text itself comes back as a space"
                ),
            )
        )

    return MarkingResponse(
        sentence=RUNNING_SENTENCE,
        corpus=SENTENCE_CORPUS,
        n_distinct_words=len(counts),
        n_word_occurrences=sum(counts.values()),
        distinct_characters=len(
            {character for text in SENTENCE_CORPUS for character in text}
        ),
        mark=MARK,
        mark_codepoint=ord(MARK),
        marked_sentence="".join(front.units_of(RUNNING_SENTENCE)),
        marked_units=list(front.units_of(RUNNING_SENTENCE)),
        schemes=[
            _scheme_summary("front", "The space marked in front", front),
            _scheme_summary("end", "The word end marked behind", end),
        ],
        word_pairs=[
            WordPair(
                word=word,
                front_pieces=list(front.encode(word).texts),
                end_pieces=list(end.encode(word).texts),
            )
            for word in NEIGHBOUR_WORDS
        ],
        losses=losses,
        unspaced_corpus=UNSPACED_CORPUS,
        unspaced_words_found=len(
            Corpus.of(UNSPACED_CORPUS).word_counts(WhitespacePreTokenizer())
        ),
    )


def _reading(model: Tokenizer, text: str) -> Reading:
    encoding = model.encode(text)
    decoded = model.decode(encoding.ids)
    unknown = model.vocabulary.unknown_token
    return Reading(
        pieces=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        decoded=decoded,
        round_trip_exact=decoded == text,
        n_unspellable=sum(1 for piece in encoding.texts if piece == unknown),
    )


def _scheme_summary(key: str, label: str, model: Tokenizer) -> SchemeSummary:
    alphabet = (
        _front_alphabet(SENTENCE_CORPUS)
        if key == "front"
        else _end_alphabet(SENTENCE_CORPUS)
    )
    return SchemeSummary(
        key=key,
        label=label,
        alphabet=alphabet,
        alphabet_size=len(alphabet),
        floor=len(alphabet) + 1,
        learned=model.vocabulary.n_tokens,
        n_merges=_n_merges(model),
        corpus_pieces=_corpus_pieces(model, SENTENCE_CORPUS),
        sentence=_reading(model, RUNNING_SENTENCE),
    )


@router.get("/marking", response_model=MarkingResponse)
def marking() -> MarkingResponse:
    """The two boundary schemes on the eighteen sentences, and what each loses."""
    return _marking()


# --------------------------------------------------------------------------
# /learning: the four-word ladder, and what the sentences learn.
# --------------------------------------------------------------------------


class LadderStep(BaseModel):
    """One merge, the candidates that tied with it, and the corpus before it.

    ``tied`` names every pair that reached the winning count, so a reader can
    see the steps where the count alone did not decide, and
    ``mark_pair_was_tied`` says whether one of those candidates began with the
    mark, which is the case the page turns on.
    """

    step: int
    left: str
    right: str
    merged: str
    count: int
    tied: list[str]
    mark_pair_was_tied: bool
    spellings: list[list[str]]
    corpus_pieces: int
    unseen_pieces: list[str]


class LearnerSummary(BaseModel):
    """One of the two learners on the four-word corpus."""

    key: str
    label: str
    learned: int
    tokens: list[str]
    unseen_pieces: list[str]
    corpus_pieces: int
    sentence_corpus_pieces: int
    sentence_pieces: int


class LearningResponse(BaseModel):
    """The four words merged step by step, then the eighteen sentences."""

    words: list[str]
    counts: list[int]
    mark: str
    mark_codepoint: int
    rival_character: str
    rival_codepoint: int
    starting_pieces: int
    steps: list[LadderStep]
    final_spellings: list[list[str]]
    final_pieces: int
    final_unseen_pieces: list[str]
    tokens: list[str]
    unseen_word: str
    sentence_merges: list[str]
    sentence_merge_counts: list[int]
    sentence_n_merges: int
    sentence_n_mark_merges: int
    first_mark_merge_rank: int
    twin_rows: list[str]
    sentence_tokens: list[str]
    learners: list[LearnerSummary]
    shared_tokens: int


@cache
def _learning() -> LearningResponse:
    counts = FOUR_WORD_COUNTS
    current = _front_spelled(counts)
    starting = sum(len(word.symbols) * word.count for word in current)

    steps: list[LadderStep] = []
    while True:
        pair_counts = _pair_counts(current)
        candidates = {pair: count for pair, count in pair_counts.items() if count >= 2}
        if not candidates:
            break
        best = max(candidates.values())
        tied = sorted(pair for pair, count in candidates.items() if count == best)
        merge = learn_merges(current, n_merges=1, minimum_pair_frequency=2)[0]
        steps.append(
            LadderStep(
                step=len(steps),
                left=merge.left,
                right=merge.right,
                merged=merge.merged,
                count=int(merge.score),
                tied=[f"{left} + {right}" for left, right in tied],
                mark_pair_was_tied=any(left == MARK for left, _ in tied),
                spellings=[list(word.symbols) for word in current],
                corpus_pieces=sum(len(word.symbols) * word.count for word in current),
                unseen_pieces=list(
                    _front_marked(FOUR_WORD_CORPUS, 12 + len(steps))
                    .encode(UNSEEN_WORD)
                    .texts
                ),
            )
        )
        current = [word.with_pair_merged(merge) for word in current]

    merging = _front_marked(FOUR_WORD_CORPUS, 30)
    shrinking = SentencePiece(vocabulary_size=30, random_seed=0).fit(FOUR_WORD_CORPUS)
    sentence_merging = _front_marked(SENTENCE_CORPUS, MAX_VOCABULARY_SIZE)
    sentence_shrinking = SentencePiece(
        vocabulary_size=sentence_merging.vocabulary.n_tokens, random_seed=0
    ).fit(SENTENCE_CORPUS)

    sentence_merges = list(sentence_merging.merges or [])
    mark_merges = [
        rank
        for rank, merge in enumerate(sentence_merges)
        if merge.left.startswith(MARK) or merge.right.startswith(MARK)
    ]
    sentence_tokens = list(sentence_merging.vocabulary)
    token_set = set(sentence_tokens)

    return LearningResponse(
        words=list(counts),
        counts=list(counts.values()),
        mark=MARK,
        mark_codepoint=ord(MARK),
        rival_character="l",
        rival_codepoint=ord("l"),
        starting_pieces=starting,
        steps=steps,
        final_spellings=[list(word.symbols) for word in current],
        final_pieces=sum(len(word.symbols) * word.count for word in current),
        final_unseen_pieces=list(merging.encode(UNSEEN_WORD).texts),
        tokens=list(merging.vocabulary),
        unseen_word=UNSEEN_WORD,
        sentence_merges=[f"{merge.left} + {merge.right}" for merge in sentence_merges],
        sentence_merge_counts=[int(merge.score) for merge in sentence_merges],
        sentence_n_merges=len(sentence_merges),
        sentence_n_mark_merges=len(mark_merges),
        first_mark_merge_rank=mark_merges[0],
        twin_rows=sorted(
            token
            for token in token_set
            if token.startswith(MARK) and token[1:] in token_set
        ),
        sentence_tokens=sentence_tokens,
        learners=[
            LearnerSummary(
                key="merging",
                label="Joining the commonest pair",
                learned=merging.vocabulary.n_tokens,
                tokens=list(merging.vocabulary),
                unseen_pieces=list(merging.encode(UNSEEN_WORD).texts),
                corpus_pieces=_corpus_pieces(merging, FOUR_WORD_CORPUS),
                sentence_corpus_pieces=_corpus_pieces(
                    sentence_merging, SENTENCE_CORPUS
                ),
                sentence_pieces=sentence_merging.encode(RUNNING_SENTENCE).n_tokens,
            ),
            LearnerSummary(
                key="shrinking",
                label="Dropping the least missed piece",
                learned=shrinking.vocabulary.n_tokens,
                tokens=list(shrinking.vocabulary),
                unseen_pieces=list(shrinking.encode(UNSEEN_WORD).texts),
                corpus_pieces=_corpus_pieces(shrinking, FOUR_WORD_CORPUS),
                sentence_corpus_pieces=_corpus_pieces(
                    sentence_shrinking, SENTENCE_CORPUS
                ),
                sentence_pieces=sentence_shrinking.encode(RUNNING_SENTENCE).n_tokens,
            ),
        ],
        shared_tokens=len(set(merging.vocabulary) & set(shrinking.vocabulary)),
    )


@router.get("/learning", response_model=LearningResponse)
def learning() -> LearningResponse:
    """The four-word corpus merged step by step, and what the sentences learn."""
    return _learning()


# --------------------------------------------------------------------------
# /costs: what the treatment costs, in pieces.
# --------------------------------------------------------------------------


class SizeRow(BaseModel):
    """One vocabulary size, both schemes.

    The end-marked entries are ``None`` below its own floor, where it cannot be
    fitted at all and the front-marked scheme can.
    """

    asked: int
    front_learned: int
    front_corpus_pieces: int
    front_sentence_pieces: int
    front_exact: bool
    end_learned: int | None
    end_corpus_pieces: int | None
    end_sentence_pieces: int | None
    end_exact: bool | None


class UnspacedSummary(BaseModel):
    """One scheme on the corpus written without spaces."""

    key: str
    label: str
    asked: int
    learned: int
    n_merges: int
    tokens: list[str]
    corpus_pieces: int
    text_pieces: list[str]
    n_text_pieces: int
    decoded: str
    round_trip_exact: bool


class CostsResponse(BaseModel):
    """The size sweep on the eighteen sentences, then the unspaced corpus."""

    sentence: str
    rows: list[SizeRow]
    front_floor: int
    end_floor: int
    front_exhausted: int
    end_exhausted: int
    unspaced_corpus: list[str]
    unspaced_glosses: list[list[str]]
    unspaced_text: str
    unspaced: list[UnspacedSummary]
    unspaced_end_final_marks: list[str]


@cache
def _costs() -> CostsResponse:
    front_floor = len(_front_alphabet(SENTENCE_CORPUS)) + 1
    end_floor = len(_end_alphabet(SENTENCE_CORPUS)) + 1

    rows: list[SizeRow] = []
    for asked in (*FRONT_ONLY_SIZES, *SWEEP_SIZES):
        front = _front_marked(SENTENCE_CORPUS, asked)
        front_reading = _reading(front, RUNNING_SENTENCE)
        if asked < end_floor:
            rows.append(
                SizeRow(
                    asked=asked,
                    front_learned=front.vocabulary.n_tokens,
                    front_corpus_pieces=_corpus_pieces(front, SENTENCE_CORPUS),
                    front_sentence_pieces=front_reading.n_pieces,
                    front_exact=front_reading.round_trip_exact,
                    end_learned=None,
                    end_corpus_pieces=None,
                    end_sentence_pieces=None,
                    end_exact=None,
                )
            )
            continue
        end = _end_marked(SENTENCE_CORPUS, asked)
        end_reading = _reading(end, RUNNING_SENTENCE)
        rows.append(
            SizeRow(
                asked=asked,
                front_learned=front.vocabulary.n_tokens,
                front_corpus_pieces=_corpus_pieces(front, SENTENCE_CORPUS),
                front_sentence_pieces=front_reading.n_pieces,
                front_exact=front_reading.round_trip_exact,
                end_learned=end.vocabulary.n_tokens,
                end_corpus_pieces=_corpus_pieces(end, SENTENCE_CORPUS),
                end_sentence_pieces=end_reading.n_pieces,
                end_exact=end_reading.round_trip_exact,
            )
        )

    unspaced: list[UnspacedSummary] = []
    for key, label, model in (
        ("front", "The space marked in front", _front_marked(UNSPACED_CORPUS, 40)),
        ("end", "The word end marked behind", _end_marked(UNSPACED_CORPUS, 40)),
    ):
        reading = _reading(model, UNSPACED_TEXT)
        unspaced.append(
            UnspacedSummary(
                key=key,
                label=label,
                asked=40,
                learned=model.vocabulary.n_tokens,
                n_merges=_n_merges(model),
                tokens=list(model.vocabulary),
                corpus_pieces=_corpus_pieces(model, UNSPACED_CORPUS),
                text_pieces=reading.pieces,
                n_text_pieces=reading.n_pieces,
                decoded=reading.decoded,
                round_trip_exact=reading.round_trip_exact,
            )
        )

    unspaced_end = _end_marked(UNSPACED_CORPUS, 40)
    return CostsResponse(
        sentence=RUNNING_SENTENCE,
        rows=rows,
        front_floor=front_floor,
        end_floor=end_floor,
        front_exhausted=_front_marked(
            SENTENCE_CORPUS, MAX_VOCABULARY_SIZE
        ).vocabulary.n_tokens,
        end_exhausted=_end_marked(
            SENTENCE_CORPUS, MAX_VOCABULARY_SIZE
        ).vocabulary.n_tokens,
        unspaced_corpus=UNSPACED_CORPUS,
        unspaced_glosses=[list(pair) for pair in UNSPACED_GLOSSES],
        unspaced_text=UNSPACED_TEXT,
        unspaced=unspaced,
        unspaced_end_final_marks=sorted(
            token
            for token in unspaced_end.vocabulary
            if token.endswith(END_OF_WORD)
            and token[: -len(END_OF_WORD)] in set(unspaced_end.vocabulary)
        ),
    )


@router.get("/costs", response_model=CostsResponse)
def costs() -> CostsResponse:
    """The size sweep for both schemes, and the corpus written without spaces."""
    return _costs()


# --------------------------------------------------------------------------
# /limits: what nothing in the method decides.
# --------------------------------------------------------------------------


class UncutMerge(BaseModel):
    """One merge learned with nothing cut at the marks."""

    rank: int
    left: str
    right: str
    merged: str
    count: int
    crosses: bool


class MarkChoice(BaseModel):
    """The same corpus fitted under one mark, with the mark written neutrally.

    ``tokens_shown`` writes every learned row with the mark replaced by a
    neutral glyph, so two fits under two marks can be compared row for row
    rather than character for character.
    """

    mark: str
    codepoint: int
    sorts_before_the_letters: bool
    learned: int
    merges: list[str]
    tokens: list[str]
    tokens_shown: list[str]
    unseen_pieces: list[str]


class LimitsResponse(BaseModel):
    """The facts the closing part rests on, every one of them measured."""

    uncut_merges: list[UncutMerge]
    n_uncut_crossing: int
    n_uncut: int
    first_uncut_crossing_rank: int
    mark_choices: list[MarkChoice]
    shared_rows: int
    unseen_word: str
    twin_rows: list[str]
    n_sentence_rows: int
    foreign_text: str
    foreign_pieces: list[str]
    foreign_decoded: str


UNCUT_MERGE_COUNT = 40
"""How many merges the uncut walk learns before the page stops looking."""


@cache
def _limits() -> LimitsResponse:
    marked = [MARK + text.replace(" ", MARK) for text in SENTENCE_CORPUS]
    counts = Counter(marked)
    spelled = [SpelledWord(tuple(text), count) for text, count in counts.items()]
    uncut = learn_merges(spelled, n_merges=UNCUT_MERGE_COUNT, minimum_pair_frequency=2)

    uncut_rows = [
        UncutMerge(
            rank=rank,
            left=merge.left,
            right=merge.right,
            merged=merge.merged,
            count=int(merge.score),
            crosses=MARK in merge.merged[1:],
        )
        for rank, merge in enumerate(uncut)
    ]

    choices: list[MarkChoice] = []
    shown: list[set[str]] = []
    for mark in (MARK, "_"):
        model = SentencePiece(
            vocabulary_size=30,
            algorithm=SubwordAlgorithm.BYTE_PAIR,
            whitespace_marker=mark,
        ).fit(FOUR_WORD_CORPUS)
        tokens = list(model.vocabulary)
        neutral = [token.replace(mark, "␣") for token in tokens]
        shown.append(set(neutral))
        choices.append(
            MarkChoice(
                mark=mark,
                codepoint=ord(mark),
                sorts_before_the_letters=mark < "a",
                learned=model.vocabulary.n_tokens,
                merges=_merge_labels(model.merges) if model.merges else [],
                tokens=tokens,
                tokens_shown=neutral,
                unseen_pieces=list(model.encode(UNSEEN_WORD).texts),
            )
        )

    sentence_model = _front_marked(SENTENCE_CORPUS, MAX_VOCABULARY_SIZE)
    token_set = set(sentence_model.vocabulary)
    foreign = sentence_model.encode(UNSPACED_TEXT)

    return LimitsResponse(
        uncut_merges=uncut_rows,
        n_uncut_crossing=sum(1 for row in uncut_rows if row.crosses),
        n_uncut=len(uncut_rows),
        first_uncut_crossing_rank=next(row.rank for row in uncut_rows if row.crosses),
        mark_choices=choices,
        shared_rows=len(shown[0] & shown[1]),
        unseen_word=UNSEEN_WORD,
        twin_rows=sorted(
            token
            for token in token_set
            if token.startswith(MARK) and token[1:] in token_set
        ),
        n_sentence_rows=sentence_model.vocabulary.n_tokens,
        foreign_text=UNSPACED_TEXT,
        foreign_pieces=list(foreign.texts),
        foreign_decoded=sentence_model.decode(foreign.ids),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """Merging with nothing cut, and the same fit under a different mark."""
    return _limits()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class FitRequest(BaseModel):
    """A corpus to learn from, a size to stop at, and a text to read back."""

    corpus: CorpusChoice = CorpusChoice.SENTENCES
    learner: Learner = Learner.MERGING
    vocabulary_size: int = Field(ge=2, le=MAX_VOCABULARY_SIZE)
    text: str = Field(default=RUNNING_SENTENCE, max_length=MAX_TEXT_LENGTH)

    model_config = {"extra": "forbid"}


class MergeRow(BaseModel):
    """One merge, with the count that chose it."""

    rank: int
    left: str
    right: str
    merged: str
    count: int


class FitResponse(BaseModel):
    """Everything the playground shows for one fit.

    ``end_*`` carries the same request answered by the scheme that splits on
    spaces first, or ``None`` where that scheme's own floor puts the size out
    of its reach.
    """

    corpus: list[str]
    alphabet: list[str]
    floor: int
    asked: int
    learned: int
    n_merges: int
    merges: list[MergeRow]
    tokens: list[str]
    corpus_pieces: int
    text: str
    marked_units: list[str]
    marked_text: str
    pieces: list[str]
    n_pieces: int
    n_unspellable: int
    decoded: str
    round_trip_exact: bool
    end_floor: int
    end_pieces: list[str] | None
    end_n_pieces: int | None
    end_decoded: str | None
    end_round_trip_exact: bool | None


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Learn a vocabulary of the requested size and read one text with it."""
    corpus = CORPORA[request.corpus]
    alphabet = _front_alphabet(corpus)
    end_floor = len(_end_alphabet(corpus)) + 1
    try:
        model = SentencePiece(
            vocabulary_size=request.vocabulary_size,
            algorithm=LEARNERS[request.learner],
            random_seed=0,
        ).fit(corpus)
        encoding = model.encode(request.text)
        units = model.units_of(request.text)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    end_reading: Reading | None = None
    if request.vocabulary_size >= end_floor:
        end_reading = _reading(
            _end_marked(corpus, request.vocabulary_size), request.text
        )

    unknown = model.vocabulary.unknown_token
    decoded = model.decode(encoding.ids)
    merges = list(model.merges) if model.merges is not None else []
    return FitResponse(
        corpus=corpus,
        alphabet=alphabet,
        floor=len(alphabet) + 1,
        asked=request.vocabulary_size,
        learned=model.vocabulary.n_tokens,
        n_merges=len(merges),
        merges=[
            MergeRow(
                rank=rank,
                left=merge.left,
                right=merge.right,
                merged=merge.merged,
                count=int(merge.score),
            )
            for rank, merge in enumerate(merges)
        ],
        tokens=list(model.vocabulary),
        corpus_pieces=_corpus_pieces(model, corpus),
        text=request.text,
        marked_units=list(units),
        marked_text="".join(units),
        pieces=list(encoding.texts),
        n_pieces=encoding.n_tokens,
        n_unspellable=sum(1 for piece in encoding.texts if piece == unknown),
        decoded=decoded,
        round_trip_exact=decoded == request.text,
        end_floor=end_floor,
        end_pieces=None if end_reading is None else end_reading.pieces,
        end_n_pieces=None if end_reading is None else end_reading.n_pieces,
        end_decoded=None if end_reading is None else end_reading.decoded,
        end_round_trip_exact=(
            None if end_reading is None else end_reading.round_trip_exact
        ),
    )
