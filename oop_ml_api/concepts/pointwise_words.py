"""Learning Boundaries From Examples.

Every endpoint here is registered under
``/concepts/learning-boundaries-from-examples``, and the router is created and
registered before any endpoint exists, so the pages of this section can be built
in parallel without many edits to one shared file.

The page is about PointwiseSegmenter, and about RunSegmenter, which is the base
deciding what a whitespace-separated stretch is.

What this module adds to the library
------------------------------------
The segmenter learns weights, scores every gap and answers words; nothing here
re-implements the perceptron or the scoring. What is assembled around it is the
bookkeeping the page quotes, and there are six pieces of it.

First, a gap taken apart. ``gap_features`` is public and ``weight_of`` answers
one averaged weight, so the contribution of every feature at one gap can be
listed and totalled here; a test asserts that the total equals the score the
model itself reports for that gap, to the last bit, so this is a decomposition
rather than a second opinion.

Second, the truth of a text, gap by gap. A reader's reading is turned into the
set of gaps that carry a boundary, so each decision can be marked right or wrong
on its own, which is the thing this method makes possible and the sequence model
does not.

Third, the head-to-head against the hidden-model page's segmenter, on that
page's own generated languages. The language generator, the training and testing
seeds and the span-level scorer are imported from that module rather than
rewritten, so the two pages' numbers are comparable and the two methods are
handed the identical sentences.

Fourth, the arithmetic of partial annotation. A gap is one training instance
here whether or not its neighbours were marked, while a character's place in a
word is known only when the gaps on both sides of it are, so the two methods get
very different amounts out of a corpus somebody marked in part. Both shares are
counted here over one drawn marking.

Fifth, the window measurement. A small language is built in which the boundary
between two characters is decided by a character a fixed distance away, and the
same language is fitted at every window, which turns "a dependency wider than
the window cannot be represented" into a table.

Sixth, the coherence check. The pointwise answer is scored under the sequence
model's own tables, out of that model's public log probabilities, so the claim
that an answer made of independent decisions need not maximise anything is a
comparison of two numbers.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The only free text returned is the labels written here, the
words of the corpora, and the reader's own text; a test asserts that no
identifier survives into any of them, including into a refusal.
"""

from __future__ import annotations

import itertools
import random
from enum import StrEnum
from functools import cache
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (
    BoundaryTag,
    HiddenMarkovSegmenter,
    tags_of_word,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.pointwise import (
    BIAS_FEATURE,
    PointwiseSegmenter,
    gap_features,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.hidden_markov_words import (
    ENGLISH_TESTING_SEED,
    ENGLISH_TRAINING_SEED,
    N_TEST_SENTENCES,
    N_TRAINING_SENTENCES,
    TESTING_SEED,
    TRAINING_SEED,
    _generated_sentences,
    _language_words,
    _quality,
)
from oop_ml_api.concepts.hidden_markov_words import (
    Quality as SharedQuality,
)
from oop_ml_api.concepts.maximum_matching import ORDINARY_ENGLISH

router = APIRouter(
    prefix="/concepts/learning-boundaries-from-examples",
    tags=["learning-boundaries-from-examples"],
)


# --------------------------------------------------------------------------
# The texts and the marked-up sentences the page carries.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page of this section carries, spaces and all."""

SPACELESS_SENTENCE = RUNNING_SENTENCE.replace(" ", "")
"""The same sentence with its spaces taken out, which is what a segmenter meets."""

SENTENCE_READING: tuple[str, ...] = (
    "Dr.",
    "Alvarez",
    "didn't",
    "expect",
    "the",
    "low-cost",
    "re-analysis.",
)
"""What a reader answers for it."""

WRITTEN_CORPUS: tuple[tuple[str, ...], ...] = (
    ("Dr.", "Sato", "read", "the", "re-print."),
    ("Mr.", "Okonkwo", "didn't", "read", "it."),
    ("the", "high-cost", "test", "didn't", "run."),
    ("Dr.", "Chen", "wrote", "a", "re-run."),
    ("we", "can't", "trust", "the", "low-yield", "re-count."),
    ("Ms.", "Watson", "read", "the", "report."),
    ("the", "test", "didn't", "need", "a", "re-print."),
    ("Dr.", "Sato", "wrote", "the", "report."),
)
"""Eight marked-up sentences carrying the running sentence's shapes.

An abbreviation whose full stop ends it, a contraction, a hyphenated compound
and a prefix, in words that are deliberately not the running sentence's own:
four of its seven words are absent from these eight sentences, which is what
makes what comes back on them worth anything.
"""

FIVE_SENTENCE_CORPUS: tuple[tuple[str, ...], ...] = (
    ("研究", "生命", "起源"),
    ("学生", "很", "多"),
    ("生命", "很", "好"),
    ("起源", "研究"),
    ("学生", "研究", "生命"),
)
"""The five marked-up sentences the hidden-model page reads its tables off.

Written here in the same order, so a reader coming from that page meets the
same five sentences and the difference between the two answers is the method.
"""

MIXED_CORPUS: tuple[tuple[str, ...], ...] = (
    ("Alvarez", "研究", "生命"),
    ("Watson", "起源", "研究"),
    ("生命", "起源"),
    ("Curie", "起源"),
    ("学生", "研究", "Watson"),
    ("研究", "生命", "起源"),
    ("Curie", "研究", "学生"),
    ("起源", "Alvarez", "生命"),
)
"""Eight sentences in which a foreign name keeps its own script.

Every boundary between the two scripts is a boundary, and every gap inside a
run of one script is not, which is a rule about the kinds of character rather
than about any character in particular.
"""

PAIRED_CORPUS: tuple[tuple[str, ...], ...] = (
    ("研究", "生命"),
    ("生命", "研究"),
    ("起源", "研究"),
    ("研究", "起源"),
)
"""Four sentences whose every word is exactly two characters long."""


RESEARCH_TEXT = "研究生命起源"
RESEARCH_READING: tuple[str, ...] = ("研究", "生命", "起源")

SURNAME = "阿尔瓦雷斯"
"""The running sentence's surname, written the way Chinese writes a foreign name."""

SURNAME_TEXT = SURNAME + RESEARCH_TEXT
SURNAME_READING: tuple[str, ...] = ("阿尔瓦雷斯", "研究", "生命", "起源")

UNFAMILIAR_TEXT = "Ortiz漢字文"
"""A name and three characters, none of which the mixed sentences ever held."""

UNFAMILIAR_READING: tuple[str, ...] = ("Ortiz", "漢字", "文")

CHEN_TEXT = "Chen研究起源"
CHEN_READING: tuple[str, ...] = ("Chen", "研究", "起源")

PAIRED_TEXT = "生命起源"
PAIRED_READING: tuple[str, ...] = ("生命", "起源")

SPACED_TEXT = "研究生命 起源研究"
"""Two runs with a space between them, for the rule about what is never asked."""

SPACED_READING: tuple[str, ...] = ("研究", "生命", "起源", "研究")

ODD_TEXT = "命起源生"
"""Four characters a corpus of two-character words answers with three and one."""


HAND_WORKED_TEXT = "ab"
"""Two characters and one gap, which is the whole model small enough to add up."""


# --------------------------------------------------------------------------
# How each corpus is fitted, so a number on the page has one arrangement behind it.
# --------------------------------------------------------------------------

MIN_WINDOW = 1
MAX_WINDOW = 6
MIN_EPOCHS = 1
MAX_EPOCHS = 40
MAX_TEXT_CHARACTERS = 60
"""How long a text a request may carry, a little longer than the running sentence."""

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


class CorpusKey(StrEnum):
    """Which marked-up sentences a request is to learn from.

    Attributes
    ----------
    WRITTEN:
        Eight English sentences carrying the running sentence's shapes.
    FIVE_SENTENCES:
        The five sentences the hidden-model page reads its tables off.
    MIXED:
        Eight sentences in which a foreign name keeps its own script.
    PAIRED:
        Four sentences whose every word is exactly two characters long.
    """

    WRITTEN = "written"
    FIVE_SENTENCES = "five sentences"
    MIXED = "mixed"
    PAIRED = "paired"


CORPORA: dict[CorpusKey, tuple[tuple[str, ...], ...]] = {
    CorpusKey.WRITTEN: WRITTEN_CORPUS,
    CorpusKey.FIVE_SENTENCES: FIVE_SENTENCE_CORPUS,
    CorpusKey.MIXED: MIXED_CORPUS,
    CorpusKey.PAIRED: PAIRED_CORPUS,
}

CORPUS_LABELS: dict[CorpusKey, str] = {
    CorpusKey.WRITTEN: "eight English sentences somebody marked up",
    CorpusKey.FIVE_SENTENCES: "five sentences that agree with themselves",
    CorpusKey.MIXED: "eight sentences mixing two scripts",
    CorpusKey.PAIRED: "four sentences whose every word is two characters",
}

CORPUS_SETTINGS: dict[CorpusKey, tuple[int, int, int]] = {
    CorpusKey.WRITTEN: (3, 20, 0),
    CorpusKey.FIVE_SENTENCES: (3, 10, 0),
    CorpusKey.MIXED: (3, 10, 1),
    CorpusKey.PAIRED: (3, 10, 0),
}
"""The window, the number of passes and the seed each corpus is fitted with.

Fixed per corpus rather than chosen per widget, so two numbers the page quotes
about one corpus came from one fit.
"""

DEFAULT_WINDOW = 3
DEFAULT_EPOCHS = 10

SEED_LADDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
"""The seeds the page walks, to show what the order of the corrections decides."""

WINDOW_LADDER: tuple[int, ...] = (1, 2, 3, 4, 5)

MARKED_SHARES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
"""The shares of a corpus's gaps a part-time annotator is imagined to have marked."""

MARKING_SEED = 3
N_MARKING_SENTENCES = 200

LANGUAGE_SIZES: tuple[int, ...] = (2, 5, 10, 20, 30, 40, 80, 160, 320)
ENGLISH_SIZES: tuple[int, ...] = (10, 20, 40, 80, 160, 320, 640, 1280)

FILLERS: tuple[int, ...] = (0, 1, 2, 3)
"""How many characters stand between the deciding character and the gap."""

N_WIDE_REPEATS = 6
"""How many times each of the two sentence shapes appears in the wide-dependency corpus."""


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("must be a str", "must be text"),
    ("a sentence must be a sequence of words", "a sentence must be a list of words"),
    ("the window must be at least one", "the reach must be at least one"),
)
"""The substitutions that take an identifier out of a message a reader sees."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it.

    A test asserts that nothing identifier-shaped survives, so a message that
    grows one later fails there rather than reaching the page.
    """
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


# --------------------------------------------------------------------------
# The shapes every endpoint answers in.
# --------------------------------------------------------------------------


class GapDecision(BaseModel):
    """One gap of a text, the score it was given, and whether that was right.

    ``expected`` is ``None`` for a text nobody has said the reading of, which is
    every text a reader types.
    """

    index: int
    position: int
    before: str
    after: str
    score: float
    cut: bool
    expected: bool | None
    correct: bool | None


class Scenario(BaseModel):
    """One text taken all the way through, beside what the sequence model answers.

    Both methods are fitted on the same marked-up sentences, so the difference
    between the two readings is the method and nothing else.
    """

    key: str
    label: str
    corpus_key: str
    corpus_label: str
    sentences: list[list[str]]
    corpus_words: list[str]
    window: int
    epochs: int
    random_seed: int
    text: str
    gaps: list[GapDecision]
    words: list[str]
    n_gaps: int
    n_gaps_wrong: int | None
    reader_reading: list[str] | None
    matches_reader: bool
    from_the_sequence_model: list[str]
    sequence_matches_reader: bool
    n_new_words: int
    n_features: int
    n_free_boundaries: int


class ScenariosResponse(BaseModel):
    """Every fixed text of the page, in the order the page meets them."""

    scenarios: list[Scenario]


class FeatureVote(BaseModel):
    """One feature of one gap, what it names, and the weight it carries.

    ``family`` is ``base rate``, ``which character`` or ``what kind of
    character``, which is the only distinction the page makes between features.
    """

    feature: str
    family: str
    reach: str
    reads: str
    weight: float
    seen: bool


class OpenedGap(BaseModel):
    """One gap with every feature that voted on it, and the three subtotals.

    ``total`` is the sum of the weights below it and is checked against the score
    the model itself reports for that gap, so this is the decision taken apart
    rather than an account of it.
    """

    key: str
    label: str
    corpus_key: str
    corpus_label: str
    text: str
    gap: int
    before: str
    after: str
    votes: list[FeatureVote]
    total: float
    cut: bool
    expected: bool | None
    base_rate_total: float
    identity_total: float
    kind_total: float
    n_identity: int
    n_identity_seen: int
    n_kind: int
    n_kind_seen: int


class HandWorked(BaseModel):
    """The smallest fit there is: two characters, one gap, one correction.

    A reach of one gives seven features, a first score of zero counts as a
    mistake, one is added to each of the seven, and the average over one step is
    that same set of weights, so the gap then scores seven.
    """

    text: str
    features: list[str]
    n_features: int
    score_when_cut: float
    score_when_joined: float
    words_when_cut: list[str]
    words_when_joined: list[str]
    n_updates_by_epoch: list[int]


class GapsResponse(BaseModel):
    """The gaps the page opens up, and the fit small enough to add up by hand."""

    gaps: list[OpenedGap]
    hand_worked: HandWorked
    features_by_reach: list[list[str]]


class EpochRow(BaseModel):
    """One pass over the marked gaps, and how many of them it corrected."""

    epoch: int
    n_updates: int


class SeedRow(BaseModel):
    """One order of the corrections, and the reading it ends up answering."""

    random_seed: int
    words: list[str]
    scores: list[float]
    n_updates_by_epoch: list[int]
    matches_reader: bool
    n_passes_before_settling: int


class TrainingResponse(BaseModel):
    """What the learning does, on one corpus, and what the order of it decides."""

    corpus_key: str
    corpus_label: str
    sentences: list[list[str]]
    text: str
    reader_reading: list[str]
    window: int
    epochs: int
    n_marked_gaps: int
    n_boundaries: int
    n_features: int
    epoch_rows: list[EpochRow]
    seed_rows: list[SeedRow]
    n_seeds_matching_reader: int


class Quality(BaseModel):
    """How well one method answered the held-out sentences.

    A word counts as recovered only when both its ends land where the sentence
    put them. ``recall_on_seen`` and ``recall_on_unseen`` split that same recall
    by whether the word appeared in the marked-up sentences at all.
    """

    f_measure: float
    precision: float
    recall: float
    recall_on_seen: float
    recall_on_unseen: float


class SizeStep(BaseModel):
    """One amount of marked-up text, and what each method scored on it."""

    n_sentences: int
    n_marked_gaps: int
    n_features: int
    n_known_words: int
    share_unseen: float
    pointwise: Quality
    sequence: Quality


class LanguageRace(BaseModel):
    """One generated language, both methods, along the prefixes of one corpus."""

    key: str
    label: str
    n_words: int
    n_alphabet: int
    n_test_sentences: int
    steps: list[SizeStep]
    crosses_between: list[int]


class AnnotationRow(BaseModel):
    """One share of marked gaps, and how much each method can use of it.

    ``share_usable_gaps`` is what a gap-at-a-time learner keeps, which is every
    marked gap. ``share_known_places`` is what a whole-sequence learner keeps,
    which is a character whose gaps on both sides were marked.
    """

    share_marked: float
    n_marked: int
    n_gaps: int
    share_usable_gaps: float
    n_known_places: int
    n_characters: int
    share_known_places: float


class ComparisonResponse(BaseModel):
    """The head-to-head, and the two things that decide which way it goes.

    ``identity_share_seen`` and ``kind_share_seen`` count, over every gap of two
    hundred held-out sentences, how often a feature naming particular characters
    carries a weight at all and how often a feature naming their kinds does.
    """

    languages: list[LanguageRace]
    annotation: list[AnnotationRow]
    identity_share_seen: float
    kind_share_seen: float
    n_features_counted: int


class WindowRow(BaseModel):
    """One distance from the deciding character to the gap, at every reach tried."""

    n_filler: int
    offset: int
    n_right_by_window: list[int]
    n_texts: int
    smallest_window_that_works: int | None


class CoherenceRow(BaseModel):
    """One reading of one text, and what the whole-sequence score makes of it."""

    name: str
    words: list[str]
    n_pieces: int
    sequence_log_score: float
    is_the_sequence_answer: bool


class LimitsResponse(BaseModel):
    """Where the method stops being defined, in the two ways the page shows.

    ``n_odd_pointwise`` and ``n_odd_sequence`` count, over every four-character
    text spellable out of a corpus whose every word is two characters, how often
    each method answers with a piece of odd length. The pointwise count being the
    smaller of the two is the measurement contradicting what the argument from
    independence predicts.
    """

    window_rows: list[WindowRow]
    windows: list[int]
    wide_example: list[str]
    coherence_text: str
    coherence_rows: list[CoherenceRow]
    n_four_character_texts: int
    n_odd_pointwise: int
    n_odd_sequence: int
    odd_example_text: str
    odd_from_pointwise: list[str]
    odd_from_the_sequence_model: list[str]
    paired_words: list[str]


class SegmentRequest(BaseModel):
    """A text, the sentences to learn from, and how the learning is arranged."""

    text: BoundedText
    corpus: CorpusKey = CorpusKey.WRITTEN
    window: int = Field(default=DEFAULT_WINDOW, ge=MIN_WINDOW, le=MAX_WINDOW)
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=MIN_EPOCHS, le=MAX_EPOCHS)
    random_seed: int = Field(default=0, ge=0, le=99)
    gap: int | None = Field(default=None, ge=0)

    model_config = {"extra": "forbid"}


class SegmentResponse(BaseModel):
    """The requested text decided gap by gap, and one of those gaps opened up."""

    scenario: Scenario
    opened: OpenedGap | None


# --------------------------------------------------------------------------
# The work.
# --------------------------------------------------------------------------


def _sentences(corpus: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    """The marked-up sentences as plain lists, which is what a fit is handed."""
    return [list(sentence) for sentence in corpus]


def _fitted(
    corpus: tuple[tuple[str, ...], ...],
    window: int,
    epochs: int,
    random_seed: int,
) -> PointwiseSegmenter:
    """One gap classifier, learned from the sentences as they are written above."""
    return PointwiseSegmenter(
        window=window, epochs=epochs, random_seed=random_seed
    ).fit(_sentences(corpus))


def _sequence_model(corpus: tuple[tuple[str, ...], ...]) -> HiddenMarkovSegmenter:
    """The previous page's segmenter, learned from the very same sentences."""
    return HiddenMarkovSegmenter().fit(_sentences(corpus))


def _words_of(corpus: tuple[tuple[str, ...], ...]) -> list[str]:
    """Every word the sentences contain, once each, in the order they first appear."""
    seen: dict[str, None] = {}
    for sentence in corpus:
        for word in sentence:
            seen.setdefault(word, None)
    return list(seen)


def _boundary_positions(text: str, reading: tuple[str, ...]) -> set[int]:
    """The positions in the text a word of the reading ends at, bar the last.

    Counted by walking the reading and the text together and stepping over any
    whitespace, so a boundary that falls where a space already is contributes
    nothing. That is the whole of what makes the count of wrong gaps mean
    something on a text with spaces in it, since a boundary at a space is never
    asked about and cannot be got wrong.
    """
    ends: list[int] = []
    index = 0
    for word in reading:
        taken = 0
        last = index
        while taken < len(word) and index < len(text):
            if text[index].isspace():
                index += 1
                continue
            last = index
            taken += 1
            index += 1
        ends.append(last)
    return set(ends[:-1])


def _feature_family(feature: str) -> str:
    """Which of the three kinds of feature this string is, in the page's words."""
    if feature == BIAS_FEATURE:
        return "base rate"
    return "what kind of character" if feature.startswith("t[") else "which character"


def _feature_reach(feature: str) -> str:
    """The offsets a feature reads, written as they appear inside its own name."""
    if feature == BIAS_FEATURE:
        return "no characters"
    inside = feature[feature.index("[") + 1 : feature.index("]")]
    return inside


def _feature_reads(feature: str) -> str:
    """What the feature actually saw, which is the part after the equals sign."""
    if feature == BIAS_FEATURE:
        return ""
    return feature[feature.index("=") + 1 :]


def _votes(model: PointwiseSegmenter, run: str, gap: int) -> list[FeatureVote]:
    """Every feature of one gap, with the weight the fit gave it."""
    return [
        FeatureVote(
            feature=feature,
            family=_feature_family(feature),
            reach=_feature_reach(feature),
            reads=_feature_reads(feature),
            weight=model.weight_of(feature),
            seen=model.weight_of(feature) != 0.0,
        )
        for feature in gap_features(run, gap, model.window)
    ]


def _opened_gap(
    key: str,
    label: str,
    corpus_key: CorpusKey,
    text: str,
    gap: int,
    reading: tuple[str, ...] | None,
    model: PointwiseSegmenter | None = None,
) -> OpenedGap:
    """One gap of one text, taken apart into the features that decided it."""
    if model is None:
        model = _fitted(CORPORA[corpus_key], *CORPUS_SETTINGS[corpus_key])
    votes = _votes(model, text, gap)
    total = sum(vote.weight for vote in votes)
    boundaries = _boundary_positions(text, reading) if reading is not None else None
    identity = [vote for vote in votes if vote.family == "which character"]
    kinds = [vote for vote in votes if vote.family == "what kind of character"]
    return OpenedGap(
        key=key,
        label=label,
        corpus_key=corpus_key.value,
        corpus_label=CORPUS_LABELS[corpus_key],
        text=text,
        gap=gap,
        before=text[gap],
        after=text[gap + 1],
        votes=votes,
        total=total,
        cut=total > 0,
        expected=None if boundaries is None else gap in boundaries,
        base_rate_total=sum(
            vote.weight for vote in votes if vote.family == "base rate"
        ),
        identity_total=sum(vote.weight for vote in identity),
        kind_total=sum(vote.weight for vote in kinds),
        n_identity=len(identity),
        n_identity_seen=sum(vote.seen for vote in identity),
        n_kind=len(kinds),
        n_kind_seen=sum(vote.seen for vote in kinds),
    )


def _scenario(
    key: str,
    label: str,
    corpus_key: CorpusKey,
    text: str,
    reading: tuple[str, ...] | None,
    window: int | None = None,
    epochs: int | None = None,
    random_seed: int | None = None,
) -> Scenario:
    """One text decided gap by gap, and the same text read by the sequence model."""
    corpus = CORPORA[corpus_key]
    settings = CORPUS_SETTINGS[corpus_key]
    reach = settings[0] if window is None else window
    passes = settings[1] if epochs is None else epochs
    seed = settings[2] if random_seed is None else random_seed

    model = _fitted(corpus, reach, passes, seed)
    scores = model.gap_scores(text)
    words = list(model.split(text).texts)
    corpus_words = _words_of(corpus)
    sequence_words = list(_sequence_model(corpus).split(text).texts)

    boundaries = _boundary_positions(text, reading) if reading is not None else None
    gaps: list[GapDecision] = []
    index = 0
    for match in NON_WHITESPACE_RUN.finditer(text):
        run = match.group()
        for offset in range(len(run) - 1):
            score = scores[index]
            position = match.start() + offset
            expected = None if boundaries is None else position in boundaries
            gaps.append(
                GapDecision(
                    index=index,
                    position=position,
                    before=run[offset],
                    after=run[offset + 1],
                    score=score,
                    cut=score > 0,
                    expected=expected,
                    correct=None if expected is None else (score > 0) == expected,
                )
            )
            index += 1

    return Scenario(
        key=key,
        label=label,
        corpus_key=corpus_key.value,
        corpus_label=CORPUS_LABELS[corpus_key],
        sentences=_sentences(corpus),
        corpus_words=corpus_words,
        window=reach,
        epochs=passes,
        random_seed=seed,
        text=text,
        gaps=gaps,
        words=words,
        n_gaps=len(gaps),
        n_gaps_wrong=(
            None
            if boundaries is None
            else sum(1 for gap in gaps if gap.correct is False)
        ),
        reader_reading=list(reading) if reading is not None else None,
        matches_reader=reading is not None and tuple(words) == reading,
        from_the_sequence_model=sequence_words,
        sequence_matches_reader=reading is not None
        and tuple(sequence_words) == reading,
        n_new_words=sum(1 for word in words if word not in corpus_words),
        n_features=model.n_features,
        n_free_boundaries=(
            0
            if reading is None
            else len(reading) - 1 - sum(1 for gap in gaps if gap.expected)
        ),
    )


@cache
def _scenarios() -> ScenariosResponse:
    """The page's fixed texts, each decided gap by gap, computed once."""
    return ScenariosResponse(
        scenarios=[
            _scenario(
                key="sentence",
                label="our own sentence with its spaces taken out",
                corpus_key=CorpusKey.WRITTEN,
                text=SPACELESS_SENTENCE,
                reading=SENTENCE_READING,
            ),
            _scenario(
                key="research",
                label="the sentence with two defensible readings",
                corpus_key=CorpusKey.FIVE_SENTENCES,
                text=RESEARCH_TEXT,
                reading=RESEARCH_READING,
            ),
            _scenario(
                key="surname",
                label="the same sentence behind a transliterated surname",
                corpus_key=CorpusKey.FIVE_SENTENCES,
                text=SURNAME_TEXT,
                reading=SURNAME_READING,
            ),
            _scenario(
                key="unfamiliar",
                label="a name and three characters none of the sentences held",
                corpus_key=CorpusKey.MIXED,
                text=UNFAMILIAR_TEXT,
                reading=UNFAMILIAR_READING,
            ),
            _scenario(
                key="chen",
                label="a name the sentences never held, beside words they did",
                corpus_key=CorpusKey.MIXED,
                text=CHEN_TEXT,
                reading=CHEN_READING,
            ),
            _scenario(
                key="spaced",
                label="two runs with a space between them",
                corpus_key=CorpusKey.FIVE_SENTENCES,
                text=SPACED_TEXT,
                reading=SPACED_READING,
            ),
            _scenario(
                key="paired",
                label="four characters where every word is two characters long",
                corpus_key=CorpusKey.PAIRED,
                text=PAIRED_TEXT,
                reading=PAIRED_READING,
            ),
            _scenario(
                key="odd",
                label="the same sentences answering with a piece of one character",
                corpus_key=CorpusKey.PAIRED,
                text=ODD_TEXT,
                reading=None,
            ),
        ]
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    """The page's texts, each decided one gap at a time, with every score."""
    return _scenarios()


@cache
def _gaps() -> GapsResponse:
    """The gaps the page opens up, and the fit small enough to check by hand."""
    written = _fitted(WRITTEN_CORPUS, *CORPUS_SETTINGS[CorpusKey.WRITTEN])
    mixed = _fitted(MIXED_CORPUS, *CORPUS_SETTINGS[CorpusKey.MIXED])

    stop = SPACELESS_SENTENCE.index("Dr.") + 2
    apostrophe = SPACELESS_SENTENCE.index("didn't") + 3
    hyphen = SPACELESS_SENTENCE.index("low-cost") + 2
    script = UNFAMILIAR_TEXT.index("漢") - 1

    cut = PointwiseSegmenter(window=1, epochs=1).fit([["a", "b"]])
    joined = PointwiseSegmenter(window=1, epochs=1).fit([["ab"]])

    return GapsResponse(
        gaps=[
            _opened_gap(
                key="full-stop",
                label="the full stop closing an abbreviation, against the name after it",
                corpus_key=CorpusKey.WRITTEN,
                text=SPACELESS_SENTENCE,
                gap=stop,
                reading=SENTENCE_READING,
                model=written,
            ),
            _opened_gap(
                key="apostrophe",
                label="the apostrophe inside a contraction",
                corpus_key=CorpusKey.WRITTEN,
                text=SPACELESS_SENTENCE,
                gap=apostrophe,
                reading=SENTENCE_READING,
                model=written,
            ),
            _opened_gap(
                key="hyphen",
                label="the hyphen inside a compound",
                corpus_key=CorpusKey.WRITTEN,
                text=SPACELESS_SENTENCE,
                gap=hyphen,
                reading=SENTENCE_READING,
                model=written,
            ),
            _opened_gap(
                key="script",
                label="a change of script, with nothing familiar on either side",
                corpus_key=CorpusKey.MIXED,
                text=UNFAMILIAR_TEXT,
                gap=script,
                reading=UNFAMILIAR_READING,
                model=mixed,
            ),
        ],
        hand_worked=HandWorked(
            text=HAND_WORKED_TEXT,
            features=list(gap_features(HAND_WORKED_TEXT, 0, 1)),
            n_features=cut.n_features,
            score_when_cut=cut.gap_scores(HAND_WORKED_TEXT)[0],
            score_when_joined=joined.gap_scores(HAND_WORKED_TEXT)[0],
            words_when_cut=list(cut.split(HAND_WORKED_TEXT).texts),
            words_when_joined=list(joined.split(HAND_WORKED_TEXT).texts),
            n_updates_by_epoch=list(cut.n_updates_by_epoch),
        ),
        features_by_reach=[
            list(gap_features("a" * (4 * reach), 2 * reach - 1, reach))
            for reach in (1, 2, 3)
        ],
    )


@router.get("/gaps", response_model=GapsResponse)
def gaps() -> GapsResponse:
    """The features that voted at four gaps, and the smallest fit there is."""
    return _gaps()


@cache
def _training() -> TrainingResponse:
    """What the corrections do, and what the order of them decides."""
    window, epochs, _ = CORPUS_SETTINGS[CorpusKey.WRITTEN]
    settled = _fitted(WRITTEN_CORPUS, window, epochs, 0)

    n_gaps = 0
    n_boundaries = 0
    for sentence in WRITTEN_CORPUS:
        run = "".join(sentence)
        n_gaps += len(run) - 1
        n_boundaries += len(_boundary_positions(run, sentence))

    seed_rows: list[SeedRow] = []
    for seed in SEED_LADDER:
        model = _fitted(WRITTEN_CORPUS, window, epochs, seed)
        updates = list(model.n_updates_by_epoch)
        settling = next(
            (position for position, count in enumerate(updates) if count == 0),
            len(updates),
        )
        words = list(model.split(SPACELESS_SENTENCE).texts)
        seed_rows.append(
            SeedRow(
                random_seed=seed,
                words=words,
                scores=list(model.gap_scores(SPACELESS_SENTENCE)),
                n_updates_by_epoch=updates,
                matches_reader=tuple(words) == SENTENCE_READING,
                n_passes_before_settling=settling,
            )
        )

    return TrainingResponse(
        corpus_key=CorpusKey.WRITTEN.value,
        corpus_label=CORPUS_LABELS[CorpusKey.WRITTEN],
        sentences=_sentences(WRITTEN_CORPUS),
        text=SPACELESS_SENTENCE,
        reader_reading=list(SENTENCE_READING),
        window=window,
        epochs=epochs,
        n_marked_gaps=n_gaps,
        n_boundaries=n_boundaries,
        n_features=settled.n_features,
        epoch_rows=[
            EpochRow(epoch=position + 1, n_updates=count)
            for position, count in enumerate(settled.n_updates_by_epoch)
        ],
        seed_rows=seed_rows,
        n_seeds_matching_reader=sum(row.matches_reader for row in seed_rows),
    )


@router.get("/training", response_model=TrainingResponse)
def training() -> TrainingResponse:
    """How the weights are corrected, and what the order of the corrections decides."""
    return _training()


# --------------------------------------------------------------------------
# The head-to-head, on the languages the hidden-model page generated.
# --------------------------------------------------------------------------


def _as_quality(measured: SharedQuality) -> Quality:
    """One score set, copied into this page's own shape.

    The scoring itself is the hidden-model page's, imported rather than
    rewritten, so the two pages' figures are the same measurement.
    """
    return Quality(
        f_measure=measured.f_measure,
        precision=measured.precision,
        recall=measured.recall,
        recall_on_seen=measured.recall_on_seen,
        recall_on_unseen=measured.recall_on_unseen,
    )


def _race(
    key: str,
    label: str,
    vocabulary: list[str],
    training_seed: int,
    testing_seed: int,
    sizes: tuple[int, ...],
) -> LanguageRace:
    """One language, both methods, along the prefixes of one marked-up corpus."""
    marked = _generated_sentences(vocabulary, N_TRAINING_SENTENCES, training_seed)
    testing = _generated_sentences(vocabulary, N_TEST_SENTENCES, testing_seed)
    texts = ["".join(sentence) for sentence in testing]
    wanted = sum(len(sentence) for sentence in testing)

    steps: list[SizeStep] = []
    behind = 0
    ahead = 0
    for size in sizes:
        part = marked[:size]
        known = {word for sentence in part for word in sentence}
        n_gaps = sum(len("".join(sentence)) - 1 for sentence in part)
        point = PointwiseSegmenter(
            window=DEFAULT_WINDOW, epochs=DEFAULT_EPOCHS, random_seed=0
        ).fit(part)
        sequence = HiddenMarkovSegmenter().fit(part)

        point_quality = _as_quality(
            _quality([list(point.split(text).texts) for text in texts], testing, known)
        )
        sequence_quality = _as_quality(
            _quality(
                [list(sequence.split(text).texts) for text in texts], testing, known
            )
        )
        if point_quality.f_measure <= sequence_quality.f_measure:
            behind = size
        elif ahead == 0:
            ahead = size

        unseen = sum(
            1 for sentence in testing for word in sentence if word not in known
        )
        steps.append(
            SizeStep(
                n_sentences=size,
                n_marked_gaps=n_gaps,
                n_features=point.n_features,
                n_known_words=len(known),
                share_unseen=unseen / wanted,
                pointwise=point_quality,
                sequence=sequence_quality,
            )
        )

    return LanguageRace(
        key=key,
        label=label,
        n_words=len(vocabulary),
        n_alphabet=len({character for word in vocabulary for character in word}),
        n_test_sentences=len(testing),
        steps=steps,
        crosses_between=[behind, ahead],
    )


def _annotation() -> list[AnnotationRow]:
    """What a partly marked corpus is worth to a gap learner and to a sequence one.

    One marking is drawn per share from a seeded generator, and both quantities
    are counted over that same marking, so the pair of numbers on a row describes
    one corpus rather than two.
    """
    marked_sentences = _generated_sentences(
        _language_words(), N_TRAINING_SENTENCES, TRAINING_SEED
    )[:N_MARKING_SENTENCES]
    generator = random.Random(MARKING_SEED)

    rows: list[AnnotationRow] = []
    for share in MARKED_SHARES:
        n_marked = n_gaps = n_known = n_characters = 0
        for sentence in marked_sentences:
            run = "".join(sentence)
            gap_count = len(run) - 1
            marked = {gap for gap in range(gap_count) if generator.random() < share}
            n_gaps += gap_count
            n_marked += len(marked)
            n_characters += len(run)
            for index in range(len(run)):
                left_known = index - 1 < 0 or index - 1 in marked
                right_known = index >= gap_count or index in marked
                n_known += left_known and right_known
        rows.append(
            AnnotationRow(
                share_marked=share,
                n_marked=n_marked,
                n_gaps=n_gaps,
                share_usable_gaps=n_marked / n_gaps,
                n_known_places=n_known,
                n_characters=n_characters,
                share_known_places=n_known / n_characters,
            )
        )
    return rows


def _feature_coverage() -> tuple[float, float, int]:
    """How often each kind of feature carries a weight, over held-out sentences.

    A feature naming particular characters is silent whenever that combination
    of characters never turned up in the marked-up sentences; a feature naming
    their kinds never is, because there are only seven kinds.
    """
    vocabulary = _language_words()
    part = _generated_sentences(vocabulary, N_TRAINING_SENTENCES, TRAINING_SEED)[:80]
    testing = _generated_sentences(vocabulary, N_TEST_SENTENCES, TESTING_SEED)
    model = PointwiseSegmenter(
        window=DEFAULT_WINDOW, epochs=DEFAULT_EPOCHS, random_seed=0
    ).fit(part)

    identity_seen = identity_total = kind_seen = kind_total = 0
    for sentence in testing:
        run = "".join(sentence)
        for gap in range(len(run) - 1):
            for feature in gap_features(run, gap, DEFAULT_WINDOW):
                if feature == BIAS_FEATURE:
                    continue
                seen = model.weight_of(feature) != 0.0
                if feature.startswith("t["):
                    kind_total += 1
                    kind_seen += seen
                else:
                    identity_total += 1
                    identity_seen += seen
    return (
        identity_seen / identity_total,
        kind_seen / kind_total,
        identity_total + kind_total,
    )


@cache
def _comparison() -> ComparisonResponse:
    """Both languages measured, and the two quantities that decide the outcome."""
    identity_share, kind_share, counted = _feature_coverage()
    return ComparisonResponse(
        languages=[
            _race(
                key="characters",
                label="a language of seventy words over forty characters",
                vocabulary=_language_words(),
                training_seed=TRAINING_SEED,
                testing_seed=TESTING_SEED,
                sizes=LANGUAGE_SIZES,
            ),
            _race(
                key="letters",
                label="ordinary English words with the spaces taken out",
                vocabulary=list(ORDINARY_ENGLISH),
                training_seed=ENGLISH_TRAINING_SEED,
                testing_seed=ENGLISH_TESTING_SEED,
                sizes=ENGLISH_SIZES,
            ),
        ],
        annotation=_annotation(),
        identity_share_seen=identity_share,
        kind_share_seen=kind_share,
        n_features_counted=counted,
    )


@router.get("/comparison", response_model=ComparisonResponse)
def comparison() -> ComparisonResponse:
    """Both methods on the same generated languages, scored the same way."""
    return _comparison()


# --------------------------------------------------------------------------
# Where the method stops being defined.
# --------------------------------------------------------------------------


def _wide_corpus(n_filler: int) -> tuple[list[list[str]], list[tuple[str, ...]]]:
    """A language whose boundary is decided by a character a fixed distance away.

    A run beginning with one character is two words and a run beginning with the
    other is one, and everything between the deciding character and the gap is
    identical, so the gap can only be settled by reading that far back.
    """
    filler = "x" * n_filler
    corpus: list[list[str]] = []
    for _ in range(N_WIDE_REPEATS):
        corpus.append([f"p{filler}ab", "cd"])
        corpus.append([f"q{filler}abcd"])
    tests = [(f"p{filler}ab", "cd"), (f"q{filler}abcd",)]
    return corpus, tests


def _sequence_log_score(
    model: HiddenMarkovSegmenter, run: str, words: list[str]
) -> float:
    """What the whole-sequence score makes of one reading of one run.

    Every term comes from the model's own public log probabilities, so this is
    that model scoring a reading it did not choose rather than a second scoring
    rule invented here.
    """
    tags: list[BoundaryTag] = [tag for word in words for tag in tags_of_word(word)]
    total = model.initial_log_probability(tags[0]) + model.emission_log_probability(
        tags[0], run[0]
    )
    for index in range(1, len(run)):
        total += model.transition_log_probability(tags[index - 1], tags[index])
        total += model.emission_log_probability(tags[index], run[index])
    return total


@cache
def _limits() -> LimitsResponse:
    """The reach of the window, and what an answer of independent parts is worth."""
    window_rows: list[WindowRow] = []
    wide_example: list[str] = []
    for n_filler in FILLERS:
        corpus, tests = _wide_corpus(n_filler)
        if not wide_example:
            wide_example = ["".join(sentence) for sentence in corpus[:2]]
        right_by_window: list[int] = []
        smallest: int | None = None
        for reach in WINDOW_LADDER:
            model = PointwiseSegmenter(window=reach, epochs=20, random_seed=0).fit(
                corpus
            )
            right = sum(model.split("".join(truth)).texts == truth for truth in tests)
            right_by_window.append(right)
            if right == len(tests) and smallest is None:
                smallest = reach
        window_rows.append(
            WindowRow(
                n_filler=n_filler,
                offset=-(n_filler + 3),
                n_right_by_window=right_by_window,
                n_texts=len(tests),
                smallest_window_that_works=smallest,
            )
        )

    written_point = _fitted(WRITTEN_CORPUS, *CORPUS_SETTINGS[CorpusKey.WRITTEN])
    written_sequence = _sequence_model(WRITTEN_CORPUS)
    pointwise_reading = list(written_point.split(SPACELESS_SENTENCE).texts)
    sequence_reading = list(written_sequence.split(SPACELESS_SENTENCE).texts)
    coherence_rows = [
        CoherenceRow(
            name="one answer per gap",
            words=pointwise_reading,
            n_pieces=len(pointwise_reading),
            sequence_log_score=_sequence_log_score(
                written_sequence, SPACELESS_SENTENCE, pointwise_reading
            ),
            is_the_sequence_answer=False,
        ),
        CoherenceRow(
            name="the best whole sequence",
            words=sequence_reading,
            n_pieces=len(sequence_reading),
            sequence_log_score=_sequence_log_score(
                written_sequence, SPACELESS_SENTENCE, sequence_reading
            ),
            is_the_sequence_answer=True,
        ),
        CoherenceRow(
            name="what a reader answers",
            words=list(SENTENCE_READING),
            n_pieces=len(SENTENCE_READING),
            sequence_log_score=_sequence_log_score(
                written_sequence, SPACELESS_SENTENCE, list(SENTENCE_READING)
            ),
            is_the_sequence_answer=False,
        ),
    ]

    paired_point = _fitted(PAIRED_CORPUS, *CORPUS_SETTINGS[CorpusKey.PAIRED])
    paired_sequence = _sequence_model(PAIRED_CORPUS)
    alphabet = sorted(
        {character for word in _words_of(PAIRED_CORPUS) for character in word}
    )
    n_texts = odd_point = odd_sequence = 0
    for combination in itertools.product(alphabet, repeat=4):
        text = "".join(combination)
        n_texts += 1
        odd_point += any(len(piece) % 2 for piece in paired_point.split(text).texts)
        odd_sequence += any(
            len(piece) % 2 for piece in paired_sequence.split(text).texts
        )

    return LimitsResponse(
        window_rows=window_rows,
        windows=list(WINDOW_LADDER),
        wide_example=wide_example,
        coherence_text=SPACELESS_SENTENCE,
        coherence_rows=coherence_rows,
        n_four_character_texts=n_texts,
        n_odd_pointwise=odd_point,
        n_odd_sequence=odd_sequence,
        odd_example_text=ODD_TEXT,
        odd_from_pointwise=list(paired_point.split(ODD_TEXT).texts),
        odd_from_the_sequence_model=list(paired_sequence.split(ODD_TEXT).texts),
        paired_words=_words_of(PAIRED_CORPUS),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """How far a gap can see, and what an answer assembled from parts is worth."""
    return _limits()


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest) -> SegmentResponse:
    """Decide the requested text gap by gap, and open up one of those gaps."""
    if not NON_WHITESPACE_RUN.search(request.text):
        raise HTTPException(
            status_code=422,
            detail="a text needs at least one character that is not a space",
        )
    try:
        scenario = _scenario(
            key="requested",
            label="your text",
            corpus_key=request.corpus,
            text=request.text,
            reading=None,
            window=request.window,
            epochs=request.epochs,
            random_seed=request.random_seed,
        )
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    opened: OpenedGap | None = None
    if request.gap is not None and 0 <= request.gap < len(scenario.gaps):
        chosen = scenario.gaps[request.gap]
        run_start = chosen.position - _offset_within_run(request.text, chosen.position)
        run = NON_WHITESPACE_RUN.match(request.text, run_start)
        model = _fitted(
            CORPORA[request.corpus],
            request.window,
            request.epochs,
            request.random_seed,
        )
        assert run is not None
        opened = _opened_gap(
            key="requested",
            label="the gap you chose",
            corpus_key=request.corpus,
            text=run.group(),
            gap=chosen.position - run_start,
            reading=None,
            model=model,
        )

    return SegmentResponse(scenario=scenario, opened=opened)


def _offset_within_run(text: str, position: int) -> int:
    """How far into its own whitespace-free run a character of the text sits."""
    for match in NON_WHITESPACE_RUN.finditer(text):
        if match.start() <= position < match.end():
            return position - match.start()
    return 0
