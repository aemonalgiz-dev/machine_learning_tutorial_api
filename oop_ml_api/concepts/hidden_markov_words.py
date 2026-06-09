"""Segmenting With a Hidden Model.

Every endpoint here is registered under ``/concepts/segmenting-with-a-hidden-model``,
and the router is created and registered before any endpoint exists, so the pages
of this section can be built in parallel without many edits to one shared file.

The page is about HiddenMarkovSegmenter.

What this module adds to the library
------------------------------------
The segmenter counts the three tables, decodes with Viterbi and answers words;
nothing here re-implements the decoding as a way of answering. What is assembled
around it is the bookkeeping the page quotes, and there are five pieces of it.

First, the trellis. The segmenter keeps its columns internally and hands back a
tag per character, so the page cannot show what the losing tags scored. The
trellis here is built entirely out of the model's own public log probabilities,
and the tests assert that the path it walks back is the one the model itself
answers with, on every text the page carries; without that it would be a second
opinion rather than evidence.

Second, the counts behind the smoothed probabilities. The model exposes the
smoothed numbers and not the counts they came from, and a reader needs to see
``(2 + 1) / (2 + 2)`` rather than ``0.75``. The counts are recounted here from
the same corpus, and a test asserts that every smoothed fraction assembled from
them equals the model's own number to the last bit.

Third, what a word list built from the identical corpus answers, so the claim
that a tagger reaches words a list cannot is a difference between two numbers
rather than an assertion. Both learners are handed the same sentences.

Fourth, a measurement of how much tagged text is needed. Two languages are
generated with a fixed seed, one over a large character set and one over the
ordinary English words of the maximum-matching page, and both segmenters are
fitted on prefixes of the same corpus and scored on the same held-out sentences.
The scoring is word-level precision and recall over spans, written from the
definition here.

Fifth, the conditional entropy of a character's place given the character, which
is the quantity that explains why the two languages come out so differently.

Nothing this module returns may name a class, a parameter or an exception,
because a widget renders these strings and the page's readers have no reason to
meet an identifier. The only free text returned is the labels written here, the
words of the corpora, and the reader's own text; a test asserts that no
identifier survives into any of them, including into a refusal.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from enum import StrEnum
from functools import cache
from itertools import pairwise
from typing import Annotated

from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (
    ADMISSIBLE_SUCCESSORS,
    FINAL_TAGS,
    INITIAL_TAGS,
    BoundaryTag,
    HiddenMarkovSegmenter,
    tags_of_word,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.lattice import (
    DictionaryLatticeSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.maximum_matching import (
    MatchingDirection,
    MaximumMatchingSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.maximum_matching import ORDINARY_ENGLISH

router = APIRouter(
    prefix="/concepts/segmenting-with-a-hidden-model",
    tags=["segmenting-with-a-hidden-model"],
)


# --------------------------------------------------------------------------
# The texts and the corpora the page carries.
# --------------------------------------------------------------------------

RUNNING_SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
"""The sentence every page of this section carries, spaces and all."""

SPACELESS_SENTENCE = RUNNING_SENTENCE.replace(" ", "")
"""The same sentence with its spaces taken out."""

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

TWO_SENTENCE_CORPUS: tuple[tuple[str, ...], ...] = (("ab", "c"), ("abc",))
"""Two sentences over three characters, whose tables can be read whole.

The fixture the segmenter's own spec pins its smoothed fractions against, worked
on paper before the code existed.
"""

FIVE_SENTENCE_CORPUS: tuple[tuple[str, ...], ...] = (
    ("研究", "生命", "起源"),
    ("学生", "很", "多"),
    ("生命", "很", "好"),
    ("起源", "研究"),
    ("学生", "研究", "生命"),
)
"""Five segmented sentences over ten characters, agreeing with themselves.

No character sequence is one word in one sentence and two in another, which is
what a corpus has to be before anything can be recovered from it exactly.
"""

CONTRADICTORY_CORPUS: tuple[tuple[str, ...], ...] = (
    ("研究", "生命", "起源"),
    ("研究生", "很", "多"),
    ("生命", "很", "好"),
    ("起源", "研究"),
)
"""Four sentences that disagree about three characters.

The first two characters are a word twice and all three are a word once, so the
corpus cannot be reproduced by any single reading of them. The segmenter's own
module docstring says three times against once; counted, it is two against one,
and the threshold it records reproduces either way.
"""

BOUNDARY_CORPUS: tuple[tuple[str, ...], ...] = (
    ("ab", "cd"),
    ("ab", "ef"),
    ("cd", "ef"),
)
"""Three sentences in which a, c and e only begin words and b, d and f only end them."""

PAIRED_CORPUS: tuple[tuple[str, ...], ...] = (
    ("ab", "cd"),
    ("cd", "ab"),
    ("ab", "cd"),
)
"""One language in which the words are the first character with the second."""

CROSSED_CORPUS: tuple[tuple[str, ...], ...] = (
    ("ad", "cb"),
    ("cb", "ad"),
    ("ad", "cb"),
)
"""The other language, in which they are crossed over; every count comes out the same."""

RESEARCH_SENTENCE = "研究生命起源"
"""Research into the origin of life, or a graduate student's fate and origin."""

LIFE_READING: tuple[str, ...] = ("研究", "生命", "起源")
STUDENT_SENTENCE = "学生研究生命"
STUDENT_SENTENCE_READING: tuple[str, ...] = ("学生", "研究", "生命")

MINORITY_SENTENCE = "研究生很多"
"""The sentence the contradictory corpus contains once and contradicts twice."""

MAJORITY_READING: tuple[str, ...] = ("研究", "生", "很", "多")
MINORITY_READING: tuple[str, ...] = ("研究生", "很", "多")

NOVEL_TEXT = "adcfeb"
"""Three words the corpus never held, spelled out of characters it did."""

NOVEL_READING: tuple[str, ...] = ("ad", "cf", "eb")

SURNAME = "阿尔瓦雷斯"
"""The running sentence's surname, written the way Chinese writes a foreign name."""

NAME_SENTENCE = SURNAME + RESEARCH_SENTENCE
NAME_READING: tuple[str, ...] = ("阿尔瓦雷斯", "研究", "生命", "起源")

UNSEEN_PROBE = "￿"
"""A codepoint no corpus can contain, for reading the floor a place gives an unseen character."""

CROSSED_TEXT = "adcb"
"""A text both twin languages answer identically although only one contains its words."""


# --------------------------------------------------------------------------
# Two generated languages, for the measurement of how much text is needed.
# --------------------------------------------------------------------------

LANGUAGE_CHARACTERS = (
    "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生"
)
"""Forty characters the generated language spells its words out of."""

N_LANGUAGE_WORDS = 70
"""How many words that language has: ten of one character, forty of two, twenty of three."""

LANGUAGE_SEED = 11
TRAINING_SEED = 12
TESTING_SEED = 13
ENGLISH_TRAINING_SEED = 21
ENGLISH_TESTING_SEED = 22

N_TEST_SENTENCES = 200
N_TRAINING_SENTENCES = 2000

LANGUAGE_SIZES: tuple[int, ...] = (2, 5, 10, 20, 30, 40, 60, 80, 160, 320)
ENGLISH_SIZES: tuple[int, ...] = (10, 20, 40, 80, 160, 320, 640, 1280)

SMOOTHING_LADDER: tuple[float, ...] = (
    2.0,
    1.0,
    0.5,
    0.3,
    0.25,
    0.24,
    0.23,
    0.2,
    0.1,
    0.01,
)
"""The smoothing values the page walks, straddling the one at which the answer turns."""

UNSEEN_RUN_LENGTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

UNSEEN_RUNS: tuple[str, ...] = ("阿尔瓦雷斯", "ABCDE", "☃☂☘☕☔")
"""Three runs of five characters the five-sentence corpus never held."""


MAX_TEXT_CHARACTERS = 60
"""How long a text a request may carry, a little longer than the running sentence."""

MIN_SMOOTHING = 0.001
MAX_SMOOTHING = 5.0

BoundedText = Annotated[str, Field(max_length=MAX_TEXT_CHARACTERS)]


class CorpusKey(StrEnum):
    """Which segmented sentences a request is to be fitted on.

    Attributes
    ----------
    TWO_SENTENCES:
        Two sentences over three characters, small enough to read whole.
    FIVE_SENTENCES:
        Five sentences over ten characters, agreeing with themselves.
    CONTRADICTORY:
        Four sentences that disagree about one stretch of three characters.
    BOUNDARY:
        Three sentences in which every character plays one part only.
    PAIRED:
        One of the two languages whose counts are indistinguishable.
    CROSSED:
        The other one.
    """

    TWO_SENTENCES = "two sentences"
    FIVE_SENTENCES = "five sentences"
    CONTRADICTORY = "contradictory"
    BOUNDARY = "boundary"
    PAIRED = "paired"
    CROSSED = "crossed"


CORPORA: dict[CorpusKey, tuple[tuple[str, ...], ...]] = {
    CorpusKey.TWO_SENTENCES: TWO_SENTENCE_CORPUS,
    CorpusKey.FIVE_SENTENCES: FIVE_SENTENCE_CORPUS,
    CorpusKey.CONTRADICTORY: CONTRADICTORY_CORPUS,
    CorpusKey.BOUNDARY: BOUNDARY_CORPUS,
    CorpusKey.PAIRED: PAIRED_CORPUS,
    CorpusKey.CROSSED: CROSSED_CORPUS,
}

CORPUS_LABELS: dict[CorpusKey, str] = {
    CorpusKey.TWO_SENTENCES: "two sentences over three characters",
    CorpusKey.FIVE_SENTENCES: "five sentences that agree with themselves",
    CorpusKey.CONTRADICTORY: "four sentences that disagree about one word",
    CorpusKey.BOUNDARY: "three sentences where each character plays one part",
    CorpusKey.PAIRED: "one word-forming rule",
    CorpusKey.CROSSED: "the crossed-over rule",
}

TAG_NAMES: dict[BoundaryTag, str] = {
    BoundaryTag.BEGIN: "begins a word",
    BoundaryTag.MIDDLE: "inside a word",
    BoundaryTag.END: "ends a word",
    BoundaryTag.SINGLE: "a word on its own",
}
"""What each of the four places is, in words a reader can take at face value."""


# --------------------------------------------------------------------------
# Turning a refusal into the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("a word dictionary needs at least one entry", "a word list cannot be empty"),
    ("dictionary word", "word of the list"),
    ("dictionary", "word list"),
    ("must be a str", "must be text"),
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


class Cell(BaseModel):
    """What one of the four places scores at one character, and where it came from.

    ``score`` is ``None`` where the place cannot be reached at all, which is what
    the eight forbidden steps look like from inside the search.
    """

    tag: str
    tag_name: str
    score: float | None
    came_from: str | None
    on_best_path: bool


class TrellisColumn(BaseModel):
    """One character of a text, with all four places scored."""

    position: int
    character: str
    in_alphabet: bool
    cells: list[Cell]
    tag: str


class Tagging(BaseModel):
    """One text tagged all the way through, and what a word list answers for it.

    ``from_a_word_list`` and ``from_a_greedy_scan`` are both fitted on the very
    same sentences, so the three answers differ by method and by nothing else.
    """

    key: str
    label: str
    text: str
    corpus_label: str
    smoothing: float
    columns: list[TrellisColumn]
    tags: str
    words: list[str]
    total_log_score: float
    n_characters: int
    n_unseen_characters: int
    reader_reading: list[str] | None
    matches_reader: bool
    from_a_word_list: list[str]
    list_matches_reader: bool
    from_a_greedy_scan: list[str]
    words_in_the_corpus: list[str]
    n_new_words: int


class ScenariosResponse(BaseModel):
    """Every fixed text of the page, in the order the page meets them."""

    scenarios: list[Tagging]


class StartRow(BaseModel):
    """How often a run started in each place, and what that is worth once smoothed."""

    tag: str
    tag_name: str
    count: int
    total: int
    admissible: bool
    probability: float | None
    log_probability: float | None


class StepRow(BaseModel):
    """How often one place followed another, smoothed over the steps that can happen."""

    previous: str
    following: str
    count: int
    total: int
    admissible: bool
    probability: float | None
    log_probability: float | None


class CharacterRow(BaseModel):
    """How often one place produced one character, smoothed over the alphabet and a spare slot."""

    tag: str
    character: str
    count: int
    total: int
    probability: float
    log_probability: float


class Tables(BaseModel):
    """Everything one fit learned, with the counts the smoothed numbers were built from.

    ``denominator`` is the alphabet plus one spare slot, multiplied by the
    smoothing and added to the count of characters each place produced, which is
    what makes a character nobody wrote down cost something rather than nothing.
    """

    key: str
    label: str
    sentences: list[list[str]]
    n_sentences: int
    n_words: int
    alphabet: list[str]
    n_alphabet: int
    smoothing: float
    n_slots: int
    starts: list[StartRow]
    steps: list[StepRow]
    characters: list[CharacterRow]
    floors: list[CharacterRow]
    n_admissible_steps: int
    n_forbidden_steps: int


class TablesResponse(BaseModel):
    """The two corpora the page reads tables off, each fitted at the usual smoothing."""

    tables: list[Tables]


class SetupResponse(BaseModel):
    """The failure both word-list methods share, on one corpus and one text.

    Every field here is an answer to the same six characters from a method fitted
    on the same three sentences, so the difference between them is the method.
    """

    corpus_sentences: list[list[str]]
    corpus_words: list[str]
    text: str
    reader_reading: list[str]
    from_a_greedy_scan: list[str]
    from_a_word_list: list[str]
    from_the_tagger: list[str]
    n_new_words: int
    sentence: str
    spaceless_sentence: str
    sentence_reading: list[str]
    surname: str
    name_sentence: str
    name_reading: list[str]
    name_from_a_word_list: list[str]
    name_from_the_tagger: list[str]
    name_from_both: list[str]


class SmoothingStep(BaseModel):
    """One smoothing value, and the reading the corpus's own sentence comes back as."""

    smoothing: float
    words: list[str]
    tags: str
    reproduces_the_corpus: bool
    begin_after_begin_is_middle: float
    probability_of_the_compound_start: float


class SmoothingResponse(BaseModel):
    """The value at which a corpus that contradicts itself changes its mind.

    ``turns_between`` is the pair of smoothing values the answer changes between,
    found by halving the interval until the two agree to twelve places.
    """

    sentences: list[list[str]]
    text: str
    majority_reading: list[str]
    minority_reading: list[str]
    n_majority: int
    n_minority: int
    steps: list[SmoothingStep]
    turns_between: list[float]


class Quality(BaseModel):
    """How well one method answered the held-out sentences.

    ``recall_on_seen`` and ``recall_on_unseen`` split the same recall by whether
    the word was in the training sentences at all, which is where the whole
    difference between the two methods lives.
    """

    f_measure: float
    precision: float
    recall: float
    recall_on_seen: float
    recall_on_unseen: float


class ScaleStep(BaseModel):
    """One training size, and what each method scored on the held-out sentences."""

    n_sentences: int
    n_known_words: int
    share_unseen: float
    tagger: Quality
    word_list: Quality
    together: Quality


class ScaleCorpus(BaseModel):
    """One generated language, measured along the same held-out sentences throughout.

    ``bits_told`` is how much of the uncertainty about a character's place the
    character itself removes, which is the quantity that decides whether the
    method can work on a writing system at all.
    """

    key: str
    label: str
    n_words: int
    n_alphabet: int
    tag_bits: float
    bits_left: float
    bits_told: float
    n_test_sentences: int
    steps: list[ScaleStep]
    turns_between: list[int]
    example_text: str
    example_truth: list[str]
    example_from_the_tagger: list[str]
    example_from_a_word_list: list[str]
    running_text: str | None
    running_truth: list[str] | None
    running_from_the_tagger: list[str] | None
    running_from_a_word_list: list[str] | None


class ScaleResponse(BaseModel):
    """Both generated languages, fitted on prefixes of one corpus and scored alike."""

    corpora: list[ScaleCorpus]


class TwinText(BaseModel):
    """One text, what the two fits answer, and whether either language contains it."""

    text: str
    answer: list[str]
    in_the_first_language: bool
    in_the_second_language: bool


class UnseenRun(BaseModel):
    """A run of characters the corpus never held, and what the steps alone decide."""

    n_characters: int
    tags: str
    n_words: int


class LimitsResponse(BaseModel):
    """Two places where the method answers without having anything to answer from.

    The first is a pair of languages whose counts are equal in every entry, so no
    quantity of either could tell them apart. The second is a run of characters
    none of which was ever seen, where every character is worth the same and the
    answer comes out of the steps alone.
    """

    first_sentences: list[list[str]]
    second_sentences: list[list[str]]
    n_numbers_compared: int
    n_numbers_equal: int
    twin_texts: list[TwinText]
    unseen_runs: list[UnseenRun]
    unseen_texts: list[str]
    unseen_tags: str


class SegmentRequest(BaseModel):
    """A text, the sentences to learn from, and how much to smooth the counts."""

    text: BoundedText
    corpus: CorpusKey = CorpusKey.FIVE_SENTENCES
    smoothing: float = Field(default=1.0, ge=MIN_SMOOTHING, le=MAX_SMOOTHING)

    model_config = {"extra": "forbid"}


class SegmentResponse(BaseModel):
    """The requested text tagged, with the tables it was tagged by."""

    tagging: Tagging
    tables: Tables


# --------------------------------------------------------------------------
# The work.
# --------------------------------------------------------------------------


def _fitted(
    corpus: tuple[tuple[str, ...], ...], smoothing: float
) -> HiddenMarkovSegmenter:
    """One model, fitted on the sentences as they are written above."""
    return HiddenMarkovSegmenter(smoothing=smoothing).fit(
        [list(sentence) for sentence in corpus]
    )


def _dictionary_of(corpus: tuple[tuple[str, ...], ...]) -> WordDictionary:
    """A word list counted from the very same sentences, for the comparison."""
    return WordDictionary.from_segmented_corpus([list(sentence) for sentence in corpus])


def _words_of(corpus: tuple[tuple[str, ...], ...]) -> list[str]:
    """Every word the sentences contain, once each, in the order they first appear."""
    seen: dict[str, None] = {}
    for sentence in corpus:
        for word in sentence:
            seen.setdefault(word, None)
    return list(seen)


def _trellis(
    model: HiddenMarkovSegmenter, run: str
) -> tuple[list[dict[BoundaryTag, float]], list[dict[BoundaryTag, BoundaryTag]]]:
    """Every place scored at every character, out of the model's own numbers.

    The recurrence is the one the segmenter runs; what is different here is that
    the losing places are kept rather than dropped, since a page showing only the
    winner cannot show what it beat. The tests assert the path walked back out of
    this equals the one the model answers with.
    """
    scores: dict[BoundaryTag, float] = {tag: -math.inf for tag in BoundaryTag}
    for tag in INITIAL_TAGS:
        scores[tag] = model.initial_log_probability(
            tag
        ) + model.emission_log_probability(tag, run[0])
    columns = [dict(scores)]
    pointers: list[dict[BoundaryTag, BoundaryTag]] = []

    for character in run[1:]:
        following_scores: dict[BoundaryTag, float] = {
            tag: -math.inf for tag in BoundaryTag
        }
        chosen: dict[BoundaryTag, BoundaryTag] = {}
        for previous, successors in ADMISSIBLE_SUCCESSORS.items():
            if scores[previous] == -math.inf:
                continue
            for following in successors:
                candidate = (
                    scores[previous]
                    + model.transition_log_probability(previous, following)
                    + model.emission_log_probability(following, character)
                )
                if candidate > following_scores[following]:
                    following_scores[following] = candidate
                    chosen[following] = previous
        columns.append(dict(following_scores))
        pointers.append(chosen)
        scores = following_scores

    return columns, pointers


def _path_of(
    columns: list[dict[BoundaryTag, float]],
    pointers: list[dict[BoundaryTag, BoundaryTag]],
) -> list[BoundaryTag]:
    """The best tag sequence, walked back from the better of the two ending places."""
    last = max(FINAL_TAGS, key=lambda tag: columns[-1][tag])
    path = [last]
    for step in reversed(pointers):
        path.append(step[path[-1]])
    path.reverse()
    return path


def _tagging(
    key: str,
    label: str,
    text: str,
    corpus: tuple[tuple[str, ...], ...],
    corpus_key: CorpusKey,
    reader: tuple[str, ...] | None,
    smoothing: float = 1.0,
) -> Tagging:
    """One whole text tagged, scored, and set beside what a word list answers.

    Whitespace ends a run before anything is tagged, so each run is walked on its
    own and carries no column; that is why the places after a space always begin
    a word rather than continuing one.
    """
    model = _fitted(corpus, smoothing)
    alphabet = set(model.alphabet)
    words = list(model.split(text).texts)
    corpus_words = _words_of(corpus)

    dictionary = _dictionary_of(corpus)
    from_list = list(
        DictionaryLatticeSegmenter(dictionary=dictionary).split(text).texts
    )
    from_greedy = list(
        MaximumMatchingSegmenter(
            dictionary=dictionary, direction=MatchingDirection.FORWARD
        )
        .split(text)
        .texts
    )

    columns: list[TrellisColumn] = []
    tags: list[BoundaryTag] = []
    total = 0.0
    for match in NON_WHITESPACE_RUN.finditer(text):
        run = match.group()
        columns_scores, pointers = _trellis(model, run)
        path = _path_of(columns_scores, pointers)
        tags.extend(path)
        total += max(columns_scores[-1][tag] for tag in FINAL_TAGS)
        for offset, character in enumerate(run):
            cells = [
                Cell(
                    tag=tag.value,
                    tag_name=TAG_NAMES[tag],
                    score=(
                        None
                        if columns_scores[offset][tag] == -math.inf
                        else columns_scores[offset][tag]
                    ),
                    came_from=(
                        pointers[offset - 1][tag].value
                        if offset > 0 and tag in pointers[offset - 1]
                        else None
                    ),
                    on_best_path=path[offset] is tag,
                )
                for tag in BoundaryTag
            ]
            columns.append(
                TrellisColumn(
                    position=match.start() + offset,
                    character=character,
                    in_alphabet=character in alphabet,
                    cells=cells,
                    tag=path[offset].value,
                )
            )

    return Tagging(
        key=key,
        label=label,
        text=text,
        corpus_label=CORPUS_LABELS[corpus_key],
        smoothing=smoothing,
        columns=columns,
        tags="".join(tag.value for tag in tags),
        words=words,
        total_log_score=total,
        n_characters=len(columns),
        n_unseen_characters=sum(1 for column in columns if not column.in_alphabet),
        reader_reading=list(reader) if reader is not None else None,
        matches_reader=reader is not None and tuple(words) == reader,
        from_a_word_list=from_list,
        list_matches_reader=reader is not None and tuple(from_list) == reader,
        from_a_greedy_scan=from_greedy,
        words_in_the_corpus=corpus_words,
        n_new_words=sum(1 for word in words if word not in corpus_words),
    )


def _tables(corpus_key: CorpusKey, smoothing: float = 1.0) -> Tables:
    """The three tables one fit learned, with the counts underneath them."""
    corpus = CORPORA[corpus_key]
    model = _fitted(corpus, smoothing)

    start_counts: Counter[BoundaryTag] = Counter()
    step_counts: dict[BoundaryTag, Counter[BoundaryTag]] = {
        tag: Counter() for tag in BoundaryTag
    }
    character_counts: dict[BoundaryTag, Counter[str]] = {
        tag: Counter() for tag in BoundaryTag
    }
    for sentence in corpus:
        tags = [tag for word in sentence for tag in tags_of_word(word)]
        start_counts[tags[0]] += 1
        for previous, following in pairwise(tags):
            step_counts[previous][following] += 1
        for character, tag in zip("".join(sentence), tags, strict=True):
            character_counts[tag][character] += 1

    start_total = sum(start_counts.values())
    starts = [
        StartRow(
            tag=tag.value,
            tag_name=TAG_NAMES[tag],
            count=start_counts[tag],
            total=start_total,
            admissible=tag in INITIAL_TAGS,
            probability=(
                math.exp(model.initial_log_probability(tag))
                if tag in INITIAL_TAGS
                else None
            ),
            log_probability=(
                model.initial_log_probability(tag) if tag in INITIAL_TAGS else None
            ),
        )
        for tag in BoundaryTag
    ]

    steps: list[StepRow] = []
    for previous in BoundaryTag:
        total = sum(step_counts[previous].values())
        for following in BoundaryTag:
            admissible = following in ADMISSIBLE_SUCCESSORS[previous]
            steps.append(
                StepRow(
                    previous=previous.value,
                    following=following.value,
                    count=step_counts[previous][following],
                    total=total,
                    admissible=admissible,
                    probability=(
                        math.exp(model.transition_log_probability(previous, following))
                        if admissible
                        else None
                    ),
                    log_probability=(
                        model.transition_log_probability(previous, following)
                        if admissible
                        else None
                    ),
                )
            )

    alphabet = list(model.alphabet)
    n_slots = len(alphabet) + 1
    characters: list[CharacterRow] = []
    floors: list[CharacterRow] = []
    for tag in BoundaryTag:
        total = sum(character_counts[tag].values())
        for character in alphabet:
            characters.append(
                CharacterRow(
                    tag=tag.value,
                    character=character,
                    count=character_counts[tag][character],
                    total=total,
                    probability=math.exp(
                        model.emission_log_probability(tag, character)
                    ),
                    log_probability=model.emission_log_probability(tag, character),
                )
            )
        floors.append(
            CharacterRow(
                tag=tag.value,
                character="",
                count=0,
                total=total,
                probability=math.exp(model.emission_log_probability(tag, UNSEEN_PROBE)),
                log_probability=model.emission_log_probability(tag, UNSEEN_PROBE),
            )
        )

    return Tables(
        key=corpus_key.value,
        label=CORPUS_LABELS[corpus_key],
        sentences=[list(sentence) for sentence in corpus],
        n_sentences=len(corpus),
        n_words=sum(len(sentence) for sentence in corpus),
        alphabet=alphabet,
        n_alphabet=len(alphabet),
        smoothing=smoothing,
        n_slots=n_slots,
        starts=starts,
        steps=steps,
        characters=characters,
        floors=floors,
        n_admissible_steps=sum(
            len(successors) for successors in ADMISSIBLE_SUCCESSORS.values()
        ),
        n_forbidden_steps=len(BoundaryTag) ** 2
        - sum(len(successors) for successors in ADMISSIBLE_SUCCESSORS.values()),
    )


@cache
def _scenarios() -> ScenariosResponse:
    """The page's fixed texts, each tagged all the way through, computed once."""
    return ScenariosResponse(
        scenarios=[
            _tagging(
                key="research",
                label="the sentence with two defensible readings",
                text=RESEARCH_SENTENCE,
                corpus=FIVE_SENTENCE_CORPUS,
                corpus_key=CorpusKey.FIVE_SENTENCES,
                reader=LIFE_READING,
            ),
            _tagging(
                key="student",
                label="another sentence of the same corpus",
                text=STUDENT_SENTENCE,
                corpus=FIVE_SENTENCE_CORPUS,
                corpus_key=CorpusKey.FIVE_SENTENCES,
                reader=STUDENT_SENTENCE_READING,
            ),
            _tagging(
                key="novel",
                label="three words the sentences never held",
                text=NOVEL_TEXT,
                corpus=BOUNDARY_CORPUS,
                corpus_key=CorpusKey.BOUNDARY,
                reader=NOVEL_READING,
            ),
            _tagging(
                key="minority",
                label="the sentence its own corpus contradicts",
                text=MINORITY_SENTENCE,
                corpus=CONTRADICTORY_CORPUS,
                corpus_key=CorpusKey.CONTRADICTORY,
                reader=MINORITY_READING,
            ),
            _tagging(
                key="name",
                label="the research sentence behind a surname",
                text=NAME_SENTENCE,
                corpus=FIVE_SENTENCE_CORPUS,
                corpus_key=CorpusKey.FIVE_SENTENCES,
                reader=NAME_READING,
            ),
            _tagging(
                key="two-sentence",
                label="the small corpus reading its own characters",
                text="abcab",
                corpus=TWO_SENTENCE_CORPUS,
                corpus_key=CorpusKey.TWO_SENTENCES,
                reader=None,
            ),
        ]
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    """The page's texts, each tagged all the way through with every place scored."""
    return _scenarios()


@cache
def _tables_response() -> TablesResponse:
    """The tables of the two corpora the page reads numbers off, computed once."""
    return TablesResponse(
        tables=[
            _tables(CorpusKey.TWO_SENTENCES),
            _tables(CorpusKey.FIVE_SENTENCES),
        ]
    )


@router.get("/tables", response_model=TablesResponse)
def tables() -> TablesResponse:
    """Everything two fits learned, with the counts the smoothed numbers came from."""
    return _tables_response()


@cache
def _setup() -> SetupResponse:
    """What each method answers for a word its training sentences never held."""
    dictionary = _dictionary_of(BOUNDARY_CORPUS)
    tagger = _fitted(BOUNDARY_CORPUS, 1.0)
    tagged = list(tagger.split(NOVEL_TEXT).texts)
    corpus_words = _words_of(BOUNDARY_CORPUS)

    five_dictionary = _dictionary_of(FIVE_SENTENCE_CORPUS)
    five_tagger = _fitted(FIVE_SENTENCE_CORPUS, 1.0)

    return SetupResponse(
        corpus_sentences=[list(sentence) for sentence in BOUNDARY_CORPUS],
        corpus_words=corpus_words,
        text=NOVEL_TEXT,
        reader_reading=list(NOVEL_READING),
        from_a_greedy_scan=list(
            MaximumMatchingSegmenter(
                dictionary=dictionary, direction=MatchingDirection.FORWARD
            )
            .split(NOVEL_TEXT)
            .texts
        ),
        from_a_word_list=list(
            DictionaryLatticeSegmenter(dictionary=dictionary).split(NOVEL_TEXT).texts
        ),
        from_the_tagger=tagged,
        n_new_words=sum(1 for word in tagged if word not in corpus_words),
        sentence=RUNNING_SENTENCE,
        spaceless_sentence=SPACELESS_SENTENCE,
        sentence_reading=list(SENTENCE_READING),
        surname=SURNAME,
        name_sentence=NAME_SENTENCE,
        name_reading=list(NAME_READING),
        name_from_a_word_list=list(
            DictionaryLatticeSegmenter(dictionary=five_dictionary)
            .split(NAME_SENTENCE)
            .texts
        ),
        name_from_the_tagger=list(five_tagger.split(NAME_SENTENCE).texts),
        name_from_both=list(
            DictionaryLatticeSegmenter(
                dictionary=five_dictionary, unknown_segmenter=five_tagger
            )
            .split(NAME_SENTENCE)
            .texts
        ),
    )


@router.get("/setup", response_model=SetupResponse)
def setup() -> SetupResponse:
    """The word no list can produce, answered by three methods taught the same sentences."""
    return _setup()


def _turning_smoothing() -> tuple[float, float]:
    """The two smoothing values the contradictory corpus changes its mind between."""
    low, high = 0.01, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        answer = _fitted(CONTRADICTORY_CORPUS, middle).split(MINORITY_SENTENCE).texts
        if tuple(answer) == MINORITY_READING:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return low, high


@cache
def _smoothing() -> SmoothingResponse:
    """Where a corpus that disagrees with itself stops reproducing its own sentence."""
    steps: list[SmoothingStep] = []
    for smoothing in SMOOTHING_LADDER:
        model = _fitted(CONTRADICTORY_CORPUS, smoothing)
        words = list(model.split(MINORITY_SENTENCE).texts)
        steps.append(
            SmoothingStep(
                smoothing=smoothing,
                words=words,
                tags="".join(tag.value for tag in model.tags_of(MINORITY_SENTENCE)),
                reproduces_the_corpus=tuple(words) == MINORITY_READING,
                begin_after_begin_is_middle=math.exp(
                    model.transition_log_probability(
                        BoundaryTag.BEGIN, BoundaryTag.MIDDLE
                    )
                ),
                probability_of_the_compound_start=math.exp(
                    model.emission_log_probability(BoundaryTag.MIDDLE, "究")
                ),
            )
        )

    low, high = _turning_smoothing()
    majority = sum(
        1 for sentence in CONTRADICTORY_CORPUS for word in sentence if word == "研究"
    )
    minority = sum(
        1 for sentence in CONTRADICTORY_CORPUS for word in sentence if word == "研究生"
    )
    return SmoothingResponse(
        sentences=[list(sentence) for sentence in CONTRADICTORY_CORPUS],
        text=MINORITY_SENTENCE,
        majority_reading=list(MAJORITY_READING),
        minority_reading=list(MINORITY_READING),
        n_majority=majority,
        n_minority=minority,
        steps=steps,
        turns_between=[low, high],
    )


@router.get("/smoothing", response_model=SmoothingResponse)
def smoothing() -> SmoothingResponse:
    """One free number, and the two readings it decides between on one corpus."""
    return _smoothing()


# --------------------------------------------------------------------------
# The two generated languages, and how much tagged text each method needs.
# --------------------------------------------------------------------------


def _language_words() -> list[str]:
    """Seventy words spelled out of forty characters, drawn once with a fixed seed."""
    generator = random.Random(LANGUAGE_SEED)
    characters = list(LANGUAGE_CHARACTERS)
    words: list[str] = []
    seen: set[str] = set()
    for character in generator.sample(characters, 10):
        words.append(character)
        seen.add(character)
    while len(words) < N_LANGUAGE_WORDS:
        length = 2 if len(words) < 50 else 3
        word = "".join(generator.choices(characters, k=length))
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def _generated_sentences(
    vocabulary: list[str], count: int, seed: int
) -> list[list[str]]:
    """Sentences of four to nine words, drawn with a rank-weighted hand.

    The weight of the word at rank r is one over r, which is the shape a real
    vocabulary has: a few words carry most of the text and the tail is long. That
    tail is the whole reason the two methods come apart, since it is what puts
    words in a held-out sentence that no training sentence contained.
    """
    generator = random.Random(seed)
    weights = [1.0 / (rank + 1) for rank in range(len(vocabulary))]
    return [
        generator.choices(vocabulary, weights=weights, k=generator.randint(4, 9))
        for _ in range(count)
    ]


def _spans(words: list[str]) -> set[tuple[int, int]]:
    """Where each word starts and ends, which is what two answers are compared by."""
    found: set[tuple[int, int]] = set()
    position = 0
    for word in words:
        found.add((position, position + len(word)))
        position += len(word)
    return found


def _quality(
    answers: list[list[str]], truth: list[list[str]], known: set[str]
) -> Quality:
    """Precision and recall over word spans, split by whether the word was seen.

    A word counts as recovered only when both its ends land where the sentence
    put them, which is the ordinary way segmentation is scored and is stricter
    than counting boundaries: a word with one end right and one wrong is wrong.
    """
    hit = answered = wanted = 0
    seen_hit = seen_total = unseen_hit = unseen_total = 0
    for answer, gold in zip(answers, truth, strict=True):
        given = _spans(answer)
        expected = _spans(gold)
        hit += len(given & expected)
        answered += len(given)
        wanted += len(expected)
        position = 0
        for word in gold:
            span = (position, position + len(word))
            position += len(word)
            if word in known:
                seen_total += 1
                seen_hit += span in given
            else:
                unseen_total += 1
                unseen_hit += span in given
    precision = hit / answered if answered else 0.0
    recall = hit / wanted if wanted else 0.0
    return Quality(
        f_measure=(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        precision=precision,
        recall=recall,
        recall_on_seen=seen_hit / seen_total if seen_total else 0.0,
        recall_on_unseen=unseen_hit / unseen_total if unseen_total else 0.0,
    )


def _bits_told(corpus: list[list[str]]) -> tuple[float, float, float]:
    """How much of the uncertainty about a character's place the character removes.

    The first number is how uncertain the place is before the character is read,
    the second how uncertain it still is after, and the third the difference. All
    three are in bits, and the third is the one that decides whether a writing
    system suits this method at all.
    """
    joint: Counter[tuple[str, BoundaryTag]] = Counter()
    for sentence in corpus:
        for word in sentence:
            for character, tag in zip(word, tags_of_word(word), strict=True):
                joint[(character, tag)] += 1
    total = sum(joint.values())
    by_character: Counter[str] = Counter()
    by_tag: Counter[BoundaryTag] = Counter()
    for (character, tag), count in joint.items():
        by_character[character] += count
        by_tag[tag] += count

    def entropy(counts: list[int], denominator: int) -> float:
        return -sum(
            (count / denominator) * math.log2(count / denominator)
            for count in counts
            if count
        )

    before = entropy(list(by_tag.values()), total)
    after = 0.0
    for character, count in by_character.items():
        inner = [
            joint[(character, tag)] for tag in BoundaryTag if joint[(character, tag)]
        ]
        after += (count / total) * entropy(inner, count)
    return before, after, before - after


def _scale_corpus(
    key: str,
    label: str,
    vocabulary: list[str],
    training_seed: int,
    testing_seed: int,
    sizes: tuple[int, ...],
    running: tuple[str, tuple[str, ...]] | None = None,
) -> ScaleCorpus:
    """One language, both methods, along the prefixes of one training corpus."""
    training = _generated_sentences(vocabulary, N_TRAINING_SENTENCES, training_seed)
    testing = _generated_sentences(vocabulary, N_TEST_SENTENCES, testing_seed)
    texts = ["".join(sentence) for sentence in testing]

    steps: list[ScaleStep] = []
    turns_low = 0
    turns_high = 0
    for size in sizes:
        part = training[:size]
        known = {word for sentence in part for word in sentence}
        tagger = HiddenMarkovSegmenter().fit(part)
        dictionary = WordDictionary.from_segmented_corpus(part)
        lister = DictionaryLatticeSegmenter(dictionary=dictionary)
        together = DictionaryLatticeSegmenter(
            dictionary=dictionary, unknown_segmenter=tagger
        )

        tagged = _quality(
            [list(tagger.split(text).texts) for text in texts], testing, known
        )
        listed = _quality(
            [list(lister.split(text).texts) for text in texts], testing, known
        )
        both = _quality(
            [list(together.split(text).texts) for text in texts], testing, known
        )
        unseen = sum(
            1 for sentence in testing for word in sentence if word not in known
        )
        wanted = sum(len(sentence) for sentence in testing)

        if listed.f_measure <= tagged.f_measure:
            turns_low = size
        elif turns_high == 0:
            turns_high = size

        steps.append(
            ScaleStep(
                n_sentences=size,
                n_known_words=len(known),
                share_unseen=unseen / wanted,
                tagger=tagged,
                word_list=listed,
                together=both,
            )
        )

    small = training[: sizes[len(sizes) // 3]]
    tagger = HiddenMarkovSegmenter().fit(small)
    dictionary = WordDictionary.from_segmented_corpus(small)
    example = testing[0]
    example_text = "".join(example)

    # The running sentence is asked of the largest fit rather than the small one,
    # so what comes back is the best this method has to offer on that language.
    largest = training[: sizes[-1]]
    widest = HiddenMarkovSegmenter().fit(largest)
    widest_dictionary = WordDictionary.from_segmented_corpus(largest)

    before, after, told = _bits_told(training)
    return ScaleCorpus(
        key=key,
        label=label,
        n_words=len(vocabulary),
        n_alphabet=len({character for word in vocabulary for character in word}),
        tag_bits=before,
        bits_left=after,
        bits_told=told,
        n_test_sentences=len(testing),
        steps=steps,
        turns_between=[turns_low, turns_high],
        example_text=example_text,
        example_truth=list(example),
        example_from_the_tagger=list(tagger.split(example_text).texts),
        example_from_a_word_list=list(
            DictionaryLatticeSegmenter(dictionary=dictionary).split(example_text).texts
        ),
        running_text=running[0] if running is not None else None,
        running_truth=list(running[1]) if running is not None else None,
        running_from_the_tagger=(
            list(widest.split(running[0]).texts) if running is not None else None
        ),
        running_from_a_word_list=(
            list(
                DictionaryLatticeSegmenter(dictionary=widest_dictionary)
                .split(running[0])
                .texts
            )
            if running is not None
            else None
        ),
    )


@cache
def _scale() -> ScaleResponse:
    """Both languages measured, computed once because the fits are the slow part."""
    return ScaleResponse(
        corpora=[
            _scale_corpus(
                key="characters",
                label="a language of seventy words over forty characters",
                vocabulary=_language_words(),
                training_seed=TRAINING_SEED,
                testing_seed=TESTING_SEED,
                sizes=LANGUAGE_SIZES,
            ),
            _scale_corpus(
                key="letters",
                label="ordinary English words with the spaces taken out",
                vocabulary=list(ORDINARY_ENGLISH),
                training_seed=ENGLISH_TRAINING_SEED,
                testing_seed=ENGLISH_TESTING_SEED,
                sizes=ENGLISH_SIZES,
                running=(SPACELESS_SENTENCE, SENTENCE_READING),
            ),
        ]
    )


@router.get("/scale", response_model=ScaleResponse)
def scale() -> ScaleResponse:
    """How much tagged text each method needs, on two languages scored the same way."""
    return _scale()


@cache
def _limits() -> LimitsResponse:
    """Two arrangements on which the method answers with nothing to answer from."""
    first = _fitted(PAIRED_CORPUS, 1.0)
    second = _fitted(CROSSED_CORPUS, 1.0)

    compared = 0
    equal = 0
    for tag in BoundaryTag:
        compared += 1
        equal += first.initial_log_probability(tag) == second.initial_log_probability(
            tag
        )
        for following in BoundaryTag:
            compared += 1
            equal += first.transition_log_probability(
                tag, following
            ) == second.transition_log_probability(tag, following)
        for character in first.alphabet:
            compared += 1
            equal += first.emission_log_probability(
                tag, character
            ) == second.emission_log_probability(tag, character)

    first_words = set(_words_of(PAIRED_CORPUS))
    second_words = set(_words_of(CROSSED_CORPUS))
    twin_texts = [
        TwinText(
            text=text,
            answer=list(first.split(text).texts),
            in_the_first_language=all(
                piece in first_words for piece in first.split(text).texts
            ),
            in_the_second_language=all(
                piece in second_words for piece in first.split(text).texts
            ),
        )
        for text in ("abcd", CROSSED_TEXT)
    ]

    five = _fitted(FIVE_SENTENCE_CORPUS, 1.0)
    runs = [
        UnseenRun(
            n_characters=length,
            tags="".join(
                tag.value
                for tag in five.tags_of(
                    "".join(chr(0x9000 + index) for index in range(length))
                )
            ),
            n_words=five.split(
                "".join(chr(0x9000 + index) for index in range(length))
            ).n_words,
        )
        for length in UNSEEN_RUN_LENGTHS
    ]

    return LimitsResponse(
        first_sentences=[list(sentence) for sentence in PAIRED_CORPUS],
        second_sentences=[list(sentence) for sentence in CROSSED_CORPUS],
        n_numbers_compared=compared,
        n_numbers_equal=equal,
        twin_texts=twin_texts,
        unseen_runs=runs,
        unseen_texts=list(UNSEEN_RUNS),
        unseen_tags="".join(tag.value for tag in five.tags_of(UNSEEN_RUNS[0])),
    )


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """Where the counts cannot tell two languages apart, and where they say nothing."""
    return _limits()


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest) -> SegmentResponse:
    """Tag the requested text with a model fitted on the chosen sentences."""
    corpus = CORPORA[request.corpus]
    if not NON_WHITESPACE_RUN.search(request.text):
        raise HTTPException(
            status_code=422,
            detail="a text needs at least one character that is not a space",
        )
    try:
        tagging = _tagging(
            key="requested",
            label="your text",
            text=request.text,
            corpus=corpus,
            corpus_key=request.corpus,
            reader=None,
            smoothing=request.smoothing,
        )
        learned = _tables(request.corpus, request.smoothing)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal

    return SegmentResponse(tagging=tagging, tables=learned)
