"""Markov chains, a process that remembers only where it is.

Every endpoint here is registered under ``/concepts/markov-chains``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about the transition table counted from sequences, what it predicts
several steps ahead, where it settles, and what the assumption that only the
present matters throws away.

The running text
----------------
The first three chapters of *Alice's Adventures in Wonderland*, as the API
already holds them in :mod:`oop_ml_api.alice_corpus`. The first two chapters,
128 sentences, are what every chain here is counted on; the third, 61
sentences, is held back and only ever read. Each sentence is one sequence, so
counting stops at a full stop and nothing is counted across the join between two
sentences, since nobody observed that step.

Two alphabets are counted on the same letters. The first is Markov's own from
1913, every letter a vowel or a consonant, spaces dropped as he dropped them,
with ``y`` counted as a consonant, which is a convention and is said so on the
page. The second is every letter and the space, twenty-seven states, which is
the alphabet of Shannon's 1948 approximations to English.

Five endpoints
--------------
``/vowels`` is everything about the two-state chain: one short sentence counted
by hand, the same count on two chapters, the likelihood of the vowel row as a
curve, the walk several steps ahead, what the chain says two steps ahead against
what the text does, and the table counted chapter by chapter. ``/letters`` is
the twenty-seven state chain: its table, its long-run shares against the
letters of three texts, how fast it forgets, what it writes at three lengths of
memory, and what longer memory costs in cells. ``/memory`` scores the held-back
chapter under every model on one set of positions. ``/edges`` is the four small
chains, each a single word or two, on which the stationary distribution is not
the ordinary case, and the two states the real text strands. ``/fit`` is the
playground.

What is computed here rather than by a chain
--------------------------------------------
Three things, each named where it happens. The log likelihood of the vowel row
as a curve is the definition, a sum of counts times logarithms, evaluated on
the chain's own counts; a widget may draw a definition. Scoring a held-back text
reads each step's probability off the fitted table, once per step, because the
comparison needs every model scored on exactly the same positions and a step to
a state the table never met has to be set aside rather than scored; the sum of
those readings is what ``log_probability_of`` would return on the same steps.
And the text written by a model with no memory at all is drawn letter by letter
from the chain's own stationary distribution, since a chain always remembers one
state and so has no method for forgetting it.

Every refusal a widget can show is written here in the page's own words, and the
test module asserts that no identifier reaches any string a widget prints.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from itertools import pairwise

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml.core.exceptions import (
    MLLibError,
    NonUniqueStationaryDistributionError,
    TooFewValuesError,
)
from oop_ml.numpy.sequences.markov_chain import MarkovChain
from pydantic import BaseModel, Field

from oop_ml_api.alice_corpus import SENTENCES

router = APIRouter(prefix="/concepts/markov-chains", tags=["markov-chains"])


# --------------------------------------------------------------------------
# The texts and the two alphabets.
# --------------------------------------------------------------------------

FIRST_CHAPTER_END = 54
"""Where the second chapter, *The Pool of Tears*, begins in the sentence list."""

COUNTED_END = 128
"""Where the third chapter, *A Caucus-Race and a Long Tale*, begins."""

COUNTED: tuple[str, ...] = SENTENCES[:COUNTED_END]
"""The first two chapters, which every chain on the page is counted on."""

HELD_BACK: tuple[str, ...] = SENTENCES[COUNTED_END:]
"""The third chapter, never counted, only read."""

CHAPTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("the first chapter", SENTENCES[:FIRST_CHAPTER_END]),
    ("the second chapter", SENTENCES[FIRST_CHAPTER_END:COUNTED_END]),
    ("the third chapter", SENTENCES[COUNTED_END:]),
)
"""The three chapters, each counted on its own for the homogeneity check."""

VOWEL = "V"
CONSONANT = "C"
CLASS_ORDER: tuple[str, str] = (VOWEL, CONSONANT)
"""The two states of Markov's alphabet, in the order every answer is given in."""

VOWEL_LETTERS = frozenset("aeiou")
"""The letters counted as vowels. ``y`` is a consonant here, which is a choice."""

SPACE = " "
LETTER_ORDER: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz") + (SPACE,)
"""The twenty-seven states of the letter alphabet, in the order tables are drawn."""

HAND_SENTENCE = "oh my dear dinah"
"""Alice calling her cat, from the third chapter: thirteen letters, twelve steps."""

LETTER_SMOOTHING = 0.5
"""The smoothing the held-back letter scores use wherever one setting is wanted.

The sweep in ``/memory`` shows the one-letter chain barely cares, reading
between 3.2436 and 3.2645 bits across a thousandfold range of settings, so the
choice is not doing the work in any comparison it is used for.
"""

SMOOTHING_SWEEP: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
"""The additive counts the held-back letter scores are swept over."""

CLASS_SMOOTHINGS: tuple[float, ...] = (0.0, 0.5)
"""The two settings the vowel-and-consonant memory sweep is read at."""

CLASS_WIDTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
"""How many classes back each vowel-and-consonant model remembers."""

LETTER_WIDTHS: tuple[int, ...] = (0, 1, 2)
"""How many letters back each letter model remembers, for the held-back scores."""

COST_WIDTHS: tuple[int, ...] = (1, 2, 3)
"""How many letters back each chain in the cost table remembers."""

AHEAD_STEPS: tuple[int, ...] = (1, 2, 3, 4, 6, 10)
"""How far ahead the one-letter chain is asked to predict the held-back text."""

AHEAD_FIRST_POSITION = 10
"""The first position of each sentence scored by the steps-ahead series.

Ten, so that even the ten-step prediction has a letter to predict from, and so
every row of that series is scored on the same positions.
"""

MEMORY_FIRST_POSITION = 2
"""The first position the letter memory sweep scores, so two letters stand behind it."""

CLASS_FIRST_POSITION = 6
"""The first position the class memory sweep scores, so six classes stand behind it."""

TWO_STEP_GAPS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""How many steps apart the chain and the text are compared."""

AFTER_STEPS: tuple[int, ...] = tuple(range(13))
"""The step counts the two-state walk is followed across."""

MIXING_STEPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 15)
"""The step counts at which the letter chain's worst start is measured."""

LONG_WALK = 20000
"""How many letters the chain writes when its shares are checked against a walk."""

SAMPLE_LENGTH = 150
"""How many letters each written sample runs to."""

SAMPLE_SEED = 1
"""The seed every written sample on the page uses, so they can be quoted."""

TRIPLE_SMOOTHING = 0.001
"""The smoothing the three-letter chain needs before it will fit at all.

Two runs of three letters end a sentence and occur nowhere else, so their rows
are zero over zero; this is the smallest setting in the sweep, chosen so its
invented steps stay well below a row's real ones.
"""

LIKELIHOOD_GRID: tuple[float, ...] = tuple(
    round(0.01 * step, 2) for step in range(1, 61)
)
"""The candidate values of P(vowel after a vowel) the likelihood curve is drawn at."""

EDGE_STEPS: tuple[int, ...] = tuple(range(9))
"""How many steps each small chain's walk is followed across."""

LONG_RUN_STEPS = 1000
"""How many steps the long-run share of time is averaged over."""

SETTLE_CHECK = 1000
"""How far a walk is taken before asking whether it has stopped moving."""

SETTLE_TOLERANCE = 1e-9
"""How close two consecutive distributions must be to count as settled."""

EDGE_SMOOTHING = 0.5
"""The smoothing each small chain is refitted at to show what it repairs."""

MAX_TEXT_LENGTH = 4000
"""The longest text the playground may be asked to count."""

MAX_SMOOTHING = 5.0
"""The largest smoothing the playground offers, far past where every row is flat."""

MAX_STEPS = 30
"""The most steps ahead the playground will walk."""

RUN_LENGTH = 6
"""How many consecutive sentences the random passage holds."""

WRITTEN_LENGTH = 1500
"""How many symbols the chain writes for the playground's ideal case."""

PLAYGROUND_WALK = 90
"""How many symbols the playground's drawn walk runs to."""

MIXING_HORIZON: tuple[int, ...] = tuple(range(21))
"""The steps at which the playground measures how far the walk is from settled."""


# --------------------------------------------------------------------------
# Turning text into sequences of states.
# --------------------------------------------------------------------------


def _classes(sentence: str) -> list[str]:
    """Every letter of ``sentence`` as a vowel or a consonant, spaces dropped."""
    return [
        VOWEL if letter in VOWEL_LETTERS else CONSONANT
        for letter in sentence
        if letter != SPACE
    ]


def _letters(sentence: str) -> list[str]:
    """Every letter of ``sentence`` and every space, in order."""
    return list(sentence)


def _grouped(sequence: Sequence[str], width: int) -> list[str]:
    """Each run of ``width`` adjacent symbols, overlapping, joined into one state.

    A chain that remembers two letters is a chain whose states are pairs of
    letters, and the step from ``th`` to ``he`` is the step that predicts ``e``
    from ``th``. Width one is the sequence itself.
    """
    return [
        "".join(sequence[position : position + width])
        for position in range(len(sequence) - width + 1)
    ]


def _offset_runs(sequences: Sequence[Sequence[str]], gap: int) -> list[list[str]]:
    """Every sequence split into ``gap`` interleaved subsequences.

    A chain counted on these counts each symbol against the one ``gap`` places
    later, so its table is what the text itself does over that distance. No
    symbol is dropped and no pair is counted twice.
    """
    return [
        list(sequence[offset::gap])
        for sequence in sequences
        for offset in range(gap)
        if sequence[offset::gap]
    ]


def _class_sequences(sentences: Sequence[str]) -> list[list[str]]:
    return [classes for sentence in sentences if (classes := _classes(sentence))]


def _letter_sequences(sentences: Sequence[str]) -> list[list[str]]:
    return [_letters(sentence) for sentence in sentences]


def _shares(sequences: Sequence[Sequence[str]], states: Sequence[str]) -> list[float]:
    """How often each state occurs among every symbol of ``sequences``."""
    tally = Counter(symbol for sequence in sequences for symbol in sequence)
    total = sum(tally.values())
    return [tally[state] / total for state in states]


# --------------------------------------------------------------------------
# The fitted chains, each counted once.
# --------------------------------------------------------------------------


@cache
def _class_chain(width: int = 1, smoothing: float = 0.0) -> MarkovChain:
    """The vowel-and-consonant chain on the first two chapters."""
    return MarkovChain(smoothing=smoothing).fit(
        [
            _grouped(sequence, width)
            for sequence in _class_sequences(COUNTED)
            if len(sequence) >= width
        ]
    )


@cache
def _letter_chain(width: int = 1, smoothing: float = 0.0) -> MarkovChain:
    """The letter chain on the first two chapters, remembering ``width`` letters."""
    return MarkovChain(smoothing=smoothing).fit(
        [
            _grouped(sequence, width)
            for sequence in _letter_sequences(COUNTED)
            if len(sequence) >= width
        ]
    )


def _total_variation(first: Sequence[float], second: Sequence[float]) -> float:
    """Half the summed absolute difference, the largest gap in any event's probability."""
    return 0.5 * float(np.sum(np.abs(np.asarray(first) - np.asarray(second))))


def _in_order(chain: MarkovChain, states: Sequence[str], values: object) -> list[float]:
    """A distribution the chain answered in its own order, read in ``states`` order."""
    by_name = dict(zip(chain.states, np.asarray(values).tolist(), strict=True))
    return [float(by_name[state]) for state in states]


# --------------------------------------------------------------------------
# Refusals, in the page's own words.
# --------------------------------------------------------------------------

PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("NonUniqueStationaryDistributionError", "this refusal"),
    ("TooFewValuesError", "this refusal"),
    ("InvalidValuesError", "this refusal"),
    ("EmptyValuesError", "this refusal"),
    ("MarkovChain", "this chain"),
    ("n_steps", "the number of steps"),
    ("random_seed", "the seed"),
)
"""The substitutions that take an identifier out of a message a reader sees."""

STATE_NAMES = {VOWEL: "vowel", CONSONANT: "consonant", SPACE: "the space"}


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence with the identifiers taken out of it."""
    sentence = str(refusal)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


def _named(states: Sequence[str]) -> str:
    """States as a reader would name them, joined into a phrase."""
    names = [STATE_NAMES.get(state, f"“{state}”") for state in states]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _never_left_note(states: Sequence[str]) -> str:
    return (
        f"{_named(states)[0].upper()}{_named(states)[1:]} never "
        f"{'begins' if len(states) == 1 else 'begin'} a counted step, so "
        f"{'its row is' if len(states) == 1 else 'their rows are'} zero over zero "
        "and there is no table until a smoothing above zero is added."
    )


SEVERAL_GROUPS_NOTE = (
    "This table holds more than one group of states that a walk can enter and "
    "never leave, so each group settles somewhere of its own and no single "
    "distribution is the place the chain settles."
)


# --------------------------------------------------------------------------
# /vowels: the two-state chain, from one sentence to two chapters.
# --------------------------------------------------------------------------


class ClassTable(BaseModel):
    """A two-state count and its table, both in the order vowel, consonant."""

    counts: list[list[int]]
    table: list[list[float]]
    n_symbols: int
    n_steps: int
    stationary: list[float]
    vowel_share: float


class HandSentence(BaseModel):
    """The sentence counted by hand, symbol by symbol."""

    sentence: str
    letters: str
    classes: str
    pairs: list[str]
    tally: ClassTable


class LikelihoodPoint(BaseModel):
    """The log likelihood of the vowel row at one candidate value."""

    probability: float
    log_likelihood: float


class AfterRow(BaseModel):
    """Where a walk is after ``n_steps``, from each of the two starts."""

    n_steps: int
    vowel_from_vowel: float
    vowel_from_consonant: float
    distance_from_vowel: float
    distance_from_consonant: float


class GapRow(BaseModel):
    """What the chain says ``gap`` steps ahead, against what three texts do."""

    gap: int
    chain_vowel_after_vowel: float
    chain_vowel_after_consonant: float
    text_vowel_after_vowel: float
    text_vowel_after_consonant: float
    held_back_vowel_after_vowel: float
    written_chain_vowel_after_vowel: float
    written_text_vowel_after_vowel: float


class ChapterRow(BaseModel):
    """One chapter counted on its own."""

    chapter: str
    n_symbols: int
    vowel_after_vowel: float
    vowel_after_consonant: float
    stationary_vowel: float


class VowelsResponse(BaseModel):
    """Everything the page shows about Markov's two-state alphabet."""

    states: list[str]
    hand: HandSentence
    counted: ClassTable
    n_sentences: int
    held_back_vowel_share: float
    walk_vowel_share: float
    walk_length: int
    likelihood: list[LikelihoodPoint]
    best_probability: float
    best_log_likelihood: float
    counted_log_probability: float
    second_eigenvalue: float
    after: list[AfterRow]
    gaps: list[GapRow]
    written_length: int
    chapters: list[ChapterRow]


def _class_table(chain: MarkovChain, sequences: Sequence[Sequence[str]]) -> ClassTable:
    counts = chain.transition_counts
    stationary = chain.stationary_distribution()
    return ClassTable(
        counts=[
            [counts.count_of(source, target) for target in CLASS_ORDER]
            for source in CLASS_ORDER
        ],
        table=[
            [chain.probability_of(source, target) for target in CLASS_ORDER]
            for source in CLASS_ORDER
        ],
        n_symbols=sum(len(sequence) for sequence in sequences),
        n_steps=counts.n_transitions,
        stationary=[stationary[state] for state in CLASS_ORDER],
        vowel_share=_shares(sequences, CLASS_ORDER)[0],
    )


def _vowel_after_vowel(chain: MarkovChain, gap: int) -> float:
    return chain.distribution_after(VOWEL, gap)[VOWEL]


@cache
def _vowels() -> VowelsResponse:
    hand_classes = _classes(HAND_SENTENCE)
    hand_chain = MarkovChain().fit([hand_classes])
    counted_sequences = _class_sequences(COUNTED)
    chain = _class_chain()
    counted = _class_table(chain, counted_sequences)

    held_back_share = _shares(_class_sequences(HELD_BACK), CLASS_ORDER)[0]
    walk = chain.sample(CONSONANT, LONG_WALK, random_seed=0)
    walk_share = walk.count(VOWEL) / len(walk)

    stays = counted.counts[0][0]
    leaves = counted.counts[0][1]
    likelihood = [
        LikelihoodPoint(
            probability=value,
            log_likelihood=stays * math.log(value) + leaves * math.log(1.0 - value),
        )
        for value in LIKELIHOOD_GRID
    ]
    best = stays / (stays + leaves)

    stationary = counted.stationary
    after = [
        AfterRow(
            n_steps=n_steps,
            vowel_from_vowel=chain.distribution_after(VOWEL, n_steps)[VOWEL],
            vowel_from_consonant=chain.distribution_after(CONSONANT, n_steps)[VOWEL],
            distance_from_vowel=_total_variation(
                _in_order(chain, CLASS_ORDER, chain.distribution_after(VOWEL, n_steps)),
                stationary,
            ),
            distance_from_consonant=_total_variation(
                _in_order(
                    chain, CLASS_ORDER, chain.distribution_after(CONSONANT, n_steps)
                ),
                stationary,
            ),
        )
        for n_steps in AFTER_STEPS
    ]

    written = [_classes("".join(_letter_chain().sample("t", LONG_WALK, random_seed=0)))]
    written_chain = MarkovChain().fit(written)
    held_back_sequences = _class_sequences(HELD_BACK)
    gaps: list[GapRow] = []
    for gap in TWO_STEP_GAPS:
        text = MarkovChain().fit(_offset_runs(counted_sequences, gap))
        held_back = MarkovChain().fit(_offset_runs(held_back_sequences, gap))
        written_text = MarkovChain().fit(_offset_runs(written, gap))
        gaps.append(
            GapRow(
                gap=gap,
                chain_vowel_after_vowel=_vowel_after_vowel(chain, gap),
                chain_vowel_after_consonant=chain.distribution_after(CONSONANT, gap)[
                    VOWEL
                ],
                text_vowel_after_vowel=text.probability_of(VOWEL, VOWEL),
                text_vowel_after_consonant=text.probability_of(CONSONANT, VOWEL),
                held_back_vowel_after_vowel=held_back.probability_of(VOWEL, VOWEL),
                written_chain_vowel_after_vowel=_vowel_after_vowel(written_chain, gap),
                written_text_vowel_after_vowel=written_text.probability_of(
                    VOWEL, VOWEL
                ),
            )
        )

    chapters: list[ChapterRow] = []
    for name, sentences in CHAPTERS:
        sequences = _class_sequences(sentences)
        part = MarkovChain().fit(sequences)
        chapters.append(
            ChapterRow(
                chapter=name,
                n_symbols=sum(len(sequence) for sequence in sequences),
                vowel_after_vowel=part.probability_of(VOWEL, VOWEL),
                vowel_after_consonant=part.probability_of(CONSONANT, VOWEL),
                stationary_vowel=part.stationary_distribution()[VOWEL],
            )
        )

    return VowelsResponse(
        states=list(CLASS_ORDER),
        hand=HandSentence(
            sentence=HAND_SENTENCE,
            letters=HAND_SENTENCE.replace(SPACE, ""),
            classes="".join(hand_classes),
            pairs=[first + second for first, second in pairwise(hand_classes)],
            tally=_class_table(hand_chain, [hand_classes]),
        ),
        counted=counted,
        n_sentences=len(COUNTED),
        held_back_vowel_share=held_back_share,
        walk_vowel_share=walk_share,
        walk_length=LONG_WALK,
        likelihood=likelihood,
        best_probability=best,
        best_log_likelihood=stays * math.log(best) + leaves * math.log(1.0 - best),
        counted_log_probability=sum(
            chain.log_probability_of(sequence) for sequence in counted_sequences
        ),
        second_eigenvalue=counted.table[0][0] - counted.table[1][0],
        after=after,
        gaps=gaps,
        written_length=LONG_WALK,
        chapters=chapters,
    )


@router.get("/vowels", response_model=VowelsResponse)
def vowels() -> VowelsResponse:
    """The vowel-and-consonant chain, counted by hand and on two chapters."""
    return _vowels()


# --------------------------------------------------------------------------
# /letters: twenty-seven states.
# --------------------------------------------------------------------------


class Follower(BaseModel):
    """One state that follows another, and how often."""

    state: str
    probability: float


class FollowerRow(BaseModel):
    """The likeliest few states after one letter."""

    state: str
    n_leaving: int
    n_followers: int
    top: list[Follower]


class ShareRow(BaseModel):
    """One letter's long-run share against its share of three texts."""

    state: str
    stationary: float
    counted: float
    held_back: float
    walk: float


class MixingRow(BaseModel):
    """How far the letter chain is from settled, from its worst start."""

    n_steps: int
    worst_distance: float
    worst_start: str


class Sample(BaseModel):
    """What one model writes."""

    label: str
    width: int
    smoothing: float
    text: str


class CostRow(BaseModel):
    """What remembering ``width`` letters costs in states and cells."""

    width: int
    n_possible_states: int
    n_states: int
    n_cells: int
    n_nonzero: int
    share_nonzero: float
    n_steps: int
    steps_per_state: float
    never_left: list[str]


class LettersResponse(BaseModel):
    """The twenty-seven state chain on the first two chapters."""

    states: list[str]
    counts: list[list[int]]
    table: list[list[float]]
    n_symbols: int
    n_steps: int
    n_cells: int
    n_nonzero: int
    followers: list[FollowerRow]
    shares: list[ShareRow]
    largest_gap_counted: float
    largest_gap_held_back: float
    largest_gap_walk: float
    walk_length: int
    mixing: list[MixingRow]
    samples: list[Sample]
    costs: list[CostRow]
    stranded_sentences: list[str]


FOLLOWED_LETTERS: tuple[str, ...] = ("q", "h", SPACE, "e")
"""The letters whose rows the page quotes."""


def _written(chain: MarkovChain, start: str, width: int, seed: int) -> str:
    """A walk of the chain, turned back into letters."""
    walk = chain.sample(start, SAMPLE_LENGTH - width + 1, random_seed=seed)
    return walk[0] + "".join(state[-1] for state in walk[1:])


@cache
def _letters_view() -> LettersResponse:
    chain = _letter_chain()
    counts = chain.transition_counts
    table = chain.transitions
    stationary = chain.stationary_distribution()

    followers: list[FollowerRow] = []
    for state in FOLLOWED_LETTERS:
        row = table[state]
        ranked = sorted(
            ((row[target], target) for target in LETTER_ORDER if row[target] > 0.0),
            reverse=True,
        )
        followers.append(
            FollowerRow(
                state=state,
                n_leaving=counts.leaving(state),
                n_followers=len(ranked),
                top=[
                    Follower(state=target, probability=probability)
                    for probability, target in ranked[:5]
                ],
            )
        )

    counted_shares = _shares(_letter_sequences(COUNTED), LETTER_ORDER)
    held_back_shares = _shares(_letter_sequences(HELD_BACK), LETTER_ORDER)
    walk = chain.sample("t", LONG_WALK, random_seed=0)
    walk_shares = _shares([walk], LETTER_ORDER)
    shares = [
        ShareRow(
            state=state,
            stationary=stationary[state],
            counted=counted_shares[position],
            held_back=held_back_shares[position],
            walk=walk_shares[position],
        )
        for position, state in enumerate(LETTER_ORDER)
    ]

    settled = np.asarray(stationary)
    mixing: list[MixingRow] = []
    for n_steps in MIXING_STEPS:
        distances = {
            state: _total_variation(
                np.asarray(chain.distribution_after(state, n_steps)), settled
            )
            for state in chain.states
        }
        worst = max(distances, key=lambda state: distances[state])
        mixing.append(
            MixingRow(
                n_steps=n_steps, worst_distance=distances[worst], worst_start=worst
            )
        )

    generator = np.random.default_rng(SAMPLE_SEED)
    memoryless = "".join(
        str(letter)
        for letter in generator.choice(
            np.asarray(chain.states), size=SAMPLE_LENGTH, p=settled
        )
    )
    samples = [
        Sample(label="no memory", width=0, smoothing=0.0, text=memoryless),
        Sample(
            label="one letter of memory",
            width=1,
            smoothing=0.0,
            text=_written(chain, "t", 1, SAMPLE_SEED),
        ),
        Sample(
            label="two letters of memory",
            width=2,
            smoothing=0.0,
            text=_written(_letter_chain(2), "th", 2, SAMPLE_SEED),
        ),
        Sample(
            label="three letters of memory",
            width=3,
            smoothing=TRIPLE_SMOOTHING,
            text=_written(_letter_chain(3, TRIPLE_SMOOTHING), "the", 3, SAMPLE_SEED),
        ),
        Sample(
            label="two letters of memory, smoothed by a half",
            width=2,
            smoothing=LETTER_SMOOTHING,
            text=_written(_letter_chain(2, LETTER_SMOOTHING), "th", 2, SAMPLE_SEED),
        ),
    ]

    costs: list[CostRow] = []
    for width in COST_WIDTHS:
        wide = _letter_chain(width, 0.0 if width < 3 else TRIPLE_SMOOTHING)
        wide_counts = np.asarray(wide.transition_counts)
        n_nonzero = int(np.count_nonzero(wide_counts))
        costs.append(
            CostRow(
                width=width,
                n_possible_states=len(LETTER_ORDER) ** width,
                n_states=wide.n_states,
                n_cells=int(wide_counts.size),
                n_nonzero=n_nonzero,
                share_nonzero=n_nonzero / wide_counts.size,
                n_steps=wide.transition_counts.n_transitions,
                steps_per_state=wide.transition_counts.n_transitions / wide.n_states,
                never_left=list(wide.transition_counts.never_left),
            )
        )

    stranded = costs[-1].never_left
    stranded_sentences = [
        sentence
        for sentence in COUNTED
        if any(sentence.endswith(state) for state in stranded)
    ]

    return LettersResponse(
        states=list(LETTER_ORDER),
        counts=[
            [counts.count_of(source, target) for target in LETTER_ORDER]
            for source in LETTER_ORDER
        ],
        table=[
            [table.probability_of(source, target) for target in LETTER_ORDER]
            for source in LETTER_ORDER
        ],
        n_symbols=sum(len(sentence) for sentence in COUNTED),
        n_steps=counts.n_transitions,
        n_cells=len(LETTER_ORDER) ** 2,
        n_nonzero=int(np.count_nonzero(np.asarray(counts))),
        followers=followers,
        shares=shares,
        largest_gap_counted=max(abs(row.stationary - row.counted) for row in shares),
        largest_gap_held_back=max(
            abs(row.stationary - row.held_back) for row in shares
        ),
        largest_gap_walk=max(abs(row.stationary - row.walk) for row in shares),
        walk_length=LONG_WALK,
        mixing=mixing,
        samples=samples,
        costs=costs,
        stranded_sentences=stranded_sentences,
    )


@router.get("/letters", response_model=LettersResponse)
def letters() -> LettersResponse:
    """The letter chain: its table, its shares, how fast it forgets, what it writes."""
    return _letters_view()


# --------------------------------------------------------------------------
# /memory: the held-back chapter, scored on one set of positions.
# --------------------------------------------------------------------------


class ScoreRow(BaseModel):
    """One model's score on the held-back chapter.

    ``bits`` is the mean number of bits per predicted symbol, and is absent when
    the model gave some scored step no probability at all, which makes the
    whole text impossible; ``n_impossible`` counts those steps.
    """

    width: int
    smoothing: float
    bits: float | None
    n_impossible: int


class AheadRow(BaseModel):
    """The one-letter chain predicting ``n_steps`` letters ahead of what it read."""

    n_steps: int
    bits: float


class MemoryResponse(BaseModel):
    """What memory buys on text the chains never counted."""

    letter_positions: int
    letter_positions_set_aside: int
    letter_rows: list[ScoreRow]
    ahead_positions: int
    ahead: list[AheadRow]
    frequencies_bits: float
    ahead_smoothing: float
    class_positions: int
    class_positions_set_aside: int
    class_rows: list[ScoreRow]


def _scored_positions(
    sequences: Sequence[Sequence[str]],
    first_position: int,
    widest: MarkovChain,
    width: int,
) -> list[tuple[int, int]]:
    """Every (sequence, position) the widest model can score, and so every model can.

    A position is set aside when the run of symbols before it, or the run ending
    on it, is a state the widest chain never counted, since that chain then has
    no row to read. Every narrower chain met every state the widest one did.
    """
    known = set(widest.states)
    positions: list[tuple[int, int]] = []
    for index, sequence in enumerate(sequences):
        for position in range(first_position, len(sequence)):
            source = "".join(sequence[position - width : position])
            target = "".join(sequence[position - width + 1 : position + 1])
            if source in known and target in known:
                positions.append((index, position))
    return positions


def _score(
    chain: MarkovChain,
    width: int,
    sequences: Sequence[Sequence[str]],
    positions: Sequence[tuple[int, int]],
) -> tuple[float | None, int]:
    """Mean bits per position under a chain remembering ``width`` symbols.

    Width zero reads each symbol's probability off the chain's stationary
    distribution, which is the model that knows how common each symbol is and
    nothing about what came before it.
    """
    table = chain.transitions
    settled = chain.stationary_distribution() if width == 0 else None
    total = 0.0
    impossible = 0
    for index, position in positions:
        sequence = sequences[index]
        if settled is not None:
            probability = settled[sequence[position]]
        else:
            probability = table.probability_of(
                "".join(sequence[position - width : position]),
                "".join(sequence[position - width + 1 : position + 1]),
            )
        if probability == 0.0:
            impossible += 1
        else:
            total += math.log2(probability)
    if impossible:
        return None, impossible
    return -total / len(positions), 0


def _all_positions(sequences: Sequence[Sequence[str]], first_position: int) -> int:
    return sum(max(0, len(sequence) - first_position) for sequence in sequences)


@cache
def _memory() -> MemoryResponse:
    letter_sequences = _letter_sequences(HELD_BACK)
    letter_positions = _scored_positions(
        letter_sequences, MEMORY_FIRST_POSITION, _letter_chain(2), 2
    )
    letter_rows = [
        ScoreRow(
            width=width,
            smoothing=smoothing,
            bits=scored[0],
            n_impossible=scored[1],
        )
        for smoothing in SMOOTHING_SWEEP
        for width in LETTER_WIDTHS
        for scored in [
            _score(
                _letter_chain(max(width, 1), smoothing),
                width,
                letter_sequences,
                letter_positions,
            )
        ]
    ]

    ahead_chain = _letter_chain(1, LETTER_SMOOTHING)
    ahead_positions = [
        (index, position)
        for index, sequence in enumerate(letter_sequences)
        for position in range(AHEAD_FIRST_POSITION, len(sequence))
    ]
    ahead: list[AheadRow] = []
    for n_steps in AHEAD_STEPS:
        predictions = {
            state: ahead_chain.distribution_after(state, n_steps)
            for state in ahead_chain.states
        }
        total = sum(
            math.log2(
                predictions[letter_sequences[index][position - n_steps]][
                    letter_sequences[index][position]
                ]
            )
            for index, position in ahead_positions
        )
        ahead.append(AheadRow(n_steps=n_steps, bits=-total / len(ahead_positions)))
    frequencies_bits, _ = _score(ahead_chain, 0, letter_sequences, ahead_positions)

    class_sequences = _class_sequences(HELD_BACK)
    widest = max(CLASS_WIDTHS)
    class_positions = _scored_positions(
        class_sequences, CLASS_FIRST_POSITION, _class_chain(widest), widest
    )
    class_rows = [
        ScoreRow(
            width=width,
            smoothing=smoothing,
            bits=scored[0],
            n_impossible=scored[1],
        )
        for smoothing in CLASS_SMOOTHINGS
        for width in CLASS_WIDTHS
        for scored in [
            _score(
                _class_chain(max(width, 1), smoothing),
                width,
                class_sequences,
                class_positions,
            )
        ]
    ]

    return MemoryResponse(
        letter_positions=len(letter_positions),
        letter_positions_set_aside=_all_positions(
            letter_sequences, MEMORY_FIRST_POSITION
        )
        - len(letter_positions),
        letter_rows=letter_rows,
        ahead_positions=len(ahead_positions),
        ahead=ahead,
        frequencies_bits=frequencies_bits if frequencies_bits is not None else math.inf,
        ahead_smoothing=LETTER_SMOOTHING,
        class_positions=len(class_positions),
        class_positions_set_aside=_all_positions(class_sequences, CLASS_FIRST_POSITION)
        - len(class_positions),
        class_rows=class_rows,
    )


@router.get("/memory", response_model=MemoryResponse)
def memory() -> MemoryResponse:
    """The held-back chapter scored under every length of memory and smoothing."""
    return _memory()


# --------------------------------------------------------------------------
# /edges: small chains where settling is not the ordinary case.
# --------------------------------------------------------------------------


class EdgeKind(StrEnum):
    """What the stationary distribution does on one small chain."""

    SETTLES = "settles"
    NEVER_SETTLES = "never_settles"
    SEVERAL_GROUPS = "several_groups"
    NEVER_LEFT = "never_left"


class EdgeCase(BaseModel):
    """One small chain, counted on a word or two, and what its long run does."""

    title: str
    words: list[str]
    classes: list[str]
    kind: EdgeKind
    counts: list[list[int]]
    table: list[list[float]] | None
    stationary: list[float] | None
    note: str
    start: str
    walk: list[float] | None
    long_run_share: list[float] | None
    smoothed_table: list[list[float]]
    smoothed_stationary: list[float]


class EdgesResponse(BaseModel):
    """The four small chains, and the two states the real text strands."""

    states: list[str]
    smoothing: float
    cases: list[EdgeCase]
    stranded_width: int
    stranded_states: list[str]
    stranded_sentences: list[str]


EDGE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A word that alternates", ("banana",)),
    ("Two words that never meet", ("rhythm", "aeiou")),
    ("A vowel left once and never again", ("asks",)),
    ("A vowel never left", ("spa",)),
)
"""The four small chains, each named for what it does rather than what it is."""


def _edge(title: str, words: Sequence[str]) -> EdgeCase:
    sequences = [_classes(word) for word in words]
    smoothed = MarkovChain(smoothing=EDGE_SMOOTHING).fit(sequences)
    counts = smoothed.transition_counts
    counted = [
        [counts.count_of(source, target) for target in CLASS_ORDER]
        for source in CLASS_ORDER
    ]
    start = sequences[0][0]

    table: list[list[float]] | None = None
    stationary: list[float] | None = None
    walk: list[float] | None = None
    long_run: list[float] | None = None
    try:
        chain = MarkovChain().fit(sequences)
    except TooFewValuesError:
        kind = EdgeKind.NEVER_LEFT
        note = _never_left_note(counts.never_left)
    else:
        table = [
            [chain.probability_of(source, target) for target in CLASS_ORDER]
            for source in CLASS_ORDER
        ]
        walk = [chain.distribution_after(start, n)[VOWEL] for n in EDGE_STEPS]
        long_run = [
            float(value)
            for value in np.mean(
                [
                    _in_order(chain, CLASS_ORDER, chain.distribution_after(start, n))
                    for n in range(LONG_RUN_STEPS)
                ],
                axis=0,
            )
        ]
        try:
            settled = chain.stationary_distribution()
        except NonUniqueStationaryDistributionError:
            kind = EdgeKind.SEVERAL_GROUPS
            note = SEVERAL_GROUPS_NOTE
        else:
            stationary = [settled[state] for state in CLASS_ORDER]
            far = _in_order(
                chain, CLASS_ORDER, chain.distribution_after(start, SETTLE_CHECK)
            )
            further = _in_order(
                chain, CLASS_ORDER, chain.distribution_after(start, SETTLE_CHECK + 1)
            )
            if _total_variation(far, further) > SETTLE_TOLERANCE:
                kind = EdgeKind.NEVER_SETTLES
                note = (
                    "One distribution is left unchanged by a step, and the walk "
                    "never arrives at it, since it changes sides at every step."
                )
            else:
                kind = EdgeKind.SETTLES
                note = (
                    "One distribution is left unchanged by a step and every walk "
                    "arrives at it, though a state the walk leaves for good holds "
                    "no share of it."
                )

    settled_smoothed = smoothed.stationary_distribution()
    return EdgeCase(
        title=title,
        words=list(words),
        classes=["".join(sequence) for sequence in sequences],
        kind=kind,
        counts=counted,
        table=table,
        stationary=stationary,
        note=note,
        start=start,
        walk=walk,
        long_run_share=long_run,
        smoothed_table=[
            [smoothed.probability_of(source, target) for target in CLASS_ORDER]
            for source in CLASS_ORDER
        ],
        smoothed_stationary=[settled_smoothed[state] for state in CLASS_ORDER],
    )


@cache
def _edges() -> EdgesResponse:
    stranded = _letter_chain(3, TRIPLE_SMOOTHING).transition_counts.never_left
    return EdgesResponse(
        states=list(CLASS_ORDER),
        smoothing=EDGE_SMOOTHING,
        cases=[_edge(title, words) for title, words in EDGE_WORDS],
        stranded_width=3,
        stranded_states=list(stranded),
        stranded_sentences=[
            sentence
            for sentence in COUNTED
            if any(sentence.endswith(state) for state in stranded)
        ],
    )


@router.get("/edges", response_model=EdgesResponse)
def edges() -> EdgesResponse:
    """Four small chains on which the long run is not the ordinary case."""
    return _edges()


# --------------------------------------------------------------------------
# /fit: the playground.
# --------------------------------------------------------------------------


class Alphabet(StrEnum):
    """Which states the letters are counted as."""

    VOWELS_AND_CONSONANTS = "vowels_and_consonants"
    LETTERS = "letters"


class Source(StrEnum):
    """Where the text being counted comes from."""

    TYPED = "typed"
    COUNTED_CHAPTERS = "counted_chapters"
    ALICE_RUN = "alice_run"
    WRITTEN_BY_THE_CHAIN = "written_by_the_chain"


class FitRequest(BaseModel):
    """A text, an alphabet to count it in, a smoothing and a number of steps."""

    source: Source = Source.COUNTED_CHAPTERS
    text: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    alice_start: int = Field(default=0, ge=0, le=len(SENTENCES) - RUN_LENGTH)
    alphabet: Alphabet = Alphabet.VOWELS_AND_CONSONANTS
    smoothing: float = Field(default=0.0, ge=0.0, le=MAX_SMOOTHING)
    n_steps: int = Field(default=2, ge=0, le=MAX_STEPS)
    random_seed: int = Field(default=0, ge=0, le=9999)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """Everything the playground shows for one text in one alphabet."""

    source: Source
    alphabet: Alphabet
    smoothing: float
    shown: str
    n_sequences: int
    n_symbols: int
    n_steps_counted: int
    states: list[str]
    counts: list[list[int]]
    table: list[list[float]] | None
    table_note: str | None
    never_left: list[str]
    text_shares: list[float]
    stationary: list[float] | None
    stationary_note: str | None
    settles: bool | None
    n_steps: int
    after: list[list[float]] | None
    distances: list[float] | None
    chain_two_step: list[list[float]] | None
    text_two_step: list[list[float]] | None
    two_step_gap: float | None
    walk: str | None


SENTENCE_BREAK = re.compile(r"[.!?\n]+")
NOT_A_LETTER = re.compile(r"[^a-z]+")


def _typed_sentences(text: str) -> list[str]:
    """A typed text lowercased, cut at sentence ends, reduced to letters and spaces."""
    return [
        cleaned
        for part in SENTENCE_BREAK.split(text.lower())
        if (cleaned := NOT_A_LETTER.sub(" ", part).strip())
    ]


def _sequences_for(
    request: FitRequest,
) -> tuple[list[list[str]], str]:
    """The sequences to count, and the text to show beside them."""
    as_sequences = (
        _class_sequences
        if request.alphabet is Alphabet.VOWELS_AND_CONSONANTS
        else _letter_sequences
    )
    if request.source is Source.WRITTEN_BY_THE_CHAIN:
        if request.alphabet is Alphabet.VOWELS_AND_CONSONANTS:
            walk = _class_chain().sample(
                CONSONANT, WRITTEN_LENGTH, random_seed=request.random_seed
            )
        else:
            walk = _letter_chain().sample(
                "t", WRITTEN_LENGTH, random_seed=request.random_seed
            )
        return [list(walk)], "".join(walk)
    if request.source is Source.COUNTED_CHAPTERS:
        sentences: Sequence[str] = COUNTED
    elif request.source is Source.ALICE_RUN:
        sentences = SENTENCES[request.alice_start : request.alice_start + RUN_LENGTH]
    else:
        sentences = _typed_sentences(request.text)
    return [sequence for sequence in as_sequences(sentences) if sequence], ". ".join(
        sentences
    )


def _ordered_states(
    sequences: Sequence[Sequence[str]], alphabet: Alphabet
) -> list[str]:
    present = {symbol for sequence in sequences for symbol in sequence}
    order = CLASS_ORDER if alphabet is Alphabet.VOWELS_AND_CONSONANTS else LETTER_ORDER
    return [state for state in order if state in present]


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Count one text in one alphabet, and ask the chain everything the page asks."""
    sequences, shown = _sequences_for(request)
    if not sequences:
        raise HTTPException(
            status_code=422, detail="There are no letters here to count."
        )
    states = _ordered_states(sequences, request.alphabet)

    try:
        counted = MarkovChain(smoothing=max(request.smoothing, 1.0)).fit(sequences)
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal
    counts = counted.transition_counts
    count_table = [
        [counts.count_of(source, target) for target in states] for source in states
    ]

    table: list[list[float]] | None = None
    table_note: str | None = None
    stationary: list[float] | None = None
    stationary_note: str | None = None
    settles: bool | None = None
    after: list[list[float]] | None = None
    distances: list[float] | None = None
    chain_two_step: list[list[float]] | None = None
    text_two_step: list[list[float]] | None = None
    two_step_gap: float | None = None
    walk: str | None = None

    try:
        chain = MarkovChain(smoothing=request.smoothing).fit(sequences)
    except TooFewValuesError:
        table_note = _never_left_note(counts.never_left)
    else:
        table = [
            [chain.probability_of(source, target) for target in states]
            for source in states
        ]
        after = [
            _in_order(chain, states, chain.distribution_after(state, request.n_steps))
            for state in states
        ]
        chain_two_step = [
            _in_order(chain, states, chain.distribution_after(state, 2))
            for state in states
        ]
        drawn = chain.sample(
            sequences[0][0], PLAYGROUND_WALK, random_seed=request.random_seed
        )
        walk = "".join(drawn)
        try:
            settled = chain.stationary_distribution()
        except NonUniqueStationaryDistributionError:
            stationary_note = SEVERAL_GROUPS_NOTE
        else:
            stationary = [settled[state] for state in states]
            settles = all(
                _total_variation(
                    _in_order(
                        chain, states, chain.distribution_after(state, SETTLE_CHECK)
                    ),
                    _in_order(
                        chain,
                        states,
                        chain.distribution_after(state, SETTLE_CHECK + 1),
                    ),
                )
                <= SETTLE_TOLERANCE
                for state in states
            )
            if not settles:
                stationary_note = (
                    "One distribution is left unchanged by a step, and no walk "
                    "arrives at it, since the walk cycles in lockstep; it is the "
                    "share of time a long walk spends in each state."
                )
            distances = [
                max(
                    _total_variation(
                        _in_order(
                            chain, states, chain.distribution_after(state, n_steps)
                        ),
                        stationary,
                    )
                    for state in states
                )
                for n_steps in MIXING_HORIZON
            ]
        try:
            text_chain = MarkovChain(smoothing=request.smoothing).fit(
                _offset_runs(sequences, 2)
            )
        except TooFewValuesError:
            text_chain = None
        if text_chain is not None and set(text_chain.states) == set(states):
            text_two_step = [
                [text_chain.probability_of(source, target) for target in states]
                for source in states
            ]
            shares = _shares(sequences, states)
            two_step_gap = sum(
                share * _total_variation(chain_row, text_row)
                for share, chain_row, text_row in zip(
                    shares, chain_two_step, text_two_step, strict=True
                )
            )

    return FitResponse(
        source=request.source,
        alphabet=request.alphabet,
        smoothing=request.smoothing,
        shown=shown[:MAX_TEXT_LENGTH],
        n_sequences=len(sequences),
        n_symbols=sum(len(sequence) for sequence in sequences),
        n_steps_counted=counts.n_transitions,
        states=states,
        counts=count_table,
        table=table,
        table_note=table_note,
        never_left=list(counts.never_left),
        text_shares=_shares(sequences, states),
        stationary=stationary,
        stationary_note=stationary_note,
        settles=settles,
        n_steps=request.n_steps,
        after=after,
        distances=distances,
        chain_two_step=chain_two_step,
        text_two_step=text_two_step,
        two_step_gap=two_step_gap,
        walk=walk,
    )
