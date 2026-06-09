"""GloVe.

Every endpoint here is registered under ``/concepts/glove``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about GloVe, CooccurrenceMatrix.

The running corpus is the one the library's own GloVe spec fits: twelve cooking
sentences and twelve astronomy sentences whose content words never mix, sharing
only seven function words. It is reproduced here rather than imported, because
``test`` is not part of the installed package, and a test asserts the two agree
sentence for sentence.

One thing is computed here rather than in the library, and it is the reason the
page has anything to say about the step rule. The library's fit accumulates each
parameter's squared slopes from zero and divides after adding, which is the
textbook rule; the released C program starts every accumulator at one and
divides before adding. Both are written out in ``_walk`` and differ by two
lines, so the two curves the page draws come from one loop. Every piece of
arithmetic inside it -- the co-occurrence counts, the weighting, one pair's term
and its four slopes, and the objective at rest -- is the library's, and
``test_glove.py`` asserts that the accumulating branch reproduces the library's
own fit to the last bit, which is what makes the comparison a comparison of step
rules and not of implementations.

Nothing a widget renders may name a class, a parameter or a module, so the two
weightings and the two step rules cross the wire as the page's own words,
``harmonic`` and ``uniform``, ``accumulating`` and ``released``, and
``_WEIGHTINGS`` is the only place a weighting meets the library's own name.
"""

from __future__ import annotations

import math
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.embeddings.prediction.glove import (
    ADAGRAD_EPSILON,
    GloVe,
    GloVeParameters,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/glove", tags=["glove"])


# --- The corpus ------------------------------------------------------------

COOKING: tuple[str, ...] = (
    "the chef chops onion and garlic",
    "the chef stirs soup with butter and salt",
    "the chef stirs onion and garlic in the pan",
    "the chef bakes bread with flour in the oven",
    "the chef stirs salt and pepper into the broth",
    "the chef bakes bread and stirs the broth",
    "flour and butter and salt in the pan",
    "the chef chops onion for the soup and the broth",
    "garlic and pepper in the soup with bread",
    "the chef bakes flour and butter into bread in the oven",
    "onion and garlic and salt in the broth",
    "the chef stirs the soup in the pan with pepper",
)
"""Twelve sentences about cooking."""

ASTRONOMY: tuple[str, ...] = (
    "the astronomer watches the star and the planet",
    "the astronomer charts the orbit of the moon",
    "the comet and the meteor orbit the star",
    "the telescope watches the galaxy and the nebula",
    "the moon orbits the planet with gravity",
    "the astronomer watches the eclipse of the moon with the telescope",
    "the comet orbits the star and the astronomer charts the orbit",
    "the telescope watches the nebula and the galaxy",
    "the astronomer charts the meteor and the comet with the telescope",
    "the planet orbits the star and the moon orbits the planet",
    "the eclipse of the star and the eclipse of the moon",
    "gravity and the orbit of the galaxy and the nebula",
)
"""Twelve sentences about astronomy, sharing no content word with the others."""

FUNCTION_WORDS: frozenset[str] = frozenset(
    {"the", "and", "with", "in", "of", "into", "for"}
)
"""The seven words both halves use, which are the only overlap there is."""

TWO_TOPICS: list[str] = list(COOKING) + list(ASTRONOMY)
"""The twenty-four sentences every fixed measurement on this page is made on."""

ONE_SENTENCE: tuple[str, ...] = ("the chef stirs the soup",)
"""Five words, whose whole table can be checked with a pencil."""

TWO_SENTENCES: tuple[str, ...] = (
    "the chef stirs the soup",
    "the chef stirs the broth",
)
"""Ten words, where a repeated pair first pushes a count above one."""

TOPIC_OF: dict[str, str] = {
    **{word: "cooking" for sentence in COOKING for word in sentence.split()},
    **{word: "astronomy" for sentence in ASTRONOMY for word in sentence.split()},
    **{word: "shared" for word in FUNCTION_WORDS},
}
"""Which half each word belongs to, for the widgets that colour by it."""

COOKING_WORDS: tuple[str, ...] = tuple(
    sorted({word for sentence in COOKING for word in sentence.split()} - FUNCTION_WORDS)
)
ASTRONOMY_WORDS: tuple[str, ...] = tuple(
    sorted(
        {word for sentence in ASTRONOMY for word in sentence.split()} - FUNCTION_WORDS
    )
)


# --- Translating the page's words into the library's -------------------------

WeightingName = Literal["harmonic", "uniform"]
StepRuleName = Literal["accumulating", "released"]
TextChoice = Literal["one-sentence", "two-sentences", "corpus"]

_WEIGHTINGS: dict[WeightingName, ContextWeighting] = {
    "harmonic": ContextWeighting.HARMONIC,
    "uniform": ContextWeighting.UNIFORM,
}

_TEXTS: dict[TextChoice, list[str]] = {
    "one-sentence": list(ONE_SENTENCE),
    "two-sentences": list(TWO_SENTENCES),
    "corpus": TWO_TOPICS,
}


# --- What a request may ask for ----------------------------------------------

PAGE_DIMENSION = 8
"""How many numbers each vector holds in every fixed measurement here."""

PAGE_WINDOW = 2
"""How far the window reaches in every fixed measurement here.

Two rather than the published five, and the reason is measured on the
``/windows`` endpoint rather than chosen.
"""

PAGE_EPOCHS = 25
"""How many passes over the table every fixed measurement here makes."""

PAGE_LEARNING_RATE = 0.05
"""The published base step size, which every fixed measurement here uses."""

PAGE_SEED = 0
"""The one start every fixed measurement here begins from."""

PUBLISHED_WINDOW = 5
PUBLISHED_DIMENSION = 50
PUBLISHED_MAXIMUM_COUNT = 100.0
PUBLISHED_EXPONENT = 0.75
"""The four settings the paper ships with."""

MAX_DIMENSION = 24
"""The widest vector an interactive fit may learn.

Width costs little at this vocabulary, measured, but the bound keeps a crafted
request from turning a teaching endpoint into a compute job.
"""

MAX_EPOCHS = 40
"""The most passes one interactive fit may make.

A pass is a Python loop over every pair that ever co-occurred, so forty at the
widest allowed vector is already longer than a reader waits for.
"""


# --- Shared pieces -----------------------------------------------------------


@cache
def _tokenised(choice: TextChoice) -> TokenisedCorpus:
    """One of the three texts, split into words, once."""
    return TokenisedCorpus.from_texts(_TEXTS[choice], WhitespacePreTokenizer())


@cache
def _vocabulary(choice: TextChoice) -> Vocabulary:
    """The words of one text, commonest first."""
    return _tokenised(choice).vocabulary(1)


@cache
def _matrix(
    choice: TextChoice, window: int, weighting: WeightingName
) -> CooccurrenceMatrix:
    """How often each word of one text appeared near each other word."""
    tokenised = _tokenised(choice)
    vocabulary = _vocabulary(choice)
    return CooccurrenceMatrix.from_id_sequences(
        vocabulary,
        tokenised.id_sequences(vocabulary),
        window,
        _WEIGHTINGS[weighting],
    )


def _within_and_across(embeddings: WordEmbeddings) -> tuple[float, float]:
    """Mean cosine inside a half, and mean cosine between the two halves.

    Only content words count, since the seven shared words belong to neither
    half and comparing them to anything says nothing about whether the halves
    came apart.
    """
    inside: list[float] = []
    for words in (COOKING_WORDS, ASTRONOMY_WORDS):
        for position, first in enumerate(words):
            for second in words[position + 1 :]:
                inside.append(embeddings.similarity(first, second))
    between = [
        embeddings.similarity(first, second)
        for first in COOKING_WORDS
        for second in ASTRONOMY_WORDS
    ]
    return float(np.mean(inside)), float(np.mean(between))


def _same_topic_share(embeddings: WordEmbeddings, n_results: int = 5) -> float:
    """Of the nearest content words to each content word, how many share a half.

    The shared words are skipped on both sides, so this is the question the
    corpus was built to ask and nothing else.
    """
    right = 0
    total = 0
    for word in COOKING_WORDS + ASTRONOMY_WORDS:
        near = [
            similar.word
            for similar in embeddings.most_similar(word, embeddings.n_words - 1)
            if TOPIC_OF[similar.word] != "shared"
        ][:n_results]
        right += sum(1 for other in near if TOPIC_OF[other] == TOPIC_OF[word])
        total += len(near)
    return right / total


class Neighbour(BaseModel):
    """One nearby word and how near it is."""

    word: str
    similarity: float
    topic: str


def _neighbours(
    embeddings: WordEmbeddings, word: str, n_results: int
) -> list[Neighbour]:
    return [
        Neighbour(
            word=similar.word,
            similarity=float(similar.similarity),
            topic=TOPIC_OF.get(similar.word, "shared"),
        )
        for similar in embeddings.most_similar(word, n_results)
    ]


# --- The two step rules, in one loop -----------------------------------------


class Walk:
    """What one run of the fitting loop produced, whichever rule it stepped by.

    Held together rather than returned as a tuple, because the objective at rest
    means nothing without the parameters it was computed from and the curve
    means nothing without both.
    """

    __slots__ = ("_divisors", "_embeddings", "_objective", "_parameters", "_totals")

    def __init__(
        self,
        totals: list[float],
        objective: float,
        parameters: GloVeParameters,
        embeddings: WordEmbeddings,
        divisors: tuple[float, float],
    ) -> None:
        self._totals = totals
        self._objective = objective
        self._parameters = parameters
        self._embeddings = embeddings
        self._divisors = divisors

    @property
    def totals(self) -> list[float]:
        """The summed pair terms of each pass, as they were computed."""
        return self._totals

    @property
    def objective(self) -> float:
        """The objective recomputed where the walk stopped."""
        return self._objective

    @property
    def parameters(self) -> GloVeParameters:
        """The four tables the walk ended on."""
        return self._parameters

    @property
    def embeddings(self) -> WordEmbeddings:
        """The word table and the context table added, bound to the words."""
        return self._embeddings

    @property
    def divisors(self) -> tuple[float, float]:
        """The smallest and largest amount a slope was divided by at the end."""
        return self._divisors


def _walk(
    model: GloVe,
    cooccurrence: CooccurrenceMatrix,
    step_rule: StepRuleName,
) -> Walk:
    """Visit every pair that co-occurred, once per pass, stepping by one rule.

    ``accumulating`` adds each slope's square to the running sum and then
    divides by the root of the sum, so a parameter that has moved a lot moves
    less next time; ``released`` starts every sum at one and adds the square
    *after* stepping, which is what the released program does. The two branches
    differ by which side of the step the three lines sit on, and everything
    between them is the same arithmetic.
    """
    counts = np.asarray(cooccurrence)
    rows, columns = np.nonzero(counts)
    n_pairs = int(rows.shape[0])
    n_words = cooccurrence.n_words
    dimension = model.dimension
    generator = np.random.default_rng(model.random_seed)

    word_vectors = (generator.random((n_words, dimension)) - 0.5) / dimension
    context_vectors = (generator.random((n_words, dimension)) - 0.5) / dimension
    biases = (generator.random((n_words,)) - 0.5) / dimension
    context_biases = (generator.random((n_words,)) - 0.5) / dimension

    start = 1.0 if step_rule == "released" else 0.0
    word_sums = np.full((n_words, dimension), start)
    context_sums = np.full((n_words, dimension), start)
    bias_sums = np.full(n_words, start)
    context_bias_sums = np.full(n_words, start)

    totals: list[float] = []
    for _ in range(model.epochs):
        total = 0.0
        for position in generator.permutation(n_pairs):
            word_id = int(rows[position])
            context_id = int(columns[position])
            gradient = model.pair_gradient(
                float(counts[word_id, context_id]),
                word_vectors[word_id],
                context_vectors[context_id],
                float(biases[word_id]),
                float(context_biases[context_id]),
            )
            total += gradient.loss

            if step_rule == "accumulating":
                word_sums[word_id] += gradient.word_gradient**2
                context_sums[context_id] += gradient.context_gradient**2
                bias_sums[word_id] += gradient.bias_gradient**2
                context_bias_sums[context_id] += gradient.context_bias_gradient**2

            rate = model.learning_rate
            word_vectors[word_id] -= (
                rate
                * gradient.word_gradient
                / (np.sqrt(word_sums[word_id]) + ADAGRAD_EPSILON)
            )
            context_vectors[context_id] -= (
                rate
                * gradient.context_gradient
                / (np.sqrt(context_sums[context_id]) + ADAGRAD_EPSILON)
            )
            biases[word_id] -= (
                rate
                * gradient.bias_gradient
                / (np.sqrt(bias_sums[word_id]) + ADAGRAD_EPSILON)
            )
            context_biases[context_id] -= (
                rate
                * gradient.context_bias_gradient
                / (np.sqrt(context_bias_sums[context_id]) + ADAGRAD_EPSILON)
            )

            if step_rule == "released":
                word_sums[word_id] += gradient.word_gradient**2
                context_sums[context_id] += gradient.context_gradient**2
                bias_sums[word_id] += gradient.bias_gradient**2
                context_bias_sums[context_id] += gradient.context_bias_gradient**2

        totals.append(total)

    parameters = GloVeParameters(word_vectors, context_vectors, biases, context_biases)
    divisors = np.sqrt(word_sums) + ADAGRAD_EPSILON
    return Walk(
        totals=totals,
        objective=model.objective_of(cooccurrence, parameters),
        parameters=parameters,
        embeddings=WordEmbeddings(cooccurrence.vocabulary, parameters.combined_vectors),
        divisors=(float(divisors.min()), float(divisors.max())),
    )


# --- One fit, on request -----------------------------------------------------


class PassReport(BaseModel):
    """What one pass over the pairs cost.

    ``total`` is the summed pair terms as they were computed during the pass,
    with the parameters moving between one pair and the next, so it is not the
    objective at any one set of parameters.
    """

    pass_number: int
    total: float
    mean_per_pair: float


class FitRequest(BaseModel):
    """One fit of the running corpus, and the word to look around afterwards."""

    window: int = Field(default=PAGE_WINDOW, ge=1, le=5)
    dimension: int = Field(default=PAGE_DIMENSION, ge=2, le=MAX_DIMENSION)
    epochs: int = Field(default=PAGE_EPOCHS, ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(default=PAGE_LEARNING_RATE, gt=0.0, le=1.0)
    maximum_count: float = Field(default=PUBLISHED_MAXIMUM_COUNT, gt=0.0, le=1e6)
    weighting_exponent: float = Field(default=PUBLISHED_EXPONENT, gt=0.0, le=1.0)
    weighting: WeightingName = "harmonic"
    step_rule: StepRuleName = "accumulating"
    random_seed: int = Field(default=PAGE_SEED, ge=0, le=999)
    word: str = Field(default="moon", min_length=1, max_length=32)
    n_neighbours: int = Field(default=8, ge=1, le=12)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """A fit of the table, the objective falling, and where one word ended up.

    ``objective_at_rest`` is the quantity the fit minimises, recomputed where
    the walk stopped, and it is the number to read; the per-pass totals were
    accumulated while the parameters moved and are only an indication.
    """

    n_sentences: int
    n_occurrences: int
    n_words: int
    n_pairs: int
    n_cells: int
    zero_share: float
    passes: list[PassReport]
    objective_at_rest: float
    within_topic: float
    across_topic: float
    gap: float
    neighbours: list[Neighbour]
    coordinates: list[float]
    numbers_kept: int
    smallest_divisor: float
    largest_divisor: float
    seconds: float


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Fit the running corpus once and report the road as well as the answer."""
    started = time.perf_counter()
    model = GloVe(
        dimension=request.dimension,
        window=request.window,
        weighting=_WEIGHTINGS[request.weighting],
        maximum_count=request.maximum_count,
        weighting_exponent=request.weighting_exponent,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        random_seed=request.random_seed,
    )
    cooccurrence = _matrix("corpus", request.window, request.weighting)
    walk = _walk(model, cooccurrence, request.step_rule)
    seconds = time.perf_counter() - started

    counts = np.asarray(cooccurrence)
    within, across = _within_and_across(walk.embeddings)
    smallest, largest = walk.divisors
    n_pairs = int(np.count_nonzero(counts))

    return FitResponse(
        n_sentences=_tokenised("corpus").n_sentences,
        n_occurrences=_tokenised("corpus").n_words,
        n_words=cooccurrence.n_words,
        n_pairs=n_pairs,
        n_cells=int(counts.size),
        zero_share=float(np.mean(counts == 0.0)),
        passes=[
            PassReport(
                pass_number=position + 1,
                total=total,
                mean_per_pair=total / n_pairs,
            )
            for position, total in enumerate(walk.totals)
        ],
        objective_at_rest=walk.objective,
        within_topic=within,
        across_topic=across,
        gap=within - across,
        neighbours=_neighbours(walk.embeddings, request.word, request.n_neighbours),
        coordinates=[
            float(value)
            for value in np.asarray(walk.embeddings.vector_of(request.word).values)
        ],
        numbers_kept=int(walk.embeddings.table.size),
        smallest_divisor=smallest,
        largest_divisor=largest,
        seconds=seconds,
    )


# --- The table, built once ---------------------------------------------------


class CountedPair(BaseModel):
    """One cell of the table that is not zero."""

    word: str
    context: str
    count: float
    logarithm: float


class TableRequest(BaseModel):
    """Which text to count, how far to reach, and how to discount distance."""

    text: TextChoice = "one-sentence"
    window: int = Field(default=PAGE_WINDOW, ge=1, le=10)
    weighting: WeightingName = "harmonic"

    model_config = {"extra": "forbid"}


class TableResponse(BaseModel):
    """The counts, and what their logarithms look like.

    ``below_one`` counts the cells whose logarithm is negative, which is the
    quantity the window silently decides on short sentences.
    """

    words: list[str]
    topics: list[str]
    occurrences: list[int]
    table: list[list[float]]
    n_cells: int
    n_nonzero: int
    zero_share: float
    total: float
    largest: float
    smallest: float
    below_one: int
    below_one_share: float
    mean_logarithm: float
    largest_pairs: list[CountedPair]
    n_sentences: int
    n_occurrences: int


@router.post("/table", response_model=TableResponse)
def table(request: TableRequest) -> TableResponse:
    """Count which words appeared near which, in one of the three texts."""
    matrix = _matrix(request.text, request.window, request.weighting)
    counts = np.asarray(matrix)
    listed = list(matrix.vocabulary)
    tokenised = _tokenised(request.text)
    rows, columns = np.nonzero(counts)
    values = counts[rows, columns]
    order = np.argsort(-values)[:10]

    return TableResponse(
        words=listed,
        topics=[TOPIC_OF.get(word, "shared") for word in listed],
        occurrences=[tokenised.word_counts[word] for word in listed],
        table=[[float(value) for value in line] for line in counts],
        n_cells=int(counts.size),
        n_nonzero=int(values.size),
        zero_share=float(np.mean(counts == 0.0)),
        total=matrix.total,
        largest=float(values.max()),
        smallest=float(values.min()),
        below_one=int(np.sum(values < 1.0)),
        below_one_share=float(np.mean(values < 1.0)),
        mean_logarithm=float(np.log(values).mean()),
        largest_pairs=[
            CountedPair(
                word=listed[int(rows[position])],
                context=listed[int(columns[position])],
                count=float(values[position]),
                logarithm=float(math.log(values[position])),
            )
            for position in order
        ],
        n_sentences=tokenised.n_sentences,
        n_occurrences=tokenised.n_words,
    )


# --- The ratio the method argues from ----------------------------------------


class ProbeRow(BaseModel):
    """One probe word, and how often it sits beside each of the two words.

    ``ratio`` is absent where the second word never met the probe, since the
    quantity is a division by zero and not a very large number.
    """

    probe: str
    topic: str
    first_probability: float
    second_probability: float
    ratio: float | None
    verdict: str


class RatioRequest(BaseModel):
    """Two words to compare, through everything either of them met."""

    first_word: str = Field(default="chef", min_length=1, max_length=32)
    second_word: str = Field(default="astronomer", min_length=1, max_length=32)
    window: int = Field(default=PAGE_WINDOW, ge=1, le=5)

    model_config = {"extra": "forbid"}


class RatioResponse(BaseModel):
    """Every probe either word reached, with the two shares and their ratio."""

    first_word: str
    second_word: str
    rows: list[ProbeRow]
    n_shared: int
    n_only_first: int
    n_only_second: int
    n_neither: int


@router.post("/ratios", response_model=RatioResponse)
def ratios(request: RatioRequest) -> RatioResponse:
    """Compare two words through the probes one or both of them appeared near."""
    matrix = _matrix("corpus", request.window, "harmonic")
    vocabulary = matrix.vocabulary
    counts = np.asarray(matrix)
    totals = np.asarray(matrix.word_totals)
    first = vocabulary.id_of(request.first_word)
    second = vocabulary.id_of(request.second_word)

    rows: list[ProbeRow] = []
    shared = only_first = only_second = neither = 0
    for probe_id, probe in enumerate(vocabulary):
        one = float(counts[first, probe_id] / totals[first])
        two = float(counts[second, probe_id] / totals[second])
        if one == 0.0 and two == 0.0:
            neither += 1
            continue
        if one > 0.0 and two > 0.0:
            shared += 1
            verdict = "both"
        elif one > 0.0:
            only_first += 1
            verdict = "only the first"
        else:
            only_second += 1
            verdict = "only the second"
        rows.append(
            ProbeRow(
                probe=probe,
                topic=TOPIC_OF.get(probe, "shared"),
                first_probability=one,
                second_probability=two,
                ratio=one / two if two > 0.0 else None,
                verdict=verdict,
            )
        )

    rows.sort(key=lambda row: -(row.first_probability + row.second_probability))
    return RatioResponse(
        first_word=request.first_word,
        second_word=request.second_word,
        rows=rows,
        n_shared=shared,
        n_only_first=only_first,
        n_only_second=only_second,
        n_neither=neither,
    )


# --- The weighting, and what the cap does ------------------------------------


class CurvePoint(BaseModel):
    """One reading of the weighting curve."""

    count: float
    weight: float


class WeightedPair(BaseModel):
    """One pair of the running corpus, its count and what its term is worth."""

    word: str
    context: str
    count: float
    weight: float
    at_the_cap: bool


class WeightingRequest(BaseModel):
    """Where the cap sits, and how steeply the weight rises below it."""

    maximum_count: float = Field(default=PUBLISHED_MAXIMUM_COUNT, gt=0.0, le=1e6)
    weighting_exponent: float = Field(default=PUBLISHED_EXPONENT, gt=0.0, le=1.0)
    window: int = Field(default=PAGE_WINDOW, ge=1, le=5)

    model_config = {"extra": "forbid"}


class WeightingResponse(BaseModel):
    """The curve, the corpus's own counts on it, and what the cap redistributes.

    ``top_pair_share`` and ``top_ten_share`` are the shares of the summed
    weights the largest one and the largest ten pairs carry, which is what a
    cap moves and a proportional weight does not.
    """

    maximum_count: float
    weighting_exponent: float
    curve: list[CurvePoint]
    pairs: list[WeightedPair]
    n_pairs: int
    n_at_the_cap: int
    largest_count: float
    largest_weight: float
    smallest_weight: float
    top_pair_share: float
    top_ten_share: float
    below_one_share_of_weight: float
    proportional_top_ten_share: float
    alike_top_ten_share: float


CURVE_STEPS = 60
"""How many readings of the weighting curve a widget is handed."""


@router.post("/weighting", response_model=WeightingResponse)
def weighting(request: WeightingRequest) -> WeightingResponse:
    """The weight each count earns, and how much of the objective it buys."""
    model = GloVe(
        maximum_count=request.maximum_count,
        weighting_exponent=request.weighting_exponent,
    )
    matrix = _matrix("corpus", request.window, "harmonic")
    counts = np.asarray(matrix)
    listed = list(matrix.vocabulary)
    rows, columns = np.nonzero(counts)
    values = counts[rows, columns]
    weights = np.array([model.weight_of_count(float(value)) for value in values])
    order = np.argsort(-values)
    top_ten = order[:10]

    ceiling = max(float(values.max()) * 1.25, request.maximum_count * 1.25)
    return WeightingResponse(
        maximum_count=request.maximum_count,
        weighting_exponent=request.weighting_exponent,
        curve=[
            CurvePoint(
                count=step * ceiling / CURVE_STEPS,
                weight=model.weight_of_count(step * ceiling / CURVE_STEPS),
            )
            for step in range(CURVE_STEPS + 1)
        ],
        pairs=[
            WeightedPair(
                word=listed[int(rows[position])],
                context=listed[int(columns[position])],
                count=float(values[position]),
                weight=float(weights[position]),
                at_the_cap=bool(values[position] >= request.maximum_count),
            )
            for position in order
        ],
        n_pairs=int(values.size),
        n_at_the_cap=int(np.sum(values >= request.maximum_count)),
        largest_count=float(values.max()),
        largest_weight=float(weights.max()),
        smallest_weight=float(weights.min()),
        top_pair_share=float(weights[order[0]] / weights.sum()),
        top_ten_share=float(weights[top_ten].sum() / weights.sum()),
        below_one_share_of_weight=float(weights[values < 1.0].sum() / weights.sum()),
        proportional_top_ten_share=float(values[top_ten].sum() / values.sum()),
        alike_top_ten_share=float(len(top_ten) / values.size),
    )


# --- The two step rules ------------------------------------------------------


class RuleReport(BaseModel):
    """One rule's curve, where it stopped, and how far it ever divided a slope."""

    step_rule: StepRuleName
    totals: list[float]
    first_total: float
    final_total: float
    objective_at_rest: float
    smallest_divisor: float
    largest_divisor: float
    within_topic: float
    across_topic: float


class StepRulesResponse(BaseModel):
    """Both rules, at this page's settings and at the published ones.

    The two branches share every piece of arithmetic and differ only in where
    each parameter's running sum of squared slopes starts and when it is added
    to, so the gap between the two curves is that difference and nothing else.
    """

    window: int
    dimension: int
    epochs: int
    learning_rate: float
    reports: list[RuleReport]
    published_window: int
    published_dimension: int
    published_reports: list[RuleReport]


def _rule_report(
    model: GloVe, cooccurrence: CooccurrenceMatrix, step_rule: StepRuleName
) -> RuleReport:
    walk = _walk(model, cooccurrence, step_rule)
    within, across = _within_and_across(walk.embeddings)
    smallest, largest = walk.divisors
    return RuleReport(
        step_rule=step_rule,
        totals=walk.totals,
        first_total=walk.totals[0],
        final_total=walk.totals[-1],
        objective_at_rest=walk.objective,
        smallest_divisor=smallest,
        largest_divisor=largest,
        within_topic=within,
        across_topic=across,
    )


@cache
def _step_rules() -> StepRulesResponse:
    page = GloVe(
        dimension=PAGE_DIMENSION,
        window=PAGE_WINDOW,
        epochs=PAGE_EPOCHS,
        learning_rate=PAGE_LEARNING_RATE,
        random_seed=PAGE_SEED,
    )
    page_matrix = _matrix("corpus", PAGE_WINDOW, "harmonic")
    published = GloVe(
        dimension=PUBLISHED_DIMENSION,
        window=PUBLISHED_WINDOW,
        epochs=PAGE_EPOCHS,
        learning_rate=PAGE_LEARNING_RATE,
        random_seed=PAGE_SEED,
    )
    published_matrix = _matrix("corpus", PUBLISHED_WINDOW, "harmonic")

    return StepRulesResponse(
        window=PAGE_WINDOW,
        dimension=PAGE_DIMENSION,
        epochs=PAGE_EPOCHS,
        learning_rate=PAGE_LEARNING_RATE,
        reports=[
            _rule_report(page, page_matrix, rule)
            for rule in ("accumulating", "released")
        ],
        published_window=PUBLISHED_WINDOW,
        published_dimension=PUBLISHED_DIMENSION,
        published_reports=[
            _rule_report(published, published_matrix, rule)
            for rule in ("accumulating", "released")
        ],
    )


@router.get("/step-rules", response_model=StepRulesResponse)
def step_rules() -> StepRulesResponse:
    """The same walk under both step rules, at two sets of settings."""
    return _step_rules()


# --- What the window decides -------------------------------------------------


class WindowRow(BaseModel):
    """One reach, and what it does to the counts and to the answer.

    ``mean_dot_product`` is the mean of the fitted dot products over the pairs
    that co-occurred, which is where a negative mean logarithm ends up once the
    two biases have failed to absorb it.
    """

    window: int
    n_pairs: int
    below_one: int
    below_one_share: float
    mean_logarithm: float
    within_topic: float
    across_topic: float
    gap: float
    gaps_by_seed: list[float]
    mean_word_bias: float
    mean_dot_product: float


class WindowsResponse(BaseModel):
    """Every reach the page sweeps, fitted from three starts each."""

    seeds: list[int]
    rows: list[WindowRow]


WINDOW_SWEEP: tuple[int, ...] = (1, 2, 3, 5)
"""The reaches the sweep runs, ending at the published one."""

WINDOW_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Three starts, which is enough to show that one reach changes the sign."""


@cache
def _windows() -> WindowsResponse:
    rows: list[WindowRow] = []
    for window in WINDOW_SWEEP:
        cooccurrence = _matrix("corpus", window, "harmonic")
        counts = np.asarray(cooccurrence)
        row_ids, column_ids = np.nonzero(counts)
        values = counts[row_ids, column_ids]

        gaps: list[float] = []
        first_within = first_across = 0.0
        mean_bias = mean_dot = 0.0
        for position, seed in enumerate(WINDOW_SEEDS):
            model = GloVe(
                dimension=PAGE_DIMENSION,
                window=window,
                epochs=PAGE_EPOCHS,
                learning_rate=PAGE_LEARNING_RATE,
                random_seed=seed,
            )
            walk = _walk(model, cooccurrence, "accumulating")
            within, across = _within_and_across(walk.embeddings)
            gaps.append(within - across)
            if position == 0:
                first_within, first_across = within, across
                parameters = walk.parameters
                mean_bias = float(np.asarray(parameters.biases).mean())
                mean_dot = float(
                    np.einsum(
                        "ij,ij->i",
                        np.asarray(parameters.word_vectors)[row_ids],
                        np.asarray(parameters.context_vectors)[column_ids],
                    ).mean()
                )

        rows.append(
            WindowRow(
                window=window,
                n_pairs=int(values.size),
                below_one=int(np.sum(values < 1.0)),
                below_one_share=float(np.mean(values < 1.0)),
                mean_logarithm=float(np.log(values).mean()),
                within_topic=first_within,
                across_topic=first_across,
                gap=first_within - first_across,
                gaps_by_seed=gaps,
                mean_word_bias=mean_bias,
                mean_dot_product=mean_dot,
            )
        )
    return WindowsResponse(seeds=list(WINDOW_SEEDS), rows=rows)


@router.get("/windows", response_model=WindowsResponse)
def windows() -> WindowsResponse:
    """Every reach, with what it does to the logarithms and to the answer."""
    return _windows()


# --- What one pass over a table costs ----------------------------------------


class GrowthRow(BaseModel):
    """One size of corpus, and the two quantities that grow differently.

    ``ordered_positions`` is how many neighbour readings a pass over the text
    makes and ``nonzero_pairs`` how many terms a pass over the table has.
    """

    copies: int
    n_sentences: int
    n_occurrences: int
    n_words: int
    nonzero_pairs: int
    ordered_positions: int


class CostResponse(BaseModel):
    """What the table costs to build, to store and to fit, against the text.

    ``rows_*`` are the same two questions asked of the raw counts, comparing
    whole rows of the table by cosine, which is the thing the fit is meant to
    improve on and does not always.
    """

    growth: list[GrowthRow]
    n_cells: int
    zero_share: float
    numbers_per_word_rows: int
    numbers_per_word_vectors: int
    numbers_in_all_rows: int
    numbers_in_all_vectors: int
    table_seconds: float
    fit_seconds: float
    rows_within_topic: float
    rows_across_topic: float
    rows_gap: float
    rows_same_topic_share: float
    fitted_within_topic: float
    fitted_across_topic: float
    fitted_gap: float
    fitted_same_topic_share: float
    rows_neighbours: list[Neighbour]
    fitted_neighbours: list[Neighbour]
    compared_word: str


GROWTH_COPIES: tuple[int, ...] = (1, 2, 4, 8)
"""How many times over the corpus is repeated, which fixes the vocabulary."""

COMPARED_WORD = "onion"
"""The word whose two neighbour lists the page prints side by side."""


@cache
def _cost() -> CostResponse:
    growth: list[GrowthRow] = []
    for copies in GROWTH_COPIES:
        corpus = TWO_TOPICS * copies
        tokenised = TokenisedCorpus.from_texts(corpus, WhitespacePreTokenizer())
        vocabulary = tokenised.vocabulary(1)
        sequences = tokenised.id_sequences(vocabulary)
        harmonic = CooccurrenceMatrix.from_id_sequences(
            vocabulary, sequences, PAGE_WINDOW, ContextWeighting.HARMONIC
        )
        uniform = CooccurrenceMatrix.from_id_sequences(
            vocabulary, sequences, PAGE_WINDOW, ContextWeighting.UNIFORM
        )
        growth.append(
            GrowthRow(
                copies=copies,
                n_sentences=tokenised.n_sentences,
                n_occurrences=tokenised.n_words,
                n_words=vocabulary.n_tokens,
                nonzero_pairs=int(np.count_nonzero(np.asarray(harmonic))),
                ordered_positions=int(uniform.total),
            )
        )

    cooccurrence = _matrix("corpus", PAGE_WINDOW, "harmonic")
    counts = np.asarray(cooccurrence)
    rows_embeddings = WordEmbeddings(cooccurrence.vocabulary, counts)

    started = time.perf_counter()
    CooccurrenceMatrix.from_id_sequences(
        cooccurrence.vocabulary,
        _tokenised("corpus").id_sequences(cooccurrence.vocabulary),
        PAGE_WINDOW,
        ContextWeighting.HARMONIC,
    )
    table_seconds = time.perf_counter() - started

    model = GloVe(
        dimension=PAGE_DIMENSION,
        window=PAGE_WINDOW,
        epochs=PAGE_EPOCHS,
        learning_rate=PAGE_LEARNING_RATE,
        random_seed=PAGE_SEED,
    )
    started = time.perf_counter()
    walk = _walk(model, cooccurrence, "accumulating")
    fit_seconds = time.perf_counter() - started

    rows_within, rows_across = _within_and_across(rows_embeddings)
    fitted_within, fitted_across = _within_and_across(walk.embeddings)

    return CostResponse(
        growth=growth,
        n_cells=int(counts.size),
        zero_share=float(np.mean(counts == 0.0)),
        numbers_per_word_rows=int(counts.shape[1]),
        numbers_per_word_vectors=PAGE_DIMENSION,
        numbers_in_all_rows=int(counts.size),
        numbers_in_all_vectors=int(walk.embeddings.table.size),
        table_seconds=table_seconds,
        fit_seconds=fit_seconds,
        rows_within_topic=rows_within,
        rows_across_topic=rows_across,
        rows_gap=rows_within - rows_across,
        rows_same_topic_share=_same_topic_share(rows_embeddings),
        fitted_within_topic=fitted_within,
        fitted_across_topic=fitted_across,
        fitted_gap=fitted_within - fitted_across,
        fitted_same_topic_share=_same_topic_share(walk.embeddings),
        rows_neighbours=_neighbours(rows_embeddings, COMPARED_WORD, 5),
        fitted_neighbours=_neighbours(walk.embeddings, COMPARED_WORD, 5),
        compared_word=COMPARED_WORD,
    )


@router.get("/cost", response_model=CostResponse)
def cost() -> CostResponse:
    """What the table costs, and what comparing its rows already answers."""
    return _cost()
