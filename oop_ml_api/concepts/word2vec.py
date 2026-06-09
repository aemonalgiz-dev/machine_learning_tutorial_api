"""Word2vec.

Every endpoint here is registered under ``/concepts/word2vec``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about Word2Vec, UnigramSampler, HuffmanTree.

The running corpus is the one the library's own word2vec spec fits, two word
lists that never share a sentence: twenty verb forms and eleven money words,
drawn into two hundred sentences of five to eight words under a fixed seed. It
is reproduced here rather than imported, because ``test`` is not part of the
installed package, and a test asserts the two agree word for word.

Three things are computed here rather than in the library, and each is a
definition rather than a fit. Enumerating the pairs a window makes is the loop's
own rule written out for a reader to check; the cosine between two rows of the
counting table is the baseline the method is measured against; and the
count-weighted mean path length of a tree over a made-up vocabulary is
arithmetic on counts. Everything that is a fit, a draw, a tree or a gradient is
the library's.

Nothing a widget renders may name a class, a parameter or a module, so the
architectures and objectives cross the wire as the page's own words,
``skip-gram`` and ``bag-of-words``, ``negative-sampling`` and ``tree``, and
``_ARCHITECTURES`` and ``_OBJECTIVES`` are the only places those meet the
library's own names.
"""

from __future__ import annotations

import random
import time
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanTree,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    binary_logistic_gradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    Word2Vec,
    Word2VecArchitecture,
    Word2VecObjective,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts/word2vec", tags=["word2vec"])


# --- The corpus ------------------------------------------------------------

VERB_WORDS: tuple[str, ...] = (
    "play",
    "played",
    "plays",
    "player",
    "stay",
    "stayed",
    "stays",
    "staying",
    "say",
    "said",
    "says",
    "saying",
    "walk",
    "walked",
    "walks",
    "walking",
    "talk",
    "talked",
    "talks",
    "talking",
)
"""One list: twenty forms of five verbs."""

MONEY_WORDS: tuple[str, ...] = (
    "stock",
    "bond",
    "market",
    "price",
    "trade",
    "fund",
    "bank",
    "cash",
    "debt",
    "yield",
    "share",
)
"""The other list: eleven words about money."""


def _two_topic_sentences(n_per_list: int = 100, seed: int = 11) -> list[str]:
    """Sentences of five to eight words, each drawn from one list only.

    The lists alternate so neither sits at one end of the corpus, and the draw
    is seeded, so the corpus is a function of nothing but this code.
    """
    draw = random.Random(seed)
    sentences: list[str] = []
    for _ in range(n_per_list):
        for words in (VERB_WORDS, MONEY_WORDS):
            length = draw.randint(5, 8)
            sentences.append(" ".join(draw.choice(words) for _ in range(length)))
    return sentences


TWO_LIST_CORPUS: list[str] = _two_topic_sentences()
"""Two hundred sentences, a hundred drawn from each list."""

SMALL_CORPUS: tuple[str, ...] = (
    "walk walked talks",
    "bond market price",
    "walked talks walk",
)
"""Nine occurrences of six words, small enough to follow with a pencil."""

TOPIC_OF: dict[str, str] = {
    **{word: "verb" for word in VERB_WORDS},
    **{word: "money" for word in MONEY_WORDS},
}
"""Which list each word came from, for the widgets that colour by it."""


# --- Translating the page's words into the library's -------------------------

ArchitectureName = Literal["skip-gram", "bag-of-words"]
ObjectiveName = Literal["negative-sampling", "tree"]

_ARCHITECTURES: dict[ArchitectureName, Word2VecArchitecture] = {
    "skip-gram": Word2VecArchitecture.SKIP_GRAM,
    "bag-of-words": Word2VecArchitecture.CONTINUOUS_BAG_OF_WORDS,
}
_OBJECTIVES: dict[ObjectiveName, Word2VecObjective] = {
    "negative-sampling": Word2VecObjective.NEGATIVE_SAMPLING,
    "tree": Word2VecObjective.HIERARCHICAL_SOFTMAX,
}

ARCHITECTURE_NAMES: tuple[ArchitectureName, ...] = ("skip-gram", "bag-of-words")
OBJECTIVE_NAMES: tuple[ObjectiveName, ...] = ("negative-sampling", "tree")

MAX_EPOCHS = 8
"""The most passes one interactive fit may make.

A pass over the two hundred sentences is a Python loop over every position, so
eight is already a couple of seconds and is far more than a reader waits for.
The fixed sweeps that need more passes are computed once and cached.
"""

MAX_DIMENSION = 24
"""The widest vector an interactive fit may learn.

Width costs almost nothing here, measured, but the bound keeps a crafted
request from turning a teaching endpoint into a compute job.
"""

DEFAULT_DIMENSION = 12
"""The width every fixed measurement on the page uses."""

DEFAULT_WINDOW = 3
"""The reach every fixed measurement on the page uses."""


# --- Shared pieces -----------------------------------------------------------


@cache
def _tokenised() -> TokenisedCorpus:
    """The two-list corpus split into words, once."""
    return TokenisedCorpus.from_texts(TWO_LIST_CORPUS, WhitespacePreTokenizer())


@cache
def _vocabulary_and_counts() -> tuple[Vocabulary, tuple[int, ...]]:
    """The vocabulary in the library's own order, with each word's count."""
    tokenised = _tokenised()
    vocabulary = tokenised.vocabulary(1)
    counts = tuple(tokenised.word_counts[word] for word in vocabulary)
    return vocabulary, counts


def _within_and_across(embeddings: WordEmbeddings) -> tuple[float, float]:
    """Mean cosine inside a list, and mean cosine between the two lists."""
    inside: list[float] = []
    for words in (VERB_WORDS, MONEY_WORDS):
        for position, first in enumerate(words):
            for second in words[position + 1 :]:
                inside.append(embeddings.similarity(first, second))
    between = [
        embeddings.similarity(first, second)
        for first in VERB_WORDS
        for second in MONEY_WORDS
    ]
    return float(np.mean(inside)), float(np.mean(between))


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
            topic=TOPIC_OF[similar.word],
        )
        for similar in embeddings.most_similar(word, n_results)
    ]


class EpochReport(BaseModel):
    """What one pass over the corpus cost."""

    epoch: int
    mean_loss: float
    n_pairs: int
    learning_rate: float


def _epoch_reports(model: Word2Vec) -> list[EpochReport]:
    return [
        EpochReport(
            epoch=record.epoch,
            mean_loss=record.mean_loss,
            n_pairs=record.n_pairs,
            learning_rate=record.learning_rate,
        )
        for record in model.history
    ]


def _fit(
    architecture: ArchitectureName,
    objective: ObjectiveName,
    learning_rate: float,
    epochs: int,
    window: int = DEFAULT_WINDOW,
    dimension: int = DEFAULT_DIMENSION,
    random_seed: int = 0,
    corpus: list[str] | None = None,
) -> Word2Vec:
    """One fit of the running corpus, with the page's words translated."""
    return Word2Vec(
        dimension=dimension,
        window=window,
        architecture=_ARCHITECTURES[architecture],
        objective=_OBJECTIVES[objective],
        epochs=epochs,
        learning_rate=learning_rate,
        random_seed=random_seed,
    ).fit(TWO_LIST_CORPUS if corpus is None else corpus)


# --- One fit, on request -----------------------------------------------------


class FitRequest(BaseModel):
    """One fit of the running corpus, and the word to look around afterwards."""

    architecture: ArchitectureName = "skip-gram"
    objective: ObjectiveName = "negative-sampling"
    learning_rate: float = Field(default=0.05, gt=0.0, le=0.5)
    epochs: int = Field(default=5, ge=1, le=MAX_EPOCHS)
    window: int = Field(default=DEFAULT_WINDOW, ge=1, le=5)
    dimension: int = Field(default=DEFAULT_DIMENSION, ge=2, le=MAX_DIMENSION)
    random_seed: int = Field(default=0, ge=0, le=999)
    word: str = Field(default="walked", min_length=1, max_length=32)
    n_neighbours: int = Field(default=6, ge=1, le=12)

    model_config = {"extra": "forbid"}


class FitResponse(BaseModel):
    """A fit, the objective falling, and where one word ended up.

    ``kept_numbers`` is the size of the table the fit answers with and
    ``discarded_numbers`` the size of the table it needed and does not
    return, so the two together are what the fit actually learned.
    ``predicted`` is what the discarded half says, for each word in turn, about
    whether it belongs beside the chosen one, and the two means beside it are
    that same reading averaged over the chosen word's own list and over the
    other. Both are zero under the tree, whose rows are branches rather than
    words and cannot be read one word at a time.
    """

    n_sentences: int
    n_occurrences: int
    n_words: int
    epochs: list[EpochReport]
    total_pairs: int
    within_topic: float
    across_topic: float
    neighbours: list[Neighbour]
    coordinates: list[float]
    kept_numbers: int
    discarded_numbers: int
    output_rows: int
    predicted: list[Neighbour]
    predicted_mean_same_list: float
    predicted_mean_other_list: float
    seconds: float


@router.post("/fit", response_model=FitResponse)
def fit(request: FitRequest) -> FitResponse:
    """Fit the running corpus once and report the road as well as the answer."""
    started = time.perf_counter()
    model = _fit(
        request.architecture,
        request.objective,
        request.learning_rate,
        request.epochs,
        window=request.window,
        dimension=request.dimension,
        random_seed=request.random_seed,
    )
    seconds = time.perf_counter() - started

    embeddings = model.embeddings
    within, across = _within_and_across(embeddings)
    outputs = np.asarray(model.output_vectors)
    centre = np.asarray(embeddings.vector_of(request.word).values)
    words = list(embeddings.vocabulary)

    predicted: list[Neighbour] = []
    same_list_mean = 0.0
    other_list_mean = 0.0
    if request.objective == "negative-sampling":
        scores = outputs @ centre
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        order = np.argsort(-probabilities)[: request.n_neighbours]
        predicted = [
            Neighbour(
                word=words[position],
                similarity=float(probabilities[position]),
                topic=TOPIC_OF[words[position]],
            )
            for position in order
        ]
        own = TOPIC_OF[request.word]
        same = [
            probabilities[position]
            for position, word in enumerate(words)
            if TOPIC_OF[word] == own
        ]
        other = [
            probabilities[position]
            for position, word in enumerate(words)
            if TOPIC_OF[word] != own
        ]
        same_list_mean = float(np.mean(same))
        other_list_mean = float(np.mean(other))

    return FitResponse(
        n_sentences=len(TWO_LIST_CORPUS),
        n_occurrences=_tokenised().n_words,
        n_words=embeddings.n_words,
        epochs=_epoch_reports(model),
        total_pairs=sum(record.n_pairs for record in model.history),
        within_topic=within,
        across_topic=across,
        neighbours=_neighbours(embeddings, request.word, request.n_neighbours),
        coordinates=[float(value) for value in centre],
        kept_numbers=int(embeddings.table.size),
        discarded_numbers=int(outputs.size),
        output_rows=int(outputs.shape[0]),
        seconds=seconds,
        predicted=predicted,
        predicted_mean_same_list=same_list_mean,
        predicted_mean_other_list=other_list_mean,
    )


# --- The table counting gives you --------------------------------------------


class CountingResponse(BaseModel):
    """The neighbour counts, their zeros, and what they separate.

    ``within_topic`` and ``across_topic`` are cosines between whole rows of the
    count table, which is the baseline a learned table has to beat.
    ``row_zero_shares`` is, per word, the share of its row that is zero.
    """

    words: list[str]
    topics: list[str]
    counts: list[int]
    table: list[list[float]]
    n_cells: int
    n_zero_cells: int
    zero_share: float
    row_width: int
    vector_width: int
    vector_numbers: int
    within_topic: float
    across_topic: float
    largest_for_walked: list[Neighbour]


@cache
def _counting() -> CountingResponse:
    vocabulary, counts = _vocabulary_and_counts()
    matrix = CooccurrenceMatrix.from_id_sequences(
        vocabulary, _tokenised().id_sequences(vocabulary), window=DEFAULT_WINDOW
    )
    table = np.asarray(matrix)
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    identifier = {word: vocabulary.id_of(word) for word in vocabulary}

    inside: list[float] = []
    for words in (VERB_WORDS, MONEY_WORDS):
        for position, first in enumerate(words):
            for second in words[position + 1 :]:
                inside.append(float(unit[identifier[first]] @ unit[identifier[second]]))
    between = [
        float(unit[identifier[first]] @ unit[identifier[second]])
        for first in VERB_WORDS
        for second in MONEY_WORDS
    ]

    listed = list(vocabulary)
    row = table[identifier["walked"]]
    order = np.argsort(-row)[:5]
    return CountingResponse(
        words=listed,
        topics=[TOPIC_OF[word] for word in listed],
        counts=list(counts),
        table=[[float(value) for value in line] for line in table],
        n_cells=int(table.size),
        n_zero_cells=int(np.sum(table == 0.0)),
        zero_share=float(np.mean(table == 0.0)),
        row_width=int(table.shape[1]),
        vector_width=DEFAULT_DIMENSION,
        vector_numbers=int(table.shape[0]) * DEFAULT_DIMENSION,
        within_topic=float(np.mean(inside)),
        across_topic=float(np.mean(between)),
        largest_for_walked=[
            Neighbour(
                word=listed[position],
                similarity=float(row[position]),
                topic=TOPIC_OF[listed[position]],
            )
            for position in order
        ],
    )


@router.get("/counting", response_model=CountingResponse)
def counting() -> CountingResponse:
    """How often every word of the corpus appears beside every other."""
    return _counting()


# --- The pairs a window makes ------------------------------------------------


class WindowPair(BaseModel):
    """One position of the window, and the training it produces.

    ``reads`` are the words whose vectors are read to form the prediction and
    ``predicts`` the words that have to be predicted from them, so under
    skip-gram one position gives one entry per neighbour and under the bag of
    words one entry whatever the reach.
    """

    sentence: int
    position: int
    centre: str
    context: list[str]
    reads: list[str]
    predicts: list[str]


class WindowsResponse(BaseModel):
    """The small corpus, and every pair the window makes at one reach."""

    sentences: list[list[str]]
    words: list[str]
    counts: list[int]
    pairs: list[WindowPair]
    n_pairs: int
    n_pairs_skip_gram: int
    n_pairs_bag_of_words: int


class WindowsRequest(BaseModel):
    """A reach and a direction, over the small corpus."""

    reach: int = Field(default=1, ge=1, le=3)
    architecture: ArchitectureName = "skip-gram"

    model_config = {"extra": "forbid"}


def _small_pieces() -> tuple[Vocabulary, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    tokenised = TokenisedCorpus.from_texts(list(SMALL_CORPUS), WhitespacePreTokenizer())
    vocabulary = tokenised.vocabulary(1)
    counts = tuple(tokenised.word_counts[word] for word in vocabulary)
    return vocabulary, counts, tokenised.id_sequences(vocabulary)


def _pairs_at(reach: int, architecture: ArchitectureName) -> list[WindowPair]:
    """Write out the rule the fitting loop applies at every position.

    The loop reads a slice of ``reach`` words on each side, skipping the centre,
    and then either trains the centre against each neighbour in turn or trains
    the averaged neighbours against the centre. That is arithmetic on indices
    rather than a fit, which is why it is written here.
    """
    vocabulary, _, sequences = _small_pieces()
    listed = list(vocabulary)
    pairs: list[WindowPair] = []
    for sentence_index, sentence in enumerate(sequences):
        for position, centre in enumerate(sentence):
            context = list(sentence[max(0, position - reach) : position]) + list(
                sentence[position + 1 : position + 1 + reach]
            )
            if not context:
                continue
            names = [listed[word_id] for word_id in context]
            if architecture == "skip-gram":
                for name in names:
                    pairs.append(
                        WindowPair(
                            sentence=sentence_index,
                            position=position,
                            centre=listed[centre],
                            context=names,
                            reads=[listed[centre]],
                            predicts=[name],
                        )
                    )
            else:
                pairs.append(
                    WindowPair(
                        sentence=sentence_index,
                        position=position,
                        centre=listed[centre],
                        context=names,
                        reads=names,
                        predicts=[listed[centre]],
                    )
                )
    return pairs


@router.post("/windows", response_model=WindowsResponse)
def windows(request: WindowsRequest) -> WindowsResponse:
    """Every training pair the small corpus makes at the chosen reach."""
    vocabulary, counts, sequences = _small_pieces()
    listed = list(vocabulary)
    pairs = _pairs_at(request.reach, request.architecture)
    return WindowsResponse(
        sentences=[[listed[word_id] for word_id in sentence] for sentence in sequences],
        words=listed,
        counts=list(counts),
        pairs=pairs,
        n_pairs=len(pairs),
        n_pairs_skip_gram=len(_pairs_at(request.reach, "skip-gram")),
        n_pairs_bag_of_words=len(_pairs_at(request.reach, "bag-of-words")),
    )


# --- One pair, scored and differentiated -------------------------------------


class PlanePoint(BaseModel):
    """A vector in the plane, so a pair can be worked by hand."""

    x: float
    y: float


class ScoredRow(BaseModel):
    """One output row, its score, and where the step pushes it."""

    label: int
    score: float
    probability: float
    loss: float
    error: float
    push_x: float
    push_y: float


class PairStepResponse(BaseModel):
    """What one training pair costs and what it asks to move.

    ``loss`` is the sum over the rows and ``hidden_push`` the direction the
    word's own vector is moved in, both before the learning rate multiplies
    them. The pushes are the negatives of the gradients, which is the direction
    a step actually travels.
    """

    hidden: PlanePoint
    rows: list[ScoredRow]
    loss: float
    hidden_push: PlanePoint


class PairStepRequest(BaseModel):
    """A word's vector, the true neighbour's row, and the drawn wrong rows."""

    hidden: PlanePoint
    true_row: PlanePoint
    negative_rows: list[PlanePoint] = Field(min_length=1, max_length=8)

    model_config = {"extra": "forbid"}


@router.post("/pair-step", response_model=PairStepResponse)
def pair_step(request: PairStepRequest) -> PairStepResponse:
    """Score one pair against its negatives and report the step it asks for."""
    hidden = np.array([request.hidden.x, request.hidden.y], dtype=np.float64)
    rows = np.array(
        [[request.true_row.x, request.true_row.y]]
        + [[row.x, row.y] for row in request.negative_rows],
        dtype=np.float64,
    )
    labels = np.zeros(len(rows))
    labels[0] = 1.0
    gradients = binary_logistic_gradients(hidden, np.arange(len(rows)), rows, labels)

    scores = rows @ hidden
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    errors = probabilities - labels
    return PairStepResponse(
        hidden=request.hidden,
        rows=[
            ScoredRow(
                label=int(labels[position]),
                score=float(scores[position]),
                probability=float(probabilities[position]),
                loss=float(
                    np.logaddexp(
                        0.0, -(2.0 * labels[position] - 1.0) * scores[position]
                    )
                ),
                error=float(errors[position]),
                push_x=float(-gradients.output_gradients[position, 0]),
                push_y=float(-gradients.output_gradients[position, 1]),
            )
            for position in range(len(rows))
        ],
        loss=float(gradients.loss),
        hidden_push=PlanePoint(
            x=float(-gradients.hidden_gradient[0]),
            y=float(-gradients.hidden_gradient[1]),
        ),
    )


# --- Where the wrong answers come from ---------------------------------------


class SamplerRow(BaseModel):
    """One word's count and its two shares of the draw."""

    word: str
    topic: str
    count: int
    raw_share: float
    flattened_share: float
    ratio: float


class SamplerResponse(BaseModel):
    """What the draw looks like at one exponent, against raw frequency.

    ``toy_counts`` is the three-word illustration the page works through,
    counts of a thousand, ten and one, where the effect is large enough to
    read off without a chart.
    """

    exponent: float
    rows: list[SamplerRow]
    commonest: SamplerRow
    rarest: SamplerRow
    raw_odds: float
    flattened_odds: float
    toy_counts: list[int]
    toy_raw: list[float]
    toy_flattened: list[float]


class SamplerRequest(BaseModel):
    """The flattening power to draw the wrong answers with."""

    exponent: float = Field(default=0.75, gt=0.0, le=1.0)

    model_config = {"extra": "forbid"}


@router.post("/sampler", response_model=SamplerResponse)
def sampler(request: SamplerRequest) -> SamplerResponse:
    """The share of the draw each word receives, raw and flattened."""
    vocabulary, counts = _vocabulary_and_counts()
    raw = UnigramSampler(list(counts), 1.0)
    flattened = UnigramSampler(list(counts), request.exponent)
    listed = list(vocabulary)

    rows = [
        SamplerRow(
            word=word,
            topic=TOPIC_OF[word],
            count=counts[position],
            raw_share=raw.probability_of(position),
            flattened_share=flattened.probability_of(position),
            ratio=flattened.probability_of(position) / raw.probability_of(position),
        )
        for position, word in enumerate(listed)
    ]
    commonest = max(rows, key=lambda row: row.count)
    rarest = min(rows, key=lambda row: row.count)

    toy_counts = [1000, 10, 1]
    return SamplerResponse(
        exponent=request.exponent,
        rows=rows,
        commonest=commonest,
        rarest=rarest,
        raw_odds=commonest.raw_share / rarest.raw_share,
        flattened_odds=commonest.flattened_share / rarest.flattened_share,
        toy_counts=toy_counts,
        toy_raw=[
            float(value) for value in UnigramSampler(toy_counts, 1.0).probabilities
        ],
        toy_flattened=[
            float(value)
            for value in UnigramSampler(toy_counts, request.exponent).probabilities
        ],
    )


# --- The tree, and what it saves ---------------------------------------------


class TreePath(BaseModel):
    """One word's route from the root, and how long it is."""

    word: str
    count: int
    depth: int
    nodes: list[int]
    bits: list[int]


class SizeRow(BaseModel):
    """What one vocabulary size costs under each of the three rules.

    ``skewed_depth`` is the same tree over counts falling as one over the rank,
    which is roughly how word counts really fall, and ``balanced_depth`` the
    depth of a tree that ignores the counts entirely.
    """

    n_words: int
    sweep_rows: int
    negative_rows: int
    uniform_depth: float
    skewed_depth: float
    balanced_depth: float


class TreeResponse(BaseModel):
    """The small corpus's tree, the running corpus's, and the cost by size."""

    small_words: list[str]
    small_counts: list[int]
    small_paths: list[TreePath]
    small_internal_nodes: int
    small_expected_depth: float
    small_sweep_rows: int
    corpus_words: int
    corpus_expected_depth: float
    corpus_balanced_depth: float
    corpus_sweep_rows: int
    corpus_negative_rows: int
    corpus_shortest_depth: int
    corpus_longest_depth: int
    sizes: list[SizeRow]


TREE_SIZES: tuple[int, ...] = (31, 100, 1000, 10000, 40000)
"""Vocabulary sizes the cost table runs over, ending near a real one."""

DEFAULT_NEGATIVES = 5
"""How many wrong answers a pair is contrasted against, by default."""


@cache
def _tree() -> TreeResponse:
    small_vocabulary, small_counts, _ = _small_pieces()
    small_tree = HuffmanTree.from_counts(list(small_counts))
    small_listed = list(small_vocabulary)

    vocabulary, counts = _vocabulary_and_counts()
    corpus_tree = HuffmanTree.from_counts(list(counts))
    depths = [
        corpus_tree.code_of(position).depth for position in range(corpus_tree.n_words)
    ]

    sizes: list[SizeRow] = []
    for n_words in TREE_SIZES:
        uniform = [1] * n_words
        skewed = [max(1, int(1_000_000 / (rank + 1))) for rank in range(n_words)]
        sizes.append(
            SizeRow(
                n_words=n_words,
                sweep_rows=n_words,
                negative_rows=DEFAULT_NEGATIVES + 1,
                uniform_depth=HuffmanTree.from_counts(uniform).expected_depth(uniform),
                skewed_depth=HuffmanTree.from_counts(skewed).expected_depth(skewed),
                balanced_depth=float(np.log2(n_words)),
            )
        )

    return TreeResponse(
        small_words=small_listed,
        small_counts=list(small_counts),
        small_paths=[
            TreePath(
                word=word,
                count=small_counts[position],
                depth=small_tree.code_of(position).depth,
                nodes=list(small_tree.code_of(position).nodes),
                bits=list(small_tree.code_of(position).bits),
            )
            for position, word in enumerate(small_listed)
        ],
        small_internal_nodes=small_tree.n_internal_nodes,
        small_expected_depth=small_tree.expected_depth(list(small_counts)),
        small_sweep_rows=len(small_listed),
        corpus_words=corpus_tree.n_words,
        corpus_expected_depth=corpus_tree.expected_depth(list(counts)),
        corpus_balanced_depth=float(np.log2(corpus_tree.n_words)),
        corpus_sweep_rows=vocabulary.n_tokens,
        corpus_negative_rows=DEFAULT_NEGATIVES + 1,
        corpus_shortest_depth=min(depths),
        corpus_longest_depth=max(depths),
        sizes=sizes,
    )


@router.get("/tree", response_model=TreeResponse)
def tree() -> TreeResponse:
    """The Huffman paths, and what a path costs against a sweep."""
    return _tree()


# --- The four combinations, at two learning rates ----------------------------


class CombinationReport(BaseModel):
    """One of the four, fitted at one learning rate."""

    architecture: ArchitectureName
    objective: ObjectiveName
    learning_rate: float
    within_topic: float
    across_topic: float
    gap: float
    mean_losses: list[float]
    pairs_per_epoch: int
    output_rows: int
    neighbours: dict[str, list[Neighbour]]


class CombinationsResponse(BaseModel):
    """Every architecture against every objective, at both rates.

    The reference learning rate is 0.025 and the doubled one is 0.05; the page
    shows both because one of the four does not separate the lists at the
    reference value.
    """

    reference_rate: float
    doubled_rate: float
    epochs: int
    dimension: int
    window: int
    words: list[str]
    reports: list[CombinationReport]


REFERENCE_RATE = 0.025
"""The published starting step size."""

DOUBLED_RATE = 0.05
"""Twice it, which is what this corpus needs."""

FIXED_EPOCHS = 5
"""How many passes every fixed measurement on the page makes."""


@cache
def _combinations() -> CombinationsResponse:
    vocabulary, _ = _vocabulary_and_counts()
    listed = list(vocabulary)
    reports: list[CombinationReport] = []
    for rate in (REFERENCE_RATE, DOUBLED_RATE):
        for architecture in ARCHITECTURE_NAMES:
            for objective in OBJECTIVE_NAMES:
                model = _fit(architecture, objective, rate, FIXED_EPOCHS)
                embeddings = model.embeddings
                within, across = _within_and_across(embeddings)
                reports.append(
                    CombinationReport(
                        architecture=architecture,
                        objective=objective,
                        learning_rate=rate,
                        within_topic=within,
                        across_topic=across,
                        gap=within - across,
                        mean_losses=list(model.history.mean_losses),
                        pairs_per_epoch=model.history[0].n_pairs,
                        output_rows=int(model.output_vectors.shape[0]),
                        neighbours={
                            word: _neighbours(embeddings, word, 5) for word in listed
                        },
                    )
                )
    return CombinationsResponse(
        reference_rate=REFERENCE_RATE,
        doubled_rate=DOUBLED_RATE,
        epochs=FIXED_EPOCHS,
        dimension=DEFAULT_DIMENSION,
        window=DEFAULT_WINDOW,
        words=listed,
        reports=reports,
    )


@router.get("/combinations", response_model=CombinationsResponse)
def combinations() -> CombinationsResponse:
    """The four shapes of the method, fitted at both learning rates."""
    return _combinations()


# --- Training for longer -----------------------------------------------------


class LengthReport(BaseModel):
    """One fit of a given number of passes, scored two ways."""

    epochs: int
    first_loss: float
    final_loss: float
    within_topic: float
    across_topic: float
    gap: float


class LongerTrainingResponse(BaseModel):
    """What more passes do to the objective and to the vectors separately."""

    points: list[LengthReport]


TRAINING_LENGTHS: tuple[int, ...] = (1, 2, 5, 10, 20)
"""Pass counts the sweep runs, ending well past the point the vectors turn."""


@cache
def _longer_training() -> LongerTrainingResponse:
    points: list[LengthReport] = []
    for epochs in TRAINING_LENGTHS:
        model = _fit("skip-gram", "negative-sampling", DOUBLED_RATE, epochs)
        within, across = _within_and_across(model.embeddings)
        points.append(
            LengthReport(
                epochs=epochs,
                first_loss=model.history.mean_losses[0],
                final_loss=model.history.mean_losses[-1],
                within_topic=within,
                across_topic=across,
                gap=within - across,
            )
        )
    return LongerTrainingResponse(points=points)


@router.get("/longer-training", response_model=LongerTrainingResponse)
def longer_training() -> LongerTrainingResponse:
    """Fit the same corpus for one pass and for twenty, and score both ways."""
    return _longer_training()


# --- What a seed fixes -------------------------------------------------------


class SeedReport(BaseModel):
    """One fit's own answer for the word the page follows."""

    seed: int
    coordinates: list[float]
    neighbours: list[Neighbour]


class SeedsResponse(BaseModel):
    """Two starts, and what agrees between them.

    The coordinates are two different lists of numbers; the cosines between
    every pair of words are very nearly the same list, which is the difference
    between what the fit determines and what it merely happens to land on.
    """

    word: str
    reports: list[SeedReport]
    mean_coordinate_gap: float
    largest_coordinate_gap: float
    similarity_correlation: float
    mean_similarity_gap: float
    shared_neighbours: list[str]


@cache
def _seeds() -> SeedsResponse:
    word = "walked"
    models = [
        _fit(
            "skip-gram",
            "negative-sampling",
            DOUBLED_RATE,
            FIXED_EPOCHS,
            random_seed=seed,
        )
        for seed in (0, 1)
    ]
    tables = [np.asarray(model.embeddings.table) for model in models]

    def similarity_matrix(table: np.ndarray) -> np.ndarray:
        unit = table / np.linalg.norm(table, axis=1, keepdims=True)
        return unit @ unit.T

    upper = np.triu_indices(tables[0].shape[0], 1)
    first = similarity_matrix(tables[0])[upper]
    second = similarity_matrix(tables[1])[upper]
    lists = [_neighbours(model.embeddings, word, 5) for model in models]
    shared = sorted(
        {entry.word for entry in lists[0]} & {entry.word for entry in lists[1]}
    )

    return SeedsResponse(
        word=word,
        reports=[
            SeedReport(
                seed=seed,
                coordinates=[
                    float(value)
                    for value in np.asarray(model.embeddings.vector_of(word).values)
                ],
                neighbours=entries,
            )
            for seed, model, entries in zip((0, 1), models, lists, strict=True)
        ],
        mean_coordinate_gap=float(np.mean(np.abs(tables[0] - tables[1]))),
        largest_coordinate_gap=float(np.max(np.abs(tables[0] - tables[1]))),
        similarity_correlation=float(np.corrcoef(first, second)[0, 1]),
        mean_similarity_gap=float(np.mean(np.abs(first - second))),
        shared_neighbours=shared,
    )


@router.get("/seeds", response_model=SeedsResponse)
def seeds() -> SeedsResponse:
    """The same corpus from two starts, compared coordinate by coordinate."""
    return _seeds()


# --- A word seen a handful of times ------------------------------------------


class RareWordResponse(BaseModel):
    """How much a rare word's position moves when only the start changes.

    The corpus is the running one with two extra sentences, so one made-up
    word appears twice against thirty-eight appearances of the word it is
    compared with. Both are fitted under each seed; the frequent pair barely
    moves and the rare one moves by an order more.
    """

    rare_word: str
    rare_count: int
    common_word: str
    common_count: int
    against: str
    seeds: list[int]
    rare_similarities: list[float]
    common_similarities: list[float]
    rare_spread: float
    common_spread: float
    rare_neighbour_lists: list[list[str]]
    common_neighbour_lists: list[list[str]]


RARE_SEEDS: tuple[int, ...] = (0, 1, 2, 3)
"""Four starts, which is enough to show the spread and cheap enough to cache."""

RARE_SENTENCES: tuple[str, ...] = (
    "walk quibble walked",
    "talks quibble talk",
)
"""Two sentences carrying a word that appears nowhere else."""


@cache
def _rare_word() -> RareWordResponse:
    corpus = list(TWO_LIST_CORPUS) + list(RARE_SENTENCES)
    rare, common, against = "quibble", "walk", "walked"
    rare_similarities: list[float] = []
    common_similarities: list[float] = []
    rare_lists: list[list[str]] = []
    common_lists: list[list[str]] = []
    rare_count = 0
    common_count = 0
    for seed in RARE_SEEDS:
        model = _fit(
            "skip-gram",
            "negative-sampling",
            DOUBLED_RATE,
            FIXED_EPOCHS,
            random_seed=seed,
            corpus=corpus,
        )
        embeddings = model.embeddings
        rare_similarities.append(embeddings.similarity(rare, against))
        common_similarities.append(embeddings.similarity(common, against))
        rare_lists.append(list(embeddings.most_similar(rare, 3).words))
        common_lists.append(list(embeddings.most_similar(common, 3).words))
    tokenised = TokenisedCorpus.from_texts(corpus, WhitespacePreTokenizer())
    rare_count = tokenised.word_counts[rare]
    common_count = tokenised.word_counts[common]

    return RareWordResponse(
        rare_word=rare,
        rare_count=rare_count,
        common_word=common,
        common_count=common_count,
        against=against,
        seeds=list(RARE_SEEDS),
        rare_similarities=rare_similarities,
        common_similarities=common_similarities,
        rare_spread=max(rare_similarities) - min(rare_similarities),
        common_spread=max(common_similarities) - min(common_similarities),
        rare_neighbour_lists=rare_lists,
        common_neighbour_lists=common_lists,
    )


@router.get("/rare-word", response_model=RareWordResponse)
def rare_word() -> RareWordResponse:
    """A word seen twice and a word seen many times, across four starts."""
    return _rare_word()
