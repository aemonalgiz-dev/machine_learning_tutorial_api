"""One small convolutional network over the picture collection, built and
trained the same way for every page that needs one.

Four pages train on pictures: the one assembling a convolutional network, the
one reading a vector off it, the one training that vector for nearness
directly, and the one searching a collection of those vectors. If each built
its own network, a reader moving between them would meet four networks that
differed in ways no page mentioned. So the arrangement, the starting weights
and the loop live here, and a page that varies one of them says which.

The arrangement
---------------
Two rounds of a three-by-three convolution with one pixel of padding, each
followed by a two-by-two pooling that halves the picture, then a flattening,
then a dense hidden layer, then the answer::

    (1, 16, 16)  convolve  (4, 16, 16)  pool  (4, 8, 8)
                 convolve  (8, 8, 8)    pool  (8, 4, 4)
                 flatten   (128,)
                 dense     (16,)        the vector a picture is given
                 dense     (4,)         one score per kind

The hidden layer is there because two pages need a small vector for a
picture, and the layer before the answer is exactly that. Sixteen numbers is
few enough to draw after a projection and enough for four kinds to separate.

The loop
--------
Plain descent on shuffled batches, the loop the training page builds, with
the shuffling drawn from a seed so a run can be repeated exactly. Nothing here
is tuned per page; a page that wants a different rate or length passes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from oop_ml import (
    Conv2d,
    DenseLayer,
    Flatten,
    HyperbolicTangent,
    Identity,
    LayerStack,
    MaxPool2d,
    Neuron,
    RectifiedLinear,
    SoftmaxCrossEntropy,
)

from oop_ml_api.picture_collection import (
    DEFAULT_SEED,
    SIDE,
    PictureCollection,
    drawn_collection,
)

DEFAULT_FILTERS = (4, 8)
DEFAULT_HIDDEN = 16
DEFAULT_EPOCHS = 40
DEFAULT_RATE = 0.1
DEFAULT_BATCH = 16
REFERENCE_PER_KIND = 120
"""Pictures of each kind in the reference collection, before it is halved.

The first half is for training and the second is held out, which the
interleaved drawing makes a plain split at the middle, so the network learns
from 60 of each kind and is scored on 60 it never saw.

Chosen by measurement over three weight seeds. At 40 of each kind for
training, a rate of 0.05 and 40 epochs, the held-out accuracy was 0.77 to
0.85 with the loss still falling; at 60 of each, a rate of 0.1 and 40 epochs,
it was 0.92 to 0.97, in about 3 s, and doubling the epochs beyond that bought
nothing measurable on the held-out half.
"""


@dataclass(frozen=True)
class NetworkDesign:
    """What a page may vary about the network, and nothing else."""

    n_filters: tuple[int, int] = DEFAULT_FILTERS
    hidden_width: int = DEFAULT_HIDDEN
    with_pooling: bool = True
    weight_seed: int = 0


REFERENCE_DESIGN = NetworkDesign()
"""The design every picture page means when it says "the network"."""


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    loss: float
    training_accuracy: float


@dataclass(frozen=True)
class TrainedNetwork:
    """A trained stack, how it got there, and the design it was built from."""

    stack: LayerStack
    design: NetworkDesign
    history: tuple[EpochRecord, ...] = field(default_factory=tuple)

    @property
    def vector_position(self) -> int:
        """Which layer's answer is the picture's vector: the one before the last."""
        return len(self.stack) - 2


def _dense(
    n_inputs: int, n_outputs: int, activation, generator: np.random.Generator
) -> DenseLayer:
    spread = (2.0 / n_inputs) ** 0.5
    return DenseLayer(
        [
            Neuron(
                weights=generator.normal(0.0, spread, n_inputs),
                bias=0.0,
                activation=activation,
            )
            for _ in range(n_outputs)
        ]
    )


def built_network(design: NetworkDesign, n_kinds: int) -> LayerStack:
    """The arrangement in the module docstring, with starting weights drawn."""
    first, second = design.n_filters
    generator = np.random.default_rng(design.weight_seed)
    layers = []
    side = SIDE
    layers.append(
        Conv2d(
            reads=(1, side, side),
            n_filters=first,
            kernel_size=3,
            activation=RectifiedLinear(),
            padding=1,
            random_seed=design.weight_seed,
        )
    )
    if design.with_pooling:
        layers.append(MaxPool2d(reads=(first, side, side)))
        side //= 2
    layers.append(
        Conv2d(
            reads=(first, side, side),
            n_filters=second,
            kernel_size=3,
            activation=RectifiedLinear(),
            padding=1,
            random_seed=design.weight_seed + 1,
        )
    )
    if design.with_pooling:
        layers.append(MaxPool2d(reads=(second, side, side)))
        side //= 2
    layers.append(Flatten(reads=(second, side, side)))
    flat_width = second * side * side
    layers.append(
        _dense(flat_width, design.hidden_width, HyperbolicTangent(), generator)
    )
    layers.append(_dense(design.hidden_width, n_kinds, Identity(), generator))
    return LayerStack(layers)


def scores_of(stack: LayerStack, pictures: np.ndarray) -> np.ndarray:
    """One raw score per kind for every picture."""
    return np.asarray(stack.respond_to(pictures).outputs, dtype=np.float64)


def accuracy_of(stack: LayerStack, collection: PictureCollection) -> float:
    """The share of pictures whose highest score is their own kind."""
    called = scores_of(stack, collection.pictures).argmax(axis=1)
    return float((called == collection.labels).mean())


def vectors_of(trained: TrainedNetwork, pictures: np.ndarray) -> np.ndarray:
    """``(n, hidden_width)``, each picture's answer at the layer before the last."""
    response = trained.stack.respond_to(pictures)
    return np.asarray(response[trained.vector_position].outputs, dtype=np.float64)


def trained_network(
    collection: PictureCollection,
    design: NetworkDesign = REFERENCE_DESIGN,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_RATE,
    batch_size: int = DEFAULT_BATCH,
    shuffle_seed: int = 0,
) -> TrainedNetwork:
    """Train from the design's starting weights and record every epoch."""
    stack = built_network(design, collection.n_kinds)
    loss = SoftmaxCrossEntropy()
    targets = collection.one_hot()
    generator = np.random.default_rng(shuffle_seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = generator.permutation(len(collection))
        losses = []
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            backward = stack.backward_pass(
                collection.pictures[chosen], targets[chosen], loss
            )
            losses.append(backward.loss * len(chosen))
            stack = stack.stepped_by(backward, learning_rate)
        history.append(
            EpochRecord(
                epoch=epoch,
                loss=float(sum(losses) / len(order)),
                training_accuracy=accuracy_of(stack, collection),
            )
        )
    return TrainedNetwork(stack=stack, design=design, history=tuple(history))


@lru_cache(maxsize=4)
def reference_split(
    seed: int = DEFAULT_SEED,
) -> tuple[PictureCollection, PictureCollection]:
    """The reference collection, cut into a training half and a held-out half."""
    collection = drawn_collection(REFERENCE_PER_KIND, seed=seed)
    middle = len(collection) // 2
    return (
        collection.subset(np.arange(middle)),
        collection.subset(np.arange(middle, len(collection))),
    )


@lru_cache(maxsize=4)
def reference_network(seed: int = DEFAULT_SEED) -> TrainedNetwork:
    """The network every picture page shares, trained once per process."""
    training, _ = reference_split(seed)
    return trained_network(training)
