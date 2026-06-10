"""A labelled collection of small pictures, drawn by arithmetic, for every page
past the classical ones.

Why drawn rather than downloaded
--------------------------------
The pages run their fits inside a request, so a network on them has to learn
something in well under a second. A real collection of photographs, or even of
handwritten digits, cannot be learned at that scale, and a page that trained on
a tenth of one would be quoting numbers about a network nobody would build.

So the collection is drawn, and it is drawn so that every page can say exactly
what is in it. There are four kinds of shape, and every picture holds one of
them at a random position, a random size inside a range, a random brightness
against a random background, and a little noise on every pixel. The variation
is the point. A cross that always sat in the same place at the same brightness
could be recognised by one pixel, and the network would learn that pixel rather
than anything worth calling a shape.

* **A cross**, two bars through a common centre.
* **A square outline**, one pixel thick, so it has corners and straight edges
  and almost nothing inside.
* **A filled disc**, whose edge points every way and which has the most lit
  pixels of the four.
* **A diagonal bar**, drawn in either direction, which is the one kind whose
  two versions share no lit pixel at the same position.

A fifth kind, a **ring**, is drawn only when asked for by name. It is the
picture a network has never been shown, which is what a page needs to ask
whether a network's arrangement of pictures says anything about a kind it was
not trained to name.

Every picture carries its mask, which pixels belong to the shape, because one
page asks a network for an answer at every pixel rather than one for the
picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SIDE = 16
"""How many pixels on each side of one picture.

Sixteen halves twice without a remainder, which a network that pools twice and
then grows the picture back needs, and it is small enough that a hundred
pictures train in a fraction of a second.
"""

KINDS = ("cross", "square", "disc", "bar")
"""The kinds a network is trained to name, in label order."""

UNSEEN_KIND = "ring"
"""The kind drawn only on request, as the one a network never saw."""

NOISE = 0.05
"""The spread of the noise added to every pixel."""

DEFAULT_SEED = 2
"""The draw every page uses unless it is showing what a different draw does."""


@dataclass(frozen=True)
class PictureCollection:
    """Pictures, what kind each one is, and which of its pixels are the shape.

    ``pictures`` is ``(n, 1, SIDE, SIDE)``, one channel, which is the
    arrangement a convolution reads. ``labels`` is ``(n,)`` of whole numbers
    indexing ``kind_names``. ``masks`` is ``(n, SIDE, SIDE)`` of booleans.
    """

    pictures: np.ndarray
    labels: np.ndarray
    kind_names: tuple[str, ...]
    masks: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_kinds(self) -> int:
        return len(self.kind_names)

    def one_hot(self) -> np.ndarray:
        """``(n, n_kinds)``, a single one in each picture's kind column."""
        return np.eye(self.n_kinds)[self.labels]

    def kind_of(self, position: int) -> str:
        return self.kind_names[int(self.labels[position])]

    def subset(self, positions: Sequence[int] | np.ndarray) -> PictureCollection:
        """The pictures at ``positions``, in that order, sharing the kind names."""
        chosen = np.asarray(positions, dtype=np.int64)
        return PictureCollection(
            pictures=self.pictures[chosen],
            labels=self.labels[chosen],
            kind_names=self.kind_names,
            masks=self.masks[chosen],
        )


def _cross(generator: np.random.Generator) -> np.ndarray:
    size = int(generator.integers(6, 10))
    thickness = 2
    top, left = generator.integers(1, SIDE - size, size=2)
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    middle = (size - thickness) // 2
    mask[top + middle : top + middle + thickness, left : left + size] = True
    mask[top : top + size, left + middle : left + middle + thickness] = True
    return mask


def _square(generator: np.random.Generator) -> np.ndarray:
    size = int(generator.integers(6, 11))
    top, left = generator.integers(1, SIDE - size, size=2)
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[top, left : left + size] = True
    mask[top + size - 1, left : left + size] = True
    mask[top : top + size, left] = True
    mask[top : top + size, left + size - 1] = True
    return mask


def _disc(generator: np.random.Generator) -> np.ndarray:
    radius = float(generator.uniform(2.6, 4.6))
    reach = int(np.ceil(radius)) + 1
    centre_row, centre_column = generator.uniform(reach, SIDE - reach, size=2)
    rows, columns = np.ogrid[:SIDE, :SIDE]
    return (rows - centre_row) ** 2 + (columns - centre_column) ** 2 <= radius**2


def _bar(generator: np.random.Generator) -> np.ndarray:
    length = int(generator.integers(7, 12))
    top, left = generator.integers(1, SIDE - length, size=2)
    rising = bool(generator.integers(0, 2))
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for step in range(length):
        row = top + (length - 1 - step if rising else step)
        mask[row, left + step : left + step + 2] = True
    return mask


def _ring(generator: np.random.Generator) -> np.ndarray:
    outer = float(generator.uniform(3.8, 5.6))
    reach = int(np.ceil(outer)) + 1
    centre_row, centre_column = generator.uniform(reach, SIDE - reach, size=2)
    rows, columns = np.ogrid[:SIDE, :SIDE]
    distance = np.sqrt((rows - centre_row) ** 2 + (columns - centre_column) ** 2)
    return (distance <= outer) & (distance >= outer - 1.5)


_DRAWERS = {
    "cross": _cross,
    "square": _square,
    "disc": _disc,
    "bar": _bar,
    "ring": _ring,
}


def drawn_collection(
    n_per_kind: int,
    seed: int = DEFAULT_SEED,
    kinds: Sequence[str] = KINDS,
) -> PictureCollection:
    """``n_per_kind`` pictures of each kind, interleaved, drawn under ``seed``.

    Interleaved rather than grouped, kind 0 then kind 1 and so on round again,
    so the first half of a collection holds every kind in equal number and a
    page can split it into a training half and a held-out half without
    shuffling.

    Parameters
    ----------
    n_per_kind:
        How many of each kind, at least one.
    seed:
        The draw. The same seed always gives the same pictures.
    kinds:
        Which kinds, in label order. Any of :data:`KINDS` and
        :data:`UNSEEN_KIND`.
    """
    if n_per_kind < 1:
        raise ValueError(
            f"a collection needs at least one of each kind, got {n_per_kind}"
        )
    unknown = [kind for kind in kinds if kind not in _DRAWERS]
    if unknown:
        raise ValueError(f"no such kind of picture: {unknown}")

    generator = np.random.default_rng(seed)
    pictures = []
    labels = []
    masks = []
    for _ in range(n_per_kind):
        for label, kind in enumerate(kinds):
            mask = _DRAWERS[kind](generator)
            background = float(generator.uniform(0.0, 0.3))
            lift = float(generator.uniform(0.4, 0.7))
            picture = (
                background
                + lift * mask.astype(np.float64)
                + generator.normal(0.0, NOISE, (SIDE, SIDE))
            )
            pictures.append(picture)
            labels.append(label)
            masks.append(mask)

    return PictureCollection(
        pictures=np.stack(pictures)[:, np.newaxis, :, :],
        labels=np.asarray(labels, dtype=np.int64),
        kind_names=tuple(kinds),
        masks=np.stack(masks),
    )


def as_rows(picture: np.ndarray) -> list[list[float]]:
    """One picture as nested lists rounded to three places, for a browser.

    Accepts ``(SIDE, SIDE)`` or ``(1, SIDE, SIDE)``.
    """
    flat = np.asarray(picture, dtype=np.float64).reshape(SIDE, SIDE)
    return [[round(float(value), 3) for value in row] for row in flat]
