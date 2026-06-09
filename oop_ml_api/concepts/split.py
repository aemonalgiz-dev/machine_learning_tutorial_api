"""The seeded hold-out split the evaluation and penalty pages share.

Both pages hold back the same share of the same points, and both need to
know *which* positions were held out so a widget can ring them, which is why
this is by index rather than a ``DataSplit`` of rows. It lives on its own so
the penalty router can import it without importing the whole evaluation
router.
"""

from __future__ import annotations

import numpy as np

HELD_OUT_FRACTION = 0.3
"""The share of the points kept back from the fit."""

SPLIT_SEED = 4
"""The shuffle is seeded so the pages can quote the numbers it produces."""


def held_out_indices(n_points: int) -> np.ndarray:
    """The seeded shuffle-and-cut, by position, sorted ascending."""
    generator = np.random.default_rng(SPLIT_SEED)
    shuffled = generator.permutation(n_points)
    n_held = max(1, int(n_points * HELD_OUT_FRACTION))
    return np.sort(shuffled[:n_held])
