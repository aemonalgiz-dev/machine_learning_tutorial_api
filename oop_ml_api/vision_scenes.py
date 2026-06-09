"""The pictures every classical vision page works on, drawn once here.

Why one scene rather than five
------------------------------
The language pages share one sentence, and they share it so that a reader who
has met it on one page recognises it on the next and can compare what each
method did to the same thing. The vision pages need the same, and a picture
is much easier to invent badly: a scene arranged to flatter one method will
quietly flatter it, and five separately invented scenes cannot be compared at
all.

So there is one workbench scene, and it is drawn by arithmetic rather than
loaded from a file, which means every page can state exactly what is in it.
It is built to contain, deliberately, one of each thing the five methods
disagree about:

* **A square**, which has four corners and four straight edges, so the corner
  measure has something to separate and the aperture problem has somewhere to
  show itself.
* **A disc**, whose edge points in every direction, so an oriented histogram
  has something with no preferred direction to describe.
* **A diagonal bar**, because a diagonal is where the gradient operators
  disagree with each other and where a rotation-sensitive method loses most.
* **The motif, three times**, which is what template matching looks for, and
  the three copies are what makes "the best position" an interesting question
  rather than an obvious one.
* **A brightness ramp across the whole scene**, so that the left copy of the
  motif is darker than the right one. This is the single most important thing
  in the picture. Without it, plain correlation and normalised correlation
  agree and the reason anybody normalises is invisible.

The relit and turned versions are the same scene, so a page comparing them is
comparing a method against itself rather than against a different picture.
"""

from __future__ import annotations

import numpy as np
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.types import FloatArray

SCENE_SIDE = 48
"""How many pixels on each side of the workbench scene.

Small enough that a browser can draw one rectangle per pixel without
struggling, and that a reader can be shown the whole thing at a readable size;
large enough to hold four separated shapes with flat ground between them.
"""

MOTIF_SIDE = 7
"""The side of the small cross that template matching looks for."""

BACKGROUND = 0.12
"""What empty ground reads, before the ramp is added. Not zero, so that a
method reading a ratio has something to divide by, and so that the difference
between empty ground and a dark shape is visible rather than absolute."""

SHAPE_BRIGHTNESS = 0.78
"""What a shape reads, before the ramp."""

RAMP_ACROSS = 0.30
"""How much brighter the right edge of the scene is than the left.

The load-bearing number of this module. It is what makes an unnormalised
score prefer a bright patch of nothing to a dark copy of the thing being
looked for.
"""

SQUARE_AT = (6, 5)
SQUARE_SIDE = 11
DISC_CENTRE = (12, 34)
DISC_RADIUS = 6
BAR_FROM = (30, 6)
BAR_LENGTH = 14
BAR_THICKNESS = 3
MOTIF_AT = ((26, 26), (38, 12), (39, 36))
"""Where each thing sits, as ``(row, column)`` of its top-left pixel.

Written out rather than computed so that a page can quote a position and a
reader can check it against the drawing.
"""


def _blank() -> FloatArray:
    return np.full((SCENE_SIDE, SCENE_SIDE), BACKGROUND, dtype=np.float64)


def _with_square(canvas: FloatArray) -> FloatArray:
    row, column = SQUARE_AT
    canvas[row : row + SQUARE_SIDE, column : column + SQUARE_SIDE] = SHAPE_BRIGHTNESS
    return canvas


def _with_disc(canvas: FloatArray) -> FloatArray:
    centre_row, centre_column = DISC_CENTRE
    rows, columns = np.ogrid[:SCENE_SIDE, :SCENE_SIDE]
    inside = (rows - centre_row) ** 2 + (columns - centre_column) ** 2 <= DISC_RADIUS**2
    canvas[inside] = SHAPE_BRIGHTNESS
    return canvas


def _with_bar(canvas: FloatArray) -> FloatArray:
    row, column = BAR_FROM
    for step in range(BAR_LENGTH):
        canvas[row - step, column + step : column + step + BAR_THICKNESS] = (
            SHAPE_BRIGHTNESS
        )
    return canvas


def motif_values() -> FloatArray:
    """The small cross, on its own, as a bare array.

    A cross rather than a blob because it has corners, straight edges and a
    centre, so the same shape is worth looking for with a template and worth
    describing with a histogram.
    """
    motif = np.full((MOTIF_SIDE, MOTIF_SIDE), BACKGROUND, dtype=np.float64)
    middle = MOTIF_SIDE // 2
    motif[middle - 1 : middle + 2, :] = SHAPE_BRIGHTNESS
    motif[:, middle - 1 : middle + 2] = SHAPE_BRIGHTNESS
    return motif


def _with_motifs(canvas: FloatArray) -> FloatArray:
    motif = motif_values()
    for row, column in MOTIF_AT:
        patch = canvas[row : row + MOTIF_SIDE, column : column + MOTIF_SIDE]
        canvas[row : row + MOTIF_SIDE, column : column + MOTIF_SIDE] = np.maximum(
            patch, motif
        )
    return canvas


def _ramp() -> FloatArray:
    across = np.linspace(0.0, RAMP_ACROSS, SCENE_SIDE, dtype=np.float64)
    return np.tile(across, (SCENE_SIDE, 1))


def workbench(with_ramp: bool = True) -> Picture:
    """The scene every classical vision page works on.

    Parameters
    ----------
    with_ramp:
        Whether the left-to-right brightness ramp is added. A page that wants
        to show what the ramp is responsible for asks for the scene both ways
        and compares, which is the only honest way to attribute an effect to
        it.
    """
    canvas = _with_motifs(_with_bar(_with_disc(_with_square(_blank()))))
    if with_ramp:
        canvas = canvas + _ramp()
    return Picture(canvas)


def motif() -> Picture:
    """The cross being looked for, as a picture of its own."""
    return Picture(motif_values())


def relit(scale: float = 1.6, shift: float = 0.15) -> Picture:
    """The workbench under different lighting, and nothing else changed.

    Every pixel multiplied and then shifted, which is what turning a lamp up
    and adding a second one does to a photograph to a first approximation. A
    description that survives this is doing what it claims; a description that
    moves has been reading the lighting.
    """
    return Picture(np.asarray(workbench()) * scale + shift)


def turned() -> Picture:
    """The workbench turned a quarter circle.

    A quarter turn rather than an arbitrary angle because it is exact: every
    pixel lands on a pixel and nothing is interpolated, so a method that fails
    on it has failed on the rotation itself rather than on the resampling.
    """
    return Picture(np.rot90(np.asarray(workbench())))


def hand_sized() -> Picture:
    """A six by six picture with one edge, for a worked example on paper.

    Dark on the left, bright on the right, so its one edge sits between
    columns two and three and every sweep over it can be checked by hand.
    """
    return Picture(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] for _ in range(6)],
    )


def as_rows(picture: Picture) -> list[list[float]]:
    """A picture as plain nested lists, rounded, for sending to a browser.

    Rounded to four places because a browser draws a grey and cannot show the
    difference beyond that, and an unrounded 48 by 48 scene is a large amount
    of text to send for no gain.
    """
    return [[round(float(value), 4) for value in row] for row in picture.values]
