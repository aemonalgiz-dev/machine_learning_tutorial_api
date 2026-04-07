"""Hebbian principal components, for the page that reaches PCA by a local rule.

One endpoint. It fits the library's ``HebbianPrincipalComponents`` on the
request's people under a fixed seed, two directions, for the epoch budget the
request sets, and beside it fits ``PrincipalComponentAnalysis`` on the same
people, so the response carries both answers to one question. The page needs
the pair rather than either alone, because its whole claim is that a unit
obeying a rule which reads one row at a time turns toward the very direction an
eigensolver finds, and the number that says so is the angle between them. Each
Hebbian direction comes back as a unit vector with the variance along it and
that variance's share of the total, the eigen twin comes back in the PCA
endpoint's own shape, and the angle between each pair is reported in degrees,
taken modulo a half turn because a direction and its negative are the same
direction and the rule is free to land on either.

The page needs the resting state rather than a walk because the library records
none, and the rate schedule is a fraction of the epoch budget, so a fit stopped
at epoch ten of two hundred is not the two-hundred-epoch fit caught early but a
different fit with a faster decay. The widget says instead that every change is
a fresh fit, and its epochs slider stretches the whole walk.

The rate is set from the cloud rather than fixed here, and the reason is the
trap the library's own docstring names. Oja's rule multiplies each update by
the unit's output, which carries the data's units, and subtracts a term in the
output squared, so a rate that behaves on data of spread one overflows on
centimetres. What keeps the weights bounded is that no single row may push the
vector further out than the subtraction pulls it back, and that is a condition
on the rate times the square of the longest row. So the rate starts at a fixed
fraction over the square of the longest deviation from the mean, and falls a
hundredfold across the walk as the library's default does. A uniform scale like
this leaves every direction and every share exactly where they were, which is
what lets the eigen twin be fitted on the raw people and still be the right
comparison. The starting rate is reported, so the page can quote the one it
works a step of by hand.

The seed fixes both the starting weights and the order the people are presented
in each epoch, so the same people for the same budget answer the same way every
time and the page can quote what the widget shows. Two components rather than a
request field, because the plane has two directions and the page teaches the
deflation that finds the second, which a single unit never exercises.
"""

from __future__ import annotations

import math

from fastapi import APIRouter
from oop_ml import (
    ExponentialDecaySchedule,
    Feature,
    HebbianPrincipalComponents,
    PrincipalComponentAnalysis,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import ComponentDocument, PlainPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["hebbian-pca"])

N_COMPONENTS = 2
"""Both directions of the plane, so the deflation that finds the second is exercised."""

MIN_EPOCHS = 1
"""The shortest walk. One epoch presents every person once at the starting rate."""

MAX_EPOCHS = 500
"""The longest walk the widget may ask for.

A hundred people for five hundred epochs is fifty thousand row updates through
a Python loop, measured at about 1.2 seconds, which is the worst a request can
cost here and why the widget waits for the slider to stop moving before it
asks. The four-person worked example has settled long before two hundred.
"""

FIXED_SEED = 11
"""The seeding is fixed so the page can quote what the widget shows.

Chosen so that on the PCA page's four people the first direction starts a
visible swing away from the diagonal, about twenty-three degrees after one
epoch, and lands on the diagonal with a positive sign, so the learned vector
comes back as the PCA page's own (0.71, 0.71) rather than its negative.
"""

RATE_SCALE = 0.5
"""The starting rate, as a fraction of one over the longest deviation squared.

Oja's subtraction moves the squared length toward one by twice the rate times
the output squared, and the output squared is at most the squared length of the
longest row, so at a half the largest single kick lands exactly on length one
rather than carrying past it. At one it would carry an excess straight through
to the same excess on the other side, and above that a kick grows what it was
meant to remove.
"""

RATE_DECAY = 100.0
"""How far the rate falls across the walk, a hundredfold, as the library's default does."""


class LearnRequest(BaseModel):
    """The people, and how many passes over them the rule is allowed."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class HebbianDirectionDocument(BaseModel):
    """One direction the rule learned, at unit length, and how it came out.

    ``variance`` is the cloud's variance along the unit direction and ``share``
    that variance over the cloud's total, both comparable with the eigen twin's.
    ``length`` is how long the learned weight vector was left by the walk, which
    Oja's rule drives to one and nothing here normalises, so it is the report on
    whether this direction settled.
    """

    dx: float
    dy: float
    variance: float
    share: float
    length: float


class LearnResponse(BaseModel):
    """Both answers to one question, and the angle between them.

    ``hebbian`` and ``eigen`` are in matching order, the first direction of each
    first, and ``angles_degrees`` holds the angle between each pair, from zero
    to ninety because a direction and its negative are the same direction.
    ``epochs_run`` is how many passes the walk took before its weights stopped
    moving or the budget ran out, ``worst_orthogonality`` the largest dot
    product between two learned directions, zero for an eigensolver's answer
    and never quite zero here, and ``starting_rate`` the rate the first epoch
    ran at, which the page works one step of by hand.
    """

    mean: PlainPoint
    hebbian: list[HebbianDirectionDocument]
    eigen: list[ComponentDocument]
    angles_degrees: list[float]
    epochs_run: int
    worst_orthogonality: float
    starting_rate: float


@router.post("/concepts/hebbian-pca/learn", response_model=LearnResponse)
def learn(request: LearnRequest) -> LearnResponse:
    """Learn the cloud's directions one row at a time, and solve for them too."""
    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])

    solved = PrincipalComponentAnalysis().fit([heights, weights])
    eigen_shares = list(solved.components.variance_shares)
    eigen = [
        ComponentDocument(
            dx=float(component.direction[0]),
            dy=float(component.direction[1]),
            variance=float(component.variance),
            share=float(share),
        )
        for component, share in zip(solved.components, eigen_shares, strict=True)
    ]

    centre = _centre_of(request.points)
    schedule = _rate_schedule(request.points, centre)
    learned = HebbianPrincipalComponents(
        n_components=N_COMPONENTS,
        learning_rate=schedule,
        max_epochs=request.max_epochs,
        random_seed=FIXED_SEED,
    ).fit([heights, weights])

    hebbian_shares = list(learned.directions.variance_shares)
    hebbian = [
        HebbianDirectionDocument(
            dx=float(direction.direction[0]),
            dy=float(direction.direction[1]),
            variance=float(direction.variance),
            share=float(share),
            length=float(direction.length),
        )
        for direction, share in zip(learned.directions, hebbian_shares, strict=True)
    ]

    return LearnResponse(
        mean=centre,
        hebbian=hebbian,
        eigen=eigen,
        angles_degrees=[
            _angle_between(learned_direction, eigen_twin)
            for learned_direction, eigen_twin in zip(hebbian, eigen, strict=True)
        ],
        epochs_run=int(learned.epochs_run),
        worst_orthogonality=float(learned.directions.worst_orthogonality),
        starting_rate=float(schedule.start),
    )


def _centre_of(points: list[PlainPoint]) -> PlainPoint:
    """The cloud's mean point, which the rule centres every row on."""
    return PlainPoint(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def _rate_schedule(
    points: list[PlainPoint], centre: PlainPoint
) -> ExponentialDecaySchedule:
    """The rate, a fixed fraction over the longest deviation squared, then falling.

    Called after the eigensolver has accepted the people, so a cloud with no
    spread at all has already been refused by the library and the longest
    deviation here is never zero.
    """
    longest_squared = max(
        (point.x - centre.x) ** 2 + (point.y - centre.y) ** 2 for point in points
    )

    return ExponentialDecaySchedule(
        start=RATE_SCALE / longest_squared,
        end=RATE_SCALE / RATE_DECAY / longest_squared,
    )


def _angle_between(
    learned: HebbianDirectionDocument, eigen_twin: ComponentDocument
) -> float:
    """The angle in degrees between a learned direction and its eigen twin.

    Taken modulo a half turn, so it runs from zero to ninety. The rule may land
    on either sign of the eigenvector, and the eigensolver's own sign is
    arbitrary, so the absolute value of the dot product is the comparison. The
    dot product of two unit vectors is clamped at one before the arccosine,
    because rounding can leave it a last bit over.
    """
    alignment = abs(learned.dx * eigen_twin.dx + learned.dy * eigen_twin.dy)

    return math.degrees(math.acos(min(1.0, alignment)))
