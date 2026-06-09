"""Hebbian principal components, for the page that reaches PCA by a local rule.

Three endpoints, and the reason for each.

``learn`` fits the library's ``HebbianPrincipalComponents`` on the request's
people under a fixed seed, two directions, for the epoch budget the request
sets, and beside it fits ``PrincipalComponentAnalysis`` on the same people, so
the response carries both answers to one question. The page needs the pair
rather than either alone, because its whole claim is that a unit obeying a rule
which reads one row at a time turns toward the very direction an eigensolver
finds, and the number that says so is the angle between them. Each Hebbian
direction comes back as a unit vector with the variance along it and that
variance's share of the total, the eigen twin comes back in the PCA endpoint's
own shape, and the angle between each pair is reported in degrees, taken modulo
a half turn because a direction and its negative are the same direction and
the rule is free to land on either. An optional multiplier on the rate lets the
page show the library refusing a diverged walk by name.

``walk`` records the walk itself, one row at a time, which the library does not.
Its fit reports where the weights came to rest and how many passes it took, and
nothing in between, so a page that wants to show the weight vector turning as
each person is presented has to step the rule itself. For Sanger's rule, which
is the library's, every step here is the library's own update method applied to
the library's own centred rows under the library's own schedule, from the same
seeded start and in the same shuffled order, so the trace lands exactly where
the fit lands and a test says so. For the two rules the library deliberately
does not implement, plain Hebb and Oja's rule on independent units, the
arithmetic is written here, because the page has to show the plain rule's
weights running away and two independent units finding the same direction, and
the library refuses to contain either mistake. The trace is short by
construction, twenty epochs at most, because every step is returned.

``sweeps`` runs the library's fit many times over on one small cloud and reports
what changed: the angle and length against the epoch budget, against a
multiplier on the starting rate, across seeds, and under a constant rate that
never falls. Every fit in it is the library's, a diverged one is caught and
reported as diverged rather than raised, and the cloud is capped at thirty
people because the endpoint runs some three thousand epochs.

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
request field on ``learn``, because the plane has two directions and the page
teaches the deflation that finds the second, which a single unit never
exercises.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    Feature,
    HebbianPrincipalComponents,
    PrincipalComponentAnalysis,
)
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import DivergenceError
from oop_ml.core.schedule import Schedule
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

MAX_TRACE_EPOCHS = 20
"""The longest walk the trace records, since every step of it is returned."""

MAX_SWEEP_POINTS = 30
"""The largest cloud the sweeps run on, since they fit it some thirty times over."""

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

MIN_RATE_MULTIPLIER = 0.01
MAX_RATE_MULTIPLIER = 100.0
"""How far a request may scale the starting rate either way, to show it failing."""

EPOCH_BUDGETS = (1, 2, 3, 5, 10, 20, 50, 100, 200)
"""The budgets the epoch sweep fits, dense where most of the swing happens."""

RATE_MULTIPLIERS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""The multipliers the rate sweep applies to the starting rate."""

SWEEP_SEEDS = (11, 1, 2, 3, 4)
"""The seeds the seed sweep fits from, the page's own first."""

SWEEP_EPOCHS = 200
"""The budget every sweep fit other than the epoch sweep runs for."""

Rule = Literal["hebb", "oja", "sanger"]
"""Which update the trace steps.

``hebb`` is the plain rule, which the library refuses to implement because it
diverges; ``oja`` is Oja's rule on every unit independently, which the library
also refuses because two such units find one direction; ``sanger`` is the
library's own update, Oja's rule plus deflation.
"""

Start = Literal["seeded", "along_height"]
"""Where the trace starts its weights.

``seeded`` draws the library's own random unit vectors from the seed, so the
trace reproduces the fit. ``along_height`` starts every unit at (1, 0), which
is what the page works one step of by hand.
"""


class LearnRequest(BaseModel):
    """The people, how many passes over them the rule is allowed, and the rate.

    ``rate_multiplier`` scales the starting rate the endpoint sets from the
    cloud, and exists so the page can show what a rate too large for the data
    does, which is a diverged walk the library refuses by name.
    """

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_epochs: int = Field(ge=MIN_EPOCHS, le=MAX_EPOCHS)
    rate_multiplier: float = Field(
        default=1.0, ge=MIN_RATE_MULTIPLIER, le=MAX_RATE_MULTIPLIER
    )

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
    moving or the budget ran out, ``converged`` whether it was the former,
    ``worst_orthogonality`` the largest dot product between two learned
    directions, zero for an eigensolver's answer and never quite zero here,
    ``total_variance`` the cloud's variance summed over its two columns,
    ``kept_variance`` the variance along the learned directions added up, which
    can exceed the total by a little because the directions overlap a little,
    and ``starting_rate`` the rate the first epoch ran at, which the page works
    one step of by hand. ``learned_scores`` and ``eigen_scores`` are every
    person's coordinates along the two learned directions and along the two
    eigen directions, one row per person, which is what each model's transform
    hands back and the place the two answers meet.
    """

    mean: PlainPoint
    hebbian: list[HebbianDirectionDocument]
    eigen: list[ComponentDocument]
    angles_degrees: list[float]
    epochs_run: int
    converged: bool
    worst_orthogonality: float
    total_variance: float
    kept_variance: float
    starting_rate: float
    learned_scores: list[list[float]]
    eigen_scores: list[list[float]]


@router.post("/concepts/hebbian-pca/learn", response_model=LearnResponse)
def learn(request: LearnRequest) -> LearnResponse:
    """Learn the cloud's directions one row at a time, and solve for them too."""
    heights, weights = _features_of(request.points)
    solved = PrincipalComponentAnalysis().fit([heights, weights])
    eigen = _eigen_documents(solved)

    centre = _centre_of(request.points)
    schedule = _rate_schedule(request.points, centre, request.rate_multiplier)
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
            _angle_between((learned_direction.dx, learned_direction.dy), eigen_twin)
            for learned_direction, eigen_twin in zip(hebbian, eigen, strict=True)
        ],
        epochs_run=int(learned.epochs_run),
        converged=bool(learned.converged),
        worst_orthogonality=float(learned.directions.worst_orthogonality),
        total_variance=float(learned.directions.total_variance),
        kept_variance=float(learned.directions.kept_variance),
        starting_rate=float(schedule.start),
        learned_scores=_scores_of(learned.transform([heights, weights])),
        eigen_scores=_scores_of(solved.transform([heights, weights])),
    )


class WalkRequest(BaseModel):
    """The people and every choice the trace lets the page vary.

    ``rule`` picks the update, ``n_components`` how many units read each row,
    ``decay`` whether the rate falls across the walk or holds at its start,
    ``centre`` whether the rows are the deviations from the mean or the raw
    measurements, and ``start`` where the weights begin. The defaults are the
    library's fit exactly, one unit, Sanger's rule, decaying rate, centred rows,
    seeded start.
    """

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    rule: Rule = "sanger"
    n_components: int = Field(default=1, ge=1, le=N_COMPONENTS)
    max_epochs: int = Field(default=1, ge=MIN_EPOCHS, le=MAX_TRACE_EPOCHS)
    rate_multiplier: float = Field(
        default=1.0, ge=MIN_RATE_MULTIPLIER, le=MAX_RATE_MULTIPLIER
    )
    decay: bool = True
    centre: bool = True
    start: Start = "seeded"
    random_seed: int = Field(default=FIXED_SEED, ge=0, le=10_000)

    model_config = {"extra": "forbid"}


class Vector(BaseModel):
    """A two-vector in the height-weight plane, unbounded.

    The point models are bounded because they are hand-entered people; a
    residual or an update under plain Hebb is neither and grows without limit.
    """

    x: float
    y: float


class WeightState(BaseModel):
    """One unit's weight vector at one moment of the walk.

    Unnormalised, since the length is the whole point of recording it.
    ``angle_degrees`` is the angle to this unit's eigen twin, modulo a half
    turn, and ``angle_to_mean_degrees`` the angle to the direction from the
    origin to the cloud's mean, which is what an uncentred walk learns.
    """

    dx: float
    dy: float
    length: float
    angle_degrees: float
    angle_to_mean_degrees: float


class WalkStep(BaseModel):
    """One person presented to the units, and what every unit did about it.

    ``row`` is the person's position in the request, ``presented`` the row the
    units read, which is the deviation from the mean when centring is on and
    the raw measurement when it is not. ``outputs`` is what each unit fired,
    ``residuals`` the bracket each unit's update multiplies, the row minus what
    the rule subtracts, and ``updates`` the whole change to each unit's
    weights. ``weights`` is where every unit stood after the step.
    """

    epoch: int
    row: int
    rate: float
    presented: Vector
    outputs: list[float]
    residuals: list[Vector]
    updates: list[Vector]
    weights: list[WeightState]


class EpochSummary(BaseModel):
    """Where every unit stood at the end of one pass over the people.

    ``movement`` is the largest change any weight made across the pass, which
    is the number the library's convergence test reads.
    """

    epoch: int
    rate: float
    weights: list[WeightState]
    movement: float


class WalkResponse(BaseModel):
    """The whole walk, step by step.

    ``initial`` is where the units started, ``steps`` every presentation in
    order, ``epochs`` a summary at the end of each pass. ``diverged_at_epoch``
    names the pass on which a weight stopped being finite, after which nothing
    more is recorded; it is null for a walk that stayed finite.
    ``converged_at_epoch`` names the pass on which the walk would have stopped
    itself by the library's own test, or null if it never did. The eigen twins
    are always those of the centred cloud, since that is the answer the rule is
    meant to reach.
    """

    mean: PlainPoint
    eigen: list[ComponentDocument]
    starting_rate: float
    initial: list[WeightState]
    steps: list[WalkStep]
    epochs: list[EpochSummary]
    diverged_at_epoch: int | None
    converged_at_epoch: int | None
    worst_orthogonality: float | None


@router.post("/concepts/hebbian-pca/walk", response_model=WalkResponse)
def walk(request: WalkRequest) -> WalkResponse:
    """Step the rule over the people one at a time and record every step."""
    heights, weights_feature = _features_of(request.points)
    eigen = _eigen_documents(
        PrincipalComponentAnalysis().fit([heights, weights_feature])
    )
    centre = _centre_of(request.points)

    matrix = np.asarray(
        [[point.x, point.y] for point in request.points], dtype=np.float64
    )
    presented = rows_of(
        matrix - np.array([centre.x, centre.y]) if request.centre else matrix,
        ("height", "weight"),
    )
    # The rate is set from the longest row the units actually read, so an
    # uncentred walk reads positions some fourteen times longer than the
    # deviations and gets a rate some two hundred times smaller, which is what
    # lets it run to its own wrong answer rather than overflowing at once.
    schedule = _rate_schedule_for(
        float(np.max(np.sum(presented.values**2, axis=1))), request.rate_multiplier
    )
    rate_schedule: Schedule = (
        schedule if request.decay else ConstantSchedule(value=schedule.start)
    )

    model = HebbianPrincipalComponents(
        n_components=request.n_components,
        learning_rate=rate_schedule,
        max_epochs=request.max_epochs,
        random_seed=request.random_seed,
    )
    generator = np.random.default_rng(request.random_seed)
    weights = _initial_weights(request, presented, generator)

    describe = _describer(eigen, centre)
    steps: list[WalkStep] = []
    epochs: list[EpochSummary] = []
    diverged_at: int | None = None
    converged_at: int | None = None

    for epoch in range(1, request.max_epochs + 1):
        rate = rate_schedule.value_at(epoch, request.max_epochs)
        before = weights.copy()

        for row in generator.permutation(presented.n_rows):
            current_row = presented.row(int(row))
            outputs = weights @ current_row
            residuals = _residuals(request.rule, weights, current_row, outputs)
            updated = (
                model._updated_weights(weights, current_row, rate)
                if request.rule == "sanger"
                else weights + rate * outputs[:, None] * residuals
            )

            if not np.all(np.isfinite(updated)):
                # The step that overflowed is not recorded, so nothing here
                # ever carries an infinity into the response.
                diverged_at = epoch
                break

            steps.append(
                WalkStep(
                    epoch=epoch,
                    row=int(row),
                    rate=rate,
                    presented=_vector(current_row),
                    outputs=[float(output) for output in outputs],
                    residuals=[_vector(vector) for vector in residuals],
                    updates=[_vector(vector) for vector in updated - weights],
                    weights=describe(updated),
                )
            )
            weights = updated

        if diverged_at is not None:
            break

        movement = float(np.max(np.abs(weights - before)))
        epochs.append(
            EpochSummary(
                epoch=epoch, rate=rate, weights=describe(weights), movement=movement
            )
        )
        if converged_at is None and movement < model.tolerance:
            converged_at = epoch

    return WalkResponse(
        mean=centre,
        eigen=eigen,
        starting_rate=float(schedule.start),
        initial=describe(_initial_weights(request, presented, None)),
        steps=steps,
        epochs=epochs,
        diverged_at_epoch=diverged_at,
        converged_at_epoch=converged_at,
        worst_orthogonality=_worst_orthogonality(weights),
    )


class SweepsRequest(BaseModel):
    """A small cloud, fitted many times over."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_SWEEP_POINTS)

    model_config = {"extra": "forbid"}


class BudgetFit(BaseModel):
    """One fit of the library's model for one epoch budget."""

    max_epochs: int
    epochs_run: int
    converged: bool
    angles_degrees: list[float]
    lengths: list[float]
    shares: list[float]


class RateFit(BaseModel):
    """One fit at one multiple of the starting rate.

    A diverged fit is the library refusing by name; its angles and lengths are
    null because there is no answer to report.
    """

    rate_multiplier: float
    starting_rate: float
    diverged: bool
    epochs_run: int | None
    angles_degrees: list[float] | None
    lengths: list[float] | None


class SeedFit(BaseModel):
    """The first direction one seed's walk landed on.

    ``angle_to_first_seed_degrees`` is the angle between this seed's first
    direction and the page's own seed's, so a cloud on which every seed agrees
    reports zeros and a cloud with no preferred direction reports anything.
    """

    seed: int
    dx: float
    dy: float
    length: float
    angle_degrees: float
    angle_to_first_seed_degrees: float
    epochs_run: int


class SweepsResponse(BaseModel):
    """What the library's fit did as one thing at a time was varied.

    ``epochs`` varies the budget under the falling rate, ``constant`` the same
    budgets under a rate held at its starting value, ``rates`` a multiplier on
    the starting rate at a fixed budget, and ``seeds`` the seed at that budget.
    """

    starting_rate: float
    eigen: list[ComponentDocument]
    epochs: list[BudgetFit]
    constant: list[BudgetFit]
    rates: list[RateFit]
    seeds: list[SeedFit]


@router.post("/concepts/hebbian-pca/sweeps", response_model=SweepsResponse)
def sweeps(request: SweepsRequest) -> SweepsResponse:
    """Fit the cloud under every budget, rate, seed and schedule the page quotes."""
    features = list(_features_of(request.points))
    eigen = _eigen_documents(PrincipalComponentAnalysis().fit(features))
    centre = _centre_of(request.points)
    schedule = _rate_schedule(request.points, centre, 1.0)
    constant = ConstantSchedule(value=schedule.start)

    budget_fits = [
        _budget_fit(features, eigen, schedule, budget) for budget in EPOCH_BUDGETS
    ]
    constant_fits = [
        _budget_fit(features, eigen, constant, budget) for budget in EPOCH_BUDGETS
    ]
    rate_fits = [
        _rate_fit(
            features,
            eigen,
            _rate_schedule(request.points, centre, multiplier),
            multiplier,
        )
        for multiplier in RATE_MULTIPLIERS
    ]

    seed_fits: list[SeedFit] = []
    first_seed_direction: tuple[float, float] | None = None
    for seed in SWEEP_SEEDS:
        fitted = HebbianPrincipalComponents(
            n_components=1,
            learning_rate=schedule,
            max_epochs=SWEEP_EPOCHS,
            random_seed=seed,
        ).fit(features)
        direction = next(iter(fitted.directions))
        unit = (float(direction.direction[0]), float(direction.direction[1]))
        if first_seed_direction is None:
            first_seed_direction = unit
        seed_fits.append(
            SeedFit(
                seed=seed,
                dx=unit[0],
                dy=unit[1],
                length=float(direction.length),
                angle_degrees=_angle_between(unit, eigen[0]),
                angle_to_first_seed_degrees=_angle_between_units(
                    unit, first_seed_direction
                ),
                epochs_run=int(fitted.epochs_run),
            )
        )

    return SweepsResponse(
        starting_rate=float(schedule.start),
        eigen=eigen,
        epochs=budget_fits,
        constant=constant_fits,
        rates=rate_fits,
        seeds=seed_fits,
    )


def _budget_fit(
    features: list[Feature],
    eigen: list[ComponentDocument],
    schedule: Schedule,
    budget: int,
) -> BudgetFit:
    """The library's two-direction fit for one budget under one schedule."""
    fitted = HebbianPrincipalComponents(
        n_components=N_COMPONENTS,
        learning_rate=schedule,
        max_epochs=budget,
        random_seed=FIXED_SEED,
    ).fit(features)
    directions = list(fitted.directions)

    return BudgetFit(
        max_epochs=budget,
        epochs_run=int(fitted.epochs_run),
        converged=bool(fitted.converged),
        angles_degrees=[
            _angle_between(
                (float(direction.direction[0]), float(direction.direction[1])), twin
            )
            for direction, twin in zip(directions, eigen, strict=True)
        ],
        lengths=[float(direction.length) for direction in directions],
        shares=[float(share) for share in fitted.directions.variance_shares],
    )


def _rate_fit(
    features: list[Feature],
    eigen: list[ComponentDocument],
    schedule: ExponentialDecaySchedule,
    multiplier: float,
) -> RateFit:
    """The library's fit at one multiple of the starting rate, or its refusal."""
    try:
        fitted = HebbianPrincipalComponents(
            n_components=N_COMPONENTS,
            learning_rate=schedule,
            max_epochs=SWEEP_EPOCHS,
            random_seed=FIXED_SEED,
        ).fit(features)
    except DivergenceError:
        return RateFit(
            rate_multiplier=multiplier,
            starting_rate=float(schedule.start),
            diverged=True,
            epochs_run=None,
            angles_degrees=None,
            lengths=None,
        )

    directions = list(fitted.directions)

    return RateFit(
        rate_multiplier=multiplier,
        starting_rate=float(schedule.start),
        diverged=False,
        epochs_run=int(fitted.epochs_run),
        angles_degrees=[
            _angle_between(
                (float(direction.direction[0]), float(direction.direction[1])), twin
            )
            for direction, twin in zip(directions, eigen, strict=True)
        ],
        lengths=[float(direction.length) for direction in directions],
    )


def _features_of(points: list[PlainPoint]) -> tuple[Feature, Feature]:
    """The people as the two named features every fit here reads."""
    return (
        Feature("height", [point.x for point in points]),
        Feature("weight", [point.y for point in points]),
    )


def _eigen_documents(solved: PrincipalComponentAnalysis) -> list[ComponentDocument]:
    """The eigensolver's two directions, in the PCA endpoint's own shape."""
    return [
        ComponentDocument(
            dx=float(component.direction[0]),
            dy=float(component.direction[1]),
            variance=float(component.variance),
            share=float(share),
        )
        for component, share in zip(
            solved.components, solved.components.variance_shares, strict=True
        )
    ]


def _scores_of(transformed: list[Feature]) -> list[list[float]]:
    """A transform's output features as one row of coordinates per person."""
    columns = [np.asarray(feature.values, dtype=np.float64) for feature in transformed]

    return [[float(value) for value in row] for row in np.column_stack(columns)]


def _centre_of(points: list[PlainPoint]) -> PlainPoint:
    """The cloud's mean point, which the rule centres every row on."""
    return PlainPoint(
        x=sum(point.x for point in points) / len(points),
        y=sum(point.y for point in points) / len(points),
    )


def _rate_schedule(
    points: list[PlainPoint], centre: PlainPoint, multiplier: float
) -> ExponentialDecaySchedule:
    """The rate, a fixed fraction over the longest deviation squared, then falling.

    Called after the eigensolver has accepted the people, so a cloud with no
    spread at all has already been refused by the library and the longest
    deviation here is never zero.
    """
    longest_squared = max(
        (point.x - centre.x) ** 2 + (point.y - centre.y) ** 2 for point in points
    )

    return _rate_schedule_for(longest_squared, multiplier)


def _rate_schedule_for(
    longest_squared: float, multiplier: float
) -> ExponentialDecaySchedule:
    """The schedule for a walk whose longest row has the given squared length."""
    start = multiplier * RATE_SCALE / longest_squared

    return ExponentialDecaySchedule(start=start, end=start / RATE_DECAY)


def _initial_weights(
    request: WalkRequest, presented: RowBlock, generator: np.random.Generator | None
) -> np.ndarray:
    """Where the trace's units start.

    The seeded start draws exactly what the library's fit draws, in the same
    order from the same generator, which is what lets the trace reproduce the
    fit. Handed no generator it draws from a fresh one on the request's seed,
    which is the same numbers again, for the report of where the walk began.
    """
    if request.start == "along_height":
        return np.tile(np.array([1.0, 0.0]), (request.n_components, 1))

    source = (
        generator
        if generator is not None
        else np.random.default_rng(request.random_seed)
    )
    drawn = source.normal(0.0, 1.0, (request.n_components, presented.n_features))

    return drawn / np.linalg.norm(drawn, axis=1, keepdims=True)


def _residuals(
    rule: Rule, weights: np.ndarray, row: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """The bracket each unit's update multiplies, under each rule.

    Plain Hebb subtracts nothing, so the bracket is the row. Oja subtracts the
    unit's own reconstruction of the row. Sanger subtracts every earlier
    unit's reconstruction as well as its own, which is what the library does,
    written out here only so the trace can report it.
    """
    if rule == "hebb":
        return np.tile(row, (weights.shape[0], 1))

    if rule == "oja":
        return row[None, :] - outputs[:, None] * weights

    return np.array(
        [
            row - outputs[: unit + 1] @ weights[: unit + 1]
            for unit in range(weights.shape[0])
        ]
    )


def _describer(
    eigen: list[ComponentDocument], centre: PlainPoint
) -> Callable[[np.ndarray], list[WeightState]]:
    """A function turning a finite weight block into one report per unit."""
    mean_unit = _unit((centre.x, centre.y))

    def describe(weights: np.ndarray) -> list[WeightState]:
        states: list[WeightState] = []
        for unit, vector in enumerate(weights):
            direction = _unit((float(vector[0]), float(vector[1])))
            states.append(
                WeightState(
                    dx=float(vector[0]),
                    dy=float(vector[1]),
                    length=math.hypot(float(vector[0]), float(vector[1])),
                    angle_degrees=_angle_between(direction, eigen[unit]),
                    angle_to_mean_degrees=_angle_between_units(direction, mean_unit),
                )
            )
        return states

    return describe


def _worst_orthogonality(weights: np.ndarray) -> float | None:
    """The largest dot product between two units' unit directions, or null.

    Null for a single unit, which has no pair, and for a walk whose weights
    are no longer finite.
    """
    if weights.shape[0] < 2 or not np.all(np.isfinite(weights)):
        return None

    directions = np.array([_unit((float(row[0]), float(row[1]))) for row in weights])
    products = directions @ directions.T
    np.fill_diagonal(products, 0.0)

    return float(np.max(np.abs(products)))


def _vector(vector: np.ndarray) -> Vector:
    """A two-vector as the unbounded response type."""
    return Vector(x=float(vector[0]), y=float(vector[1]))


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    """A vector scaled to unit length, or the zero vector if it has none."""
    length = math.hypot(vector[0], vector[1])
    if length == 0.0 or not math.isfinite(length):
        return (0.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _angle_between(
    learned: tuple[float, float], eigen_twin: ComponentDocument
) -> float:
    """The angle in degrees between a learned unit direction and its eigen twin.

    Taken modulo a half turn, so it runs from zero to ninety. The rule may land
    on either sign of the eigenvector, and the eigensolver's own sign is
    arbitrary, so the absolute value of the dot product is the comparison. The
    dot product of two unit vectors is clamped at one before the arccosine,
    because rounding can leave it a last bit over.
    """
    return _angle_between_units(learned, (eigen_twin.dx, eigen_twin.dy))


def _angle_between_units(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """The angle in degrees between two unit vectors, modulo a half turn."""
    alignment = abs(first[0] * second[0] + first[1] * second[1])

    return math.degrees(math.acos(min(1.0, alignment)))
