"""Fitting by walking, for the gradient-descent-regression page.

The line page solved for its slope and intercept in one algebraic step. This
module reaches the same line by walking, and hands the page every step of the
walk rather than only where it ended, which is what the library's observed
route, ``solver_path``, exists for. One call records one walk, and the widget
scrubs through the record. Nothing is refitted under a growing budget.

The heights are measured from their mean before the library sees them, and
that is a modelling decision rather than a convenience. On raw heights near
170 the loss bowl is a valley tens of thousands of times longer than it is
wide, because the intercept and the slope pull against each other, so a step
small enough to be safe across the valley crawls along it and no rate reaches
the line inside the page's budget. Centring makes the two directions
independent, one for the level of the line at the average height and one for
its slope, and the walk becomes two one-dimensional descents side by side.
The endpoint reports the divergence threshold both ways, centred and raw, so
the page can state that claim as a pair of numbers.

A walk whose rate sits above the threshold is recorded only up to the pass
where the coefficients could overflow, decided in closed form before the walk
starts, so the recorded route never provokes the library's ``DivergenceError``
and the page can draw the steps that run away. The library's own verdict at
the requested rate is reported beside the record, from one plain fit at the
full budget, and that is where the refusal is allowed to happen and be named.
The rate sweep repeats that plain fit at several rates and nothing more.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    DesignMatrix,
    DivergenceError,
    Feature,
    GradientDescentRegression,
    MultipleLinearRegression,
    RegressionEvaluation,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["gradient-descent-regression"])

MAX_EPOCHS = 500
"""The most passes one walk may take, so a request stays a teaching request."""

MAX_LEARNING_RATE = 1.0
"""The largest rate accepted. Far above every threshold the page meets."""

MAX_SWEEP_RATES = 12
"""How many rates one sweep may ask the library to fit at."""

RUNAWAY_MAGNITUDE = 1e12
"""A runaway walk is recorded until its coefficients could reach this size.

Well inside what a float64 holds, so the loss at the last recorded pass is
still a finite number the page can draw, and far enough out that the
running away is unmistakable.
"""

CENTRED_HEIGHT = "height_from_mean"
"""The name of the one predictor, the height measured from the mean height."""

WEIGHT = "weight"
"""The name of the target."""

WalkOutcome = Literal["converged", "pass_limit_reached", "diverged"]
FitVerdict = Literal["converged", "pass_limit_reached", "refused"]


class WalkRequest(BaseModel):
    """The people, the step size, and how many passes the walk may take."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    learning_rate: float = Field(gt=0.0, le=MAX_LEARNING_RATE)
    max_epochs: int = Field(ge=1, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class LinePosition(BaseModel):
    """Where a line stands, and the loss it earns there.

    ``level`` is the weight the line predicts at the average height, which is
    the intercept the library learns on centred heights. ``intercept`` is the
    same line's weight at a height of zero, the number the line page reports,
    and it is ``level - slope * mean_height``. ``loss`` is the mean squared
    residual, the quantity whose gradient the walk follows.
    """

    slope: float
    level: float
    intercept: float
    loss: float


class WalkPass(BaseModel):
    """One pass of the walk, the gradient it read, and where it landed.

    The gradient is the one the library stepped against at the start of the
    pass, and ``movement`` is the largest change any coefficient made, the
    number the library's convergence test reads.
    """

    pass_number: int
    gradient_level: float
    gradient_slope: float
    slope: float
    level: float
    intercept: float
    loss: float
    movement: float


class LibraryVerdict(BaseModel):
    """What the library answered when asked for a plain fit at one rate.

    A refused fit carries the library's error by name and its message in the
    library's own words, and no numbers, because there were none to report.
    A fit that ran out of passes carries whatever numbers it stopped on, and
    ``converged`` on the model is the attribute that says not to trust them.
    """

    learning_rate: float
    verdict: FitVerdict
    epochs_run: int | None
    slope: float | None
    level: float | None
    error_factor_per_pass: float
    error_name: str | None
    message: str | None


class WalkResponse(BaseModel):
    """One recorded walk, the line it was walking toward, and the verdicts.

    ``passes`` holds every pass the walk took, in order, and ``start`` is
    where it stood before the first, a flat line at zero. ``outcome`` is
    ``diverged`` when the rate sits above ``divergence_threshold``, in which
    case the record stops before the coefficients could overflow and
    ``passes_run`` is shorter than asked.

    ``level_factor_per_pass`` and ``slope_factor_per_pass`` are what one pass
    multiplies the remaining error by in each direction, ``1 - rate * 2`` for
    the level and ``1 - rate * 2 * variance`` for the slope. A walk settles
    when both sit strictly inside minus one and one, and
    ``error_factor_per_pass`` is the larger of their sizes.
    """

    mean_height: float
    start: LinePosition
    passes: list[WalkPass]
    closed_form: LinePosition
    outcome: WalkOutcome
    passes_run: int
    tolerance: float
    divergence_threshold: float
    divergence_threshold_uncentred: float
    level_factor_per_pass: float
    slope_factor_per_pass: float
    error_factor_per_pass: float
    library_verdict: LibraryVerdict


class PreparedPeople:
    """The people as the library will see them, heights measured from the mean.

    Built once per request and handed to the closed form, the recorded walk
    and the plain fit alike, so the three cannot disagree about what was
    fitted.
    """

    def __init__(self, points: list[PlainPoint]) -> None:
        heights = np.asarray([point.x for point in points], dtype=np.float64)
        weights = np.asarray([point.y for point in points], dtype=np.float64)

        self.mean_height = float(np.mean(heights))
        self.raw_heights = heights
        self.centred_heights = heights - self.mean_height
        self.weights = weights

        row_count = len(points)
        values = np.empty((row_count, 2), order="F")
        values[:, 0] = 1.0
        values[:, 1] = self.centred_heights
        self.design = DesignMatrix(values, [CENTRED_HEIGHT], True)
        self.target = Feature(WEIGHT, weights).column

    @property
    def height_feature(self) -> Feature:
        """The one predictor the library fits on."""
        return Feature(CENTRED_HEIGHT, self.centred_heights)

    @property
    def weight_feature(self) -> Feature:
        """The target, as it came."""
        return Feature(WEIGHT, self.weights)

    def position(self, level: float, slope: float) -> LinePosition:
        """Describe the line with the given level and slope, and score it."""
        predicted = self.design.values @ np.asarray([level, slope], dtype=np.float64)
        evaluation = RegressionEvaluation(self.target.values, predicted)

        return LinePosition(
            slope=float(slope),
            level=float(level),
            intercept=float(level - slope * self.mean_height),
            loss=float(evaluation.mean_squared_error),
        )

    def closed_form(self) -> LinePosition:
        """The line the walk approaches, solved by the normal equations."""
        model = MultipleLinearRegression().fit(
            [self.height_feature], self.weight_feature
        )

        return self.position(model.intercept, model.coefficients[CENTRED_HEIGHT])

    @property
    def level_curvature(self) -> float:
        """How steeply the loss bends in the level direction, ``(2/n) * n``."""
        return _curvature_of(self.design.values[:, 0])

    @property
    def slope_curvature(self) -> float:
        """How steeply the loss bends in the slope direction, twice the variance."""
        return _curvature_of(self.design.values[:, 1])

    @property
    def divergence_threshold(self) -> float:
        """The rate above which the steeper direction's steps grow, ``2 / L``."""
        return 2.0 / max(self.level_curvature, self.slope_curvature)

    @property
    def divergence_threshold_uncentred(self) -> float:
        """The same threshold had the heights been handed over raw.

        On raw heights the two directions are not independent, so the bowl's
        steepest direction is read off the eigenvalues of ``(2/n) X'X``.
        """
        raw_design = np.column_stack([np.ones_like(self.raw_heights), self.raw_heights])
        row_count = raw_design.shape[0]
        hessian = (2.0 / row_count) * raw_design.T @ raw_design

        return float(2.0 / np.max(np.linalg.eigvalsh(hessian)))

    def factors_at(self, learning_rate: float) -> tuple[float, float]:
        """What one pass multiplies the remaining error by, level then slope.

        Signed, because a factor between minus one and zero is a walk that
        overshoots and comes back, and a factor of exactly minus one is a walk
        that hops between two places forever.
        """
        return (
            1.0 - learning_rate * self.level_curvature,
            1.0 - learning_rate * self.slope_curvature,
        )

    def error_factor_at(self, learning_rate: float) -> float:
        """The larger of the two factors in size, which decides the outcome."""
        level_factor, slope_factor = self.factors_at(learning_rate)

        return max(abs(level_factor), abs(slope_factor))

    def library_verdict(self, learning_rate: float, max_epochs: int) -> LibraryVerdict:
        """Ask the library for a plain fit at this rate and report what it said.

        The efficient route, the same walk the recorded one takes. At a rate
        far enough above the threshold the coefficients overflow inside the
        budget and the library refuses by name. numpy's overflow warnings are
        silenced for the duration, since the overflow is the thing being
        demonstrated and the library already turns it into a typed refusal.
        """
        model = GradientDescentRegression(
            learning_rate=learning_rate, max_epochs=max_epochs
        )
        error_factor = self.error_factor_at(learning_rate)

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                model.fit([self.height_feature], self.weight_feature)
        except DivergenceError as refusal:
            return LibraryVerdict(
                learning_rate=learning_rate,
                verdict="refused",
                epochs_run=None,
                slope=None,
                level=None,
                error_factor_per_pass=error_factor,
                error_name=type(refusal).__name__,
                message=str(refusal),
            )

        return LibraryVerdict(
            learning_rate=learning_rate,
            verdict="converged" if model.converged else "pass_limit_reached",
            epochs_run=int(model.epochs_run),
            slope=float(model.coefficients[CENTRED_HEIGHT]),
            level=float(model.intercept),
            error_factor_per_pass=error_factor,
            error_name=None,
            message=None,
        )


def _curvature_of(design_column: np.ndarray) -> float:
    """The diagonal entry of ``(2/n) X'X`` for one column of the design.

    The whole matrix is diagonal on centred heights, because the ones column
    and the centred column are orthogonal, so the two entries are the bowl's
    curvatures along its two axes.
    """
    row_count = design_column.shape[0]

    return float((2.0 / row_count) * design_column @ design_column)


def _passes_to_record(
    max_epochs: int, error_factor: float, closed_form_size: float
) -> int:
    """How many passes to record before a runaway walk could overflow.

    The walk starts at zero, so its distance from the closed-form answer
    starts at that answer's size and is multiplied by at most
    ``error_factor`` each pass. The record stops where that bound reaches
    ``RUNAWAY_MAGNITUDE``. A settling walk is recorded in full.
    """
    if error_factor <= 1.0:
        return max_epochs

    room = math.log(RUNAWAY_MAGNITUDE / max(1.0, closed_form_size)) / math.log(
        error_factor
    )

    return max(1, min(max_epochs, math.floor(room)))


@router.post("/concepts/gradient-descent-regression/walk", response_model=WalkResponse)
def walk(request: WalkRequest) -> WalkResponse:
    """Record one walk down the loss, and the line it was walking toward.

    The closed form is solved first, because it carries the library's own
    refusals, a column of identical heights among them, and those should
    arrive before any walking is done.
    """
    people = PreparedPeople(request.points)
    closed_form = people.closed_form()

    level_factor, slope_factor = people.factors_at(request.learning_rate)
    error_factor = people.error_factor_at(request.learning_rate)
    passes_to_record = _passes_to_record(
        request.max_epochs,
        error_factor,
        math.hypot(closed_form.level, closed_form.slope),
    )

    model = GradientDescentRegression(
        learning_rate=request.learning_rate, max_epochs=passes_to_record
    )
    path = model.solver_path(people.design, people.target)

    passes = []
    for step in path:
        gradient = -step.step / request.learning_rate
        landed = people.position(
            float(step.weights_after[0]), float(step.weights_after[1])
        )
        passes.append(
            WalkPass(
                pass_number=step.pass_number,
                gradient_level=float(gradient[0]),
                gradient_slope=float(gradient[1]),
                slope=landed.slope,
                level=landed.level,
                intercept=landed.intercept,
                loss=landed.loss,
                movement=float(step.largest_movement),
            )
        )

    if path.converged:
        outcome: WalkOutcome = "converged"
    elif error_factor > 1.0:
        outcome = "diverged"
    else:
        outcome = "pass_limit_reached"

    return WalkResponse(
        mean_height=people.mean_height,
        start=people.position(0.0, 0.0),
        passes=passes,
        closed_form=closed_form,
        outcome=outcome,
        passes_run=path.passes_run,
        tolerance=model.tolerance,
        divergence_threshold=people.divergence_threshold,
        divergence_threshold_uncentred=people.divergence_threshold_uncentred,
        level_factor_per_pass=level_factor,
        slope_factor_per_pass=slope_factor,
        error_factor_per_pass=error_factor,
        library_verdict=people.library_verdict(
            request.learning_rate, request.max_epochs
        ),
    )


class RateSweepRequest(BaseModel):
    """The people, and the rates to ask the library for a plain fit at."""

    points: list[PlainPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    learning_rates: list[Annotated[float, Field(gt=0.0, le=MAX_LEARNING_RATE)]] = Field(
        min_length=1, max_length=MAX_SWEEP_RATES
    )
    max_epochs: int = Field(ge=1, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class RateSweepResponse(BaseModel):
    """One verdict per rate, in the order asked, and the threshold they turn on."""

    divergence_threshold: float
    closed_form: LinePosition
    verdicts: list[LibraryVerdict]


@router.post(
    "/concepts/gradient-descent-regression/rate-sweep",
    response_model=RateSweepResponse,
)
def rate_sweep(request: RateSweepRequest) -> RateSweepResponse:
    """Ask the library for a plain fit at each rate and report what it said."""
    people = PreparedPeople(request.points)
    closed_form = people.closed_form()

    return RateSweepResponse(
        divergence_threshold=people.divergence_threshold,
        closed_form=closed_form,
        verdicts=[
            people.library_verdict(learning_rate, request.max_epochs)
            for learning_rate in request.learning_rates
        ],
    )
