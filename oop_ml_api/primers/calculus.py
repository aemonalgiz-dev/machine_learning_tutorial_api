"""Gradient descent on a small fixed function, for the calculus primer widget.

Not one of the library's models, and not in ``oop_ml``. This exists only to draw
the calculus primer's picture: given one of a couple of preset functions, a
starting point and a learning rate, it walks gradient descent down the curve and
hands back the whole trajectory. Everything the widget teaches is in that
trajectory. The derivative is the slope of the curve, the walk steps against the
slope, the slope reaches zero at the bottom, and a learning rate that is too
large makes each step overshoot so the walk climbs the far wall and runs away.

The functions are plain polynomials, so their derivatives are exact rather than
estimated, and the walk is short and bounded so a diverging run is something the
widget can show rather than an error it has to handle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/primers/calculus", tags=["calculus-primer"])

MAX_STEPS = 100
"""The most descent steps a single request may take, converged or not."""

MAX_LEARNING_RATE = 5.0
"""An upper bound on the step size, so a request cannot ask for a huge walk."""

CURVE_SAMPLES = 160
"""How many points the returned curve is drawn from across its window."""

CONVERGENCE_STEP = 1e-3
"""Stop once a step moves the point less than this: the slope has gone flat."""

FunctionKey = Literal["bowl", "valley"]


class PresetFunction:
    """One fixed function to descend on, with the window it is drawn in.

    The coefficients are in ascending order, so ``(2.5, -1.0, 0.5)`` is
    ``0.5 x^2 - x + 2.5``. The derivative is taken once, exactly, and reused.
    """

    __slots__ = ("_slope", "_value", "key", "name", "x_max", "x_min")

    def __init__(
        self,
        key: FunctionKey,
        name: str,
        coefficients: tuple[float, ...],
        x_min: float,
        x_max: float,
    ) -> None:
        self.key = key
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self._value = np.polynomial.Polynomial(coefficients)
        self._slope = self._value.deriv()

    def value(self, point: float) -> float:
        """The height of the curve at ``point``."""
        return float(self._value(point))

    def slope(self, point: float) -> float:
        """The derivative at ``point``: the slope of the curve there."""
        return float(self._slope(point))


PRESETS: dict[str, PresetFunction] = {
    # A single convex bowl. One minimum, no way to get stuck: the clean case
    # where descent always arrives if the step is not too large.
    "bowl": PresetFunction("bowl", "One bowl", (2.5, -1.0, 0.5), -4.0, 6.0),
    # Two valleys with a hill between them. Which valley the walk falls into
    # depends on which side of the hill it starts, so the starting point matters.
    "valley": PresetFunction(
        "valley", "Two valleys", (4.0, 0.1, -0.8, 0.0, 0.1), -4.0, 4.0
    ),
}


class DescendRequest(BaseModel):
    """A run of gradient descent to draw."""

    function: FunctionKey
    start: float
    learning_rate: float = Field(gt=0.0, le=MAX_LEARNING_RATE)
    steps: int = Field(default=MAX_STEPS, gt=0, le=MAX_STEPS)


class CurvePoint(BaseModel):
    """A point on the drawn curve."""

    x: float
    y: float


class DescentPoint(BaseModel):
    """A point the walk stood on, and the slope it saw there."""

    x: float
    y: float
    slope: float


class Window(BaseModel):
    """The drawing window: the x-range of the function and the y-range it spans."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class DescendResponse(BaseModel):
    """Everything the widget needs to draw the curve and the walk down it."""

    curve: list[CurvePoint]
    window: Window
    path: list[DescentPoint]
    outcome: Literal["converged", "step_limit_reached", "diverged"]
    minimum: CurvePoint | None


def _sample_curve(function: PresetFunction) -> tuple[list[CurvePoint], Window]:
    """Sample the curve across its window, and measure the height it spans."""
    xs = np.linspace(function.x_min, function.x_max, CURVE_SAMPLES)
    ys = [function.value(float(x)) for x in xs]

    lowest, highest = min(ys), max(ys)
    padding = 0.08 * (highest - lowest or 1.0)

    curve = [CurvePoint(x=float(x), y=y) for x, y in zip(xs, ys, strict=True)]
    window = Window(
        x_min=function.x_min,
        x_max=function.x_max,
        y_min=lowest - padding,
        y_max=highest + padding,
    )
    return curve, window


def _descend(
    function: PresetFunction, start: float, learning_rate: float, steps: int
) -> tuple[list[DescentPoint], str]:
    """Walk gradient descent from ``start`` and record every point it stood on.

    Each step moves against the slope, scaled by the learning rate. The walk
    stops when a step barely moves (the slope has flattened, so it converged),
    when it runs out of steps, or when it leaves a generous window around the
    function, which is what a step too large for the curve does.
    """
    width = function.x_max - function.x_min
    lower, upper = function.x_min - width, function.x_max + width

    trajectory: list[DescentPoint] = []
    point = start

    for _ in range(steps):
        slope = function.slope(point)
        trajectory.append(DescentPoint(x=point, y=function.value(point), slope=slope))

        step = -learning_rate * slope
        if abs(step) < CONVERGENCE_STEP:
            return trajectory, "converged"

        point = point + step
        if not np.isfinite(point) or point < lower or point > upper:
            return trajectory, "diverged"

    return trajectory, "step_limit_reached"


@router.post("/descend", response_model=DescendResponse)
def descend(request: DescendRequest) -> DescendResponse:
    """Draw a curve and walk gradient descent down it from the given start.

    A synchronous endpoint, like the concept endpoints: the work is a short numpy
    loop. The start is clamped into the function's own window so the walk always
    begins on the drawn curve.
    """
    function = PRESETS.get(request.function)
    if function is None:  # pragma: no cover - Literal makes this unreachable
        raise HTTPException(status_code=422, detail="unknown function")

    start = float(np.clip(request.start, function.x_min, function.x_max))

    curve, window = _sample_curve(function)
    path, outcome = _descend(function, start, request.learning_rate, request.steps)

    minimum = None
    if outcome == "converged" and path:
        landed = path[-1]
        minimum = CurvePoint(x=landed.x, y=landed.y)

    return DescendResponse(
        curve=curve,
        window=window,
        path=path,
        outcome=outcome,  # type: ignore[arg-type]
        minimum=minimum,
    )


class TangentRequest(BaseModel):
    """A point on a curve to read the slope at."""

    function: FunctionKey
    x: float


class CurveSlope(BaseModel):
    """A point on the curve, and the slope of the tangent there."""

    x: float
    y: float
    slope: float


class TangentResponse(BaseModel):
    """The curve to draw, and the one point the tangent is taken at."""

    curve: list[CurvePoint]
    window: Window
    point: CurveSlope


@router.post("/tangent", response_model=TangentResponse)
def tangent(request: TangentRequest) -> TangentResponse:
    """The slope of a curve at a single point, for the tangent explorer.

    The step before descent: no walk, just the derivative at one place. Drag the
    point and the slope changes, negative where the curve falls, positive where
    it rises, and zero where it turns around. The x is clamped onto the drawn
    curve so the point cannot be dragged off it.
    """
    function = PRESETS.get(request.function)
    if function is None:  # pragma: no cover - Literal makes this unreachable
        raise HTTPException(status_code=422, detail="unknown function")

    x = float(np.clip(request.x, function.x_min, function.x_max))
    curve, window = _sample_curve(function)
    point = CurveSlope(x=x, y=function.value(x), slope=function.slope(x))

    return TangentResponse(curve=curve, window=window, point=point)
