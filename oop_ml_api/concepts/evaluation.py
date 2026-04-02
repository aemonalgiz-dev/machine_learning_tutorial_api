"""Held-out scoring and cross-validation, for the evaluation page.

Every concept page ends on the same sentence, that the honest judge is data
the fit never saw, and this module is that sentence given machinery. One
endpoint fits a polynomial on a fixed seeded split of the points and scores
it twice, once on the rows it saw and once on the rows held back, so the
widget can colour the held-out people and show both numbers. One traces that
pair of scores across every degree, the curve that turns around. One runs the
library's five-fold cross-validation at each degree, so the page can show
every row taking a turn as the judge, and the spread that says how much the
mean deserves to be trusted.

The split is drawn by index here, the same shuffle-and-cut the library's
``TrainTestSplitter`` performs, because the widget needs to know which of the
caller's points were held out, and a ``DataSplit`` hands back the rows
themselves rather than their positions.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    CrossValidation,
    Dataset,
    Feature,
    KFold,
    MultipleLinearRegression,
    PipelineStep,
    PipelineSteps,
    PolynomialFeatures,
    RegressionPipeline,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint, Point
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/evaluation", tags=["evaluation"])

MAX_DEGREE = 9
"""The wildest curve the page lets a reader ask for."""

HELD_OUT_FRACTION = 0.3
"""The share of the points kept back from the fit."""

SPLIT_SEED = 4
"""The shuffle is seeded so the page can quote the numbers it produces."""

FOLD_COUNT = 5
"""How many folds the cross-validation section deals."""

CURVE_SAMPLES = 81


def _held_out_indices(n_points: int) -> np.ndarray:
    """The seeded shuffle-and-cut, by position, smallest indices last."""
    generator = np.random.default_rng(SPLIT_SEED)
    shuffled = generator.permutation(n_points)
    n_held = max(1, int(n_points * HELD_OUT_FRACTION))
    return np.sort(shuffled[:n_held])


def _fit_on_train(
    x_values: np.ndarray, y_values: np.ndarray, held: np.ndarray, degree: int
):
    """Fit the degree on the training rows only, returning the fitted pair."""
    training = np.setdiff1d(np.arange(x_values.size), held)
    train_inputs = Feature("t", x_values[training])
    train_target = Feature("h", y_values[training])

    polynomial = PolynomialFeatures(degree=degree).fit([train_inputs])
    model = MultipleLinearRegression().fit(
        polynomial.transform([train_inputs]), train_target
    )
    return polynomial, model, training


class SplitFitRequest(BaseModel):
    """The points and the degree to fit on the training share alone."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}


class SplitFitResponse(BaseModel):
    """The fit, judged twice.

    ``held_out_indices`` are positions into the request's points, so the
    widget can ring the very dots the fit never saw.
    """

    held_out_indices: list[int]
    train_r_squared: float
    held_out_r_squared: float
    curve: list[CurvePoint]


@router.post("/split-fit", response_model=SplitFitResponse)
def split_fit(request: SplitFitRequest) -> SplitFitResponse:
    """Fit on the training share, then score on both shares."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    held = _held_out_indices(x_values.size)

    n_training = x_values.size - held.size
    if n_training <= request.degree:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a degree {request.degree} curve has {request.degree + 1} "
                f"numbers to pin down and the training share holds only "
                f"{n_training} points"
            ),
        )

    polynomial, model, _ = _fit_on_train(x_values, y_values, held, request.degree)

    train_mask = np.ones(x_values.size, dtype=bool)
    train_mask[held] = False
    train_terms = polynomial.transform([Feature("t", x_values[train_mask])])
    held_terms = polynomial.transform([Feature("t", x_values[held])])

    grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), CURVE_SAMPLES)
    curve_heights = np.asarray(
        model.predict(polynomial.transform([Feature("t", grid)])), dtype=np.float64
    )

    return SplitFitResponse(
        held_out_indices=[int(index) for index in held],
        train_r_squared=float(
            model.score(train_terms, Feature("h", y_values[train_mask]))
        ),
        held_out_r_squared=float(model.score(held_terms, Feature("h", y_values[held]))),
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
    )


class GapCurveRequest(BaseModel):
    """The points whose two score curves are to be traced across degrees."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class GapCurveResponse(BaseModel):
    """One training score and one held-out score per degree.

    The sweep stops where the training share runs out of points to pin the
    next degree down.
    """

    degrees: list[int]
    train_scores: list[float]
    held_out_scores: list[float]


@router.post("/degree-curve", response_model=GapCurveResponse)
def degree_curve(request: GapCurveRequest) -> GapCurveResponse:
    """Trace both scores across every degree the training share can pin."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    held = _held_out_indices(x_values.size)

    train_mask = np.ones(x_values.size, dtype=bool)
    train_mask[held] = False

    degrees: list[int] = []
    train_scores: list[float] = []
    held_out_scores: list[float] = []
    for degree in range(1, MAX_DEGREE + 1):
        if int(np.sum(train_mask)) <= degree:
            break
        polynomial, model, _ = _fit_on_train(x_values, y_values, held, degree)
        train_terms = polynomial.transform([Feature("t", x_values[train_mask])])
        held_terms = polynomial.transform([Feature("t", x_values[held])])
        degrees.append(degree)
        train_scores.append(
            float(model.score(train_terms, Feature("h", y_values[train_mask])))
        )
        held_out_scores.append(
            float(model.score(held_terms, Feature("h", y_values[held])))
        )

    return GapCurveResponse(
        degrees=degrees,
        train_scores=train_scores,
        held_out_scores=held_out_scores,
    )


class ValidationCurveRequest(BaseModel):
    """The points to cross-validate a polynomial on, degree by degree."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DegreeValidation(BaseModel):
    """One degree's five-fold verdict, the mean and how far the folds spread."""

    degree: int
    mean_r_squared: float
    spread: float


class ValidationCurveResponse(BaseModel):
    """The library's cross-validation run at every degree the folds can pin."""

    validations: list[DegreeValidation]


@router.post("/validation-curve", response_model=ValidationCurveResponse)
def validation_curve(request: ValidationCurveRequest) -> ValidationCurveResponse:
    """Five folds per degree, every row taking one turn as the judge.

    The candidate at each degree is a pipeline, the term manufacturing and
    the regression refit together inside every fold, so no fold ever sees a
    column built from rows it is about to judge.
    """
    dataset = Dataset(
        [Feature("t", [point.x for point in request.points])],
        Feature("h", [point.y for point in request.points]),
    )

    smallest_training = dataset.n_samples - dataset.n_samples // FOLD_COUNT - 1
    validations: list[DegreeValidation] = []
    for degree in range(1, MAX_DEGREE + 1):
        if smallest_training <= degree:
            break
        candidate = RegressionPipeline(
            steps=PipelineSteps(
                [
                    PipelineStep(
                        name="terms", transformer=PolynomialFeatures(degree=degree)
                    )
                ]
            ),
            model=MultipleLinearRegression(),
        )
        result = CrossValidation(
            folds=KFold(n_folds=FOLD_COUNT, random_seed=SPLIT_SEED)
        ).evaluate(candidate, dataset)
        validations.append(
            DegreeValidation(
                degree=degree,
                mean_r_squared=float(result.mean_r2_score),
                spread=float(result.r2_score_spread),
            )
        )

    return ValidationCurveResponse(validations=validations)
