"""The principal component analysis page's further endpoints.

The analyze and variance-sweep endpoints in ``unsupervised`` report what the
library's fit found. These report the road there: what the cloud looks like
along a direction the reader chose, which is the arithmetic the fitted
components are the best case of; what changes when a feature's unit
changes; how the first component differs from the two regression lines on
the same cloud; and a five-measurement dataset where keeping fewer
components is a real choice rather than a choice between one and two.

The projection onto an arbitrary direction is computed here rather than in
the library, because the library has no reason to project onto a direction
that is not a component. Everything that is a fit is the library's.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from fastapi import APIRouter
from oop_ml import Feature, PrincipalComponentAnalysis, SimpleLinearRegression
from oop_ml.numpy.preprocessing.standardization.standardizer import Standardizer
from pydantic import BaseModel, Field

from oop_ml_api.concepts.unsupervised import PlainPoint
from oop_ml_api.limits import MAX_POINTS

router = APIRouter(tags=["unsupervised"])


class Arrow(BaseModel):
    """A direction in the height-weight plane."""

    dx: float
    dy: float


class CovarianceEntries(BaseModel):
    """The three numbers a two-feature covariance matrix holds."""

    var_x: float
    cov: float
    var_y: float


class DirectionRequest(BaseModel):
    """The cloud, the direction to project it onto, and two switches.

    ``centre`` off projects the raw coordinates rather than the deviations,
    which is the mistake the page shows. ``length`` scales the direction
    vector, which is the other mistake the page shows: a longer arrow makes
    every score larger without changing what the direction is.
    """

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)
    angle_degrees: float = Field(ge=0.0, le=360.0)
    centre: bool = True
    length: float = Field(default=1.0, ge=0.1, le=3.0)

    model_config = {"extra": "forbid"}


class ProjectedRow(BaseModel):
    """One person seen along the chosen direction."""

    x: float
    y: float
    deviation_x: float
    deviation_y: float
    contribution_x: float
    contribution_y: float
    score: float
    unit_score: float
    foot: PlainPoint
    squared_residual: float


class DirectionResponse(BaseModel):
    """The cloud along one direction, and what the covariance matrix does to it.

    ``score`` uses the vector as given, ``unit_score`` the unit direction, so
    the two agree at length one and part elsewhere. ``score_variance`` is
    the variance of the scores about their own mean and
    ``score_second_moment`` the mean of their squares; the two agree when
    the cloud was centred and not otherwise. ``transformed`` is the
    population covariance matrix applied to the unit direction, and
    ``angle_between_degrees`` how far it turned.
    """

    pivot: PlainPoint
    unit: Arrow
    vector: Arrow
    rows: list[ProjectedRow]
    score_mean: float
    score_variance: float
    score_second_moment: float
    unit_score_variance: float
    total_squared_residual: float
    retained_sum_squares: float
    total_sum_squares: float
    covariance: CovarianceEntries
    transformed: Arrow
    transformed_length: float
    angle_between_degrees: float


def _matrix(points: list[PlainPoint]) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in points], dtype=np.float64)


@router.post("/concepts/pca/direction", response_model=DirectionResponse)
def direction(request: DirectionRequest) -> DirectionResponse:
    """Project every person onto the chosen direction and measure the spread."""
    matrix = _matrix(request.points)
    mean = matrix.mean(axis=0)
    pivot = mean if request.centre else np.zeros(2)
    deviations = matrix - pivot

    radians = np.deg2rad(request.angle_degrees)
    unit = np.array([np.cos(radians), np.sin(radians)])
    vector = request.length * unit

    unit_scores = deviations @ unit
    scores = deviations @ vector
    contributions = deviations * vector
    feet = pivot + unit_scores[:, None] * unit
    squared_lengths = np.sum(deviations**2, axis=1)
    squared_residuals = squared_lengths - unit_scores**2

    centred = matrix - mean
    covariance = centred.T @ centred / len(request.points)
    transformed = covariance @ unit
    transformed_length = float(np.linalg.norm(transformed))
    cosine = (
        float(transformed @ unit / transformed_length)
        if transformed_length > 0
        else 1.0
    )
    angle_between = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    rows = [
        ProjectedRow(
            x=float(matrix[index, 0]),
            y=float(matrix[index, 1]),
            deviation_x=float(deviations[index, 0]),
            deviation_y=float(deviations[index, 1]),
            contribution_x=float(contributions[index, 0]),
            contribution_y=float(contributions[index, 1]),
            score=float(scores[index]),
            unit_score=float(unit_scores[index]),
            foot=PlainPoint(x=float(feet[index, 0]), y=float(feet[index, 1])),
            squared_residual=float(max(squared_residuals[index], 0.0)),
        )
        for index in range(len(request.points))
    ]
    return DirectionResponse(
        pivot=PlainPoint(x=float(pivot[0]), y=float(pivot[1])),
        unit=Arrow(dx=float(unit[0]), dy=float(unit[1])),
        vector=Arrow(dx=float(vector[0]), dy=float(vector[1])),
        rows=rows,
        score_mean=float(scores.mean()),
        score_variance=float(scores.var()),
        score_second_moment=float(np.mean(scores**2)),
        unit_score_variance=float(unit_scores.var()),
        total_squared_residual=float(np.sum(np.maximum(squared_residuals, 0.0))),
        retained_sum_squares=float(np.sum(unit_scores**2)),
        total_sum_squares=float(np.sum(squared_lengths)),
        covariance=CovarianceEntries(
            var_x=float(covariance[0, 0]),
            cov=float(covariance[0, 1]),
            var_y=float(covariance[1, 1]),
        ),
        transformed=Arrow(dx=float(transformed[0]), dy=float(transformed[1])),
        transformed_length=transformed_length,
        angle_between_degrees=angle_between,
    )


class PointsRequest(BaseModel):
    """The cloud, and nothing else."""

    points: list[PlainPoint] = Field(min_length=2, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class ComponentReport(BaseModel):
    """One fitted direction, its variance and its share of the total."""

    dx: float
    dy: float
    variance: float
    share: float


def _component_reports(fitted: PrincipalComponentAnalysis) -> list[ComponentReport]:
    shares = fitted.components.variance_shares
    return [
        ComponentReport(
            dx=float(component.direction[0]),
            dy=float(component.direction[1]),
            variance=float(component.variance),
            share=float(share),
        )
        for component, share in zip(fitted.components, shares, strict=True)
    ]


UNIT_FACTORS = (("metres", 0.01), ("centimetres", 1.0), ("millimetres", 10.0))
"""Height rewritten in three units, each a factor on the same numbers."""


class UnitVariant(BaseModel):
    """The same people with height in one unit, and what the fit makes of them."""

    unit: str
    factor: float
    covariance: list[list[float]]
    components: list[ComponentReport]


class UnitScalingResponse(BaseModel):
    """The fit repeated with height in metres, centimetres and millimetres."""

    variants: list[UnitVariant]


@router.post("/concepts/pca/unit-scaling", response_model=UnitScalingResponse)
def unit_scaling(request: PointsRequest) -> UnitScalingResponse:
    """Refit the cloud with height in each unit and report what moved."""
    matrix = _matrix(request.points)
    variants: list[UnitVariant] = []
    for unit, factor in UNIT_FACTORS:
        heights = Feature("height", matrix[:, 0] * factor)
        weights = Feature("weight", matrix[:, 1])
        fitted = PrincipalComponentAnalysis().fit([heights, weights])
        scaled = np.column_stack([matrix[:, 0] * factor, matrix[:, 1]])
        centred = scaled - scaled.mean(axis=0)
        covariance = centred.T @ centred / len(request.points)
        variants.append(
            UnitVariant(
                unit=unit,
                factor=factor,
                covariance=[[float(value) for value in row] for row in covariance],
                components=_component_reports(fitted),
            )
        )
    return UnitScalingResponse(variants=variants)


class FittedLine(BaseModel):
    """A regression line and the total of its squared misses."""

    slope: float
    intercept: float
    residual_sum: float


class VersusRegressionResponse(BaseModel):
    """The two regression lines and the first component, on one cloud.

    ``weight_on_height`` predicts weight from height and misses vertically;
    ``height_on_weight`` predicts height from weight, written as
    ``height = intercept + slope * weight``, and misses horizontally; the
    first component misses perpendicularly, and its total is the one-component
    reconstruction error.
    """

    mean: PlainPoint
    weight_on_height: FittedLine
    height_on_weight: FittedLine
    first_component: Arrow
    perpendicular_residual_sum: float


@router.post("/concepts/pca/versus-regression", response_model=VersusRegressionResponse)
def versus_regression(request: PointsRequest) -> VersusRegressionResponse:
    """Fit both regressions and the first component to the same cloud."""
    matrix = _matrix(request.points)
    heights, weights = matrix[:, 0], matrix[:, 1]

    forward = SimpleLinearRegression().fit(heights, weights)
    forward_residual = float(
        np.sum((weights - np.asarray(forward.predict(heights))) ** 2)
    )
    backward = SimpleLinearRegression().fit(weights, heights)
    backward_residual = float(
        np.sum((heights - np.asarray(backward.predict(weights))) ** 2)
    )

    features = [Feature("height", heights), Feature("weight", weights)]
    fitted = PrincipalComponentAnalysis(n_components=1).fit(features)
    rebuilt = fitted.inverse_transform(fitted.transform(features))
    rebuilt_matrix = np.column_stack(
        [np.asarray(rebuilt[0].values), np.asarray(rebuilt[1].values)]
    )
    perpendicular = float(np.sum((matrix - rebuilt_matrix) ** 2))
    first = next(iter(fitted.components))

    return VersusRegressionResponse(
        mean=PlainPoint(x=float(heights.mean()), y=float(weights.mean())),
        weight_on_height=FittedLine(
            slope=float(forward.slope),
            intercept=float(forward.intercept),
            residual_sum=forward_residual,
        ),
        height_on_weight=FittedLine(
            slope=float(backward.slope),
            intercept=float(backward.intercept),
            residual_sum=backward_residual,
        ),
        first_component=Arrow(
            dx=float(first.direction[0]), dy=float(first.direction[1])
        ),
        perpendicular_residual_sum=perpendicular,
    )


BODY_ROWS = 60
"""How many people the five-measurement dataset holds."""

BODY_SEED = 5
"""The dataset is drawn once, from this seed."""

BODY_NAMES = ("height", "arm_span", "leg_length", "sitting_height", "shoe_size")
"""Five measurements that all follow overall size, each with something of its own."""


class BodyComponent(BaseModel):
    """One of the five components, with its loadings by feature name."""

    name: str
    variance: float
    share: float
    cumulative_share: float
    loadings: dict[str, float]


class BodyMeasurementsResponse(BaseModel):
    """Sixty people, five standardised measurements, and the whole fit.

    The measurements are standardised before the fit, because shoe sizes and
    heights are not in comparable units, so every variance and every
    reconstruction error is in units of standard deviations.
    ``reconstruction_errors[k - 1]`` is the total squared error over every
    person and feature when only the first ``k`` components are kept.
    """

    names: list[str]
    rows: list[list[float]]
    correlation: list[list[float]]
    components: list[BodyComponent]
    reconstruction_errors: list[float]


def _body_features() -> list[Feature]:
    generator = np.random.default_rng(BODY_SEED)
    size = generator.normal(size=BODY_ROWS)
    height = 170.0 + 9.0 * size + generator.normal(scale=2.5, size=BODY_ROWS)
    columns = {
        "height": height,
        "arm_span": height + generator.normal(scale=3.0, size=BODY_ROWS),
        "leg_length": 80.0 + 5.0 * size + generator.normal(scale=2.5, size=BODY_ROWS),
        "sitting_height": 90.0
        + 3.0 * size
        + generator.normal(scale=2.5, size=BODY_ROWS),
        "shoe_size": 42.0 + 1.5 * size + generator.normal(scale=1.2, size=BODY_ROWS),
    }
    return [Feature(name, columns[name]) for name in BODY_NAMES]


@cache
def _body_measurements() -> BodyMeasurementsResponse:
    raw = _body_features()
    standardised = Standardizer().fit(raw).transform(raw)
    matrix = np.column_stack([np.asarray(feature.values) for feature in standardised])

    whole = PrincipalComponentAnalysis().fit(standardised)
    cumulative = whole.components.cumulative_shares
    components = [
        BodyComponent(
            name=component.name,
            variance=float(component.variance),
            share=float(share),
            cumulative_share=float(total),
            loadings={name: float(component.loading_for(name)) for name in BODY_NAMES},
        )
        for component, share, total in zip(
            whole.components, whole.components.variance_shares, cumulative, strict=True
        )
    ]

    errors: list[float] = []
    for kept in range(1, len(BODY_NAMES) + 1):
        fitted = PrincipalComponentAnalysis(n_components=kept).fit(standardised)
        rebuilt = fitted.inverse_transform(fitted.transform(standardised))
        rebuilt_matrix = np.column_stack(
            [np.asarray(feature.values) for feature in rebuilt]
        )
        errors.append(float(np.sum((matrix - rebuilt_matrix) ** 2)))

    raw_matrix = np.column_stack([np.asarray(feature.values) for feature in raw])
    correlation = np.corrcoef(raw_matrix, rowvar=False)
    return BodyMeasurementsResponse(
        names=list(BODY_NAMES),
        rows=[[float(value) for value in row] for row in raw_matrix],
        correlation=[[float(value) for value in row] for row in correlation],
        components=components,
        reconstruction_errors=errors,
    )


@router.get("/concepts/pca/body-measurements", response_model=BodyMeasurementsResponse)
def body_measurements() -> BodyMeasurementsResponse:
    """The five-measurement dataset and every way of keeping fewer of them."""
    return _body_measurements()
