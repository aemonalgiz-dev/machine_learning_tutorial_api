"""Feature scaling, for the page that puts every column on one footing.

One endpoint, one feature. It fits the library's ``Standardizer`` and its four
affine scalers, ``MinMaxScaler``, ``MaxAbsScaler``, ``RobustScaler`` and
``RootMeanSquareScaler``, on the same list of values, and returns each one's
centre, spread and scaled values keyed by a closed enum. The page needs all five
side by side because the lesson is not any one recipe but the fact that they are
one recipe, ``(value - centre) / spread``, differing only in how the two numbers
are read off the column, and that they disagree most when the column carries an
outlier. The widget drags one value far to the right and watches which rows
squash and which hold still.

Nothing here is seeded, because nothing here is random. A zero spread is left to
the library to refuse, since ``AllSameValuesError`` already becomes a 422
carrying the library's own words.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Feature,
    FeatureScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    RootMeanSquareScaler,
    Standardizer,
)
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["feature-scaling"])

FEATURE_NAME = "value"
"""The one column every request carries, named so the library can address it."""


class ScalingMethod(StrEnum):
    """The five readings of a centre and a spread the page compares."""

    STANDARDIZE = "standardize"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    ROOT_MEAN_SQUARE = "root_mean_square"


AFFINE_SCALERS: dict[ScalingMethod, type[FeatureScaler]] = {
    ScalingMethod.MIN_MAX: MinMaxScaler,
    ScalingMethod.MAX_ABS: MaxAbsScaler,
    ScalingMethod.ROBUST: RobustScaler,
    ScalingMethod.ROOT_MEAN_SQUARE: RootMeanSquareScaler,
}
"""The four methods that share the library's affine base.

``Standardizer`` predates that base and keeps its own vocabulary of a mean and a
standard deviation, so it is fitted separately below and read through its own
names.
"""

BoundedValue = Annotated[float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)]


class ScaleRequest(BaseModel):
    """One feature's raw values, in the order the widget holds them."""

    values: list[BoundedValue] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class MethodScaling(BaseModel):
    """What one method learned, and what it did to every value.

    ``scaled`` is in the input order, so the widget can draw each raw dot's
    scaled twin without re-sorting anything.
    """

    centre: float
    spread: float
    scaled: list[float]


class ScaleResponse(BaseModel):
    """All five methods, keyed by name, fitted on the same column."""

    scalings: dict[ScalingMethod, MethodScaling]


@router.post("/concepts/feature-scaling/scale", response_model=ScaleResponse)
def scale(request: ScaleRequest) -> ScaleResponse:
    """Fit all five scalers on one column and report what each one did."""
    feature = Feature(FEATURE_NAME, request.values)

    standardizer = Standardizer().fit([feature])
    standard_scaling = standardizer.scalings[FEATURE_NAME]
    scalings = {
        ScalingMethod.STANDARDIZE: MethodScaling(
            centre=float(standard_scaling.mean),
            spread=float(standard_scaling.standard_deviation),
            scaled=_scaled_values(standardizer, feature),
        )
    }

    for method, scaler_type in AFFINE_SCALERS.items():
        scaler = scaler_type().fit([feature])
        affine_scaling = scaler.scalings.scaling_for(FEATURE_NAME)
        scalings[method] = MethodScaling(
            centre=float(affine_scaling.centre),
            spread=float(affine_scaling.spread),
            scaled=_scaled_values(scaler, feature),
        )

    return ScaleResponse(scalings=scalings)


def _scaled_values(
    scaler: Standardizer | FeatureScaler, feature: Feature
) -> list[float]:
    """The feature put through a fitted scaler, as plain floats in input order."""
    transformed = scaler.transform([feature])[0]
    return [float(value) for value in np.asarray(transformed.values, dtype=np.float64)]
