"""Bagging, random forests and gradient boosting, for the three ensemble pages.

The two committee endpoints share a shape. Both fit a crowd of the library's
deep trees on the two-feature people, both hand back the shaded regions, the
training accuracy and the out-of-bag score, the honest score bagging gets for
free, and both report which feature each member's first question asked about,
since that census is where bagging and the forest visibly part company. The
seed is fixed so the pages can quote the census. The boosting endpoint fits
the library's ``GradientBoostingRegressor`` on the thrown ball with stump
members, so a reader can watch the fit assemble round by round.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BaggingClassifier,
    Feature,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint, Point
from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/ensembles", tags=["ensembles"])

MAX_MEMBERS = 50
"""The largest committee a request may ask for."""

MAX_ROUNDS = 300
"""The most boosting rounds a request may ask for."""

FIXED_SEED = 7
"""Every committee draws its resamples from this seed, so the pages can quote
which features the members rooted on and the widget reproduces the numbers."""

CURVE_SAMPLES = 81


class CommitteeRequest(BaseModel):
    """The people, and how many trees sit on the committee."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_members: int = Field(ge=1, le=MAX_MEMBERS)

    model_config = {"extra": "forbid"}


class CommitteeResponse(BaseModel):
    """The committee's answer, its two scores, and its members' first questions.

    ``accuracy`` is scored on the people the committee trained on, and
    ``out_of_bag_score`` on the honest route, each person judged only by the
    members whose resample happened to miss them.
    """

    accuracy: float
    out_of_bag_score: float
    roots_on_height: int
    roots_on_weight: int
    regions: RegionGrid


def _committee_response(model, points: list[LabelledPoint]) -> CommitteeResponse:
    heights, weights, labels = features_of(points)
    fitted = model.fit([heights, weights], labels)

    root_features = [
        member.root.split.feature_name
        for member in fitted.members
        if hasattr(member.root, "split")
    ]

    return CommitteeResponse(
        accuracy=float(fitted.score([heights, weights], labels)),
        out_of_bag_score=float(fitted.out_of_bag_score()),
        roots_on_height=root_features.count("height"),
        roots_on_weight=root_features.count("weight"),
        regions=region_grid(fitted, points),
    )


@router.post("/bagging", response_model=CommitteeResponse)
def bagging(request: CommitteeRequest) -> CommitteeResponse:
    """A committee of deep trees, each grown on its own bootstrap resample."""
    return _committee_response(
        BaggingClassifier(n_members=request.n_members, random_seed=FIXED_SEED),
        request.points,
    )


@router.post("/random-forest", response_model=CommitteeResponse)
def random_forest(request: CommitteeRequest) -> CommitteeResponse:
    """The same committee with each split shown only one feature of the two."""
    return _committee_response(
        RandomForestClassifier(
            n_members=request.n_members, random_seed=FIXED_SEED, max_features=1
        ),
        request.points,
    )


class BoostingRequest(BaseModel):
    """The points, how many rounds to run, and how bold each round is.

    The rate is accepted up to 2 so the widget's slider can cross 1 and meet
    the refusal below, though the library itself will not construct a model
    past 1.
    """

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    learning_rate: float = Field(gt=0.0, le=2.0)

    model_config = {"extra": "forbid"}


class BoostingResponse(BaseModel):
    """The assembled fit after the requested rounds, sampled for drawing."""

    r_squared: float
    predictions: list[float]
    curve: list[CurvePoint]


@router.post("/gradient-boosting", response_model=BoostingResponse)
def gradient_boosting(request: BoostingRequest) -> BoostingResponse:
    """Fit stump-by-stump for the requested number of rounds.

    The members are depth-one stumps, the smallest tree there is, so the
    assembly is visible, each round adding one shelf to the curve. A rate past
    1 is refused in words rather than passed to the library, whose constructor
    would reject it anyway, since past 1 each round's correction overshoots
    the very mistake it just fitted, and past 2 the leftover errors grow every
    round instead of shrinking.
    """
    if request.learning_rate > 1.0:
        raise HTTPException(
            status_code=422,
            detail=(
                "past a learning rate of 1 each round's correction overshoots "
                "the very mistake it just fitted, so the fit gets worse round "
                "by round, and the library refuses to build such a model"
            ),
        )

    x_values = [point.x for point in request.points]
    y_values = [point.y for point in request.points]
    inputs = Feature("t", x_values)
    target = Feature("h", y_values)

    model = GradientBoostingRegressor(
        n_rounds=request.rounds, learning_rate=request.learning_rate, max_depth=1
    ).fit([inputs], target)

    grid = np.linspace(min(x_values), max(x_values), CURVE_SAMPLES)
    curve_heights = np.asarray(model.predict([Feature("t", grid)]), dtype=np.float64)
    fitted_points = np.asarray(model.predict([inputs]), dtype=np.float64)

    return BoostingResponse(
        r_squared=float(model.score([inputs], target)),
        predictions=[float(value) for value in fitted_points],
        curve=[
            CurvePoint(x=float(x), y=float(y))
            for x, y in zip(grid, curve_heights, strict=True)
        ],
    )
