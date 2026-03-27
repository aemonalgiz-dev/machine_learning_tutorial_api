"""Classify one query point by its nearest neighbours, for the k-NN page.

The library's ``KNearestNeighboursClassifier`` does the work, and its observed
``neighbour_search`` route supplies what a widget uniquely needs, which of the
stored people were the k nearest and how far away each sat, so the picture can
draw the lines the vote travelled along. The decision regions come back as the
shared lattice, the fitted model's own answer across the plane.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import Feature, KNearestNeighboursClassifier
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS

router = APIRouter(
    prefix="/concepts/k-nearest-neighbours", tags=["k-nearest-neighbours"]
)

MAX_K = 15
"""The most neighbours the widget may ask to consult."""


class QueryPoint(BaseModel):
    """The new person to classify."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class ClassifyRequest(BaseModel):
    """The stored people, the query, and how many neighbours vote."""

    points: list[LabelledPoint] = Field(min_length=1, max_length=MAX_POINTS)
    query: QueryPoint
    k: int = Field(ge=1, le=MAX_K)

    model_config = {"extra": "forbid"}


class ChosenNeighbour(BaseModel):
    """One of the k nearest, by its index into the request's points."""

    index: int
    distance: float
    label: int


class ClassifyResponse(BaseModel):
    """The vote, its voters, and the model's answer across the plane."""

    prediction: int
    neighbours: list[ChosenNeighbour]
    votes_for_zero: int
    votes_for_one: int
    regions: RegionGrid


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Fit on the stored people and put the query to its k nearest."""
    if request.k > len(request.points):
        raise HTTPException(
            status_code=422,
            detail=(
                f"k is {request.k} but only {len(request.points)} people are "
                "stored, and a vote cannot have more voters than people"
            ),
        )

    heights, weights, labels = features_of(request.points)
    model = KNearestNeighboursClassifier(n_neighbours=request.k).fit(
        [heights, weights], labels
    )

    query_features = [
        Feature("height", [request.query.x]),
        Feature("weight", [request.query.y]),
    ]

    prediction = int(np.asarray(model.predict(query_features), dtype=np.float64)[0])

    search = model.neighbour_search(query_features)
    neighbours: list[ChosenNeighbour] = []
    for query in search:
        for index, distance in zip(
            query.chosen_indices, query.chosen_distances, strict=True
        ):
            neighbours.append(
                ChosenNeighbour(
                    index=int(index),
                    distance=float(distance),
                    label=request.points[int(index)].label,
                )
            )

    votes_for_one = sum(1 for neighbour in neighbours if neighbour.label == 1)

    return ClassifyResponse(
        prediction=prediction,
        neighbours=sorted(neighbours, key=lambda each: each.distance),
        votes_for_zero=len(neighbours) - votes_for_one,
        votes_for_one=votes_for_one,
        regions=region_grid(model, request.points),
    )
