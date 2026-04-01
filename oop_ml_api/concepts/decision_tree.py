"""Grow a decision tree and hand back its questions, for the trees page.

The library's ``DecisionTreeClassifier`` grows the tree, and this endpoint
walks the grown structure into a nested document the widget renders as an
indented flowchart, each split a readable question and each leaf the class it
answers. The decision regions come back as the shared lattice, which for a
tree draws the axis-aligned boxes its questions carve the plane into.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from oop_ml import DecisionTreeClassifier, FeatureSet, RowBlock
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/decision-trees", tags=["decision-trees"])

MAX_DEPTH = 6
"""The deepest tree the widget may ask to grow."""


class TreeFitRequest(BaseModel):
    """The people to grow the tree from, and how deep it may go."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_depth: int = Field(ge=1, le=MAX_DEPTH)

    model_config = {"extra": "forbid"}


class TreeNodeDocument(BaseModel):
    """One node of the grown tree, a question or an answer.

    A split node carries its question and two children. A leaf carries the
    class it predicts. Both carry how many training people reached them.
    """

    kind: str
    samples: int
    question: str | None = None
    left: TreeNodeDocument | None = None
    right: TreeNodeDocument | None = None
    label: int | None = None


class TreeFitResponse(BaseModel):
    """The grown tree, its size, its training accuracy, and its regions."""

    tree: TreeNodeDocument
    depth: int
    n_leaves: int
    accuracy: float
    regions: RegionGrid


def _walk(node) -> TreeNodeDocument:
    """Turn the library's grown nodes into the nested document, recursively.

    A node with a split has children and its question is written the way the
    routing works, values below the threshold go left. Anything else is a
    leaf answering with its class.
    """
    if hasattr(node, "split"):
        return TreeNodeDocument(
            kind="split",
            samples=int(node.n_samples),
            question=(f"{node.split.feature_name} < {float(node.split.threshold):g}?"),
            left=_walk(node.left),
            right=_walk(node.right),
        )
    return TreeNodeDocument(
        kind="leaf",
        samples=int(node.n_samples),
        label=int(node.prediction),
    )


@router.post("/fit", response_model=TreeFitResponse)
def fit(request: TreeFitRequest) -> TreeFitResponse:
    """Grow the tree, walk it, and shade its regions."""
    labels = {point.label for point in request.points}
    if len(labels) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "every person here belongs to one class, and a tree has "
                "nothing to separate until both classes appear"
            ),
        )

    heights, weights, target = features_of(request.points)
    model = DecisionTreeClassifier(max_depth=request.max_depth).fit(
        [heights, weights], target
    )

    return TreeFitResponse(
        tree=_walk(model.root),
        depth=int(model.depth),
        n_leaves=int(model.n_leaves),
        accuracy=float(model.score([heights, weights], target)),
        regions=region_grid(model, request.points),
    )


class RootCandidateDocument(BaseModel):
    """One question the root considered, and what asking it was worth."""

    feature: str
    threshold: float
    gain: float
    admitted: bool


class RootSearchResponse(BaseModel):
    """Every candidate question the root scored, and the one that won.

    The library's observed split search keeps all of its working, in scan
    order, which is what lets the page draw the sweep instead of asserting
    its outcome.
    """

    candidates: list[RootCandidateDocument]
    best_feature: str
    best_threshold: float
    best_gain: float


class RootSearchRequest(BaseModel):
    """The people whose root question is to be searched for."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


@router.post("/root-candidates", response_model=RootSearchResponse)
def root_candidates(request: RootSearchRequest) -> RootSearchResponse:
    """Score every candidate first question and report the sweep."""
    heights, weights, labels = features_of(request.points)

    tree = DecisionTreeClassifier(max_depth=1).fit([heights, weights], labels)

    block = RowBlock(
        FeatureSet([heights, weights]).feature_matrix, ["height", "weight"]
    )
    search = tree.split_search(block, labels.column)

    best = search.best
    return RootSearchResponse(
        candidates=[
            RootCandidateDocument(
                feature=candidate.split.feature_name,
                threshold=float(candidate.split.threshold),
                gain=float(candidate.split.gain),
                admitted=candidate.was_admitted,
            )
            for candidate in search
        ],
        best_feature=best.feature_name,
        best_threshold=float(best.threshold),
        best_gain=float(best.gain),
    )
