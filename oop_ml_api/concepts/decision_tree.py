"""Grow a decision tree and hand back its questions, for the trees page.

The library's ``DecisionTreeClassifier`` grows the tree, and this endpoint
walks the grown structure into a nested document the widget renders as an
indented flowchart, each split a readable question and each leaf the class it
answers. The decision regions come back as the shared lattice, which for a
tree draws the axis-aligned boxes its questions carve the plane into.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from oop_ml import DecisionTreeClassifier
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
