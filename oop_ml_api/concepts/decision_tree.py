"""Grow a decision tree and hand back its questions, for the trees page.

The library's ``DecisionTreeClassifier`` grows the tree, and this endpoint
walks the grown structure into a nested document the widget renders as an
indented flowchart, each split a readable question and each leaf the class it
answers. The decision regions come back as the shared lattice, which for a
tree draws the axis-aligned boxes its questions carve the plane into.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import DecisionTreeClassifier, FeatureSet, RowBlock
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    LabelledPoint,
    RegionGrid,
    features_of,
    region_grid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

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


MAX_WALK_DEPTH = 3
"""The deepest tree the walking widget may grow.

Deliberately shallower than the fitting endpoint's cap. The walk exists to
be read node by node, and a tree past three questions stops fitting in one
look, which is the one job this endpoint has.
"""


class WalkVisitor(BaseModel):
    """The person to send down the tree, height then weight."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class TreeWalkRequest(BaseModel):
    """The crowd to grow the tree from, how deep, and who walks it."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_depth: int = Field(ge=1, le=MAX_WALK_DEPTH)
    visitor: WalkVisitor

    model_config = {"extra": "forbid"}


class WalkNodeDocument(BaseModel):
    """One node of the walkable tree, carrying an id the path can point at."""

    id: int
    kind: Literal["question", "answer"]
    samples: int
    feature: str | None = None
    threshold: float | None = None
    label: int | None = None
    left: WalkNodeDocument | None = None
    right: WalkNodeDocument | None = None


class WalkStep(BaseModel):
    """One question asked on the way down, and which way the visitor went."""

    node_id: int
    feature: str
    threshold: float
    value: float
    direction: Literal["left", "right"]


class TreeWalkResponse(BaseModel):
    """The tree, the route one visitor took through it, and where they landed."""

    tree: WalkNodeDocument
    steps: list[WalkStep]
    leaf_id: int
    prediction: int


def _document_with_ids(node, identities: dict[int, int], counter: list[int]):
    """Walk the fitted tree into a document, remembering each node's id."""
    assigned = counter[0]
    counter[0] += 1
    identities[id(node)] = assigned

    if hasattr(node, "split"):
        return WalkNodeDocument(
            id=assigned,
            kind="question",
            samples=int(node.n_samples),
            feature=node.split.feature_name,
            threshold=float(node.split.threshold),
            left=_document_with_ids(node.left, identities, counter),
            right=_document_with_ids(node.right, identities, counter),
        )
    return WalkNodeDocument(
        id=assigned,
        kind="answer",
        samples=int(node.n_samples),
        label=int(node.prediction),
    )


@router.post("/walk", response_model=TreeWalkResponse)
def walk(request: TreeWalkRequest) -> TreeWalkResponse:
    """Grow a small tree and route one visitor through it, step by step.

    The routing is the library's own ``Split.sends_left`` at every node, the
    same rule ``predict`` applies in bulk, so the recorded path cannot
    disagree with the model's answer, and a test asserts the landing leaf's
    label is the prediction.
    """
    heights, weights, labels = features_of(request.points)
    tree = DecisionTreeClassifier(max_depth=request.max_depth).fit(
        [heights, weights], labels
    )

    identities: dict[int, int] = {}
    document = _document_with_ids(tree.root, identities, [0])

    visitor_block = RowBlock(
        np.asarray([[request.visitor.x, request.visitor.y]], dtype=np.float64),
        ["height", "weight"],
    )

    steps: list[WalkStep] = []
    node = tree.root
    while hasattr(node, "split"):
        goes_left = bool(node.split.sends_left(visitor_block)[0])
        observed = (
            request.visitor.x
            if node.split.feature_name == "height"
            else request.visitor.y
        )
        steps.append(
            WalkStep(
                node_id=identities[id(node)],
                feature=node.split.feature_name,
                threshold=float(node.split.threshold),
                value=float(observed),
                direction="left" if goes_left else "right",
            )
        )
        node = node.left if goes_left else node.right

    return TreeWalkResponse(
        tree=document,
        steps=steps,
        leaf_id=identities[id(node)],
        prediction=int(node.prediction),
    )
