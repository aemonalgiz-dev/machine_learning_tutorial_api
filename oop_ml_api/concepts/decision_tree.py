"""Grow a decision tree and hand back its questions, for the trees page.

The library's ``DecisionTreeClassifier`` grows the tree, and this endpoint
walks the grown structure into a nested document the widget renders as an
indented flowchart, each split a readable question and each leaf the class it
answers. The decision regions come back as the shared lattice, which for a
tree draws the axis-aligned boxes its questions carve the plane into.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DecisionTreeClassifier,
    Feature,
    FeatureSet,
    LogisticRegression,
    Predictions,
    RowBlock,
    Standardizer,
)
from oop_ml.core.tree.impurity import GiniImpurity
from oop_ml.core.tree.weights import WeightedTargets
from pydantic import BaseModel, Field

from oop_ml_api.concepts.split import held_out_indices
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
    best_feature: str | None
    best_threshold: float | None
    best_gain: float | None


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
    # The observed route pairs the targets with their weights, uniform here.
    search = tree.split_search(block, WeightedTargets(labels.column))

    # On a crowd where no single question improves on the parent, the
    # search admits nothing and there is no winner to report. That is a real
    # answer rather than a failure, and the greedy section of the page is
    # built on it.
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
        best_feature=None if best is None else best.feature_name,
        best_threshold=None if best is None else float(best.threshold),
        best_gain=None if best is None else float(best.gain),
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


FEATURE_NAMES = ("height", "weight")
"""The two columns every crowd on the trees page carries, in request order."""


class SplitInspectorRequest(BaseModel):
    """One candidate question, named by its feature and threshold."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    feature: str = Field(pattern="^(height|weight)$")
    threshold: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)

    model_config = {"extra": "forbid"}


class NodeMixture(BaseModel):
    """What one side of a candidate split holds, and how mixed it is."""

    children: int
    adults: int
    n_samples: int
    child_share: float
    adult_share: float
    gini: float


class SplitInspectorResponse(BaseModel):
    """The question scored the way the split search scores it.

    ``candidates`` are every threshold the search would try for this feature,
    the midpoints between neighbouring distinct values, so the page can snap
    a slider to them.
    """

    parent: NodeMixture
    left: NodeMixture
    right: NodeMixture
    weighted_after: float
    gain: float
    candidates: list[float]


def _mixture(labels: np.ndarray, gini: GiniImpurity) -> NodeMixture:
    """Count and score one group of labels; an empty group scores zero."""
    n_samples = int(labels.size)
    children = int(np.sum(labels == 0))
    adults = n_samples - children
    if n_samples == 0:
        return NodeMixture(
            children=0,
            adults=0,
            n_samples=0,
            child_share=0.0,
            adult_share=0.0,
            gini=0.0,
        )
    impurity = gini.of(WeightedTargets(Feature("is_adult", labels).column))
    return NodeMixture(
        children=children,
        adults=adults,
        n_samples=n_samples,
        child_share=children / n_samples,
        adult_share=adults / n_samples,
        gini=float(impurity),
    )


def _candidate_thresholds(values: np.ndarray) -> list[float]:
    """The midpoints between neighbouring distinct sorted values."""
    distinct = np.unique(values)
    return [float((low + high) / 2) for low, high in pairwise(distinct)]


@router.post("/split-inspector", response_model=SplitInspectorResponse)
def split_inspector(request: SplitInspectorRequest) -> SplitInspectorResponse:
    """Score one candidate question with the library's own impurity."""
    values = np.asarray(
        [
            point.x if request.feature == "height" else point.y
            for point in request.points
        ],
        dtype=np.float64,
    )
    labels = np.asarray([point.label for point in request.points], dtype=np.float64)
    goes_left = values < request.threshold
    gini = GiniImpurity()

    parent = _mixture(labels, gini)
    left = _mixture(labels[goes_left], gini)
    right = _mixture(labels[~goes_left], gini)
    weighted_after = (
        left.n_samples / parent.n_samples * left.gini
        + right.n_samples / parent.n_samples * right.gini
    )

    return SplitInspectorResponse(
        parent=parent,
        left=left,
        right=right,
        weighted_after=weighted_after,
        gain=parent.gini - weighted_after,
        candidates=_candidate_thresholds(values),
    )


class GrowthRequest(BaseModel):
    """The crowd, and the growth controls the library's tree accepts."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_depth: int = Field(default=MAX_DEPTH, ge=1, le=MAX_DEPTH)
    min_samples_split: int = Field(default=2, ge=2, le=MAX_POINTS)
    min_samples_leaf: int = Field(default=1, ge=1, le=MAX_POINTS)
    min_impurity_decrease: float = Field(default=0.0, ge=0.0, le=1.0)
    # The forest's one extra rule. Left unset, every split sees both
    # features and this is an ordinary tree.
    max_features: int | None = Field(default=None, ge=1, le=2)
    random_seed: int | None = Field(default=None, ge=0, le=10_000)

    model_config = {"extra": "forbid"}


class GrownNode(BaseModel):
    """One node of the grown tree, with everything the growth widget draws.

    ``split_order`` numbers the splits in the order a breadth-first growth
    would make them, ``None`` on a leaf, so the widget can show the tree
    after any number of splits by treating later splits as leaves.
    ``bounds`` is the rectangle of the plane this node owns, in feature
    units, clipped to the region window. ``majority`` is the class the node
    would predict if it were a leaf, and the counts say how pure it is.
    """

    id: int
    parent_id: int | None
    depth: int
    split_order: int | None
    feature: str | None
    threshold: float | None
    gain: float | None
    children: int
    adults: int
    majority: int
    left_id: int | None
    right_id: int | None
    bounds: list[float]


class GrowthResponse(BaseModel):
    """The grown tree as a flat list of nodes, plus its size and score."""

    nodes: list[GrownNode]
    n_splits: int
    depth: int
    n_leaves: int
    accuracy: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def _window(points: list[LabelledPoint]) -> tuple[float, float, float, float]:
    """The same padded window ``region_grid`` draws on."""
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
    y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
    return (
        min(x_values) - x_pad,
        max(x_values) + x_pad,
        min(y_values) - y_pad,
        max(y_values) + y_pad,
    )


@router.post("/growth", response_model=GrowthResponse)
def growth(request: GrowthRequest) -> GrowthResponse:
    """Grow the tree and lay it out node by node for a step-through.

    The library grows depth first. The widget shows growth breadth first,
    one split at a time, and the two reach the same tree because every
    split is chosen from its own node's rows alone.
    """
    labels_present = {point.label for point in request.points}
    if len(labels_present) < 2:
        raise HTTPException(
            status_code=422,
            detail="a tree has nothing to separate until both classes appear",
        )

    heights, weights, target = features_of(request.points)
    model = DecisionTreeClassifier(
        max_depth=request.max_depth,
        min_samples_split=request.min_samples_split,
        min_samples_leaf=request.min_samples_leaf,
        min_impurity_decrease=request.min_impurity_decrease,
        max_features=request.max_features,
        random_seed=request.random_seed,
    ).fit([heights, weights], target)

    rows = np.column_stack([heights.values, weights.values]).astype(np.float64)
    labels = np.asarray(target.values, dtype=np.float64)
    x_min, x_max, y_min, y_max = _window(request.points)

    nodes: list[GrownNode] = []
    queue: list[tuple[object, int | None, int, np.ndarray, list[float]]] = [
        (
            model.root,
            None,
            0,
            np.ones(labels.size, dtype=bool),
            [x_min, x_max, y_min, y_max],
        )
    ]
    split_order = 0
    while queue:
        node, parent_id, depth, mask, bounds = queue.pop(0)
        node_id = len(nodes)
        here = labels[mask]
        children = int(np.sum(here == 0))
        adults = int(here.size - children)
        majority = 0 if children >= adults else 1

        if hasattr(node, "split"):
            block = RowBlock(rows[mask], list(FEATURE_NAMES))
            goes_left = np.asarray(node.split.sends_left(block), dtype=bool)
            left_mask = mask.copy()
            left_mask[mask] = goes_left
            right_mask = mask.copy()
            right_mask[mask] = ~goes_left
            threshold = float(node.split.threshold)
            if node.split.feature_name == "height":
                left_bounds = [
                    bounds[0],
                    min(bounds[1], threshold),
                    bounds[2],
                    bounds[3],
                ]
                right_bounds = [
                    max(bounds[0], threshold),
                    bounds[1],
                    bounds[2],
                    bounds[3],
                ]
            else:
                left_bounds = [
                    bounds[0],
                    bounds[1],
                    bounds[2],
                    min(bounds[3], threshold),
                ]
                right_bounds = [
                    bounds[0],
                    bounds[1],
                    max(bounds[2], threshold),
                    bounds[3],
                ]
            left_id = node_id + 1 + len(queue)
            right_id = left_id + 1
            nodes.append(
                GrownNode(
                    id=node_id,
                    parent_id=parent_id,
                    depth=depth,
                    split_order=split_order,
                    feature=node.split.feature_name,
                    threshold=threshold,
                    gain=float(node.split.gain),
                    children=children,
                    adults=adults,
                    majority=majority,
                    left_id=left_id,
                    right_id=right_id,
                    bounds=bounds,
                )
            )
            split_order += 1
            queue.append((node.left, node_id, depth + 1, left_mask, left_bounds))
            queue.append((node.right, node_id, depth + 1, right_mask, right_bounds))
        else:
            nodes.append(
                GrownNode(
                    id=node_id,
                    parent_id=parent_id,
                    depth=depth,
                    split_order=None,
                    feature=None,
                    threshold=None,
                    gain=None,
                    children=children,
                    adults=adults,
                    majority=int(node.prediction),
                    left_id=None,
                    right_id=None,
                    bounds=bounds,
                )
            )

    # Breadth-first ids were assigned by position in the queue, which is only
    # right if children are appended in the order their ids were promised.
    # They are, two at a time, so verify rather than trust.
    for grown in nodes:
        if grown.left_id is not None:
            assert nodes[grown.left_id].parent_id == grown.id
            assert nodes[grown.right_id].parent_id == grown.id

    return GrowthResponse(
        nodes=nodes,
        n_splits=split_order,
        depth=int(model.depth),
        n_leaves=int(model.n_leaves),
        accuracy=float(model.score([heights, weights], target)),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


class DepthSweepRequest(BaseModel):
    """The crowd to split once, seeded, and grow to every depth on."""

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    max_depth: int = Field(default=MAX_DEPTH, ge=1, le=MAX_DEPTH)

    model_config = {"extra": "forbid"}


class DepthScore(BaseModel):
    """One depth cap, and what a tree grown under it did on each share."""

    max_depth: int
    depth_reached: int
    n_leaves: int
    train_accuracy: float
    held_out_accuracy: float
    regions: RegionGrid


class DepthSweepResponse(BaseModel):
    """Every depth cap, on the same seeded split."""

    held_out_indices: list[int]
    scores: list[DepthScore]


@router.post("/depth-sweep", response_model=DepthSweepResponse)
def depth_sweep(request: DepthSweepRequest) -> DepthSweepResponse:
    """Hold back a share of the crowd, then grow a tree at every depth cap."""
    held = held_out_indices(len(request.points))
    train_mask = np.ones(len(request.points), dtype=bool)
    train_mask[held] = False
    training = [
        point for point, keep in zip(request.points, train_mask, strict=True) if keep
    ]
    held_out = [request.points[index] for index in held]
    if len({point.label for point in training}) < 2:
        raise HTTPException(
            status_code=422,
            detail="the training share holds one class only, so no tree can be grown on it",
        )

    train_heights, train_weights, train_target = features_of(training)
    held_heights, held_weights, held_target = features_of(held_out)

    scores: list[DepthScore] = []
    for max_depth in range(1, request.max_depth + 1):
        model = DecisionTreeClassifier(max_depth=max_depth).fit(
            [train_heights, train_weights], train_target
        )
        scores.append(
            DepthScore(
                max_depth=max_depth,
                depth_reached=int(model.depth),
                n_leaves=int(model.n_leaves),
                train_accuracy=float(
                    model.score([train_heights, train_weights], train_target)
                ),
                held_out_accuracy=float(
                    model.score([held_heights, held_weights], held_target)
                ),
                regions=region_grid(model, request.points),
            )
        )

    return DepthSweepResponse(
        held_out_indices=[int(index) for index in held], scores=scores
    )


class VersusLogisticRequest(BaseModel):
    """The crowd both a tree and a logistic boundary are fitted to."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_depth: int = Field(default=MAX_DEPTH, ge=1, le=MAX_DEPTH)

    model_config = {"extra": "forbid"}


class VersusLogisticResponse(BaseModel):
    """Two decision maps for one crowd, and each model's training accuracy."""

    tree_regions: RegionGrid
    tree_accuracy: float
    tree_leaves: int
    logistic_regions: RegionGrid
    logistic_accuracy: float


class _StandardisedLogistic:
    """A logistic fit that standardises what it is asked about first.

    ``region_grid`` asks in raw units and the curve was fitted on standardised
    columns, so this stands between the two, as the judging page's does.
    """

    def __init__(self, standardizer: Standardizer, model: LogisticRegression) -> None:
        self._standardizer = standardizer
        self._model = model

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        return self._model.predict(self._standardizer.transform(input_values))


@router.post("/versus-logistic", response_model=VersusLogisticResponse)
def versus_logistic(request: VersusLogisticRequest) -> VersusLogisticResponse:
    """Fit a tree and a logistic boundary to the same crowd and map both."""
    if len({point.label for point in request.points}) < 2:
        raise HTTPException(
            status_code=422,
            detail="both classes have to appear before either model has a boundary to draw",
        )

    heights, weights, target = features_of(request.points)
    tree = DecisionTreeClassifier(max_depth=request.max_depth).fit(
        [heights, weights], target
    )

    standardizer = Standardizer().fit([heights, weights])
    standardised = standardizer.transform([heights, weights])
    logistic = LogisticRegression(learning_rate=1.0, max_epochs=2000).fit(
        standardised, target
    )

    return VersusLogisticResponse(
        tree_regions=region_grid(tree, request.points),
        tree_accuracy=float(tree.score([heights, weights], target)),
        tree_leaves=int(tree.n_leaves),
        logistic_regions=region_grid(
            _StandardisedLogistic(standardizer, logistic), request.points
        ),
        logistic_accuracy=float(logistic.score(standardised, target)),
    )
