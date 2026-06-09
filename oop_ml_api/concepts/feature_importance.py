"""Which feature mattered, for the feature-importance page.

Four endpoints, three of them fixtures the page quotes from.

``measure`` draws the parity puzzle, two coin-flip columns whose disagreement
is the class and a third column of pure noise that offers the split search a
threshold between every pair of its values, and fits the library's lone
``DecisionTreeClassifier`` and its ``RandomForestClassifier`` on it. Each
fitted model is asked the page's two questions. What its own splits credit
comes from ``feature_importances``, and what breaks when a column is
scrambled comes from ``PermutationImportance``, once on the rows the model
learned and once on as many rows it never saw. The lone tree's decision nodes
come back with their rows and their gain, which is everything the impurity
shares are made of, and every scramble walk is one loop over the model's
public ``score`` with the same seeded draws the library makes, so the drops
it records normalise to exactly the shares the library reports.

``ledger`` takes the site's own crowd of people, grows the lone tree on it,
and hands back every split in the order a reader steps through them with
the running credit per feature after each, which is the impurity measure
assembled a split at a time. The seed-7 forest the random-forests page ends
on is fitted beside it, so the two pages quote one committee.

``drawn-crowd`` is a larger crowd the endpoint draws once, two hundred
people and as many held out, with two columns added to height and weight
that a careful reader should distrust: a raffle ticket number, which is pure
chance with two hundred distinct values, and a second reading of height
with a little measurement error. The rule that assigns adult reads height
alone, so weight follows the label only through overall body size. The four
columns are fitted in six arrangements, two of them a single column, so what
a column does to the others' shares is a comparison between fits rather than
an assertion.

``edges`` probes what the library does at each boundary the page's closing
table lists, so the table reports behaviour that was run rather than
remembered.

A forest member restricted to one column at its root can be handed a coin
whose split buys exactly nothing, and a member that grew no split has no
shares for the forest to average. The library refuses that in its own words,
which are about one member and read as if they were about the forest, so the
parity endpoint checks for such a member first and refuses in a sentence a
reader can act on. Measured over one hundred and eighty draws across the
widget's range, four grew one.
"""

from __future__ import annotations

import math
from collections import deque
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BaggingRegressor,
    DecisionNode,
    DecisionTreeClassifier,
    Feature,
    FeatureImportances,
    PermutationImportance,
    RandomForestClassifier,
    RidgeRegression,
    TreeNode,
)
from oop_ml.core.exceptions import InvalidValuesError, MLLibError
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.two_feature_classification import LabelledPoint, features_of
from oop_ml_api.limits import MAX_POINTS

router = APIRouter(tags=["feature-importance"])

MIN_ROWS = 200
"""The fewest training rows a parity request may draw.

The page's finding needs a forest that works, and below this it often does
not. Measured over twenty draws at each size, the forest's held-out accuracy
fell below 0.9 on eleven draws at a hundred rows, nine at a hundred and fifty,
and two at two hundred, where its median was 0.965.
"""

MAX_ROWS = 400
"""The most training rows a parity request may draw. As many again are held
out. A cost bound rather than a teaching one."""

MAX_SEED = 9999
"""The largest seed the redraw control may reach."""

FEATURE_NAMES = ("first", "second", "distractor")
"""The two coin flips whose disagreement is the class, and the column of
noise, in the order every fit sees them."""

TARGET_NAME = "parity"
"""What the parity class column is called."""

LONE_TREE_DEPTH = 3
"""Deep enough for parity, which takes two questions, and far short of
memorising the rows, which an unstopped tree does and which would hide the
point. The forest's members grow to the same depth."""

FOREST_MEMBERS = 20
"""How many trees the parity forest grows."""

FOREST_SEED = 0
"""Fixes the parity forest's resamples and its per-node feature draws, so the
page can quote its shares. Only the data's own draw follows the request's
seed."""

SHUFFLE_SEED = 0
"""Fixes every scramble, so the permutation reading is reproducible and the
page can quote it."""

SHUFFLE_REPEATS = 5
"""How many times each column is scrambled before the drops are averaged."""

CROWD_SEED = 7
"""The seed the random-forests page's committee draws from, reused here so
the ledger's forest is that page's forest."""

MAX_LEDGER_MEMBERS = 50
"""The largest committee the ledger endpoint grows."""

MAX_LEDGER_DEPTH = 8
"""The deepest a ledger tree may be asked to grow. Unset, it grows until
every leaf is pure, which on the site's crowd takes eight splits."""

DRAWN_ROWS = 200
"""How many people the drawn crowd holds. As many again are held out."""

DRAWN_SEED = 3
"""The drawn crowd is drawn once, from this seed."""

DRAWN_FOREST_MEMBERS = 30
"""How many trees the drawn crowd's forest grows. Fifty measured 15.7
seconds for the whole fixture on first call, thirty is under ten, and the
readings tell the same story at both."""

DRAWN_FOREST_SEED = 0
"""Fixes the drawn crowd's forest."""

DRAWN_COLUMNS = ("height", "weight", "ticket", "height_again")
"""The four columns of the drawn crowd, in the order every fit sees them."""

ARRANGEMENTS: dict[str, tuple[str, ...]] = {
    "measurements": ("height", "weight"),
    "with_ticket": ("height", "weight", "ticket"),
    "with_twin": ("height", "weight", "height_again"),
    "all_four": ("height", "weight", "ticket", "height_again"),
    "weight_only": ("weight",),
    "ticket_only": ("ticket",),
}
"""Which columns each arrangement hands the fit. The two single-column
arrangements exist for what a fit on one column shows: weight alone predicts
adulthood usefully without any role in the rule, and the ticket alone is
memorised on the training rows and worth nothing held out."""

ADULT_THRESHOLD = 152.0
"""The height, in centimetres, above which the drawn crowd's rule calls a
person an adult, before the rule's own noise is added."""


class MeasureRequest(BaseModel):
    """How many parity rows to draw, and which draw."""

    n_rows: int = Field(ge=MIN_ROWS, le=MAX_ROWS)
    seed: int = Field(ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class ImportanceShare(BaseModel):
    """One feature's share of a model's explanation, in the fit's order."""

    name: str
    share: float


class SplitCredit(BaseModel):
    """One decision node, and the credit it hands its feature.

    ``credit`` is ``n_rows`` times ``gain``, the library's own weighting, so a
    question asked of every row counts for more than one asked of a dozen.
    """

    feature: str
    depth: int
    n_rows: int
    gain: float
    credit: float


class FeatureCredit(BaseModel):
    """Everything one feature's questions earned, summed."""

    name: str
    credit: float


class ShuffleStep(BaseModel):
    """What scrambling one column did to the score.

    ``shuffled_score`` is averaged over the repeats and ``drop`` is the intact
    score minus it, before the clamp at zero the library applies, so a column
    the model was slightly better off without shows its negative drop here.
    """

    name: str
    shuffled_score: float
    drop: float


class ScrambleReading(BaseModel):
    """The scramble experiment on one set of rows.

    ``shares`` is what the library reports, and is ``None`` when it refused
    because no column's scramble lowered the score, with its own words in
    ``refusal``. ``drop_total`` is the sum of the clamped drops, the number
    every share is divided by, and ``steps`` is the recorded walk.
    """

    intact_score: float
    steps: list[ShuffleStep]
    drop_total: float
    shares: list[ImportanceShare] | None
    leading: str | None
    refusal: str | None


class ModelReport(BaseModel):
    """Both importance readings for one fitted model, and its two accuracies.

    ``training`` scrambles the rows the model learned and ``held_out`` the
    rows it never saw.
    """

    training_accuracy: float
    held_out_accuracy: float
    impurity: list[ImportanceShare]
    leading_by_impurity: str
    training: ScrambleReading
    held_out: ScrambleReading


class LoneTreeReport(ModelReport):
    """The lone tree's report, with every decision node it grew.

    ``credit_total`` is the sum of every node's credit, the number every
    impurity share is divided by.
    """

    splits: list[SplitCredit]
    credits: list[FeatureCredit]
    credit_total: float


class MeasureResponse(BaseModel):
    """Both models measured both ways on one draw of the parity puzzle."""

    n_rows: int
    feature_names: list[str]
    lone_tree: LoneTreeReport
    forest: ModelReport


class LabelledRows:
    """Some rows of the puzzle, the three columns and the class they carry."""

    def __init__(self, n_rows: int, generator: np.random.Generator) -> None:
        first = generator.integers(0, 2, size=n_rows).astype(np.float64)
        second = generator.integers(0, 2, size=n_rows).astype(np.float64)
        distractor = generator.normal(size=n_rows)

        self.features = [
            Feature(FEATURE_NAMES[0], first),
            Feature(FEATURE_NAMES[1], second),
            Feature(FEATURE_NAMES[2], distractor),
        ]
        self.target = Feature(TARGET_NAME, (first != second).astype(np.float64))


class ParityDraw:
    """One draw of the puzzle, the training rows first and as many held out.

    The class is 1 where exactly one of the two coin flips came up 1, and the
    distractor is standard normal noise that never touches the class. Drawing
    the training rows first, in the order first, second, distractor, is what
    makes the worked draw the library's own parity fixture.
    """

    def __init__(self, n_rows: int, seed: int) -> None:
        generator = np.random.default_rng(seed)
        self.training = LabelledRows(n_rows, generator)
        self.held_out = LabelledRows(n_rows, generator)


class Rows:
    """Some features and the target they are paired with."""

    def __init__(self, features: list[Feature], target: Feature) -> None:
        self.features = features
        self.target = target


@router.post("/concepts/feature-importance/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> MeasureResponse:
    """Fit a lone tree and a forest on the parity puzzle and ask both questions."""
    draw = ParityDraw(request.n_rows, request.seed)
    training = Rows(draw.training.features, draw.training.target)
    held_out = Rows(draw.held_out.features, draw.held_out.target)

    lone_tree = DecisionTreeClassifier(max_depth=LONE_TREE_DEPTH).fit(
        training.features, training.target
    )
    forest = RandomForestClassifier(
        n_members=FOREST_MEMBERS,
        max_features=1,
        max_depth=LONE_TREE_DEPTH,
        random_seed=FOREST_SEED,
    ).fit(training.features, training.target)
    _refuse_a_forest_with_a_member_that_never_split(forest, request.seed)

    splits: list[SplitCredit] = []
    _collect_splits(lone_tree.root, 0, splits)
    credits = _credits_by_feature(splits, FEATURE_NAMES)

    return MeasureResponse(
        n_rows=request.n_rows,
        feature_names=list(FEATURE_NAMES),
        lone_tree=LoneTreeReport(
            **_report(lone_tree, training, held_out).model_dump(),
            splits=splits,
            credits=credits,
            credit_total=sum(credit.credit for credit in credits),
        ),
        forest=_report(forest, training, held_out),
    )


def _refuse_a_forest_with_a_member_that_never_split(
    forest: RandomForestClassifier, seed: int
) -> None:
    """Refuse, readably, a draw on which the forest cannot report shares."""
    for position, member in enumerate(forest.members):
        if isinstance(member, DecisionTreeClassifier) and not isinstance(
            member.root, DecisionNode
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"On draw {seed} member {position} of the forest was handed a "
                    "single column at its root, found no gain in it, and grew no "
                    "split at all, so there are no impurity shares to average "
                    "across the members. Redraw the rows."
                ),
            )


Fitted = DecisionTreeClassifier | RandomForestClassifier


def _report(model: Fitted, training: Rows, held_out: Rows) -> ModelReport:
    """Both readings and both accuracies for one fitted model."""
    impurity = model.feature_importances

    return ModelReport(
        training_accuracy=float(model.score(training.features, training.target)),
        held_out_accuracy=float(model.score(held_out.features, held_out.target)),
        impurity=_shares(impurity),
        leading_by_impurity=impurity.most_important.name,
        training=_scramble(model, training),
        held_out=_scramble(model, held_out),
    )


def _shares(importances: FeatureImportances) -> list[ImportanceShare]:
    """The shares in the order the fit saw the features."""
    return [
        ImportanceShare(name=importance.name, share=float(importance.value))
        for importance in importances
    ]


def _scramble(model: Fitted, rows: Rows) -> ScrambleReading:
    """The library's permutation reading on these rows, and the recorded walk."""
    intact_score = float(model.score(rows.features, rows.target))
    steps = _shuffle_walk(model, rows, intact_score)

    shares: list[ImportanceShare] | None
    leading: str | None
    refusal: str | None
    try:
        importances = PermutationImportance(
            n_repeats=SHUFFLE_REPEATS, random_seed=SHUFFLE_SEED
        ).measure(model, rows.features, rows.target)
    except InvalidValuesError as error:
        shares, leading, refusal = None, None, str(error)
    else:
        shares, leading, refusal = (
            _shares(importances),
            importances.most_important.name,
            None,
        )

    return ScrambleReading(
        intact_score=intact_score,
        steps=steps,
        drop_total=sum(max(0.0, step.drop) for step in steps),
        shares=shares,
        leading=leading,
        refusal=refusal,
    )


def _shuffle_walk(model: Fitted, rows: Rows, intact_score: float) -> list[ShuffleStep]:
    """Scramble each column in turn and record what the score became.

    One loop over the model's public ``score``, drawing every permutation from
    one generator seeded once, columns in fit order and the repeats of each
    column together, which is the order the library's own measure draws in.
    """
    generator = np.random.default_rng(SHUFFLE_SEED)
    steps: list[ShuffleStep] = []

    for position, feature in enumerate(rows.features):
        shuffled_scores: list[float] = []
        for _ in range(SHUFFLE_REPEATS):
            scrambled = [
                Feature(feature.name, generator.permutation(feature.values))
                if index == position
                else other
                for index, other in enumerate(rows.features)
            ]
            shuffled_scores.append(float(model.score(scrambled, rows.target)))

        mean_shuffled = float(np.mean(shuffled_scores))
        steps.append(
            ShuffleStep(
                name=feature.name,
                shuffled_score=mean_shuffled,
                drop=intact_score - mean_shuffled,
            )
        )

    return steps


def _collect_splits(node: TreeNode, depth: int, splits: list[SplitCredit]) -> None:
    """Walk the tree top down and record every question it asks."""
    if not isinstance(node, DecisionNode):
        return

    splits.append(
        SplitCredit(
            feature=node.split.feature_name,
            depth=depth,
            n_rows=int(node.n_samples),
            gain=float(node.split.gain),
            credit=float(node.n_samples * node.split.gain),
        )
    )
    _collect_splits(node.left, depth + 1, splits)
    _collect_splits(node.right, depth + 1, splits)


def _credits_by_feature(
    splits: list[SplitCredit], names: tuple[str, ...]
) -> list[FeatureCredit]:
    """Each feature's credits summed, in fit order, zero where it won nothing."""
    totals = dict.fromkeys(names, 0.0)
    for split in splits:
        totals[split.feature] += split.credit

    return [FeatureCredit(name=name, credit=totals[name]) for name in names]


# --- The ledger: the site's crowd, one split at a time ----------------------


class LedgerRequest(BaseModel):
    """The crowd, how deep the lone tree may grow, and the committee beside it.

    ``max_depth`` unset grows the tree until every leaf is pure. The forest
    defaults to the random-forests page's own, twenty-five members offered one
    feature per split under seed 7.
    """

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    max_depth: int | None = Field(default=None, ge=1, le=MAX_LEDGER_DEPTH)
    n_members: int = Field(default=25, ge=1, le=MAX_LEDGER_MEMBERS)
    max_features: int = Field(default=1, ge=1, le=2)
    seed: int = Field(default=CROWD_SEED, ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class LedgerSplit(BaseModel):
    """One split of the ledger tree, and the running totals once it is added.

    Splits are numbered breadth first, the order the decision-tree page steps
    through them. ``impurity`` is the node's own before the question was
    asked, ``credit`` is ``n_rows`` times ``gain``, and ``running_credits``
    and ``running_shares`` are the state of the ledger after this split, so
    the last split's shares are the model's own.
    """

    order: int
    feature: str
    threshold: float
    depth: int
    n_rows: int
    impurity: float
    gain: float
    credit: float
    running_credits: list[FeatureCredit]
    running_shares: list[ImportanceShare]


class LedgerTree(BaseModel):
    """The lone tree, split by split."""

    depth: int
    n_leaves: int
    n_splits: int
    training_accuracy: float
    splits: list[LedgerSplit]
    credits: list[FeatureCredit]
    credit_total: float
    impurity: list[ImportanceShare]
    training: ScrambleReading


class LedgerForest(BaseModel):
    """The committee grown beside the tree, and its two readings.

    ``member_shares`` holds every member's own impurity shares, in member
    order, which the forest averages into ``impurity``. ``roots`` counts
    which feature each member's first question asked about.
    """

    n_members: int
    max_features: int
    training_accuracy: float
    out_of_bag_accuracy: float
    impurity: list[ImportanceShare]
    member_shares: list[list[ImportanceShare]]
    roots: list[FeatureCredit]
    training: ScrambleReading


class LedgerResponse(BaseModel):
    """The lone tree's ledger and the forest's readings on one crowd."""

    feature_names: list[str]
    tree: LedgerTree
    forest: LedgerForest


@router.post("/concepts/feature-importance/ledger", response_model=LedgerResponse)
def ledger(request: LedgerRequest) -> LedgerResponse:
    """Grow the lone tree on the crowd and account for every split it makes."""
    heights, weights, labels = features_of(request.points)
    rows = Rows([heights, weights], labels)
    names = ("height", "weight")

    tree = DecisionTreeClassifier(max_depth=request.max_depth).fit(
        rows.features, rows.target
    )
    splits = _ledger_splits(tree.root, names)
    credits = (
        splits[-1].running_credits
        if splits
        else [FeatureCredit(name=name, credit=0.0) for name in names]
    )

    forest = RandomForestClassifier(
        n_members=request.n_members,
        max_features=request.max_features,
        random_seed=request.seed,
    ).fit(rows.features, rows.target)
    roots = dict.fromkeys(names, 0.0)
    member_shares: list[list[ImportanceShare]] = []
    for member in forest.members:
        assert isinstance(member, DecisionTreeClassifier)
        if isinstance(member.root, DecisionNode):
            roots[member.root.split.feature_name] += 1.0
        member_shares.append(_shares(member.feature_importances))

    return LedgerResponse(
        feature_names=list(names),
        tree=LedgerTree(
            depth=int(tree.root.depth),
            n_leaves=int(tree.root.n_leaves),
            n_splits=len(splits),
            training_accuracy=float(tree.score(rows.features, rows.target)),
            splits=splits,
            credits=credits,
            credit_total=sum(credit.credit for credit in credits),
            impurity=_shares(tree.feature_importances),
            training=_scramble(tree, rows),
        ),
        forest=LedgerForest(
            n_members=request.n_members,
            max_features=request.max_features,
            training_accuracy=float(forest.score(rows.features, rows.target)),
            out_of_bag_accuracy=float(forest.out_of_bag_score()),
            impurity=_shares(forest.feature_importances),
            member_shares=member_shares,
            roots=[FeatureCredit(name=name, credit=roots[name]) for name in names],
            training=_scramble(forest, rows),
        ),
    )


def _ledger_splits(root: TreeNode, names: tuple[str, ...]) -> list[LedgerSplit]:
    """Every decision node breadth first, with the ledger's state after each."""
    totals = dict.fromkeys(names, 0.0)
    splits: list[LedgerSplit] = []
    queue: deque[tuple[TreeNode, int]] = deque([(root, 0)])

    while queue:
        node, depth = queue.popleft()
        if not isinstance(node, DecisionNode):
            continue

        credit = float(node.n_samples * node.split.gain)
        totals[node.split.feature_name] += credit
        total = sum(totals.values())
        splits.append(
            LedgerSplit(
                order=len(splits) + 1,
                feature=node.split.feature_name,
                threshold=float(node.split.threshold),
                depth=depth,
                n_rows=int(node.n_samples),
                impurity=float(node.impurity),
                gain=float(node.split.gain),
                credit=credit,
                running_credits=[
                    FeatureCredit(name=name, credit=totals[name]) for name in names
                ],
                running_shares=[
                    ImportanceShare(name=name, share=totals[name] / total)
                    for name in names
                ],
            )
        )
        queue.append((node.left, depth + 1))
        queue.append((node.right, depth + 1))

    return splits


# --- The drawn crowd: four columns, four arrangements ------------------------


class FeatureCount(BaseModel):
    """How many splits one feature won."""

    name: str
    count: int


class ArrangementModel(BaseModel):
    """One model fitted on one arrangement of the drawn crowd's columns.

    ``split_counts`` answers which columns were used often, ``impurity``
    which removed the most, and the two scramble readings which the
    predictions depend on. ``out_of_bag_accuracy`` is ``None`` for the lone
    tree, which has no rows it left out.
    """

    training_accuracy: float
    held_out_accuracy: float
    out_of_bag_accuracy: float | None
    n_splits: int
    depth: int
    split_counts: list[FeatureCount]
    impurity: list[ImportanceShare]
    leading_by_impurity: str
    training: ScrambleReading
    held_out: ScrambleReading


class Arrangement(BaseModel):
    """One choice of columns, fitted by both models."""

    name: str
    feature_names: list[str]
    max_features: int
    tree: ArrangementModel
    forest: ArrangementModel


class DrawnCrowdResponse(BaseModel):
    """The drawn crowd and every arrangement of its columns, fitted both ways.

    ``correlations`` pairs each column with height, so the reader can see
    that the second reading of height is very nearly height and that weight
    follows it closely. ``adult_share`` is the fraction of training rows the
    rule called adult.
    """

    n_rows: int
    n_held_out: int
    column_names: list[str]
    correlations_with_height: list[FeatureCredit]
    adult_share: float
    arrangements: list[Arrangement]


class DrawnCrowd:
    """The drawn crowd, its four columns and its rule, training and held out.

    Overall body size is one hidden number per person. Height and weight both
    follow it, so they are correlated without either causing the other, and
    the rule that assigns adult reads height alone. The ticket is a raffle
    number drawn without reference to anything, and the second reading of
    height is height plus a little measurement error.
    """

    def __init__(self, n_rows: int, seed: int) -> None:
        generator = np.random.default_rng(seed)
        self.training = self._draw(n_rows, generator)
        self.held_out = self._draw(n_rows, generator)

    @staticmethod
    def _draw(n_rows: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
        size = generator.normal(size=n_rows)
        height = 150.0 + 15.0 * size + generator.normal(scale=4.0, size=n_rows)
        weight = 45.0 + 12.0 * size + generator.normal(scale=5.0, size=n_rows)
        is_adult = (
            height + generator.normal(scale=6.0, size=n_rows) > ADULT_THRESHOLD
        ).astype(np.float64)
        return {
            "height": height,
            "weight": weight,
            "ticket": generator.uniform(0.0, 1000.0, size=n_rows),
            "height_again": height + generator.normal(scale=1.5, size=n_rows),
            "is_adult": is_adult,
        }

    @staticmethod
    def rows(columns: dict[str, np.ndarray], names: tuple[str, ...]) -> Rows:
        return Rows(
            [Feature(name, columns[name]) for name in names],
            Feature("is_adult", columns["is_adult"]),
        )


def _split_counts(root: TreeNode, totals: dict[str, int]) -> None:
    """Count the decision nodes beneath ``root`` by feature, itself included."""
    if not isinstance(root, DecisionNode):
        return
    totals[root.split.feature_name] += 1
    _split_counts(root.left, totals)
    _split_counts(root.right, totals)


def _arrangement_model(
    model: Fitted, names: tuple[str, ...], training: Rows, held_out: Rows
) -> ArrangementModel:
    """Every reading of one fitted model on one arrangement."""
    counts = dict.fromkeys(names, 0)
    out_of_bag: float | None
    if isinstance(model, DecisionTreeClassifier):
        _split_counts(model.root, counts)
        depth = int(model.root.depth)
        out_of_bag = None
    else:
        depths: list[int] = []
        for member in model.members:
            assert isinstance(member, DecisionTreeClassifier)
            _split_counts(member.root, counts)
            depths.append(int(member.root.depth))
        depth = max(depths)
        out_of_bag = float(model.out_of_bag_score())

    report = _report(model, training, held_out)
    return ArrangementModel(
        training_accuracy=report.training_accuracy,
        held_out_accuracy=report.held_out_accuracy,
        out_of_bag_accuracy=out_of_bag,
        n_splits=sum(counts.values()),
        depth=depth,
        split_counts=[FeatureCount(name=name, count=counts[name]) for name in names],
        impurity=report.impurity,
        leading_by_impurity=report.leading_by_impurity,
        training=report.training,
        held_out=report.held_out,
    )


@cache
def _drawn_crowd() -> DrawnCrowdResponse:
    crowd = DrawnCrowd(DRAWN_ROWS, DRAWN_SEED)
    height = crowd.training["height"]
    correlations = [
        FeatureCredit(
            name=name,
            credit=float(np.corrcoef(height, crowd.training[name])[0, 1]),
        )
        for name in DRAWN_COLUMNS
    ]

    arrangements: list[Arrangement] = []
    for name, names in ARRANGEMENTS.items():
        training = DrawnCrowd.rows(crowd.training, names)
        held_out = DrawnCrowd.rows(crowd.held_out, names)
        max_features = math.isqrt(len(names))

        tree = DecisionTreeClassifier().fit(training.features, training.target)
        forest = RandomForestClassifier(
            n_members=DRAWN_FOREST_MEMBERS,
            max_features=max_features,
            random_seed=DRAWN_FOREST_SEED,
        ).fit(training.features, training.target)

        arrangements.append(
            Arrangement(
                name=name,
                feature_names=list(names),
                max_features=max_features,
                tree=_arrangement_model(tree, names, training, held_out),
                forest=_arrangement_model(forest, names, training, held_out),
            )
        )

    return DrawnCrowdResponse(
        n_rows=DRAWN_ROWS,
        n_held_out=DRAWN_ROWS,
        column_names=list(DRAWN_COLUMNS),
        correlations_with_height=correlations,
        adult_share=float(crowd.training["is_adult"].mean()),
        arrangements=arrangements,
    )


@router.get(
    "/concepts/feature-importance/drawn-crowd", response_model=DrawnCrowdResponse
)
def drawn_crowd() -> DrawnCrowdResponse:
    """The drawn crowd, fitted in every arrangement of its columns."""
    return _drawn_crowd()


# --- The edges: what the library does at each boundary ----------------------


class EdgeProbe(BaseModel):
    """What happened when one edge was tried.

    ``outcome`` is ``refused`` when the library raised, with the error's type
    and its own words, and ``accepted`` otherwise, with a note on what came
    back.
    """

    edge: str
    outcome: Literal["refused", "accepted"]
    error: str | None
    detail: str


class EdgesResponse(BaseModel):
    """Every edge the page's closing table lists, probed."""

    probes: list[EdgeProbe]


def _probe(edge: str, attempt) -> EdgeProbe:
    """Run one attempt and record whether the library refused it."""
    try:
        detail = attempt()
    except (MLLibError, ValidationError) as error:
        return EdgeProbe(
            edge=edge, outcome="refused", error=type(error).__name__, detail=str(error)
        )
    return EdgeProbe(edge=edge, outcome="accepted", error=None, detail=detail)


def _small_rows() -> Rows:
    """Eight people, four of each class, on which a tree grows two splits."""
    return Rows(
        [
            Feature("height", [120.0, 125.0, 130.0, 135.0, 160.0, 165.0, 170.0, 175.0]),
            Feature("weight", [25.0, 28.0, 60.0, 62.0, 30.0, 33.0, 70.0, 72.0]),
        ],
        Feature("is_adult", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )


def _fitted_small_tree() -> DecisionTreeClassifier:
    rows = _small_rows()
    return DecisionTreeClassifier().fit(rows.features, rows.target)


def _empty_feature() -> str:
    Feature("height", [])
    return "a feature of no values was built"


def _one_row() -> str:
    DecisionTreeClassifier().fit(
        [Feature("height", [150.0]), Feature("weight", [50.0])],
        Feature("is_adult", [1.0]),
    )
    return "a tree was fitted on one row"


def _one_class() -> str:
    rows = _small_rows()
    DecisionTreeClassifier().fit(rows.features, Feature("is_adult", [1.0] * 8))
    return "a tree was fitted on one class"


def _every_column_constant() -> str:
    rows = _small_rows()
    tree = DecisionTreeClassifier().fit(
        [Feature("height", [150.0] * 8), Feature("weight", [50.0] * 8)], rows.target
    )
    _ = tree.feature_importances
    return "shares were reported for a tree that never split"


def _one_constant_column() -> str:
    rows = _small_rows()
    tree = DecisionTreeClassifier().fit(
        [*rows.features, Feature("shoe_size", [40.0] * 8)], rows.target
    )
    return f"accepted; the constant column's share is {tree.feature_importances['shoe_size']}"


def _non_finite_value() -> str:
    Feature("height", [150.0, float("nan"), 160.0])
    return "a feature holding a non-finite value was built"


def _no_repeats() -> str:
    PermutationImportance(n_repeats=0)
    return "a scramble of zero repeats was configured"


def _unfitted_importances() -> str:
    _ = DecisionTreeClassifier().feature_importances
    return "shares were read from an unfitted tree"


def _unfitted_scramble() -> str:
    rows = _small_rows()
    PermutationImportance(random_seed=0).measure(
        DecisionTreeClassifier(), rows.features, rows.target
    )
    return "an unfitted tree was scrambled"


def _renamed_column() -> str:
    rows = _small_rows()
    tree = _fitted_small_tree()
    renamed = [Feature("stature", rows.features[0].values), rows.features[1]]
    PermutationImportance(random_seed=0).measure(tree, renamed, rows.target)
    return "a scramble ran with a column the fit never saw"


def _reordered_columns() -> str:
    rows = _small_rows()
    tree = _fitted_small_tree()
    reordered = [rows.features[1], rows.features[0]]
    importances = PermutationImportance(random_seed=0).measure(
        tree, reordered, rows.target
    )
    return "accepted; the shares come back in the order handed, " + ", ".join(
        f"{one.name} {one.value:.3f}" for one in importances
    )


def _unknown_name() -> str:
    _fitted_small_tree().feature_importances["age"]
    return "a share was read for a feature the fit never saw"


def _nothing_drops() -> str:
    rows = _small_rows()
    tree = DecisionTreeClassifier().fit(
        [Feature("height", [150.0] * 8), Feature("weight", [50.0] * 8)], rows.target
    )
    PermutationImportance(random_seed=0).measure(tree, rows.features, rows.target)
    return "a scramble reported shares for a model no column could hurt"


def _members_that_cannot_report() -> str:
    heights = Feature("height", [150.0, 155.0, 160.0, 165.0, 170.0, 175.0])
    weights = Feature("weight", [50.0, 54.0, 57.0, 63.0, 66.0, 71.0])
    committee = BaggingRegressor(
        n_members=3, base_model=RidgeRegression(penalty=1.0), random_seed=0
    ).fit([heights], weights)
    _ = committee.feature_importances
    return "a bagged ridge committee reported impurity shares"


def _duplicate_names() -> str:
    rows = _small_rows()
    DecisionTreeClassifier().fit(
        [rows.features[0], Feature("height", rows.features[1].values)], rows.target
    )
    return "a tree was fitted on two columns of one name"


def _max_features_past_the_count() -> str:
    rows = _small_rows()
    forest = RandomForestClassifier(n_members=3, max_features=5, random_seed=0).fit(
        rows.features, rows.target
    )
    return (
        "accepted; every split is offered both columns, and the shares are "
        + ", ".join(f"{one.name} {one.value:.3f}" for one in forest.feature_importances)
    )


EDGE_ATTEMPTS = (
    ("a feature of no values", _empty_feature),
    ("one row", _one_row),
    ("one class only", _one_class),
    ("a non-finite value", _non_finite_value),
    ("every column constant, then the split reading", _every_column_constant),
    ("one constant column beside two real ones", _one_constant_column),
    ("every column constant, then the scramble reading", _nothing_drops),
    ("zero scramble repeats", _no_repeats),
    ("the split reading before fit", _unfitted_importances),
    ("the scramble reading before fit", _unfitted_scramble),
    ("a scramble with a column renamed", _renamed_column),
    ("a scramble with the columns reordered", _reordered_columns),
    ("a share asked for by a name the fit never saw", _unknown_name),
    ("two columns of one name", _duplicate_names),
    ("more features per split than there are features", _max_features_past_the_count),
    ("a committee whose members have no splits to read", _members_that_cannot_report),
)
"""Every edge the page's table lists, and how to try it."""


@cache
def _edges() -> EdgesResponse:
    return EdgesResponse(
        probes=[_probe(edge, attempt) for edge, attempt in EDGE_ATTEMPTS]
    )


@router.get("/concepts/feature-importance/edges", response_model=EdgesResponse)
def edges() -> EdgesResponse:
    """What the library does at each boundary, tried rather than remembered."""
    return _edges()
