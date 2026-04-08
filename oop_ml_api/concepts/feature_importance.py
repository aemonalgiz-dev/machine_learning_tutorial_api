"""Which feature mattered, for the feature-importance page.

One endpoint draws the parity puzzle, two coin-flip columns whose disagreement
is the class and a third column of pure noise that offers the split search a
threshold between every pair of its values, and fits the library's lone
``DecisionTreeClassifier`` and its ``RandomForestClassifier`` on it. Each
fitted model is then asked the page's two questions. What its own splits
credit comes from ``feature_importances``, and what breaks when a column is
scrambled comes from ``PermutationImportance``. Both readings go back with the
model's accuracy on the rows it learned and on as many rows it never saw.

Two recorded routes ride along so the page can do the arithmetic by hand. The
lone tree's decision nodes come back with their rows and their gain, which is
everything the impurity shares are made of. And the scramble walk is one loop
over the model's public ``score``, shuffling each column in turn with the same
seeded draws the library's own measure makes, so the drops it records
normalise to exactly the shares the library reports, and a test says so.

Both readings are taken on the training rows, the library's own reading, and
deliberately so. The lone tree scores at chance on rows it never saw, and a
column cannot lower a score that is already at chance, so a held-out reading
would have nothing to say about the model whose reliance the page most wants
to show. Held-out accuracy is reported beside it, which is what tells the
reader which of the two models actually works.

A forest member restricted to one column at its root can be handed a coin
whose split buys exactly nothing, and a member that grew no split has no
shares for the forest to average. The library refuses that in its own words,
which are about one member and read as if they were about the forest, so the
endpoint checks for such a member first and refuses in a sentence a reader can
act on. Measured over one hundred and eighty draws across the widget's range,
four grew one.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DecisionNode,
    DecisionTreeClassifier,
    Feature,
    FeatureImportances,
    PermutationImportance,
    RandomForestClassifier,
    TreeNode,
)
from pydantic import BaseModel, Field

router = APIRouter(tags=["feature-importance"])

MIN_ROWS = 200
"""The fewest training rows a request may draw.

The page's finding needs a forest that works, and below this it often does
not. Measured over twenty draws at each size, the forest's held-out accuracy
fell below 0.9 on eleven draws at a hundred rows, nine at a hundred and fifty,
and two at two hundred, where its median was 0.965.
"""

MAX_ROWS = 400
"""The most training rows a request may draw. As many again are held out.

A cost bound rather than a teaching one. The endpoint measured 0.17 seconds at
this size, and nothing on the page needs more rows than the worked draw's
three hundred.
"""

MAX_SEED = 9999
"""The largest seed the redraw control may reach."""

FEATURE_NAMES = ("first", "second", "distractor")
"""The two coin flips whose disagreement is the class, and the column of
noise, in the order every fit sees them."""

TARGET_NAME = "parity"
"""What the class column is called."""

LONE_TREE_DEPTH = 3
"""Deep enough for parity, which takes two questions, and far short of
memorising the rows, which an unstopped tree does and which would hide the
point. The forest's members grow to the same depth."""

FOREST_MEMBERS = 20
"""How many trees the forest grows."""

FOREST_SEED = 0
"""Fixes the forest's resamples and its per-node feature draws, so the page
can quote its shares. Only the data's own draw follows the request's seed."""

SHUFFLE_SEED = 0
"""Fixes every scramble, so the permutation reading is reproducible and the
page can quote it."""

SHUFFLE_REPEATS = 5
"""How many times each column is scrambled before the drops are averaged."""


class MeasureRequest(BaseModel):
    """How many rows to draw, and which draw."""

    n_rows: int = Field(ge=MIN_ROWS, le=MAX_ROWS)
    seed: int = Field(ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class ImportanceShare(BaseModel):
    """One feature's share of a model's explanation, in the fit's order."""

    name: str
    share: float


class SplitCredit(BaseModel):
    """One decision node of the lone tree, and the credit it hands its feature.

    ``credit`` is ``n_rows`` times ``gain``, the library's own weighting, so a
    question asked of every row counts for more than one asked of a dozen.
    """

    feature: str
    depth: int
    n_rows: int
    gain: float
    credit: float


class FeatureCredit(BaseModel):
    """Everything one feature's questions earned in the lone tree, summed."""

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


class ModelReport(BaseModel):
    """Both importance readings for one fitted model, and its two accuracies.

    ``drop_total`` is the sum of the clamped drops, the number every
    permutation share is divided by.
    """

    training_accuracy: float
    held_out_accuracy: float
    impurity: list[ImportanceShare]
    permutation: list[ImportanceShare]
    leading_by_impurity: str
    leading_by_permutation: str
    intact_score: float
    shuffles: list[ShuffleStep]
    drop_total: float


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


@router.post("/concepts/feature-importance/measure", response_model=MeasureResponse)
def measure(request: MeasureRequest) -> MeasureResponse:
    """Fit a lone tree and a forest on the parity puzzle and ask both questions."""
    draw = ParityDraw(request.n_rows, request.seed)

    lone_tree = DecisionTreeClassifier(max_depth=LONE_TREE_DEPTH).fit(
        draw.training.features, draw.training.target
    )
    forest = RandomForestClassifier(
        n_members=FOREST_MEMBERS,
        max_features=1,
        max_depth=LONE_TREE_DEPTH,
        random_seed=FOREST_SEED,
    ).fit(draw.training.features, draw.training.target)
    _refuse_a_forest_with_a_member_that_never_split(forest, request.seed)

    splits: list[SplitCredit] = []
    _collect_splits(lone_tree.root, 0, splits)
    credits = _credits_by_feature(splits)

    return MeasureResponse(
        n_rows=request.n_rows,
        feature_names=list(FEATURE_NAMES),
        lone_tree=LoneTreeReport(
            **_report(lone_tree, draw).model_dump(),
            splits=splits,
            credits=credits,
            credit_total=sum(credit.credit for credit in credits),
        ),
        forest=_report(forest, draw),
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


def _report(
    model: DecisionTreeClassifier | RandomForestClassifier, draw: ParityDraw
) -> ModelReport:
    """Both readings and both accuracies for one fitted model."""
    impurity = model.feature_importances
    permutation = PermutationImportance(
        n_repeats=SHUFFLE_REPEATS, random_seed=SHUFFLE_SEED
    ).measure(model, draw.training.features, draw.training.target)

    intact_score = float(model.score(draw.training.features, draw.training.target))
    shuffles = _shuffle_walk(model, draw, intact_score)

    return ModelReport(
        training_accuracy=intact_score,
        held_out_accuracy=float(
            model.score(draw.held_out.features, draw.held_out.target)
        ),
        impurity=_shares(impurity),
        permutation=_shares(permutation),
        leading_by_impurity=impurity.most_important.name,
        leading_by_permutation=permutation.most_important.name,
        intact_score=intact_score,
        shuffles=shuffles,
        drop_total=sum(max(0.0, step.drop) for step in shuffles),
    )


def _shares(importances: FeatureImportances) -> list[ImportanceShare]:
    """The shares in the order the fit saw the features."""
    return [
        ImportanceShare(name=importance.name, share=float(importance.value))
        for importance in importances
    ]


def _shuffle_walk(
    model: DecisionTreeClassifier | RandomForestClassifier,
    draw: ParityDraw,
    intact_score: float,
) -> list[ShuffleStep]:
    """Scramble each column in turn and record what the score became.

    One loop over the model's public ``score``, drawing every permutation from
    one generator seeded once, columns in fit order and the repeats of each
    column together, which is the order the library's own measure draws in.
    """
    generator = np.random.default_rng(SHUFFLE_SEED)
    steps: list[ShuffleStep] = []

    for position, feature in enumerate(draw.training.features):
        shuffled_scores: list[float] = []
        for _ in range(SHUFFLE_REPEATS):
            scrambled = [
                Feature(feature.name, generator.permutation(feature.values))
                if index == position
                else other
                for index, other in enumerate(draw.training.features)
            ]
            shuffled_scores.append(float(model.score(scrambled, draw.training.target)))

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


def _credits_by_feature(splits: list[SplitCredit]) -> list[FeatureCredit]:
    """Each feature's credits summed, in fit order, zero where it won nothing."""
    totals = dict.fromkeys(FEATURE_NAMES, 0.0)
    for split in splits:
        totals[split.feature] += split.credit

    return [FeatureCredit(name=name, credit=totals[name]) for name in FEATURE_NAMES]
