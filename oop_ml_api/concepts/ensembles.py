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

from functools import cache
from itertools import combinations

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    BaggingClassifier,
    BaggingRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Feature,
    FeatureSet,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RegressionEvaluation,
    RowBlock,
)
from oop_ml.core.exceptions import MLLibError
from oop_ml.core.tree.node import DecisionNode, LeafNode
from oop_ml.core.tree.weights import WeightedTargets
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import CurvePoint, Point
from oop_ml_api.concepts.split import held_out_indices
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
                "by round, and a committee at such a rate is refused rather "
                "than built"
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


MAX_CURVE_CROWD = 500
"""The largest crowd the leave-out curve is traced up to."""

MAX_CURVE_MEMBERS = 200
"""The largest committee the variance curves are traced up to."""

VARIANCE_CORRELATIONS = (0.0, 0.3, 0.6)
"""The member correlations the variance chart draws, one curve each."""


class LeaveOutCurveRequest(BaseModel):
    """How large a crowd to trace the bootstrap's missing share up to."""

    max_crowd: int = Field(ge=2, le=MAX_CURVE_CROWD)

    model_config = {"extra": "forbid"}


class LeaveOutCurveResponse(BaseModel):
    """The chance one person dodges every draw, at each crowd size.

    ``limit`` is 1/e, the number the curve settles toward as the crowd grows.
    """

    sizes: list[int]
    probabilities: list[float]
    limit: float


@router.post("/leave-out-curve", response_model=LeaveOutCurveResponse)
def leave_out_curve(request: LeaveOutCurveRequest) -> LeaveOutCurveResponse:
    """Trace (1 - 1/n)^n, the bagging page's missing third, against n.

    Formula rather than library arithmetic, the same standing the primers
    have: the page derives this expression and the chart draws it settling.
    """
    sizes = list(range(2, request.max_crowd + 1))
    probabilities = [float((1.0 - 1.0 / size) ** size) for size in sizes]
    return LeaveOutCurveResponse(
        sizes=sizes,
        probabilities=probabilities,
        limit=float(np.exp(-1.0)),
    )


class CommitteeVarianceRequest(BaseModel):
    """How large a committee to trace the variance curves up to."""

    max_members: int = Field(ge=2, le=MAX_CURVE_MEMBERS)

    model_config = {"extra": "forbid"}


class VarianceCurve(BaseModel):
    """One correlation's curve, the committee variance at each size."""

    correlation: float
    variances: list[float]


class CommitteeVarianceResponse(BaseModel):
    """The committee variance formula traced for each drawn correlation.

    A member's own variance is taken as 1, so every number is a share of it,
    and each curve's floor is simply its correlation.
    """

    members: list[int]
    curves: list[VarianceCurve]


@router.post("/committee-variance", response_model=CommitteeVarianceResponse)
def committee_variance(
    request: CommitteeVarianceRequest,
) -> CommitteeVarianceResponse:
    """Trace rho + (1 - rho)/B for each correlation the chart draws."""
    members = list(range(1, request.max_members + 1))
    curves = [
        VarianceCurve(
            correlation=correlation,
            variances=[
                float(correlation + (1.0 - correlation) / size) for size in members
            ],
        )
        for correlation in VARIANCE_CORRELATIONS
    ]
    return CommitteeVarianceResponse(members=members, curves=curves)


class BoostingDescentRequest(BaseModel):
    """The points, and the boosting run whose leftovers are to be traced."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    learning_rate: float = Field(gt=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class BoostingDescentResponse(BaseModel):
    """The residual sum of squares after each round, round zero included.

    Round zero is the flat mean model the sequence starts from, so the first
    entry is the target's total spread and every later one is what the
    corrections had left.
    """

    rounds: list[int]
    residual_sums: list[float]


@router.post("/boosting-descent", response_model=BoostingDescentResponse)
def boosting_descent(request: BoostingDescentRequest) -> BoostingDescentResponse:
    """Replay a boosting fit round by round and measure each leftover.

    One model is fitted at the full round count; the running predictions are
    then rebuilt by walking its members in order from the initial mean, the
    same accumulation ``predict`` performs, so the last entry agrees with the
    fitted model exactly.
    """
    x_values = [point.x for point in request.points]
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)
    inputs = Feature("t", x_values)

    model = GradientBoostingRegressor(
        n_rounds=request.rounds, learning_rate=request.learning_rate, max_depth=1
    ).fit([inputs], Feature("h", [float(value) for value in y_values]))

    running = np.full(len(x_values), float(model.initial_prediction))
    residual_sums = [float(np.sum((y_values - running) ** 2))]
    for member in model.members:
        member_answers = np.asarray(member.predict([inputs]), dtype=np.float64)
        running = running + request.learning_rate * member_answers
        residual_sums.append(float(np.sum((y_values - running) ** 2)))

    return BoostingDescentResponse(
        rounds=list(range(len(residual_sums))),
        residual_sums=residual_sums,
    )


MAX_ANATOMY_MEMBERS = 25
"""The largest committee the anatomy endpoint lays out member by member."""


class AnatomyRequest(BaseModel):
    """The crowd, how many trees, and how deep each may grow.

    ``member_depth`` left unset grows every member to full depth, the
    canonical bagged tree. Set to one it grows stumps, which is how the page
    shows that averaging cannot supply flexibility no member has.
    """

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    n_members: int = Field(default=MAX_ANATOMY_MEMBERS, ge=1, le=MAX_ANATOMY_MEMBERS)
    member_depth: int | None = Field(default=None, ge=1, le=6)
    max_features: int | None = Field(default=None, ge=1, le=2)
    seed: int = Field(default=FIXED_SEED, ge=0, le=10_000)

    model_config = {"extra": "forbid"}


class MemberDocument(BaseModel):
    """One member of the committee, as the page reads it.

    ``draws`` is the bootstrap sample in the order it was drawn, as row
    positions into the request, so the page can replay the drawing.
    ``multiplicities`` counts how many times each row was drawn, and
    ``predictions`` is the member's own answer for every row of the crowd,
    whether or not it trained on that row.
    """

    position: int
    draws: list[int]
    multiplicities: list[int]
    distinct_rows: int
    omitted_rows: int
    root_feature: str | None
    root_threshold: float | None
    depth: int
    n_leaves: int
    predictions: list[int]
    regions: RegionGrid


class CommitteeSize(BaseModel):
    """The committee cut to its first ``n_members`` trees, refitted and scored.

    ``vote_share`` is, at every lattice cell, the share of the members
    voting adult, so the page can shade agreement separately from the
    winning class. ``changed_cells`` counts the cells whose majority differs
    from the committee one member smaller.
    """

    n_members: int
    train_accuracy: float
    out_of_bag_accuracy: float | None
    out_of_bag_covered: int
    changed_cells: int
    regions: RegionGrid
    vote_share: list[list[float]]


class OutOfBagRow(BaseModel):
    """One person's two verdicts: the whole committee's, and their judges'.

    ``eligible`` lists the members whose bootstrap sample omitted this
    person, the only members allowed to judge them out of bag.
    """

    label: int
    eligible: list[int]
    out_of_bag_adult_votes: int
    out_of_bag_prediction: int | None
    full_adult_votes: int
    full_prediction: int


class AnatomyResponse(BaseModel):
    """Everything the bagging and forest pages draw, from one seeded committee.

    ``member_strengths`` scores each member on the rows its own sample
    omitted. ``pairwise_agreement`` and ``error_correlation`` are averaged
    over every pair of members, each pair measured on the rows both of them
    omitted, so no member is judged on a row it memorised.
    """

    n_rows: int
    members: list[MemberDocument]
    sizes: list[CommitteeSize]
    out_of_bag: list[OutOfBagRow]
    roots_on_height: int
    roots_on_weight: int
    expected_omitted: float
    expected_distinct: float
    member_strengths: list[float | None]
    mean_strength: float | None
    pairwise_agreement: float | None
    error_correlation: float | None


Committee = BaggingClassifier | RandomForestClassifier


def _committee(
    n_members: int,
    member_depth: int | None,
    max_features: int | None,
    seed: int = FIXED_SEED,
) -> Committee:
    """A bagged committee, or a forest when a feature restriction is asked for."""
    if max_features is not None:
        return RandomForestClassifier(
            n_members=n_members,
            random_seed=seed,
            max_features=max_features,
            max_depth=member_depth,
        )
    if member_depth is None:
        return BaggingClassifier(n_members=n_members, random_seed=seed)
    return BaggingClassifier(
        n_members=n_members,
        random_seed=seed,
        base_model=DecisionTreeClassifier(max_depth=member_depth),
    )


def _diversity(
    votes: np.ndarray, in_bag: np.ndarray, labels: np.ndarray
) -> tuple[list[float | None], float | None, float | None, float | None]:
    """Member strength, and how alike the members are, all measured out of bag.

    ``votes`` is ``(n_rows, n_members)`` and ``in_bag`` the same shape. A
    pair's agreement is the share of rows both omitted on which they answer
    alike, and its error correlation is the correlation of their two
    wrong-or-not indicators over those rows, skipped where either member
    was never wrong there, since a correlation of a constant is undefined.
    """
    n_members = votes.shape[1]
    strengths: list[float | None] = []
    for member in range(n_members):
        judged = ~in_bag[:, member]
        strengths.append(
            None
            if not judged.any()
            else float(np.mean(votes[judged, member] == labels[judged]))
        )
    present = [value for value in strengths if value is not None]
    mean_strength = float(np.mean(present)) if present else None

    agreements: list[float] = []
    correlations: list[float] = []
    for first, second in combinations(range(n_members), 2):
        both = ~in_bag[:, first] & ~in_bag[:, second]
        if np.sum(both) < 3:
            continue
        agreements.append(float(np.mean(votes[both, first] == votes[both, second])))
        wrong_first = (votes[both, first] != labels[both]).astype(np.float64)
        wrong_second = (votes[both, second] != labels[both]).astype(np.float64)
        if wrong_first.std() == 0 or wrong_second.std() == 0:
            continue
        correlations.append(float(np.corrcoef(wrong_first, wrong_second)[0, 1]))

    return (
        strengths,
        mean_strength,
        float(np.mean(agreements)) if agreements else None,
        float(np.mean(correlations)) if correlations else None,
    )


def _vote_share(model: Committee, regions: RegionGrid) -> list[list[float]]:
    """The share of members voting adult at every cell of the region lattice."""
    x_centres = np.linspace(regions.x_min, regions.x_max, regions.cells)
    y_centres = np.linspace(regions.y_min, regions.y_max, regions.cells)
    lattice_x, lattice_y = np.meshgrid(x_centres, y_centres)
    shares = np.asarray(
        model.predict_probabilities(
            [Feature("height", lattice_x.ravel()), Feature("weight", lattice_y.ravel())]
        ),
        dtype=np.float64,
    )[:, 1].reshape(regions.cells, regions.cells)
    return [[float(value) for value in row] for row in shares]


@router.post("/bagging-anatomy", response_model=AnatomyResponse)
def bagging_anatomy(request: AnatomyRequest) -> AnatomyResponse:
    """Fit the seeded committee and lay out every part the page reads.

    The committee draws its resamples from one seeded generator, so a
    committee of ``k`` members is the first ``k`` members of a larger one.
    Each size is refitted through the library rather than read off a prefix,
    so every score on the sweep is the library's own, and a test pins that
    the two agree.
    """
    if len({point.label for point in request.points}) < 2:
        raise HTTPException(
            status_code=422,
            detail="both classes have to appear before a committee has anything to vote on",
        )

    heights, weights, labels = features_of(request.points)
    label_values = np.asarray(labels.values, dtype=np.int64)
    n_rows = label_values.size

    full = _committee(
        request.n_members, request.member_depth, request.max_features, request.seed
    ).fit([heights, weights], labels)

    members: list[MemberDocument] = []
    member_predictions: list[np.ndarray] = []
    for position, (member, sample) in enumerate(
        zip(full.members, full.samples, strict=True)
    ):
        draws = [int(index) for index in sample.drawn]
        multiplicities = np.bincount(sample.drawn, minlength=n_rows)
        predictions = np.asarray(member.predict([heights, weights]), dtype=np.int64)
        member_predictions.append(predictions)
        root = member.root
        has_split = hasattr(root, "split")
        members.append(
            MemberDocument(
                position=position,
                draws=draws,
                multiplicities=[int(value) for value in multiplicities],
                distinct_rows=int(np.sum(multiplicities > 0)),
                omitted_rows=int(np.sum(multiplicities == 0)),
                root_feature=root.split.feature_name if has_split else None,
                root_threshold=float(root.split.threshold) if has_split else None,
                depth=int(member.depth),
                n_leaves=int(member.n_leaves),
                predictions=[int(value) for value in predictions],
                regions=region_grid(member, request.points),
            )
        )

    sizes: list[CommitteeSize] = []
    previous_labels: list[list[int]] | None = None
    for n_members in range(1, request.n_members + 1):
        committee = _committee(
            n_members, request.member_depth, request.max_features, request.seed
        ).fit([heights, weights], labels)
        estimate = committee.out_of_bag_estimate()
        covered = int(np.sum(estimate.covered))
        regions = region_grid(committee, request.points)
        changed = 0
        if previous_labels is not None:
            changed = int(
                sum(
                    1
                    for row_now, row_before in zip(
                        regions.labels, previous_labels, strict=True
                    )
                    for now, before in zip(row_now, row_before, strict=True)
                    if now != before
                )
            )
        previous_labels = regions.labels
        sizes.append(
            CommitteeSize(
                n_members=n_members,
                train_accuracy=float(committee.score([heights, weights], labels)),
                out_of_bag_accuracy=(
                    float(committee.out_of_bag_score()) if covered > 0 else None
                ),
                out_of_bag_covered=covered,
                changed_cells=changed,
                regions=regions,
                vote_share=_vote_share(committee, regions),
            )
        )

    in_bag = np.column_stack(
        [np.asarray(sample.in_bag, dtype=bool) for sample in full.samples]
    )
    votes = np.column_stack(member_predictions)
    out_of_bag: list[OutOfBagRow] = []
    for row in range(n_rows):
        eligible = [int(index) for index in np.flatnonzero(~in_bag[row])]
        oob_adult = int(np.sum(votes[row, eligible] == 1)) if eligible else 0
        full_adult = int(np.sum(votes[row] == 1))
        out_of_bag.append(
            OutOfBagRow(
                label=int(label_values[row]),
                eligible=eligible,
                out_of_bag_adult_votes=oob_adult,
                out_of_bag_prediction=(
                    None if not eligible else int(oob_adult * 2 > len(eligible))
                ),
                full_adult_votes=full_adult,
                full_prediction=int(full_adult * 2 > request.n_members),
            )
        )

    root_features = [member.root_feature for member in members]
    expected_omitted = float(n_rows * (1 - 1 / n_rows) ** n_rows)
    strengths, mean_strength, agreement, correlation = _diversity(
        votes, in_bag, label_values
    )

    return AnatomyResponse(
        n_rows=n_rows,
        members=members,
        sizes=sizes,
        out_of_bag=out_of_bag,
        roots_on_height=root_features.count("height"),
        roots_on_weight=root_features.count("weight"),
        expected_omitted=expected_omitted,
        expected_distinct=n_rows - expected_omitted,
        member_strengths=strengths,
        mean_strength=mean_strength,
        pairwise_agreement=agreement,
        error_correlation=correlation,
    )


class LotteryRequest(BaseModel):
    """The crowd, how many features each split may see, and how many draws."""

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    max_features: int = Field(default=1, ge=1, le=2)
    n_draws: int = Field(default=10, ge=1, le=50)
    seed: int = Field(default=FIXED_SEED, ge=0, le=10_000)

    model_config = {"extra": "forbid"}


class FeatureBest(BaseModel):
    """The best question one feature can ask at the root, and what it earns."""

    feature: str
    threshold: float
    gain: float


class LotteryDraw(BaseModel):
    """One draw of the feature lottery at the root."""

    offered: list[str]
    withheld: list[str]
    winner: str
    winner_gain: float
    best_denied: bool


class LotteryResponse(BaseModel):
    """The full candidate board at the root, and the lottery replayed on it.

    The draws use the same rule the library's tree applies at every node,
    ``max_features`` names chosen without replacement from a seeded
    generator, replayed here so the page can show them side by side. The
    library does not keep a record of the draws inside a grown tree, so
    these are the rule demonstrated rather than a particular tree's history.
    """

    board: list[FeatureBest]
    draws: list[LotteryDraw]


def _root_board(
    heights: Feature, weights: Feature, labels: Feature
) -> list[FeatureBest]:
    """The best root question each feature can ask, scored by the library's search."""
    tree = DecisionTreeClassifier(max_depth=1).fit([heights, weights], labels)
    block = RowBlock(
        FeatureSet([heights, weights]).feature_matrix, ["height", "weight"]
    )
    search = tree.split_search(block, WeightedTargets(labels.column))

    best: dict[str, FeatureBest] = {}
    for candidate in search:
        name = candidate.split.feature_name
        gain = float(candidate.split.gain)
        if name not in best or gain > best[name].gain:
            best[name] = FeatureBest(
                feature=name, threshold=float(candidate.split.threshold), gain=gain
            )
    return [best[name] for name in ("height", "weight") if name in best]


@router.post("/forest-lottery", response_model=LotteryResponse)
def forest_lottery(request: LotteryRequest) -> LotteryResponse:
    """Score every feature's best root question, then replay the lottery."""
    heights, weights, labels = features_of(request.points)
    board = _root_board(heights, weights, labels)
    strongest = max(board, key=lambda each: each.gain).feature

    generator = np.random.default_rng(request.seed)
    names = [each.feature for each in board]
    draws: list[LotteryDraw] = []
    for _ in range(request.n_draws):
        chosen = sorted(
            generator.choice(
                len(names), size=min(request.max_features, len(names)), replace=False
            )
        )
        offered = [names[index] for index in chosen]
        winner = max(
            (each for each in board if each.feature in offered),
            key=lambda each: each.gain,
        )
        draws.append(
            LotteryDraw(
                offered=offered,
                withheld=[name for name in names if name not in offered],
                winner=winner.feature,
                winner_gain=winner.gain,
                best_denied=strongest not in offered,
            )
        )

    return LotteryResponse(board=board, draws=draws)


class CompetitionRequest(BaseModel):
    """The crowd and how many bootstrap samples to hold the root competition on."""

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    n_samples: int = Field(default=8, ge=1, le=25)
    seed: int = Field(default=FIXED_SEED, ge=0, le=10_000)

    model_config = {"extra": "forbid"}


class SampleCompetition(BaseModel):
    """One bootstrap sample's root board, and which feature won it."""

    position: int
    distinct_rows: int
    board: list[FeatureBest]
    winner: str | None


class CompetitionResponse(BaseModel):
    """The root competition held on each of the committee's first samples.

    Every sample is a different set of rows, so every board is different,
    and the question is whether the same feature keeps winning anyway. The
    samples are the ones the library's bagged committee draws under the
    given seed, so they are the samples the bagging page's members grew on.
    """

    samples: list[SampleCompetition]
    wins: dict[str, int]


@router.post("/split-competition", response_model=CompetitionResponse)
def split_competition(request: CompetitionRequest) -> CompetitionResponse:
    """Hold the root competition on each of the first bootstrap samples."""
    heights, weights, labels = features_of(request.points)
    committee = BaggingClassifier(
        n_members=request.n_samples, random_seed=request.seed
    ).fit([heights, weights], labels)

    samples: list[SampleCompetition] = []
    wins: dict[str, int] = {"height": 0, "weight": 0}
    for position, sample in enumerate(committee.samples):
        drawn = np.asarray(sample.drawn)
        board = _root_board(
            Feature("height", np.asarray(heights.values)[drawn]),
            Feature("weight", np.asarray(weights.values)[drawn]),
            Feature("is_adult", np.asarray(labels.values)[drawn]),
        )
        winner = max(board, key=lambda each: each.gain).feature if board else None
        if winner is not None:
            wins[winner] += 1
        samples.append(
            SampleCompetition(
                position=position,
                distinct_rows=len(np.unique(drawn)),
                board=board,
                winner=winner,
            )
        )
    return CompetitionResponse(samples=samples, wins=wins)


class SameSampleRequest(BaseModel):
    """One bootstrap sample, and the seeds of the feature lotteries to grow on it."""

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    seeds: list[int] = Field(default=[1, 2], min_length=2, max_length=4)

    model_config = {"extra": "forbid"}


class SameSampleTree(BaseModel):
    """A tree grown on the shared sample under one feature-lottery seed."""

    seed: int
    root_feature: str | None
    root_threshold: float | None
    depth: int
    n_leaves: int
    predictions: list[int]
    regions: RegionGrid


class SameSampleResponse(BaseModel):
    """The shared sample's multiplicities, the trees, and how far they agree."""

    multiplicities: list[int]
    unrestricted_root: str | None
    trees: list[SameSampleTree]
    agreement: float


@router.post("/forest-same-sample", response_model=SameSampleResponse)
def forest_same_sample(request: SameSampleRequest) -> SameSampleResponse:
    """Hold the bootstrap sample fixed and vary only the feature lottery."""
    heights, weights, labels = features_of(request.points)
    sample = (
        BaggingClassifier(n_members=1, random_seed=FIXED_SEED)
        .fit([heights, weights], labels)
        .samples[0]
    )
    drawn = np.asarray(sample.drawn)
    resampled = [
        Feature("height", np.asarray(heights.values)[drawn]),
        Feature("weight", np.asarray(weights.values)[drawn]),
    ]
    resampled_labels = Feature("is_adult", np.asarray(labels.values)[drawn])

    unrestricted = DecisionTreeClassifier().fit(resampled, resampled_labels)
    trees: list[SameSampleTree] = []
    for seed in request.seeds:
        tree = DecisionTreeClassifier(max_features=1, random_seed=seed).fit(
            resampled, resampled_labels
        )
        root = tree.root
        has_split = hasattr(root, "split")
        trees.append(
            SameSampleTree(
                seed=seed,
                root_feature=root.split.feature_name if has_split else None,
                root_threshold=float(root.split.threshold) if has_split else None,
                depth=int(tree.depth),
                n_leaves=int(tree.n_leaves),
                predictions=[
                    int(value) for value in np.asarray(tree.predict([heights, weights]))
                ],
                regions=region_grid(tree, request.points),
            )
        )

    first = np.asarray(trees[0].predictions)
    second = np.asarray(trees[1].predictions)
    unrestricted_root = unrestricted.root
    return SameSampleResponse(
        multiplicities=[
            int(value) for value in np.bincount(drawn, minlength=len(request.points))
        ],
        unrestricted_root=(
            unrestricted_root.split.feature_name
            if hasattr(unrestricted_root, "split")
            else None
        ),
        trees=trees,
        agreement=float(np.mean(first == second)),
    )


class SeedComparisonRequest(BaseModel):
    """The crowd and how many seeds to repeat the bagging-versus-forest comparison over."""

    points: list[LabelledPoint] = Field(min_length=4, max_length=MAX_POINTS)
    n_seeds: int = Field(default=8, ge=2, le=12)
    n_members: int = Field(default=MAX_ANATOMY_MEMBERS, ge=2, le=MAX_MEMBERS)

    model_config = {"extra": "forbid"}


class SeedResult(BaseModel):
    """One seed's pair of committees."""

    seed: int
    bagging_out_of_bag: float
    forest_out_of_bag: float
    bagging_roots_on_height: int
    forest_roots_on_height: int


class SeedComparisonResponse(BaseModel):
    """Every seed's result, and the mean difference between the two routes."""

    results: list[SeedResult]
    mean_difference: float


@router.post("/forest-seeds", response_model=SeedComparisonResponse)
def forest_seeds(request: SeedComparisonRequest) -> SeedComparisonResponse:
    """Repeat the comparison under several seeds, so one census is not a verdict."""
    heights, weights, labels = features_of(request.points)
    results: list[SeedResult] = []
    for seed in range(1, request.n_seeds + 1):
        bagged = BaggingClassifier(n_members=request.n_members, random_seed=seed).fit(
            [heights, weights], labels
        )
        forest = RandomForestClassifier(
            n_members=request.n_members, random_seed=seed, max_features=1
        ).fit([heights, weights], labels)
        results.append(
            SeedResult(
                seed=seed,
                bagging_out_of_bag=float(bagged.out_of_bag_score()),
                forest_out_of_bag=float(forest.out_of_bag_score()),
                bagging_roots_on_height=sum(
                    1
                    for member in bagged.members
                    if hasattr(member.root, "split")
                    and member.root.split.feature_name == "height"
                ),
                forest_roots_on_height=sum(
                    1
                    for member in forest.members
                    if hasattr(member.root, "split")
                    and member.root.split.feature_name == "height"
                ),
            )
        )
    return SeedComparisonResponse(
        results=results,
        mean_difference=float(
            np.mean(
                [each.forest_out_of_bag - each.bagging_out_of_bag for each in results]
            )
        ),
    )


WORLD_ROWS = 120
"""How many rows each simulated world holds."""

WORLD_FEATURES = 6
"""How many features each simulated world offers, informative or not."""

WORLD_SEED = 4
"""The worlds are drawn once, from this seed."""

WORLD_MEMBERS = 100
"""How many trees each forest in the sweep grows.

Twenty-five is too few for the comparison to settle: measured at 25 the
two worlds' sweeps wobble by more than the effect. At 100 the direction is
stable across seeds 3, 4 and 5, which is why the response is computed once
and cached rather than made cheap.
"""

WORLD_NOISE = 0.5
"""The spread of the noise added to the score before it is thresholded."""

WORLD_WEIGHTS: dict[str, tuple[float, ...]] = {
    "sparse": (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "shared": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
}
"""How much of the score each feature carries, per world.

Sparse: one feature carries the whole signal and five are noise. Shared:
every feature carries an equal part of it, so a node denied one feature
is always offered another that matters. A third arrangement, three noisy
copies of one latent signal beside three noise columns, was measured and
behaved like the sparse world, which the page says in prose.
"""

WORLD_DESCRIPTIONS = {
    "sparse": "one feature carries the whole signal and five are pure noise",
    "shared": "all six features carry an equal part of the signal",
}


class WorldPoint(BaseModel):
    """One max_features setting's result in one world."""

    max_features: int
    mean_strength: float | None
    error_correlation: float | None
    out_of_bag: float


class SignalWorld(BaseModel):
    """One simulated world and its sweep over max_features."""

    name: str
    description: str
    sweep: list[WorldPoint]


class SignalWorldsResponse(BaseModel):
    """The sparse world and the shared world, swept identically."""

    n_rows: int
    n_features: int
    n_members: int
    worlds: list[SignalWorld]


def _world(kind: str) -> tuple[list[Feature], Feature]:
    """Six standard-normal features and a label thresholded from a weighted score.

    Both worlds draw the same feature matrix from the same seed and differ
    only in the weights, so the rows are literally the same people.
    """
    generator = np.random.default_rng(WORLD_SEED)
    matrix = generator.normal(size=(WORLD_ROWS, WORLD_FEATURES))
    noise = generator.normal(scale=WORLD_NOISE, size=WORLD_ROWS)
    score = matrix @ np.asarray(WORLD_WEIGHTS[kind]) + noise
    features = [
        Feature(f"feature_{index + 1}", matrix[:, index])
        for index in range(WORLD_FEATURES)
    ]
    return features, Feature("label", (score > 0).astype(np.float64))


@cache
def _signal_worlds() -> SignalWorldsResponse:
    """Sweep max_features across both worlds, once per process."""
    worlds: list[SignalWorld] = []
    for kind, description in WORLD_DESCRIPTIONS.items():
        features, labels = _world(kind)
        label_values = np.asarray(labels.values, dtype=np.int64)
        sweep: list[WorldPoint] = []
        for max_features in range(1, WORLD_FEATURES + 1):
            forest = RandomForestClassifier(
                n_members=WORLD_MEMBERS,
                random_seed=FIXED_SEED,
                max_features=max_features,
            ).fit(features, labels)
            votes = np.column_stack(
                [
                    np.asarray(member.predict(features), dtype=np.int64)
                    for member in forest.members
                ]
            )
            in_bag = np.column_stack(
                [np.asarray(sample.in_bag, dtype=bool) for sample in forest.samples]
            )
            _, mean_strength, _, correlation = _diversity(votes, in_bag, label_values)
            sweep.append(
                WorldPoint(
                    max_features=max_features,
                    mean_strength=mean_strength,
                    error_correlation=correlation,
                    out_of_bag=float(forest.out_of_bag_score()),
                )
            )
        worlds.append(SignalWorld(name=kind, description=description, sweep=sweep))
    return SignalWorldsResponse(
        n_rows=WORLD_ROWS,
        n_features=WORLD_FEATURES,
        n_members=WORLD_MEMBERS,
        worlds=worlds,
    )


@router.get("/signal-worlds", response_model=SignalWorldsResponse)
def signal_worlds() -> SignalWorldsResponse:
    """Sweep max_features across a sparse world and a shared one."""
    return _signal_worlds()


# --- Gradient boosting laid out round by round, for the gradient boosting page

MAX_STEPPER_ROUNDS = 60
"""The most rounds the stepper lays out one at a time.

Every round carries its residuals, its member's curve and the running curve,
so the payload grows with the count, and the stepper is about the early rounds
where a single correction is still visible on the picture.
"""

MAX_MEMBER_DEPTH = 6
"""The deepest member a boosting request may ask for.

Six is already past unpruned on the throw: eleven training rows fit inside
sixteen leaves, so a depth of four can reproduce every one of them in a single
round, which is the comparison the depth section makes.
"""

MAX_CURVE_CONFIGURATIONS = 6
"""How many rate-and-depth configurations one learning-curve request sweeps."""

MIN_CURVE_POINTS = 7
"""The fewest points a held-out learning curve can be traced on.

The seeded deal keeps back three tenths of the rows, so seven points hold out
two, which is the fewest that leave any spread for a held-out score to explain.
"""

TRAIN_THRESHOLD = 0.95
"""The training score a configuration is timed against, in rounds."""

FITTED_TARGETS_SHOWN = 3
"""How many members' fitted targets the bagging comparison lays out."""

RATE_PAST_ONE = (
    "past a learning rate of 1 each round's correction overshoots the very "
    "mistake it just fitted, so the fit gets worse round by round, and a "
    "committee at such a rate is refused rather than built"
)


def _refuse_rate_past_one(learning_rate: float) -> None:
    """The same refusal the older endpoint makes, in the same words."""
    if learning_rate > 1.0:
        raise HTTPException(status_code=422, detail=RATE_PAST_ONE)


class _Throw:
    """The request's points as arrays and as the library's two named features."""

    __slots__ = ("inputs", "target", "x_values", "y_values")

    def __init__(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        self.x_values = x_values
        self.y_values = y_values
        self.inputs = Feature("t", x_values)
        self.target = Feature("h", y_values)

    @classmethod
    def of(cls, points: list[Point]) -> _Throw:
        return cls(
            np.asarray([point.x for point in points], dtype=np.float64),
            np.asarray([point.y for point in points], dtype=np.float64),
        )

    def rows(self, positions: np.ndarray) -> _Throw:
        """The same throw cut to these row positions."""
        return _Throw(self.x_values[positions], self.y_values[positions])


def _curve_grid(x_values: np.ndarray) -> np.ndarray:
    return np.linspace(float(x_values.min()), float(x_values.max()), CURVE_SAMPLES)


def _curve_points(grid: np.ndarray, heights: object) -> list[CurvePoint]:
    values = np.asarray(heights, dtype=np.float64)
    return [
        CurvePoint(x=float(x), y=float(y)) for x, y in zip(grid, values, strict=True)
    ]


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(RegressionEvaluation(actual, predicted).r2_score)


def _member_answers(member: object, throw: _Throw) -> np.ndarray:
    return np.asarray(member.predict([throw.inputs]), dtype=np.float64)  # type: ignore[attr-defined]


class BoostingModelRequest(BaseModel):
    """The points, the rounds, the rate, and how deep each member may grow.

    The rate is accepted up to 2 so the playground's slider can cross 1 and
    meet the refusal in words, as the older endpoint does; the library itself
    will not construct a model past 1.
    """

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    rounds: int = Field(ge=1, le=MAX_ROUNDS)
    learning_rate: float = Field(gt=0.0, le=2.0)
    max_depth: int = Field(default=1, ge=1, le=MAX_MEMBER_DEPTH)

    model_config = {"extra": "forbid"}


class BoostingModelResponse(BaseModel):
    """The assembled fit, scored on its own rows and sampled for drawing.

    ``initial_prediction`` is the flat start every round corrects,
    ``total_leaves`` counts the shelves across every member, and
    ``residual_sum_of_squares`` is what the rounds left unexplained.
    """

    initial_prediction: float
    r_squared: float
    residual_sum_of_squares: float
    total_leaves: int
    predictions: list[float]
    curve: list[CurvePoint]


@router.post("/boosting-model", response_model=BoostingModelResponse)
def boosting_model(request: BoostingModelRequest) -> BoostingModelResponse:
    """Fit the boosted committee at the requested depth and read it out."""
    _refuse_rate_past_one(request.learning_rate)
    throw = _Throw.of(request.points)
    model = GradientBoostingRegressor(
        n_rounds=request.rounds,
        learning_rate=request.learning_rate,
        max_depth=request.max_depth,
    ).fit([throw.inputs], throw.target)

    grid = _curve_grid(throw.x_values)
    fitted = _member_answers(model, throw)
    evaluation = RegressionEvaluation(throw.y_values, fitted)
    return BoostingModelResponse(
        initial_prediction=float(model.initial_prediction),
        r_squared=float(evaluation.r2_score),
        residual_sum_of_squares=float(evaluation.residual_sum_of_squares),
        total_leaves=sum(int(member.n_leaves) for member in model.members),  # type: ignore[attr-defined]
        predictions=[float(value) for value in fitted],
        curve=_curve_points(grid, model.predict([Feature("t", grid)])),
    )


class BoostingRoundsRequest(BaseModel):
    """The points and the run the stepper walks through one round at a time."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    rounds: int = Field(ge=1, le=MAX_STEPPER_ROUNDS)
    learning_rate: float = Field(gt=0.0, le=1.0)
    max_depth: int = Field(default=1, ge=1, le=MAX_MEMBER_DEPTH)

    model_config = {"extra": "forbid"}


class RoundDocument(BaseModel):
    """One round of the fit, laid out as the page steps through it.

    ``residuals_before`` is what the committee so far had wrong on every row
    when the round began, which is exactly the target its member was fitted
    to. For a stump the member is one question, ``threshold`` with
    ``left_value`` answering strictly below it and ``right_value`` at or
    above; a deeper member reports its root's threshold and leaves the two
    values unset, since it answers with more than two shelves.
    ``member_curve`` is the member's own answer along the grid before the rate
    is applied, and ``running_curve`` the committee's answer once the round
    is folded in.
    """

    number: int
    residuals_before: list[float]
    residual_sum_before: float
    threshold: float | None
    left_value: float | None
    right_value: float | None
    n_leaves: int
    member_predictions: list[float]
    member_curve: list[float]
    running_predictions: list[float]
    running_curve: list[float]
    residual_sum_after: float
    r_squared_after: float


class BoostingRoundsResponse(BaseModel):
    """The flat start, the grid the curves are sampled on, and every round.

    ``residual_sum_at_start`` is the squared leftover of the flat mean, the
    target's own total spread, which is what round one begins from.
    """

    initial_prediction: float
    residual_sum_at_start: float
    grid: list[float]
    rounds: list[RoundDocument]


def _stump_reading(member: object) -> tuple[float | None, float | None, float | None]:
    """A member's root threshold, and its two leaf values when it is a stump."""
    root = member.root  # type: ignore[attr-defined]
    if not isinstance(root, DecisionNode):
        return None, None, None
    threshold = float(root.split.threshold)
    if isinstance(root.left, LeafNode) and isinstance(root.right, LeafNode):
        return threshold, float(root.left.prediction), float(root.right.prediction)
    return threshold, None, None


@router.post("/boosting-rounds", response_model=BoostingRoundsResponse)
def boosting_rounds(request: BoostingRoundsRequest) -> BoostingRoundsResponse:
    """Fit once, then replay the members in order and record each round.

    The replay is the same accumulation ``predict`` performs, from the initial
    mean, so the last round's running predictions agree with the fitted
    model exactly, and a test pins that they do.
    """
    throw = _Throw.of(request.points)
    model = GradientBoostingRegressor(
        n_rounds=request.rounds,
        learning_rate=request.learning_rate,
        max_depth=request.max_depth,
    ).fit([throw.inputs], throw.target)

    grid = _curve_grid(throw.x_values)
    grid_throw = _Throw(grid, np.zeros_like(grid))
    running = np.full(throw.y_values.size, float(model.initial_prediction))
    running_curve = np.full(grid.size, float(model.initial_prediction))
    at_start = float(np.sum((throw.y_values - running) ** 2))

    rounds: list[RoundDocument] = []
    for number, member in enumerate(model.members, start=1):
        residuals = throw.y_values - running
        member_predictions = _member_answers(member, throw)
        member_curve = _member_answers(member, grid_throw)
        running = running + request.learning_rate * member_predictions
        running_curve = running_curve + request.learning_rate * member_curve
        threshold, left_value, right_value = _stump_reading(member)
        rounds.append(
            RoundDocument(
                number=number,
                residuals_before=[float(value) for value in residuals],
                residual_sum_before=float(np.sum(residuals**2)),
                threshold=threshold,
                left_value=left_value,
                right_value=right_value,
                n_leaves=int(member.n_leaves),  # type: ignore[attr-defined]
                member_predictions=[float(value) for value in member_predictions],
                member_curve=[float(value) for value in member_curve],
                running_predictions=[float(value) for value in running],
                running_curve=[float(value) for value in running_curve],
                residual_sum_after=float(np.sum((throw.y_values - running) ** 2)),
                r_squared_after=_r_squared(throw.y_values, running),
            )
        )

    return BoostingRoundsResponse(
        initial_prediction=float(model.initial_prediction),
        residual_sum_at_start=at_start,
        grid=[float(value) for value in grid],
        rounds=rounds,
    )


class CurveConfiguration(BaseModel):
    """One rate and one member depth to trace a learning curve for."""

    learning_rate: float = Field(gt=0.0, le=1.0)
    max_depth: int = Field(default=1, ge=1, le=MAX_MEMBER_DEPTH)

    model_config = {"extra": "forbid"}


class BoostingCurvesRequest(BaseModel):
    """The points, how far to run, and which configurations to run.

    The held-out rows are the same seeded deal the evaluation page makes,
    three tenths of the points under seed 4, so the round to stop at is read
    off rows the fit never saw.
    """

    points: list[Point] = Field(min_length=MIN_CURVE_POINTS, max_length=MAX_POINTS)
    max_rounds: int = Field(ge=1, le=MAX_ROUNDS)
    configurations: list[CurveConfiguration] = Field(
        min_length=1, max_length=MAX_CURVE_CONFIGURATIONS
    )

    model_config = {"extra": "forbid"}


class LearningCurve(BaseModel):
    """One configuration's two scores after every round, round zero included.

    Round zero is the flat start, the training mean, whose training score is
    zero by definition and whose held-out score is whatever that mean is
    worth on rows it never saw. ``best_round`` is the first round at which
    the held-out score reached its highest value, and ``rounds_to_train`` the
    first round at which the training score passed the threshold, null if it
    never did.
    """

    learning_rate: float
    max_depth: int
    train_r_squared: list[float]
    held_out_r_squared: list[float]
    best_round: int
    best_held_out_r_squared: float
    final_train_r_squared: float
    final_held_out_r_squared: float
    rounds_to_train: int | None


class BoostingCurvesResponse(BaseModel):
    """The deal, the training threshold, and one curve per configuration."""

    held_out_indices: list[int]
    n_training: int
    n_held_out: int
    train_threshold: float
    curves: list[LearningCurve]


def _learning_curve(
    training: _Throw,
    held_out: _Throw,
    configuration: CurveConfiguration,
    max_rounds: int,
) -> LearningCurve:
    """Fit once at the full round count and score every prefix of it.

    A boosted fit of ``k`` rounds is the first ``k`` members of a longer one,
    because each round depends only on the rounds before it, so one fit
    replayed gives every round's scores and a test pins the prefix claim.
    """
    model = GradientBoostingRegressor(
        n_rounds=max_rounds,
        learning_rate=configuration.learning_rate,
        max_depth=configuration.max_depth,
    ).fit([training.inputs], training.target)

    running_train = np.full(training.y_values.size, float(model.initial_prediction))
    running_held = np.full(held_out.y_values.size, float(model.initial_prediction))
    train_scores = [_r_squared(training.y_values, running_train)]
    held_scores = [_r_squared(held_out.y_values, running_held)]
    for member in model.members:
        running_train = running_train + configuration.learning_rate * _member_answers(
            member, training
        )
        running_held = running_held + configuration.learning_rate * _member_answers(
            member, held_out
        )
        train_scores.append(_r_squared(training.y_values, running_train))
        held_scores.append(_r_squared(held_out.y_values, running_held))

    best_round = int(np.argmax(held_scores))
    passed = [
        number for number, score in enumerate(train_scores) if score >= TRAIN_THRESHOLD
    ]
    return LearningCurve(
        learning_rate=configuration.learning_rate,
        max_depth=configuration.max_depth,
        train_r_squared=train_scores,
        held_out_r_squared=held_scores,
        best_round=best_round,
        best_held_out_r_squared=held_scores[best_round],
        final_train_r_squared=train_scores[-1],
        final_held_out_r_squared=held_scores[-1],
        rounds_to_train=passed[0] if passed else None,
    )


def _dealt(points: list[Point]) -> tuple[_Throw, _Throw, np.ndarray]:
    """The throw cut into its training rows and its held-out rows."""
    throw = _Throw.of(points)
    held = held_out_indices(len(points))
    training_positions = np.setdiff1d(np.arange(len(points)), held)
    return throw.rows(training_positions), throw.rows(held), held


@router.post("/boosting-curves", response_model=BoostingCurvesResponse)
def boosting_curves(request: BoostingCurvesRequest) -> BoostingCurvesResponse:
    """Trace training and held-out scores against rounds, per configuration."""
    training, held_out, held = _dealt(request.points)
    return BoostingCurvesResponse(
        held_out_indices=[int(index) for index in held],
        n_training=int(training.y_values.size),
        n_held_out=int(held_out.y_values.size),
        train_threshold=TRAIN_THRESHOLD,
        curves=[
            _learning_curve(training, held_out, configuration, request.max_rounds)
            for configuration in request.configurations
        ],
    )


class VersusRequest(BaseModel):
    """The points, the committee size, and the boosting rate and member depth.

    The bagged committees and the boosted one all get ``n_members`` members,
    fitted on the same training rows of the same seeded deal, so the three
    are compared on rows none of them saw.
    """

    points: list[Point] = Field(min_length=MIN_CURVE_POINTS, max_length=MAX_POINTS)
    n_members: int = Field(default=25, ge=1, le=MAX_MEMBERS)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    max_depth: int = Field(default=1, ge=1, le=MAX_MEMBER_DEPTH)

    model_config = {"extra": "forbid"}


class FittedTarget(BaseModel):
    """What one member was actually fitted to.

    ``times`` and ``targets`` are the rows the member saw and the value it was
    asked to predict at each, so a bagged member lists resampled heights with
    repeats and a boosted member lists every training time against the
    residual its round began with. ``distinct_rows`` counts the rows without
    their repeats.
    """

    position: int
    times: list[float]
    targets: list[float]
    distinct_rows: int


class CommitteeScores(BaseModel):
    """One committee's scores, its curve, and what its first members fitted.

    ``out_of_bag_r_squared`` exists only for the bagged committees, because
    a boosted round sees every row and leaves none out to judge it.
    """

    name: str
    train_r_squared: float
    held_out_r_squared: float
    out_of_bag_r_squared: float | None
    curve: list[CurvePoint]
    fitted_targets: list[FittedTarget]


class VersusResponse(BaseModel):
    """Deep bagged trees, bagged stumps and boosted stumps on one deal."""

    held_out_indices: list[int]
    training_times: list[float]
    training_heights: list[float]
    bagged_deep: CommitteeScores
    bagged_stumps: CommitteeScores
    boosted: CommitteeScores


def _out_of_bag_or_none(model: BaggingRegressor) -> float | None:
    """The bagged committee's out-of-bag score, or null where it is undefined.

    At one or two members most rows have no judge, and the library's own
    route refuses a score with nothing to explain, so the null is the honest
    reading rather than a number.
    """
    try:
        return float(model.out_of_bag_score())
    except MLLibError:
        return None


def _bagged_scores(
    name: str,
    model: BaggingRegressor,
    training: _Throw,
    held_out: _Throw,
    grid: np.ndarray,
) -> CommitteeScores:
    fitted = model.fit([training.inputs], training.target)
    targets: list[FittedTarget] = []
    for position, sample in enumerate(fitted.samples[:FITTED_TARGETS_SHOWN]):
        drawn = np.asarray(sample.drawn)
        targets.append(
            FittedTarget(
                position=position,
                times=[float(value) for value in training.x_values[drawn]],
                targets=[float(value) for value in training.y_values[drawn]],
                distinct_rows=int(np.unique(drawn).size),
            )
        )
    return CommitteeScores(
        name=name,
        train_r_squared=_r_squared(
            training.y_values, _member_answers(fitted, training)
        ),
        held_out_r_squared=_r_squared(
            held_out.y_values, _member_answers(fitted, held_out)
        ),
        out_of_bag_r_squared=_out_of_bag_or_none(fitted),
        curve=_curve_points(grid, fitted.predict([Feature("t", grid)])),
        fitted_targets=targets,
    )


def _boosted_scores(
    request: VersusRequest, training: _Throw, held_out: _Throw, grid: np.ndarray
) -> CommitteeScores:
    model = GradientBoostingRegressor(
        n_rounds=request.n_members,
        learning_rate=request.learning_rate,
        max_depth=request.max_depth,
    ).fit([training.inputs], training.target)

    running = np.full(training.y_values.size, float(model.initial_prediction))
    targets: list[FittedTarget] = []
    for position, member in enumerate(model.members[:FITTED_TARGETS_SHOWN]):
        residuals = training.y_values - running
        targets.append(
            FittedTarget(
                position=position,
                times=[float(value) for value in training.x_values],
                targets=[float(value) for value in residuals],
                distinct_rows=int(training.x_values.size),
            )
        )
        running = running + request.learning_rate * _member_answers(member, training)

    return CommitteeScores(
        name="boosted stumps",
        train_r_squared=_r_squared(training.y_values, _member_answers(model, training)),
        held_out_r_squared=_r_squared(
            held_out.y_values, _member_answers(model, held_out)
        ),
        out_of_bag_r_squared=None,
        curve=_curve_points(grid, model.predict([Feature("t", grid)])),
        fitted_targets=targets,
    )


@router.post("/boosting-versus-bagging", response_model=VersusResponse)
def boosting_versus_bagging(request: VersusRequest) -> VersusResponse:
    """Three committees of the same size on the same deal, scored alike.

    The deep bagged trees are the canonical bagged member, the bagged stumps
    are the boosted member handed to the wrong frame, and the boosted stumps
    are the page's own model, so the comparison separates what the member
    contributes from what the frame does.
    """
    training, held_out, held = _dealt(request.points)
    grid = _curve_grid(_Throw.of(request.points).x_values)
    return VersusResponse(
        held_out_indices=[int(index) for index in held],
        training_times=[float(value) for value in training.x_values],
        training_heights=[float(value) for value in training.y_values],
        bagged_deep=_bagged_scores(
            "bagged deep trees",
            BaggingRegressor(n_members=request.n_members, random_seed=FIXED_SEED),
            training,
            held_out,
            grid,
        ),
        bagged_stumps=_bagged_scores(
            "bagged stumps",
            BaggingRegressor(
                n_members=request.n_members,
                random_seed=FIXED_SEED,
                base_model=DecisionTreeRegressor(max_depth=request.max_depth),
            ),
            training,
            held_out,
            grid,
        ),
        boosted=_boosted_scores(request, training, held_out, grid),
    )
