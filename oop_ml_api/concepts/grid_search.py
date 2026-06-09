"""Searching for a setting, for the grid-search page.

A model has dials no fit can turn, and the obvious move is to try every
setting and keep the best. Every endpoint here hands that move to the
library's ``GridSearch``, and every one hands back each candidate's score
rather than only the winner's, because the losing scores are what tell a
reader whether the winner meant anything.

The search endpoint takes the reader's people, heights against weights, and
searches the ``n_neighbours`` dial of ``KNearestNeighboursRegressor``. A
quarter of them is held out by a seeded shuffle before the search begins,
the rest are dealt into seeded folds, every candidate is cross-validated on
that one deal, and the winner is refitted on the searched rows and scored
once on the held-out ones, which is the library's ``honest_score_on``. The
fold-by-fold scores and predictions come from a second pass of
``CrossValidation`` over the identical folds, so the page can walk one fold
by hand and a test can check that the two routes agree. Each candidate is
also scored on the very rows it was fitted to, because that is the number a
fit would choose the dial by if it were allowed to, and at one neighbour it
is exactly one.

The same-folds endpoint is the detector for a mistake the library once made.
Two copies of one setting go through one search whose splitter is unseeded,
and they tie to the last bit because the search draws one seed and pins it;
the same setting cross-validated on deals drawn under different seeds does
not tie, and the spread of those is what a search without the pin would have
been comparing its candidates across.

The space endpoint builds a search space from a parameter name the reader
typed, so the page can show what the library does with a misspelling: it is
refused before any data is seen, with the fields the model does have named
in the message. A value the model's own constructor refuses is refused the
same way, because every candidate is rebuilt through that constructor rather
than copied.

The grid endpoint searches two dials at once, the degree of a polynomial in
height and the penalty of the ridge fit that follows it, as one pipeline.
The candidates are every combination of the two ranges, which is what makes
it a grid, and the count of fits is the product of the ranges' lengths and
the fold count, which is what makes a grid expensive.

The optimism endpoint runs the library's own measurement of selection bias.
On a target that is pure noise, where no setting can genuinely help, the
winner among the candidates is compared with the same setting scored again
on fresh fold deals it was never chosen on. The re-score uses the same ruler
as the selection, cross-validation over the same rows, rather than a fresh
holdout, because the two rulers do not agree with each other. Five folds of
sixteen rows read every setting a little low, and measured over twenty draws
the winner sat 0.05 above a fresh-fold re-score and 0.02 below a
four-hundred-row holdout, so a holdout would hide the flattering inside the
difference between rulers. The re-score is averaged over several deals
because one deal carries as much noise as the number it is checking, and the
whole experiment is repeated over a run of seeds because a single draw's
optimism can land on either side of zero. The candidate count and the
target are the reader's, so the page can show the bias growing with the
number of candidates and shrinking on a target with signal in it.

The nested endpoints are the fix the library does not yet have as an object.
A second layer of folds is dealt round the whole search, the search runs
inside each outer training fold, and the winner is scored on the outer fold
it never saw. That loop is written here over the library's ``KFold`` and
``GridSearch``, one outer fold at a time, and every score inside it is the
library's own.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Candidate,
    CrossValidation,
    Dataset,
    Feature,
    GridSearch,
    KFold,
    KNearestNeighboursRegressor,
    PipelineSteps,
    PolynomialFeatures,
    RegressionPipeline,
    RidgeRegression,
    SearchResult,
    SearchSpace,
    Standardizer,
)
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.polynomial_regression import Point
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["grid-search"])

MAX_FOLDS = 10
"""The most folds the widget may deal."""

MAX_NEIGHBOURS = 25
"""The largest neighbour count a candidate may try."""

HELD_OUT_FRACTION = 0.25
"""The share of the points kept back from the search, to judge the winner."""

SPLIT_SEED = 1
"""The held-out shuffle is seeded so the page can name the rows it holds out."""

FOLD_SEED = 3
"""The fold deal is seeded so every candidate sees the same folds, and so the
page can name which rows sit in which fold."""

MAX_DEGREE = 4
"""The highest polynomial degree the two-dial grid may try."""

MAX_PENALTY = 1000.0
"""The heaviest ridge penalty the two-dial grid may try."""

MAX_GRID_SIDE = 6
"""The most values either range of the two-dial grid may hold."""

NOISE_ROWS = 80
"""How many rows the pure-noise target has, matching the library's measurement."""

NOISE_FEATURES = 4
"""How many columns the noise and signal targets are searched over."""

NOISE_SEED = 12
"""The draw the page quotes."""

NOISE_CANDIDATES = 25
"""The winner is by default the best of this many neighbour counts."""

NOISE_FOLDS = 5
"""The library's measurement cross-validated on five folds."""

FIRST_FRESH_DEAL_SEED = 101
"""The first of the fresh fold deals the winner is re-scored on."""

FRESH_DEALS = 12
"""How many fresh deals the re-score is averaged over."""

DRAWS = 12
"""How many consecutive seeds the optimism is averaged over."""

MAX_SEED = 10_000
"""The largest seed the endpoints accept."""

OUTER_SEED = 21
"""The outer deal of the nested loop on the noise target."""

MAX_OUTER_FOLDS = 5
"""The most outer folds the nested endpoint deals round the reader's people."""

MAX_SAME_FOLD_DEALS = 10
"""The most fresh deals the same-folds detector compares."""

MODEL_TYPE = KNearestNeighboursRegressor
"""The model every neighbour search here varies."""

PARAMETER = "n_neighbours"
"""The dial every neighbour search here turns."""


class _SearchDeal:
    """The reader's people, a quarter held out, the rest ready to be searched.

    ``held`` and ``searched`` are positions into the request's points, sorted
    ascending, and ``dataset`` and ``holdout`` are the two shares as the
    library reads them.
    """

    __slots__ = ("dataset", "held", "holdout", "searched")

    def __init__(self, points: list[Point]) -> None:
        heights = np.asarray([point.x for point in points], dtype=np.float64)
        weights = np.asarray([point.y for point in points], dtype=np.float64)
        self.held = _held_out_indices(heights.size)
        self.searched = np.setdiff1d(np.arange(heights.size), self.held)
        self.dataset = Dataset(
            [Feature("height", heights[self.searched])],
            Feature("weight", weights[self.searched]),
        )
        self.holdout = Dataset(
            [Feature("height", heights[self.held])],
            Feature("weight", weights[self.held]),
        )


def _held_out_indices(n_points: int) -> np.ndarray:
    """The seeded shuffle-and-cut, by position, smallest indices first."""
    generator = np.random.default_rng(SPLIT_SEED)
    shuffled = generator.permutation(n_points)
    n_held = max(1, int(n_points * HELD_OUT_FRACTION))
    return np.sort(shuffled[:n_held])


def _fold_positions(n_rows: int, folds: KFold) -> list[list[int]]:
    """Which rows each fold holds out, in the fold's own row order.

    The rows' positions are folded as if they were data, so the deal reported
    is the deal the search used rather than a re-derivation of it, and the
    order inside a fold is the order the fold's predictions come back in.
    """
    positions = np.arange(n_rows, dtype=np.float64)
    return [
        [int(position) for position in split.testing.input_features[0].values]
        for split in folds.split(
            Dataset([Feature("position", positions)], Feature("target", positions))
        )
    ]


def _neighbour_space(neighbour_counts: list[int]) -> SearchSpace:
    """The dial and the values to try for it, checked against the model."""
    return SearchSpace.over(MODEL_TYPE, n_neighbours=neighbour_counts)


def _search_neighbours(
    dataset: Dataset, folds: KFold, neighbour_counts: list[int]
) -> SearchResult:
    """Cross-validate every neighbour count on one deal and keep the best."""
    return GridSearch(folds=folds).search(
        MODEL_TYPE(), _neighbour_space(neighbour_counts), dataset
    )


def _neighbour_count(candidate: Candidate) -> int:
    """The one value a neighbour candidate assigns."""
    return int(candidate.value_for(PARAMETER))


def _mean_candidate_score(result: SearchResult) -> float:
    """The average over every candidate, the yardstick the winner beats."""
    return float(np.mean([scored.score for scored in result]))


def _training_score(dataset: Dataset, n_neighbours: int) -> float:
    """A setting fitted on the searched rows and scored on those same rows.

    The number a fit would choose the dial by if it were allowed to, and at
    one neighbour it is exactly one, because every row's nearest neighbour
    is itself.
    """
    model = MODEL_TYPE(n_neighbours=n_neighbours).fit(
        dataset.input_features, dataset.target_feature
    )
    return float(model.score(dataset.input_features, dataset.target_feature))


class CandidateScore(BaseModel):
    """One setting, its cross-validated score, and the folds behind it.

    ``fold_scores`` and ``fold_predictions`` are in fold order, and each
    fold's predictions are in that fold's own row order, which the response's
    ``fold_members`` names. ``training_score`` is the setting fitted on every
    searched row and scored on those same rows.
    """

    n_neighbours: int
    score: float
    fold_scores: list[float]
    fold_predictions: list[list[float]]
    training_score: float


class SearchRequest(BaseModel):
    """The points, the neighbour counts to try, and how many folds to deal."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_neighbours_from: int = Field(ge=1, le=MAX_NEIGHBOURS)
    n_neighbours_to: int = Field(ge=1, le=MAX_NEIGHBOURS)
    n_folds: int = Field(ge=2, le=MAX_FOLDS)

    model_config = {"extra": "forbid"}


class SearchResponse(BaseModel):
    """Every candidate's score, the winner, the spread, and the honest score.

    ``held_out_indices`` are positions into the request's points.
    ``fold_members`` holds, per fold, the positions of the people that fold
    judges, in the order the fold's predictions come back in, and
    ``fold_of`` is the same deal read the other way, the fold each point was
    dealt into or ``None`` for a held-out point. ``ranked_n_neighbours`` is
    every candidate best first, ties keeping the earlier one.
    ``same_rows_score`` is the winner refitted on the searched rows and scored
    on those same rows, the score the honest one must not be confused with.
    ``n_fits`` is the number of models the search fitted, candidates times
    folds.
    """

    held_out_indices: list[int]
    fold_members: list[list[int]]
    fold_of: list[int | None]
    candidates: list[CandidateScore]
    n_candidates: int
    n_fits: int
    ranked_n_neighbours: list[int]
    best_n_neighbours: int
    best_score: float
    mean_candidate_score: float
    score_spread: float
    honest_score: float
    same_rows_score: float


def _candidate_scores(
    result: SearchResult, folds: KFold, dataset: Dataset
) -> list[CandidateScore]:
    """Every candidate's score, paired with the fold scores that averaged to it.

    The search reports only the mean. A second pass of cross-validation over
    the identical seeded folds recovers the per-fold figures, and a test
    asserts that the mean of the folds is the score the search reported.
    """
    scores: list[CandidateScore] = []
    for scored in result:
        n_neighbours = _neighbour_count(scored.candidate)
        evaluated = CrossValidation(folds=folds).evaluate(
            MODEL_TYPE(n_neighbours=n_neighbours), dataset
        )
        scores.append(
            CandidateScore(
                n_neighbours=n_neighbours,
                score=float(scored.score),
                fold_scores=[float(evaluation.r2_score) for evaluation in evaluated],
                fold_predictions=[
                    [float(value) for value in evaluation.predicted_values]
                    for evaluation in evaluated
                ],
                training_score=_training_score(dataset, n_neighbours),
            )
        )
    return scores


@router.post("/concepts/grid-search/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Hold a quarter out, search the rest, and score the winner honestly."""
    deal = _SearchDeal(request.points)
    folds = KFold(n_folds=request.n_folds, random_seed=FOLD_SEED)
    neighbour_counts = list(
        range(request.n_neighbours_from, request.n_neighbours_to + 1)
    )
    result = _search_neighbours(deal.dataset, folds, neighbour_counts)

    fold_members = [
        [int(deal.searched[position]) for position in fold]
        for fold in _fold_positions(deal.searched.size, folds)
    ]
    fold_of: list[int | None] = [None] * len(request.points)
    for fold_index, members in enumerate(fold_members):
        for position in members:
            fold_of[position] = fold_index

    winner = _neighbour_count(result.best.candidate)
    return SearchResponse(
        held_out_indices=[int(index) for index in deal.held],
        fold_members=fold_members,
        fold_of=fold_of,
        candidates=_candidate_scores(result, folds, deal.dataset),
        n_candidates=result.n_candidates,
        n_fits=result.n_candidates * request.n_folds,
        ranked_n_neighbours=[
            _neighbour_count(scored.candidate) for scored in result.ranked()
        ],
        best_n_neighbours=winner,
        best_score=float(result.best_score),
        mean_candidate_score=_mean_candidate_score(result),
        score_spread=float(result.score_spread),
        honest_score=float(
            result.honest_score_on(MODEL_TYPE(), deal.dataset, deal.holdout)
        ),
        same_rows_score=_training_score(deal.dataset, winner),
    )


# --- The same folds for every candidate, and the detector for it ------------


class SameFoldsRequest(BaseModel):
    """One setting, and how many fresh deals to compare it across."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_neighbours: int = Field(ge=1, le=MAX_NEIGHBOURS)
    n_folds: int = Field(ge=2, le=MAX_FOLDS)
    first_seed: int = Field(default=0, ge=0, le=MAX_SEED)
    n_deals: int = Field(default=6, ge=2, le=MAX_SAME_FOLD_DEALS)

    model_config = {"extra": "forbid"}


class SameFoldsResponse(BaseModel):
    """Two copies of one setting through one search, and the same setting on
    fresh deals.

    ``pinned_scores`` are the two copies' scores from a single search whose
    splitter was handed over unseeded; ``scores_tie`` is whether they are
    equal to the last bit. ``fresh_deal_scores`` are the same setting
    cross-validated once per seed in ``fresh_deal_seeds``, and
    ``fresh_deal_spread`` is the highest less the lowest, which is the noise
    a search without the pin would have been comparing across.
    """

    pinned_scores: list[float]
    scores_tie: bool
    fresh_deal_seeds: list[int]
    fresh_deal_scores: list[float]
    fresh_deal_spread: float


@router.post("/concepts/grid-search/same-folds", response_model=SameFoldsResponse)
def same_folds(request: SameFoldsRequest) -> SameFoldsResponse:
    """Search two copies of one setting on an unseeded splitter, then re-deal."""
    deal = _SearchDeal(request.points)
    result = _search_neighbours(
        deal.dataset,
        KFold(n_folds=request.n_folds),
        [request.n_neighbours, request.n_neighbours],
    )
    pinned = [float(scored.score) for scored in result]

    seeds = list(range(request.first_seed, request.first_seed + request.n_deals))
    fresh = [
        float(
            CrossValidation(folds=KFold(n_folds=request.n_folds, random_seed=seed))
            .evaluate(MODEL_TYPE(n_neighbours=request.n_neighbours), deal.dataset)
            .mean_r2_score
        )
        for seed in seeds
    ]
    return SameFoldsResponse(
        pinned_scores=pinned,
        scores_tie=pinned[0] == pinned[1],
        fresh_deal_seeds=seeds,
        fresh_deal_scores=fresh,
        fresh_deal_spread=float(max(fresh) - min(fresh)),
    )


# --- Building a space from a name the reader typed --------------------------


class SpaceRequest(BaseModel):
    """A parameter name, as typed, and the values to try for it.

    The values may be empty and may include zero, so the library's own
    refusals of an empty range and of a value the model rejects can be seen.
    """

    parameter_name: str = Field(min_length=1, max_length=40)
    values: list[int] = Field(max_length=MAX_NEIGHBOURS)

    model_config = {"extra": "forbid"}


class CandidateDocument(BaseModel):
    """One candidate's assignment, and the model it was rebuilt into."""

    assignments: dict[str, int]
    built_n_neighbours: int


class SpaceResponse(BaseModel):
    """What the space holds, and what the model it was built for has.

    ``declared_fields`` are the model's declared hyperparameters, the list a
    misspelling is checked against. Each candidate has been rebuilt through
    the model's constructor, and ``built_n_neighbours`` is the value the
    built model actually carries.
    """

    declared_fields: list[str]
    parameter_names: list[str]
    n_candidates: int
    candidates: list[CandidateDocument]


@router.post("/concepts/grid-search/space", response_model=SpaceResponse)
def space(request: SpaceRequest) -> SpaceResponse:
    """Build the space and rebuild every candidate through the constructor.

    A name the model does not have is the library's refusal and names what
    the model does have. A value the model's constructor rejects is caught
    when the candidate is rebuilt, before any fold would have been fitted.
    """
    built_space = SearchSpace.over(
        MODEL_TYPE, **{request.parameter_name: request.values}
    )
    prototype = MODEL_TYPE()
    documents: list[CandidateDocument] = []
    for candidate in built_space.candidates():
        try:
            model = candidate.applied_to(prototype)
        except ValidationError as error:
            message = error.errors()[0]["msg"]
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the candidate {candidate!r} was refused by the model's own "
                    f"constructor: {message}"
                ),
            ) from error
        documents.append(
            CandidateDocument(
                assignments={
                    name: int(value) for name, value in candidate.assignments.items()
                },
                built_n_neighbours=int(model.n_neighbours),
            )
        )
    return SpaceResponse(
        declared_fields=sorted(MODEL_TYPE.model_fields),
        parameter_names=list(built_space.parameter_names),
        n_candidates=built_space.n_candidates,
        candidates=documents,
    )


# --- Two dials at once: a polynomial degree and a ridge penalty --------------


class GridRequest(BaseModel):
    """The points, and the values of each of the two dials to try."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degrees: list[int] = Field(min_length=1, max_length=MAX_GRID_SIDE)
    penalties: list[float] = Field(min_length=1, max_length=MAX_GRID_SIDE)
    n_folds: int = Field(ge=2, le=MAX_FOLDS)

    model_config = {"extra": "forbid"}


class GridCell(BaseModel):
    """One combination of the two dials and what it scored."""

    degree: int
    penalty: float
    score: float


class GridResponse(BaseModel):
    """Every combination scored, the winner, and what the search cost.

    ``n_candidates`` is the product of the two ranges' lengths and ``n_fits``
    is that times the fold count, which is every model the search fitted.
    ``cells`` are in the order the search ran them, degree slowest.
    """

    n_searched: int
    n_held_out: int
    n_candidates: int
    n_fits: int
    cells: list[GridCell]
    best_degree: int
    best_penalty: float
    best_score: float
    worst_score: float
    score_spread: float
    honest_score: float


def _steps_at(degree: int) -> PipelineSteps:
    """Expand height to this degree, then standardise the expanded columns."""
    return PipelineSteps.of(
        terms=PolynomialFeatures(degree=degree), scaler=Standardizer()
    )


def _degree_of(candidate: Candidate) -> int:
    """Read the degree back off the whole steps object the candidate carried."""
    transformer = candidate.value_for("steps")["terms"].transformer
    assert isinstance(transformer, PolynomialFeatures)
    return transformer.degree


def _penalty_of(candidate: Candidate) -> float:
    """Read the penalty back off the whole model object the candidate carried."""
    model = candidate.value_for("model")
    assert isinstance(model, RidgeRegression)
    return float(model.penalty)


def _check_grid_ranges(request: GridRequest) -> None:
    """Refuse a degree or penalty outside what the page lets a reader try."""
    for degree in request.degrees:
        if not 1 <= degree <= MAX_DEGREE:
            raise HTTPException(
                status_code=422,
                detail=f"a degree must be between 1 and {MAX_DEGREE}, got {degree}",
            )
    for penalty in request.penalties:
        if not 0.0 < penalty <= MAX_PENALTY:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"a penalty must be above zero and at most {MAX_PENALTY:g}, "
                    f"got {penalty:g}"
                ),
            )


@router.post("/concepts/grid-search/grid", response_model=GridResponse)
def grid(request: GridRequest) -> GridResponse:
    """Search every combination of degree and penalty as one pipeline."""
    _check_grid_ranges(request)
    deal = _SearchDeal(request.points)
    prototype = RegressionPipeline(
        steps=_steps_at(1), model=RidgeRegression(penalty=request.penalties[0])
    )
    built_space = SearchSpace.over(
        RegressionPipeline,
        steps=[_steps_at(degree) for degree in request.degrees],
        model=[RidgeRegression(penalty=penalty) for penalty in request.penalties],
    )
    result = GridSearch(
        folds=KFold(n_folds=request.n_folds, random_seed=FOLD_SEED)
    ).search(prototype, built_space, deal.dataset)

    scores = [float(scored.score) for scored in result]
    return GridResponse(
        n_searched=int(deal.searched.size),
        n_held_out=int(deal.held.size),
        n_candidates=built_space.n_candidates,
        n_fits=built_space.n_candidates * request.n_folds,
        cells=[
            GridCell(
                degree=_degree_of(scored.candidate),
                penalty=_penalty_of(scored.candidate),
                score=float(scored.score),
            )
            for scored in result
        ],
        best_degree=_degree_of(result.best.candidate),
        best_penalty=_penalty_of(result.best.candidate),
        best_score=float(result.best_score),
        worst_score=float(min(scores)),
        score_spread=float(result.score_spread),
        honest_score=float(
            result.honest_score_on(prototype, deal.dataset, deal.holdout)
        ),
    )


# --- The winner's optimism, measured where nothing can be learned -----------

Target = Literal["noise", "signal"]


class OptimismRequest(BaseModel):
    """Which draw to search, how many settings to try, and what the target is.

    ``noise`` is a target with no relationship to its columns, where no
    setting can genuinely beat zero. ``signal`` is a target driven by two of
    the four columns with noise over the top, where the settings genuinely
    differ and the selection has something real to find.
    """

    seed: int = Field(ge=0, le=MAX_SEED)
    n_candidates: int = Field(default=NOISE_CANDIDATES, ge=2, le=NOISE_CANDIDATES)
    target: Target = "noise"

    model_config = {"extra": "forbid"}


class OptimismResponse(BaseModel):
    """What the winner reported, what it was worth, and the gap between them.

    ``refolded_score`` is the winner's neighbour count cross-validated again
    on fresh fold deals it was never chosen on, averaged over
    ``FRESH_DEALS`` of them. ``optimism`` is the winner's score minus that.
    ``mean_optimism_over_draws`` repeats the whole experiment on ``n_draws``
    consecutive seeds starting at the requested one and averages the gap,
    which is the number that survives any single draw landing the wrong side
    of zero. ``nested_score`` is the honest figure a second layer of folds
    gives on this draw, the search run inside each of five outer training
    folds and its winner scored on the outer fold it never saw, and
    ``nested_winners`` are the five winners those inner searches chose.
    """

    seed: int
    n_candidates: int
    target: Target
    candidates: list[CandidateScore]
    best_n_neighbours: int
    best_score: float
    mean_candidate_score: float
    score_spread: float
    refolded_score: float
    optimism: float
    n_draws: int
    mean_optimism_over_draws: float
    nested_score: float
    nested_winners: list[int]


class _DrawnColumns:
    """The four columns both targets are searched over, and the generator that
    drew them, still ready to draw the target."""

    __slots__ = ("generator", "matrix")

    def __init__(self, n_rows: int, seed: int) -> None:
        self.generator = np.random.default_rng(seed)
        self.matrix = self.generator.normal(size=(n_rows, NOISE_FEATURES))

    @property
    def features(self) -> list[Feature]:
        return [
            Feature(f"feature_{position}", self.matrix[:, position])
            for position in range(NOISE_FEATURES)
        ]


def _pure_noise(n_rows: int, seed: int) -> Dataset:
    """A target with no relationship to its features, the library's fixture."""
    drawn = _DrawnColumns(n_rows, seed)
    return Dataset(
        drawn.features, Feature("outcome", drawn.generator.normal(size=n_rows))
    )


def _signal(n_rows: int, seed: int) -> Dataset:
    """A target driven by two of the four columns, with noise over the top."""
    drawn = _DrawnColumns(n_rows, seed)
    outcome = (
        3.0 * drawn.matrix[:, 0]
        - 2.0 * drawn.matrix[:, 1]
        + drawn.generator.normal(scale=0.5, size=n_rows)
    )
    return Dataset(drawn.features, Feature("outcome", outcome))


def _drawn_target(target: Target, seed: int) -> Dataset:
    if target == "noise":
        return _pure_noise(NOISE_ROWS, seed)
    return _signal(NOISE_ROWS, seed)


def _noise_folds() -> KFold:
    """The one deal every candidate on a drawn target is scored on."""
    return KFold(n_folds=NOISE_FOLDS, random_seed=FOLD_SEED)


def _refolded_score(dataset: Dataset, n_neighbours: int) -> float:
    """The setting scored again on fresh deals, none of which chose it."""
    return float(
        np.mean(
            [
                CrossValidation(folds=KFold(n_folds=NOISE_FOLDS, random_seed=deal_seed))
                .evaluate(MODEL_TYPE(n_neighbours=n_neighbours), dataset)
                .mean_r2_score
                for deal_seed in range(
                    FIRST_FRESH_DEAL_SEED, FIRST_FRESH_DEAL_SEED + FRESH_DEALS
                )
            ]
        )
    )


class _NestedOutcome:
    """What a second layer of folds round a whole search reported."""

    __slots__ = ("honest_scores", "winners")

    def __init__(self, winners: list[int], honest_scores: list[float]) -> None:
        self.winners = winners
        self.honest_scores = honest_scores

    @property
    def mean_honest_score(self) -> float:
        return float(np.mean(self.honest_scores))

    @property
    def honest_spread(self) -> float:
        return float(max(self.honest_scores) - min(self.honest_scores))


def _nested(
    dataset: Dataset, outer: KFold, inner: KFold, neighbour_counts: list[int]
) -> _NestedOutcome:
    """Search inside each outer training fold, score on the outer held-out one.

    The outer loop the library does not yet have as an object, written over
    its ``KFold`` and ``GridSearch``. Every inner search is the ordinary one,
    and every honest score is ``honest_score_on`` with the outer held-out
    fold as the rows the search never saw.
    """
    winners: list[int] = []
    honest_scores: list[float] = []
    for split in outer.split(dataset):
        result = _search_neighbours(split.training, inner, neighbour_counts)
        winners.append(_neighbour_count(result.best.candidate))
        honest_scores.append(
            float(result.honest_score_on(MODEL_TYPE(), split.training, split.testing))
        )
    return _NestedOutcome(winners, honest_scores)


@cache
def _draw_optimism(seed: int, n_candidates: int, target: Target) -> float:
    """One draw's gap between the winner's score and its fresh-deal re-score.

    Cached because the averaged figure asks for a run of consecutive seeds,
    and the next draw the widget asks for shares all but one of them.
    """
    dataset = _drawn_target(target, seed)
    result = _search_neighbours(
        dataset, _noise_folds(), list(range(1, n_candidates + 1))
    )
    winner = _neighbour_count(result.best.candidate)
    return float(result.best_score) - _refolded_score(dataset, winner)


@cache
def _optimism_report(seed: int, n_candidates: int, target: Target) -> OptimismResponse:
    """The whole measurement on one draw, cached because the page re-reads it."""
    dataset = _drawn_target(target, seed)
    folds = _noise_folds()
    neighbour_counts = list(range(1, n_candidates + 1))
    result = _search_neighbours(dataset, folds, neighbour_counts)
    winner = _neighbour_count(result.best.candidate)
    refolded = _refolded_score(dataset, winner)
    nested = _nested(
        dataset,
        KFold(n_folds=NOISE_FOLDS, random_seed=OUTER_SEED),
        folds,
        neighbour_counts,
    )

    return OptimismResponse(
        seed=seed,
        n_candidates=n_candidates,
        target=target,
        candidates=_candidate_scores(result, folds, dataset),
        best_n_neighbours=winner,
        best_score=float(result.best_score),
        mean_candidate_score=_mean_candidate_score(result),
        score_spread=float(result.score_spread),
        refolded_score=refolded,
        optimism=float(result.best_score) - refolded,
        n_draws=DRAWS,
        mean_optimism_over_draws=float(
            np.mean(
                [
                    _draw_optimism(draw_seed, n_candidates, target)
                    for draw_seed in range(seed, seed + DRAWS)
                ]
            )
        ),
        nested_score=nested.mean_honest_score,
        nested_winners=nested.winners,
    )


@router.post("/concepts/grid-search/optimism", response_model=OptimismResponse)
def optimism(request: OptimismRequest) -> OptimismResponse:
    """Search a drawn target, then re-score the winner where it was never chosen."""
    return _optimism_report(request.seed, request.n_candidates, request.target)


# --- A second layer of folds round the whole search, on the reader's people --


class NestedRequest(BaseModel):
    """The points, the two fold counts, and the largest neighbour count."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_outer_folds: int = Field(default=4, ge=2, le=MAX_OUTER_FOLDS)
    n_inner_folds: int = Field(default=3, ge=2, le=MAX_FOLDS)
    n_neighbours_to: int = Field(default=5, ge=1, le=MAX_NEIGHBOURS)

    model_config = {"extra": "forbid"}


class OuterFoldReading(BaseModel):
    """One outer fold, the setting the inner search chose, and what it earned.

    ``best_inner_score`` is the inner search's own selection score, which
    carries the optimism; ``honest_score`` is the same winner refitted on the
    outer training rows and scored on the outer held-out rows.
    """

    held_out_indices: list[int]
    best_n_neighbours: int
    best_inner_score: float
    honest_score: float


class NestedResponse(BaseModel):
    """Every outer fold, the honest summary, and the flat search beside it.

    ``flat_best_score`` is what a single search over every row reports for
    its winner, which is the number the nested figure corrects.
    ``n_fits`` counts every model fitted, the inner searches and the outer
    refits together.
    """

    outer_folds: list[OuterFoldReading]
    mean_honest_score: float
    honest_spread: float
    flat_best_n_neighbours: int
    flat_best_score: float
    n_candidates: int
    n_fits: int


@router.post("/concepts/grid-search/nested", response_model=NestedResponse)
def nested(request: NestedRequest) -> NestedResponse:
    """Deal outer folds round every person and run the search inside each."""
    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    weights = np.asarray([point.y for point in request.points], dtype=np.float64)
    dataset = Dataset([Feature("height", heights)], Feature("weight", weights))
    neighbour_counts = list(range(1, request.n_neighbours_to + 1))
    inner = KFold(n_folds=request.n_inner_folds, random_seed=FOLD_SEED)
    outer = KFold(n_folds=request.n_outer_folds, random_seed=OUTER_SEED)

    readings: list[OuterFoldReading] = []
    for held in _fold_positions(heights.size, outer):
        training = np.setdiff1d(np.arange(heights.size), held)
        training_rows = Dataset(
            [Feature("height", heights[training])], Feature("weight", weights[training])
        )
        held_rows = Dataset(
            [Feature("height", heights[held])], Feature("weight", weights[held])
        )
        result = _search_neighbours(training_rows, inner, neighbour_counts)
        readings.append(
            OuterFoldReading(
                held_out_indices=sorted(held),
                best_n_neighbours=_neighbour_count(result.best.candidate),
                best_inner_score=float(result.best_score),
                honest_score=float(
                    result.honest_score_on(MODEL_TYPE(), training_rows, held_rows)
                ),
            )
        )

    flat = _search_neighbours(dataset, inner, neighbour_counts)
    honest_scores = [reading.honest_score for reading in readings]
    return NestedResponse(
        outer_folds=readings,
        mean_honest_score=float(np.mean(honest_scores)),
        honest_spread=float(max(honest_scores) - min(honest_scores)),
        flat_best_n_neighbours=_neighbour_count(flat.best.candidate),
        flat_best_score=float(flat.best_score),
        n_candidates=len(neighbour_counts),
        n_fits=request.n_outer_folds
        * (len(neighbour_counts) * request.n_inner_folds + 1),
    )
