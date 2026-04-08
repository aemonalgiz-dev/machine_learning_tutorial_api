"""Searching for a setting, for the grid-search page.

A model has dials no fit can turn, and the obvious move is to try every
setting and keep the best. Both endpoints here hand that move to the
library's ``GridSearch`` over the ``n_neighbours`` dial of
``KNearestNeighboursRegressor``, and both hand back every candidate's score
rather than only the winner's, because the losing scores are what tell a
reader whether the winner meant anything.

The search endpoint takes the reader's points. A quarter of them is held out
by a seeded shuffle before the search begins, the rest are dealt into seeded
folds, every candidate is cross-validated on that one deal, and the winner is
refitted on the searched rows and scored once on the held-out ones, which is
the library's ``honest_score_on``. The fold-by-fold scores come from a second
pass of ``CrossValidation`` over the identical folds, so the page can walk one
fold by hand and a test can check that the two routes agree.

The optimism endpoint runs the library's own measurement of selection bias.
On a target that is pure noise, where no setting can genuinely help, the
winner of twenty-five candidates is compared with the same setting scored
again on fresh fold deals it was never chosen on. The re-score uses the same
ruler as the selection, cross-validation over the same rows, rather than a
fresh holdout, because the two rulers do not agree with each other. Five
folds of sixteen rows read every setting a little low, and measured over
twenty draws the winner sat 0.05 above a fresh-fold re-score and 0.02 below
a four-hundred-row holdout, so a holdout would hide the flattering inside the
difference between rulers. The re-score is averaged over several deals
because one deal carries as much noise as the number it is checking, and the
whole experiment is repeated over a run of seeds because a single draw's
optimism can land on either side of zero.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    CrossValidation,
    Dataset,
    Feature,
    GridSearch,
    KFold,
    KNearestNeighboursRegressor,
    SearchResult,
    SearchSpace,
)
from pydantic import BaseModel, Field

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

NOISE_ROWS = 80
"""How many rows the pure-noise target has, matching the library's measurement."""

NOISE_FEATURES = 4
"""How many noise columns the pure-noise target is searched over."""

NOISE_SEED = 12
"""The draw the page quotes."""

NOISE_CANDIDATES = 25
"""The winner is the best of this many neighbour counts, one to twenty-five."""

NOISE_FOLDS = 5
"""The library's measurement cross-validated on five folds."""

FIRST_FRESH_DEAL_SEED = 101
"""The first of the fresh fold deals the winner is re-scored on."""

FRESH_DEALS = 12
"""How many fresh deals the re-score is averaged over."""

DRAWS = 12
"""How many consecutive seeds the optimism is averaged over."""

MAX_SEED = 10_000
"""The largest seed the noise endpoint accepts."""


class CandidateScore(BaseModel):
    """One setting, its cross-validated score, and the fold scores behind it."""

    n_neighbours: int
    score: float
    fold_scores: list[float]


class SearchRequest(BaseModel):
    """The points, the neighbour counts to try, and how many folds to deal."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    n_neighbours_from: int = Field(ge=1, le=MAX_NEIGHBOURS)
    n_neighbours_to: int = Field(ge=1, le=MAX_NEIGHBOURS)
    n_folds: int = Field(ge=2, le=MAX_FOLDS)

    model_config = {"extra": "forbid"}


class SearchResponse(BaseModel):
    """Every candidate's score, the winner, the spread, and the honest score.

    ``held_out_indices`` are positions into the request's points, and
    ``fold_of`` names the fold each point was dealt into, or ``None`` for a
    held-out point, so the widget can colour the deal the page describes.
    """

    held_out_indices: list[int]
    fold_of: list[int | None]
    candidates: list[CandidateScore]
    best_n_neighbours: int
    best_score: float
    mean_candidate_score: float
    score_spread: float
    honest_score: float


def _held_out_indices(n_points: int) -> np.ndarray:
    """The seeded shuffle-and-cut, by position, smallest indices first."""
    generator = np.random.default_rng(SPLIT_SEED)
    shuffled = generator.permutation(n_points)
    n_held = max(1, int(n_points * HELD_OUT_FRACTION))
    return np.sort(shuffled[:n_held])


def _fold_of_searched(n_searched: int, folds: KFold) -> list[int]:
    """Which fold each searched row was dealt into, read back from the library.

    The rows' positions are folded as if they were data, so the deal reported
    is the deal the search used rather than a re-derivation of it.
    """
    positions = np.arange(n_searched, dtype=np.float64)
    fold_of = [0] * n_searched
    for fold_index, split in enumerate(
        folds.split(
            Dataset([Feature("position", positions)], Feature("target", positions))
        )
    ):
        for position in split.testing.input_features[0].values:
            fold_of[int(position)] = fold_index
    return fold_of


def _search_neighbours(
    dataset: Dataset, folds: KFold, neighbour_counts: list[int]
) -> SearchResult:
    """Cross-validate every neighbour count on one deal and keep the best."""
    return GridSearch(folds=folds).search(
        KNearestNeighboursRegressor(),
        SearchSpace.over(KNearestNeighboursRegressor, n_neighbours=neighbour_counts),
        dataset,
    )


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
        n_neighbours = int(scored.candidate.value_for("n_neighbours"))
        evaluated = CrossValidation(folds=folds).evaluate(
            KNearestNeighboursRegressor(n_neighbours=n_neighbours), dataset
        )
        scores.append(
            CandidateScore(
                n_neighbours=n_neighbours,
                score=float(scored.score),
                fold_scores=[float(evaluation.r2_score) for evaluation in evaluated],
            )
        )
    return scores


def _mean_candidate_score(result: SearchResult) -> float:
    """The average over every candidate, the yardstick the winner beats."""
    return float(np.mean([scored.score for scored in result]))


@router.post("/concepts/grid-search/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Hold a quarter out, search the rest, and score the winner honestly."""
    x_values = np.asarray([point.x for point in request.points], dtype=np.float64)
    y_values = np.asarray([point.y for point in request.points], dtype=np.float64)

    held = _held_out_indices(x_values.size)
    searched = np.setdiff1d(np.arange(x_values.size), held)

    dataset = Dataset(
        [Feature("x", x_values[searched])], Feature("y", y_values[searched])
    )
    holdout = Dataset([Feature("x", x_values[held])], Feature("y", y_values[held]))
    folds = KFold(n_folds=request.n_folds, random_seed=FOLD_SEED)

    result = _search_neighbours(
        dataset,
        folds,
        list(range(request.n_neighbours_from, request.n_neighbours_to + 1)),
    )

    fold_of: list[int | None] = [None] * x_values.size
    for position, fold_index in zip(
        searched, _fold_of_searched(searched.size, folds), strict=True
    ):
        fold_of[int(position)] = fold_index

    return SearchResponse(
        held_out_indices=[int(index) for index in held],
        fold_of=fold_of,
        candidates=_candidate_scores(result, folds, dataset),
        best_n_neighbours=int(result.best.candidate.value_for("n_neighbours")),
        best_score=float(result.best_score),
        mean_candidate_score=_mean_candidate_score(result),
        score_spread=float(result.score_spread),
        honest_score=float(
            result.honest_score_on(KNearestNeighboursRegressor(), dataset, holdout)
        ),
    )


class OptimismRequest(BaseModel):
    """Which draw of pure noise to search."""

    seed: int = Field(ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class OptimismResponse(BaseModel):
    """What the winner reported, what it was worth, and the gap between them.

    ``refolded_score`` is the winner's neighbour count cross-validated again
    on fresh fold deals it was never chosen on, averaged over
    ``FRESH_DEALS`` of them. ``optimism`` is the winner's score minus that.
    ``mean_optimism_over_draws`` repeats the whole experiment on ``n_draws``
    consecutive seeds starting at the requested one and averages the gap,
    which is the number that survives any single draw landing the wrong side
    of zero.
    """

    seed: int
    candidates: list[CandidateScore]
    best_n_neighbours: int
    best_score: float
    mean_candidate_score: float
    score_spread: float
    refolded_score: float
    optimism: float
    n_draws: int
    mean_optimism_over_draws: float


def _pure_noise(n_rows: int, seed: int) -> Dataset:
    """A target with no relationship to its features, the library's fixture."""
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_rows, NOISE_FEATURES))
    return Dataset(
        [
            Feature(f"feature_{position}", matrix[:, position])
            for position in range(NOISE_FEATURES)
        ],
        Feature("outcome", generator.normal(size=n_rows)),
    )


def _noise_folds() -> KFold:
    """The one deal every noise candidate is scored on."""
    return KFold(n_folds=NOISE_FOLDS, random_seed=FOLD_SEED)


def _refolded_score(dataset: Dataset, n_neighbours: int) -> float:
    """The setting scored again on fresh deals, none of which chose it."""
    return float(
        np.mean(
            [
                CrossValidation(folds=KFold(n_folds=NOISE_FOLDS, random_seed=deal_seed))
                .evaluate(
                    KNearestNeighboursRegressor(n_neighbours=n_neighbours), dataset
                )
                .mean_r2_score
                for deal_seed in range(
                    FIRST_FRESH_DEAL_SEED, FIRST_FRESH_DEAL_SEED + FRESH_DEALS
                )
            ]
        )
    )


@cache
def _draw_optimism(seed: int) -> float:
    """One draw's gap between the winner's score and its fresh-deal re-score.

    Cached because the averaged figure asks for a run of consecutive seeds,
    and the next draw the widget asks for shares all but one of them.
    """
    dataset = _pure_noise(NOISE_ROWS, seed)
    result = _search_neighbours(
        dataset, _noise_folds(), list(range(1, NOISE_CANDIDATES + 1))
    )
    winner = int(result.best.candidate.value_for("n_neighbours"))
    return float(result.best_score) - _refolded_score(dataset, winner)


@router.post("/concepts/grid-search/optimism", response_model=OptimismResponse)
def optimism(request: OptimismRequest) -> OptimismResponse:
    """Search pure noise, then re-score the winner where it was never chosen."""
    dataset = _pure_noise(NOISE_ROWS, request.seed)
    folds = _noise_folds()
    result = _search_neighbours(dataset, folds, list(range(1, NOISE_CANDIDATES + 1)))
    winner = int(result.best.candidate.value_for("n_neighbours"))
    refolded = _refolded_score(dataset, winner)

    return OptimismResponse(
        seed=request.seed,
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
                    _draw_optimism(seed)
                    for seed in range(request.seed, request.seed + DRAWS)
                ]
            )
        ),
    )
