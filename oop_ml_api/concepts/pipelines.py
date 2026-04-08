"""Pipelines, for the page where preprocessing and a model become one object.

Two endpoints, one for each reason the library has a pipeline at all.

The leak endpoint measures what fitting a transformer outside the fold costs,
instead of repeating a scare. It draws three columns on three different scales
with a price that is a plain line through them plus noise, then cross-validates
k-nearest neighbours twice on the same seeded folds. Once the ``Standardizer``
is fitted on every row before the folds are dealt, and once it sits inside a
``RegressionPipeline`` so each fold refits it on that fold's training rows. The
difference between the two mean scores is the leak. For a transformer that
never reads the target it comes out within noise of zero and changes sign from
one seed to the next, which is why a family of thirty seeds is run per request
and the count of seeds that flattered is reported beside the mean. The same
request then runs the comparison the library deliberately cannot express, a
column chosen by its correlation with the target, over columns that are pure
noise. There the gap is systematic. The library has no transformer that
consults the target, so that fold loop is written here over the library's
public ``KFold`` and ``KNearestNeighboursRegressor``, one run, with every fold
scored through the library's own evaluation.

The degree-search endpoint is the capability. ``degree`` is a field of
``PolynomialFeatures`` and not of ``RidgeRegression``, so a grid over the model
alone could never ask which degree to use. Wrapping both in one pipeline makes
the whole configuration one model, and the search varies whole ``PipelineSteps``
objects, each validated by its own constructor when it was written, never a step
name and a field name joined by a string and parsed at runtime.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Candidate,
    CrossValidation,
    Dataset,
    DataSplit,
    Feature,
    GridSearch,
    KFold,
    KNearestNeighboursRegressor,
    PipelineSteps,
    PolynomialFeatures,
    RegressionCrossValidationResult,
    RegressionEvaluation,
    RegressionPipeline,
    RidgeRegression,
    SearchSpace,
    Standardizer,
)
from pydantic import BaseModel, Field

from oop_ml_api.concepts.polynomial_regression import Point
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["pipelines"])

MIN_LEAK_ROWS = 30
"""The fewest rows the leak is measured on, six per held-out fold."""

MAX_LEAK_ROWS = 200
"""The most rows the leak is measured on, where the gap has all but vanished."""

MAX_FIRST_SEED = 10_000
"""The largest first seed a request may name."""

SEED_FAMILY = 30
"""How many seeds each leak request averages over.

One seed's gap at thirty rows is a coin toss either way, so the page needs the
mean across a family of seeds, and the count of seeds that flattered, to say so
honestly. The request names the first seed and the family runs upward from it.
"""

FOLD_COUNT = 5
"""How many folds every cross-validation here deals."""

NEIGHBOUR_COUNT = 5
"""How many neighbours the k-nearest regressor averages, in both comparisons."""

NOISE_COLUMNS = 20
"""How many pure-noise columns the target-consulting comparison chooses among."""

HIGHEST_DEGREE = 5
"""The degree search tries every degree from one up to this."""

SEARCH_PENALTY = 0.1
"""The ridge penalty behind every candidate of the degree search."""

SEARCH_FOLD_SEED = 4
"""The degree search's folds are seeded so the page can quote its scores."""


class LeakRequest(BaseModel):
    """How many rows to draw, and the first of the thirty seeds to draw them by."""

    n_rows: int = Field(ge=MIN_LEAK_ROWS, le=MAX_LEAK_ROWS)
    seed: int = Field(ge=0, le=MAX_FIRST_SEED)

    model_config = {"extra": "forbid"}


class ArrangementGap(BaseModel):
    """One transformer's two arrangements, seed by seed and then summarised.

    ``outside_scores`` are the mean held-out R squared when the transformer was
    fitted on every row before folding, ``inside_scores`` when it was refitted
    inside each fold, one entry per seed in the order the seeds ran.
    ``flattered_by`` is the mean of the first less the mean of the second, and
    ``seeds_flattered`` counts the seeds on which the outside arrangement
    scored higher, which is the number that says whether the gap is a bias or
    a coin toss.
    """

    outside_scores: list[float]
    inside_scores: list[float]
    mean_outside: float
    mean_inside: float
    flattered_by: float
    seeds_flattered: int


class LeakResponse(BaseModel):
    """Both comparisons, and what the pipeline object looked like afterwards.

    ``scaling`` is the standardizer, which never reads the target, and
    ``selection`` is a column picked by its correlation with the target. The
    two booleans are read off the one pipeline object that was handed to every
    fold of every seed. It comes back fitted to the last fold it saw, while the
    steps and model it was configured with come back untouched.
    """

    n_rows: int
    first_seed: int
    n_seeds: int
    n_folds: int
    n_neighbours: int
    n_noise_columns: int
    scaling: ArrangementGap
    selection: ArrangementGap
    pipeline_fitted_after_folds: bool
    configuration_fitted_after_folds: bool


def _three_scales(n_rows: int, seed: int) -> Dataset:
    """Three columns on three scales, and a price that is a line through them.

    Rooms sit around five, age runs to sixty years and distance to twenty-five
    kilometres, so an unscaled k-nearest neighbours would let the two wide
    columns decide who is near and the standardizer genuinely has work to do.
    """
    generator = np.random.default_rng(seed)
    rooms = generator.normal(loc=5.0, scale=1.5, size=n_rows)
    age_years = generator.uniform(0.0, 60.0, size=n_rows)
    distance_km = generator.uniform(0.5, 25.0, size=n_rows)
    price = (
        120.0
        + 30.0 * rooms
        - 1.2 * age_years
        - 4.0 * distance_km
        + generator.normal(scale=1.0, size=n_rows)
    )

    return Dataset(
        [
            Feature("rooms", rooms),
            Feature("age_years", age_years),
            Feature("distance_km", distance_km),
        ],
        Feature("price", price),
    )


def _pure_noise(n_rows: int, seed: int) -> Dataset:
    """Twenty columns of noise beside a target of noise, with nothing to find."""
    generator = np.random.default_rng(seed)
    columns = generator.normal(size=(n_rows, NOISE_COLUMNS))
    outcome = generator.normal(size=n_rows)

    return Dataset(
        [
            Feature(f"noise_{position + 1}", columns[:, position])
            for position in range(NOISE_COLUMNS)
        ],
        Feature("outcome", outcome),
    )


def _scaling_pipeline() -> RegressionPipeline:
    """Standardize, then average the nearest neighbours, as one regressor."""
    return RegressionPipeline(
        steps=PipelineSteps.of(scaler=Standardizer()),
        model=KNearestNeighboursRegressor(n_neighbours=NEIGHBOUR_COUNT),
    )


def _scaled_before_folding(dataset: Dataset, folds: KFold) -> float:
    """The scaler sees every row, then the folds are dealt over its output."""
    leaked = Standardizer().fit_transform(dataset.input_features)
    result = CrossValidation(folds=folds).evaluate(
        KNearestNeighboursRegressor(n_neighbours=NEIGHBOUR_COUNT),
        Dataset(leaked, dataset.target_feature),
    )

    return float(result.mean_r2_score)


def _scaled_inside_each_fold(
    pipeline: RegressionPipeline, dataset: Dataset, folds: KFold
) -> float:
    """The pipeline refits its scaler on each fold's training rows alone."""
    result = CrossValidation(folds=folds).evaluate(pipeline, dataset)

    return float(result.mean_r2_score)


def _most_correlated(dataset: Dataset) -> str:
    """The column whose correlation with the target is largest in size."""
    target = np.asarray(dataset.target_feature.values, dtype=np.float64)
    strengths = {
        feature.name: abs(
            float(
                np.corrcoef(np.asarray(feature.values, dtype=np.float64), target)[0, 1]
            )
        )
        for feature in dataset.input_features
    }

    return max(strengths, key=strengths.__getitem__)


def _only(dataset: Dataset, name: str) -> Dataset:
    """The dataset reduced to one named column beside its target."""
    return Dataset(
        [feature for feature in dataset.input_features if feature.name == name],
        dataset.target_feature,
    )


def _scored_on_column(split: DataSplit, name: str) -> RegressionEvaluation:
    """Fit the neighbours on one column of the training half, judge on the other."""
    training = _only(split.training, name)
    testing = _only(split.testing, name)
    model = KNearestNeighboursRegressor(n_neighbours=NEIGHBOUR_COUNT)
    model.fit(training.input_features, training.target_feature)

    return model.evaluate(testing.input_features, testing.target_feature)


def _selected_before_folding(dataset: Dataset, folds: KFold) -> float:
    """The column is chosen on every row, then the folds are dealt."""
    chosen = _most_correlated(dataset)
    result = CrossValidation(folds=folds).evaluate(
        KNearestNeighboursRegressor(n_neighbours=NEIGHBOUR_COUNT),
        _only(dataset, chosen),
    )

    return float(result.mean_r2_score)


def _selected_inside_each_fold(dataset: Dataset, folds: KFold) -> float:
    """Each fold chooses its column on its own training rows alone.

    The loop ``CrossValidation`` runs, written out, because the choosing step
    has no library transformer to sit inside a pipeline. The folds are the same
    seeded folds the other arrangement used, and every fold's score is the
    library's own evaluation.
    """
    evaluations = [
        _scored_on_column(split, _most_correlated(split.training))
        for split in folds.split(dataset)
    ]

    return float(RegressionCrossValidationResult(evaluations).mean_r2_score)


def _arrangement_gap(outside: list[float], inside: list[float]) -> ArrangementGap:
    """Summarise the two arrangements across the seeds they were run on."""
    outside_array = np.asarray(outside, dtype=np.float64)
    inside_array = np.asarray(inside, dtype=np.float64)

    return ArrangementGap(
        outside_scores=outside,
        inside_scores=inside,
        mean_outside=float(outside_array.mean()),
        mean_inside=float(inside_array.mean()),
        flattered_by=float(outside_array.mean() - inside_array.mean()),
        seeds_flattered=int(np.sum(outside_array > inside_array)),
    )


@router.post("/concepts/pipelines/leak", response_model=LeakResponse)
def leak(request: LeakRequest) -> LeakResponse:
    """Measure both leaks over thirty seeds, on the same folds each time."""
    pipeline = _scaling_pipeline()
    scaling_outside: list[float] = []
    scaling_inside: list[float] = []
    selection_outside: list[float] = []
    selection_inside: list[float] = []

    for seed in range(request.seed, request.seed + SEED_FAMILY):
        folds = KFold(n_folds=FOLD_COUNT, random_seed=seed)

        scaled = _three_scales(request.n_rows, seed)
        scaling_outside.append(_scaled_before_folding(scaled, folds))
        scaling_inside.append(_scaled_inside_each_fold(pipeline, scaled, folds))

        noise = _pure_noise(request.n_rows, seed)
        selection_outside.append(_selected_before_folding(noise, folds))
        selection_inside.append(_selected_inside_each_fold(noise, folds))

    configuration_fitted = pipeline.model.is_fitted or any(
        step.transformer.is_fitted for step in pipeline.steps
    )

    return LeakResponse(
        n_rows=request.n_rows,
        first_seed=request.seed,
        n_seeds=SEED_FAMILY,
        n_folds=FOLD_COUNT,
        n_neighbours=NEIGHBOUR_COUNT,
        n_noise_columns=NOISE_COLUMNS,
        scaling=_arrangement_gap(scaling_outside, scaling_inside),
        selection=_arrangement_gap(selection_outside, selection_inside),
        pipeline_fitted_after_folds=pipeline.is_fitted,
        configuration_fitted_after_folds=configuration_fitted,
    )


class DegreeSearchRequest(BaseModel):
    """The points whose best polynomial degree the search is to choose."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class DegreeSearchResponse(BaseModel):
    """Every degree the search tried, what it scored, and the winner.

    ``scores`` are mean held-out R squared across the seeded folds, in the
    order of ``degrees``. ``score_spread`` is the best less the worst, which
    is what says whether the winner meant anything.
    """

    degrees: list[int]
    scores: list[float]
    best_degree: int
    best_score: float
    score_spread: float
    n_candidates: int
    n_folds: int
    penalty: float


def _steps_at(degree: int) -> PipelineSteps:
    """Expand to this degree, then standardize the expanded columns."""
    return PipelineSteps.of(
        terms=PolynomialFeatures(degree=degree), scaler=Standardizer()
    )


def _degree_of(candidate: Candidate) -> int:
    """Read the degree back off the whole object the candidate carried."""
    transformer = candidate.value_for("steps")["terms"].transformer
    assert isinstance(transformer, PolynomialFeatures)

    return transformer.degree


@router.post("/concepts/pipelines/degree-search", response_model=DegreeSearchResponse)
def degree_search(request: DegreeSearchRequest) -> DegreeSearchResponse:
    """Cross-validate a pipeline at every degree and let the search pick one."""
    dataset = Dataset(
        [Feature("t", [point.x for point in request.points])],
        Feature("h", [point.y for point in request.points]),
    )
    prototype = RegressionPipeline(
        steps=_steps_at(1), model=RidgeRegression(penalty=SEARCH_PENALTY)
    )
    space = SearchSpace.over(
        RegressionPipeline,
        steps=[_steps_at(degree) for degree in range(1, HIGHEST_DEGREE + 1)],
    )
    result = GridSearch(
        folds=KFold(n_folds=FOLD_COUNT, random_seed=SEARCH_FOLD_SEED)
    ).search(prototype, space, dataset)

    return DegreeSearchResponse(
        degrees=[_degree_of(scored.candidate) for scored in result],
        scores=[float(scored.score) for scored in result],
        best_degree=_degree_of(result.best.candidate),
        best_score=float(result.best_score),
        score_spread=float(result.score_spread),
        n_candidates=result.n_candidates,
        n_folds=FOLD_COUNT,
        penalty=SEARCH_PENALTY,
    )
