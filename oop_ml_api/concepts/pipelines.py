"""Pipelines, for the page where preprocessing and a model become one object.

Five endpoints, each one measurement the page makes rather than a claim it
repeats.

The anatomy endpoint follows one chain, expand height to a degree, standardize
the expanded columns, then ridge, through one fit on the training people and
one predict on the held-out ones, and reports everything the fit learned on the
way: the columns each step produced, the centre and spread the standardizer
learned for each of them, what a held-out person's height became under those
numbers, and the refusal a single person meets when standardized alone. It
then reads the configuration back after the fit, refits on other rows, breaks
a refit on purpose, and hands the same object to every fold of a
cross-validation, because the deep copy inside ``fit`` is a promise and each of
those is a way of checking it.

The leak endpoint measures what fitting a transformer outside the fold costs,
instead of repeating a scare. It draws people measured three ways on three
scales, height in centimetres, age in years and how far they walk in a day in
kilometres, with a weight that is a plain line through them plus noise, then
cross-validates k-nearest neighbours twice on the same seeded folds. Once with
the ``Standardizer`` fitted on every row before the folds are dealt, and once
inside a ``RegressionPipeline`` so each fold refits it on that fold's training
rows. The difference between the two mean scores is the leak. For a
transformer that never reads the target it comes out within noise of zero and
changes sign from one seed to the next, which is why a family of thirty seeds
is run per request and the count of seeds that flattered is reported beside
the mean. The same request then runs the comparison the library deliberately
cannot express, a column chosen by its correlation with the target, over
columns that are pure noise. There the gap is systematic. The library has no
transformer that consults the target, so that fold loop is written here over
the library's public ``KFold`` and ``KNearestNeighboursRegressor``, with every
fold scored through the library's own evaluation.

The search endpoint is the capability. ``degree`` is a field of
``PolynomialFeatures`` and not of ``RidgeRegression``, so a grid over the model
alone could never ask which degree to use. Wrapping both in one pipeline makes
the whole configuration one model, and the search varies whole
``PipelineSteps`` and whole ``RidgeRegression`` objects, each validated by its
own constructor when it was written, never a step name and a field name joined
by a string and parsed at runtime. The response carries the two refusals that
make the point, a misspelt field name and the degree asked of the model alone.

The order endpoint runs the same chain both ways round, standardize then expand
against expand then standardize, at a range of penalties on the same folds. At
a penalty of zero the two orders span the same columns and agree to rounding;
at any positive penalty they differ, because the penalty meets the squared
column at a different spread in each.

The contracts endpoint asks every edge the page's last part tabulates, on each
load, so the table is probed rather than copied once.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Candidate,
    ClassificationPipeline,
    CrossValidation,
    Dataset,
    DataSplit,
    Feature,
    GridSearch,
    KFold,
    KNearestNeighboursClassifier,
    KNearestNeighboursRegressor,
    LogisticRegression,
    MultipleLinearRegression,
    PipelineStep,
    PipelineSteps,
    PolynomialFeatures,
    RegressionCrossValidationResult,
    RegressionEvaluation,
    RegressionPipeline,
    RidgeRegression,
    SearchSpace,
    Standardizer,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field, ValidationError

from oop_ml_api.concepts.polynomial_regression import Point
from oop_ml_api.limits import MAX_POINTS, MIN_POINTS

router = APIRouter(tags=["pipelines"])

HEIGHT = "height"
"""The one input column of the running example, in centimetres."""

WEIGHT = "weight"
"""Its target, in kilograms."""

MAX_DEGREE = 6
"""The highest degree the chain expands height to."""

MAX_PENALTY = 1000.0
"""The largest ridge penalty a request may name."""

MAX_FOLDS = 10
"""The most folds a request may deal."""

MAX_SEED = 10_000
"""The largest seed a request may name, for the folds or the leak family."""

MAX_GRID_SIDE = 8
"""The most values one dial of the search may take."""

MIN_LEAK_ROWS = 30
"""The fewest rows the leak is measured on, six per held-out fold."""

MAX_LEAK_ROWS = 200
"""The most rows the leak is measured on, where the gap has all but vanished."""

SEED_FAMILY = 30
"""How many seeds each leak request averages over.

One seed's gap at thirty rows is a coin toss either way, so the page needs the
mean across a family of seeds, and the count of seeds that flattered, to say so
honestly. The request names the first seed and the family runs upward from it.
"""

FOLD_COUNT = 5
"""How many folds the leak's cross-validations deal."""

NEIGHBOUR_COUNT = 5
"""How many neighbours the leak's k-nearest regressor averages."""

NOISE_COLUMNS = 20
"""How many pure-noise columns the target-consulting comparison chooses among."""


class Refusal(BaseModel):
    """A refusal by name and in the library's own words."""

    error: str
    detail: str


def _refusal_from(attempt: Callable[[], object]) -> Refusal:
    """Run something the library is expected to refuse and record how it did.

    Raises
    ------
    AssertionError
        If the library accepted, since the page quotes the refusal.
    """
    try:
        attempt()
    except MLLibError as refusal:
        return Refusal(error=type(refusal).__name__, detail=str(refusal))
    raise AssertionError("the library accepted what the page says it refuses")


def _height_feature(points: list[Point]) -> Feature:
    return Feature(HEIGHT, [point.x for point in points])


def _weight_feature(points: list[Point]) -> Feature:
    return Feature(WEIGHT, [point.y for point in points])


def _dataset(points: list[Point]) -> Dataset:
    return Dataset([_height_feature(points)], _weight_feature(points))


def _select(points: list[Point], indices: list[int]) -> list[Point]:
    return [points[index] for index in indices]


def _expand_then_scale(degree: int) -> PipelineSteps:
    """The chain's canonical order: every column the model sees is standardized."""
    return PipelineSteps.of(
        terms=PolynomialFeatures(degree=degree), scaler=Standardizer()
    )


def _scale_then_expand(degree: int) -> PipelineSteps:
    """The reverse order, where the squared column keeps its own spread."""
    return PipelineSteps.of(
        scaler=Standardizer(), terms=PolynomialFeatures(degree=degree)
    )


def _chain(degree: int, penalty: float) -> RegressionPipeline:
    return RegressionPipeline(
        steps=_expand_then_scale(degree), model=RidgeRegression(penalty=penalty)
    )


def _max_gap(first: object, second: object) -> float:
    """The largest absolute difference between two prediction arrays."""
    return float(
        np.max(
            np.abs(
                np.asarray(first, dtype=np.float64)
                - np.asarray(second, dtype=np.float64)
            )
        )
    )


# ---------------------------------------------------------------------------
# One chain, one fit, one predict, and the configuration read back
# ---------------------------------------------------------------------------


class AnatomyRequest(BaseModel):
    """The people, the chain's two settings, and which people to hold out.

    ``held_out`` indexes into ``points``; at least two must remain to train on,
    since a standardizer needs a spread, and the request is refused otherwise.
    """

    points: list[Point] = Field(min_length=4, max_length=MAX_POINTS)
    degree: int = Field(default=2, ge=1, le=MAX_DEGREE)
    penalty: float = Field(default=1.0, ge=0.0, le=MAX_PENALTY)
    held_out: list[int] = Field(min_length=1, max_length=MAX_POINTS)
    n_folds: int = Field(default=4, ge=2, le=MAX_FOLDS)
    seed: int = Field(default=4, ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class LearnedScaling(BaseModel):
    """One column's centre and spread, as the standardizer learned them."""

    name: str
    mean: float
    standard_deviation: float


class StepReport(BaseModel):
    """One step of the fitted chain: what it read, what it produced, what it learned.

    ``learned`` is empty for the expansion, which learns only which columns to
    make, and holds one scaling per column for the standardizer.
    """

    name: str
    reads: list[str]
    produces: list[str]
    learned: list[LearnedScaling]


class HeldOutPerson(BaseModel):
    """One held-out person on the way through the fitted chain."""

    index: int
    height: float
    weight: float
    standardized_height: float
    predicted_weight: float


class NamedValue(BaseModel):
    """A coefficient bound to the column it multiplies."""

    name: str
    value: float


class FoldReport(BaseModel):
    """One fold's held-out score from the shared object, and its gap to a fresh one."""

    fold: int
    n_training: int
    n_held_out: int
    r2_score: float
    gap_to_fresh_fit: float


class AnatomyResponse(BaseModel):
    """Everything the fit learned, and every check on the configuration.

    ``alone`` is the refusal one held-out person meets when standardized on
    their own. ``mean_before_predict`` and ``mean_after_predict`` are the
    fitted standardizer's centre for height read either side of a predict on
    the held-out people. The four booleans after them are read off the one
    object: the configured model and steps stay unfitted, the fitted copies
    are fitted, and the copies are different objects from the configuration.
    ``refit_gap`` is the largest difference between this object refitted on
    other people and a fresh chain fitted on the same people.
    ``failed_refit`` is the refusal a refit on a constant column meets, and
    ``predictions_after_failed_refit_gap`` how far the predictions moved,
    which is nothing. ``folds`` hands the same object to every fold.
    ``no_steps_gap`` is a chain with no steps against the bare model.
    """

    n_training: int
    n_held_out: int
    degree: int
    penalty: float
    steps: list[StepReport]
    held_out: list[HeldOutPerson]
    held_out_r2: float
    training_r2: float
    alone: Refusal
    coefficients: list[NamedValue]
    intercept: float
    mean_before_predict: float
    mean_after_predict: float
    configuration_model_fitted: bool
    configuration_steps_fitted: list[bool]
    fitted_model_fitted: bool
    fitted_parts_are_separate: bool
    refit_gap: float
    failed_refit: Refusal
    still_fitted_after_failed_refit: bool
    predictions_after_failed_refit_gap: float
    n_folds: int
    fold_seed: int
    folds: list[FoldReport]
    mean_fold_r2: float
    configuration_fitted_after_folds: bool
    pipeline_fitted_after_folds: bool
    no_steps_gap: float


def _step_reports(
    fitted: RegressionPipeline, training: list[Point]
) -> list[StepReport]:
    """Walk the fitted steps, recording what each read and produced."""
    carried: list[Feature] = [_height_feature(training)]
    reports: list[StepReport] = []

    for step in fitted.fitted_steps:
        produced = list(step.transformer.transform(carried))
        learned: list[LearnedScaling] = []
        if isinstance(step.transformer, Standardizer):
            learned = [
                LearnedScaling(
                    name=scaling.name,
                    mean=float(scaling.mean),
                    standard_deviation=float(scaling.standard_deviation),
                )
                for scaling in step.transformer.scalings
            ]
        reports.append(
            StepReport(
                name=step.name,
                reads=[feature.name for feature in carried],
                produces=[feature.name for feature in produced],
                learned=learned,
            )
        )
        carried = produced

    return reports


def _height_scaling(fitted: RegressionPipeline) -> LearnedScaling:
    """The centre and spread the fitted chain's standardizer learned for height."""
    scaler = fitted.fitted_steps["scaler"].transformer
    assert isinstance(scaler, Standardizer)
    scaling = scaler.scalings[HEIGHT]

    return LearnedScaling(
        name=HEIGHT,
        mean=float(scaling.mean),
        standard_deviation=float(scaling.standard_deviation),
    )


def _fold_reports(
    shared: RegressionPipeline,
    degree: int,
    penalty: float,
    dataset: Dataset,
    folds: KFold,
) -> list[FoldReport]:
    """Hand one object to every fold, and a fresh chain beside it each time."""
    reports: list[FoldReport] = []

    for position, split in enumerate(folds.split(dataset)):
        shared.fit(split.training.input_features, split.training.target_feature)
        fresh = _chain(degree, penalty).fit(
            split.training.input_features, split.training.target_feature
        )
        evaluation = shared.evaluate(
            split.testing.input_features, split.testing.target_feature
        )
        reports.append(
            FoldReport(
                fold=position + 1,
                n_training=len(split.training.target_feature.values),
                n_held_out=len(split.testing.target_feature.values),
                r2_score=float(evaluation.r2_score),
                gap_to_fresh_fit=_max_gap(
                    shared.predict(split.testing.input_features),
                    fresh.predict(split.testing.input_features),
                ),
            )
        )

    return reports


def _configuration_fitted(pipeline: RegressionPipeline) -> bool:
    return pipeline.model.is_fitted or any(
        step.transformer.is_fitted for step in pipeline.steps
    )


@router.post("/concepts/pipelines/anatomy", response_model=AnatomyResponse)
def anatomy(request: AnatomyRequest) -> AnatomyResponse:
    """Fit the chain once, predict once, and read the configuration back."""
    count = len(request.points)
    held = sorted(set(request.held_out))
    if any(index < 0 or index >= count for index in held):
        raise HTTPException(
            status_code=422,
            detail=f"held_out indexes {count} people, from 0 to {count - 1}",
        )
    training_indices = [index for index in range(count) if index not in set(held)]
    if len(training_indices) < 2:
        raise HTTPException(
            status_code=422, detail="at least two people must remain to train on"
        )

    training = _select(request.points, training_indices)
    held_out = _select(request.points, held)
    training_features = [_height_feature(training)]
    training_target = _weight_feature(training)
    held_features = [_height_feature(held_out)]
    held_target = _weight_feature(held_out)

    pipeline = _chain(request.degree, request.penalty)
    pipeline.fit(training_features, training_target)
    step_reports = _step_reports(pipeline, training)
    height_scaling = _height_scaling(pipeline)
    predictions = np.asarray(pipeline.predict(held_features), dtype=np.float64)
    mean_after = _height_scaling(pipeline).mean

    configuration_model_fitted = pipeline.model.is_fitted
    configuration_steps_fitted = [step.transformer.is_fitted for step in pipeline.steps]
    fitted_model_fitted = pipeline.fitted_model.is_fitted
    separate = pipeline.fitted_model is not pipeline.model and all(
        fitted.transformer is not configured.transformer
        for fitted, configured in zip(
            pipeline.fitted_steps, pipeline.steps, strict=True
        )
    )

    fitted_model = pipeline.fitted_model
    assert isinstance(fitted_model, RidgeRegression)
    coefficients = [
        NamedValue(name=coefficient.name, value=float(coefficient.value))
        for coefficient in fitted_model.coefficients
    ]
    intercept = float(fitted_model.intercept)

    alone = _refusal_from(
        lambda: Standardizer().fit([Feature(HEIGHT, [held_out[0].x])])
    )

    # Refit on the other arrangement, held-out people swapped for the first
    # few training ones, and compare with a fresh chain on the same people.
    swapped = held + training_indices[len(held) :]
    other = _select(request.points, sorted(swapped))
    pipeline.fit([_height_feature(other)], _weight_feature(other))
    fresh = _chain(request.degree, request.penalty).fit(
        [_height_feature(other)], _weight_feature(other)
    )
    everyone = [_height_feature(request.points)]
    refit_gap = _max_gap(pipeline.predict(everyone), fresh.predict(everyone))

    # Back to the training people, then break a refit on purpose.
    pipeline.fit(training_features, training_target)
    before_failure = pipeline.predict(held_features)
    failed_refit = _refusal_from(
        lambda: pipeline.fit(
            [Feature(HEIGHT, [training[0].x] * len(training))], training_target
        )
    )
    still_fitted = pipeline.is_fitted
    after_failure_gap = _max_gap(before_failure, pipeline.predict(held_features))

    dataset = _dataset(request.points)
    folds = KFold(n_folds=request.n_folds, random_seed=request.seed)
    shared = _chain(request.degree, request.penalty)
    fold_reports = _fold_reports(
        shared, request.degree, request.penalty, dataset, folds
    )

    no_steps_gap = _max_gap(
        RegressionPipeline(model=RidgeRegression(penalty=request.penalty))
        .fit(training_features, training_target)
        .predict(held_features),
        RidgeRegression(penalty=request.penalty)
        .fit(training_features, training_target)
        .predict(held_features),
    )

    return AnatomyResponse(
        n_training=len(training),
        n_held_out=len(held_out),
        degree=request.degree,
        penalty=request.penalty,
        steps=step_reports,
        held_out=[
            HeldOutPerson(
                index=index,
                height=person.x,
                weight=person.y,
                standardized_height=float(
                    (person.x - height_scaling.mean) / height_scaling.standard_deviation
                ),
                predicted_weight=float(predictions[position]),
            )
            for position, (index, person) in enumerate(zip(held, held_out, strict=True))
        ],
        held_out_r2=float(pipeline.score(held_features, held_target)),
        training_r2=float(pipeline.score(training_features, training_target)),
        alone=alone,
        coefficients=coefficients,
        intercept=intercept,
        mean_before_predict=height_scaling.mean,
        mean_after_predict=mean_after,
        configuration_model_fitted=configuration_model_fitted,
        configuration_steps_fitted=configuration_steps_fitted,
        fitted_model_fitted=fitted_model_fitted,
        fitted_parts_are_separate=separate,
        refit_gap=refit_gap,
        failed_refit=failed_refit,
        still_fitted_after_failed_refit=still_fitted,
        predictions_after_failed_refit_gap=after_failure_gap,
        n_folds=request.n_folds,
        fold_seed=request.seed,
        folds=fold_reports,
        mean_fold_r2=float(np.mean([report.r2_score for report in fold_reports])),
        configuration_fitted_after_folds=_configuration_fitted(shared),
        pipeline_fitted_after_folds=shared.is_fitted,
        no_steps_gap=no_steps_gap,
    )


# ---------------------------------------------------------------------------
# The leak, measured over a family of seeds
# ---------------------------------------------------------------------------


class LeakRequest(BaseModel):
    """How many people to draw, and the first of the thirty seeds to draw them by."""

    n_rows: int = Field(ge=MIN_LEAK_ROWS, le=MAX_LEAK_ROWS)
    seed: int = Field(ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class ArrangementGap(BaseModel):
    """One transformer's two arrangements, seed by seed and then summarised.

    ``outside_scores`` are the mean held-out R squared when the transformer was
    fitted on every row before folding, ``inside_scores`` when it was refitted
    inside each fold, one entry per seed in the order the seeds ran.
    ``flattered_by`` is the mean of the first less the mean of the second, and
    ``seeds_flattered`` counts the seeds on which the outside arrangement
    scored higher, which is the number that says whether the gap is a bias or
    a coin toss. ``largest_gap`` is the largest single-seed gap in size.
    """

    outside_scores: list[float]
    inside_scores: list[float]
    mean_outside: float
    mean_inside: float
    flattered_by: float
    seeds_flattered: int
    largest_gap: float


class LeakResponse(BaseModel):
    """Both comparisons, and what the pipeline object looked like afterwards.

    ``scaling`` is the standardizer, which never reads the target, and
    ``selection`` is a column picked by its correlation with the target. The
    two booleans are read off the one pipeline object that was handed to every
    fold of every seed. It comes back fitted to the last fold it saw, while the
    steps and model it was configured with come back untouched.
    ``training_share`` is the share of rows each training fold holds, which is
    the weight the fold's own centre carries in a centre computed over all.
    """

    n_rows: int
    first_seed: int
    n_seeds: int
    n_folds: int
    n_neighbours: int
    n_noise_columns: int
    training_share: float
    scaling: ArrangementGap
    selection: ArrangementGap
    pipeline_fitted_after_folds: bool
    configuration_fitted_after_folds: bool


def _people(n_rows: int, seed: int) -> Dataset:
    """People measured three ways on three scales, and a weight through them.

    Height in centimetres spreads about nine, age in years runs from eighteen
    to seventy, and the daily walk in kilometres runs to twelve, so an unscaled
    k-nearest neighbours would let age decide who is near and the standardizer
    genuinely has work to do.
    """
    generator = np.random.default_rng(seed)
    height = generator.normal(loc=168.0, scale=9.0, size=n_rows)
    age_years = generator.uniform(18.0, 70.0, size=n_rows)
    walk_km = generator.uniform(0.5, 12.0, size=n_rows)
    weight = (
        -45.0
        + 0.62 * height
        + 0.12 * age_years
        - 1.4 * walk_km
        + generator.normal(scale=2.0, size=n_rows)
    )

    return Dataset(
        [
            Feature(HEIGHT, height),
            Feature("age_years", age_years),
            Feature("walk_km", walk_km),
        ],
        Feature(WEIGHT, weight),
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
        largest_gap=float(np.max(np.abs(outside_array - inside_array))),
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

        people = _people(request.n_rows, seed)
        scaling_outside.append(_scaled_before_folding(people, folds))
        scaling_inside.append(_scaled_inside_each_fold(pipeline, people, folds))

        noise = _pure_noise(request.n_rows, seed)
        selection_outside.append(_selected_before_folding(noise, folds))
        selection_inside.append(_selected_inside_each_fold(noise, folds))

    return LeakResponse(
        n_rows=request.n_rows,
        first_seed=request.seed,
        n_seeds=SEED_FAMILY,
        n_folds=FOLD_COUNT,
        n_neighbours=NEIGHBOUR_COUNT,
        n_noise_columns=NOISE_COLUMNS,
        training_share=(FOLD_COUNT - 1) / FOLD_COUNT,
        scaling=_arrangement_gap(scaling_outside, scaling_inside),
        selection=_arrangement_gap(selection_outside, selection_inside),
        pipeline_fitted_after_folds=pipeline.is_fitted,
        configuration_fitted_after_folds=_configuration_fitted(pipeline),
    )


# ---------------------------------------------------------------------------
# A step's setting, searched by varying whole objects
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """The people and the two dials, a degree for the expansion and a penalty for the ridge."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degrees: list[int] = Field(min_length=1, max_length=MAX_GRID_SIDE)
    penalties: list[float] = Field(min_length=1, max_length=MAX_GRID_SIDE)
    n_folds: int = Field(default=4, ge=2, le=MAX_FOLDS)
    seed: int = Field(default=4, ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class SearchCell(BaseModel):
    """One candidate, a whole chain at one degree and one penalty, and its score."""

    degree: int
    penalty: float
    score: float


class SearchResponse(BaseModel):
    """Every candidate scored, the winner, and the two refusals that make the point.

    ``model_hyperparameters`` are the fields the ridge model alone exposes and
    ``chain_hyperparameters`` the fields the chain exposes, read from the
    library. ``degree_on_model`` is the refusal a search meets when it asks the
    model for a degree it does not have, and ``misspelt_field`` the refusal a
    misspelt chain field meets at construction. ``n_fits`` is candidates times
    folds.
    """

    cells: list[SearchCell]
    best_degree: int
    best_penalty: float
    best_score: float
    score_spread: float
    n_candidates: int
    n_folds: int
    n_fits: int
    model_hyperparameters: list[str]
    chain_hyperparameters: list[str]
    degree_on_model: Refusal
    misspelt_field: Refusal


def _degree_of(candidate: Candidate) -> int:
    """Read the degree back off the whole object the candidate carried."""
    transformer = candidate.value_for("steps")["terms"].transformer
    assert isinstance(transformer, PolynomialFeatures)

    return transformer.degree


def _penalty_of(candidate: Candidate) -> float:
    model = candidate.value_for("model")
    assert isinstance(model, RidgeRegression)

    return float(model.penalty)


@router.post("/concepts/pipelines/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Cross-validate the chain at every degree and penalty and let the search pick."""
    degrees = sorted(set(request.degrees))
    penalties = sorted(set(request.penalties))
    if any(degree < 1 or degree > MAX_DEGREE for degree in degrees):
        raise HTTPException(
            status_code=422, detail=f"a degree must lie between 1 and {MAX_DEGREE}"
        )
    if any(penalty < 0.0 or penalty > MAX_PENALTY for penalty in penalties):
        raise HTTPException(
            status_code=422, detail=f"a penalty must lie between 0 and {MAX_PENALTY}"
        )

    prototype = _chain(degrees[0], penalties[0])
    space = SearchSpace.over(
        RegressionPipeline,
        steps=[_expand_then_scale(degree) for degree in degrees],
        model=[RidgeRegression(penalty=penalty) for penalty in penalties],
    )
    result = GridSearch(
        folds=KFold(n_folds=request.n_folds, random_seed=request.seed)
    ).search(prototype, space, _dataset(request.points))

    return SearchResponse(
        cells=[
            SearchCell(
                degree=_degree_of(scored.candidate),
                penalty=_penalty_of(scored.candidate),
                score=float(scored.score),
            )
            for scored in result
        ],
        best_degree=_degree_of(result.best.candidate),
        best_penalty=_penalty_of(result.best.candidate),
        best_score=float(result.best_score),
        score_spread=float(result.score_spread),
        n_candidates=result.n_candidates,
        n_folds=request.n_folds,
        n_fits=result.n_candidates * request.n_folds,
        model_hyperparameters=sorted(RidgeRegression.model_fields),
        chain_hyperparameters=sorted(RegressionPipeline.model_fields),
        degree_on_model=_refusal_from(
            lambda: SearchSpace.over(RidgeRegression, degree=degrees)
        ),
        misspelt_field=_refusal_from(
            lambda: SearchSpace.over(
                RegressionPipeline, stepz=[_expand_then_scale(degrees[0])]
            )
        ),
    )


# ---------------------------------------------------------------------------
# The order of the steps
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    """The people, the degree, the penalties to sweep, and the people to hold out.

    The held-out people are where the two orders' predictions are compared at
    each penalty; the scores come from the seeded folds over everyone.
    """

    points: list[Point] = Field(min_length=4, max_length=MAX_POINTS)
    degree: int = Field(default=2, ge=1, le=MAX_DEGREE)
    penalties: list[float] = Field(min_length=1, max_length=MAX_GRID_SIDE)
    held_out: list[int] = Field(min_length=1, max_length=MAX_POINTS)
    n_folds: int = Field(default=4, ge=2, le=MAX_FOLDS)
    seed: int = Field(default=4, ge=0, le=MAX_SEED)

    model_config = {"extra": "forbid"}


class OrderAtPenalty(BaseModel):
    """Both orders at one penalty: their mean fold scores and their held-out gap."""

    penalty: float
    expand_then_scale_score: float
    scale_then_expand_score: float
    held_out_gap: float


class ColumnAsSeen(BaseModel):
    """One column as the model met it, its centre and spread over the training people.

    The spread is the population deviation, the one the standardizer divides
    by, so a standardized column reads exactly one.
    """

    name: str
    mean: float
    standard_deviation: float
    coefficient: float


class OrderResponse(BaseModel):
    """The sweep, and the columns each order handed the model at ``shown_penalty``.

    ``least_squares_gap`` is the two orders' held-out gap under a plain
    least-squares model, which has no penalty to feel the spread with.
    """

    degree: int
    sweep: list[OrderAtPenalty]
    shown_penalty: float
    expand_then_scale_columns: list[ColumnAsSeen]
    scale_then_expand_columns: list[ColumnAsSeen]
    least_squares_gap: float


def _columns_as_seen(
    steps: PipelineSteps, penalty: float, training: list[Point]
) -> list[ColumnAsSeen]:
    """Fit the chain, then read the columns its model was handed and its weights."""
    features = [_height_feature(training)]
    target = _weight_feature(training)
    fitted = RegressionPipeline(
        steps=steps, model=RidgeRegression(penalty=penalty)
    ).fit(features, target)
    carried: list[Feature] = list(features)
    for step in fitted.fitted_steps:
        carried = list(step.transformer.transform(carried))
    model = fitted.fitted_model
    assert isinstance(model, RidgeRegression)
    weights = {
        coefficient.name: float(coefficient.value) for coefficient in model.coefficients
    }

    return [
        ColumnAsSeen(
            name=column.name,
            mean=float(np.mean(np.asarray(column.values, dtype=np.float64))),
            standard_deviation=float(
                np.std(np.asarray(column.values, dtype=np.float64))
            ),
            coefficient=weights[column.name],
        )
        for column in carried
    ]


@router.post("/concepts/pipelines/order", response_model=OrderResponse)
def order(request: OrderRequest) -> OrderResponse:
    """Run the chain both ways round at every penalty on the same folds."""
    count = len(request.points)
    held = sorted(set(request.held_out))
    if any(index < 0 or index >= count for index in held):
        raise HTTPException(
            status_code=422,
            detail=f"held_out indexes {count} people, from 0 to {count - 1}",
        )
    training_indices = [index for index in range(count) if index not in set(held)]
    if len(training_indices) < 2:
        raise HTTPException(
            status_code=422, detail="at least two people must remain to train on"
        )
    penalties = sorted(set(request.penalties))
    if any(penalty < 0.0 or penalty > MAX_PENALTY for penalty in penalties):
        raise HTTPException(
            status_code=422, detail=f"a penalty must lie between 0 and {MAX_PENALTY}"
        )

    training = _select(request.points, training_indices)
    held_out = _select(request.points, held)
    training_features = [_height_feature(training)]
    training_target = _weight_feature(training)
    held_features = [_height_feature(held_out)]
    dataset = _dataset(request.points)
    folds = KFold(n_folds=request.n_folds, random_seed=request.seed)
    validation = CrossValidation(folds=folds)

    sweep: list[OrderAtPenalty] = []
    for penalty in penalties:
        expand_first = RegressionPipeline(
            steps=_expand_then_scale(request.degree),
            model=RidgeRegression(penalty=penalty),
        )
        scale_first = RegressionPipeline(
            steps=_scale_then_expand(request.degree),
            model=RidgeRegression(penalty=penalty),
        )
        sweep.append(
            OrderAtPenalty(
                penalty=penalty,
                expand_then_scale_score=float(
                    validation.evaluate(expand_first, dataset).mean_r2_score
                ),
                scale_then_expand_score=float(
                    validation.evaluate(scale_first, dataset).mean_r2_score
                ),
                held_out_gap=_max_gap(
                    expand_first.fit(training_features, training_target).predict(
                        held_features
                    ),
                    scale_first.fit(training_features, training_target).predict(
                        held_features
                    ),
                ),
            )
        )

    shown_penalty = (
        penalties[-1] if len(penalties) == 1 else penalties[len(penalties) // 2]
    )
    least_squares_gap = _max_gap(
        RegressionPipeline(
            steps=_expand_then_scale(request.degree), model=MultipleLinearRegression()
        )
        .fit(training_features, training_target)
        .predict(held_features),
        RegressionPipeline(
            steps=_scale_then_expand(request.degree), model=MultipleLinearRegression()
        )
        .fit(training_features, training_target)
        .predict(held_features),
    )

    return OrderResponse(
        degree=request.degree,
        sweep=sweep,
        shown_penalty=shown_penalty,
        expand_then_scale_columns=_columns_as_seen(
            _expand_then_scale(request.degree), shown_penalty, training
        ),
        scale_then_expand_columns=_columns_as_seen(
            _scale_then_expand(request.degree), shown_penalty, training
        ),
        least_squares_gap=least_squares_gap,
    )


# ---------------------------------------------------------------------------
# The contracts, probed on every load
# ---------------------------------------------------------------------------

Outcome = Literal["accepted", "refused", "escaped"]


class ContractProbe(BaseModel):
    """One edge put to the chain, and what came back.

    ``accepted`` carries the library's answer in ``result``; ``refused`` carries
    a typed refusal by name in ``error`` and in its own words in ``detail``,
    either from the library's hierarchy or from the constructor's validation;
    ``escaped`` is a failure that is neither, a bare exception the page reports
    as documented rather than defended.
    """

    edge: str
    outcome: Outcome
    result: str | None
    error: str | None
    detail: str | None


class ContractsResponse(BaseModel):
    """Every edge the page tabulates, probed."""

    probes: list[ContractProbe]


TWELVE_HEIGHTS = [
    150.0,
    153.0,
    157.0,
    159.0,
    162.0,
    165.0,
    168.0,
    171.0,
    174.0,
    177.0,
    180.0,
    183.0,
]
"""The twelve people the page's chain is fitted on, by height in centimetres."""

TWELVE_WEIGHTS = [
    50.0,
    52.0,
    55.0,
    57.0,
    61.0,
    63.0,
    66.0,
    71.0,
    72.0,
    76.0,
    78.0,
    82.0,
]
"""Their weights in kilograms."""

TWELVE_ADULT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
"""A two-class label over the same twelve, for the classification chain's probes."""


def _probe(edge: str, attempt: Callable[[], object]) -> ContractProbe:
    """Run one attempt and record whether the library accepted, refused or let it escape."""
    try:
        result = attempt()
    except MLLibError as refusal:
        return ContractProbe(
            edge=edge,
            outcome="refused",
            result=None,
            error=type(refusal).__name__,
            detail=str(refusal),
        )
    except ValidationError as refusal:
        first = refusal.errors()[0]
        return ContractProbe(
            edge=edge,
            outcome="refused",
            result=None,
            error="ValidationError",
            detail=f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}",
        )
    except Exception as escaped:  # noqa: BLE001 - the whole point is to report what escaped
        return ContractProbe(
            edge=edge,
            outcome="escaped",
            result=None,
            error=type(escaped).__name__,
            detail=str(escaped),
        )
    return ContractProbe(
        edge=edge,
        outcome="accepted",
        result=None if result is None else str(result),
        error=None,
        detail=None,
    )


@cache
def _contract_probes() -> ContractsResponse:
    """Every edge in the page's table, asked once and remembered."""
    heights = Feature(HEIGHT, TWELVE_HEIGHTS)
    weights = Feature(WEIGHT, TWELVE_WEIGHTS)
    adult = Feature("adult", TWELVE_ADULT)
    fitted = _chain(2, 1.0).fit([heights], weights)
    two_column_chain = RegressionPipeline(
        steps=PipelineSteps.of(scaler=Standardizer()),
        model=RidgeRegression(penalty=1.0),
    ).fit([heights, Feature("age", TWELVE_WEIGHTS)], weights)
    prefitted = Standardizer().fit([Feature(HEIGHT, [100.0, 200.0, 300.0])])
    # Built as dictionaries so the wrong keyword and the wrong model type reach
    # the constructor at runtime rather than being refused by a type checker.
    unknown_keyword: dict[str, object] = {
        "stepz": PipelineSteps(),
        "model": RidgeRegression(penalty=1.0),
    }
    binary_model: dict[str, object] = {
        "steps": PipelineSteps(),
        "model": LogisticRegression(),
    }

    def fitted_step_handed_in() -> str:
        chain = RegressionPipeline(
            steps=PipelineSteps.of(scaler=prefitted), model=RidgeRegression(penalty=1.0)
        ).fit([heights], weights)
        scaler = chain.fitted_steps["scaler"].transformer
        assert isinstance(scaler, Standardizer)
        return (
            f"refitted from scratch, centre {scaler.scalings[HEIGHT].mean:.4f}; "
            f"the handed-in scaler still says {prefitted.scalings[HEIGHT].mean:.1f}"
        )

    def no_steps() -> str:
        gap = _max_gap(
            RegressionPipeline(model=RidgeRegression(penalty=1.0))
            .fit([heights], weights)
            .predict([heights]),
            RidgeRegression(penalty=1.0).fit([heights], weights).predict([heights]),
        )
        return f"the bare model, to a gap of {gap:.1e}"

    probes = [
        _probe("no steps at all", no_steps),
        _probe("no features at all", lambda: _chain(2, 1.0).fit([], weights)),
        _probe(
            "one training person",
            lambda: _chain(2, 1.0).fit(
                [Feature(HEIGHT, [170.0])], Feature(WEIGHT, [70.0])
            ),
        ),
        _probe(
            "a constant height column",
            lambda: _chain(2, 1.0).fit([Feature(HEIGHT, [170.0] * 12)], weights),
        ),
        _probe(
            "a height that is not a number",
            lambda: _chain(2, 1.0).fit(
                [Feature(HEIGHT, [150.0, float("nan"), 170.0, 180.0])],
                Feature(WEIGHT, [50.0, 60.0, 70.0, 80.0]),
            ),
        ),
        _probe(
            "a target shorter than the features",
            lambda: _chain(2, 1.0).fit([heights], Feature(WEIGHT, TWELVE_WEIGHTS[:5])),
        ),
        _probe(
            "a degree the training people cannot support, 9 on 8 of them",
            lambda: (
                CrossValidation(folds=KFold(n_folds=3, random_seed=4))
                .evaluate(_chain(9, 1.0), Dataset([heights], weights))
                .mean_r2_score
            ),
        ),
        _probe("predict before fit", lambda: _chain(2, 1.0).predict([heights])),
        _probe("read the fitted steps before fit", lambda: _chain(2, 1.0).fitted_steps),
        _probe(
            "predict with a feature missing, from a chain fitted on height and age",
            lambda: two_column_chain.predict([heights]),
        ),
        _probe(
            "predict with a feature the fit never saw, met by the expansion first",
            lambda: len(
                np.asarray(fitted.predict([heights, Feature("age", TWELVE_WEIGHTS)]))
            ),
        ),
        _probe(
            "predict with a feature the fit never saw, met by the standardizer first",
            lambda: two_column_chain.predict(
                [
                    heights,
                    Feature("age", TWELVE_WEIGHTS),
                    Feature("shoe", TWELVE_WEIGHTS),
                ]
            ),
        ),
        _probe(
            "predict with the features reordered",
            lambda: len(
                np.asarray(
                    two_column_chain.predict([Feature("age", TWELVE_WEIGHTS), heights])
                )
            ),
        ),
        _probe(
            "a step already fitted on other people, handed in", fitted_step_handed_in
        ),
        _probe(
            "a model handed in as a step",
            lambda: RegressionPipeline(
                steps=PipelineSteps.of(ridge=RidgeRegression()),
                model=RidgeRegression(penalty=1.0),
            ).fit([heights], weights),
        ),
        _probe(
            "a transformer handed in as the model",
            lambda: RegressionPipeline(steps=PipelineSteps(), model=Standardizer()),
        ),
        _probe(
            "a classifier handed in as a regression chain's model",
            lambda: RegressionPipeline(
                steps=PipelineSteps(), model=KNearestNeighboursClassifier()
            ),
        ),
        _probe(
            "two steps with one name",
            lambda: PipelineSteps(
                [
                    PipelineStep("scaler", Standardizer()),
                    PipelineStep("scaler", Standardizer()),
                ]
            ),
        ),
        _probe("a blank step name", lambda: PipelineStep("   ", Standardizer())),
        _probe(
            "ask for a step the chain does not run",
            lambda: PipelineSteps.of(scaler=Standardizer())["terms"],
        ),
        _probe(
            "an unknown constructor keyword",
            lambda: RegressionPipeline(**unknown_keyword),
        ),
        _probe(
            "a search over a misspelt chain field",
            lambda: SearchSpace.over(RegressionPipeline, stepz=[_expand_then_scale(2)]),
        ),
        _probe(
            "a search asking the ridge model for a degree",
            lambda: SearchSpace.over(RidgeRegression, degree=[1, 2]),
        ),
        _probe(
            "four people dealt into five folds",
            lambda: CrossValidation(folds=KFold(n_folds=5, random_seed=4)).evaluate(
                _chain(2, 1.0),
                Dataset(
                    [Feature(HEIGHT, TWELVE_HEIGHTS[:4])],
                    Feature(WEIGHT, TWELVE_WEIGHTS[:4]),
                ),
            ),
        ),
        _probe(
            "a two-class chain fitted on the twelve",
            lambda: (
                ClassificationPipeline(
                    steps=PipelineSteps.of(scaler=Standardizer()),
                    model=KNearestNeighboursClassifier(n_neighbours=3),
                )
                .fit([heights], adult)
                .n_classes
            ),
        ),
        _probe(
            "a classification chain with one class in the target",
            lambda: ClassificationPipeline(
                steps=PipelineSteps.of(scaler=Standardizer()),
                model=KNearestNeighboursClassifier(n_neighbours=3),
            ).fit([heights], Feature("adult", [1.0] * 12)),
        ),
        _probe(
            "a binary-only classifier in a classification chain",
            lambda: ClassificationPipeline(**binary_model),
        ),
    ]
    return ContractsResponse(probes=probes)


@router.get("/concepts/pipelines/contracts", response_model=ContractsResponse)
def contracts() -> ContractsResponse:
    """Every edge the page's last part tabulates, asked of the library."""
    return _contract_probes()
