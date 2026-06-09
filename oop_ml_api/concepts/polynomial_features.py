"""Polynomial features, and buying curvature with a linear model.

Every endpoint here is registered under ``/concepts/polynomial-features``, and the router is
created and registered before any endpoint exists, so the pages of this section
can be built in parallel without many edits to one shared file.

The page is about raising and multiplying columns so a straight-line fit can bend, and what that costs in columns.

The page's people are a crowd of twenty-two measured three ways, height in
centimetres, waist girth in centimetres and weight in kilograms, running from a
104 cm child to a 191 cm adult. Weight follows height and girth the way the
volume of a cylinder does, so the relationship genuinely bends and the term that
would express it exactly is a product of a power. A second group of twelve is
laid out as four builds at each of three heights, so that height and girth are
uncorrelated there and the cross term has something to say that the powers
cannot; on the crowd the two columns move together and it has almost nothing to
say, which is measured rather than glossed over.

Seven endpoints. ``expansion`` names and computes every column the expansion
builds; ``column-counts`` reports how many there are at each degree and each
original width; ``curve-fit`` fits one degree through one column and hands back
the curve to draw; ``degree-gap`` sweeps the degree with and without a penalty
on rows the fit never saw; ``two-columns`` compares a plane against the powers
alone against the powers and the products; ``units`` measures what raising a
column measured in the hundreds to a power does to it; and
``degenerate-columns`` probes the two expansions whose extra columns carry no
extra information, a two-valued column whose square is itself and a column with
fewer distinct values than the degree asks for.

The library's refusals are readable and are shown, with the two that name an
identifier rewritten by ``_in_plain_words``, which a test pins.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from typing import TypeVar

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    Dataset,
    DecisionTreeRegressor,
    Feature,
    MultipleLinearRegression,
    PolynomialFeatures,
    RidgeRegression,
    Standardizer,
    TrainTestSplitter,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/polynomial-features", tags=["polynomial-features"])

MAX_DEGREE = 14
"""The highest total degree any request may ask for.

Fourteen is the interpolation point for the page's fifteen training rows, which
is where the sweep in the fifth part is going, and nothing above it teaches
anything the number below already showed.
"""

MAX_COLUMNS = 4
"""How many original columns an expansion request may carry.

Four at degree five is 125 columns, already past anything the page draws, and
the count grows fast enough that a wider request is a compute job rather than a
teaching one.
"""

CURVE_SAMPLES = 121
"""How many places the fitted curve is sampled at across the range of the data."""

HELD_OUT_FRACTION = 0.3
"""The share of the people kept back from every fit that reports a second score."""

HELD_OUT_SEED = 3
"""One seed, drawn once, so every widget on the page holds back the same people."""

COUNT_WIDTHS = (1, 2, 3, 5, 10)
"""Original column counts the growth table reports."""

COUNT_DEGREES = (1, 2, 3, 4, 5)
"""Degrees the growth table reports."""

TREE_DEPTHS = (1, 2, 3, 4, 5)
"""Depths the comparison against a stepped fit runs at."""


NOT_FITTED_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z]* must be fit before this is available"
)
"""A refusal that opens on the class name of whatever was asked too early."""

COLLINEAR_PATTERN = re.compile(r"the features are collinear.*?solvable", re.DOTALL)
"""The refusal for a duplicated column, which names a class in its own advice."""

COLLINEAR_IN_PLAIN_WORDS = (
    "one of the columns says exactly what the others already say, so there is no "
    "single set of weights the fit can settle on; drop the repeated column, or "
    "shrink the weights, which picks one of the answers"
)
"""The same refusal written for a reader who has met no code."""

PLAIN_WORDS = (
    ("feature_values", "a column’s values"),
    (" -- ", ", "),
)
"""Wording a reader has no reason to meet, and what to say instead."""


def _in_plain_words(refusal: MLLibError) -> str:
    """One refusal's sentence, with the identifiers taken out of it.

    Most of these messages are already written to be read, so this rewrites the
    two that name a class or a parameter and leaves the rest alone. A test
    asserts that no identifier survives, so a message that grows one later fails
    there rather than reaching the page.
    """
    sentence = NOT_FITTED_PATTERN.sub(
        "this expansion must be fitted before that can be read", str(refusal)
    )
    sentence = COLLINEAR_PATTERN.sub(COLLINEAR_IN_PLAIN_WORDS, sentence)
    for identifier, replacement in PLAIN_WORDS:
        sentence = sentence.replace(identifier, replacement)
    return sentence


Computed = TypeVar("Computed")


def _computed(attempt: Callable[[], Computed]) -> Computed:
    """Run one fit, turning a refusal into a typed 422 in the page's own words."""
    try:
        return attempt()
    except MLLibError as refusal:
        raise HTTPException(
            status_code=422, detail=_in_plain_words(refusal)
        ) from refusal


def _expanded(
    degree: int, columns: list[Feature], *, products: bool = True
) -> PolynomialFeatures:
    """Fix an expansion's columns on the rows the fit is allowed to see.

    Named rather than written inline because every endpoint here builds one the
    same way inside a loop, and a closure over a loop variable is the one thing
    that would quietly build the wrong degree.
    """
    return _computed(
        lambda: PolynomialFeatures(degree=degree, include_interactions=products).fit(
            columns
        )
    )


def _fitted(
    model: MultipleLinearRegression | RidgeRegression | DecisionTreeRegressor,
    columns: list[Feature],
    target: Feature,
) -> MultipleLinearRegression | RidgeRegression | DecisionTreeRegressor:
    """Fit one model on one set of columns, refusing in the page's own words."""
    return _computed(lambda: model.fit(columns, target))


class Point(BaseModel):
    """One person as the single-column widgets read them, height and weight."""

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class Person(BaseModel):
    """One person measured three ways, for the two-column comparisons."""

    height: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    girth: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    weight: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)


class NamedColumn(BaseModel):
    """One original column, its name and its values."""

    name: str = Field(min_length=1, max_length=24)
    values: list[float] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)


class ExpansionRequest(BaseModel):
    """The columns to expand, how high to go, and whether to build the products."""

    columns: list[NamedColumn] = Field(min_length=1, max_length=MAX_COLUMNS)
    degree: int = Field(default=2, ge=1, le=MAX_DEGREE)
    include_interactions: bool = True

    model_config = {"extra": "forbid"}


class Exponent(BaseModel):
    """One factor of a term, the column and the power it is raised to."""

    feature: str
    exponent: int


class ExpandedTerm(BaseModel):
    """One column the expansion built, named for the term that produced it."""

    name: str
    exponents: list[Exponent]
    total_degree: int
    values: list[float]


class ExpansionResponse(BaseModel):
    """Every column the expansion produced, in the order it fixed them.

    ``source_names`` are the original columns in the order they were supplied,
    which is the order the terms are written in, and ``n_terms`` is how many
    columns the fit will be handed.
    """

    source_names: list[str]
    terms: list[ExpandedTerm]
    n_terms: int


@router.post("/expansion", response_model=ExpansionResponse)
def expansion(request: ExpansionRequest) -> ExpansionResponse:
    """Build every power and product up to the degree, and hand back the columns."""
    lengths = {len(column.values) for column in request.columns}
    if len(lengths) != 1:
        raise HTTPException(
            status_code=422,
            detail="every column has to carry the same number of people",
        )

    features = [Feature(column.name, column.values) for column in request.columns]
    built = _expanded(request.degree, features, products=request.include_interactions)
    columns = _computed(lambda: built.transform(features))

    return ExpansionResponse(
        source_names=[feature.name for feature in features],
        terms=[
            ExpandedTerm(
                name=term.name,
                exponents=[
                    Exponent(feature=name, exponent=power)
                    for name, power in term.powers.items()
                ],
                total_degree=term.total_degree,
                values=[float(value) for value in column.values],
            )
            for term, column in zip(built.terms, columns, strict=True)
        ],
        n_terms=built.terms.n_terms,
    )


class CountRow(BaseModel):
    """How many columns one original width produces at each degree."""

    width: int
    counts: list[int]


class ColumnCountsResponse(BaseModel):
    """The growth table, every entry read off a built expansion."""

    widths: list[int]
    degrees: list[int]
    rows: list[CountRow]


@cache
def _column_counts() -> ColumnCountsResponse:
    """Build each expansion once and count the columns it produced."""
    rows: list[CountRow] = []
    for width in COUNT_WIDTHS:
        columns = [
            Feature(f"column_{position}", np.arange(4.0) + position + 1.0)
            for position in range(width)
        ]
        counts = [
            PolynomialFeatures(degree=degree).fit(columns).terms.n_terms
            for degree in COUNT_DEGREES
        ]
        rows.append(CountRow(width=width, counts=counts))
    return ColumnCountsResponse(
        widths=list(COUNT_WIDTHS), degrees=list(COUNT_DEGREES), rows=rows
    )


@router.get("/column-counts", response_model=ColumnCountsResponse)
def column_counts() -> ColumnCountsResponse:
    """How many columns an expansion produces, by degree and by original width."""
    return _column_counts()


def _split_of(points: list[Point]) -> tuple[Dataset, Dataset]:
    """Cut the people into rows to fit on and rows to score on, under one seed."""
    data = Dataset(
        [Feature("height", [point.x for point in points])],
        Feature("weight", [point.y for point in points]),
    )
    split = _computed(
        lambda: TrainTestSplitter(
            test_fraction=HELD_OUT_FRACTION, random_seed=HELD_OUT_SEED
        ).split(data)
    )
    return split.training, split.testing


def _held_out_positions(points: list[Point], held_out: Dataset) -> list[int]:
    """Which of the supplied people landed in the held-back block, by position."""
    kept = [
        (float(height), float(weight))
        for height, weight in zip(
            held_out.input_features[0].values,
            held_out.target_feature.values,
            strict=True,
        )
    ]
    positions: list[int] = []
    for wanted in kept:
        for position, point in enumerate(points):
            if position in positions:
                continue
            if (point.x, point.y) == wanted:
                positions.append(position)
                break
    return sorted(positions)


class CurveFitRequest(BaseModel):
    """The people, the degree to bend through them, and the two switches.

    ``penalty`` above zero shrinks the fitted weights, which is the repair the
    fifth part measures; ``standardise`` puts the height column on a common
    scale before it is raised, which the sixth part shows is not optional at a
    high degree.
    """

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(default=1, ge=1, le=MAX_DEGREE)
    penalty: float = Field(default=0.0, ge=0.0, le=1000.0)
    standardise: bool = True

    model_config = {"extra": "forbid"}


class NamedCoefficient(BaseModel):
    """One fitted weight, bound to the name of the column it multiplies."""

    name: str
    value: float


class CurvePoint(BaseModel):
    """One sample of the fitted curve."""

    x: float
    y: float


class CurveFitResponse(BaseModel):
    """The fit at one degree, scored twice and sampled for drawing.

    ``fitted`` and ``residuals`` carry one entry per supplied person, in the
    order they were sent, whichever half they landed in, so a residual can be
    drawn under the person that produced it. ``train_r_squared`` is read on the
    rows the fit saw and ``held_out_r_squared`` on the rows it did not.
    """

    n_terms: int
    term_names: list[str]
    coefficients: list[NamedCoefficient]
    intercept: float
    n_training_rows: int
    held_out_indices: list[int]
    train_r_squared: float
    held_out_r_squared: float
    residual_sum_of_squares: float
    fitted: list[float]
    residuals: list[float]
    curve: list[CurvePoint]


def _refuse_unfittable(degree: int, n_training_rows: int) -> None:
    """Refuse a degree that asks for more numbers than the rows can pin down."""
    if degree + 1 > n_training_rows:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a degree {degree} curve has {degree + 1} numbers to pin down, "
                f"and only {n_training_rows} people are being fitted"
            ),
        )


def _model_of(penalty: float) -> MultipleLinearRegression | RidgeRegression:
    """Least squares when no penalty was asked for, and shrinkage when one was."""
    return (
        RidgeRegression(penalty=penalty)
        if penalty > 0.0
        else MultipleLinearRegression()
    )


@router.post("/curve-fit", response_model=CurveFitResponse)
def curve_fit(request: CurveFitRequest) -> CurveFitResponse:
    """Expand one column to the chosen degree, fit it, and sample the curve."""
    training, held_out = _split_of(request.points)
    _refuse_unfittable(request.degree, training.n_samples)

    train_columns = list(training.input_features)
    held_columns = list(held_out.input_features)
    scaler = Standardizer().fit(train_columns) if request.standardise else None

    def prepared(columns: list[Feature]) -> list[Feature]:
        return columns if scaler is None else scaler.transform(columns)

    built = _expanded(request.degree, prepared(train_columns))
    fitted_model = _fitted(
        _model_of(request.penalty),
        built.transform(prepared(train_columns)),
        training.target_feature,
    )

    heights = [point.x for point in request.points]
    weights = [point.y for point in request.points]
    everyone = [Feature("height", heights)]
    predicted = np.asarray(
        fitted_model.predict(built.transform(prepared(everyone))), dtype=np.float64
    )

    grid = np.linspace(min(heights), max(heights), CURVE_SAMPLES)
    drawn = np.asarray(
        fitted_model.predict(built.transform(prepared([Feature("height", grid)]))),
        dtype=np.float64,
    )

    training_score = float(
        fitted_model.score(
            built.transform(prepared(train_columns)), training.target_feature
        )
    )
    held_out_score = float(
        fitted_model.score(
            built.transform(prepared(held_columns)), held_out.target_feature
        )
    )
    residuals = np.asarray(weights, dtype=np.float64) - predicted

    return CurveFitResponse(
        n_terms=built.terms.n_terms,
        term_names=list(built.terms.names),
        coefficients=[
            NamedCoefficient(name=coefficient.name, value=float(coefficient.value))
            for coefficient in fitted_model.coefficients
        ],
        intercept=float(fitted_model.intercept),
        n_training_rows=training.n_samples,
        held_out_indices=_held_out_positions(request.points, held_out),
        train_r_squared=training_score,
        held_out_r_squared=held_out_score,
        residual_sum_of_squares=float(np.sum(residuals**2)),
        fitted=[float(value) for value in predicted],
        residuals=[float(value) for value in residuals],
        curve=[
            CurvePoint(x=float(x), y=float(y)) for x, y in zip(grid, drawn, strict=True)
        ],
    )


class DegreeGapRequest(BaseModel):
    """The people, how far up the degree to climb, and which penalties to try."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_degree: int = Field(default=MAX_DEGREE, ge=1, le=MAX_DEGREE)
    penalties: list[float] = Field(default_factory=list, max_length=4)
    standardise: bool = True

    model_config = {"extra": "forbid"}


class PenalisedReading(BaseModel):
    """One degree scored again with the fitted weights shrunk."""

    penalty: float
    train_r_squared: float
    held_out_r_squared: float


class DegreeReading(BaseModel):
    """One degree of the sweep, scored on both halves."""

    degree: int
    n_terms: int
    train_r_squared: float
    held_out_r_squared: float
    penalised: list[PenalisedReading]


class DegreeGapResponse(BaseModel):
    """The whole staircase, with the held-back rows named once.

    ``best_held_out_degree`` is the degree whose second score is highest, which
    is a reading rather than a recommendation, and ``interpolating_degree`` is
    the degree at which the fit has as many numbers to set as there are rows to
    set them from.
    """

    n_training_rows: int
    held_out_indices: list[int]
    readings: list[DegreeReading]
    best_held_out_degree: int
    best_held_out_score: float
    worst_held_out_degree: int
    worst_held_out_score: float
    interpolating_degree: int


@router.post("/degree-gap", response_model=DegreeGapResponse)
def degree_gap(request: DegreeGapRequest) -> DegreeGapResponse:
    """Fit every degree up to the ceiling and score each on both halves."""
    training, held_out = _split_of(request.points)
    train_columns = list(training.input_features)
    held_columns = list(held_out.input_features)
    scaler = Standardizer().fit(train_columns) if request.standardise else None

    def prepared(columns: list[Feature]) -> list[Feature]:
        return columns if scaler is None else scaler.transform(columns)

    highest = min(request.max_degree, training.n_samples - 1)
    readings: list[DegreeReading] = []
    for degree in range(1, highest + 1):
        built = _expanded(degree, prepared(train_columns))
        train_terms = built.transform(prepared(train_columns))
        held_terms = built.transform(prepared(held_columns))

        penalised: list[PenalisedReading] = []
        scores: list[float] = []
        for penalty in [0.0, *request.penalties]:
            model = _fitted(_model_of(penalty), train_terms, training.target_feature)
            train_score = float(model.score(train_terms, training.target_feature))
            held_score = float(model.score(held_terms, held_out.target_feature))
            if penalty == 0.0:
                scores = [train_score, held_score]
            else:
                penalised.append(
                    PenalisedReading(
                        penalty=penalty,
                        train_r_squared=train_score,
                        held_out_r_squared=held_score,
                    )
                )

        readings.append(
            DegreeReading(
                degree=degree,
                n_terms=built.terms.n_terms,
                train_r_squared=scores[0],
                held_out_r_squared=scores[1],
                penalised=penalised,
            )
        )

    best = max(readings, key=lambda reading: reading.held_out_r_squared)
    worst = min(readings, key=lambda reading: reading.held_out_r_squared)
    return DegreeGapResponse(
        n_training_rows=training.n_samples,
        held_out_indices=_held_out_positions(request.points, held_out),
        readings=readings,
        best_held_out_degree=best.degree,
        best_held_out_score=best.held_out_r_squared,
        worst_held_out_degree=worst.degree,
        worst_held_out_score=worst.held_out_r_squared,
        interpolating_degree=training.n_samples - 1,
    )


class TwoColumnRequest(BaseModel):
    """The people measured three ways, and how high the expansion should go."""

    people: list[Person] = Field(min_length=3, max_length=MAX_POINTS)
    degree: int = Field(default=2, ge=1, le=5)

    model_config = {"extra": "forbid"}


class ColumnSetFit(BaseModel):
    """One set of columns, fitted, with the misses it left."""

    label: str
    n_terms: int
    term_names: list[str]
    r_squared: float
    residual_sum_of_squares: float


class GirthGain(BaseModel):
    """What a fixed amount of girth is worth at one height, three ways.

    ``measured`` reads it straight off the people, ``plane`` is what the fit
    without any product says, and ``expanded`` is what the fit with the products
    says. The plane is obliged to give one number for every height.
    """

    height: float
    from_girth: float
    to_girth: float
    measured: float
    plane: float
    expanded: float


class TwoColumnResponse(BaseModel):
    """A plane, the powers alone, and the powers with the products.

    ``correlation`` is between the two original columns, and it is the number
    that decides whether the products have anything to add, since two columns
    that move together already stand in for each other.
    """

    correlation: float
    fits: list[ColumnSetFit]
    gains: list[GirthGain]


def _fit_columns(
    label: str, columns: list[Feature], target: Feature, degree: int, products: bool
) -> tuple[ColumnSetFit, MultipleLinearRegression, PolynomialFeatures]:
    """Expand one way, fit it, and report the score and the total of the misses."""
    built = _expanded(degree, columns, products=products)
    terms = built.transform(columns)
    model = _fitted(MultipleLinearRegression(), terms, target)
    evaluation = model.evaluate(terms, target)
    return (
        ColumnSetFit(
            label=label,
            n_terms=built.terms.n_terms,
            term_names=list(built.terms.names),
            r_squared=float(evaluation.r2_score),
            residual_sum_of_squares=float(evaluation.residual_sum_of_squares),
        ),
        model,
        built,
    )


@router.post("/two-columns", response_model=TwoColumnResponse)
def two_columns(request: TwoColumnRequest) -> TwoColumnResponse:
    """Compare a plane, the pure powers, and the powers with the products."""
    heights = [person.height for person in request.people]
    girths = [person.girth for person in request.people]
    weights = [person.weight for person in request.people]
    columns = [Feature("height", heights), Feature("girth", girths)]
    target = Feature("weight", weights)

    plane_fit, plane_model, plane_built = _fit_columns(
        "height and girth", columns, target, 1, True
    )
    powers_fit, _, _ = _fit_columns(
        "the powers alone", columns, target, request.degree, False
    )
    products_fit, product_model, product_built = _fit_columns(
        "the powers and the products", columns, target, request.degree, True
    )

    gains: list[GirthGain] = []
    for height in sorted(set(heights)):
        at_height = [
            (girth, weight)
            for person_height, girth, weight in zip(
                heights, girths, weights, strict=True
            )
            if person_height == height
        ]
        if len(at_height) < 2:
            continue
        low = min(at_height, key=lambda pair: pair[0])
        high = max(at_height, key=lambda pair: pair[0])
        ends = [
            Feature("height", [height, height]),
            Feature("girth", [low[0], high[0]]),
        ]
        plane_ends = np.asarray(
            plane_model.predict(plane_built.transform(ends)), dtype=np.float64
        )
        product_ends = np.asarray(
            product_model.predict(product_built.transform(ends)), dtype=np.float64
        )
        gains.append(
            GirthGain(
                height=float(height),
                from_girth=float(low[0]),
                to_girth=float(high[0]),
                measured=float(high[1] - low[1]),
                plane=float(plane_ends[1] - plane_ends[0]),
                expanded=float(product_ends[1] - product_ends[0]),
            )
        )

    correlation = float(np.corrcoef(heights, girths)[0, 1])
    return TwoColumnResponse(
        correlation=correlation,
        fits=[plane_fit, powers_fit, products_fit],
        gains=gains,
    )


class UnitsRequest(BaseModel):
    """The people, the degree to raise the column to, and a penalty to feel it."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    degree: int = Field(default=3, ge=2, le=6)
    penalty: float = Field(default=1.0, gt=0.0, le=1000.0)
    tree_depth: int = Field(default=3, ge=1, le=8)

    model_config = {"extra": "forbid"}


class ColumnSpan(BaseModel):
    """One expanded column, and the largest value in it."""

    name: str
    smallest: float
    largest: float


class UnitReading(BaseModel):
    """The same people with height written in one unit, and what changed.

    ``plain_r_squared`` is least squares, which is untouched by the unit because
    a coefficient can absorb it; ``penalised_r_squared`` is the same fit with the
    weights shrunk, which is not, because the penalty is a number about
    coefficients and the coefficients changed size. ``tree_r_squared`` is a
    stepped fit on the raw column, which asks only which side of a threshold a
    value falls.
    """

    unit: str
    factor: float
    columns: list[ColumnSpan]
    span_ratio: float
    plain_r_squared: float
    penalised_r_squared: float
    penalised_coefficients: list[NamedCoefficient]
    tree_r_squared: float


class HighDegreeReading(BaseModel):
    """The interpolating degree, reached from one reading of the column."""

    reading: str
    degree: int
    largest_value: float
    train_r_squared: float
    held_out_r_squared: float


class UnitsResponse(BaseModel):
    """What a change of unit does to an expansion, and what it does not.

    ``high_degree`` runs the interpolating degree twice, once on the column as
    measured and once with it put on a common scale first, and the training
    score is the honest report, since a fit with as many numbers as rows should
    reach one.
    """

    readings: list[UnitReading]
    high_degree: list[HighDegreeReading]


UNIT_FACTORS = (("centimetres", 1.0), ("metres", 0.01))
"""Height written two ways, each a factor on the same numbers."""


@router.post("/units", response_model=UnitsResponse)
def units(request: UnitsRequest) -> UnitsResponse:
    """Rewrite the height column in each unit and refit everything on it."""
    heights = np.asarray([point.x for point in request.points], dtype=np.float64)
    target = Feature("weight", [point.y for point in request.points])

    readings: list[UnitReading] = []
    for unit, factor in UNIT_FACTORS:
        column = Feature("height", heights * factor)
        built = _expanded(request.degree, [column])
        terms = built.transform([column])
        plain = _fitted(MultipleLinearRegression(), terms, target)
        shrunk = _fitted(RidgeRegression(penalty=request.penalty), terms, target)
        tree = _fitted(
            DecisionTreeRegressor(max_depth=request.tree_depth), [column], target
        )
        spans = [
            ColumnSpan(
                name=term.name,
                smallest=float(np.min(np.abs(term.values))),
                largest=float(np.max(np.abs(term.values))),
            )
            for term in terms
        ]
        widest = max(span.largest for span in spans)
        narrowest = min(span.largest for span in spans)
        readings.append(
            UnitReading(
                unit=unit,
                factor=factor,
                columns=spans,
                span_ratio=float(widest / narrowest),
                plain_r_squared=float(plain.score(terms, target)),
                penalised_r_squared=float(shrunk.score(terms, target)),
                penalised_coefficients=[
                    NamedCoefficient(name=weight.name, value=float(weight.value))
                    for weight in shrunk.coefficients
                ],
                tree_r_squared=float(tree.score([column], target)),
            )
        )

    training, held_out = _split_of(request.points)
    train_columns = list(training.input_features)
    held_columns = list(held_out.input_features)
    scaler = Standardizer().fit(train_columns)
    interpolating = training.n_samples - 1

    high_degree: list[HighDegreeReading] = []
    for label, train_side, held_side in (
        ("as measured", train_columns, held_columns),
        (
            "on a common scale",
            scaler.transform(train_columns),
            scaler.transform(held_columns),
        ),
    ):
        built = _expanded(interpolating, train_side)
        train_terms = built.transform(train_side)
        held_terms = built.transform(held_side)
        model = _fitted(
            MultipleLinearRegression(), train_terms, training.target_feature
        )
        high_degree.append(
            HighDegreeReading(
                reading=label,
                degree=interpolating,
                largest_value=float(
                    max(float(np.max(np.abs(term.values))) for term in train_terms)
                ),
                train_r_squared=float(
                    model.score(train_terms, training.target_feature)
                ),
                held_out_r_squared=float(
                    model.score(held_terms, held_out.target_feature)
                ),
            )
        )

    return UnitsResponse(readings=readings, high_degree=high_degree)


class TreeComparisonRequest(BaseModel):
    """The people, how far up the degree to climb, and how deep to grow."""

    points: list[Point] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    max_degree: int = Field(default=6, ge=1, le=MAX_DEGREE)

    model_config = {"extra": "forbid"}


class StepReading(BaseModel):
    """One depth of the stepped fit, scored on both halves."""

    depth: int
    n_leaves: int
    train_r_squared: float
    held_out_r_squared: float


class TreeComparisonResponse(BaseModel):
    """A curve built from powers against a fit made of straight cuts.

    ``curve_readings`` are the expanded linear fits at each degree and
    ``step_readings`` the stepped fits at each depth, both scored on the same
    held-back people, so the two columns are comparable. ``worst`` on each side
    is what the method does when its one dial is set badly, which is the honest
    half of the comparison.
    """

    n_training_rows: int
    held_out_indices: list[int]
    curve_readings: list[DegreeReading]
    step_readings: list[StepReading]
    best_curve_degree: int
    best_curve_score: float
    worst_curve_score: float
    best_step_depth: int
    best_step_score: float
    worst_step_score: float


@router.post("/against-a-tree", response_model=TreeComparisonResponse)
def against_a_tree(request: TreeComparisonRequest) -> TreeComparisonResponse:
    """Score the expansion and a stepped fit on the same held-back people."""
    training, held_out = _split_of(request.points)
    train_columns = list(training.input_features)
    held_columns = list(held_out.input_features)
    scaler = Standardizer().fit(train_columns)

    highest = min(request.max_degree, training.n_samples - 1)
    curve_readings: list[DegreeReading] = []
    for degree in range(1, highest + 1):
        built = _expanded(degree, scaler.transform(train_columns))
        train_terms = built.transform(scaler.transform(train_columns))
        held_terms = built.transform(scaler.transform(held_columns))
        model = _fitted(
            MultipleLinearRegression(), train_terms, training.target_feature
        )
        curve_readings.append(
            DegreeReading(
                degree=degree,
                n_terms=built.terms.n_terms,
                train_r_squared=float(
                    model.score(train_terms, training.target_feature)
                ),
                held_out_r_squared=float(
                    model.score(held_terms, held_out.target_feature)
                ),
                penalised=[],
            )
        )

    step_readings: list[StepReading] = []
    for depth in TREE_DEPTHS:
        tree = _fitted(
            DecisionTreeRegressor(max_depth=depth),
            train_columns,
            training.target_feature,
        )
        step_readings.append(
            StepReading(
                depth=depth,
                n_leaves=tree.n_leaves,
                train_r_squared=float(
                    tree.score(train_columns, training.target_feature)
                ),
                held_out_r_squared=float(
                    tree.score(held_columns, held_out.target_feature)
                ),
            )
        )

    best_curve = max(curve_readings, key=lambda reading: reading.held_out_r_squared)
    best_step = max(step_readings, key=lambda reading: reading.held_out_r_squared)
    return TreeComparisonResponse(
        n_training_rows=training.n_samples,
        held_out_indices=_held_out_positions(request.points, held_out),
        curve_readings=curve_readings,
        step_readings=step_readings,
        best_curve_degree=best_curve.degree,
        best_curve_score=best_curve.held_out_r_squared,
        worst_curve_score=min(reading.held_out_r_squared for reading in curve_readings),
        best_step_depth=best_step.depth,
        best_step_score=best_step.held_out_r_squared,
        worst_step_score=min(reading.held_out_r_squared for reading in step_readings),
    )


INDICATOR_HEIGHTS = (150.0, 160.0, 170.0, 180.0, 155.0, 165.0, 175.0, 185.0)
"""Eight people, for the probe where the second column takes only two values."""

INDICATOR_FLAGS = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0)
"""A column recording whether each of them swims, written as zero and one."""

INDICATOR_WEIGHTS = (50.0, 62.0, 70.0, 84.0, 54.0, 63.0, 77.0, 88.0)
"""Their weights, so the probe has something to fit."""

REPEATED_HEIGHTS = (
    150.0,
    150.0,
    165.0,
    165.0,
    180.0,
    180.0,
    150.0,
    165.0,
    180.0,
    150.0,
)
"""Ten people standing at only three distinct heights."""

REPEATED_WEIGHTS = (50.0, 52.0, 63.0, 65.0, 80.0, 82.0, 51.0, 64.0, 79.0, 49.0)
"""Their weights, for the probe about how many powers a column can support."""

REPEATED_DEGREES = (2, 3, 4, 5)
"""Degrees asked of that column, none of which can say anything new."""


class TwoValuedProbe(BaseModel):
    """What happens when a column takes only the values zero and one.

    ``columns_match`` compares the column with its own square value by value,
    and ``fit_succeeded`` says whether a fit on the resulting set had a unique
    answer, with ``reason`` carrying the refusal in the page's own words.
    """

    column_name: str
    squared_name: str
    original: list[float]
    squared: list[float]
    columns_match: bool
    fit_succeeded: bool
    reason: str | None


class RepeatedValueReading(BaseModel):
    """One degree asked of a column that takes only three distinct values."""

    degree: int
    n_terms: int
    r_squared: float


class DegenerateResponse(BaseModel):
    """The two cases where an expansion builds columns that say nothing new.

    Both are quiet rather than loud. The first produces a duplicate column and
    the fit has no unique answer; the second produces columns that are new
    arithmetic and old information, so the score stops moving however high the
    degree goes.
    """

    two_valued: TwoValuedProbe
    distinct_values: int
    repeated: list[RepeatedValueReading]


@cache
def _degenerate() -> DegenerateResponse:
    """Run both probes once, since neither depends on anything a reader sends."""
    height = Feature("height", list(INDICATOR_HEIGHTS))
    swims = Feature("swims", list(INDICATOR_FLAGS))
    target = Feature("weight", list(INDICATOR_WEIGHTS))

    built = PolynomialFeatures(degree=2).fit([height, swims])
    columns = {term.name: term for term in built.transform([height, swims])}
    original = [float(value) for value in columns["swims"].values]
    squared = [float(value) for value in columns["swims^2"].values]

    succeeded = True
    reason: str | None = None
    try:
        MultipleLinearRegression().fit(list(columns.values()), target)
    except MLLibError as refusal:
        succeeded = False
        reason = _in_plain_words(refusal)

    few = Feature("height", list(REPEATED_HEIGHTS))
    few_target = Feature("weight", list(REPEATED_WEIGHTS))
    readings: list[RepeatedValueReading] = []
    for degree in REPEATED_DEGREES:
        expansion = PolynomialFeatures(degree=degree).fit([few])
        terms = expansion.transform([few])
        model = MultipleLinearRegression().fit(terms, few_target)
        readings.append(
            RepeatedValueReading(
                degree=degree,
                n_terms=expansion.terms.n_terms,
                r_squared=float(model.score(terms, few_target)),
            )
        )

    return DegenerateResponse(
        two_valued=TwoValuedProbe(
            column_name="swims",
            squared_name="swims^2",
            original=original,
            squared=squared,
            columns_match=original == squared,
            fit_succeeded=succeeded,
            reason=reason,
        ),
        distinct_values=len(set(REPEATED_HEIGHTS)),
        repeated=readings,
    )


@router.get("/degenerate-columns", response_model=DegenerateResponse)
def degenerate_columns() -> DegenerateResponse:
    """The two expansions that build columns carrying nothing new."""
    return _degenerate()
