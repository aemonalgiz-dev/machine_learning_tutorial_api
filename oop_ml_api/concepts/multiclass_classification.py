"""Three classes, and the two honest ways to reach them, for the multiclass page.

The logistic curve chooses between two. The page's crowd holds children,
teenagers and adults, and the library extends the same curve to them in two
ways that do not agree about whether the answers should add up.
``MultinomialLogisticRegression`` gives every class its own score, holds class
0's score at zero as the reference, and shares one unit of probability out
across the classes with a softmax, so its ``predict_probabilities`` returns a
``ProbabilityMatrix`` whose rows sum to one by construction.
``OneVsRestClassifier`` fits one binary ``LogisticRegression`` per class, each
asked whether a row is its class or anything else, and reports the three
separate chances side by side as ``ClassScores``, the type that promises the
bound and deliberately not the sum, because three fits that were never
introduced to one another have no reason to add up.

One endpoint serves both, chosen by a ``route`` field, so the widget can flip
between them on the same crowd. The heights and weights are standardised
before the ascent, as the judging page's boundary is, because a gradient walk
on columns near 150 and 50 oscillates where the same walk on standardised
columns settles. Beside each person's scores the response carries the scores
before the squash, each class's intercept and weights applied to that person,
so the page's worked example can exponentiate and divide by hand and arrive at
the library's own numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from oop_ml import (
    DesignMatrix,
    Feature,
    LogisticRegression,
    MultinomialLogisticRegression,
    OneVsRestClassifier,
    Predictions,
    Standardizer,
)
from oop_ml.core.logistic import stable_softmax
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    GRID_CELLS,
    RegionGrid,
    region_grid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS

router = APIRouter(
    prefix="/concepts/multiclass-classification", tags=["multiclass-classification"]
)

N_CLASSES = 3
"""The page is about three classes, and the widget colours exactly that many."""

CLASS_NAMES = ("child", "teenager", "adult")
"""What the pages call classes 0, 1 and 2, for a refusal a reader can act on."""

LEARNING_RATE = 1.0
"""The stride of the ascent, on standardised columns where a stride of one is tame."""

MAX_EPOCHS = 500
"""How long each ascent is allowed.

The worked example's three people are separable and so is the crowd, so the
likelihood has no maximum, and a walk given ten thousand passes spends them
steepening boundaries it placed in the first hundred. Five hundred passes
place every boundary, cost a few hundredths of a second for both routes
together, and leave the scores readable rather than rounded to certainty,
which is what the page quotes.
"""

TOLERANCE = 1e-6
"""Stop once no coefficient moves further than this in a whole pass."""

Route = Literal["softmax", "one_vs_rest"]
"""The two extensions of the logistic curve the page compares."""


class ClassedPoint(BaseModel):
    """One person, their two measurements, and which of the three classes they are.

    Not the shared ``LabelledPoint``, whose label is bounded to the two classes
    every other classification page has, since widening that bound here would
    loosen it there.
    """

    x: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    y: float = Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE)
    label: int = Field(ge=0, le=N_CLASSES - 1)


class ClassifyRequest(BaseModel):
    """The crowd, which of the two routes to three classes to take, and how
    many passes the ascent may spend, up to the page's cap."""

    points: list[ClassedPoint] = Field(min_length=N_CLASSES, max_length=MAX_POINTS)
    route: Route
    max_epochs: int = Field(default=MAX_EPOCHS, ge=1, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class ClassCoefficients(BaseModel):
    """One class's score, as an intercept and a weight per standardised column.

    Under softmax class 0 is the reference and every entry is zero; under
    one-vs-rest they are that class's own binary fit.
    """

    class_index: int
    intercept: float
    height: float
    weight: float


class ClassFitDocument(BaseModel):
    """One of the one-vs-rest route's binary fits, as the wrapper saw it.

    ``positive_rows`` is how many of the crowd were this fit's own class, the
    number that shows a balanced three-class crowd becoming three unbalanced
    binary ones.
    """

    class_index: int
    positive_rows: int
    epochs_run: int
    converged: bool


class ClassifyResponse(BaseModel):
    """Everything the widget draws and the page quotes, for one route.

    ``scores`` has one row per person in request order and one column per
    class, the probabilities under softmax and the three separate chances
    under one-vs-rest. ``linear_scores`` is the same table before the squash,
    with class 0's column held at zero under softmax because it is the
    reference. ``row_sums`` totals each row of ``scores``, which is one
    everywhere under softmax and whatever the three fits happened to say under
    one-vs-rest. ``epochs_run`` and ``converged`` describe the softmax ascent
    and are ``null`` on the other route, where ``class_fits`` carries the same
    facts for each binary fit instead.
    """

    route: Route
    predictions: list[int]
    scores: list[list[float]]
    linear_scores: list[list[float]]
    row_sums: list[float]
    accuracy: float
    epochs_run: int | None
    converged: bool | None
    class_fits: list[ClassFitDocument]
    regions: RegionGrid
    standardised_points: list[list[float]]
    class_coefficients: list[ClassCoefficients]
    score_lattice: list[list[list[float]]]
    correct_class_scores: list[float]
    log_loss: float
    coefficient_norm: float


class _FittedRoute:
    """What one route learned, gathered so the endpoint reads both the same way.

    A pairing rather than a tuple, for the reason the library gives. The
    softmax route has one walk to report and no per-class fits; the
    one-vs-rest route has three fits and no single walk, so each field is
    empty on the route it does not describe.
    """

    def __init__(
        self,
        model: MultinomialLogisticRegression | OneVsRestClassifier,
        linear_scores: np.ndarray,
        epochs_run: int | None,
        converged: bool | None,
        class_fits: list[ClassFitDocument],
        coefficients: list[ClassCoefficients],
    ) -> None:
        self.model = model
        self.linear_scores = linear_scores
        self.epochs_run = epochs_run
        self.converged = converged
        self.class_fits = class_fits
        self.coefficients = coefficients

    def scores_at(self, standardised: Sequence[Feature]) -> np.ndarray:
        """The linear score of every class at every row of standardised columns."""
        named = {feature.name: np.asarray(feature.values) for feature in standardised}
        return np.column_stack(
            [
                each.intercept
                + each.height * named["height"]
                + each.weight * named["weight"]
                for each in self.coefficients
            ]
        )


class _StandardisedClassifier:
    """A fitted classifier that standardises what it is asked about first.

    ``region_grid`` asks a model for its answer across a lattice in the raw
    units the points arrived in, and both routes were fitted on standardised
    columns, so this stands between the two. Plumbing rather than a model, and
    it exists so the shared lattice is reused rather than rewritten with a
    standardiser threaded through it.
    """

    def __init__(
        self,
        standardizer: Standardizer,
        model: MultinomialLogisticRegression | OneVsRestClassifier,
    ) -> None:
        self._standardizer = standardizer
        self._model = model

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        return self._model.predict(self._standardizer.transform(input_values))


def _softmax_route(
    standardised: Sequence[Feature], labels: Feature, max_epochs: int
) -> _FittedRoute:
    """Fit one score per class against the reference, and read the scores back.

    The scores before the squash are each class's intercept plus its weights
    applied to the standardised columns, the library's own learned numbers
    assembled the way its softmax reads them. Class 0's column is all zeros,
    since the library holds the reference class's score at zero and learns
    every other class against it.
    """
    model = MultinomialLogisticRegression(
        learning_rate=LEARNING_RATE, max_epochs=max_epochs, tolerance=TOLERANCE
    ).fit(standardised, labels)

    intercepts = model.intercepts
    columns = []
    documents = []
    for class_index in range(model.n_classes):
        coefficients = model.coefficients_for(class_index)
        column = np.full(labels.n_samples, float(intercepts[class_index]))
        for feature in standardised:
            column = column + coefficients[feature.name] * feature.values
        columns.append(column)
        documents.append(
            ClassCoefficients(
                class_index=class_index,
                intercept=float(intercepts[class_index]),
                height=float(coefficients["height"]),
                weight=float(coefficients["weight"]),
            )
        )

    return _FittedRoute(
        model=model,
        linear_scores=np.column_stack(columns),
        epochs_run=int(model.epochs_run),
        converged=bool(model.converged),
        class_fits=[],
        coefficients=documents,
    )


def _one_vs_rest_route(
    standardised: Sequence[Feature], labels: Feature, max_epochs: int
) -> _FittedRoute:
    """Fit one binary curve per class, and read each curve's own log-odds back.

    Each class's column is that class's binary model's intercept plus its
    weights applied to the standardised columns, the log-odds its sigmoid
    turns into the chance the wrapper reports. Nothing ties the three columns
    together, which is the whole of the route.
    """
    model = OneVsRestClassifier(
        binary_model=LogisticRegression(
            learning_rate=LEARNING_RATE, max_epochs=max_epochs, tolerance=TOLERANCE
        )
    ).fit(standardised, labels)

    columns = []
    class_fits = []
    documents = []
    for class_index in range(model.n_classes):
        binary_model = model.model_for(class_index)
        column = np.full(labels.n_samples, float(binary_model.intercept))
        for feature in standardised:
            column = column + binary_model.coefficients[feature.name] * feature.values
        columns.append(column)
        class_fits.append(
            ClassFitDocument(
                class_index=class_index,
                positive_rows=int(np.sum(labels.values == float(class_index))),
                epochs_run=int(binary_model.epochs_run),
                converged=bool(binary_model.converged),
            )
        )
        documents.append(
            ClassCoefficients(
                class_index=class_index,
                intercept=float(binary_model.intercept),
                height=float(binary_model.coefficients["height"]),
                weight=float(binary_model.coefficients["weight"]),
            )
        )

    return _FittedRoute(
        model=model,
        linear_scores=np.column_stack(columns),
        epochs_run=None,
        converged=None,
        class_fits=class_fits,
        coefficients=documents,
    )


def _log_loss(route: Route, scores: np.ndarray, labels: np.ndarray) -> float:
    """The loss each route was actually fitted to, on the crowd it was fitted to.

    Softmax minimises the mean of minus the log of the probability it gave
    the correct class. One-vs-rest fits three binary losses, one per class,
    so its figure is the mean binary log loss across the three fits, which is
    a different objective and is reported as one rather than dressed up as
    the other.
    """
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    rows = np.arange(labels.size)
    if route == "softmax":
        return float(-np.mean(np.log(clipped[rows, labels.astype(int)])))
    indicator = np.zeros_like(clipped)
    indicator[rows, labels.astype(int)] = 1.0
    return float(
        -np.mean(indicator * np.log(clipped) + (1 - indicator) * np.log(1 - clipped))
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Standardise the crowd, fit the chosen route, and score every person.

    A crowd missing one of the three classes is refused here in words, because
    the library would happily fit the two classes that remain and hand back a
    table two columns wide, which is a different page. A constant column is
    refused by the library with its own typed error, which the shared handler
    turns into a 422.
    """
    present = {point.label for point in request.points}
    missing = [name for index, name in enumerate(CLASS_NAMES) if index not in present]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"there is no {' or '.join(missing)} in the crowd, and a choice "
                "among three classes needs at least one person in each"
            ),
        )

    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])
    labels = Feature("age_group", [float(point.label) for point in request.points])

    standardizer = Standardizer().fit([heights, weights])
    standardised = standardizer.transform([heights, weights])

    if request.route == "softmax":
        fitted = _softmax_route(standardised, labels, request.max_epochs)
    else:
        fitted = _one_vs_rest_route(standardised, labels, request.max_epochs)

    scores = np.asarray(
        fitted.model.predict_probabilities(standardised), dtype=np.float64
    )
    predictions = np.asarray(fitted.model.predict(standardised), dtype=np.float64)
    regions = region_grid(
        _StandardisedClassifier(standardizer, fitted.model), request.points
    )

    # The same lattice the regions were drawn on, scored class by class, so
    # the page can draw the three score planes the regions are the top of.
    lattice_x, lattice_y = np.meshgrid(
        np.linspace(regions.x_min, regions.x_max, GRID_CELLS),
        np.linspace(regions.y_min, regions.y_max, GRID_CELLS),
    )
    lattice_scores = fitted.scores_at(
        standardizer.transform(
            [
                Feature("height", lattice_x.ravel()),
                Feature("weight", lattice_y.ravel()),
            ]
        )
    )
    label_indices = labels.values.astype(int)

    return ClassifyResponse(
        route=request.route,
        predictions=[int(value) for value in predictions],
        scores=[[float(value) for value in row] for row in scores],
        linear_scores=[[float(value) for value in row] for row in fitted.linear_scores],
        row_sums=[float(value) for value in scores.sum(axis=1)],
        accuracy=float(fitted.model.score(standardised, labels)),
        epochs_run=fitted.epochs_run,
        converged=fitted.converged,
        class_fits=fitted.class_fits,
        regions=regions,
        standardised_points=[
            [float(height), float(weight)]
            for height, weight in zip(
                standardised[0].values, standardised[1].values, strict=True
            )
        ],
        class_coefficients=fitted.coefficients,
        score_lattice=[
            [
                [float(value) for value in row]
                for row in lattice_scores[:, class_index].reshape(
                    GRID_CELLS, GRID_CELLS
                )
            ]
            for class_index in range(N_CLASSES)
        ],
        correct_class_scores=[
            float(scores[row, label]) for row, label in enumerate(label_indices)
        ],
        log_loss=_log_loss(request.route, scores, labels.values),
        coefficient_norm=float(
            np.sqrt(
                sum(each.height**2 + each.weight**2 for each in fitted.coefficients)
            )
        ),
    )


class SoftmaxWalkRequest(BaseModel):
    """The crowd, and how many passes of the softmax ascent to record."""

    points: list[ClassedPoint] = Field(min_length=N_CLASSES, max_length=MAX_POINTS)
    max_epochs: int = Field(default=MAX_EPOCHS, ge=1, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class SoftmaxPass(BaseModel):
    """Where the joint ascent stood after one pass.

    ``correct_class_probabilities`` is, for each person in request order, the
    probability the model gave their own class at that pass.
    """

    pass_number: int
    coefficient_norm: float
    log_loss: float
    accuracy: float
    correct_class_probabilities: list[float]


class SoftmaxWalkResponse(BaseModel):
    """Every pass of the softmax ascent, and the library's verdict on the budget."""

    passes: list[SoftmaxPass]
    converged: bool
    passes_run: int


@router.post("/walk", response_model=SoftmaxWalkResponse)
def walk(request: SoftmaxWalkRequest) -> SoftmaxWalkResponse:
    """Record the softmax ascent pass by pass on the standardised crowd.

    The library's observed route keeps the weights as a matrix, one row per
    class above the reference, and each recorded pass is read back into
    scores, probabilities and a loss here so the page can show accuracy
    reaching one while the coefficients keep growing.
    """
    present = {point.label for point in request.points}
    if len(present) < N_CLASSES:
        raise HTTPException(
            status_code=422,
            detail="a choice among three classes needs at least one person in each",
        )

    heights = Feature("height", [point.x for point in request.points])
    weights = Feature("weight", [point.y for point in request.points])
    labels = Feature("age_group", [float(point.label) for point in request.points])
    standardised = Standardizer().fit([heights, weights]).transform([heights, weights])

    model = MultinomialLogisticRegression(
        learning_rate=LEARNING_RATE, max_epochs=request.max_epochs, tolerance=TOLERANCE
    ).fit(standardised, labels)

    values = np.column_stack(
        [np.ones(labels.n_samples), standardised[0].values, standardised[1].values]
    )
    design = DesignMatrix(np.asfortranarray(values), ["height", "weight"], True)
    path = model.solver_path(design, labels.column)

    label_indices = labels.values.astype(int)
    rows = np.arange(labels.n_samples)
    passes: list[SoftmaxPass] = []
    for step in path:
        above_reference = np.asarray(step.weights_after, dtype=np.float64)
        scores = np.column_stack(
            [np.zeros(labels.n_samples), values @ above_reference.T]
        )
        probabilities = stable_softmax(scores)
        correct = probabilities[rows, label_indices]
        passes.append(
            SoftmaxPass(
                pass_number=step.pass_number,
                coefficient_norm=float(np.sqrt(np.sum(above_reference[:, 1:] ** 2))),
                log_loss=float(-np.mean(np.log(np.clip(correct, 1e-15, 1.0)))),
                accuracy=float(
                    np.mean(np.argmax(probabilities, axis=1) == label_indices)
                ),
                correct_class_probabilities=[float(value) for value in correct],
            )
        )

    return SoftmaxWalkResponse(
        passes=passes, converged=bool(path.converged), passes_run=int(path.passes_run)
    )
