"""The kernel trick and the support vector classifier, on the clinic's vitals.

The page's claim is structural. The library's ``SupportVectorClassifier``
reads its data only through inner products, so swapping its kernel changes
the space the straight boundary is drawn in without changing the machinery,
and the endpoints here exist to let a reader check each piece of that claim
against a measurement rather than take it on trust. Every fit standardizes
the two vitals first, since a kernel built on products and distances inherits
the units of its data, the same trap the k-nearest neighbours page worked.

What each endpoint is for
-------------------------
``classify`` is the playground: one fit under a chosen kernel and capacity,
with the shaded regions, the support vectors and the solver's step count.
``lift`` carries the clinic through the worked degree-2 map by hand, fits the
same classifier on the three lifted columns under the plain inner product,
and fits it again on the raw columns under the squared kernel, so the page
can report how far apart the two routes land. ``identity`` is the 121 check.
``gram`` is the machine's whole view of the data, the matrix of every pair's
kernel value, with its eigenvalues and whether the library's Cholesky route
accepts it, which is how Mercer's condition is enforced in this library.
``margin`` draws the corridor under the linear kernel, where the weight
vector can be recovered and the margin's width measured. ``gamma-sweep`` and
``capacity-sweep`` are the two hyperparameters, each held against a second
clinic the fit never saw. ``support-only`` refits on the support vectors alone
and on the clinic with one row dropped, which is the measurement behind the
claim that the support vectors are the model. ``intercept-cost`` is the
library's own four-row fixture for the absorbed intercept.

The solver, and why its settings are not the library's defaults
----------------------------------------------------------------
The library solves the dual by projected gradient ascent, the plainest
correct method. At its default step of 0.001 and ceiling of 1000 epochs,
measured on the clinic, every fit stops at the ceiling with every multiplier
still moving, so all 24 rows report as support vectors under every kernel,
which would make the page's support vector story false. The step here is
``min(0.02, 0.5 / n_rows)``, since the Gram matrix's largest eigenvalue grows
with the row count and the step has to shrink with it, and the ceiling is
50,000. Under those settings the clinic converges in 226 steps under the
linear kernel and 5,768 under the radial one, and the response carries the
count so the page can say so.

Every endpoint is a plain ``def`` so FastAPI runs it in the worker threadpool
off the event loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter
from oop_ml import (
    Feature,
    Kernel,
    KernelRidgeRegression,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    RowBlock,
    SigmoidKernel,
    Standardizer,
    SupportVectorClassifier,
)
from oop_ml.core.exceptions import MLLibError
from pydantic import BaseModel, Field

from oop_ml_api.concepts.two_feature_classification import (
    GRID_CELLS,
    LabelledPoint,
    RegionGrid,
)
from oop_ml_api.limits import MAX_MAGNITUDE, MAX_POINTS, MIN_POINTS

router = APIRouter(prefix="/concepts/kernel-trick", tags=["kernel-trick"])

TEMPERATURE = "temperature"
HEART_RATE = "heart_rate"
TARGET = "healthy"

MAX_DEGREE = 6
"""The highest power the polynomial kernel may raise to."""

MAX_GAMMA = 1000.0
"""How sharply the radial and sigmoid kernels may be tuned."""

MAX_CONSTANT = 10.0
"""How far a kernel's constant may be shifted either way."""

MAX_CAPACITY = 1000.0
"""The largest price of a margin violation the page may ask for."""

MAX_EPOCHS = 50_000
"""The ceiling on ascent steps, high enough that the clinic converges under it."""

LARGEST_STEP = 0.02
"""The ascent step on a small clinic; it shrinks with the row count."""

STEP_SCALE = 0.5
"""``step = min(LARGEST_STEP, STEP_SCALE / n_rows)``."""

HELD_OUT_SEED = 11
"""The second clinic, the one the sweeps are scored on, is drawn from this seed."""

HELD_OUT_HEALTHY = 9
HELD_OUT_UNWELL = 12

GAMMAS = (0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0, 1000.0)
"""The reaches the radial kernel is swept through."""

CAPACITIES = (0.01, 0.1, 1.0, 10.0, 100.0)
"""The prices the capacity sweep runs through."""

INTERCEPT_CAPACITIES = (0.1, 1.0, 10.0, 100.0)
"""The capacities the four-row intercept fixture is fitted at."""

LATTICE_REACH = 2.5
"""The standardized lattice the route-agreement checks are read on runs to this."""

LATTICE_CELLS = 26


class LinearKernelChoice(BaseModel):
    """The plain inner product, which leaves the boundary straight."""

    name: Literal["linear"]

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return LinearKernel()


class PolynomialKernelChoice(BaseModel):
    """The inner product plus a constant, raised to a power.

    A constant of one, the library's default, lets every lower power ride
    along with the highest; a constant of zero is the pure power the page
    works by hand.
    """

    name: Literal["polynomial"]
    degree: int = Field(default=2, ge=1, le=MAX_DEGREE)
    constant: float = Field(default=1.0, ge=0.0, le=MAX_CONSTANT)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return PolynomialKernel(degree=self.degree, constant=self.constant)


class RadialKernelChoice(BaseModel):
    """A bump that falls with distance, with gamma setting how fast."""

    name: Literal["rbf"]
    gamma: float = Field(default=1.0, gt=0.0, le=MAX_GAMMA)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return RadialBasisKernel(gamma=self.gamma)


class SigmoidKernelChoice(BaseModel):
    """A hyperbolic tangent of the shifted inner product, not always a kernel.

    Negative constants are allowed on purpose. They are where the function
    stops being a kernel, and what the library does about that is part of
    what the page shows.
    """

    name: Literal["sigmoid"]
    gamma: float = Field(default=1.0, gt=0.0, le=MAX_GAMMA)
    constant: float = Field(default=0.0, ge=-MAX_CONSTANT, le=MAX_CONSTANT)

    model_config = {"extra": "forbid"}

    def kernel(self) -> Kernel:
        return SigmoidKernel(gamma=self.gamma, constant=self.constant)


KernelChoice = Annotated[
    LinearKernelChoice
    | PolynomialKernelChoice
    | RadialKernelChoice
    | SigmoidKernelChoice,
    Field(discriminator="name"),
]
"""Which reading of similarity to hand the machine, with exactly its own parameters."""


class StandardizedClinic:
    """The clinic's two vitals standardized once, with the labels beside them.

    ``standardize=False`` leaves the vitals in their raw units, which exists so
    the page can show what a kernel makes of a heart rate that ranges over a
    hundred beats beside a temperature that ranges over five degrees.
    """

    def __init__(self, points: list[LabelledPoint], standardize: bool = True) -> None:
        temperature = Feature(TEMPERATURE, [point.x for point in points])
        heart_rate = Feature(HEART_RATE, [point.y for point in points])
        self.standardizer = (
            Standardizer().fit([temperature, heart_rate]) if standardize else None
        )
        self.inputs = self.transform(points)
        self.labels = Feature(TARGET, [float(point.label) for point in points])
        self.points = points

    @property
    def n_rows(self) -> int:
        return len(self.points)

    @property
    def rows(self) -> np.ndarray:
        return np.column_stack([np.asarray(feature.values) for feature in self.inputs])

    def transform(self, points: list[LabelledPoint]) -> list[Feature]:
        """Other patients through this clinic's scaling, never their own."""
        return self._scaled(
            Feature(TEMPERATURE, [point.x for point in points]),
            Feature(HEART_RATE, [point.y for point in points]),
        )

    def _scaled(self, temperature: Feature, heart_rate: Feature) -> list[Feature]:
        if self.standardizer is None:
            return [temperature, heart_rate]
        return self.standardizer.transform([temperature, heart_rate])

    def lattice(self) -> tuple[list[Feature], tuple[float, float, float, float]]:
        """A lattice over the raw plane, in this clinic's scaling, and its bounds."""
        x_values = [point.x for point in self.points]
        y_values = [point.y for point in self.points]
        x_pad = 0.05 * (max(x_values) - min(x_values) or 1.0)
        y_pad = 0.05 * (max(y_values) - min(y_values) or 1.0)
        x_min, x_max = min(x_values) - x_pad, max(x_values) + x_pad
        y_min, y_max = min(y_values) - y_pad, max(y_values) + y_pad
        lattice_x, lattice_y = np.meshgrid(
            np.linspace(x_min, x_max, GRID_CELLS),
            np.linspace(y_min, y_max, GRID_CELLS),
        )
        features = self._scaled(
            Feature(TEMPERATURE, lattice_x.ravel()),
            Feature(HEART_RATE, lattice_y.ravel()),
        )
        return features, (float(x_min), float(x_max), float(y_min), float(y_max))


def learning_rate_for(n_rows: int) -> float:
    """The ascent step, shrinking with the row count."""
    return min(LARGEST_STEP, STEP_SCALE / n_rows)


def classifier(
    kernel: Kernel,
    n_rows: int,
    capacity: float,
    step: float | None = None,
    ceiling: int | None = None,
) -> SupportVectorClassifier:
    """The library's classifier at the settings the module docstring defends.

    ``step`` and ``ceiling`` override them, which is how the page shows what
    the library's own defaults do on the clinic.
    """
    return SupportVectorClassifier(
        kernel=kernel,
        capacity=capacity,
        learning_rate=learning_rate_for(n_rows) if step is None else step,
        max_epochs=MAX_EPOCHS if ceiling is None else ceiling,
    )


def regions_of(
    model: SupportVectorClassifier, clinic: StandardizedClinic
) -> RegionGrid:
    """The fitted boundary's class at every cell, through the same scaling."""
    features, (x_min, x_max, y_min, y_max) = clinic.lattice()
    predictions = np.asarray(model.predict(features), dtype=np.float64).reshape(
        GRID_CELLS, GRID_CELLS
    )
    return RegionGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cells=GRID_CELLS,
        labels=[[int(value) for value in row] for row in predictions],
    )


class FitSummary(BaseModel):
    """What one fit of the classifier reports about itself.

    ``support_positions`` are the training rows the boundary depends on, in
    the order the points arrived; ``at_the_cap`` counts those whose
    multiplier reached the capacity, which are the rows the soft margin gave
    up on. ``converged`` is whether the ascent stopped before its ceiling.
    ``decision_values`` are signed distances in the implied space's units and
    ``squashed`` is each one through a logistic, which the page is careful
    to say is not a probability.
    """

    accuracy: float
    n_support_vectors: int
    support_share: float
    support_positions: list[int]
    at_the_cap: int
    epochs_run: int
    converged: bool
    decision_values: list[float]
    squashed: list[float]


def summarise(
    model: SupportVectorClassifier, inputs: list[Feature], labels: Feature
) -> FitSummary:
    support = model.support_vectors
    return FitSummary(
        accuracy=float(model.score(inputs, labels)),
        n_support_vectors=support.n_vectors,
        support_share=float(support.share_of_training_rows),
        support_positions=support.positions(),
        at_the_cap=support.n_at_the_cap(model.capacity),
        epochs_run=model.epochs_run,
        converged=model.epochs_run < model.max_epochs,
        decision_values=[float(value) for value in model.decision_values(inputs)],
        squashed=[
            float(value) for value in np.asarray(model.predict_probability(inputs))
        ],
    )


class ClassifyRequest(BaseModel):
    """The patients, which reading of similarity to hand the machine, and the price.

    ``standardize`` off fits the raw vitals, which the page does once to show
    the units trap. ``step`` and ``ceiling`` override the solver settings the
    module docstring defends, which the page does once to show the library's
    own defaults stopping short.
    """

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice
    capacity: float = Field(default=1.0, gt=0.0, le=MAX_CAPACITY)
    standardize: bool = True
    step: float | None = Field(default=None, gt=0.0, le=LARGEST_STEP)
    ceiling: int | None = Field(default=None, ge=1, le=MAX_EPOCHS)

    model_config = {"extra": "forbid"}


class ClassifyResponse(BaseModel):
    """The fit's summary, its answer across the plane, and its held-out score.

    ``held_out_accuracy`` is the fit scored on the second clinic, the one the
    sweeps use, through the same scaling the fit was given.
    """

    fit: FitSummary
    regions: RegionGrid
    kernel_description: str
    held_out_accuracy: float


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Standardize the vitals, fit under the chosen kernel, shade the plane."""
    clinic = StandardizedClinic(request.points, standardize=request.standardize)
    kernel = request.kernel.kernel()
    model = classifier(
        kernel, clinic.n_rows, request.capacity, request.step, request.ceiling
    ).fit(clinic.inputs, clinic.labels)
    held = held_out_clinic()
    held_labels = Feature(TARGET, [float(point.label) for point in held])
    return ClassifyResponse(
        fit=summarise(model, clinic.inputs, clinic.labels),
        regions=regions_of(model, clinic),
        kernel_description=kernel.description,
        held_out_accuracy=float(model.score(clinic.transform(held), held_labels)),
    )


class PointsRequest(BaseModel):
    """The patients, and nothing else."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)

    model_config = {"extra": "forbid"}


class LiftedPoint(BaseModel):
    """One patient carried through the worked mapping, with their label."""

    u: float
    v: float
    w: float
    label: int


class SeparatingPlane(BaseModel):
    """The flat boundary found in the lifted space, a·u + b·v + c·w = d."""

    a: float
    b: float
    c: float
    d: float


class LiftResponse(BaseModel):
    """The clinic lifted into three dimensions, and the two routes compared.

    ``plane`` is the boundary the classifier drew on the three lifted
    columns under the plain inner product, with its normal scaled to unit
    length so it draws the same at any steepness; ``lifted`` is that fit's
    summary. ``kernel`` is the same classifier on the two raw columns under
    the pure squared kernel. ``multiplier_gap`` and ``decision_gap`` are the
    largest differences between the two routes, over the training rows and
    over a lattice of the standardized plane, and the page's claim is that
    both are rounding.
    """

    points: list[LiftedPoint]
    plane: SeparatingPlane
    lifted: FitSummary
    kernel: FitSummary
    multiplier_gap: float
    decision_gap: float


def lifted_columns(rows: np.ndarray) -> np.ndarray:
    """The worked degree-2 map, squares and a root-two cross term."""
    return np.column_stack(
        [rows[:, 0] ** 2, math.sqrt(2.0) * rows[:, 0] * rows[:, 1], rows[:, 1] ** 2]
    )


def lifted_features(rows: np.ndarray) -> list[Feature]:
    lifted = lifted_columns(rows)
    return [
        Feature("u", lifted[:, 0]),
        Feature("v", lifted[:, 1]),
        Feature("w", lifted[:, 2]),
    ]


def standardized_lattice() -> np.ndarray:
    axis = np.linspace(-LATTICE_REACH, LATTICE_REACH, LATTICE_CELLS)
    lattice_x, lattice_y = np.meshgrid(axis, axis)
    return np.column_stack([lattice_x.ravel(), lattice_y.ravel()])


@router.post("/lift", response_model=LiftResponse)
def lift(request: PointsRequest) -> LiftResponse:
    """Carry the patients through the worked map and fit both routes."""
    clinic = StandardizedClinic(request.points)
    rows = clinic.rows
    lifted = lifted_features(rows)

    lifted_model = classifier(LinearKernel(), clinic.n_rows, 1.0).fit(
        lifted, clinic.labels
    )
    kernel_model = classifier(
        PolynomialKernel(degree=2, constant=0.0), clinic.n_rows, 1.0
    ).fit(clinic.inputs, clinic.labels)

    coefficient = lifted_model.multipliers * lifted_model.signed_labels
    normal = lifted_columns(rows).T @ coefficient
    offset = float(coefficient.sum())
    length = float(np.linalg.norm(normal)) or 1.0

    lattice = standardized_lattice()
    lifted_decisions = lifted_model.decision_values(lifted_features(lattice))
    kernel_decisions = kernel_model.decision_values(
        [Feature(TEMPERATURE, lattice[:, 0]), Feature(HEART_RATE, lattice[:, 1])]
    )

    lifted_values = lifted_columns(rows)
    return LiftResponse(
        points=[
            LiftedPoint(u=float(u), v=float(v), w=float(w), label=point.label)
            for (u, v, w), point in zip(lifted_values, request.points, strict=True)
        ],
        plane=SeparatingPlane(
            a=float(normal[0] / length),
            b=float(normal[1] / length),
            c=float(normal[2] / length),
            d=float(-offset / length),
        ),
        lifted=summarise(lifted_model, lifted, clinic.labels),
        kernel=summarise(kernel_model, clinic.inputs, clinic.labels),
        multiplier_gap=float(
            np.max(np.abs(lifted_model.multipliers - kernel_model.multipliers))
        ),
        decision_gap=float(np.max(np.abs(lifted_decisions - kernel_decisions))),
    )


Coordinate = Annotated[
    float, Field(ge=-MAX_MAGNITUDE, le=MAX_MAGNITUDE, allow_inf_nan=False)
]
"""One finite, bounded number, so a vector cannot smuggle a NaN past the kernel."""


class IdentityRequest(BaseModel):
    """Two small vectors to check the squared kernel against its lift on."""

    a: list[Coordinate] = Field(min_length=2, max_length=2)
    b: list[Coordinate] = Field(min_length=2, max_length=2)

    model_config = {"extra": "forbid"}


class ColumnCount(BaseModel):
    """How many columns a polynomial lift of some degree over some features builds.

    ``with_lower_terms`` counts every product of up to ``degree`` features,
    which is the constant-one kernel's space; ``pure`` counts the products of
    exactly ``degree``, which is the constant-zero kernel's.
    """

    n_features: int
    degree: int
    with_lower_terms: int
    pure: int


COLUMN_COUNT_CASES = ((2, 2), (2, 3), (10, 3), (100, 2), (20, 5), (20, 10))


class IdentityResponse(BaseModel):
    """Both routes to one number, and what the long route costs at scale."""

    dot_product: float
    phi_a: list[float]
    phi_b: list[float]
    lifted_product: float
    kernel_value: float
    column_counts: list[ColumnCount]


@router.post("/identity", response_model=IdentityResponse)
def identity(request: IdentityRequest) -> IdentityResponse:
    """The kernel's way and the lifted way, both through the library's kernels."""
    left = np.asarray([request.a], dtype=np.float64)
    right = np.asarray([request.b], dtype=np.float64)
    raw_names = ["x", "y"]
    lifted_names = ["u", "v", "w"]

    dot = LinearKernel().between(RowBlock(left, raw_names), RowBlock(right, raw_names))
    lifted = LinearKernel().between(
        RowBlock(lifted_columns(left), lifted_names),
        RowBlock(lifted_columns(right), lifted_names),
    )
    squared = PolynomialKernel(degree=2, constant=0.0).between(
        RowBlock(left, raw_names), RowBlock(right, raw_names)
    )
    return IdentityResponse(
        dot_product=float(dot.values[0, 0]),
        phi_a=[float(value) for value in lifted_columns(left)[0]],
        phi_b=[float(value) for value in lifted_columns(right)[0]],
        lifted_product=float(lifted.values[0, 0]),
        kernel_value=float(squared.values[0, 0]),
        column_counts=[
            ColumnCount(
                n_features=n_features,
                degree=degree,
                with_lower_terms=math.comb(n_features + degree, degree),
                pure=math.comb(n_features + degree - 1, degree),
            )
            for n_features, degree in COLUMN_COUNT_CASES
        ],
    )


class GramRequest(BaseModel):
    """The patients and the kernel whose whole view of them is wanted."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice

    model_config = {"extra": "forbid"}


class GramResponse(BaseModel):
    """Every pair's kernel value, and whether the table came from a kernel.

    ``eigenvalues`` are sorted largest first. ``symmetric`` is the library's
    own check. ``ridge_accepts`` is whether the library's kernel ridge, whose
    Cholesky solve fails exactly when the matrix is not positive definite,
    took the matrix, and ``ridge_refusal`` is its message when it did not.
    ``classifier`` is the support vector classifier run on the same matrix,
    which the library does not refuse, and the page says why that is a
    weaker guarantee.
    """

    kernel_description: str
    values: list[list[float]]
    eigenvalues: list[float]
    smallest_eigenvalue: float
    largest_eigenvalue: float
    n_negative_eigenvalues: int
    symmetric: bool
    ridge_accepts: bool
    ridge_refusal: str | None
    classifier: FitSummary


EIGENVALUE_TOLERANCE = 1e-09
"""Below minus this an eigenvalue is negative rather than a rounded zero."""


@router.post("/gram", response_model=GramResponse)
def gram(request: GramRequest) -> GramResponse:
    """Build the training Gram matrix and test it two ways."""
    clinic = StandardizedClinic(request.points)
    kernel = request.kernel.kernel()
    block = RowBlock(clinic.rows, [TEMPERATURE, HEART_RATE])
    matrix = kernel.between(block, block)
    values = matrix.values
    eigenvalues = np.sort(np.linalg.eigvalsh(values))[::-1]

    try:
        matrix.check_symmetric()
        symmetric = True
    except MLLibError:
        symmetric = False

    ridge_refusal: str | None = None
    try:
        KernelRidgeRegression(kernel=kernel, penalty=1e-06).fit(
            clinic.inputs,
            Feature("heart_rate_target", [point.y for point in request.points]),
        )
    except MLLibError as error:
        ridge_refusal = str(error)

    model = classifier(kernel, clinic.n_rows, 1.0).fit(clinic.inputs, clinic.labels)

    return GramResponse(
        kernel_description=kernel.description,
        values=[[float(value) for value in row] for row in values],
        eigenvalues=[float(value) for value in eigenvalues],
        smallest_eigenvalue=float(eigenvalues[-1]),
        largest_eigenvalue=float(eigenvalues[0]),
        n_negative_eigenvalues=int(np.sum(eigenvalues < -EIGENVALUE_TOLERANCE)),
        symmetric=symmetric,
        ridge_accepts=ridge_refusal is None,
        ridge_refusal=ridge_refusal,
        classifier=summarise(model, clinic.inputs, clinic.labels),
    )


class MarginRequest(BaseModel):
    """The patients and the price, under the linear kernel."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    capacity: float = Field(default=1.0, gt=0.0, le=MAX_CAPACITY)

    model_config = {"extra": "forbid"}


class Segment(BaseModel):
    """A straight line across the raw plane, as two endpoints."""

    x1: float
    y1: float
    x2: float
    y2: float


class MarginResponse(BaseModel):
    """The corridor under the linear kernel, where the weights can be read.

    ``weights`` and ``offset`` are in standardized units, recovered from the
    multipliers as the dual's stationarity condition says they can be;
    ``width`` is ``2 / ||w||``. The three segments are the boundary and the
    two margin edges in raw temperature and heart rate, and the summary
    carries the support vectors and each patient's decision value.
    """

    fit: FitSummary
    weights: list[float]
    offset: float
    width: float
    boundary: Segment
    upper_edge: Segment
    lower_edge: Segment


def _margin_segment(
    clinic: StandardizedClinic, weights: np.ndarray, offset: float, level: float
) -> Segment:
    """Where ``w · z + b = level`` crosses the raw plane's temperature range."""
    assert clinic.standardizer is not None
    scalings = {scaling.name: scaling for scaling in clinic.standardizer.scalings}
    temperature = scalings[TEMPERATURE]
    heart_rate = scalings[HEART_RATE]
    x_values = [point.x for point in clinic.points]
    span = (max(x_values) - min(x_values)) or 1.0
    ends: list[tuple[float, float]] = []
    for raw_x in (min(x_values) - 0.5 * span, max(x_values) + 0.5 * span):
        z_x = (raw_x - temperature.mean) / temperature.standard_deviation
        if abs(weights[1]) > 1e-12:
            z_y = (level - offset - weights[0] * z_x) / weights[1]
        else:
            z_y = 0.0
        ends.append((raw_x, z_y * heart_rate.standard_deviation + heart_rate.mean))
    (x1, y1), (x2, y2) = ends
    return Segment(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


@router.post("/margin", response_model=MarginResponse)
def margin(request: MarginRequest) -> MarginResponse:
    """Fit the maximum-margin line and read its corridor back."""
    clinic = StandardizedClinic(request.points)
    model = classifier(LinearKernel(), clinic.n_rows, request.capacity).fit(
        clinic.inputs, clinic.labels
    )
    coefficient = model.multipliers * model.signed_labels
    weights = clinic.rows.T @ coefficient
    offset = float(coefficient.sum())
    width = float(2.0 / (np.linalg.norm(weights) or 1.0))

    return MarginResponse(
        fit=summarise(model, clinic.inputs, clinic.labels),
        weights=[float(value) for value in weights],
        offset=offset,
        width=width,
        boundary=_margin_segment(clinic, weights, offset, 0.0),
        upper_edge=_margin_segment(clinic, weights, offset, 1.0),
        lower_edge=_margin_segment(clinic, weights, offset, -1.0),
    )


def held_out_clinic() -> list[LabelledPoint]:
    """A second clinic from the same distribution as the playground's random one."""
    generator = np.random.default_rng(HELD_OUT_SEED)
    healthy = [
        LabelledPoint(
            x=round(36.2 + generator.random() * 1.4, 1),
            y=round(56 + generator.random() * 38),
            label=1,
        )
        for _ in range(HELD_OUT_HEALTHY)
    ]
    unwell = []
    for _ in range(HELD_OUT_UNWELL):
        angle = generator.random() * 2 * math.pi
        reach = 0.55 + generator.random() * 0.45
        unwell.append(
            LabelledPoint(
                x=round(37.7 + math.cos(angle) * 2.6 * reach, 1),
                y=round(88 + math.sin(angle) * 46 * reach),
                label=0,
            )
        )
    return healthy + unwell


class SweepEntry(BaseModel):
    """One setting of a hyperparameter, scored on the fit and on the second clinic."""

    value: float
    fit: FitSummary
    held_out_accuracy: float
    regions: RegionGrid


class SweepResponse(BaseModel):
    """A hyperparameter swept through fixed values, each fit reported whole."""

    held_out: list[LabelledPoint]
    entries: list[SweepEntry]


def _sweep(
    clinic: StandardizedClinic,
    settings: tuple[float, ...],
    make_model: Callable[[float], SupportVectorClassifier],
) -> SweepResponse:
    held = held_out_clinic()
    held_inputs = clinic.transform(held)
    held_labels = Feature(TARGET, [float(point.label) for point in held])
    entries = []
    for value in settings:
        model = make_model(value).fit(clinic.inputs, clinic.labels)
        entries.append(
            SweepEntry(
                value=value,
                fit=summarise(model, clinic.inputs, clinic.labels),
                held_out_accuracy=float(model.score(held_inputs, held_labels)),
                regions=regions_of(model, clinic),
            )
        )
    return SweepResponse(held_out=held, entries=entries)


@router.post("/gamma-sweep", response_model=SweepResponse)
def gamma_sweep(request: PointsRequest) -> SweepResponse:
    """The radial kernel's reach, from too wide to memorising."""
    clinic = StandardizedClinic(request.points)
    return _sweep(
        clinic,
        GAMMAS,
        lambda gamma: classifier(RadialBasisKernel(gamma=gamma), clinic.n_rows, 1.0),
    )


class CapacitySweepRequest(BaseModel):
    """The patients and which kernel to sweep the price under."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice

    model_config = {"extra": "forbid"}


@router.post("/capacity-sweep", response_model=SweepResponse)
def capacity_sweep(request: CapacitySweepRequest) -> SweepResponse:
    """The price of a violation, from every row capped to none."""
    clinic = StandardizedClinic(request.points)
    kernel = request.kernel.kernel()
    return _sweep(
        clinic,
        CAPACITIES,
        lambda capacity: classifier(kernel, clinic.n_rows, capacity),
    )


class SupportOnlyRequest(BaseModel):
    """The patients and the kernel, for the refits that test what the model is."""

    points: list[LabelledPoint] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    kernel: KernelChoice

    model_config = {"extra": "forbid"}


class RefitResult(BaseModel):
    """A refit on some subset of the rows, against the full fit's boundary.

    ``gap`` is the largest difference in decision value over a lattice of
    the standardized plane between this refit and the fit on every row.
    """

    kept_positions: list[int]
    fit: FitSummary
    regions: RegionGrid
    gap: float


class SupportOnlyResponse(BaseModel):
    """The full fit, the refit on its support vectors alone, and two one-row drops."""

    full: FitSummary
    full_regions: RegionGrid
    support_only: RefitResult | None
    dropped_non_support: RefitResult | None
    dropped_support: RefitResult | None


def _refit(
    clinic: StandardizedClinic,
    kernel: Kernel,
    keep: list[int],
    reference: SupportVectorClassifier,
    lattice: np.ndarray,
) -> RefitResult:
    kept_points = [clinic.points[position] for position in keep]
    kept_inputs = clinic.transform(kept_points)
    kept_labels = Feature(TARGET, [float(point.label) for point in kept_points])
    model = classifier(kernel, clinic.n_rows, 1.0).fit(kept_inputs, kept_labels)
    lattice_features = [
        Feature(TEMPERATURE, lattice[:, 0]),
        Feature(HEART_RATE, lattice[:, 1]),
    ]
    gap = float(
        np.max(
            np.abs(
                model.decision_values(lattice_features)
                - reference.decision_values(lattice_features)
            )
        )
    )
    return RefitResult(
        kept_positions=keep,
        fit=summarise(model, kept_inputs, kept_labels),
        regions=regions_of(model, clinic),
        gap=gap,
    )


@router.post("/support-only", response_model=SupportOnlyResponse)
def support_only(request: SupportOnlyRequest) -> SupportOnlyResponse:
    """Refit on the support vectors alone, then drop one row of each kind."""
    clinic = StandardizedClinic(request.points)
    kernel = request.kernel.kernel()
    full = classifier(kernel, clinic.n_rows, 1.0).fit(clinic.inputs, clinic.labels)
    positions = full.support_vectors.positions()
    everyone = list(range(clinic.n_rows))
    non_support = [position for position in everyone if position not in positions]
    lattice = standardized_lattice()

    dropped_non_support = (
        _refit(
            clinic,
            kernel,
            [position for position in everyone if position != non_support[0]],
            full,
            lattice,
        )
        if non_support
        else None
    )
    return SupportOnlyResponse(
        full=summarise(full, clinic.inputs, clinic.labels),
        full_regions=regions_of(full, clinic),
        support_only=(
            _refit(clinic, kernel, positions, full, lattice) if positions else None
        ),
        dropped_non_support=dropped_non_support,
        dropped_support=(
            _refit(
                clinic,
                kernel,
                [position for position in everyone if position != positions[0]],
                full,
                lattice,
            )
            if positions
            else None
        ),
    )


class InterceptEntry(BaseModel):
    """The four-row fixture at one capacity: where the boundary crosses."""

    capacity: float
    crossing: float
    multipliers: list[float]
    epochs_run: int


class InterceptCostResponse(BaseModel):
    """The library's own fixture for the absorbed intercept, at four capacities.

    Four rows at 1, 2, 3 and 4, labelled 0, 0, 1, 1, so the widest corridor
    puts the boundary at 2.5. ``crossing`` is where the fitted decision value
    passes through zero, read off a fine grid.
    """

    inputs: list[float]
    labels: list[int]
    entries: list[InterceptEntry]


INTERCEPT_INPUTS = (1.0, 2.0, 3.0, 4.0)
INTERCEPT_LABELS = (0, 0, 1, 1)
INTERCEPT_PROBE = np.linspace(0.0, 5.0, 501)


@cache
def _intercept_cost() -> InterceptCostResponse:
    feature = Feature("x", list(INTERCEPT_INPUTS))
    target = Feature("y", [float(label) for label in INTERCEPT_LABELS])
    probe = Feature("x", INTERCEPT_PROBE)
    entries = []
    for capacity in INTERCEPT_CAPACITIES:
        model = classifier(LinearKernel(), len(INTERCEPT_INPUTS), capacity).fit(
            [feature], target
        )
        decisions = model.decision_values([probe])
        crossing = float(np.interp(0.0, decisions, INTERCEPT_PROBE))
        entries.append(
            InterceptEntry(
                capacity=capacity,
                crossing=crossing,
                multipliers=[float(value) for value in model.multipliers],
                epochs_run=model.epochs_run,
            )
        )
    return InterceptCostResponse(
        inputs=list(INTERCEPT_INPUTS),
        labels=list(INTERCEPT_LABELS),
        entries=entries,
    )


@router.get("/intercept-cost", response_model=InterceptCostResponse)
def intercept_cost() -> InterceptCostResponse:
    """Where a penalised offset puts the boundary, and how capacity releases it."""
    return _intercept_cost()


SIMILARITY_SAMPLES = 161
"""How many positions the similarity curves are sampled at."""

SIMILARITY_REFERENCE = 1.5
"""Where the reference point stands on the one-dimensional axis."""

SIMILARITY_SPAN = 4.0
"""The sweep runs from minus this to this."""

SIMILARITY_KERNELS = (
    LinearKernel(),
    PolynomialKernel(degree=2),
    RadialBasisKernel(gamma=0.5),
    RadialBasisKernel(gamma=2.0),
)
"""The kernels the chart compares, two widths of radial so gamma is visible."""


class SimilarityCurve(BaseModel):
    """One kernel's answer at every swept position, labelled by its own formula."""

    label: str
    values: list[float]


class SimilarityCurveResponse(BaseModel):
    """Each kernel's similarity to one fixed reference point, swept along a line.

    The labels are the library kernels' own descriptions, so the chart cannot
    drift from what the code actually computes.
    """

    positions: list[float]
    reference: float
    curves: list[SimilarityCurve]


@router.post("/similarity-curve", response_model=SimilarityCurveResponse)
def similarity_curve() -> SimilarityCurveResponse:
    """Sweep a point along a line and ask each kernel how alike the pair is.

    No request body, since the sweep is a fixed illustration rather than a
    fit, the same standing as the ensemble pages' formula curves.
    """
    positions = np.linspace(-SIMILARITY_SPAN, SIMILARITY_SPAN, SIMILARITY_SAMPLES)
    swept = RowBlock(positions.reshape(-1, 1), ["x"])
    reference = RowBlock(np.asarray([[SIMILARITY_REFERENCE]], dtype=np.float64), ["x"])

    curves = [
        SimilarityCurve(
            label=kernel.description,
            values=[
                float(value)
                for value in np.asarray(
                    kernel.between(swept, reference), dtype=np.float64
                ).ravel()
            ],
        )
        for kernel in SIMILARITY_KERNELS
    ]

    return SimilarityCurveResponse(
        positions=[float(value) for value in positions],
        reference=SIMILARITY_REFERENCE,
        curves=curves,
    )
