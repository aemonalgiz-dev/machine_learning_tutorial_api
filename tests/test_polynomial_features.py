"""The polynomial-features endpoints, pinned to the numbers the page quotes.

The page works a crowd of twenty-two people measured by height, waist girth and
weight, and a second group of twelve laid out as four builds at each of three
heights. Under one seed seven of the twenty-two are held back, leaving fifteen
to fit on, so degree fourteen is the interpolating degree and the sweep runs
right up to it. Every figure the page prints is asserted here, the five columns
a degree-two expansion of two features produces, the growth table, the sweep
where the training score climbs while the held-out score falls to minus
twenty-five thousand, the penalty that rescues the same degree, the kilograms of
weight a plane is obliged to attribute to girth at every height at once, the
change of unit that leaves least squares alone and moves a penalised fit from
0.9774 to 0.7036, and the stepped fit that beats none of the curves and never
falls as far.
"""

from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.polynomial_features import PLAIN_WORDS
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/polynomial-features"
EXPANSION = f"{BASE}/expansion"
COLUMN_COUNTS = f"{BASE}/column-counts"
CURVE_FIT = f"{BASE}/curve-fit"
DEGREE_GAP = f"{BASE}/degree-gap"
TWO_COLUMNS = f"{BASE}/two-columns"
UNITS = f"{BASE}/units"
AGAINST_A_TREE = f"{BASE}/against-a-tree"
DEGENERATE = f"{BASE}/degenerate-columns"

HEIGHTS = [
    104.0,
    111.0,
    118.0,
    124.0,
    131.0,
    137.0,
    143.0,
    149.0,
    154.0,
    158.0,
    162.0,
    166.0,
    169.0,
    172.0,
    175.0,
    178.0,
    180.0,
    183.0,
    185.0,
    187.0,
    189.0,
    191.0,
]
GIRTHS = [
    49.0,
    55.2,
    58.2,
    57.2,
    60.9,
    63.2,
    68.5,
    69.9,
    74.0,
    70.2,
    79.6,
    77.8,
    80.9,
    80.5,
    81.4,
    84.7,
    86.4,
    85.6,
    86.6,
    89.4,
    86.9,
    86.4,
]
WEIGHTS = [
    16.1,
    18.1,
    18.1,
    21.7,
    27.6,
    29.1,
    35.5,
    43.6,
    53.1,
    45.8,
    59.0,
    61.1,
    68.2,
    65.7,
    70.9,
    79.3,
    79.6,
    77.5,
    83.9,
    89.9,
    88.5,
    81.3,
]

CROWD = [{"x": height, "y": weight} for height, weight in zip(HEIGHTS, WEIGHTS)]
CROWD_PEOPLE = [
    {"height": height, "girth": girth, "weight": weight}
    for height, girth, weight in zip(HEIGHTS, GIRTHS, WEIGHTS)
]

BUILD_HEIGHTS = [150.0] * 4 + [165.0] * 4 + [180.0] * 4
BUILD_GIRTHS = [66.0, 76.0, 86.0, 96.0] * 3
BUILD_WEIGHTS = [
    38.5,
    51.6,
    67.5,
    83.0,
    41.6,
    56.9,
    72.3,
    90.9,
    45.5,
    62.2,
    79.6,
    100.3,
]
BUILDS = [
    {"height": height, "girth": girth, "weight": weight}
    for height, girth, weight in zip(BUILD_HEIGHTS, BUILD_GIRTHS, BUILD_WEIGHTS)
]

FOUR_PEOPLE = {
    "columns": [
        {"name": "height", "values": [150.0, 150.0, 180.0, 180.0]},
        {"name": "girth", "values": [66.0, 96.0, 66.0, 96.0]},
    ],
    "degree": 2,
}

HELD_OUT_POSITIONS = [4, 5, 7, 14, 16, 19, 21]


def posted(path: str, body: dict) -> dict:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.json()
    return response.json()


class TestTheExpansionItself:
    """Which columns come back, in which order, holding which numbers."""

    def test_two_features_at_degree_two_give_five_columns(self) -> None:
        answer = posted(EXPANSION, FOUR_PEOPLE)

        assert answer["n_terms"] == 5
        assert [term["name"] for term in answer["terms"]] == [
            "height",
            "girth",
            "height^2",
            "height*girth",
            "girth^2",
        ]

    def test_the_product_column_is_the_two_columns_multiplied(self) -> None:
        answer = posted(EXPANSION, FOUR_PEOPLE)
        product = next(
            term for term in answer["terms"] if term["name"] == "height*girth"
        )

        assert product["values"] == [9900.0, 14400.0, 11880.0, 17280.0]
        assert product["total_degree"] == 2
        assert product["exponents"] == [
            {"feature": "height", "exponent": 1},
            {"feature": "girth", "exponent": 1},
        ]

    def test_a_square_is_the_column_times_itself(self) -> None:
        answer = posted(EXPANSION, FOUR_PEOPLE)
        squared = next(term for term in answer["terms"] if term["name"] == "girth^2")

        assert squared["values"] == [4356.0, 9216.0, 4356.0, 9216.0]

    def test_turning_the_products_off_drops_exactly_the_cross_term(self) -> None:
        answer = posted(EXPANSION, {**FOUR_PEOPLE, "include_interactions": False})

        assert [term["name"] for term in answer["terms"]] == [
            "height",
            "girth",
            "height^2",
            "girth^2",
        ]

    def test_degree_three_over_two_columns_names_nine(self) -> None:
        answer = posted(EXPANSION, {**FOUR_PEOPLE, "degree": 3})

        assert answer["n_terms"] == 9
        assert "height*girth^2" in [term["name"] for term in answer["terms"]]

    def test_degree_one_returns_the_columns_unchanged(self) -> None:
        answer = posted(EXPANSION, {**FOUR_PEOPLE, "degree": 1})

        assert [term["name"] for term in answer["terms"]] == ["height", "girth"]
        assert answer["terms"][0]["values"] == [150.0, 150.0, 180.0, 180.0]


class TestHowManyColumns:
    """The growth table, every entry read off a built expansion."""

    def test_the_rows_the_page_prints(self) -> None:
        answer = client.get(COLUMN_COUNTS).json()
        counts = {row["width"]: row["counts"] for row in answer["rows"]}

        assert answer["degrees"] == [1, 2, 3, 4, 5]
        assert counts[1] == [1, 2, 3, 4, 5]
        assert counts[2] == [2, 5, 9, 14, 20]
        assert counts[3] == [3, 9, 19, 34, 55]
        assert counts[5] == [5, 20, 55, 125, 251]
        assert counts[10] == [10, 65, 285, 1000, 3002]

    @pytest.mark.parametrize(
        ("width", "degree", "expected"),
        [(2, 2, 5), (3, 3, 19), (10, 3, 285), (10, 5, 3002)],
    )
    def test_the_count_is_the_binomial_less_the_constant(
        self, width: int, degree: int, expected: int
    ) -> None:
        from math import comb

        answer = client.get(COLUMN_COUNTS).json()
        counts = {row["width"]: row["counts"] for row in answer["rows"]}

        assert counts[width][degree - 1] == expected == comb(width + degree, degree) - 1


class TestOneCurveThroughTheCrowd:
    """The fit the playground draws, and the misses a straight line leaves."""

    def test_the_line_leaves_its_misses_in_a_pattern(self) -> None:
        answer = posted(CURVE_FIT, {"points": CROWD, "degree": 1, "standardise": False})
        signs = ["+" if value >= 0 else "-" for value in answer["residuals"]]
        runs = 1 + sum(1 for left, right in pairwise(signs) if left != right)

        assert answer["train_r_squared"] == pytest.approx(0.962028, abs=5e-7)
        assert answer["residual_sum_of_squares"] == pytest.approx(548.3307, abs=5e-5)
        assert runs == 10
        assert signs[:2] == ["+", "+"] and signs[2:8] == ["-"] * 6

    def test_the_line_guesses_a_child_far_too_light(self) -> None:
        answer = posted(CURVE_FIT, {"points": CROWD, "degree": 1, "standardise": False})

        assert answer["fitted"][0] == pytest.approx(7.80, abs=5e-3)
        assert WEIGHTS[0] == 16.1

    def test_one_squared_column_halves_the_misses(self) -> None:
        line = posted(CURVE_FIT, {"points": CROWD, "degree": 1, "standardise": False})
        bend = posted(CURVE_FIT, {"points": CROWD, "degree": 2, "standardise": False})

        assert bend["n_terms"] == 2
        assert bend["term_names"] == ["height", "height^2"]
        assert bend["residual_sum_of_squares"] == pytest.approx(307.1806, abs=5e-5)
        assert bend["residual_sum_of_squares"] < line["residual_sum_of_squares"]

    def test_the_cubic_reaches_the_child(self) -> None:
        answer = posted(CURVE_FIT, {"points": CROWD, "degree": 3, "standardise": False})

        assert answer["fitted"][0] == pytest.approx(16.67, abs=5e-3)
        assert answer["train_r_squared"] == pytest.approx(0.986371, abs=5e-7)

    @pytest.mark.parametrize(
        ("degree", "longest_stretch", "child"),
        [(1, 6, 7.80), (2, 5, 14.02), (3, 3, 16.67)],
    )
    def test_the_pattern_in_the_misses_shortens_with_the_degree(
        self, degree: int, longest_stretch: int, child: float
    ) -> None:
        answer = posted(
            CURVE_FIT, {"points": CROWD, "degree": degree, "standardise": False}
        )
        signs = ["+" if value >= 0 else "-" for value in answer["residuals"]]

        longest = 0
        running = 0
        for position, sign in enumerate(signs):
            running = running + 1 if position and sign == signs[position - 1] else 1
            longest = max(longest, running)

        assert longest == longest_stretch
        assert answer["fitted"][0] == pytest.approx(child, abs=5e-3)

    def test_the_same_people_are_held_back_every_time(self) -> None:
        for degree in (1, 3, 6):
            answer = posted(CURVE_FIT, {"points": CROWD, "degree": degree})
            assert answer["held_out_indices"] == HELD_OUT_POSITIONS
            assert answer["n_training_rows"] == 15

        assert [HEIGHTS[position] for position in HELD_OUT_POSITIONS] == [
            131.0,
            137.0,
            149.0,
            175.0,
            180.0,
            187.0,
            191.0,
        ]

    def test_the_curve_is_sampled_across_the_measured_range(self) -> None:
        answer = posted(CURVE_FIT, {"points": CROWD, "degree": 3})

        assert len(answer["curve"]) == 121
        assert answer["curve"][0]["x"] == pytest.approx(104.0)
        assert answer["curve"][-1]["x"] == pytest.approx(191.0)

    def test_the_interpolating_degree_answers_two_people_absurdly(self) -> None:
        answer = posted(CURVE_FIT, {"points": CROWD, "degree": 14})

        assert HEIGHTS[4] == 131.0 and WEIGHTS[4] == 27.6
        assert answer["fitted"][4] == pytest.approx(-9002.94, abs=5e-3)
        assert HEIGHTS[5] == 137.0 and WEIGHTS[5] == 29.1
        assert answer["fitted"][5] == pytest.approx(-4030.13, abs=5e-3)
        assert answer["train_r_squared"] == pytest.approx(0.999808, abs=5e-7)


class TestWhereTheDegreeStopsHelping:
    """The sweep, and the penalty that makes its top end survivable."""

    def test_the_training_score_never_falls(self) -> None:
        answer = posted(DEGREE_GAP, {"points": CROWD, "max_degree": 14})
        scores = [reading["train_r_squared"] for reading in answer["readings"]]

        assert answer["interpolating_degree"] == 14
        assert len(scores) == 14
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(scores))
        assert scores[0] == pytest.approx(0.962028, abs=5e-7)
        assert scores[-1] == pytest.approx(0.999808, abs=5e-7)

    def test_the_held_out_score_collapses_at_the_top(self) -> None:
        answer = posted(DEGREE_GAP, {"points": CROWD, "max_degree": 14})
        held_out = {
            reading["degree"]: reading["held_out_r_squared"]
            for reading in answer["readings"]
        }

        assert held_out[4] == pytest.approx(0.976150, abs=5e-7)
        assert held_out[9] == pytest.approx(0.910499, abs=5e-7)
        assert held_out[10] == pytest.approx(0.978514, abs=5e-7)
        assert held_out[11] == pytest.approx(0.364094, abs=5e-7)
        assert held_out[12] == pytest.approx(-36.3475, abs=5e-5)
        assert held_out[14] == pytest.approx(-24967.2025, abs=5e-3)

    def test_the_best_held_out_degree_is_not_the_highest(self) -> None:
        answer = posted(DEGREE_GAP, {"points": CROWD, "max_degree": 14})

        assert answer["best_held_out_degree"] == 10
        assert answer["best_held_out_score"] == pytest.approx(0.978514, abs=5e-7)
        assert answer["worst_held_out_degree"] == 14

    def test_a_small_penalty_rescues_the_top_degree(self) -> None:
        answer = posted(
            DEGREE_GAP,
            {"points": CROWD, "max_degree": 14, "penalties": [0.001, 0.1]},
        )
        top = answer["readings"][-1]
        penalised = {
            reading["penalty"]: reading["held_out_r_squared"]
            for reading in top["penalised"]
        }

        assert top["held_out_r_squared"] == pytest.approx(-24967.2025, abs=5e-3)
        assert penalised[0.001] == pytest.approx(0.963137, abs=5e-7)
        assert penalised[0.1] == pytest.approx(0.890071, abs=5e-7)

    def test_a_penalty_costs_something_at_a_degree_that_did_not_need_it(self) -> None:
        answer = posted(
            DEGREE_GAP, {"points": CROWD, "max_degree": 4, "penalties": [0.1]}
        )
        fourth = answer["readings"][3]

        assert fourth["held_out_r_squared"] == pytest.approx(0.976150, abs=5e-7)
        assert fourth["penalised"][0]["held_out_r_squared"] == pytest.approx(
            0.967513, abs=5e-7
        )

    def test_the_sweep_stops_where_the_rows_run_out(self) -> None:
        answer = posted(DEGREE_GAP, {"points": CROWD[:6], "max_degree": 14})

        assert answer["n_training_rows"] == 4
        assert [reading["degree"] for reading in answer["readings"]] == [1, 2, 3]


class TestWhatTheProductsAdd:
    """A plane, the powers alone, and the powers with the products."""

    def test_on_crossed_builds_one_product_column_does_the_work(self) -> None:
        answer = posted(TWO_COLUMNS, {"people": BUILDS, "degree": 2})
        misses = {
            fit["label"]: fit["residual_sum_of_squares"] for fit in answer["fits"]
        }

        assert answer["correlation"] == pytest.approx(0.0, abs=1e-12)
        assert misses["the powers alone"] == pytest.approx(28.0725, abs=5e-5)
        assert misses["the powers and the products"] == pytest.approx(1.8285, abs=5e-5)

    def test_a_plane_owes_every_height_the_same_kilograms(self) -> None:
        answer = posted(TWO_COLUMNS, {"people": BUILDS, "degree": 2})
        gains = {gain["height"]: gain for gain in answer["gains"]}

        assert [gain["plane"] for gain in answer["gains"]] == [
            pytest.approx(49.45, abs=5e-3)
        ] * 3
        assert gains[150.0]["measured"] == pytest.approx(44.5, abs=5e-3)
        assert gains[180.0]["measured"] == pytest.approx(54.8, abs=5e-3)
        assert gains[150.0]["expanded"] == pytest.approx(44.59, abs=5e-3)
        assert gains[180.0]["expanded"] == pytest.approx(54.31, abs=5e-3)

    def test_on_the_crowd_the_product_earns_almost_nothing(self) -> None:
        answer = posted(TWO_COLUMNS, {"people": CROWD_PEOPLE, "degree": 2})
        misses = {
            fit["label"]: fit["residual_sum_of_squares"] for fit in answer["fits"]
        }

        assert answer["correlation"] == pytest.approx(0.9886, abs=5e-5)
        assert misses["the powers alone"] == pytest.approx(98.8862, abs=5e-5)
        assert misses["the powers and the products"] == pytest.approx(91.0288, abs=5e-5)

    def test_no_gain_can_be_read_where_every_height_is_its_own(self) -> None:
        answer = posted(TWO_COLUMNS, {"people": CROWD_PEOPLE, "degree": 2})

        assert answer["gains"] == []

    def test_the_plane_is_the_same_columns_whether_products_are_asked_for(self) -> None:
        answer = posted(TWO_COLUMNS, {"people": BUILDS, "degree": 1})
        fits = {fit["label"]: fit for fit in answer["fits"]}

        assert fits["the powers alone"]["term_names"] == ["height", "girth"]
        assert fits["the powers and the products"]["term_names"] == ["height", "girth"]


class TestTheScaleOfTheColumns:
    """What a change of unit does to an expansion, and what it does not."""

    def test_the_spread_between_the_columns_moves_by_ten_thousand(self) -> None:
        answer = posted(UNITS, {"points": CROWD, "degree": 3, "penalty": 100.0})
        readings = {reading["unit"]: reading for reading in answer["readings"]}

        assert readings["centimetres"]["span_ratio"] == pytest.approx(36481.0, abs=5e-2)
        assert readings["metres"]["span_ratio"] == pytest.approx(3.6481, abs=5e-5)

    def test_least_squares_does_not_feel_the_unit(self) -> None:
        answer = posted(UNITS, {"points": CROWD, "degree": 3, "penalty": 100.0})
        plain = [reading["plain_r_squared"] for reading in answer["readings"]]

        assert plain[0] == pytest.approx(0.984072, abs=5e-7)
        assert plain[0] == pytest.approx(plain[1], abs=1e-12)

    def test_a_penalised_fit_feels_it_severely(self) -> None:
        answer = posted(UNITS, {"points": CROWD, "degree": 3, "penalty": 100.0})
        readings = {reading["unit"]: reading for reading in answer["readings"]}

        assert readings["centimetres"]["penalised_r_squared"] == pytest.approx(
            0.977374, abs=5e-7
        )
        assert readings["metres"]["penalised_r_squared"] == pytest.approx(
            0.703648, abs=5e-7
        )

    def test_a_stepped_fit_does_not_feel_it_at_all(self) -> None:
        answer = posted(UNITS, {"points": CROWD, "degree": 3, "penalty": 100.0})
        trees = [reading["tree_r_squared"] for reading in answer["readings"]]

        assert trees[0] == pytest.approx(0.990159, abs=5e-7)
        assert trees[0] == pytest.approx(trees[1], abs=1e-12)

    def test_the_interpolating_degree_cannot_be_reached_on_raw_units(self) -> None:
        answer = posted(UNITS, {"points": CROWD, "degree": 3, "penalty": 1.0})
        high = {reading["reading"]: reading for reading in answer["high_degree"]}

        assert high["as measured"]["largest_value"] == pytest.approx(7.421e31, rel=1e-3)
        assert high["as measured"]["train_r_squared"] == pytest.approx(
            0.987231, abs=5e-7
        )
        assert high["on a common scale"]["largest_value"] == pytest.approx(
            5834.36, abs=5e-2
        )
        assert high["on a common scale"]["train_r_squared"] == pytest.approx(
            0.999808, abs=5e-7
        )


class TestAgainstAFitMadeOfCuts:
    """The honest comparison, both dials swept on the same held-back people."""

    def test_the_curve_wins_at_its_best_setting(self) -> None:
        answer = posted(AGAINST_A_TREE, {"points": CROWD, "max_degree": 8})

        assert answer["best_curve_degree"] == 4
        assert answer["best_curve_score"] == pytest.approx(0.976150, abs=5e-7)
        assert answer["best_step_depth"] == 3
        assert answer["best_step_score"] == pytest.approx(0.950194, abs=5e-7)
        assert answer["best_curve_score"] > answer["best_step_score"]

    def test_the_steps_never_fall_as_far_as_the_curve_can(self) -> None:
        cut_short = posted(AGAINST_A_TREE, {"points": CROWD, "max_degree": 8})
        all_the_way = posted(DEGREE_GAP, {"points": CROWD, "max_degree": 14})

        assert cut_short["worst_step_score"] == pytest.approx(0.643820, abs=5e-7)
        assert all_the_way["worst_held_out_score"] == pytest.approx(
            -24967.2025, abs=5e-3
        )

    def test_a_deeper_tree_memorises_and_then_stops_changing(self) -> None:
        answer = posted(AGAINST_A_TREE, {"points": CROWD, "max_degree": 4})
        steps = {reading["depth"]: reading for reading in answer["step_readings"]}

        assert steps[5]["train_r_squared"] == pytest.approx(1.0, abs=1e-12)
        assert steps[5]["n_leaves"] == 14
        assert steps[4]["held_out_r_squared"] == pytest.approx(
            steps[5]["held_out_r_squared"], abs=1e-12
        )


class TestRefusals:
    """Every way of asking for something that cannot be built."""

    def test_a_column_that_never_varies_is_refused(self) -> None:
        response = client.post(
            EXPANSION,
            json={"columns": [{"name": "build", "values": [1.0] * 6}], "degree": 2},
        )

        assert response.status_code == 422
        assert "constant" in response.json()["detail"]

    def test_two_columns_of_one_name_are_refused(self) -> None:
        response = client.post(
            EXPANSION,
            json={
                "columns": [
                    {"name": "height", "values": [1.0, 2.0, 3.0]},
                    {"name": "height", "values": [2.0, 3.0, 5.0]},
                ],
                "degree": 2,
            },
        )

        assert response.status_code == 422
        assert "unique" in response.json()["detail"]

    def test_columns_of_unequal_length_are_refused(self) -> None:
        response = client.post(
            EXPANSION,
            json={
                "columns": [
                    {"name": "height", "values": [1.0, 2.0, 3.0]},
                    {"name": "girth", "values": [2.0, 3.0]},
                ],
                "degree": 2,
            },
        )

        assert response.status_code == 422
        assert "same number of people" in response.json()["detail"]

    def test_a_degree_beyond_the_rows_is_refused_with_the_arithmetic(self) -> None:
        response = client.post(CURVE_FIT, json={"points": CROWD[:4], "degree": 8})

        assert response.status_code == 422
        assert "9 numbers to pin down" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            CURVE_FIT, json={"points": CROWD, "degree": 2, "order": 3}
        )

        assert response.status_code == 422

    def test_a_degree_past_the_ceiling_is_refused(self) -> None:
        response = client.post(CURVE_FIT, json={"points": CROWD, "degree": 40})

        assert response.status_code == 422


class TestColumnsThatSayNothingNew:
    """The two expansions whose extra columns carry no extra information."""

    def test_a_two_valued_column_squares_to_itself(self) -> None:
        answer = client.get(DEGENERATE).json()
        probe = answer["two_valued"]

        assert probe["original"] == probe["squared"]
        assert probe["columns_match"] is True
        assert probe["fit_succeeded"] is False
        assert probe["reason"] is not None

    def test_more_powers_than_distinct_values_change_nothing(self) -> None:
        answer = client.get(DEGENERATE).json()
        scores = [reading["r_squared"] for reading in answer["repeated"]]

        assert answer["distinct_values"] == 3
        assert scores[0] == pytest.approx(0.992417, abs=5e-7)
        for score in scores[1:]:
            assert score == pytest.approx(scores[0], abs=1e-12)
        assert [reading["n_terms"] for reading in answer["repeated"]] == [2, 3, 4, 5]


IDENTIFIER_FRAGMENTS = (
    "feature_values",
    "PolynomialFeatures",
    "Standardizer",
    "RidgeRegression",
    "MultipleLinearRegression",
    "normal equations",
    "oop_ml",
    " -- ",
    "_",
)
"""Anything a reader has no reason to meet, in a message a widget will print."""


class TestNoIdentifierReachesThePage:
    """A refusal a widget renders is page text, so it carries no code in it."""

    @pytest.mark.parametrize(
        "body",
        [
            {"columns": [{"name": "build", "values": [1.0] * 6}], "degree": 2},
            {
                "columns": [
                    {"name": "height", "values": [1.0, 2.0, 3.0]},
                    {"name": "height", "values": [2.0, 3.0, 5.0]},
                ],
                "degree": 2,
            },
        ],
    )
    def test_every_expansion_refusal_is_in_plain_words(self, body: dict) -> None:
        detail = client.post(EXPANSION, json=body).json()["detail"]

        for fragment in IDENTIFIER_FRAGMENTS:
            assert fragment not in detail

    def test_the_duplicated_column_refusal_is_in_plain_words(self) -> None:
        reason = client.get(DEGENERATE).json()["two_valued"]["reason"]

        for fragment in IDENTIFIER_FRAGMENTS:
            assert fragment not in reason
        assert "condition number" not in reason

    def test_the_rewriting_is_still_needed(self) -> None:
        assert PLAIN_WORDS, "the substitution table has been emptied"
