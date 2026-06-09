"""Spec for the finite scalar quantisation endpoints.

Every number the page quotes in prose is pinned here, so a change in the
library that moved one would fail here rather than reaching a reader. The
central claim, that the obvious rounding rule reaches one rung too few at an
even level count, is checked across the whole sweep rather than at one value.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient
from oop_ml.core.exceptions import InvalidValuesError, UnknownTokenError

from oop_ml_api.concepts.finite_scalar_quantisation import (
    PLAIN_WORDS,
    _in_plain_words,
)
from oop_ml_api.main import app

client = TestClient(app)

BASE = "/concepts/finite-scalar-quantisation"

WORKED_INPUT = [1.0, -0.5, 0.2, 3.0]
PAPER_LEVELS = [8, 5, 5, 5]


def _strings_within(body: object) -> list[str]:
    """Every string a widget could render, which is the values and not the keys."""
    if isinstance(body, str):
        return [body]
    if isinstance(body, dict):
        return [text for value in body.values() for text in _strings_within(value)]
    if isinstance(body, list):
        return [text for value in body for text in _strings_within(value)]
    return []


def quantise(values: list[float], levels: list[int]) -> dict:
    answer = client.post(f"{BASE}/quantise", json={"values": values, "levels": levels})
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.fixture(scope="module")
def words() -> dict:
    answer = client.get(f"{BASE}/words")
    assert answer.status_code == 200
    return answer.json()


@pytest.fixture(scope="module")
def levels() -> dict:
    answer = client.get(f"{BASE}/levels")
    assert answer.status_code == 200
    return answer.json()


@pytest.fixture(scope="module")
def cost() -> dict:
    answer = client.get(f"{BASE}/cost")
    assert answer.status_code == 200
    return answer.json()


@pytest.fixture(scope="module")
def limits() -> dict:
    answer = client.get(f"{BASE}/limits")
    assert answer.status_code == 200
    return answer.json()


class TestOneVector:
    def test_the_four_numbers_the_page_walks_through(self):
        body = quantise(WORKED_INPUT, PAPER_LEVELS)

        assert body["digits"] == [6, 1, 2, 4]
        assert body["code_id"] == 894
        assert body["n_codes"] == 1000
        assert body["reconstruction"] == pytest.approx([5 / 7, -0.5, 0.0, 1.0])
        assert body["distortion"] == pytest.approx(0.04265467092229372)

    def test_the_ladder_is_reported_step_by_step(self):
        steps = quantise(WORKED_INPUT, PAPER_LEVELS)["steps"]

        assert [step["half_width"] for step in steps] == [3.5, 2.0, 2.0, 2.0]
        assert [step["offset"] for step in steps] == [0.5, 0.0, 0.0, 0.0]
        assert [step["position"] for step in steps] == [2, -1, 0, 2]
        assert [step["place_value"] for step in steps] == [1, 8, 40, 200]
        assert steps[0]["squashed"] == pytest.approx(0.7615941559557649)
        assert steps[0]["scaled"] == pytest.approx(2.1655795458451773)

    def test_the_squared_gaps_add_up_to_the_reported_cost(self):
        body = quantise(WORKED_INPUT, PAPER_LEVELS)

        assert sum(step["squared_gap"] for step in body["steps"]) == pytest.approx(
            body["distortion"]
        )

    def test_the_two_expressions_read_as_the_page_prints_them(self):
        body = quantise(WORKED_INPUT, PAPER_LEVELS)

        assert body["id_expression"] == "6 × 1 + 1 × 8 + 2 × 40 + 4 × 200 = 894"
        assert body["product_expression"] == "8 × 5 × 5 × 5 = 1000"
        assert body["bits"] == pytest.approx(math.log2(1000))

    def test_a_saturating_input_lands_on_the_last_code_and_costs_nothing(self):
        body = quantise([40.0, 40.0, 40.0, 40.0], PAPER_LEVELS)

        assert body["code_id"] == 999
        assert body["digits"] == [7, 4, 4, 4]
        assert body["distortion"] == pytest.approx(0.0)

    def test_the_opposite_corner_is_the_first_code(self):
        body = quantise([-40.0, -40.0, -40.0, -40.0], PAPER_LEVELS)

        assert body["code_id"] == 0
        assert body["digits"] == [0, 0, 0, 0]

    def test_the_word_the_page_checks_by_hand(self, words):
        by_word = {entry["word"]: entry for entry in words["named"]}
        cost_word = by_word["cost"]

        body = quantise(cost_word["values"], PAPER_LEVELS)
        assert body["digits"] == [5, 3, 2, 1]
        assert body["code_id"] == 309
        assert body["id_expression"].endswith("= 309")


class TestTheRoundingRule:
    def test_the_obvious_rule_reaches_one_rung_too_few_at_every_even_count(
        self, levels
    ):
        even = [row for row in levels["rows"] if row["n_levels"] % 2 == 0]

        assert len(even) == 6
        for row in even:
            assert row["naive_rungs"] == row["n_levels"] - 1
            assert row["corrected_rungs"] == row["n_levels"]

    def test_the_two_rules_agree_at_every_odd_count(self, levels):
        odd = [row for row in levels["rows"] if row["n_levels"] % 2 == 1]

        assert len(odd) == 5
        for row in odd:
            assert row["naive_rungs"] == row["n_levels"]
            assert row["corrected_rungs"] == row["n_levels"]
            assert row["offset"] == 0.0

    def test_two_levels_collapse_to_one_under_the_obvious_rule(self, levels):
        two = next(row for row in levels["rows"] if row["n_levels"] == 2)

        assert two["naive_rungs"] == 1
        assert two["naive_lowest"] == 0 and two["naive_highest"] == 0
        assert two["corrected_rungs"] == 2
        assert two["rungs"] == [-1.0, 1.0]

    def test_the_extra_pair_at_saturation_appears_only_at_some_even_counts(
        self, levels
    ):
        by_count = {row["n_levels"]: row for row in levels["rows"]}

        assert [by_count[count]["naive_with_saturation"] for count in (4, 8, 12)] == [
            5,
            9,
            13,
        ]
        assert [by_count[count]["naive_with_saturation"] for count in (2, 6, 10)] == [
            1,
            5,
            9,
        ]

    def test_the_corrected_rungs_are_evenly_spaced_and_reach_both_ends(self, levels):
        eight = next(row for row in levels["rows"] if row["n_levels"] == 8)

        assert eight["rungs"][0] == pytest.approx(-1.0)
        assert eight["rungs"][-1] == pytest.approx(1.0)
        gaps = [
            second - first for first, second in zip(eight["rungs"], eight["rungs"][1:])
        ]
        assert gaps == pytest.approx([2 / 7] * 7)

    def test_the_published_configuration_hides_the_fault_and_an_even_one_does_not(
        self, levels
    ):
        by_levels = {tuple(row["levels"]): row for row in levels["configurations"]}

        published = by_levels[(8, 5, 5, 5)]
        assert published["asked_for"] == 1000
        assert published["naive_reach"] == 875
        assert published["naive_distortion"] == pytest.approx(0.058718, abs=5e-6)
        assert published["corrected_distortion"] == pytest.approx(0.056450, abs=5e-6)

        all_even = by_levels[(4, 4, 4, 4)]
        assert all_even["asked_for"] == 256
        assert all_even["naive_reach"] == 81
        assert all_even["naive_distortion"] == pytest.approx(0.198371, abs=5e-6)
        assert all_even["corrected_distortion"] == pytest.approx(0.110766, abs=5e-6)

    def test_an_all_odd_configuration_is_identical_under_both_rules(self, levels):
        odd = next(
            row for row in levels["configurations"] if row["levels"] == [5, 5, 5, 5]
        )

        assert odd["naive_reach"] == odd["asked_for"] == 625
        assert odd["naive_distortion"] == pytest.approx(odd["corrected_distortion"])

    def test_the_squashing_reaches_one_exactly_at_a_finite_input(self, levels):
        assert levels["saturating_input"] == pytest.approx(18.99034, abs=1e-4)

    def test_zero_under_an_even_count_is_not_a_rung(self, limits):
        even = limits["zero_under_even"]
        odd = limits["zero_under_odd"]

        assert even["reconstruction"] == pytest.approx([1 / 7])
        assert even["distortion"] == pytest.approx(1 / 49)
        assert odd["reconstruction"] == pytest.approx([0.0])
        assert odd["distortion"] == pytest.approx(0.0)


class TestTheWords:
    def test_the_collection_is_seventy_two_words_with_thirty_five_vectors(self, words):
        assert words["n_words"] == 72
        assert words["n_distinct_vectors"] == 35
        assert len(words["corpus"]) == 18

    def test_the_running_sentence_is_almost_absent_from_the_corpus(self, words):
        held = [
            piece["piece"] for piece in words["pieces"] if piece["in_the_collection"]
        ]

        assert (
            words["sentence"] == "Dr. Alvarez didn't expect the low-cost re-analysis."
        )
        assert held == ["Dr.", "the"]
        assert words["n_pieces_in_the_collection"] == 2

    def test_nineteen_codes_hold_the_whole_vocabulary(self, words):
        assert words["n_codes"] == 1000
        assert words["n_distinct_codes"] == 19
        assert words["n_empty_codes"] == 981

    def test_two_different_words_round_onto_one_code(self, words):
        by_word = {entry["word"]: entry for entry in words["named"]}

        assert by_word["lower"]["code_id"] == by_word["estimate."]["code_id"] == 748
        assert by_word["lower"]["values"] != by_word["estimate."]["values"]
        assert (
            by_word["lower"]["reconstruction"] == by_word["estimate."]["reconstruction"]
        )

    def test_the_named_words_land_where_the_page_says(self, words):
        by_word = {entry["word"]: entry for entry in words["named"]}

        assert by_word["cost"]["digits"] == [5, 3, 2, 1]
        assert by_word["cost"]["code_id"] == 309
        assert by_word["analysis"]["code_id"] == 341
        assert by_word["report"]["code_id"] == 461
        assert by_word["expected"]["code_id"] == 509
        assert by_word["the"]["code_id"] == 495

    def test_every_word_carries_four_numbers_and_four_digits(self, words):
        for entry in words["words"]:
            assert len(entry["values"]) == 4
            assert len(entry["digits"]) == 4
            assert len(entry["reconstruction"]) == 4
            assert 0 <= entry["code_id"] < 1000


class TestWhatTheFitWouldHaveBought:
    def test_the_cost_falls_as_the_codes_multiply(self, cost):
        counts = [row["n_codes"] for row in cost["rows"]]

        assert counts == sorted(counts)
        assert cost["n_rows"] == 600
        assert cost["rows"][0]["n_codes"] == 16
        assert cost["rows"][0]["grid_distortion"] == pytest.approx(0.820028, abs=5e-6)
        assert cost["rows"][-1]["n_codes"] == 160
        assert cost["rows"][-1]["grid_distortion"] == pytest.approx(0.282809, abs=5e-6)

    def test_a_fitted_table_beats_the_grid_at_every_matched_size(self, cost):
        for row in cost["rows"]:
            assert row["table_distortion"] < row["grid_distortion"]
            assert row["ratio"] > 2.0

        ratios = [row["ratio"] for row in cost["rows"]]
        assert min(ratios) == pytest.approx(2.036, abs=5e-3)
        assert max(ratios) == pytest.approx(4.963, abs=5e-3)

    def test_the_fitted_table_uses_every_one_of_its_codes(self, cost):
        for row in cost["rows"]:
            assert row["table_codes_used"] == row["n_codes"]
            assert row["grid_codes_used"] <= row["n_codes"]

        largest = next(row for row in cost["rows"] if row["n_codes"] == 160)
        assert largest["grid_codes_used"] == 152

    def test_the_published_grid_leaves_more_than_half_its_codes_empty(self, cost):
        assert cost["published_codes"] == 1000
        assert cost["published_distinct"] == 420
        assert cost["published_empty"] == 580
        assert cost["published_busiest"] == 6
        assert cost["published_distortion"] == pytest.approx(0.056450, abs=5e-6)

    def test_the_usage_histogram_accounts_for_every_vector(self, cost):
        held = sum(row["n_vectors"] * row["n_codes"] for row in cost["published_usage"])
        chosen = sum(row["n_codes"] for row in cost["published_usage"])

        assert held == cost["n_rows"]
        assert chosen == cost["published_distinct"]


class TestWhereTheMethodStops:
    def test_the_scale_moves_everything_and_nothing_chooses_it(self, limits):
        by_factor = {row["factor"]: row for row in limits["scale_rows"]}

        assert by_factor[0.1]["codes_used"] == 30
        assert by_factor[1.0]["codes_used"] == 420
        assert by_factor[8.0]["codes_used"] == 117
        assert by_factor[0.1]["distortion"] == pytest.approx(
            by_factor[1.0]["distortion"], abs=1e-4
        )
        assert by_factor[8.0]["distortion"] < by_factor[1.0]["distortion"]
        assert by_factor[8.0]["share_past_the_edge"] > 0.8

    def test_a_lean_between_two_coordinates_costs_codes_rather_than_error(self, limits):
        row = next(
            entry for entry in limits["lean_rows"] if entry["levels"] == [4, 4, 2, 2]
        )

        assert limits["correlation"] == pytest.approx(0.9486, abs=5e-4)
        assert row["leaning_grid_used"] == 46
        assert row["straight_grid_used"] == 64
        assert row["leaning_grid"] == pytest.approx(row["straight_grid"], abs=2e-3)
        assert row["leaning_table"] < row["straight_table"]

    def test_the_same_codes_spent_differently_are_worth_different_amounts(self, limits):
        by_levels = {tuple(row["levels"]): row for row in limits["spending_rows"]}

        assert by_levels[(8, 2, 2, 2)]["n_codes"] == 64
        assert by_levels[(4, 4, 2, 2)]["n_codes"] == 64
        assert (
            by_levels[(8, 2, 2, 2)]["distortion"]
            > by_levels[(4, 4, 2, 2)]["distortion"]
        )
        assert by_levels[(3, 3, 3, 2)]["n_codes"] == 54
        assert (
            by_levels[(3, 3, 3, 2)]["distortion"]
            < by_levels[(4, 4, 2, 2)]["distortion"]
        )


class TestRefusals:
    def test_a_vector_and_its_level_counts_must_be_the_same_length(self):
        answer = client.post(
            f"{BASE}/quantise", json={"values": [1.0, 2.0], "levels": [8, 5, 5]}
        )

        assert answer.status_code == 422
        assert "each coordinate needs exactly one of each" in answer.json()["detail"]

    def test_a_grid_larger_than_the_page_works_at_is_refused(self):
        answer = client.post(
            f"{BASE}/quantise",
            json={"values": [1.0] * 4, "levels": [16, 16, 16, 16]},
        )

        assert answer.status_code == 422
        assert "65536" in answer.json()["detail"]

    def test_one_level_carries_nothing_and_is_refused(self):
        answer = client.post(f"{BASE}/quantise", json={"values": [1.0], "levels": [1]})

        assert answer.status_code == 422

    def test_an_unknown_field_is_refused(self):
        answer = client.post(
            f"{BASE}/quantise",
            json={"values": [1.0], "levels": [8], "codebook": "learned"},
        )

        assert answer.status_code == 422

    def test_a_vector_too_wide_for_the_page_is_refused(self):
        answer = client.post(
            f"{BASE}/quantise", json={"values": [0.1] * 8, "levels": [3] * 8}
        )

        assert answer.status_code == 422

    def test_no_coordinates_at_all_is_refused(self):
        answer = client.post(f"{BASE}/quantise", json={"values": [], "levels": []})

        assert answer.status_code == 422


class TestNothingNamesTheCode:
    def test_a_refusal_loses_the_vocabulary_of_the_code(self):
        rewritten = _in_plain_words(
            UnknownTokenError("no code has id 4000; code_ids run from 0 to 999")
        )

        assert "code_ids" not in rewritten
        assert "code number" in rewritten

    def test_every_substitution_removes_what_it_names(self):
        for identifier, replacement in PLAIN_WORDS:
            rewritten = _in_plain_words(InvalidValuesError(f"a {identifier} is wrong"))

            assert identifier not in rewritten or identifier == replacement

    @pytest.mark.parametrize(
        "path", ["/words", "/levels", "/cost", "/limits"], ids=lambda path: path[1:]
    )
    def test_no_endpoint_returns_an_identifier_shaped_string(self, path):
        answer = client.get(f"{BASE}{path}")

        assert answer.status_code == 200
        for text in _strings_within(answer.json()):
            assert "_" not in text, text
            assert not re.search(r"[a-z][A-Z]", text), text
            assert "oop" not in text.lower(), text
