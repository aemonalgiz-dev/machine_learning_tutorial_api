"""Spec for the distance-and-similarity endpoints.

Three things are pinned here rather than merely exercised. The first is the
three-word table, whose every entry is a real co-occurrence count and whose
every rule's answer can be worked with a pencil, so the numbers in the page's
worked section are checked against arithmetic rather than against the code that
produced them. The second is the pair of correlations between a word's use
count and its vector's length, which come out with opposite signs under the two
fits and are the page's central measured finding. The third is that nothing an
endpoint returns as prose carries an identifier from the code, since a refusal a
widget renders is page text.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.distance_and_similarity import (
    HAND_CONTEXTS,
    HAND_WORDS,
    METRIC_NAMES,
    PLAIN_NAMES,
)
from oop_ml_api.main import app

client = TestClient(app)
BASE = "/concepts/distance-and-similarity"


def get(path: str):
    response = client.get(f"{BASE}{path}")
    assert response.status_code == 200
    return response.json()


def post(path: str, body: dict):
    response = client.post(f"{BASE}{path}", json=body)
    assert response.status_code == 200
    return response.json()


class TestTheSpace:
    def test_the_corpus_is_the_one_the_topic_shares(self):
        body = get("/space")

        assert body["n_documents"] == 24
        assert body["n_words"] == 23
        assert body["dimension"] == 4
        assert body["window"] == 2

    def test_the_table_costs_one_number_per_word_per_coordinate(self):
        body = get("/space")

        assert body["table_numbers"] == 23 * 4
        assert body["n_pairs"] == 23 * 22 // 2

    def test_the_six_rules_are_named_and_each_has_a_plain_name(self):
        body = get("/space")

        assert body["metrics"] == list(METRIC_NAMES)
        assert set(body["plain_names"]) == set(METRIC_NAMES)

    def test_the_commonest_words_have_the_shortest_positions(self):
        rows = {row["word"]: row for row in get("/space")["words"]}

        for word in ("and", "the", "we"):
            assert rows[word]["use_count"] == 8
            assert rows[word]["length"] < 0.6
        assert rows["deck"]["length"] == pytest.approx(1.1628, abs=1.0e-04)


class TestTheHandTable:
    """Three words against two contexts, small enough to check with a pencil."""

    def test_the_entries_are_the_counts_the_corpus_holds(self):
        body = post("/by-hand", {})

        assert body["contexts"] == list(HAND_CONTEXTS)
        assert body["words"] == list(HAND_WORDS)
        assert body["rows"] == [[5.0, 0.0], [2.0, 0.0], [0.0, 5.0]]

    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            (
                "bake",
                "stir",
                {
                    "euclidean": 3.0,
                    "manhattan": 3.0,
                    "chebyshev": 3.0,
                    "cosine": 0.0,
                    "hamming": 0.5,
                    "canberra": 3.0 / 7.0,
                },
            ),
            (
                "bake",
                "sail",
                {
                    "euclidean": math.sqrt(50.0),
                    "manhattan": 10.0,
                    "chebyshev": 5.0,
                    "cosine": 1.0,
                    "hamming": 1.0,
                    "canberra": 2.0,
                },
            ),
            (
                "stir",
                "sail",
                {
                    "euclidean": math.sqrt(29.0),
                    "manhattan": 7.0,
                    "chebyshev": 5.0,
                    "cosine": 1.0,
                    "hamming": 1.0,
                    "canberra": 2.0,
                },
            ),
        ],
    )
    def test_every_rule_matches_the_arithmetic(self, first, second, expected):
        # Written from the definitions, so agreement is a real check rather
        # than the endpoint agreeing with itself.
        pairs = {
            (row["first"], row["second"]): row for row in post("/by-hand", {})["pairs"]
        }

        assert pairs[(first, second)]["answers"] == pytest.approx(expected)

    def test_two_words_pointing_one_way_are_apart_by_every_rule_but_the_angle(self):
        pairs = {
            (row["first"], row["second"]): row for row in post("/by-hand", {})["pairs"]
        }
        answers = pairs[("bake", "stir")]["answers"]

        assert answers["cosine"] == pytest.approx(0.0)
        assert pairs[("bake", "stir")]["cosine_similarity"] == pytest.approx(1.0)
        assert all(answers[name] > 0.0 for name in METRIC_NAMES if name != "cosine")

    def test_counting_one_word_twice_over_moves_four_rules_of_six(self):
        once = {
            (row["first"], row["second"]): row["answers"]
            for row in post("/by-hand", {"times": 1.0})["pairs"]
        }
        twice = {
            (row["first"], row["second"]): row["answers"]
            for row in post("/by-hand", {"times": 2.0})["pairs"]
        }

        moved = {
            name
            for name in METRIC_NAMES
            if once[("bake", "stir")][name] != twice[("bake", "stir")][name]
        }
        assert moved == {"euclidean", "manhattan", "chebyshev", "canberra"}

    def test_a_word_outside_the_small_table_is_refused_in_plain_words(self):
        response = client.post(f"{BASE}/by-hand", json={"scaled_word": "harbour"})

        assert response.status_code == 422
        assert "harbour" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self):
        assert client.post(f"{BASE}/by-hand", json={"scale": 2.0}).status_code == 422


class TestNeighbours:
    def test_every_rule_ranks_the_whole_vocabulary_but_the_word_itself(self):
        body = post("/neighbours", {"word": "sail", "n_results": 22})

        assert len(body["rankings"]) == len(METRIC_NAMES)
        for ranking in body["rankings"]:
            assert len(ranking["rows"]) == 22
            assert "sail" not in [row["word"] for row in ranking["rows"]]

    def test_the_angle_and_the_straight_line_name_different_first_places(self):
        body = post("/neighbours", {"word": "sail", "n_results": 3})
        first = {ranking["metric"]: ranking["rows"][0] for ranking in body["rankings"]}

        assert first["euclidean"]["word"] == "mast"
        assert first["cosine"]["word"] == "rope"
        assert first["euclidean"]["reading"] == pytest.approx(0.2750, abs=1.0e-04)
        assert first["cosine"]["reading"] == pytest.approx(0.0175, abs=1.0e-04)

    def test_reading_labels_off_a_fitted_table_answers_from_the_other_topic(self):
        body = post("/neighbours", {"word": "sail", "n_results": 2})
        labels = next(
            ranking for ranking in body["rankings"] if ranking["metric"] == "hamming"
        )

        assert labels["rows"][0]["word"] == "flour"
        assert labels["rows"][0]["reading"] == pytest.approx(0.5)
        assert labels["rows"][1]["reading"] == pytest.approx(1.0)

    def test_the_bounded_rules_declare_their_ceiling(self):
        rankings = {
            ranking["metric"]: ranking
            for ranking in post("/neighbours", {"word": "sail"})["rankings"]
        }

        assert rankings["cosine"]["ceiling"] == 2.0
        assert rankings["hamming"]["ceiling"] == 1.0
        assert rankings["euclidean"]["ceiling"] is None

    def test_one_question_reads_every_other_position_once(self):
        body = post("/neighbours", {"word": "sail"})

        assert body["comparisons"] == 22
        assert body["multiplications"] == 22 * 4

    def test_a_word_the_corpus_never_used_is_refused(self):
        response = client.post(f"{BASE}/neighbours", json={"word": "kettle"})

        assert response.status_code == 422
        assert "kettle" in response.json()["detail"]


class TestLengths:
    def test_the_two_fits_disagree_about_the_sign(self):
        body = get("/lengths")

        assert body["counting_correlation"] == pytest.approx(-0.6284, abs=1.0e-04)
        assert body["predicting_correlation"] == pytest.approx(0.7005, abs=1.0e-04)

    def test_the_commonest_words_are_named_with_their_count(self):
        body = get("/lengths")

        assert body["commonest_words"] == ["and", "the", "we"]
        assert body["commonest_count"] == 8

    def test_every_word_carries_a_length_under_both_fits(self):
        body = get("/lengths")

        assert len(body["rows"]) == 23
        assert all(row["counting_length"] > 0.0 for row in body["rows"])
        assert all(row["predicting_length"] > 0.0 for row in body["rows"])


class TestAgreement:
    def test_a_rule_agrees_with_itself_everywhere(self):
        body = get("/agreement")

        for position, row in enumerate(body["grid"]):
            assert row[position] == body["n_words"]

    def test_the_two_summed_rules_never_part_and_the_angle_often_does(self):
        body = get("/agreement")
        at = {name: position for position, name in enumerate(body["metrics"])}
        grid = body["grid"]

        assert grid[at["euclidean"]][at["manhattan"]] == 23
        assert grid[at["euclidean"]][at["cosine"]] == 15
        assert grid[at["euclidean"]][at["hamming"]] == 2

    def test_the_angle_and_the_straight_line_part_on_eight_words(self):
        body = get("/agreement")

        assert len(body["angle_against_line"]) == 8
        named = {row["word"] for row in body["angle_against_line"]}
        assert {"sail", "rope", "flour", "whisk"} <= named

    def test_exact_coordinate_matches_are_almost_absent(self):
        body = get("/agreement")

        assert body["n_pairs"] == 253
        assert body["exact_coordinate_matches"] == 6
        assert body["best_label_match"]["n_matching"] == 2

    def test_on_the_unit_sphere_the_angle_and_the_line_give_one_order(self):
        body = get("/agreement")

        assert body["unit_orders_agree"] is True
        assert body["largest_identity_gap"] < 1.0e-14
        assert body["raw_order_matches_unit_order"] is False


class TestMoved:
    def test_the_shortcut_agrees_until_the_positions_are_moved(self):
        rows = post("/moved", {})["rows"]
        at_home = rows[0]

        assert at_home["shift"] == 0.0
        assert at_home["expanded"] == pytest.approx(at_home["definition"], abs=1.0e-15)
        # One digit rather than none, since the two routes are different
        # sequences of roundings even when nothing has been moved.
        assert at_home["digits_lost"] <= 1

    def test_far_from_the_origin_the_shortcut_answers_something_else(self):
        rows = {row["shift"]: row for row in post("/moved", {})["rows"]}
        far = rows[1.0e08]

        assert far["definition"] == pytest.approx(0.2750, abs=1.0e-04)
        assert far["expanded"] > 2.0
        assert far["digits_lost"] == 16

    def test_taking_the_pair_mean_out_first_restores_every_digit(self):
        for row in post("/moved", {})["rows"]:
            assert row["shifted_first"] == pytest.approx(row["definition"], abs=1.0e-15)

    def test_the_two_routes_agree_on_ordinary_positions(self):
        body = post("/moved", {})

        assert body["largest_relative_gap"] < 1.0e-15
        assert body["n_random_pairs"] == 2400


class TestContrast:
    def test_the_fitted_table_keeps_more_contrast_until_it_does_not(self):
        rows = {row["dimension"]: row for row in get("/contrast")["rows"]}

        assert rows[2]["fitted"] > rows[2]["uniform"]
        assert rows[8]["fitted"] > rows[8]["uniform"]
        assert rows[16]["fitted"] < rows[16]["uniform"]

    def test_contrast_falls_as_coordinates_are_added(self):
        rows = get("/contrast")["rows"]
        fitted = [row["fitted"] for row in rows]

        assert fitted == sorted(fitted, reverse=True)

    def test_the_crossover_is_reported_rather_than_hidden(self):
        assert get("/contrast")["crossover_dimension"] == 16


class TestCost:
    def test_the_arithmetic_is_words_times_coordinates(self):
        for row in get("/cost")["rows"]:
            assert row["multiplications"] == row["n_words"] * row["dimension"]

    def test_the_largest_vocabulary_is_counted_and_not_timed(self):
        rows = get("/cost")["rows"]

        assert rows[-1]["n_words"] == 200_000
        assert rows[-1]["milliseconds"] is None
        assert all(row["milliseconds"] is not None for row in rows[:-1])


class TestUndefined:
    def test_the_origin_has_no_angle_and_an_ordinary_straight_line_gap(self):
        cases = {case["situation"]: case for case in get("/undefined")["cases"]}
        by_angle = cases["a word whose position is the origin, compared by angle"]
        by_line = cases["the same word, compared by straight line"]

        assert "points nowhere" in by_angle["outcome"]
        assert "0.8225" in by_line["outcome"]

    def test_nearest_is_not_a_symmetric_relation(self):
        body = get("/undefined")

        assert body["symmetry_gap"] == 0.0
        assert body["n_asymmetric"] == 5
        pairs = {(row["word"], row["nearest"]) for row in body["asymmetric"]}
        assert ("mast", "sail") in pairs

    def test_every_case_says_what_is_undefined_and_what_came_back(self):
        for case in get("/undefined")["cases"]:
            assert case["situation"]
            assert case["what_is_undefined"]
            assert case["outcome"]


class TestTriangle:
    def test_only_the_angle_admits_a_shorter_detour(self):
        body = get("/triangle")
        violations = dict(zip(body["metrics"], body["violations"], strict=True))

        assert violations["cosine"] == 1540
        assert all(count == 0 for name, count in violations.items() if name != "cosine")

    def test_the_worst_detour_is_named_with_its_three_legs(self):
        body = get("/triangle")

        assert body["n_triples"] == 10626
        assert body["worst_route"]["cosine"] == ["harbour", "mast", "crew"]
        first, second, direct = body["worst_legs"]["cosine"]
        assert first + second < direct
        excess = dict(zip(body["metrics"], body["worst_excess"], strict=True))
        assert excess["cosine"] == pytest.approx(0.4284, abs=1.0e-04)


IDENTIFIER = re.compile(r"\b(?:[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z][A-Za-z]*)\b")
"""Anything shaped like a name from the code: snake case, or camel case."""

ALLOWED = {"use_count", "n_results"}
"""Field names, which are keys in a response rather than prose a reader sees."""


def _sentences(body: object) -> list[str]:
    """Every string a widget could render, gathered from a whole response."""
    if isinstance(body, str):
        return [body]
    if isinstance(body, dict):
        return [
            sentence
            for key, value in body.items()
            if key not in {"words", "metrics", "documents", "nearest"}
            for sentence in _sentences(value)
        ]
    if isinstance(body, list):
        return [sentence for value in body for sentence in _sentences(value)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    """A refusal a widget renders is page text, so it is held to the page's rule."""

    @pytest.mark.parametrize("path", ["/undefined", "/agreement", "/triangle"])
    def test_no_sentence_carries_an_identifier(self, path):
        for sentence in _sentences(get(path)):
            found = set(IDENTIFIER.findall(sentence)) - ALLOWED
            assert not found, f"{path} says {found} in {sentence!r}"

    @pytest.mark.parametrize(
        ("path", "body"),
        [("/neighbours", {"word": "kettle"}), ("/by-hand", {"scaled_word": "harbour"})],
    )
    def test_no_refusal_carries_an_identifier(self, path, body):
        detail = client.post(f"{BASE}{path}", json=body).json()["detail"]

        assert not set(IDENTIFIER.findall(detail)) - ALLOWED

    def test_the_plain_names_are_descriptions_rather_than_names(self):
        for name, plain in PLAIN_NAMES.items():
            assert name not in plain.lower()
            assert plain == plain.lower()
            assert "_" not in plain
