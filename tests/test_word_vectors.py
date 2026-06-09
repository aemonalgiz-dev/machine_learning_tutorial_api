"""The a-vector-for-a-word page's endpoints, on the two spaces it uses.

Every number the page quotes in prose is pinned here, and the last class
asserts that no identifier reaches a reader through a refusal.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

SPACE = "/concepts/a-vector-for-a-word/space"
SKETCH = "/concepts/a-vector-for-a-word/sketch"
SPHERE = "/concepts/a-vector-for-a-word/sphere"
NEIGHBOURS = "/concepts/a-vector-for-a-word/neighbours"
ANALOGY = "/concepts/a-vector-for-a-word/analogy"
EXCLUSION = "/concepts/a-vector-for-a-word/analogy-exclusion"
ROTATION = "/concepts/a-vector-for-a-word/rotation"
SENSES = "/concepts/a-vector-for-a-word/senses"
CONTRACTS = "/concepts/a-vector-for-a-word/contracts"
TWO_FITS = "/concepts/a-vector-for-a-word/two-fits"
COMPANY = "/concepts/a-vector-for-a-word/company"


class TestDocumentsSpace:
    def test_the_shape_of_the_table(self) -> None:
        body = client.get(SPACE).json()

        assert body["n_words"] == 23
        assert body["dimension"] == 4
        assert body["table_numbers"] == 92
        assert len(body["documents"]) == 24
        assert body["n_pairs"] == 253

    def test_the_commonest_words_come_first(self) -> None:
        body = client.get(SPACE).json()
        first = body["words"][:3]

        assert [row["word"] for row in first] == ["and", "the", "we"]
        assert [row["count"] for row in first] == [8, 8, 8]
        assert [row["token_id"] for row in first] == [0, 1, 2]

    def test_the_three_commonest_words_have_the_three_shortest_vectors(self) -> None:
        body = client.get(SPACE).json()
        by_length = sorted(body["words"], key=lambda row: row["length"])

        assert [row["word"] for row in by_length[:3]] == ["the", "and", "we"]
        assert by_length[0]["length"] == pytest.approx(0.3667, abs=5e-5)
        assert by_length[1]["length"] == pytest.approx(0.3708, abs=5e-5)
        assert by_length[2]["length"] == pytest.approx(0.5770, abs=5e-5)

    def test_the_longest_vectors_belong_to_words_used_least(self) -> None:
        body = client.get(SPACE).json()
        by_length = sorted(body["words"], key=lambda row: -row["length"])

        assert {row["word"] for row in by_length[:2]} == {"deck", "dough"}
        assert by_length[0]["length"] == pytest.approx(1.1628, abs=5e-5)
        assert by_length[0]["count"] == 5

    def test_length_falls_as_a_word_is_used_more(self) -> None:
        body = client.get(SPACE).json()

        assert body["length_count_correlation"] == pytest.approx(-0.6284, abs=5e-5)

    def test_the_gap_between_two_numbers_says_almost_nothing(self) -> None:
        body = client.get(SPACE).json()

        assert body["id_gap_cosine_correlation"] == pytest.approx(-0.0852, abs=5e-5)

    def test_sail_sits_where_the_page_says(self) -> None:
        body = client.get(SPACE).json()
        sail = next(row for row in body["words"] if row["word"] == "sail")

        assert sail["token_id"] == 13
        assert sail["count"] == 6
        assert sail["vector"][0] == pytest.approx(0.5369, abs=5e-5)
        assert sail["length"] == pytest.approx(0.8225, abs=5e-5)


class TestSketchSpace:
    def test_five_words_in_two_dimensions(self) -> None:
        body = client.get(SKETCH).json()

        assert body["n_words"] == 5
        assert body["dimension"] == 2
        assert [row["word"] for row in body["words"]] == [
            "king",
            "queen",
            "man",
            "woman",
            "silent",
        ]
        assert [row["length"] for row in body["words"]] == [1.0, 2.0, 1.0, 3.0, 0.0]
        assert all(row["count"] is None for row in body["words"])


class TestNeighbours:
    def test_sail_keeps_sailing_company(self) -> None:
        body = client.post(NEIGHBOURS, json={"word": "sail", "n_results": 4}).json()

        assert [row["word"] for row in body["by_cosine"]] == [
            "rope",
            "mast",
            "deck",
            "crew",
        ]
        assert [row["cosine"] for row in body["by_cosine"]] == pytest.approx(
            [0.9825, 0.9445, 0.9421, 0.9361], abs=5e-5
        )
        assert body["comparisons"] == 22

    def test_the_two_orderings_disagree_at_the_very_top(self) -> None:
        body = client.post(NEIGHBOURS, json={"word": "sail", "n_results": 2}).json()

        assert body["by_cosine"][0]["word"] == "rope"
        assert body["by_gap"][0]["word"] == "mast"
        by_word = {row["word"]: row for row in body["by_cosine"] + body["by_gap"]}
        assert by_word["rope"]["cosine"] == pytest.approx(0.9825, abs=5e-5)
        assert by_word["rope"]["gap"] == pytest.approx(0.3046, abs=5e-5)
        assert by_word["rope"]["length"] == pytest.approx(1.0713, abs=5e-5)
        assert by_word["mast"]["cosine"] == pytest.approx(0.9445, abs=5e-5)
        assert by_word["mast"]["gap"] == pytest.approx(0.2750, abs=5e-5)
        assert by_word["mast"]["length"] == pytest.approx(0.8281, abs=5e-5)
        assert body["length"] == pytest.approx(0.8225, abs=5e-5)

    def test_a_pair_that_should_be_far_is_far(self) -> None:
        body = client.post(NEIGHBOURS, json={"word": "sail", "n_results": 22}).json()
        rows = {row["word"]: row["cosine"] for row in body["by_cosine"]}

        assert rows["flour"] == pytest.approx(0.0661, abs=5e-5)
        assert rows["oven"] == pytest.approx(-0.0518, abs=5e-5)
        assert rows["boat"] == pytest.approx(0.3638, abs=5e-5)

    def test_a_pair_the_page_calls_close_is_close(self) -> None:
        body = client.post(NEIGHBOURS, json={"word": "oven", "n_results": 22}).json()
        rows = {row["word"]: row["cosine"] for row in body["by_cosine"]}

        assert rows["bake"] == pytest.approx(0.9267, abs=5e-5)

    def test_the_sketch_orders_by_angle_and_skips_the_silent_word(self) -> None:
        body = client.post(
            NEIGHBOURS, json={"space": "sketch", "word": "king", "n_results": 5}
        ).json()

        assert [row["word"] for row in body["by_cosine"]] == ["queen", "man", "woman"]
        assert body["by_cosine"][0]["cosine"] == 1.0
        assert body["by_cosine"][0]["gap"] == 1.0

    def test_an_unknown_word_is_refused(self) -> None:
        response = client.post(NEIGHBOURS, json={"word": "kettle"})

        assert response.status_code == 422
        assert "kettle" in response.json()["detail"]

    def test_a_word_with_no_direction_is_refused(self) -> None:
        response = client.post(NEIGHBOURS, json={"space": "sketch", "word": "silent"})

        assert response.status_code == 422
        assert "no direction" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        assert (
            client.post(NEIGHBOURS, json={"word": "sail", "howMany": 3}).status_code
            == 422
        )


class TestAnalogy:
    def test_the_sketch_analogy_lands_on_queen(self) -> None:
        body = client.post(
            ANALOGY,
            json={
                "space": "sketch",
                "positive": ["king", "woman"],
                "negative": ["man"],
                "n_results": 2,
            },
        ).json()

        assert body["combination"] == pytest.approx([1.0, 0.0])
        assert body["combination_length"] == pytest.approx(1.0)
        assert body["answers"][0]["word"] == "queen"
        assert body["answers"][0]["cosine"] == pytest.approx(1.0)
        assert body["excluded"] == ["king", "woman", "man"]

    def test_the_units_are_the_intermediates_the_page_shows(self) -> None:
        body = client.post(
            ANALOGY,
            json={
                "space": "sketch",
                "positive": ["king", "woman"],
                "negative": ["man"],
            },
        ).json()
        terms = {term["word"]: term for term in body["terms"]}

        assert terms["woman"]["vector"] == pytest.approx([0.0, 3.0])
        assert terms["woman"]["length"] == pytest.approx(3.0)
        assert terms["woman"]["unit"] == pytest.approx([0.0, 1.0])
        assert terms["man"]["added"] is False

    def test_leaving_the_question_words_in_returns_king(self) -> None:
        body = client.post(
            ANALOGY,
            json={
                "space": "sketch",
                "positive": ["king", "woman"],
                "negative": ["man"],
                "exclude_question_words": False,
                "n_results": 2,
            },
        ).json()

        assert [row["word"] for row in body["answers"]] == ["king", "queen"]
        assert body["excluded"] == []

    def test_the_documents_analogy_does_not_recover_the_matching_word(self) -> None:
        body = client.post(
            ANALOGY,
            json={
                "space": "documents",
                "positive": ["sail", "sugar"],
                "negative": ["flour"],
                "n_results": 3,
            },
        ).json()

        assert [row["word"] for row in body["answers"]] == ["rope", "deck", "mast"]
        assert body["answers"][0]["cosine"] == pytest.approx(0.7847, abs=5e-5)

    def test_a_question_that_cancels_exactly_is_refused(self) -> None:
        response = client.post(
            ANALOGY,
            json={"space": "sketch", "positive": ["king"], "negative": ["queen"]},
        )

        assert response.status_code == 422
        assert "no direction" in response.json()["detail"]

    def test_a_question_with_nothing_added_is_refused(self) -> None:
        assert (
            client.post(
                ANALOGY, json={"space": "sketch", "positive": [], "negative": ["king"]}
            ).status_code
            == 422
        )


class TestExclusionSweep:
    def test_every_three_word_question_is_asked(self) -> None:
        body = client.get(EXCLUSION).json()

        assert body["n_questions"] == 10626
        assert body["n_returning_a_question_word"] == 4118
        assert body["share"] == pytest.approx(0.3875, abs=5e-5)

    def test_it_is_almost_always_one_of_the_added_words(self) -> None:
        body = client.get(EXCLUSION).json()

        assert body["n_added_word"] == 4048
        assert body["n_subtracted_word"] == 70

    def test_the_examples_are_asked_both_ways(self) -> None:
        body = client.get(EXCLUSION).json()
        first = body["examples"][0]

        assert first["positive_first"] == "sail"
        assert first["negative"] == "boat"
        assert first["positive_second"] == "flour"
        assert first["with_exclusion"] != first["without_exclusion"]


class TestRotation:
    @pytest.mark.parametrize("angle", [30.0, 45.0, 90.0, 217.0])
    def test_the_angles_and_the_lengths_survive_the_turn(self, angle: float) -> None:
        body = client.post(ROTATION, json={"angle_degrees": angle}).json()

        assert body["worst_cosine_change"] < 1e-14
        assert body["worst_length_change"] < 1e-14
        assert body["n_top_unchanged"] == body["n_words"] == 23

    def test_the_only_rankings_that_move_are_exact_ties(self) -> None:
        """Twenty-one whole rankings survive; the exceptions swap two words
        whose cosines agree to sixteen digits, which the change bound proves."""
        body = client.post(ROTATION, json={"angle_degrees": 30.0}).json()

        assert body["n_orders_unchanged"] == 21
        assert body["n_top_unchanged"] == 23
        assert body["worst_cosine_change"] < 1e-15

    def test_a_coordinate_moves_a_long_way(self) -> None:
        body = client.post(
            ROTATION, json={"angle_degrees": 90.0, "word": "sail"}
        ).json()

        assert body["watched"]["before"][0] == pytest.approx(0.5369, abs=5e-5)
        assert body["watched"]["after"][0] == pytest.approx(-0.5223, abs=5e-5)
        assert body["watched"]["length_before"] == pytest.approx(
            body["watched"]["length_after"]
        )
        assert [row["word"] for row in body["neighbours_before"]] == [
            row["word"] for row in body["neighbours_after"]
        ]

    def test_at_forty_five_degrees_the_first_coordinate_nearly_vanishes(self) -> None:
        body = client.post(
            ROTATION, json={"angle_degrees": 45.0, "word": "sail"}
        ).json()

        assert body["watched"]["after"][0] == pytest.approx(0.0103, abs=5e-5)


class TestSenses:
    def test_the_two_sense_word_is_nearest_nothing_in_particular(self) -> None:
        body = client.get(SENSES).json()

        assert body["merged_word"] == "sheet"
        assert body["replaced"] == ["pan", "rope"]
        assert body["best_cosine"] == pytest.approx(0.8635, abs=5e-5)
        assert body["neighbours"][0]["word"] == "deck"
        assert body["neighbours"][1]["word"] == "dough"

    def test_the_two_words_it_replaced_had_much_nearer_neighbours(self) -> None:
        body = client.get(SENSES).json()
        separate = {row["word"]: row for row in body["separate"]}

        assert separate["pan"]["nearest"] == "dough"
        assert separate["pan"]["cosine"] == pytest.approx(0.9981, abs=5e-5)
        assert separate["rope"]["nearest"] == "sail"
        assert separate["rope"]["cosine"] == pytest.approx(0.9825, abs=5e-5)

    def test_it_is_pulled_towards_both_topics_at_once(self) -> None:
        body = client.get(SENSES).json()

        assert body["mean_to_cooking"] == pytest.approx(0.3504, abs=5e-5)
        assert body["mean_to_sailing"] == pytest.approx(0.4318, abs=5e-5)
        assert body["comparison_word"] == "dough"
        assert body["comparison_to_cooking"] == pytest.approx(0.4015, abs=5e-5)
        assert body["comparison_to_sailing"] == pytest.approx(0.1481, abs=5e-5)


class TestContracts:
    def test_every_edge_is_answered_or_refused(self) -> None:
        probes = client.get(CONTRACTS).json()["probes"]

        assert len(probes) == 8
        assert all(probe["outcome"] in {"answered", "refused"} for probe in probes)
        assert sum(probe["outcome"] == "refused" for probe in probes) == 6

    def test_the_silent_word_never_appears_in_a_neighbour_list(self) -> None:
        probes = {
            probe["edge"]: probe for probe in client.get(CONTRACTS).json()["probes"]
        }
        listed = probes["a word with no direction sitting in a neighbour list"]

        assert listed["outcome"] == "answered"
        assert listed["result"] == "queen, man, woman"

    def test_asking_for_too_many_neighbours_gives_what_there_is(self) -> None:
        probes = {
            probe["edge"]: probe for probe in client.get(CONTRACTS).json()["probes"]
        }

        assert (
            probes["asking for more neighbours than the vocabulary holds"]["result"]
            == "22"
        )


class TestCompany:
    def test_two_words_placed_together_never_meet(self) -> None:
        body = client.post(COMPANY, json={"first": "sail", "second": "rope"}).json()

        assert body["window"] == 2
        assert body["between"] == 0.0
        assert body["cosine"] == pytest.approx(0.9825, abs=5e-5)
        assert body["shared"] == ["and", "crew", "deck", "the"]

    def test_two_words_that_do_meet_are_placed_apart(self) -> None:
        body = client.post(COMPANY, json={"first": "sail", "second": "boat"}).json()

        assert body["between"] == 2.0
        assert body["cosine"] == pytest.approx(0.3638, abs=5e-5)
        assert body["shared"] == ["and", "the", "wind"]

    def test_the_company_is_reported_heaviest_first(self) -> None:
        body = client.post(COMPANY, json={"first": "sail", "second": "boat"}).json()

        assert [entry["word"] for entry in body["first_company"][:3]] == [
            "wind",
            "crew",
            "and",
        ]
        assert body["first_company"][0]["weight"] == 5.0
        assert [entry["word"] for entry in body["second_company"][:3]] == [
            "harbour",
            "wind",
            "anchor",
        ]
        assert body["second_company"][0]["weight"] == 6.0

    def test_the_company_that_explains_the_miss(self) -> None:
        body = client.post(COMPANY, json={"first": "sail", "second": "boat"}).json()
        boat = {entry["word"]: entry["weight"] for entry in body["second_company"]}
        sail = {entry["word"]: entry["weight"] for entry in body["first_company"]}

        assert boat["harbour"] == 6.0
        assert boat["anchor"] == 3.0
        assert "harbour" not in sail
        assert "anchor" not in sail

    def test_an_unknown_word_is_refused(self) -> None:
        assert (
            client.post(COMPANY, json={"first": "kettle", "second": "sail"}).status_code
            == 422
        )


class TestTwoFits:
    def test_both_fits_put_a_topic_nearer_itself(self) -> None:
        body = client.get(TWO_FITS).json()
        first, second = body["fits"]

        assert first["within_topic"] == pytest.approx(0.8333, abs=5e-5)
        assert first["across_topic"] == pytest.approx(0.7313, abs=5e-5)
        assert second["within_topic"] == pytest.approx(0.8676, abs=5e-5)
        assert second["across_topic"] == pytest.approx(0.5593, abs=5e-5)
        for fit in body["fits"]:
            assert fit["within_topic"] > fit["across_topic"]

    def test_the_two_fits_share_no_coordinates(self) -> None:
        body = client.get(TWO_FITS).json()

        assert body["watched"] == "sail"
        assert body["self_cosine"] == pytest.approx(0.4954, abs=5e-5)
        assert body["mean_first_coordinate_gap"] == pytest.approx(0.3292, abs=5e-5)
        assert body["largest_first_coordinate_gap"] == pytest.approx(0.8760, abs=5e-5)

    def test_they_do_not_even_agree_on_the_neighbours(self) -> None:
        body = client.get(TWO_FITS).json()

        assert body["shared_neighbour_places"] == 28
        assert body["neighbour_places"] == 69
        assert [row["word"] for row in body["fits"][0]["watched_neighbours"]] == [
            "deck",
            "crew",
            "pan",
        ]
        assert [row["word"] for row in body["fits"][1]["watched_neighbours"]] == [
            "crew",
            "wind",
            "boat",
        ]


IDENTIFIER = re.compile(r"[a-zA-Z]+_[a-zA-Z_]+|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
"""A snake_case parameter or a CamelCase class name, either of which is a leak."""


class TestNothingNamesTheCodeBehindThePage:
    def test_no_refusal_carries_an_identifier(self) -> None:
        probes = client.get(CONTRACTS).json()["probes"]
        wording = " ".join(
            part
            for probe in probes
            for part in (probe["edge"], probe["reason"] or "", probe["result"] or "")
        )

        assert IDENTIFIER.search(wording) is None, wording

    def test_the_refused_neighbour_count_is_rewritten(self) -> None:
        probes = {
            probe["edge"]: probe for probe in client.get(CONTRACTS).json()["probes"]
        }
        reason = probes["asking for no nearest words at all"]["reason"]

        assert "n_results" not in reason
        assert reason == (
            "a list of 0 nearest words is not a list, so at least one must be asked for"
        )

    def test_every_endpoint_refusal_reads_as_english(self) -> None:
        details = [
            client.post(NEIGHBOURS, json={"word": "kettle"}).json()["detail"],
            client.post(NEIGHBOURS, json={"space": "sketch", "word": "silent"}).json()[
                "detail"
            ],
            client.post(
                ANALOGY,
                json={"space": "sketch", "positive": ["king"], "negative": ["queen"]},
            ).json()["detail"],
        ]

        for detail in details:
            assert IDENTIFIER.search(detail) is None, detail


class TestSphere:
    """The space drawn as directions, which is what a cosine actually compares."""

    def test_every_word_of_the_table_is_placed(self):
        answer = client.get(SPHERE).json()

        assert answer["n_drawn"] == 23
        assert len(answer["words"]) == 23

    def test_every_point_is_on_the_unit_sphere(self):
        answer = client.get(SPHERE).json()

        for word in answer["words"]:
            radius = (word["x"] ** 2 + word["y"] ** 2 + word["z"] ** 2) ** 0.5
            assert radius == pytest.approx(1.0, abs=1e-12)

    def test_words_of_one_subject_are_far_more_alike_than_words_across_two(self):
        """The clustering claim, measured in the full space rather than drawn."""
        answer = client.get(SPHERE).json()

        assert answer["within_topic"] > 0.5
        assert answer["across_topic"] < 0.1

    def test_the_drawing_says_how_much_of_the_spread_it_keeps(self):
        answer = client.get(SPHERE).json()

        assert 0.0 < answer["kept_share"] <= 1.0
        assert answer["dimension"] == 4

    def test_each_word_is_named_by_the_subject_it_belongs_to(self):
        answer = client.get(SPHERE).json()
        topics = {word["word"]: word["topic"] for word in answer["words"]}

        assert topics["flour"] == "cooking"
        assert topics["anchor"] == "sailing"
        assert topics["the"] == "shared"
