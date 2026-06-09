"""The latent semantic analysis page's endpoints, on the two corpora it carries.

Every number the page prints is pinned here. The four-document corpus has
singular values that are exactly ``sqrt(7)``, ``sqrt(3)``, 1 and 1, so its whole
fit can be asserted against arithmetic rather than against itself; the
twenty-four document one is where the claims about the leading direction and the
squeeze live.

The load-bearing tests of the module are
``TestTheLeadingDirection::test_the_two_halves_have_the_same_mean_on_it`` and
``TestNothingIdentifierShapedReachesAReader``. The first is the page's headline
claim, and it is asserted as an equality to rounding rather than as a
similarity; the second holds every string a widget can render to the rule that a
reader never meets a class, a parameter or a module.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.latent_semantic_analysis import (
    COOKING_DOCUMENTS,
    FOUR_DOCUMENTS,
    SAILING_DOCUMENTS,
    SHARED_WORDS,
    TWENTY_FOUR_DOCUMENTS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/latent-semantic-analysis"
TABLE = f"{BASE}/table"
FIT = f"{BASE}/fit"
RECONSTRUCTION = f"{BASE}/reconstruction"
SHARPNESS = f"{BASE}/sharpness"
DIRECTIONS = f"{BASE}/directions"
SENSES = f"{BASE}/senses"

SQRT_SEVEN = math.sqrt(7.0)
SQRT_THREE = math.sqrt(3.0)


class TestTheCorpora:
    def test_they_are_the_corpora_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        spec = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.counts.fixtures"
        )

        assert list(FOUR_DOCUMENTS) == spec.TINY_LSA_CORPUS
        assert list(COOKING_DOCUMENTS) == spec.COOKING_DOCUMENTS
        assert list(SAILING_DOCUMENTS) == spec.SAILING_DOCUMENTS
        assert list(SHARED_WORDS) == spec.SHARED_WORDS
        assert TWENTY_FOUR_DOCUMENTS == spec.TWO_TOPIC_CORPUS

    def test_the_larger_corpus_is_twenty_three_words_by_twenty_four_documents(
        self,
    ) -> None:
        body = client.post(TABLE, json={"corpus": "twenty-four"}).json()

        assert body["n_terms"] == 23
        assert body["n_documents"] == 24
        assert body["n_cells"] == 552
        assert body["n_nonzero"] == 144
        assert body["zero_share"] == pytest.approx(0.7391, abs=5e-5)


class TestTheFourDocumentTable:
    """Seven words over four documents, worked in the page's own prose."""

    def test_the_row_of_the_word_in_every_document_is_all_ones(self) -> None:
        body = client.post(TABLE, json={"corpus": "four"}).json()
        rows = {row["word"]: row for row in body["rows"]}

        assert body["words"] == [
            "the",
            "boat",
            "cat",
            "ran",
            "sailed",
            "sank",
            "sat",
        ]
        assert rows["the"]["counts"] == [1.0, 1.0, 1.0, 1.0]
        assert rows["the"]["document_frequency"] == 4
        assert body["everywhere_words"] == ["the"]

    def test_the_weights_reward_a_word_for_being_rare(self) -> None:
        body = client.post(TABLE, json={"corpus": "four"}).json()
        rows = {row["word"]: row for row in body["rows"]}

        assert rows["the"]["weight"] == pytest.approx(1.0)
        assert rows["cat"]["weight"] == pytest.approx(1.5108, abs=5e-5)
        assert rows["sat"]["weight"] == pytest.approx(1.9163, abs=5e-5)
        assert rows["the"]["weighted"] == [1.0, 1.0, 1.0, 1.0]
        assert rows["sat"]["weighted"][0] == pytest.approx(1.9163, abs=5e-5)

    def test_the_share_of_a_column_the_everywhere_word_takes_falls(self) -> None:
        body = client.post(TABLE, json={"corpus": "four"}).json()

        assert body["count_share_of_a_column"] == pytest.approx(1 / 3)
        assert body["weighted_share_of_a_column"] == pytest.approx(0.2259, abs=5e-5)

    def test_without_the_smoothing_the_everywhere_word_would_weigh_nothing(
        self,
    ) -> None:
        body = client.post(TABLE, json={"corpus": "four"}).json()

        assert body["unsmoothed_weight_of_the_commonest"] == 0.0
        assert body["smallest_weight"] == 1.0

    def test_the_larger_corpus_has_no_word_in_every_document(self) -> None:
        body = client.post(TABLE, json={"corpus": "twenty-four"}).json()

        assert body["everywhere_words"] == []
        assert body["count_share_of_a_column"] == pytest.approx(1 / 6)
        assert body["weighted_share_of_a_column"] == pytest.approx(0.1573, abs=5e-5)


class TestTheFourDocumentFit:
    def test_the_singular_values_are_the_roots_of_seven_three_one_one(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "cat",
            },
        ).json()
        values = [component["singular_value"] for component in body["components"]]

        assert values == pytest.approx([SQRT_SEVEN, SQRT_THREE, 1.0, 1.0])

    def test_the_shares_are_seven_three_one_and_one_twelfths(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "cat",
            },
        ).json()
        shares = [component["share"] for component in body["components"]]

        assert shares == pytest.approx([7 / 12, 3 / 12, 1 / 12, 1 / 12])
        assert body["kept_share"] == pytest.approx(10 / 12)

    def test_the_word_positions_are_whole_and_half_numbers(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "cat",
            },
        ).json()
        points = {point["word"]: point["coordinates"] for point in body["word_points"]}
        sign = 1.0 if points["cat"][1] > 0 else -1.0

        assert points["the"] == pytest.approx([2.0, 0.0], abs=1e-12)
        assert points["cat"] == pytest.approx([1.0, sign * 1.0])
        assert points["boat"] == pytest.approx([1.0, -sign * 1.0])
        assert points["sat"] == pytest.approx([0.5, sign * 0.5])
        assert points["sank"] == pytest.approx([0.5, -sign * 0.5])

    def test_every_document_sits_at_the_same_place_on_the_leading_direction(
        self,
    ) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "cat",
            },
        ).json()
        first = [point["coordinates"][0] for point in body["document_points"]]

        assert first == pytest.approx([SQRT_SEVEN / 2] * 4)
        assert body["components"][0]["all_one_sign"] is True
        assert body["components"][0]["group_difference"] == pytest.approx(
            0.0, abs=1e-12
        )

    def test_the_second_direction_splits_the_two_pairs(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "cat",
            },
        ).json()
        second = [point["coordinates"][1] for point in body["document_points"]]

        assert abs(second[0]) == pytest.approx(SQRT_THREE / 2)
        assert second[0] == pytest.approx(second[1])
        assert second[2] == pytest.approx(second[3])
        assert second[0] == pytest.approx(-second[2])
        assert body["components"][1]["all_one_sign"] is False

    def test_two_words_that_never_shared_a_document_come_out_together(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 2,
                "word": "sat",
                "n_results": 3,
            },
        ).json()
        nearest = {row["word"]: row["similarity"] for row in body["nearest_words"]}

        assert nearest["ran"] == pytest.approx(1.0)
        assert nearest["the"] == pytest.approx(2**-0.5)

    def test_the_third_and_fourth_directions_are_exactly_tied(self) -> None:
        """Two equal singular values leave their plane fixed and the pair of
        directions inside it undetermined."""
        body = client.post(
            FIT,
            json={
                "corpus": "four",
                "weighting": "counts",
                "dimension": 4,
                "word": "cat",
            },
        ).json()
        values = [component["singular_value"] for component in body["components"]]

        assert values[2] == values[3]


class TestTheLeadingDirection:
    def test_the_two_halves_have_the_same_mean_on_it(self) -> None:
        """A table of counts is never negative, so its leading direction is one
        sign throughout and cannot tell one half from the other."""
        body = client.get(DIRECTIONS).json()

        assert body["as_counted_first_difference"] == pytest.approx(0.0, abs=1e-12)
        assert all(value > 0.0 for value in body["as_counted_first"])
        assert body["smallest_first_coordinate"] == pytest.approx(2.6501, abs=5e-5)
        assert body["largest_first_coordinate"] == pytest.approx(2.8994, abs=5e-5)

    def test_the_direction_after_it_separates_them_completely(self) -> None:
        body = client.get(DIRECTIONS).json()
        second = body["as_counted_second"]

        assert all(value < 0.0 for value in second[:12])
        assert all(value > 0.0 for value in second[12:])
        assert body["as_counted_second_difference"] == pytest.approx(-5.0777, abs=5e-5)

    def test_the_words_both_halves_use_sit_at_nothing_on_it(self) -> None:
        body = client.get(DIRECTIONS).json()

        assert body["shared_words"] == ["and", "the", "we"]
        for value in body["shared_word_second_coordinates"]:
            assert value == pytest.approx(0.0, abs=1e-12)

    def test_subtracting_each_words_mean_first_moves_the_split_to_the_front(
        self,
    ) -> None:
        body = client.get(DIRECTIONS).json()
        first = body["after_shifting_first"]

        assert all(value > 0.0 for value in first[:12])
        assert all(value < 0.0 for value in first[12:])
        assert body["after_shifting_difference"] == pytest.approx(5.0777, abs=5e-5)

    def test_the_two_leading_shares(self) -> None:
        body = client.get(DIRECTIONS).json()

        assert body["as_counted_shares"][:2] == pytest.approx(
            [0.2628, 0.2171], abs=5e-5
        )
        assert body["after_shifting_shares"][0] == pytest.approx(0.2945, abs=5e-5)


class TestWhatTheSqueezeKeeps:
    def test_the_error_is_exactly_the_energy_thrown_away(self) -> None:
        """Eckart and Young: no other table of that rank is closer."""
        for dimension in (2, 4, 8):
            body = client.post(RECONSTRUCTION, json={"dimension": dimension}).json()

            assert body["squared_error"] == pytest.approx(body["discarded_energy"])

    def test_at_two_directions_the_kept_share_and_the_error(self) -> None:
        body = client.post(RECONSTRUCTION, json={"dimension": 2}).json()

        assert body["kept_share"] == pytest.approx(0.4799, abs=5e-5)
        assert body["squared_error"] == pytest.approx(371.3389, abs=5e-4)

    def test_a_word_seen_in_six_documents_is_rebuilt_across_all_twelve(self) -> None:
        body = client.post(
            RECONSTRUCTION, json={"dimension": 2, "word": "flour"}
        ).json()

        assert body["row_seen_in"] == 6
        assert body["row_filled_in"] == 12
        assert body["row_original"][1] == 0.0
        assert body["row_rebuilt"][1] == pytest.approx(1.1848, abs=5e-5)
        assert max(body["row_rebuilt"][12:]) < 0.02

    def test_cells_the_corpus_left_empty_come_back_holding_something(self) -> None:
        body = client.post(RECONSTRUCTION, json={"dimension": 2}).json()

        assert body["empty_cell_mean"] == pytest.approx(0.4043, abs=5e-5)
        assert body["empty_cell_largest"] == pytest.approx(1.2970, abs=5e-5)

    def test_a_rebuilt_count_can_be_below_zero(self) -> None:
        narrow = client.post(RECONSTRUCTION, json={"dimension": 2}).json()
        wider = client.post(RECONSTRUCTION, json={"dimension": 8}).json()

        assert narrow["n_negative"] == 104
        assert narrow["most_negative"] == pytest.approx(-0.0149, abs=5e-5)
        assert wider["n_negative"] == 208
        assert wider["most_negative"] == pytest.approx(-0.3643, abs=5e-5)

    def test_the_spectrum_runs_out_before_the_directions_do(self) -> None:
        body = client.post(FIT, json={"dimension": 2}).json()
        components = body["components"]

        assert body["rank"] == 14
        assert len(components) == 23
        assert components[13]["cumulative"] == pytest.approx(1.0)
        assert components[14]["singular_value"] == pytest.approx(0.0, abs=1e-9)


class TestWhereTheRawTableFails:
    def test_two_documents_about_one_thing_in_other_words_score_nothing(self) -> None:
        body = client.get(SHARPNESS).json()
        pair = body["document_pairs"][0]

        assert pair["first"] == "flour sugar and butter eggs oven"
        assert pair["second"] == "we bake stir whisk dough pan"
        assert pair["raw"] == 0.0
        assert pair["narrow"] == pytest.approx(0.9998, abs=5e-5)

    def test_the_raw_table_rates_a_document_of_the_other_half_higher(self) -> None:
        body = client.get(SHARPNESS).json()
        crossing = body["crossing_pair"]

        assert crossing["raw"] == pytest.approx(0.1483, abs=5e-5)
        assert crossing["raw"] > body["document_pairs"][0]["raw"]
        assert crossing["narrow"] < body["document_pairs"][0]["narrow"]

    def test_five_pairs_of_one_half_never_met_and_all_read_as_zero(self) -> None:
        body = client.get(SHARPNESS).json()
        pairs = {(row["first"], row["second"]): row for row in body["word_pairs"]}

        assert ("flour", "bake") in pairs
        assert pairs[("flour", "bake")]["raw"] == 0.0
        assert pairs[("flour", "anchor")]["raw"] == 0.0
        assert pairs[("flour", "bake")]["narrow"] == pytest.approx(1.0, abs=5e-5)
        assert pairs[("flour", "anchor")]["narrow"] == pytest.approx(0.0, abs=5e-3)

    def test_how_many_pairs_the_raw_table_cannot_place(self) -> None:
        body = client.get(SHARPNESS).json()

        assert body["n_word_pairs"] == 90
        assert body["n_word_pairs_at_zero"] == 10
        assert body["n_document_pairs"] == 132
        assert body["n_document_pairs_at_zero"] == 14
        assert body["n_across_document_pairs_at_zero"] == 96

    def test_the_narrowest_fit_loses_every_distinction_inside_a_half(self) -> None:
        body = client.get(SHARPNESS).json()
        rows = {row["label"]: row for row in body["widths"]}

        assert rows["the table itself"]["within_smallest"] == 0.0
        assert rows["the table itself"]["within_largest"] == pytest.approx(
            6 / 7, abs=5e-5
        )
        assert rows["2 directions kept"]["within_smallest"] > 0.999
        assert rows["6 directions kept"]["within_smallest"] == pytest.approx(
            0.0727, abs=5e-4
        )
        assert rows["6 directions kept"]["within_largest"] == pytest.approx(
            0.9553, abs=5e-4
        )

    def test_keeping_every_direction_reproduces_the_raw_table_exactly(self) -> None:
        body = client.get(SHARPNESS).json()
        rows = {row["label"]: row for row in body["widths"]}

        assert rows["23 directions kept"]["within_mean"] == pytest.approx(
            rows["the table itself"]["within_mean"], abs=1e-9
        )
        assert rows["23 directions kept"]["kept_share"] == pytest.approx(1.0)


class TestWhatOneFitAnswers:
    def test_the_nearest_words_to_one_word_are_all_of_its_own_half(self) -> None:
        body = client.post(FIT, json={"word": "flour", "n_results": 9}).json()

        assert body["nearest_words"][0]["similarity"] == pytest.approx(1.0, abs=5e-5)
        assert all(row["group"] == "cooking" for row in body["nearest_words"])
        assert body["within_similarity"] == pytest.approx(1.0, abs=5e-5)
        assert body["across_similarity"] == pytest.approx(0.0021, abs=5e-4)

    def test_at_six_directions_the_words_of_one_half_come_apart_again(self) -> None:
        body = client.post(
            FIT, json={"dimension": 6, "word": "flour", "n_results": 3}
        ).json()
        nearest = {row["word"]: row["similarity"] for row in body["nearest_words"]}

        assert nearest["sugar"] == pytest.approx(0.8777, abs=5e-4)
        assert nearest["pan"] == pytest.approx(0.8561, abs=5e-4)
        assert body["within_similarity"] == pytest.approx(0.4945, abs=5e-4)

    def test_the_same_fit_answers_which_documents_are_nearest(self) -> None:
        body = client.post(
            FIT, json={"dimension": 6, "document": 0, "n_results": 3}
        ).json()
        nearest = body["nearest_documents"]

        assert nearest[0]["text"] == "flour sugar butter the eggs oven"
        assert nearest[0]["similarity"] == pytest.approx(0.9963, abs=5e-4)
        assert nearest[1]["text"] == "pan flour and sugar butter eggs"
        assert nearest[1]["similarity"] == pytest.approx(0.9161, abs=5e-4)
        assert all(row["group"] == "cooking" for row in nearest)

    def test_two_directions_cannot_order_the_documents_at_all(self) -> None:
        body = client.post(FIT, json={"dimension": 2, "document": 0}).json()

        for row in body["nearest_documents"][:5]:
            assert row["similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_what_the_squeeze_costs_and_what_it_saves(self) -> None:
        body = client.post(FIT, json={"dimension": 2}).json()

        assert body["numbers_in_the_table"] == 552
        assert body["numbers_kept"] == 96
        assert body["seconds"] < 1.0

    def test_a_new_text_is_placed_on_the_side_its_words_belong_to(self) -> None:
        cooking = client.post(
            FIT, json={"new_text": "we bake the dough and whisk the eggs"}
        ).json()
        sailing = client.post(
            FIT, json={"new_text": "the crew sail the boat past the harbour"}
        ).json()

        assert cooking["placed_group"] == "cooking"
        assert cooking["placed"][1] == pytest.approx(-1.9689, abs=5e-4)
        assert sailing["placed_group"] == "sailing"
        assert sailing["placed"][1] == pytest.approx(2.0586, abs=5e-4)

    def test_a_text_of_words_the_corpus_never_had_is_placed_nowhere(self) -> None:
        body = client.post(FIT, json={"new_text": "a quiet afternoon"}).json()

        assert body["placed"] == [0.0, 0.0]
        assert body["placed_group"] is None


class TestOneSpellingTwoSenses:
    def test_it_lands_at_nothing_on_the_direction_that_separates(self) -> None:
        body = client.get(SENSES).json()

        assert body["word"] == "roll"
        assert body["second_coordinate"] == pytest.approx(0.0, abs=1e-12)
        assert abs(body["cooking_word_second_coordinate"]) > 2.0
        assert abs(body["sailing_word_second_coordinate"]) > 2.0

    def test_its_nearest_words_are_the_ones_that_mean_nothing(self) -> None:
        body = client.get(SENSES).json()
        nearest = body["nearest"][:3]

        assert [row["word"] for row in nearest] == ["the", "and", "we"]
        assert nearest[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_it_is_the_same_distance_from_both_senses(self) -> None:
        body = client.get(SENSES).json()

        assert body["similarity_to_cooking_word"] == pytest.approx(0.7219, abs=5e-4)
        assert body["similarity_to_sailing_word"] == pytest.approx(0.7050, abs=5e-4)


IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|\b[A-Z][a-z]+[A-Z]|\boop_ml\b")
"""Anything shaped like a parameter, a class or a module path."""


class TestNothingIdentifierShapedReachesAReader:
    """Every refusal a widget can provoke, read for the rule the page keeps."""

    @pytest.mark.parametrize(
        "request_body",
        [
            {"corpus": "four", "dimension": 5},
            {"corpus": "four", "document": 9},
            {"word": "zebra"},
            {"corpus": "four", "word": "harbour"},
        ],
    )
    def test_a_refusal_is_written_in_the_pages_own_words(
        self, request_body: dict[str, object]
    ) -> None:
        response = client.post(FIT, json=request_body)
        detail = response.json()["detail"]

        assert response.status_code == 422
        assert not IDENTIFIER.search(detail), detail

    def test_a_width_the_corpus_has_no_directions_for_names_the_corpus(self) -> None:
        response = client.post(FIT, json={"corpus": "four", "dimension": 5})

        assert response.json()["detail"] == (
            "this corpus of 7 words and 4 documents has 4 directions in all, so "
            "5 of them cannot be kept"
        )

    def test_a_row_of_a_word_the_corpus_never_had_is_refused_the_same_way(
        self,
    ) -> None:
        response = client.post(RECONSTRUCTION, json={"word": "zebra"})

        assert response.status_code == 422
        assert not IDENTIFIER.search(response.json()["detail"])

    def test_an_unknown_field_is_refused_before_anything_is_computed(self) -> None:
        assert client.post(FIT, json={"nonsense": 1}).status_code == 422
        assert client.post(TABLE, json={"nonsense": 1}).status_code == 422
        assert client.post(RECONSTRUCTION, json={"nonsense": 1}).status_code == 422
