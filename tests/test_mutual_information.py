"""The pointwise mutual information page's endpoints, and the numbers it quotes.

Two things are asserted here that the page depends on and could not otherwise
trust. The first is that the unclipped score the module computes from the
library's own totals, once shifted and clipped, is the library's own matrix
entry for entry, so the page's whole argument about the negative half is about
clipping rather than about two implementations. The second is that no string
any widget renders names a class, a parameter or a module.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.embeddings.counts.pointwise_mutual_information import (
    PointwiseMutualInformationMatrix,
)

from oop_ml_api.concepts.mutual_information import (
    COOKING,
    PAGE_WINDOW,
    SAILING,
    THREE_SENTENCES,
    TWENTY_FOUR,
    _cooccurrence,
    _scores,
)
from oop_ml_api.main import app

client = TestClient(app)

BASE = "/concepts/pointwise-mutual-information"


def cell(response: dict, word: str, context: str) -> dict:
    for pair in response["pairs"]:
        if pair["word"] == word and pair["context"] == context:
            return pair
    raise AssertionError(f"{word} beside {context} was not counted")


# --- The corpora are the library's own fixtures ------------------------------


class TestTheCorpora:
    def test_the_three_sentences_are_the_ones_the_spec_pins(self):
        assert list(THREE_SENTENCES) == ["the cat sat", "the dog sat", "the cat ran"]

    def test_the_two_topics_are_twelve_documents_each(self):
        assert len(COOKING) == 12
        assert len(SAILING) == 12
        assert len(TWENTY_FOUR) == 24

    def test_no_topic_word_appears_in_both_halves(self):
        cooking = {word for text in COOKING for word in text.split()}
        sailing = {word for text in SAILING for word in text.split()}

        assert cooking & sailing == {"and", "the", "we"}


# --- The unclipped score agrees with the library once it is clipped ----------


class TestTheUnclippedScore:
    @pytest.mark.parametrize("corpus", ["three-sentences", "twenty-four-documents"])
    @pytest.mark.parametrize("smoothing", [1.0, 0.75, 0.5])
    @pytest.mark.parametrize("shift", [0.0, 0.5])
    def test_clipping_it_reproduces_the_library_entry_for_entry(
        self, corpus: str, smoothing: float, shift: float
    ):
        cooccurrence = _cooccurrence(corpus, 2)
        scores = _scores(cooccurrence, smoothing)
        expected = PointwiseMutualInformationMatrix.from_cooccurrence(
            cooccurrence, smoothing, shift
        )
        clipped = np.maximum(np.where(scores.observed, scores.values, 0.0) - shift, 0.0)

        assert np.allclose(clipped, np.asarray(expected), atol=1e-12)

    def test_the_unclipped_score_carries_a_negative_half_the_library_cannot(self):
        cooccurrence = _cooccurrence("twenty-four-documents", PAGE_WINDOW)
        scores = _scores(cooccurrence, 1.0)

        assert int((scores.observed & (scores.values < 0.0)).sum()) == 56
        assert (
            PointwiseMutualInformationMatrix.from_cooccurrence(
                cooccurrence, 1.0, 0.0
            ).n_positive
            == 224
        )


# --- The three sentences, worked by hand -------------------------------------


class TestTheThreeSentences:
    @pytest.fixture(scope="class")
    def payload(self) -> dict:
        response = client.post(
            f"{BASE}/table",
            json={"corpus": "three-sentences", "window": 1, "context_smoothing": 1.0},
        )
        assert response.status_code == 200
        return response.json()

    def test_the_words_are_commonest_first(self, payload: dict):
        assert payload["words"] == ["the", "cat", "sat", "dog", "ran"]

    def test_the_counts_and_the_totals_are_as_worked(self, payload: dict):
        assert payload["total"] == 12.0
        assert payload["word_totals"] == [3.0, 4.0, 2.0, 2.0, 1.0]
        assert payload["context_totals"] == [3.0, 4.0, 2.0, 2.0, 1.0]
        assert cell(payload, "the", "cat")["count"] == 2.0
        assert cell(payload, "the", "dog")["count"] == 1.0

    def test_the_score_of_the_beside_cat_is_the_logarithm_of_two(self, payload: dict):
        pair = cell(payload, "the", "cat")

        assert pair["joint_rate"] == pytest.approx(2.0 / 12.0)
        assert pair["word_rate"] == pytest.approx(0.25)
        assert pair["context_rate"] == pytest.approx(1.0 / 3.0)
        assert pair["expected_rate"] == pytest.approx(1.0 / 12.0)
        assert pair["ratio"] == pytest.approx(2.0)
        assert pair["score"] == pytest.approx(0.693147, abs=1e-6)

    def test_a_pair_seen_once_outscores_a_pair_seen_twice(self, payload: dict):
        assert cell(payload, "cat", "ran")["count"] == 1.0
        assert cell(payload, "cat", "ran")["score"] == pytest.approx(1.098612, abs=1e-6)
        assert cell(payload, "the", "cat")["count"] == 2.0
        assert cell(payload, "the", "cat")["score"] == pytest.approx(0.693147, abs=1e-6)

    def test_cat_beside_ran_sits_exactly_on_its_ceiling(self, payload: dict):
        pair = cell(payload, "cat", "ran")

        assert pair["ceiling"] == pytest.approx(1.098612, abs=1e-6)
        assert pair["score"] == pytest.approx(pair["ceiling"], abs=1e-12)

    def test_cat_beside_sat_does_not_reach_its_ceiling(self, payload: dict):
        pair = cell(payload, "cat", "sat")

        assert pair["score"] == pytest.approx(0.405465, abs=1e-6)
        assert pair["ceiling"] == pytest.approx(1.098612, abs=1e-6)

    def test_fifteen_of_the_twenty_five_cells_have_no_score_at_all(self, payload: dict):
        assert payload["n_cells"] == 25
        assert payload["n_observed"] == 10
        assert payload["n_undefined"] == 15
        assert payload["n_below_chance"] == 0
        assert payload["n_above_chance"] == 10
        assert payload["scores"][0][2] is None

    def test_the_matrix_is_symmetric_where_nothing_is_smoothed(self, payload: dict):
        scores = payload["scores"]
        for row, first in enumerate(payload["words"]):
            for column, _ in enumerate(payload["words"]):
                assert scores[row][column] == scores[column][row], first


# --- The twenty-four documents -----------------------------------------------


class TestTheTwentyFourDocuments:
    @pytest.fixture(scope="class")
    def payload(self) -> dict:
        response = client.post(
            f"{BASE}/table",
            json={
                "corpus": "twenty-four-documents",
                "window": PAGE_WINDOW,
                "context_smoothing": 1.0,
            },
        )
        assert response.status_code == 200
        return response.json()

    def test_the_corpus_has_twenty_three_words(self, payload: dict):
        assert payload["n_texts"] == 24
        assert payload["n_occurrences"] == 144
        assert len(payload["words"]) == 23
        assert payload["n_cells"] == 529
        assert payload["total"] == 720.0

    def test_the_census_of_the_cells(self, payload: dict):
        assert payload["n_observed"] == 280
        assert payload["n_undefined"] == 249
        assert payload["n_above_chance"] == 224
        assert payload["n_below_chance"] == 56

    def test_the_three_pairs_the_page_names(self, payload: dict):
        high = cell(payload, "crew", "deck")
        chance = cell(payload, "and", "anchor")
        low = cell(payload, "and", "bake")

        assert high["count"] == 4.0
        assert high["score"] == pytest.approx(1.527794, abs=1e-6)
        assert chance["count"] == 2.0
        assert chance["score"] == pytest.approx(0.028171, abs=1e-6)
        assert low["count"] == 1.0
        assert low["score"] == pytest.approx(-0.510826, abs=1e-6)

    def test_a_pair_seen_four_times_outscores_one_seen_six_times(self, payload: dict):
        four = cell(payload, "stir", "whisk")
        six = cell(payload, "eggs", "oven")

        assert four["count"] == 4.0
        assert six["count"] == 6.0
        assert four["score"] == pytest.approx(1.527794, abs=1e-6)
        assert six["score"] == pytest.approx(1.260315, abs=1e-6)
        assert four["score"] > six["score"]

    def test_twelve_of_flours_twenty_three_numbers_carry_nothing(self, payload: dict):
        row = payload["scores"][payload["words"].index("flour")]

        assert len(row) == 23
        assert sum(1 for value in row if value is None) == 12
        assert row[payload["words"].index("bake")] is None

    def test_six_pairs_share_the_highest_score(self, payload: dict):
        highest = max(pair["score"] for pair in payload["pairs"])
        tied = [
            pair for pair in payload["pairs"] if abs(pair["score"] - highest) < 1e-9
        ]

        assert len(tied) == 12
        assert all(pair["count"] == 4.0 for pair in tied)

    def test_every_pair_at_the_lowest_score_was_seen_once(self, payload: dict):
        lowest = min(pair["score"] for pair in payload["pairs"])
        tied = [pair for pair in payload["pairs"] if abs(pair["score"] - lowest) < 1e-9]

        assert len(tied) == 8
        assert all(pair["count"] == 1.0 for pair in tied)

    def test_the_extremes_the_page_quotes(self, payload: dict):
        assert payload["highest"]["score"] == pytest.approx(1.527794, abs=1e-6)
        assert payload["lowest"]["score"] == pytest.approx(-0.510826, abs=1e-6)
        assert abs(payload["nearest_chance"]["score"]) == pytest.approx(
            0.028171, abs=1e-6
        )


# --- Clipping the negative half ----------------------------------------------


class TestClipping:
    @pytest.fixture(scope="class")
    def payload(self) -> dict:
        response = client.get(f"{BASE}/clipping")
        assert response.status_code == 200
        return response.json()

    def test_the_two_kinds_of_zero_are_counted_apart(self, payload: dict):
        assert payload["n_cells"] == 529
        assert payload["n_undefined"] == 249
        assert payload["n_below_chance"] == 56
        assert payload["n_zero_after_clipping"] == 305

    def test_the_furthest_fall_and_the_mean_of_what_is_thrown_away(self, payload: dict):
        assert payload["lowest_score"] == pytest.approx(-0.510826, abs=1e-6)
        assert payload["mean_below_chance"] == pytest.approx(-0.256225, abs=1e-6)
        assert payload["furthest_falls"][0]["score"] == pytest.approx(
            -0.510826, abs=1e-6
        )

    def test_the_shift_sweep_empties_the_table_at_the_logarithm_of_five(
        self, payload: dict
    ):
        rows = {round(row["shift"], 4): row for row in payload["shifts"]}

        assert rows[0.0]["n_kept"] == 224
        assert rows[0.6931]["n_kept"] == 96
        assert rows[1.0]["n_kept"] == 64
        assert rows[1.6094]["n_kept"] == 0
        assert rows[1.6094]["gap"] is None

    def test_the_separation_falls_as_the_shift_rises(self, payload: dict):
        rows = {round(row["shift"], 4): row for row in payload["shifts"]}

        assert rows[0.0]["gap"] == pytest.approx(0.6646, abs=1e-3)
        assert rows[0.6931]["gap"] == pytest.approx(0.5789, abs=1e-3)
        assert rows[1.0]["gap"] == pytest.approx(0.4901, abs=1e-3)


# --- Smoothing the context rates ---------------------------------------------


class TestSmoothing:
    @pytest.fixture(scope="class")
    def payload(self) -> dict:
        response = client.get(f"{BASE}/smoothing")
        assert response.status_code == 200
        return response.json()

    def test_the_two_named_contexts_are_a_rare_one_and_a_common_one(
        self, payload: dict
    ):
        assert payload["rare_context_pair"] == ["flour", "whisk"]
        assert payload["frequent_context_pair"] == ["sugar", "the"]
        assert payload["rare_context_total"] == 25.0
        assert payload["frequent_context_total"] == 40.0

    def test_smoothing_lowers_the_rare_context_and_raises_the_frequent_one(
        self, payload: dict
    ):
        rows = {round(row["exponent"], 2): row for row in payload["rows"]}

        assert rows[1.0]["rare_context"] == pytest.approx(0.652325, abs=1e-6)
        assert rows[0.75]["rare_context"] == pytest.approx(0.593342, abs=1e-6)
        assert rows[1.0]["frequent_context"] == pytest.approx(0.433636, abs=1e-6)
        assert rows[0.75]["frequent_context"] == pytest.approx(0.492154, abs=1e-6)
        assert rows[0.25]["rare_context"] < rows[0.5]["rare_context"]
        assert rows[0.25]["frequent_context"] > rows[0.5]["frequent_context"]

    def test_a_pair_already_below_chance_is_pushed_further_below(self, payload: dict):
        rows = {round(row["exponent"], 2): row for row in payload["rows"]}

        assert rows[1.0]["below_chance"] == pytest.approx(-0.510826, abs=1e-6)
        assert rows[0.75]["below_chance"] == pytest.approx(-0.524228, abs=1e-6)

    def test_smoothing_breaks_the_symmetry(self, payload: dict):
        rows = {round(row["exponent"], 2): row for row in payload["rows"]}

        assert rows[1.0]["n_asymmetric_entries"] == 0
        assert rows[0.75]["n_asymmetric_entries"] == 176
        assert rows[0.75]["n_asymmetric_pairs"] == 88
        assert rows[0.75]["n_kept"] == 224

    def test_the_largest_disagreement_between_the_two_directions(self, payload: dict):
        forward, backward = payload["largest_asymmetry"]

        assert {forward["word"], forward["context"]} == {"and", "crew"}
        assert forward["kept"] == pytest.approx(0.305660, abs=1e-6)
        assert backward["kept"] == pytest.approx(0.423161, abs=1e-6)
        assert backward["kept"] - forward["kept"] == pytest.approx(0.117501, abs=1e-6)

    def test_the_published_exponent_separates_the_topics_slightly_less_here(
        self, payload: dict
    ):
        rows = {round(row["exponent"], 2): row for row in payload["rows"]}

        assert rows[1.0]["gap"] == pytest.approx(0.6718, abs=1e-3)
        assert rows[0.75]["gap"] == pytest.approx(0.6646, abs=1e-3)
        assert rows[0.75]["gap"] < rows[1.0]["gap"]

    def test_smoothing_moves_the_across_topic_cosine_the_wrong_way(self, payload: dict):
        rows = {round(row["exponent"], 2): row for row in payload["rows"]}

        assert rows[1.0]["across_topic"] == pytest.approx(0.0016, abs=1e-3)
        assert rows[0.75]["across_topic"] == pytest.approx(0.0086, abs=1e-3)


# --- Four readings of one corpus ---------------------------------------------


class TestReadings:
    @pytest.fixture(scope="class")
    def payload(self) -> dict:
        response = client.get(f"{BASE}/readings", params={"word": "flour"})
        assert response.status_code == 200
        return response.json()

    def test_four_readings_come_back_in_order(self, payload: dict):
        assert [reading["name"] for reading in payload["readings"]] == [
            "counts",
            "plain-scores",
            "smoothed-scores",
            "vectors",
        ]
        assert [reading["numbers_per_word"] for reading in payload["readings"]] == [
            23,
            23,
            23,
            4,
        ]

    def test_scoring_alone_separates_the_topics_worse_than_counting(
        self, payload: dict
    ):
        readings = {reading["name"]: reading for reading in payload["readings"]}

        assert readings["counts"]["gap"] == pytest.approx(0.5438, abs=1e-3)
        assert readings["smoothed-scores"]["gap"] == pytest.approx(0.4936, abs=1e-3)
        assert readings["smoothed-scores"]["gap"] < readings["counts"]["gap"]

    def test_scoring_does_do_what_it_claims_to_the_across_topic_cosine(
        self, payload: dict
    ):
        readings = {reading["name"]: reading for reading in payload["readings"]}

        assert readings["counts"]["across_topic"] == pytest.approx(0.1192, abs=1e-3)
        assert readings["smoothed-scores"]["across_topic"] == pytest.approx(
            0.0292, abs=1e-3
        )
        assert readings["counts"]["within_topic"] == pytest.approx(0.6629, abs=1e-3)
        assert readings["smoothed-scores"]["within_topic"] == pytest.approx(
            0.5228, abs=1e-3
        )

    def test_the_decomposition_is_what_rescues_it(self, payload: dict):
        readings = {reading["name"]: reading for reading in payload["readings"]}

        assert readings["vectors"]["gap"] == pytest.approx(0.6646, abs=1e-3)
        assert readings["vectors"]["within_topic"] == pytest.approx(0.6732, abs=1e-3)
        assert readings["vectors"]["across_topic"] == pytest.approx(0.0086, abs=1e-3)

    def test_every_reading_answers_flour_with_cooking_words(self, payload: dict):
        cooking = {
            "flour",
            "sugar",
            "butter",
            "eggs",
            "oven",
            "bake",
            "stir",
            "whisk",
            "dough",
            "pan",
        }
        for reading in payload["readings"]:
            words = {near["word"] for near in reading["nearest"]}
            assert words <= cooking, reading["name"]

    def test_the_window_sweep_stops_mattering_once_a_document_is_covered(
        self, payload: dict
    ):
        rows = {int(row["value"]): row for row in payload["windows"]}

        assert rows[1]["n_kept"] == 112
        assert rows[5]["n_kept"] == 224
        assert rows[8]["n_kept"] == 224
        assert rows[5]["gap"] == pytest.approx(rows[8]["gap"], abs=1e-12)
        assert rows[1]["gap"] == pytest.approx(0.4238, abs=1e-3)

    def test_the_width_sweep_falls_all_the_way_down(self, payload: dict):
        rows = {int(row["value"]): row for row in payload["dimensions"]}

        assert rows[2]["gap"] == pytest.approx(0.9885, abs=1e-3)
        assert rows[4]["gap"] == pytest.approx(0.6646, abs=1e-3)
        assert rows[23]["gap"] == pytest.approx(0.1997, abs=1e-3)

    def test_the_published_power_of_the_singular_values_is_not_the_best_here(
        self, payload: dict
    ):
        rows = {round(row["value"], 2): row for row in payload["exponents"]}

        assert rows[0.0]["gap"] == pytest.approx(0.5005, abs=1e-3)
        assert rows[0.5]["gap"] == pytest.approx(0.6646, abs=1e-3)
        assert rows[1.0]["gap"] == pytest.approx(0.7762, abs=1e-3)
        assert rows[1.0]["gap"] > rows[0.5]["gap"]

    def test_five_pairs_of_cooking_words_come_out_indistinguishable(
        self, payload: dict
    ):
        collapsed = payload["collapsed"]
        pairs = {(entry["first"], entry["second"]) for entry in collapsed}

        assert len(collapsed) == 10
        assert ("flour", "bake") in pairs
        assert ("sail", "tide") in pairs
        for entry in collapsed:
            assert entry["vector_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_the_collapsed_pairs_were_not_alike_before_the_squeeze(self, payload: dict):
        entry = next(
            item
            for item in payload["collapsed"]
            if item["first"] == "flour" and item["second"] == "bake"
        )

        assert entry["full_row_similarity"] == pytest.approx(0.5356, abs=1e-3)

    def test_the_table_and_the_vectors_are_counted(self, payload: dict):
        assert payload["numbers_in_the_table"] == 529
        assert payload["numbers_in_the_vectors"] == 92

    def test_the_three_stages_are_timed(self, payload: dict):
        assert payload["counting_milliseconds"] > 0.0
        assert payload["scoring_milliseconds"] > 0.0
        assert payload["decomposing_milliseconds"] > 0.0


# --- One fit -----------------------------------------------------------------


class TestFit:
    def test_the_default_fit_puts_cooking_words_around_flour(self):
        response = client.post(f"{BASE}/fit", json={})

        assert response.status_code == 200
        payload = response.json()
        assert payload["n_words"] == 23
        assert payload["n_kept"] == 224
        assert payload["n_zero"] == 305
        assert payload["numbers_in_the_vectors"] == 92
        assert payload["note"] is None
        assert payload["neighbours"][0]["word"] == "bake"
        assert payload["neighbours"][0]["similarity"] == pytest.approx(1.0, abs=1e-6)
        assert payload["gap"] == pytest.approx(0.6646, abs=1e-3)

    def test_the_four_singular_values_the_page_quotes(self):
        payload = client.post(f"{BASE}/fit", json={}).json()

        assert payload["singular_values"][0] == pytest.approx(6.644, abs=1e-3)
        assert payload["singular_values"][1] == pytest.approx(6.1406, abs=1e-3)

    def test_a_shift_past_the_largest_score_leaves_every_word_without_a_direction(
        self,
    ):
        payload = client.post(f"{BASE}/fit", json={"shift": 1.61}).json()

        assert payload["n_kept"] == 0
        assert payload["n_words_without_a_direction"] == 23
        assert payload["within_topic"] is None
        assert payload["gap"] is None
        assert payload["neighbours"] == []
        assert payload["note"] is not None

    def test_a_wider_fit_reconstructs_the_table_more_closely(self):
        errors = {
            dimension: client.post(f"{BASE}/fit", json={"dimension": dimension}).json()[
                "reconstruction_error"
            ]
            for dimension in (2, 4, 8, 12, 23)
        }

        assert errors[2] == pytest.approx(0.6579, abs=1e-3)
        assert errors[4] == pytest.approx(0.5668, abs=1e-3)
        assert errors[8] == pytest.approx(0.4542, abs=1e-3)
        assert errors[12] == pytest.approx(0.3376, abs=1e-3)
        assert errors[23] == pytest.approx(0.0, abs=1e-9)


# --- Refusals ----------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        "body",
        [
            {"dimension": 1},
            {"dimension": 24},
            {"window": 0},
            {"context_smoothing": 0.0},
            {"context_smoothing": 1.5},
            {"shift": -0.1},
            {"shift": 9.0},
            {"singular_value_exponent": 1.5},
            {"alpha": 0.75},
        ],
    )
    def test_a_bad_fit_request_is_refused_with_a_readable_message(self, body: dict):
        response = client.post(f"{BASE}/fit", json=body)

        assert response.status_code == 422
        assert response.json()["detail"]

    @pytest.mark.parametrize(
        "body",
        [
            {"corpus": "something-else"},
            {"window": 0},
            {"window": 11},
            {"context_smoothing": 2.0},
            {"smoothing": 0.75},
        ],
    )
    def test_a_bad_table_request_is_refused(self, body: dict):
        response = client.post(f"{BASE}/table", json=body)

        assert response.status_code == 422

    def test_a_word_the_corpus_never_saw_comes_back_with_no_neighbours(self):
        payload = client.post(f"{BASE}/fit", json={"word": "telescope"}).json()

        assert payload["neighbours"] == []
        assert payload["coordinates"] == []


# --- Nothing a reader sees names the code ------------------------------------


IDENTIFIER_WORDS = (
    "oop_ml",
    "the library",
    "the API",
    "numpy",
    "PointwiseMutualInformation",
    "CooccurrenceMatrix",
    "Error",
    "_",
)


class TestNothingVisibleNamesTheCode:
    def collected(self) -> list[str]:
        strings: list[str] = []
        for payload in (
            client.get(f"{BASE}/clipping").json(),
            client.get(f"{BASE}/smoothing").json(),
            client.get(f"{BASE}/readings").json(),
            client.post(f"{BASE}/fit", json={"shift": 1.61}).json(),
        ):
            stack: list[object] = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
                elif isinstance(item, str):
                    strings.append(item)
        return strings

    def test_no_returned_string_names_an_identifier(self):
        for text in self.collected():
            for forbidden in IDENTIFIER_WORDS:
                assert forbidden not in text, text

    def test_the_note_about_a_word_with_no_direction_is_in_plain_words(self):
        payload = client.post(f"{BASE}/fit", json={"shift": 1.61}).json()

        assert "direction" in payload["note"]
        assert not any(letter.isupper() for letter in payload["note"])

    def test_the_reading_labels_are_the_pages_own_words(self):
        labels = [
            reading["label"]
            for reading in client.get(f"{BASE}/readings").json()["readings"]
        ]

        assert labels == [
            "the counts themselves",
            "the scores, unsmoothed",
            "the scores, smoothed",
            "four numbers a word",
        ]
