"""The random indexing page's endpoints, on the two corpora it carries.

Every number the page prints is pinned here. The three-sentence corpus has
whole-number counts and a fit eight numbers wide, so the accumulation can be
asserted entry by entry rather than approximately; the twenty-four document one
is where the claims about the projection and its cost live.

The load-bearing tests of the module are
``TestTheIdentity::test_one_accumulating_pass_and_the_matrix_product_agree`` and
``TestNothingIdentifierShapedReachesAReader``. The first is the page's headline
claim, that the positions really are the table projected, and it is asserted as
a gap rather than as a similarity; the second holds every string a widget can
render to the rule that a reader never meets a class, a parameter or a module.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.random_indexing import (
    COOKING_DOCUMENTS,
    SAILING_DOCUMENTS,
    SHARED_WORDS,
    THREE_SENTENCES,
    TWENTY_FOUR_DOCUMENTS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/random-indexing"
FIT = f"{BASE}/fit"
WORKED = f"{BASE}/worked"
PERPENDICULARITY = f"{BASE}/perpendicularity"
COSTS = f"{BASE}/costs"
AGAINST = f"{BASE}/against-the-table"


class TestTheCorpora:
    def test_they_are_the_corpora_the_specs_fit(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        spec = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.counts.fixtures"
        )

        assert list(THREE_SENTENCES) == spec.TINY_PMI_CORPUS
        assert list(COOKING_DOCUMENTS) == spec.COOKING_DOCUMENTS
        assert list(SAILING_DOCUMENTS) == spec.SAILING_DOCUMENTS
        assert list(SHARED_WORDS) == spec.SHARED_WORDS
        assert TWENTY_FOUR_DOCUMENTS == spec.TWO_TOPIC_CORPUS

    def test_the_larger_corpus_is_twenty_three_words_over_a_hundred_and_forty_four(
        self,
    ) -> None:
        body = client.post(FIT, json={}).json()

        assert body["n_words"] == 23
        assert body["occurrences"] == 144
        assert len(body["words"]) == 23


class TestTheIdentity:
    """The positions are the counts multiplied by the drawn directions."""

    def test_one_accumulating_pass_and_the_matrix_product_agree(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["identity_gap"] == 0.0
        assert body["largest_coordinate"] == 17.0

    def test_the_gap_is_rounding_where_the_weights_are_fractions(self) -> None:
        body = client.post(FIT, json={"weighting": "harmonic", "dimension": 64}).json()

        assert body["identity_gap"] == pytest.approx(8.88e-16, abs=1e-17)
        assert body["largest_coordinate"] == pytest.approx(11.25)

    @pytest.mark.parametrize("window", [1, 2, 5, 8])
    @pytest.mark.parametrize("weighting", ["uniform", "harmonic"])
    def test_it_holds_at_every_window_and_weighting(
        self, window: int, weighting: str
    ) -> None:
        body = client.post(
            FIT, json={"window": window, "weighting": weighting, "dimension": 32}
        ).json()

        assert body["identity_gap"] < 1e-13


class TestTheWorkedInstance:
    """Five words over three sentences, eight numbers wide, seed zero."""

    def test_the_vocabulary_is_commonest_first(self) -> None:
        body = client.post(WORKED, json={}).json()

        assert body["words"] == ["the", "cat", "sat", "dog", "ran"]

    def test_every_direction_holds_two_of_each_sign_and_four_zeros(self) -> None:
        body = client.post(WORKED, json={}).json()

        for direction in body["directions"]:
            assert direction["entries"].count(1) == 2
            assert direction["entries"].count(-1) == 2
            assert direction["entries"].count(0) == 4
            assert len(direction["positions"]) == 4

    def test_the_direction_drawn_for_each_word(self) -> None:
        body = client.post(WORKED, json={}).json()
        drawn = {one["word"]: one["entries"] for one in body["directions"]}

        assert drawn["the"] == [0, 0, 1, -1, -1, 0, 1, 0]
        assert drawn["cat"] == [1, -1, 0, 0, 0, 1, 0, -1]
        assert drawn["sat"] == [0, 0, 0, 0, 1, 1, -1, -1]
        assert drawn["dog"] == [1, 0, 0, -1, 1, -1, 0, 0]
        assert drawn["ran"] == [0, 1, 0, -1, 1, -1, 0, 0]

    def test_the_counts_are_whole_numbers_and_symmetric(self) -> None:
        body = client.post(WORKED, json={}).json()

        assert body["counts"] == [
            [0.0, 2.0, 2.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]

    def test_four_additions_build_the_position_of_cat(self) -> None:
        body = client.post(WORKED, json={}).json()

        assert body["n_occurrences"] == 2
        assert body["row_total"] == 4.0
        assert len(body["steps"]) == 4
        assert [step["neighbour"] for step in body["steps"]] == [
            "the",
            "sat",
            "the",
            "ran",
        ]
        assert [step["distance"] for step in body["steps"]] == [1, 1, 1, 1]
        assert body["steps"][0]["running"] == [0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0]
        assert body["steps"][2]["running"] == [
            0.0,
            0.0,
            2.0,
            -2.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
        ]

    def test_the_last_running_total_is_what_the_fit_reports(self) -> None:
        body = client.post(WORKED, json={}).json()

        assert body["final"] == [0.0, 1.0, 2.0, -3.0, 0.0, 0.0, 1.0, -1.0]
        assert body["steps"][-1]["running"] == body["final"]
        assert body["matches_the_fit"] is True

    def test_a_three_word_sentence_leaves_nothing_for_a_wider_window_to_reach(
        self,
    ) -> None:
        """Every neighbour here is next door already, so widening changes nothing
        and the two-topic collection is where a window can be seen to matter."""
        narrow = client.post(WORKED, json={"window": 1}).json()
        wide = client.post(WORKED, json={"window": 2}).json()

        assert len(narrow["steps"]) == len(wide["steps"]) == 4

    @pytest.mark.parametrize(("window", "steps"), [(1, 9), (2, 16), (5, 30)])
    def test_a_wider_window_adds_more_on_six_word_documents(
        self, window: int, steps: int
    ) -> None:
        body = client.post(
            WORKED,
            json={
                "corpus": "two-topic",
                "word": "flour",
                "dimension": 16,
                "window": window,
            },
        ).json()

        assert body["n_occurrences"] == 6
        assert len(body["steps"]) == steps
        assert body["row_total"] == float(steps)

    def test_distance_weighting_divides_by_how_far_away_the_neighbour_was(
        self,
    ) -> None:
        body = client.post(
            WORKED,
            json={
                "corpus": "two-topic",
                "word": "flour",
                "dimension": 16,
                "window": 5,
                "weighting": "harmonic",
            },
        ).json()

        assert sorted({step["weight"] for step in body["steps"]}) == [
            pytest.approx(0.2),
            pytest.approx(0.25),
            pytest.approx(1 / 3),
            pytest.approx(0.5),
            pytest.approx(1.0),
        ]
        assert all(
            step["weight"] == pytest.approx(1.0 / step["distance"])
            for step in body["steps"]
        )
        assert body["row_total"] == pytest.approx(16.35)


class TestNearlyPerpendicular:
    """Two hundred drawn directions, and how far from perpendicular they came."""

    def test_the_measurement_agrees_with_the_library_own_spec(self) -> None:
        body = client.get(PERPENDICULARITY).json()
        at = {row["dimension"]: row for row in body["widths"]}

        assert body["n_vectors"] == 200
        assert body["n_pairs"] == 19900
        assert at[50]["measured"] == pytest.approx(0.0744, abs=1e-4)
        assert at[500]["measured"] == pytest.approx(0.0077, abs=1e-4)
        assert at[50]["exactly_perpendicular"] == pytest.approx(0.7181, abs=1e-4)
        assert at[500]["exactly_perpendicular"] == pytest.approx(0.9694, abs=1e-4)

    def test_the_simple_rule_overstates_at_a_narrow_width(self) -> None:
        body = client.get(PERPENDICULARITY).json()
        at = {row["dimension"]: row for row in body["widths"]}

        assert at[8]["predicted"] == 0.5
        assert at[8]["measured"] == pytest.approx(0.2994, abs=1e-4)
        assert at[8]["ratio"] == pytest.approx(0.5988, abs=1e-4)
        assert at[1000]["ratio"] == pytest.approx(1.0144, abs=1e-4)

    def test_the_measured_value_falls_with_every_extra_width(self) -> None:
        body = client.get(PERPENDICULARITY).json()
        measured = [row["measured"] for row in body["widths"]]

        assert measured == sorted(measured, reverse=True)

    def test_the_rule_collapses_once_the_directions_fill_up(self) -> None:
        body = client.get(PERPENDICULARITY).json()
        at = {row["n_nonzero"]: row for row in body["fullness"]}

        assert body["fullness_dimension"] == 50
        assert at[2]["measured"] == pytest.approx(0.0404, abs=1e-4)
        assert at[50]["predicted"] == 1.0
        assert at[50]["measured"] == pytest.approx(0.1162, abs=1e-4)
        assert at[50]["exactly_perpendicular"] == 0.0


class TestTheFitTheReaderMeets:
    """Twenty-three words, sixteen numbers wide, seed zero."""

    def test_the_two_halves_come_apart(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["within_similarity"] == pytest.approx(0.6399, abs=1e-4)
        assert body["across_similarity"] == pytest.approx(0.1247, abs=1e-4)

    def test_the_whole_table_answers_almost_the_same_thing(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["table_within_similarity"] == pytest.approx(0.6629, abs=1e-4)
        assert body["table_across_similarity"] == pytest.approx(0.1192, abs=1e-4)

    def test_the_neighbours_of_one_word_are_reordered_rather_than_replaced(
        self,
    ) -> None:
        body = client.post(FIT, json={"n_results": 6}).json()
        projected = [one["word"] for one in body["nearest"]]
        by_the_table = [one["word"] for one in body["table_nearest"]]

        assert projected == ["eggs", "butter", "bake", "whisk", "pan", "sugar"]
        assert by_the_table == ["bake", "oven", "eggs", "sugar", "butter", "pan"]
        assert len(set(projected) & set(by_the_table)) == 5

    def test_the_nearest_word_by_one_reading_is_third_by_the_other(self) -> None:
        body = client.post(FIT, json={"n_results": 6}).json()
        first = body["nearest"][0]

        assert first["word"] == "eggs"
        assert first["similarity"] == pytest.approx(0.7909, abs=1e-4)
        assert first["table_similarity"] == pytest.approx(0.7047, abs=1e-4)
        assert first["table_rank"] == 3

    def test_one_pair_changes_sign_under_the_projection(self) -> None:
        body = client.post(FIT, json={}).json()
        against = {one["word"]: one for one in body["against_every_word"]}

        assert against["anchor"]["similarity"] == pytest.approx(-0.2437, abs=1e-4)
        assert against["anchor"]["table_similarity"] == pytest.approx(0.1208, abs=1e-4)

    def test_the_pair_with_little_in_common_moves_further_than_the_alike_pair(
        self,
    ) -> None:
        body = client.post(FIT, json={}).json()
        against = {one["word"]: one for one in body["against_every_word"]}
        alike = against["sugar"]
        unalike = against["anchor"]

        assert alike["table_similarity"] == pytest.approx(0.6991, abs=1e-4)
        assert alike["similarity"] == pytest.approx(0.6270, abs=1e-4)
        moved_alike = abs(alike["table_similarity"] - alike["similarity"])
        moved_unalike = abs(unalike["table_similarity"] - unalike["similarity"])
        assert moved_unalike / moved_alike == pytest.approx(5.06, abs=0.02)

    def test_every_other_word_is_reported_once(self) -> None:
        body = client.post(FIT, json={}).json()

        assert len(body["against_every_word"]) == 22
        assert "flour" not in {one["word"] for one in body["against_every_word"]}

    def test_the_geometry_of_the_directions_this_fit_drew(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["n_direction_pairs"] == 253
        assert body["colliding_pairs"] == 155
        assert body["direction_predicted_cosine"] == 0.25
        assert body["direction_mean_absolute_cosine"] == pytest.approx(0.1887, abs=1e-4)
        assert body["direction_exactly_perpendicular"] == pytest.approx(
            0.3874, abs=1e-4
        )

    def test_what_the_two_readings_cost_in_numbers(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["table_cells"] == 529
        assert body["position_cells"] == 368
        assert body["neighbour_visits"] == 720
        assert body["slot_updates"] == 2880
        assert body["product_multiplications"] == 8464

    def test_the_projection_moves_the_cosines_by_a_measurable_amount(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["table_gap_mean"] == pytest.approx(0.1441, abs=1e-4)
        assert body["table_gap_largest"] == pytest.approx(0.4690, abs=1e-4)

    def test_the_six_words_it_puts_furthest_away_are_all_of_the_other_subject(
        self,
    ) -> None:
        body = client.post(FIT, json={}).json()
        ordered = sorted(body["against_every_word"], key=lambda one: one["similarity"])

        assert [one["word"] for one in ordered[:6]] == [
            "anchor",
            "tide",
            "wind",
            "boat",
            "harbour",
            "sail",
        ]
        assert all(one["group"] == "sailing" for one in ordered[:6])

    def test_two_numbers_a_word_leaves_only_two_directions_to_hand_out(self) -> None:
        """Every pair then comes out at exactly one or exactly minus one, which
        is the degenerate case the closing part states."""
        body = client.post(FIT, json={"dimension": 2, "n_nonzero": 2}).json()
        drawn = {tuple(one["entries"]) for one in body["directions"]}

        assert drawn == {(1, -1), (-1, 1)}
        assert {round(one["similarity"], 9) for one in body["against_every_word"]} == {
            1.0,
            -1.0,
        }

    def test_a_word_alone_in_its_sentence_would_have_no_direction_to_compare(
        self,
    ) -> None:
        """Every word of these corpora has company, so the case is stated rather
        than reachable: a position summed from nothing is the origin."""
        body = client.post(FIT, json={}).json()

        assert all(one["length"] > 0.0 for one in body["positions"])
        assert all(one["occurrences"] > 0.0 for one in body["positions"])


class TestWhatTheShortcutCosts:
    def test_the_whole_table_is_the_thing_being_approximated(self) -> None:
        body = client.get(COSTS).json()

        assert body["reference_word"] == "flour"
        assert body["table_neighbours"] == [
            "bake",
            "oven",
            "eggs",
            "sugar",
            "butter",
            "pan",
        ]
        assert body["table_cells"] == 529
        assert body["n_words"] == 23
        assert body["table_separation"] == pytest.approx(0.5438, abs=1e-4)

    def test_the_gap_falls_as_the_width_grows(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["widths"]}

        assert at[8]["gap_mean"] == pytest.approx(0.3959, abs=1e-4)
        assert at[16]["gap_mean"] == pytest.approx(0.1441, abs=1e-4)
        assert at[64]["gap_mean"] == pytest.approx(0.0542, abs=1e-4)
        assert at[529]["gap_mean"] == pytest.approx(0.0181, abs=1e-4)
        assert at[2048]["gap_mean"] == pytest.approx(0.0109, abs=1e-4)

    def test_a_width_as_wide_as_the_table_still_does_not_reproduce_it(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["widths"]}

        assert at[529]["colliding_pairs"] == 8
        assert at[529]["gap_mean"] > 0.0

    def test_the_neighbour_list_is_recovered_by_sixty_four(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["widths"]}

        assert at[8]["shared_neighbours"] == 2
        assert at[16]["shared_neighbours"] == 5
        assert at[64]["shared_neighbours"] == body["n_top"]

    def test_writing_the_collection_out_again_changes_nothing_at_all(self) -> None:
        body = client.get(COSTS).json()
        rows = {row["repeats"]: row for row in body["repeats"]}

        assert rows[1]["occurrences"] == 144
        assert rows[8]["occurrences"] == 1152
        for count in (2, 4, 8):
            assert rows[count]["gap_mean"] == rows[1]["gap_mean"]
            assert rows[count]["pair_similarity"] == rows[1]["pair_similarity"]

    def test_the_answer_moves_with_the_draw_and_moves_less_when_wider(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["draws"]}

        assert at[8]["n_draws"] == 20
        assert at[8]["pair_spread"] == pytest.approx(0.9457, abs=1e-4)
        assert at[16]["pair_spread"] == pytest.approx(0.4032, abs=1e-4)
        assert at[200]["pair_spread"] == pytest.approx(0.1182, abs=1e-4)
        assert at[8]["pair_spread"] > at[16]["pair_spread"] > at[200]["pair_spread"]

    def test_most_pairs_of_topic_words_never_once_met(self) -> None:
        """The table's own answer for such a pair is what the whole idea rests
        on, so the page needs one that is high and genuinely never observed."""
        body = client.get(COSTS).json()

        assert body["n_topic_pairs"] == 190
        assert body["n_never_met"] == 110
        assert body["never_met"]["first"] == "flour"
        assert body["never_met"]["second"] == "bake"
        assert body["never_met"]["count"] == 0.0
        assert body["never_met"]["table_similarity"] == pytest.approx(0.7551, abs=1e-4)
        assert body["never_met"]["projected_similarity"] == pytest.approx(
            0.7130, abs=1e-4
        )

    def test_the_gap_never_turns_back_up_as_the_width_grows(self) -> None:
        """There is no elbow to find, which is the claim the closing part makes
        about having nothing inside the fit to choose a width by."""
        body = client.get(COSTS).json()
        gaps = [row["gap_mean"] for row in body["widths"]]

        assert gaps == sorted(gaps, reverse=True)

    def test_one_draw_orders_the_settings_the_average_reverses(self) -> None:
        body = client.get(COSTS).json()
        one = {
            (row["dimension"], row["n_nonzero"]): row["one_draw_gap"]
            for row in body["sparsity"]
        }

        assert one[(16, 2)] == pytest.approx(0.2051, abs=1e-4)
        assert one[(16, 4)] == pytest.approx(0.1441, abs=1e-4)
        assert one[(256, 2)] == pytest.approx(0.0085, abs=1e-4)
        assert one[(256, 4)] == pytest.approx(0.0340, abs=1e-4)
        assert one[(16, 2)] > one[(16, 4)]
        assert one[(256, 2)] < one[(256, 4)]

    def test_the_range_over_draws_at_the_narrowest_width(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["draws"]}

        assert at[8]["smallest_pair"] == pytest.approx(0.0189, abs=1e-4)
        assert at[8]["largest_pair"] == pytest.approx(0.9646, abs=1e-4)
        assert at[50]["pair_spread"] == pytest.approx(0.2853, abs=1e-4)

    def test_every_draw_puts_the_two_halves_the_right_way_round(self) -> None:
        body = client.get(COSTS).json()
        at = {row["dimension"]: row for row in body["draws"]}

        assert at[16]["smallest_separation"] == pytest.approx(0.1735, abs=1e-4)
        assert at[16]["largest_separation"] == pytest.approx(0.8430, abs=1e-4)
        assert all(row["smallest_separation"] > 0.0 for row in body["draws"])

    def test_how_full_a_direction_is_buys_almost_nothing(self) -> None:
        body = client.get(COSTS).json()
        rows = {
            (row["dimension"], row["n_nonzero"]): row["mean_gap"]
            for row in body["sparsity"]
        }

        assert body["sparsity_draws"] == 12
        assert rows[(16, 2)] == pytest.approx(0.1770, abs=1e-4)
        assert rows[(16, 16)] == pytest.approx(0.1732, abs=1e-4)
        assert rows[(256, 2)] == pytest.approx(0.0364, abs=1e-4)
        assert rows[(256, 32)] == pytest.approx(0.0357, abs=1e-4)

    def test_the_width_is_what_buys_the_accuracy_and_not_the_fullness(self) -> None:
        body = client.get(COSTS).json()
        at_width: dict[int, list[float]] = {}
        for row in body["sparsity"]:
            at_width.setdefault(row["dimension"], []).append(row["mean_gap"])

        for spread in at_width.values():
            assert max(spread) - min(spread) < 0.02
        assert max(at_width[256]) < min(at_width[64])
        assert max(at_width[64]) < min(at_width[16])

    def test_a_draw_with_no_shared_position_reproduces_the_table_exactly(self) -> None:
        """Exactness is a property of the draw rather than of the width, which is
        why the same width at another draw is out by a hundredth."""
        body = client.get(COSTS).json()

        assert body["exact_dimension"] == 2048
        assert body["exact_colliding"] == 0
        assert body["exact_gap"] < 1e-15
        assert body["inexact_colliding"] == 2
        assert body["inexact_gap"] == pytest.approx(0.0109, abs=1e-4)


class TestAgainstAWeightedReading:
    def test_the_three_shared_words_have_the_loudest_rows(self) -> None:
        body = client.get(AGAINST).json()
        totals = dict(zip(body["words"], body["row_totals"], strict=True))

        assert body["shared_row_total"] == 40.0
        assert body["largest_topic_row_total"] == 35.0
        assert all(totals[word] == 40.0 for word in body["shared_words"])
        assert body["shared_share_of_the_table"] == pytest.approx(1 / 6)

    def test_adding_up_raw_counts_inherits_the_loudness(self) -> None:
        body = client.get(AGAINST).json()

        assert body["table_shared_to_topic"] == pytest.approx(0.5173, abs=1e-4)
        assert body["projected_shared_to_topic"] == pytest.approx(0.5064, abs=1e-4)
        assert body["weighted_shared_to_topic"] == pytest.approx(0.2207, abs=1e-4)
        assert body["weighted_shared_to_topic"] < body["projected_shared_to_topic"]

    def test_the_weighted_reading_pushes_the_two_halves_further_apart(self) -> None:
        body = client.get(AGAINST).json()

        assert body["weighted_across"] == pytest.approx(0.0037, abs=1e-4)
        assert body["projected_across"] == pytest.approx(0.1247, abs=1e-4)
        assert body["weighted_across"] < body["projected_across"]

    def test_the_three_readings_disagree_about_the_nearest_word(self) -> None:
        body = client.get(AGAINST).json()
        rows = {row["word"]: row for row in body["rows"]}

        assert rows["flour"]["by_the_table"] == "bake"
        assert rows["flour"]["by_the_projection"] == "eggs"
        assert rows["flour"]["by_the_weighted_table"] == "sugar"
        assert rows["butter"]["by_the_table"] == rows["butter"]["by_the_projection"]

    def test_two_words_carrying_no_subject_come_out_alike_either_way(self) -> None:
        body = client.get(AGAINST).json()
        pairs = dict(
            zip(body["shared_pairs"], body["shared_pair_projected"], strict=True)
        )

        assert pairs["and and the"] == pytest.approx(0.9321, abs=1e-4)
        assert min(pairs.values()) > 0.85


class TestRefusals:
    def test_an_odd_number_of_non_zeros_is_refused(self) -> None:
        response = client.post(FIT, json={"n_nonzero": 3})

        assert response.status_code == 422
        assert "as many entries of +1" in response.json()["detail"]

    def test_more_non_zeros_than_a_direction_holds_is_refused(self) -> None:
        response = client.post(FIT, json={"dimension": 4, "n_nonzero": 8})

        assert response.status_code == 422
        assert "do not fit" in response.json()["detail"]

    def test_a_word_the_collection_never_used_is_refused(self) -> None:
        response = client.post(FIT, json={"word": "kettle"})

        assert response.status_code == 422
        assert "kettle" in response.json()["detail"]
        assert "23 words" in response.json()["detail"]

    def test_the_worked_route_refuses_the_same_way(self) -> None:
        response = client.post(WORKED, json={"word": "kettle"})

        assert response.status_code == 422
        assert "5 words" in response.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"dimension": 3},
            {"dimension": 4096},
            {"n_nonzero": 1},
            {"window": 0},
            {"random_seed": -1},
            {"n_results": 0},
            {"corpus": "nonsense"},
            {"nothing_of_the_kind": 1},
        ],
    )
    def test_a_request_outside_the_bounds_is_refused(
        self, payload: dict[str, object]
    ) -> None:
        assert client.post(FIT, json=payload).status_code == 422


IDENTIFIER = re.compile(
    r"[a-z]+_[a-z_]+|[A-Za-z]+\.[A-Za-z_]+|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b"
)
"""Anything shaped like a class, a parameter or a module path."""


class TestNothingIdentifierShapedReachesAReader:
    """Every string a widget can render, held to the page's own vocabulary.

    A refusal is page text as much as a paragraph is, so a message that grows an
    identifier later fails here rather than reaching a reader.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"n_nonzero": 3},
            {"dimension": 4, "n_nonzero": 8},
            {"word": "kettle"},
        ],
    )
    def test_no_refusal_names_a_class_a_parameter_or_a_module(
        self, payload: dict[str, object]
    ) -> None:
        detail = client.post(FIT, json=payload).json()["detail"]

        assert not IDENTIFIER.search(detail), detail

    def test_no_word_a_widget_prints_is_an_identifier(self) -> None:
        body = client.get(AGAINST).json()
        printed = (
            body["words"]
            + body["shared_words"]
            + body["shared_pairs"]
            + [row["by_the_table"] for row in body["rows"]]
            + [row["by_the_projection"] for row in body["rows"]]
            + [row["by_the_weighted_table"] for row in body["rows"]]
        )

        for one in printed:
            assert not IDENTIFIER.search(one), one

    def test_no_sentence_or_word_of_a_fit_is_an_identifier(self) -> None:
        body = client.post(WORKED, json={}).json()
        printed = (
            body["words"]
            + body["sentences"]
            + [step["neighbour"] for step in body["steps"]]
            + [step["sentence_text"] for step in body["steps"]]
        )

        for one in printed:
            assert not IDENTIFIER.search(one), one
