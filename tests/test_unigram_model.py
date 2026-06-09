"""The unigram language model page's endpoints, on the two corpora it carries.

Every number the page quotes in prose is pinned here, so a change to the
library that moved one of them would fail a test rather than quietly rewrite
the page. The refusal tests are the other half: the page may not show a
parameter name, and the last class asserts that nothing identifier-shaped
survives the rewriting.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/the-unigram-language-model"
SPELLINGS = f"{BASE}/spellings"
PRUNING = f"{BASE}/pruning"
SIZES = f"{BASE}/sizes"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."


def spelling_of(body: dict, word: str) -> dict:
    return next(entry for entry in body["words"] if entry["word"] == word)


def row_at(body: dict, asked: int) -> dict:
    return next(row for row in body["rows"] if row["asked"] == asked)


class TestSpellings:
    def test_the_four_words_hold_forty_nine_candidate_pieces(self) -> None:
        body = client.get(SPELLINGS).json()

        assert body["n_candidates"] == 49
        assert len(body["alphabet"]) == 11
        assert body["total_occurrences"] == 16
        assert body["character_pieces"] == 79

    def test_the_seed_ranks_a_candidate_by_frequency_times_length(self) -> None:
        body = client.get(SPELLINGS).json()

        by_piece = {row["piece"]: row for row in body["candidates"]}
        assert by_piece["e"]["frequency"] == 17
        assert by_piece["e"]["n_symbols"] == 1
        assert by_piece["est</w>"]["frequency"] == 9
        assert by_piece["est</w>"]["n_symbols"] == 3
        assert by_piece["est</w>"]["score"] == 27.0

    def test_the_held_out_word_has_twenty_eight_spellings(self) -> None:
        body = client.get(SPELLINGS).json()

        lowest = spelling_of(body, "lowest")
        assert lowest["in_corpus"] is False
        assert lowest["n_spellings"] == 28
        assert lowest["best_pieces"] == ["lo", "west</w>"]

    def test_the_top_three_spellings_are_two_pieces_each(self) -> None:
        body = client.get(SPELLINGS).json()

        rows = spelling_of(body, "lowest")["rows"]
        assert [row["pieces"] for row in rows[:3]] == [
            ["lo", "west</w>"],
            ["low", "est</w>"],
            ["lowe", "st</w>"],
        ]
        assert rows[0]["share"] == pytest.approx(0.4681, rel=1e-3, abs=1e-3)
        assert rows[1]["share"] == pytest.approx(0.2006, rel=1e-3, abs=1e-3)
        assert rows[1]["log_probability"] == pytest.approx(
            rows[2]["log_probability"], rel=1e-12
        )

    def test_the_merged_cut_of_the_held_out_word_ranks_fourth(self) -> None:
        body = client.get(SPELLINGS).json()

        lowest = spelling_of(body, "lowest")
        assert lowest["merged_pieces"] == ["lo", "w", "est</w>"]
        assert lowest["merged_rank"] == 4
        assert lowest["merged_share"] == pytest.approx(0.031, rel=1e-2, abs=1e-2)

    def test_the_merged_cut_of_a_corpus_word_can_rank_twelfth(self) -> None:
        body = client.get(SPELLINGS).json()

        lower = spelling_of(body, "lower")
        assert lower["best_pieces"] == ["lower</w>"]
        assert lower["merged_pieces"] == ["lo", "w", "e", "r</w>"]
        assert lower["merged_rank"] == 12

    def test_merging_and_the_model_agree_on_the_two_frequent_words(self) -> None:
        body = client.get(SPELLINGS).json()

        assert spelling_of(body, "newest")["merged_pieces"] == ["newest</w>"]
        assert spelling_of(body, "newest")["merged_rank"] == 1
        assert spelling_of(body, "low")["merged_pieces"] == ["low</w>"]
        assert spelling_of(body, "low")["merged_rank"] == 1
        assert spelling_of(body, "newest")["rows"][0]["share"] == pytest.approx(
            0.8545, rel=1e-3, abs=1e-3
        )
        assert spelling_of(body, "lower")["rows"][0]["share"] == pytest.approx(
            0.9290, rel=1e-3, abs=1e-3
        )

    def test_a_word_that_is_one_piece_still_has_other_spellings(self) -> None:
        body = client.get(SPELLINGS).json()

        low = spelling_of(body, "low")
        assert low["n_spellings"] == 4
        assert low["rows"][0]["share"] == pytest.approx(0.9461, rel=1e-3, abs=1e-3)

    def test_the_lattice_holds_seventeen_arcs_over_seven_positions(self) -> None:
        body = client.get(SPELLINGS).json()

        lattice = body["lattice"]
        assert lattice["word"] == "lowest"
        assert lattice["n_positions"] == 7
        assert len(lattice["arcs"]) == 17
        assert lattice["n_paths"] == 28
        assert lattice["best_pieces"] == ["lo", "west</w>"]

    def test_every_arc_spans_at_least_one_position(self) -> None:
        body = client.get(SPELLINGS).json()

        for arc in body["lattice"]["arcs"]:
            assert 0 <= arc["start"] < arc["end"] <= 6
            assert arc["log_probability"] <= 0.0

    def test_two_arcs_lie_on_the_best_path(self) -> None:
        body = client.get(SPELLINGS).json()

        chosen = [arc for arc in body["lattice"]["arcs"] if arc["on_best_path"]]
        assert [arc["piece"] for arc in chosen] == ["lo", "west</w>"]


class TestPruning:
    def test_the_one_word_corpus_is_worked_exactly(self) -> None:
        body = client.get(PRUNING).json()

        worked = body["worked"]
        assert worked["symbols"] == ["a", "b</w>"]
        assert [row["pieces"] for row in worked["spellings"]] == [
            ["ab</w>"],
            ["a", "b</w>"],
        ]
        assert worked["marginal_probability"] == pytest.approx(
            4 / 9, rel=1e-12, abs=1e-12
        )
        assert worked["spellings"][0]["share"] == pytest.approx(
            0.75, rel=1e-12, abs=1e-12
        )

    def test_the_expected_counts_add_to_one_and_a_quarter(self) -> None:
        body = client.get(PRUNING).json()

        counts = {row["piece"]: row["count"] for row in body["worked"]["expected"]}
        assert counts["ab</w>"] == pytest.approx(0.75, rel=1e-12, abs=1e-12)
        assert counts["a"] == pytest.approx(0.25, rel=1e-12, abs=1e-12)
        assert body["worked"]["expected_total"] == pytest.approx(
            1.25, rel=1e-12, abs=1e-12
        )

    def test_two_rounds_take_the_one_piece_spelling_to_six_and_seven_tenths(
        self,
    ) -> None:
        body = client.get(PRUNING).json()

        rounds = body["worked"]["rounds"]
        first = {row["piece"]: row["probability"] for row in rounds[0]["probabilities"]}
        second = {
            row["piece"]: row["probability"] for row in rounds[1]["probabilities"]
        }
        assert first["ab</w>"] == pytest.approx(0.6, rel=1e-12, abs=1e-12)
        assert first["a"] == pytest.approx(0.2, rel=1e-12, abs=1e-12)
        assert second["ab</w>"] == pytest.approx(0.882353, rel=1e-5, abs=1e-5)

    def test_estimation_drives_the_unused_pieces_out_of_reach(self) -> None:
        body = client.get(PRUNING).json()

        watched = [
            {row["piece"]: row["probability"] for row in step["probabilities"]}
            for step in body["concentration"]
        ]
        assert watched[0]["low</w>"] == pytest.approx(0.02008, rel=1e-3, abs=1e-3)
        assert watched[1]["low</w>"] == pytest.approx(0.264961, rel=1e-4, abs=1e-4)
        assert watched[3]["low</w>"] == pytest.approx(0.3125, rel=1e-6, abs=1e-6)
        assert watched[1]["lo"] > watched[2]["lo"] > watched[3]["lo"]

        scores = [step["unseen_log_probability"] for step in body["concentration"]]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(-7.2972, rel=1e-3, abs=1e-3)
        assert scores[4] == pytest.approx(-73.4987, rel=1e-4, abs=1e-4)

    def test_four_pieces_carry_the_whole_four_word_corpus(self) -> None:
        body = client.get(PRUNING).json()

        four = next(view for view in body["losses"] if view["label"] == "four words")
        assert four["n_pieces"] == 49
        assert four["n_removable"] == 38
        assert four["n_costing_nothing"] == 34
        costly = {row["piece"]: row["loss"] for row in four["most_costly"][:4]}
        assert costly["newest</w>"] == pytest.approx(91.2605, rel=1e-4, abs=1e-4)
        assert costly["low</w>"] == pytest.approx(85.0399, rel=1e-4, abs=1e-4)
        assert costly["widest</w>"] == pytest.approx(45.6302, rel=1e-4, abs=1e-4)
        assert costly["lower</w>"] == pytest.approx(34.0160, rel=1e-4, abs=1e-4)

    def test_most_of_the_eighteen_sentences_pieces_cost_nothing(self) -> None:
        body = client.get(PRUNING).json()

        sentences = next(
            view for view in body["losses"] if view["label"] == "eighteen sentences"
        )
        assert sentences["n_pieces"] == 866
        assert sentences["n_removable"] == 815
        assert sentences["n_costing_nothing"] == 745
        assert sentences["most_costly"][0]["piece"] == "the</w>"
        assert sentences["most_costly"][0]["loss"] == pytest.approx(
            339.3151, rel=1e-4, abs=1e-4
        )
        assert all(row["loss"] == 0.0 for row in sentences["least_costly"])

    def test_the_pruning_keeps_three_quarters_at_a_time(self) -> None:
        body = client.get(PRUNING).json()

        traces = {view["label"]: view for view in body["traces"]}
        assert traces["four words"]["sizes"] == [49, 36, 27, 20, 15]
        assert traces["four words"]["n_rounds"] == 4
        assert traces["eighteen sentences"]["sizes"] == [
            866,
            649,
            486,
            364,
            273,
            204,
            153,
            136,
        ]
        assert traces["eighteen sentences"]["n_rounds"] == 7

    def test_the_pieces_above_the_four_words_are_chosen_alphabetically(self) -> None:
        body = client.get(PRUNING).json()

        assert body["tie_broken_size"] == 22
        assert body["tie_broken_survivors"] == [
            "de",
            "des",
            "dest</w>",
            "er</w>",
            "es",
            "est</w>",
        ]


class TestSizes:
    def test_the_corpus_and_its_floor(self) -> None:
        body = client.get(SIZES).json()

        assert body["n_distinct_words"] == 72
        assert body["n_word_occurrences"] == 133
        assert body["alphabet_size"] == 51
        assert body["character_pieces"] == 648
        assert body["smallest_size"] == 52
        assert body["n_candidates"] == 866

    def test_the_corpus_reaches_one_piece_per_word_and_the_merged_one_does_not(
        self,
    ) -> None:
        body = client.get(SIZES).json()

        assert row_at(body, 137)["unigram_corpus"] == 133
        assert row_at(body, 137)["merged_corpus"] == 263
        assert row_at(body, 130)["unigram_corpus"] == 133
        assert row_at(body, 80)["unigram_corpus"] == 324
        assert row_at(body, 80)["merged_corpus"] == 414

    def test_the_held_out_sentence_goes_the_other_way(self) -> None:
        body = client.get(SIZES).json()

        assert row_at(body, 80)["unigram_sentence"] == 40
        assert row_at(body, 80)["merged_sentence"] == 28
        assert row_at(body, 137)["unigram_sentence"] == 28
        assert row_at(body, 137)["merged_sentence"] == 25

    def test_the_sentence_gets_longer_when_the_vocabulary_grows_past_the_corpus(
        self,
    ) -> None:
        body = client.get(SIZES).json()

        assert row_at(body, 160)["unigram_sentence"] == 28
        assert row_at(body, 200)["unigram_sentence"] == 29
        assert row_at(body, 200)["merged_learned"] == 137

    def test_both_methods_agree_at_the_smallest_vocabulary(self) -> None:
        body = client.get(SIZES).json()

        smallest = row_at(body, 52)
        assert smallest["unigram_corpus"] == smallest["merged_corpus"] == 648
        assert smallest["unigram_sentence"] == smallest["merged_sentence"] == 45

    def test_the_learned_pieces_are_whole_words_and_the_merged_ones_are_not(
        self,
    ) -> None:
        body = client.get(SIZES).json()

        by_size = {row["asked"]: row for row in body["compositions"]}
        assert by_size[80]["unigram_learned"] == 28
        assert by_size[80]["unigram_whole_words"] == 28
        assert by_size[80]["merged_whole_words"] == 6
        assert by_size[100]["unigram_whole_words"] == 48
        assert by_size[100]["merged_whole_words"] == 8
        assert by_size[137]["unigram_learned"] == 85
        assert by_size[137]["unigram_whole_words"] == 70
        assert by_size[137]["merged_whole_words"] == 18

    def test_the_pieces_the_page_names_are_the_pieces_that_come_back(self) -> None:
        body = client.get(SIZES).json()

        at_eighty = next(cut for cut in body["cuts"] if cut["vocabulary_size"] == 80)
        assert "the</w>" in at_eighty["unigram_pieces"]
        assert "cost</w>" in at_eighty["unigram_pieces"]
        for fragment in ("exp", "ec", "lo", "re", "analysi"):
            assert fragment in at_eighty["merged_pieces"]

        at_largest = next(cut for cut in body["cuts"] if cut["vocabulary_size"] == 137)
        for piece in ("Dr.</w>", "the</w>", "analysis.</w>", "-cost</w>"):
            assert piece in at_largest["unigram_pieces"]
        spelled_out = ["e", "x", "p", "e", "c", "t</w>"]
        pieces = at_largest["unigram_pieces"]
        assert any(
            pieces[start : start + len(spelled_out)] == spelled_out
            for start in range(len(pieces))
        )

        merged_spelled_out = ["exp", "ec", "t</w>"]
        merged = at_largest["merged_pieces"]
        assert any(
            merged[start : start + len(merged_spelled_out)] == merged_spelled_out
            for start in range(len(merged))
        )

        at_137 = next(row for row in body["compositions"] if row["asked"] == 137)
        assert len(at_137["unigram_other"]) == 15
        assert "-cost</w>" in at_137["unigram_other"]
        assert "Augus" in at_137["unigram_other"]
        assert "'t</w>" in at_137["unigram_other"]

    def test_the_round_trip_fails_on_one_symbol_and_holds_otherwise(self) -> None:
        body = client.get(SIZES).json()

        largest = next(cut for cut in body["cuts"] if cut["vocabulary_size"] == 137)
        assert largest["unigram_round_trip"] is False
        assert largest["unigram_decoded"] == (
            "Dr. Alvare[UNK]didn't expect the low-cost re-analysis."
        )
        assert body["unspellable_symbol"] == "z</w>"
        assert body["control_round_trip"] is True
        assert body["control_pieces"] == 9

    def test_one_word_of_eighty_eight_is_cut_longer_than_it_need_be(self) -> None:
        body = client.get(SIZES).json()

        assert body["n_words_examined"] == 88
        assert body["length_comparison_size"] == 200
        assert len(body["length_comparisons"]) == 1
        only = body["length_comparisons"][0]
        assert only["word"] == "didn't"
        assert only["likeliest_n"] == 6
        assert only["shortest_n"] == 5
        assert only["shortest_pieces"] == ["d", "i", "d", "n", "'t</w>"]
        assert only["likeliest_log_probability"] > only["shortest_log_probability"]


class TestFit:
    def test_the_playground_reads_the_running_sentence(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "sentences",
                "vocabulary_size": 100,
                "text": SENTENCE,
            },
        ).json()

        assert body["learned"] == 100
        assert body["n_pruning_rounds"] == 8
        assert body["n_candidates"] == 866
        assert body["corpus_pieces"] == 229
        assert body["corpus_character_pieces"] == 648
        assert body["n_pieces"] == 32
        assert body["unknown_pieces"] == 1
        assert body["round_trip_exact"] is False
        assert body["merged_corpus_pieces"] == 344
        assert body["merged_n_pieces"] == 27

    def test_the_probabilities_come_back_ordered_and_bounded(self) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 137}
        ).json()

        probabilities = [row["probability"] for row in body["pieces"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert probabilities[0] == pytest.approx(0.142857, rel=1e-4, abs=1e-4)
        assert all(0.0 <= value <= 1.0 for value in probabilities)
        assert all(row["log_probability"] <= 0.0 for row in body["pieces"])

    def test_the_four_word_corpus_lands_on_the_four_words(self) -> None:
        body = client.post(
            FIT,
            json={"corpus": "four_words", "vocabulary_size": 16, "text": "lowest"},
        ).json()

        assert body["learned"] == 16
        assert body["smallest_size"] == 12
        assert body["n_candidates"] == 49
        assert body["corpus_pieces"] == 16
        assert [row["piece"] for row in body["pieces"][:4]] == [
            "newest</w>",
            "low</w>",
            "widest</w>",
            "lower</w>",
        ]
        assert body["merged_corpus_pieces"] == 48

    def test_a_vocabulary_below_the_alphabet_is_refused(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 10})

        assert response.status_code == 422
        assert "52" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 100, "extra": 1}
        )

        assert response.status_code == 422

    def test_a_vocabulary_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 5000}
        )

        assert response.status_code == 422


class TestNothingOnThePageNamesTheCode:
    """Every string a widget can render, checked for identifiers.

    A refusal is page text as much as a paragraph is, so the message a reader
    meets may not carry a parameter name, a class name or an underscore.
    """

    IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z][A-Za-z]*")

    def test_the_refusal_carries_no_identifier(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 10})
        detail = response.json()["detail"]

        assert self.IDENTIFIER.search(detail) is None
        assert "vocabulary_size" not in detail
        assert "Error" not in detail

    def test_the_four_word_refusal_carries_no_identifier(self) -> None:
        response = client.post(FIT, json={"corpus": "four_words", "vocabulary_size": 4})
        detail = response.json()["detail"]

        assert response.status_code == 422
        assert self.IDENTIFIER.search(detail) is None

    def test_no_label_a_widget_prints_names_the_code(self) -> None:
        labels = [view["label"] for view in client.get(PRUNING).json()["losses"]] + [
            view["label"] for view in client.get(PRUNING).json()["traces"]
        ]

        for label in labels:
            assert self.IDENTIFIER.search(label) is None
