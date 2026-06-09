"""The WordPiece page's endpoints, on the two corpora the page carries.

Every number the page quotes in prose is pinned here, so a change to the
library that moved one of them would fail a test rather than quietly rewrite
the page. The refusal tests are the other half: the page may not show a
parameter name, and the last class asserts that nothing identifier-shaped
survives the rewriting.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/wordpiece"
SCORING = f"{BASE}/scoring"
SPELLING = f"{BASE}/spelling"
LIMITS = f"{BASE}/limits"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."


def _pair(body: dict, left: str, right: str, key: str = "pairs") -> dict:
    for candidate in body[key]:
        if (candidate["left"], candidate["right"]) == (left, right):
            return candidate
    raise AssertionError(f"no candidate {left} + {right}")


class TestScoring:
    def test_the_four_words_are_spelled_in_eleven_symbols(self) -> None:
        body = client.get(SCORING).json()

        assert [symbol["piece"] for symbol in body["symbols"]] == [
            "##d",
            "##e",
            "##i",
            "##o",
            "##r",
            "##s",
            "##t",
            "##w",
            "l",
            "n",
            "w",
        ]

    def test_a_letter_that_both_opens_and_continues_is_counted_twice(self) -> None:
        """``w`` opens widest three times and continues elsewhere thirteen."""
        body = client.get(SCORING).json()
        counts = {symbol["piece"]: symbol["count"] for symbol in body["symbols"]}

        assert counts["w"] == 3
        assert counts["##w"] == 13
        assert counts["##e"] == 17

    def test_the_commonest_pair_scores_badly_because_its_halves_are_common(
        self,
    ) -> None:
        body = client.get(SCORING).json()
        commonest = _pair(body, "##e", "##s")

        assert commonest["count"] == 9
        assert (commonest["left_count"], commonest["right_count"]) == (17, 9)
        assert commonest["score"] == 9 / (17 * 9)

    def test_the_pair_the_score_takes_is_seen_three_times(self) -> None:
        body = client.get(SCORING).json()
        taken = _pair(body, "##i", "##d")

        assert taken["count"] == 3
        assert (taken["left_count"], taken["right_count"]) == (3, 3)
        assert taken["score"] == 1 / 3
        assert body["pairs"][0] == taken

    def test_neither_rule_takes_the_pair_with_the_largest_total_gain(self) -> None:
        """The ratio compares what one occurrence gains, not what all of them do."""
        body = client.get(SCORING).json()
        best = body["four_word_best_total_gain"]

        assert body["four_word_symbol_total"] == 79
        assert (best["left"], best["right"]) == ("##s", "##t")
        assert round(best["total_gain"], 2) == 19.55
        assert round(_pair(body, "##e", "##s")["total_gain"], 2) == 13.83
        assert round(_pair(body, "##i", "##d")["total_gain"], 2) == 9.81

    def test_the_pair_with_the_largest_total_gain_is_fourth_by_the_ratio(self) -> None:
        body = client.get(SCORING).json()
        ordered = [(pair["left"], pair["right"]) for pair in body["pairs"]]

        assert ordered.index(("##s", "##t")) == 3

    def test_the_two_orders_part_at_the_very_first_merge_on_both_corpora(self) -> None:
        body = client.get(SCORING).json()

        assert [order["first_difference"] for order in body["orders"]] == [0, 0]

    def test_the_first_merge_under_each_rule_on_the_four_words(self) -> None:
        body = client.get(SCORING).json()
        four = body["orders"][0]

        assert four["by_count"][0]["merged"] == "##es"
        assert four["by_score"][0]["merged"] == "##id"
        assert [row["merged"] for row in four["by_score"][:3]] == [
            "##id",
            "wid",
            "lo",
        ]

    def test_twelve_merges_each_way_share_seven_pieces_on_the_four_words(
        self,
    ) -> None:
        body = client.get(SCORING).json()
        four = body["orders"][0]
        by_count = {row["merged"] for row in four["by_count"]}
        by_score = {row["merged"] for row in four["by_score"]}

        assert len(by_count) == 12
        assert len(by_score) == 12
        assert len(by_count & by_score) == 7

    def test_the_first_merge_under_each_rule_on_the_sentences(self) -> None:
        body = client.get(SCORING).json()
        sentences = body["orders"][1]

        assert sentences["by_count"][0]["merged"] == "##he"
        assert sentences["by_count"][0]["count"] == 32
        assert sentences["by_score"][0]["merged"] == "of"
        assert sentences["by_score"][0]["count"] == 3

    def test_the_largest_count_on_the_sentences_ranks_thirty_second_by_score(
        self,
    ) -> None:
        body = client.get(SCORING).json()

        assert body["sentence_n_candidates"] == 87
        assert body["sentence_largest_count_rank"] == 32
        assert body["sentence_largest_count"]["count"] == 32
        assert body["sentence_largest_count"]["left_count"] == 36
        assert body["sentence_largest_count"]["right_count"] == 98
        assert body["sentence_largest_count"]["score"] == 32 / (36 * 98)
        assert body["sentence_best_score"]["score"] == 1 / 6

    def test_the_tie_is_two_pairs_of_symbols_never_seen_apart(self) -> None:
        body = client.get(SCORING).json()
        tie = body["tie"]

        assert len(tie["tied"]) == 2
        assert {pair["score"] for pair in tie["tied"]} == {1 / 3}
        assert tie["taken"] == "##i + ##d"
        assert tie["rejected"] == "w + ##i"

    def test_the_tie_is_decided_by_two_codepoints(self) -> None:
        body = client.get(SCORING).json()
        tie = body["tie"]

        assert tie["rival_character"] == "i"
        assert tie["rival_codepoint"] == 105
        assert tie["word_start_mark_codepoint"] == 9601
        assert tie["word_start_mark_codepoint"] > tie["rival_codepoint"]

    def test_counting_shortens_the_four_words_faster_at_every_budget(self) -> None:
        body = client.get(SCORING).json()
        rows = {row["asked"]: row for row in body["lengths"][0]["rows"]}

        assert body["lengths"][0]["start_length"] == 79
        assert (rows[5]["by_count_length"], rows[5]["by_score_length"]) == (41, 50)
        assert (rows[10]["by_count_length"], rows[10]["by_score_length"]) == (20, 25)

    def test_both_rules_reach_one_piece_a_word_after_twelve_merges(self) -> None:
        body = client.get(SCORING).json()
        rows = {row["asked"]: row for row in body["lengths"][0]["rows"]}

        assert rows[12]["by_count_length"] == 16
        assert rows[12]["by_score_length"] == 16

    def test_on_the_sentences_counting_runs_out_of_pairs_first(self) -> None:
        body = client.get(SCORING).json()
        rows = {row["asked"]: row for row in body["lengths"][1]["rows"]}

        assert body["lengths"][1]["start_length"] == 648
        assert (rows[60]["by_count_length"], rows[60]["by_score_length"]) == (302, 411)
        assert (rows[108]["by_count_learned"], rows[108]["by_score_learned"]) == (
            84,
            107,
        )
        assert (rows[108]["by_count_length"], rows[108]["by_score_length"]) == (
            254,
            256,
        )


class TestSpelling:
    def test_the_two_markings_give_two_alphabets(self) -> None:
        body = client.get(SPELLING).json()

        assert body["distinct_characters"] == 35
        assert len(body["marked_alphabet"]) == 49
        assert body["ended_alphabet_size"] == 51
        assert body["character_pieces"] == 648

    def test_the_small_corpus_gives_eleven_symbols_either_way(self) -> None:
        body = client.get(SPELLING).json()

        assert len(body["four_word_marked_alphabet"]) == 11
        assert len(body["four_word_ended_alphabet"]) == 11

    def test_the_corpus_the_page_learns_from(self) -> None:
        body = client.get(SPELLING).json()

        assert body["sentence"] == SENTENCE
        assert len(body["corpus"]) == 18
        assert body["n_distinct_words"] == 72
        assert body["n_word_occurrences"] == 133

    def test_thirteen_characters_never_open_a_word_and_eighteen_never_end_one(
        self,
    ) -> None:
        body = client.get(SPELLING).json()

        assert len(body["never_opens"]) == 13
        assert len(body["never_ends"]) == 18
        assert "z" in body["never_opens"]
        assert "z" in body["never_ends"]

    def test_the_sweep_costs_more_pieces_than_the_count_scored_fit(self) -> None:
        body = client.get(SPELLING).json()
        rows = {row["asked"]: row for row in body["rows"]}
        counted = {row["asked"]: row for row in body["count_rows"]}

        assert (rows[130]["corpus_pieces"], counted[130]["corpus_pieces"]) == (343, 277)
        assert (rows[130]["sentence_pieces"], counted[130]["sentence_pieces"]) == (
            31,
            25,
        )

    def test_at_exhaustion_the_corpus_turns_and_the_sentence_does_not(self) -> None:
        body = client.get(SPELLING).json()
        rows = {row["asked"]: row for row in body["rows"]}
        counted = {row["asked"]: row for row in body["count_rows"]}

        assert (rows[200]["learned"], counted[200]["learned"]) == (157, 137)
        assert (rows[200]["corpus_pieces"], counted[200]["corpus_pieces"]) == (254, 263)
        assert (rows[200]["sentence_pieces"], counted[200]["sentence_pieces"]) == (
            29,
            25,
        )

    def test_the_marked_fit_gives_the_sentence_back_and_the_other_never_does(
        self,
    ) -> None:
        body = client.get(SPELLING).json()

        assert all(row["round_trip_exact"] for row in body["rows"])
        assert all(row["n_unknown"] == 0 for row in body["rows"])
        assert not any(row["round_trip_exact"] for row in body["count_rows"])
        assert all(row["n_unknown"] == 1 for row in body["count_rows"])

    def test_the_sentence_at_the_largest_size(self) -> None:
        body = client.get(SPELLING).json()
        largest = body["cuts"][-1]

        assert largest["n_pieces"] == 29
        assert largest["pieces"][0] == "Dr."
        assert largest["pieces"][1] == "A"
        assert "##ar" in largest["pieces"]
        assert largest["round_trip_exact"]

    def test_every_piece_after_a_word_s_first_carries_the_mark(self) -> None:
        body = client.get(SPELLING).json()
        pieces = body["cuts"][-1]["pieces"]

        assert pieces[1] == "A"
        assert all(piece.startswith("##") for piece in pieces[2:7])

    def test_a_name_ending_in_a_letter_no_word_ends_in(self) -> None:
        body = client.get(SPELLING).json()
        alvarez = next(row for row in body["comparisons"] if row["word"] == "Alvarez")

        assert alvarez["marked_unknown"] is False
        assert alvarez["marked"] == ["A", "##l", "##v", "##ar", "##e", "##z"]
        assert alvarez["ended_unknown"] is True

    def test_and_a_word_beginning_with_a_letter_no_word_begins_with(self) -> None:
        body = client.get(SPELLING).json()
        zebra = next(row for row in body["comparisons"] if row["word"] == "zebra")

        assert zebra["marked_unknown"] is True
        assert zebra["marked"] == ["[UNK]"]
        assert zebra["ended_unknown"] is False

    def test_a_word_it_cannot_spell_is_lost_whole(self) -> None:
        body = client.get(SPELLING).json()
        lox = next(row for row in body["four_word_comparisons"] if row["word"] == "lox")

        assert lox["marked"] == ["[UNK]"]
        assert lox["ended"] == ["lo", "[UNK]"]


class TestLimits:
    def test_a_pair_whose_halves_never_occur_apart_scores_one_over_its_count(
        self,
    ) -> None:
        body = client.get(LIMITS).json()
        rows = {row["repeats"]: row for row in body["bound_rows"]}

        assert rows[2]["score"] == 1 / 2
        assert rows[9]["score"] == 1 / 9

    def test_so_the_rarer_of_two_bound_pairs_is_taken(self) -> None:
        body = client.get(LIMITS).json()
        rows = {row["repeats"]: row for row in body["bound_rows"]}

        assert rows[9]["score"] < rows[9]["rival_score"]
        assert rows[9]["taken"] == "a + ##b"

    def test_the_cut_never_refused_a_word_another_cut_could_spell(self) -> None:
        body = client.get(LIMITS).json()
        greedy = body["greedy"]

        assert greedy["n_words"] == 11110
        assert greedy["n_cut"] == 1755
        assert greedy["n_refused"] == 9355
        assert greedy["n_refused_with_a_cut_available"] == 0

    def test_every_refusal_is_a_missing_symbol_rather_than_a_bad_division(
        self,
    ) -> None:
        body = client.get(LIMITS).json()
        greedy = body["greedy"]

        assert greedy["n_refused_at_the_first_letter"] == 7777
        assert greedy["n_refused_later"] == 1578
        assert greedy["missing_continuations"] == ["l", "n"]
        assert greedy["opening_pieces"][0] == "l"

    def test_the_fewest_times_rule_read_as_a_filter_learns_ten_merges(self) -> None:
        body = client.get(LIMITS).json()
        minimum = body["minimum"]

        assert minimum["minimum"] == 3
        assert len(minimum["as_filter"]) == 10
        assert minimum["as_filter"][:3] == ["##id", "wid", "lo"]

    def test_and_read_as_a_stopping_test_it_learns_five_and_quits(self) -> None:
        body = client.get(LIMITS).json()
        minimum = body["minimum"]

        assert minimum["as_stopping_test"] == ["##id", "wid", "lo", "##st", "low"]
        assert minimum["stopped_on"] == "##e + ##r"
        assert minimum["stopped_on_count"] == 2
        assert minimum["left_unmerged"] == [
            "##est",
            "##ew",
            "new",
            "newest",
            "widest",
        ]

    def test_the_vocabulary_is_cheap_on_its_own_corpus_and_dear_elsewhere(self) -> None:
        body = client.get(LIMITS).json()
        carried = body["carried"]

        assert carried["learned"] == 157
        assert carried["corpus_pieces"] == 254
        assert round(carried["corpus_per_word"], 2) == 1.91
        assert round(carried["sentence_per_word"], 2) == 4.14
        assert carried["neighbour_pieces"] == 19
        assert round(carried["neighbour_per_word"], 3) == 2.375

    def test_whole_corpus_words_reach_the_vocabulary_as_single_pieces(self) -> None:
        body = client.get(LIMITS).json()

        assert "analysis" in body["carried"]["whole_word_pieces"]
        assert "expected" in body["carried"]["whole_word_pieces"]


class TestFit:
    def test_the_small_corpus_at_ten_merges(self) -> None:
        body = client.post(
            FIT,
            json={"corpus": "four_words", "vocabulary_size": 22, "text": "lowest"},
        ).json()

        assert body["learned"] == 22
        assert body["n_merges"] == 10
        assert body["pieces"] == ["low", "##est"]
        assert body["round_trip_exact"] is True
        assert body["smallest_size"] == 12

    def test_the_merges_carry_both_figures(self) -> None:
        body = client.post(
            FIT, json={"corpus": "four_words", "vocabulary_size": 22, "text": "low"}
        ).json()

        assert body["merges"][0]["merged"] == "##id"
        assert body["merges"][0]["count"] == 3
        assert body["merges"][0]["score"] == 1 / 3

    def test_the_count_scored_fit_comes_back_beside_it(self) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 130}
        ).json()

        assert body["n_pieces"] == 31
        assert body["count_n_pieces"] == 25
        assert body["corpus_pieces"] == 343
        assert body["count_corpus_pieces"] == 277

    def test_below_the_count_scored_floor_only_one_fit_comes_back(self) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 51}
        ).json()

        assert body["learned"] == 51
        assert body["count_pieces"] is None
        assert body["count_corpus_pieces"] is None

    def test_a_size_below_the_alphabet_is_refused(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 20})

        assert response.status_code == 422
        assert "50" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(FIT, json={"vocab_size": 100})

        assert response.status_code == 422

    def test_a_size_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 5000}
        )

        assert response.status_code == 422

    def test_a_text_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={"corpus": "sentences", "vocabulary_size": 130, "text": "a" * 500},
        )

        assert response.status_code == 422


IDENTIFIER = re.compile(
    r"[A-Za-z]+_[A-Za-z_]+|\b[a-z]+[A-Z][A-Za-z]*\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b"
)
"""Anything shaped like a code name: an underscored word, or camel case."""


class TestNoIdentifierReachesThePage:
    def test_the_too_small_vocabulary_refusal_is_in_plain_words(self) -> None:
        detail = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 20}
        ).json()["detail"]

        assert IDENTIFIER.search(detail) is None, detail

    def test_no_refusal_this_page_can_reach_carries_an_identifier(self) -> None:
        for size in (2, 11, 20, 49, 51):
            for corpus in ("four_words", "sentences"):
                for minimum in (1, 2, 20):
                    response = client.post(
                        FIT,
                        json={
                            "corpus": corpus,
                            "vocabulary_size": size,
                            "minimum_pair_frequency": minimum,
                            "text": "lowest ##x",
                        },
                    )
                    if response.status_code != 422:
                        continue
                    detail = response.json()["detail"]
                    if not isinstance(detail, str):
                        continue
                    assert IDENTIFIER.search(detail) is None, detail

    def test_no_static_endpoint_returns_an_identifier_in_a_free_string(self) -> None:
        for path in (SCORING, SPELLING, LIMITS):
            body = client.get(path).json()
            for field in ("taken", "rejected"):
                if field in body.get("tie", {}):
                    assert IDENTIFIER.search(body["tie"][field]) is None
            if "minimum" in body:
                assert IDENTIFIER.search(body["minimum"]["stopped_on"]) is None
