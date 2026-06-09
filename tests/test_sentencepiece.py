"""The SentencePiece page's endpoints, on the three corpora the page carries.

Every number the page quotes in prose is pinned here, so a change to the
library that moved one of them would fail a test rather than quietly rewrite
the page. The refusal tests are the other half: the page may not show a
parameter name, and the last class asserts that nothing identifier-shaped
survives the rewriting of a refusal the page's own controls can reach.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/sentencepiece"
MARKING = f"{BASE}/marking"
LEARNING = f"{BASE}/learning"
COSTS = f"{BASE}/costs"
LIMITS = f"{BASE}/limits"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
MARK = "▁"


def _scheme(body: dict, key: str) -> dict:
    return next(scheme for scheme in body["schemes"] if scheme["key"] == key)


def _row(body: dict, asked: int) -> dict:
    return next(row for row in body["rows"] if row["asked"] == asked)


class TestMarking:
    def test_the_mark_is_the_lower_one_eighth_block(self) -> None:
        body = client.get(MARKING).json()

        assert body["mark"] == MARK
        assert body["mark_codepoint"] == 9601

    def test_the_sentence_becomes_one_stream_with_the_spaces_written_in(self) -> None:
        body = client.get(MARKING).json()

        assert body["sentence"] == SENTENCE
        assert body["marked_sentence"] == (
            "▁Dr.▁Alvarez▁didn't▁expect▁the▁low-cost▁re-analysis."
        )
        assert body["marked_units"] == [
            "▁Dr.",
            "▁Alvarez",
            "▁didn't",
            "▁expect",
            "▁the",
            "▁low-cost",
            "▁re-analysis.",
        ]

    def test_the_corpus_the_page_learns_from(self) -> None:
        body = client.get(MARKING).json()

        assert len(body["corpus"]) == 18
        assert body["n_distinct_words"] == 72
        assert body["n_word_occurrences"] == 133
        assert body["distinct_characters"] == 36

    def test_marking_the_front_costs_one_symbol_and_marking_the_end_costs_sixteen(
        self,
    ) -> None:
        body = client.get(MARKING).json()

        assert _scheme(body, "front")["alphabet_size"] == 36
        assert _scheme(body, "front")["floor"] == 37
        assert _scheme(body, "end")["alphabet_size"] == 51
        assert _scheme(body, "end")["floor"] == 52

    def test_the_mark_is_among_the_front_alphabet_and_no_letter_is_doubled(
        self,
    ) -> None:
        body = client.get(MARKING).json()
        alphabet = _scheme(body, "front")["alphabet"]

        assert MARK in alphabet
        assert len(set(alphabet)) == len(alphabet)
        assert all(len(symbol) == 1 for symbol in alphabet)

    def test_only_the_front_marked_fit_gives_the_sentence_back(self) -> None:
        body = client.get(MARKING).json()
        front = _scheme(body, "front")["sentence"]
        end = _scheme(body, "end")["sentence"]

        assert front["round_trip_exact"] is True
        assert front["n_unspellable"] == 0
        assert end["round_trip_exact"] is False
        assert end["n_unspellable"] == 1
        assert (
            end["decoded"] == "Dr. Alvare[UNK]didn't expect the low-cost re-analysis."
        )

    def test_and_it_pays_three_pieces_for_that(self) -> None:
        body = client.get(MARKING).json()

        assert _scheme(body, "front")["sentence"]["n_pieces"] == 28
        assert _scheme(body, "end")["sentence"]["n_pieces"] == 25
        assert _scheme(body, "front")["corpus_pieces"] == 261
        assert _scheme(body, "end")["corpus_pieces"] == 263

    def test_a_name_ending_in_a_letter_no_corpus_word_ends_in(self) -> None:
        body = client.get(MARKING).json()
        alvarez = next(row for row in body["word_pairs"] if row["word"] == "Alvarez")

        assert alvarez["front_pieces"][-1] == "z"
        assert alvarez["end_pieces"][-1] == "[UNK]"

    def test_a_corpus_word_is_one_piece_either_way_and_its_inside_is_not(self) -> None:
        body = client.get(MARKING).json()
        whole = next(row for row in body["word_pairs"] if row["word"] == "analysis")
        inside = next(
            row for row in body["word_pairs"] if row["word"] == "re-analysis."
        )

        assert whole["front_pieces"] == ["▁analysis"]
        assert whole["end_pieces"] == ["analysis</w>"]
        assert inside["front_pieces"] == ["▁re", "-", "an", "aly", "sis", "."]
        assert inside["end_pieces"] == ["re", "-", "analysi", "s.</w>"]

    def test_the_four_texts_whose_round_trip_is_not_exact(self) -> None:
        body = client.get(MARKING).json()
        losses = {row["text"]: row for row in body["losses"]}

        assert losses["low  lower"]["decoded"] == "low lower"
        assert losses[" low lower "]["decoded"] == "low lower"
        assert losses["low\tlower"]["decoded"] == "low lower"
        assert losses[f"low {MARK} lower"]["decoded"] == "low   lower"
        assert losses[f"low {MARK} lower"]["pieces"] == [
            "▁low",
            MARK,
            MARK,
            "▁lower",
        ]

    def test_a_whitespace_rule_finds_one_word_per_line_without_spaces(self) -> None:
        body = client.get(MARKING).json()

        assert len(body["unspaced_corpus"]) == 5
        assert body["unspaced_words_found"] == 5


class TestLearning:
    def test_the_four_words_start_at_ninety_five_pieces(self) -> None:
        body = client.get(LEARNING).json()

        assert body["words"] == ["low", "lower", "newest", "widest"]
        assert body["counts"] == [5, 2, 6, 3]
        assert body["starting_pieces"] == 95

    def test_fifteen_merges_take_them_to_one_piece_a_word(self) -> None:
        body = client.get(LEARNING).json()

        assert len(body["steps"]) == 15
        assert body["final_pieces"] == 16
        assert body["final_spellings"] == [
            ["▁low"],
            ["▁lower"],
            ["▁newest"],
            ["▁widest"],
        ]

    def test_the_first_merges_build_the_word_before_the_mark_joins_it(self) -> None:
        body = client.get(LEARNING).json()
        chosen = [
            (step["left"], step["right"], step["count"]) for step in body["steps"]
        ]

        assert chosen[:5] == [
            ("e", "s", 9),
            ("es", "t", 9),
            ("l", "o", 7),
            ("lo", "w", 7),
            (MARK, "low", 7),
        ]

    def test_the_mark_loses_two_ties_at_seven_before_it_wins_alone(self) -> None:
        body = client.get(LEARNING).json()
        steps = body["steps"]

        assert steps[2]["tied"] == ["l + o", "o + w", f"{MARK} + l"]
        assert steps[3]["tied"] == ["lo + w", f"{MARK} + lo"]
        assert steps[4]["tied"] == [f"{MARK} + low"]
        assert all(step["count"] == 7 for step in steps[2:5])

    def test_and_the_codepoints_are_what_put_it_last(self) -> None:
        body = client.get(LEARNING).json()

        assert body["rival_character"] == "l"
        assert body["rival_codepoint"] == 108
        assert body["mark_codepoint"] == 9601
        assert body["mark_codepoint"] > body["rival_codepoint"]

    def test_a_mark_pair_tied_eleven_times_and_won_only_when_left_alone(self) -> None:
        body = client.get(LEARNING).json()
        tied = [step for step in body["steps"] if step["mark_pair_was_tied"]]
        won = [step for step in tied if step["left"] == MARK]

        assert len(tied) == 11
        assert len(won) == 3
        assert all(len(step["tied"]) == 1 for step in won)

    def test_the_finished_rows_hold_a_word_and_the_same_word_marked(self) -> None:
        body = client.get(LEARNING).json()
        tokens = set(body["tokens"])
        twins = sorted(
            token for token in tokens if token.startswith(MARK) and token[1:] in tokens
        )

        assert body["tokens"].count("low") == 1
        assert body["tokens"].count("▁low") == 1
        assert twins == ["▁low", "▁newest", "▁widest"]
        assert len(body["tokens"]) - 12 == 15
        assert body["final_unseen_pieces"] == ["▁low", "est"]

    def test_the_sentences_reach_the_mark_at_the_second_merge(self) -> None:
        body = client.get(LEARNING).json()

        assert body["sentence_merges"][:3] == ["h + e", f"{MARK} + t", f"{MARK}t + he"]
        assert body["sentence_merge_counts"][:3] == [32, 25, 19]
        assert body["first_mark_merge_rank"] == 1

    def test_and_spend_forty_two_of_ninety_nine_merges_on_the_mark(self) -> None:
        body = client.get(LEARNING).json()

        assert body["sentence_n_merges"] == 99
        assert body["sentence_n_mark_merges"] == 42

    def test_seventeen_rows_differ_from_another_row_only_by_the_mark(self) -> None:
        body = client.get(LEARNING).json()

        assert len(body["twin_rows"]) == 17
        assert "▁cost" in body["twin_rows"]
        assert "▁re" in body["twin_rows"]

    def test_the_two_learners_hold_different_rows_and_cut_the_unseen_word_alike(
        self,
    ) -> None:
        body = client.get(LEARNING).json()
        merging, shrinking = body["learners"]

        assert (merging["learned"], shrinking["learned"]) == (27, 30)
        assert body["shared_tokens"] == 23
        assert merging["unseen_pieces"] == shrinking["unseen_pieces"]
        assert merging["unseen_pieces"] == ["▁low", "est"]

    def test_the_shrinking_learner_reads_its_own_corpus_in_one_piece_a_word(
        self,
    ) -> None:
        body = client.get(LEARNING).json()
        merging, shrinking = body["learners"]

        assert (
            merging["sentence_corpus_pieces"],
            shrinking["sentence_corpus_pieces"],
        ) == (
            261,
            133,
        )
        assert (merging["sentence_pieces"], shrinking["sentence_pieces"]) == (28, 37)


class TestCosts:
    def test_the_two_floors_are_fifteen_rows_apart(self) -> None:
        body = client.get(COSTS).json()

        assert body["front_floor"] == 37
        assert body["end_floor"] == 52

    def test_below_fifty_two_only_one_of_the_two_can_be_fitted(self) -> None:
        body = client.get(COSTS).json()

        assert _row(body, 37)["end_learned"] is None
        assert _row(body, 45)["end_learned"] is None
        assert _row(body, 45)["front_sentence_pieces"] == 45
        assert _row(body, 52)["end_learned"] == 52

    def test_the_front_marked_fit_reads_the_corpus_in_fewer_pieces_everywhere(
        self,
    ) -> None:
        body = client.get(COSTS).json()
        both = [row for row in body["rows"] if row["end_corpus_pieces"] is not None]

        assert all(
            row["front_corpus_pieces"] < row["end_corpus_pieces"] for row in both
        )
        assert (
            _row(body, 52)["front_corpus_pieces"],
            _row(body, 52)["end_corpus_pieces"],
        ) == (
            553,
            648,
        )

    def test_and_reads_the_held_out_sentence_in_more_from_seventy_rows_up(self) -> None:
        body = client.get(COSTS).json()

        assert (
            _row(body, 52)["front_sentence_pieces"],
            _row(body, 52)["end_sentence_pieces"],
        ) == (
            43,
            45,
        )
        assert (
            _row(body, 80)["front_sentence_pieces"],
            _row(body, 80)["end_sentence_pieces"],
        ) == (
            34,
            28,
        )
        assert (
            _row(body, 137)["front_sentence_pieces"],
            _row(body, 137)["end_sentence_pieces"],
        ) == (
            28,
            25,
        )

    def test_only_one_of_the_two_ever_gives_the_sentence_back(self) -> None:
        body = client.get(COSTS).json()

        assert all(row["front_exact"] for row in body["rows"])
        assert not any(
            row["end_exact"] for row in body["rows"] if row["end_exact"] is not None
        )

    def test_both_run_out_of_pairs_at_much_the_same_place(self) -> None:
        body = client.get(COSTS).json()

        assert body["front_exhausted"] == 136
        assert body["end_exhausted"] == 137
        assert _row(body, 200)["front_learned"] == 136
        assert _row(body, 200)["end_learned"] == 137

    def test_without_spaces_the_splitting_rule_has_almost_nothing_to_merge(
        self,
    ) -> None:
        body = client.get(COSTS).json()
        front, end = body["unspaced"]

        assert (front["n_merges"], end["n_merges"]) == (6, 3)
        assert (front["learned"], end["learned"]) == (18, 17)
        assert (front["corpus_pieces"], end["corpus_pieces"]) == (15, 18)

    def test_and_reads_a_sentence_of_three_words_as_five_pieces(self) -> None:
        body = client.get(COSTS).json()
        front, end = body["unspaced"]

        assert body["unspaced_text"] == "学生研究生命起源"
        assert front["text_pieces"] == ["▁学生", "研究生命", "起源"]
        assert end["text_pieces"] == ["学生", "研究", "生命", "起", "源</w>"]
        assert (front["n_text_pieces"], end["n_text_pieces"]) == (3, 5)

    def test_three_of_the_end_marked_rows_record_the_end_of_a_line(self) -> None:
        body = client.get(COSTS).json()

        assert body["unspaced_end_final_marks"] == ["命</w>", "源</w>", "究</w>"]


class TestLimits:
    def test_with_nothing_cut_the_very_first_merge_crosses_a_boundary(self) -> None:
        body = client.get(LIMITS).json()
        first = body["uncut_merges"][0]

        assert body["first_uncut_crossing_rank"] == 0
        assert (first["left"], first["right"]) == ("e", MARK)
        assert first["count"] == 39
        assert first["crosses"] is True

    def test_thirteen_of_the_first_forty_hold_a_mark_inside_them(self) -> None:
        body = client.get(LIMITS).json()

        assert body["n_uncut"] == 40
        assert body["n_uncut_crossing"] == 13

    def test_and_two_whole_words_are_joined_by_the_twentieth(self) -> None:
        body = client.get(LIMITS).json()
        merged = [row["merged"] for row in body["uncut_merges"]]

        assert f"{MARK}the{MARK}" in merged
        assert f"d{MARK}the{MARK}" in merged

    def test_a_mark_that_sorts_before_the_letters_builds_the_word_outward(
        self,
    ) -> None:
        body = client.get(LIMITS).json()
        late, early = body["mark_choices"]

        assert (late["mark"], late["codepoint"]) == (MARK, 9601)
        assert (early["mark"], early["codepoint"]) == ("_", 95)
        assert late["sorts_before_the_letters"] is False
        assert early["sorts_before_the_letters"] is True
        assert late["merges"][2:5] == ["l + o", "lo + w", f"{MARK} + low"]
        assert early["merges"][2:5] == ["_ + l", "_l + o", "_lo + w"]

    def test_the_two_marks_give_the_same_count_of_rows_and_nine_different_ones(
        self,
    ) -> None:
        body = client.get(LIMITS).json()
        late, early = body["mark_choices"]

        assert late["learned"] == early["learned"] == 27
        assert body["shared_rows"] == 18
        assert late["unseen_pieces"] == [f"{MARK}low", "est"]
        assert early["unseen_pieces"] == ["_low", "est"]

    def test_a_vocabulary_learned_from_english_cannot_spell_chinese(self) -> None:
        body = client.get(LIMITS).json()

        assert body["foreign_text"] == "学生研究生命起源"
        assert body["foreign_pieces"].count("[UNK]") == 8
        assert body["foreign_decoded"] == "[UNK]" * 8

    def test_seventeen_twin_rows_out_of_a_hundred_and_thirty_six(self) -> None:
        body = client.get(LIMITS).json()

        assert len(body["twin_rows"]) == 17
        assert body["n_sentence_rows"] == 136


class TestFit:
    def test_the_four_words_at_fifteen_merges(self) -> None:
        body = client.post(
            FIT,
            json={"corpus": "four_words", "vocabulary_size": 27, "text": "lowest"},
        ).json()

        assert body["learned"] == 27
        assert body["n_merges"] == 15
        assert body["floor"] == 12
        assert body["marked_text"] == "▁lowest"
        assert body["pieces"] == ["▁low", "est"]
        assert body["round_trip_exact"] is True

    def test_the_sentences_at_a_hundred_rows_answer_beside_the_other_scheme(
        self,
    ) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 100}
        ).json()

        assert body["n_pieces"] == 34
        assert body["round_trip_exact"] is True
        assert body["end_n_pieces"] == 27
        assert body["end_round_trip_exact"] is False

    def test_below_the_other_scheme_s_floor_only_one_answer_comes_back(self) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 40}
        ).json()

        assert body["learned"] == 40
        assert body["end_floor"] == 52
        assert body["end_pieces"] is None
        assert body["end_n_pieces"] is None

    def test_the_unspaced_corpus_marks_the_whole_line_once(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "unspaced",
                "vocabulary_size": 24,
                "text": "学生研究生命起源",
            },
        ).json()

        assert body["marked_text"] == "▁学生研究生命起源"
        assert body["pieces"] == ["▁学生", "研究生命", "起源"]
        assert body["end_n_pieces"] == 5

    def test_the_other_learner_answers_the_same_request(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "sentences",
                "learner": "shrinking",
                "vocabulary_size": 100,
            },
        ).json()

        assert body["learned"] == 100
        assert body["n_merges"] == 0
        assert body["pieces"][0] == "▁Dr."
        assert body["round_trip_exact"] is True

    def test_a_size_below_the_alphabet_is_refused(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 20})

        assert response.status_code == 422
        assert "37" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        assert client.post(FIT, json={"vocab_size": 100}).status_code == 422

    def test_a_size_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 5000}
        )

        assert response.status_code == 422

    def test_a_text_beyond_the_bound_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={"corpus": "sentences", "vocabulary_size": 100, "text": "a" * 500},
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

    def test_no_refusal_the_page_s_controls_can_reach_carries_an_identifier(
        self,
    ) -> None:
        for size in (2, 11, 12, 20, 36, 37):
            for corpus in ("four_words", "sentences", "unspaced"):
                for learner in ("merging", "shrinking"):
                    response = client.post(
                        FIT,
                        json={
                            "corpus": corpus,
                            "learner": learner,
                            "vocabulary_size": size,
                            "text": f"lowest {MARK} low",
                        },
                    )
                    if response.status_code != 422:
                        continue
                    detail = response.json()["detail"]
                    assert IDENTIFIER.search(detail) is None, detail

    def test_no_static_endpoint_returns_an_identifier_in_a_free_string(self) -> None:
        for path in (MARKING, LEARNING, COSTS, LIMITS):
            body = client.get(path).json()
            for row in body.get("losses", []):
                assert IDENTIFIER.search(row["what_was_lost"]) is None
            for scheme in body.get("schemes", []):
                assert IDENTIFIER.search(scheme["label"]) is None
            for entry in body.get("unspaced", []):
                assert IDENTIFIER.search(entry["label"]) is None
            for learner in body.get("learners", []):
                assert IDENTIFIER.search(learner["label"]) is None
