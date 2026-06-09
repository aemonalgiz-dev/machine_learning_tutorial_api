"""The byte pair encoding page's endpoints, on the two corpora the page carries.

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

BASE = "/concepts/byte-pair-encoding"
LEARNING = f"{BASE}/learning"
SIZES = f"{BASE}/sizes"
FALLBACK = f"{BASE}/fallback"
VARIANTS = f"{BASE}/variants"
LIMITS = f"{BASE}/limits"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."


class TestLearning:
    def test_the_four_words_are_spelled_in_eleven_symbols(self) -> None:
        body = client.get(LEARNING).json()

        assert body["alphabet"] == [
            "d",
            "e",
            "i",
            "l",
            "n",
            "o",
            "r</w>",
            "s",
            "t</w>",
            "w",
            "w</w>",
        ]
        assert body["total_occurrences"] == 16
        assert body["character_pieces"] == 79

    def test_the_ten_merges_come_in_the_order_the_counts_give(self) -> None:
        body = client.get(LEARNING).json()

        chosen = [
            (step["left"], step["right"], step["count"])
            for step in body["steps"]
            if step["left"] is not None
        ]
        assert chosen == [
            ("e", "s", 9),
            ("es", "t</w>", 9),
            ("l", "o", 7),
            ("e", "w", 6),
            ("ew", "est</w>", 6),
            ("n", "ewest</w>", 6),
            ("lo", "w</w>", 5),
            ("d", "est</w>", 3),
            ("i", "dest</w>", 3),
            ("w", "idest</w>", 3),
        ]

    def test_the_corpus_falls_from_seventy_nine_pieces_to_twenty_two(self) -> None:
        body = client.get(LEARNING).json()

        counts = [step["corpus_pieces"] for step in body["steps"]]
        assert counts == [79, 70, 61, 54, 48, 42, 36, 31, 28, 25, 22]

    def test_a_word_the_corpus_never_held_takes_three_pieces(self) -> None:
        body = client.get(LEARNING).json()

        assert body["unseen_word"] == "lowest"
        assert body["steps"][0]["unseen_pieces"] == [
            "l",
            "o",
            "w",
            "e",
            "s",
            "t</w>",
        ]
        assert body["steps"][-1]["unseen_pieces"] == ["lo", "w", "est</w>"]

    def test_the_first_step_records_two_pairs_tied_at_nine(self) -> None:
        body = client.get(LEARNING).json()

        assert body["steps"][0]["tied"] == ["e + s", "s + t</w>"]
        assert body["steps"][3]["tied"] == ["e + w", "n + e", "w + est</w>"]

    def test_the_finished_vocabulary_is_the_alphabet_then_the_merges(self) -> None:
        body = client.get(LEARNING).json()

        assert len(body["tokens"]) == 22
        assert body["tokens"][0] == "[UNK]"
        assert body["tokens"][-1] == "widest</w>"


class TestSizes:
    def test_five_of_the_seven_sentence_words_are_absent_from_the_corpus(self) -> None:
        baselines = client.get(SIZES).json()["baselines"]

        assert baselines["n_distinct_words"] == 72
        assert baselines["n_word_occurrences"] == 133
        assert baselines["sentence_words_unseen"] == [
            "Alvarez",
            "didn't",
            "expect",
            "low-cost",
            "re-analysis.",
        ]
        assert len(baselines["sentence_words"]) == 7

    def test_every_character_its_own_piece_costs_forty_five(self) -> None:
        baselines = client.get(SIZES).json()["baselines"]

        assert baselines["sentence_characters"] == 51
        assert baselines["sentence_character_pieces"] == 45
        assert baselines["alphabet_size"] == 51
        assert baselines["corpus_character_pieces"] == 648

    def test_one_symbol_of_the_sentence_has_no_row_of_its_own(self) -> None:
        baselines = client.get(SIZES).json()["baselines"]

        assert baselines["sentence_characters_unseen"] == []
        assert baselines["sentence_symbols_unspellable"] == ["z</w>"]

    def test_the_corpus_runs_out_of_pairs_at_a_hundred_and_thirty_seven(self) -> None:
        body = client.get(SIZES).json()

        assert body["smallest_size"] == 52
        assert body["largest_useful_size"] == 137
        asked_three_hundred = next(row for row in body["rows"] if row["asked"] == 300)
        assert asked_three_hundred["learned"] == 137
        assert asked_three_hundred["n_merges"] == 85

    def test_the_structural_rows_are_the_same_two_hundred_and_fifty_seven(self) -> None:
        body = client.get(SIZES).json()

        assert body["floor_rows"] == 257
        by_size = {row["asked"]: row for row in body["rows"]}
        assert by_size[52]["total_rows"] == 309
        assert by_size[137]["total_rows"] == 394
        assert by_size[300]["total_rows"] == 394

    def test_the_sentence_falls_from_forty_six_pieces_to_twenty_six(self) -> None:
        rows = client.get(SIZES).json()["rows"]

        by_size = {row["asked"]: row for row in rows}
        assert by_size[52]["sentence_pieces"] == 46
        assert by_size[80]["sentence_pieces"] == 29
        assert by_size[137]["sentence_pieces"] == 26
        assert by_size[300]["sentence_pieces"] == 26

    def test_the_corpus_falls_from_six_hundred_and_forty_eight(self) -> None:
        rows = client.get(SIZES).json()["rows"]

        by_size = {row["asked"]: row for row in rows}
        assert by_size[52]["corpus_pieces"] == 648
        assert by_size[80]["corpus_pieces"] == 414
        assert by_size[137]["corpus_pieces"] == 263

    def test_the_hyphen_and_the_final_s_are_cut_where_no_linguist_would(self) -> None:
        cuts = client.get(SIZES).json()["cuts"]

        finished = next(cut for cut in cuts if cut["vocabulary_size"] == 137)
        assert finished["pieces"] == [
            "Dr.</w>",
            "A",
            "l",
            "v",
            "a",
            "re",
            "z",
            "</w>",
            "d",
            "i",
            "d",
            "n",
            "'",
            "t</w>",
            "exp",
            "ec",
            "t</w>",
            "the</w>",
            "lo",
            "w",
            "-",
            "cost</w>",
            "re",
            "-",
            "analysi",
            "s.</w>",
        ]
        assert finished["n_pieces"] == 26

    def test_the_round_trip_is_exact_once_every_character_is_spellable(self) -> None:
        cuts = client.get(SIZES).json()["cuts"]

        finished = next(cut for cut in cuts if cut["vocabulary_size"] == 137)
        assert finished["round_trip_exact"] is True
        assert finished["decoded"] == SENTENCE

    def test_a_full_stop_costs_a_word_the_row_it_already_had(self) -> None:
        readings = {
            reading["word"]: reading["pieces"]
            for reading in client.get(SIZES).json()["neighbour_readings"]
        }

        assert readings["analysis"] == ["analysis</w>"]
        assert readings["analysis."] == ["analysi", "s.</w>"]
        assert readings["cost"] == ["cost</w>"]
        assert readings["low-cost"] == ["lo", "w", "-", "cost</w>"]

    def test_a_longer_word_the_corpus_held_is_cheaper_than_its_own_stem(
        self,
    ) -> None:
        readings = {
            reading["word"]: reading["pieces"]
            for reading in client.get(SIZES).json()["neighbour_readings"]
        }

        assert readings["expected"] == ["expected</w>"]
        assert readings["expect"] == ["exp", "ec", "t</w>"]


class TestFallback:
    def test_the_running_sentence_costs_one_piece_and_no_byte_row(self) -> None:
        body = client.get(FALLBACK).json()
        readings = _readings(body, "sentence")

        assert readings["published"]["n_pieces"] == 25
        assert readings["repaired"]["n_pieces"] == 26
        assert readings["repaired"]["n_byte_rows"] == 0
        assert readings["repaired"]["n_bare_markers"] == 1
        assert readings["published"]["n_stand_ins"] == 1
        assert readings["repaired"]["n_stand_ins"] == 0

    def test_the_loss_on_the_running_sentence_is_silent_rather_than_refused(
        self,
    ) -> None:
        readings = _readings(client.get(FALLBACK).json(), "sentence")

        assert readings["published"]["round_trip_exact"] is False
        assert readings["published"]["decoded"] == (
            "Dr. Alvare[UNK]didn't expect the low-cost re-analysis."
        )
        assert readings["repaired"]["round_trip_exact"] is True
        assert readings["repaired"]["decoded"] == SENTENCE

    def test_the_one_symbol_with_no_row_is_a_word_final_z(self) -> None:
        body = client.get(FALLBACK).json()

        assert body["unspellable_symbol"] == "z</w>"
        assert body["unspellable_character"] == "z"
        assert body["stand_in"] == "[UNK]"
        assert body["repaired_spelling"] == ["z", "</w>"]
        assert body["byte_spelled_character"] == "é"
        assert body["byte_spelling"] == ["<0xC3>", "<0xA9>"]

    def test_an_accented_sentence_costs_two_pieces_and_three_byte_rows(self) -> None:
        body = client.get(FALLBACK).json()
        accented = next(text for text in body["texts"] if text["key"] == "accented")
        readings = _readings(body, "accented")

        assert accented["characters_unseen"] == ["j", "é"]
        assert readings["published"]["n_pieces"] == 14
        assert readings["repaired"]["n_pieces"] == 16
        assert readings["repaired"]["n_byte_rows"] == 3
        assert readings["published"]["decoded"] == (
            "The caf[UNK]re[UNK]ected the re-analysis."
        )
        assert readings["repaired"]["round_trip_exact"] is True

    def test_another_script_doubles_in_length_and_survives(self) -> None:
        body = client.get(FALLBACK).json()
        greek = next(text for text in body["texts"] if text["key"] == "greek")
        readings = _readings(body, "greek")

        assert greek["n_characters"] == 25
        assert len(greek["characters_unseen"]) == 15
        assert readings["published"]["n_pieces"] == 22
        assert readings["published"]["n_stand_ins"] == 21
        assert readings["published"]["round_trip_exact"] is False
        assert readings["repaired"]["n_pieces"] == 46
        assert readings["repaired"]["n_byte_rows"] == 42
        assert readings["repaired"]["round_trip_exact"] is True

    def test_the_repair_costs_the_fitted_corpus_nothing(self) -> None:
        body = client.get(FALLBACK).json()

        assert body["corpus_pieces"] == body["corpus_pieces_published"] == 263

    def test_the_floor_is_the_same_two_hundred_and_fifty_seven_at_every_size(
        self,
    ) -> None:
        rows = {row["asked"]: row for row in client.get(FALLBACK).json()["accounting"]}

        assert body_of(rows[52]) == (0, 52, 257, 309)
        assert body_of(rows[54]) == (2, 54, 257, 311)
        assert body_of(rows[137]) == (85, 137, 257, 394)
        assert body_of(rows[300]) == (85, 137, 257, 394)


def body_of(row: dict[str, int]) -> tuple[int, int, int, int]:
    """One accounting row as merges, taught rows, floor rows and the total."""
    return (row["n_merges"], row["learned"], row["floor_rows"], row["total_rows"])


def _readings(body: dict, key: str) -> dict[str, dict]:
    """The two readings of one text, by scheme."""
    text = next(item for item in body["texts"] if item["key"] == key)
    return {reading["key"]: reading for reading in text["readings"]}


class TestVariants:
    def test_the_byte_alphabet_pays_three_more_pieces_and_gains_nothing(self) -> None:
        fits = {fit["key"]: fit for fit in client.get(VARIANTS).json()["fits"]}

        assert fits["plain"]["alphabet_size"] == 51
        assert fits["plain"]["floor_rows"] == 257
        assert fits["bytes"]["alphabet_size"] == 256
        assert fits["bytes"]["floor_rows"] == 0
        assert fits["plain"]["n_merges"] == fits["bytes"]["n_merges"] == 85
        assert fits["plain"]["sentence"]["n_pieces"] == 26
        assert fits["bytes"]["sentence"]["n_pieces"] == 29
        assert fits["plain"]["sentence"]["round_trip_exact"] is True
        assert fits["bytes"]["sentence"]["round_trip_exact"] is True

    def test_the_two_alphabets_read_another_script_at_the_same_length(self) -> None:
        fits = {fit["key"]: fit for fit in client.get(VARIANTS).json()["fits"]}

        assert fits["plain"]["greek"]["n_pieces"] == 46
        assert fits["bytes"]["greek"]["n_pieces"] == 46
        assert fits["plain"]["greek"]["round_trip_exact"] is True
        assert fits["bytes"]["greek"]["round_trip_exact"] is True

    def test_the_byte_alphabet_costs_more_on_the_corpus_too(self) -> None:
        fits = {fit["key"]: fit for fit in client.get(VARIANTS).json()["fits"]}

        assert fits["plain"]["corpus_pieces"] == 263
        assert fits["bytes"]["corpus_pieces"] == 282

    def test_the_byte_alphabet_holds_far_more_than_the_corpus_uses(self) -> None:
        body = client.get(VARIANTS).json()

        assert body["corpus_distinct_characters"] == 36

    def test_an_unseen_character_costs_a_piece_more_than_the_byte_alphabet(
        self,
    ) -> None:
        body = client.get(VARIANTS).json()

        assert body["foreign_character"] == "é"
        assert body["foreign_character_bytes"] == [195, 169]
        assert body["published_foreign_word"] == ["c", "a", "f", "[UNK]"]
        assert body["plain_foreign_word"] == [
            "c",
            "a",
            "f",
            "<0xC3>",
            "<0xA9>",
            "</w>",
        ]
        assert body["byte_foreign_word"] == ["c", "a", "f", "Ã", "©"]

    def test_the_foreign_sentence_survives_both_alphabets(self) -> None:
        fits = {fit["key"]: fit for fit in client.get(VARIANTS).json()["fits"]}

        assert fits["plain"]["foreign"]["round_trip_exact"] is True
        assert fits["plain"]["foreign"]["n_pieces"] == 16
        assert fits["bytes"]["foreign"]["round_trip_exact"] is True
        assert fits["bytes"]["foreign"]["n_pieces"] == 17

    def test_four_words_give_repeated_words_rather_than_phrases(self) -> None:
        superwords = client.get(VARIANTS).json()["superwords"]

        small = next(
            item for item in superwords if item["corpus_label"] == "four words"
        )
        assert small["n_word_merges"] == 10
        assert small["n_superword_merges"] == 3
        assert small["superword_tokens"] == [
            "newest</w>newest</w>",
            "low</w>low</w>",
        ]

    def test_a_corpus_of_sentences_gives_phrases_instead(self) -> None:
        superwords = client.get(VARIANTS).json()["superwords"]

        large = next(
            item for item in superwords if item["corpus_label"] == "eighteen sentences"
        )
        assert large["n_superword_merges"] == 8
        assert large["superword_merges"][:2] == [
            "and</w> + the</w>",
            "of</w> + the</w>",
        ]
        assert "an</w>the</w>" in large["superword_tokens"]
        assert large["example_pieces"] == [
            "The</w>",
            "report</w>was</w>",
            "the</w>",
            "reas",
            "on",
            ".</w>",
        ]

    def test_the_constraint_stops_the_fit_four_merges_early(self) -> None:
        morphology = client.get(VARIANTS).json()["morphology"]

        assert len(morphology["plain_merges"]) == 9
        assert len(morphology["constrained_merges"]) == 5
        assert morphology["constrained_merges"] == morphology["plain_merges"][:5]
        assert morphology["constrained_tokens"] == morphology["plain_tokens"][:15]

    def test_the_constraint_doubles_what_the_corpus_costs(self) -> None:
        morphology = client.get(VARIANTS).json()["morphology"]

        assert morphology["plain_cut"] == ["undo</w>"]
        assert morphology["constrained_cut"] == ["un", "do</w>"]
        assert morphology["plain_corpus_pieces"] == 40
        assert morphology["constrained_corpus_pieces"] == 80


class TestLimits:
    def test_the_first_tie_changes_one_token_and_no_piece_count(self) -> None:
        first = client.get(LIMITS).json()["ties"][0]

        assert first["count"] == 9
        assert [branch["pair"] for branch in first["branches"]] == [
            "e + s",
            "s + t</w>",
        ]
        assert [branch["chosen_by_the_rule"] for branch in first["branches"]] == [
            True,
            False,
        ]
        assert {branch["corpus_pieces"] for branch in first["branches"]} == {22}
        differing = [set(branch["tokens"]) for branch in first["branches"]]
        assert differing[0] - differing[1] == {"es"}
        assert differing[1] - differing[0] == {"st</w>"}

    def test_the_three_way_tie_changes_how_an_unseen_word_is_cut(self) -> None:
        second = client.get(LIMITS).json()["ties"][1]

        assert second["count"] == 6
        cuts = {
            branch["pair"]: branch["unseen_pieces"] for branch in second["branches"]
        }
        assert cuts["e + w"] == ["lo", "w", "est</w>"]
        assert cuts["n + e"] == ["lo", "w", "est</w>"]
        assert cuts["w + est</w>"] == ["lo", "west</w>"]
        assert {branch["corpus_pieces"] for branch in second["branches"]} == {22}

    def test_the_three_words_open_on_a_three_way_tie_at_eighteen(self) -> None:
        body = client.get(LIMITS).json()

        assert body["trap_start_pieces"] == 102
        assert body["trap_first_tie_count"] == 18
        assert body["trap_first_tie"] == ["s + t", "t + a", "t + e</w>"]

    def test_greedy_ties_at_one_merge_and_loses_by_eight_at_two(self) -> None:
        rows = {row["budget"]: row for row in client.get(LIMITS).json()["greedy_rows"]}

        assert rows[1]["greedy_pieces"] == rows[1]["best_pieces"] == 84
        assert rows[2]["greedy_pieces"] == 74
        assert rows[2]["best_pieces"] == 66
        assert rows[2]["greedy_merges"] == ["s + t", "a + t"]
        assert rows[2]["best_merges"] == ["t + a", "t + e</w>"]

    def test_the_gap_stays_eight_out_to_four_merges(self) -> None:
        rows = {row["budget"]: row for row in client.get(LIMITS).json()["greedy_rows"]}

        assert [rows[budget]["greedy_pieces"] for budget in (1, 2, 3, 4)] == [
            84,
            74,
            64,
            54,
        ]
        assert [rows[budget]["best_pieces"] for budget in (1, 2, 3, 4)] == [
            84,
            66,
            56,
            46,
        ]

    def test_an_unspellable_last_symbol_loses_the_word_boundary(self) -> None:
        body = client.get(LIMITS).json()

        assert body["lost_boundary_pieces"][5] == "[UNK]"
        assert body["lost_boundary_decoded"] == "Alvare[UNK]didn't"

    def test_spelling_the_letter_gives_the_boundary_its_own_piece(self) -> None:
        body = client.get(LIMITS).json()

        assert body["lost_boundary_repaired_pieces"][5:7] == ["z", "</w>"]
        assert body["lost_boundary_repaired_decoded"] == "Alvarez didn't"


class TestFit:
    def test_the_four_words_at_sixteen_tokens_learn_five_merges(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four_words",
                "vocabulary_size": 16,
                "text": "lowest",
            },
        ).json()

        assert body["smallest_size"] == 12
        assert body["learned"] == 16
        assert body["n_merges"] == 4
        assert body["pieces"] == ["lo", "w", "est</w>"]
        assert body["corpus_character_pieces"] == 79
        assert body["corpus_pieces"] == 48

    def test_the_sentences_at_the_full_size_read_the_running_sentence(self) -> None:
        body = client.post(
            FIT,
            json={"corpus": "sentences", "vocabulary_size": 137, "text": SENTENCE},
        ).json()

        assert body["n_pieces"] == 26
        assert body["unknown_pieces"] == 0
        assert body["round_trip_exact"] is True
        assert body["learned"] == 137
        assert body["total_rows"] == 394
        assert body["merges"][0] == {
            "rank": 0,
            "left": "h",
            "right": "e</w>",
            "merged": "he</w>",
            "count": 31,
        }
        assert body["merges"][1]["merged"] == "the</w>"
        assert body["merges"][1]["count"] == 19
        assert body["merges"][2] == {
            "rank": 2,
            "left": "r",
            "right": "e",
            "merged": "re",
            "count": 17,
        }

    def test_asking_for_more_than_the_corpus_can_give_learns_no_more(self) -> None:
        body = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 600}
        ).json()

        assert body["asked"] == 600
        assert body["learned"] == 137
        assert body["n_merges"] == 85

    def test_the_published_method_loses_the_running_sentence(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "sentences",
                "vocabulary_size": 137,
                "text": SENTENCE,
                "byte_fallback": False,
            },
        ).json()

        assert body["n_pieces"] == 25
        assert body["unknown_pieces"] == 1
        assert body["round_trip_exact"] is False
        assert body["decoded"] == (
            "Dr. Alvare[UNK]didn't expect the low-cost re-analysis."
        )
        assert body["learned"] == body["total_rows"] == 137
        assert body["floor_rows"] == 0


class TestRefusals:
    def test_a_vocabulary_below_the_alphabet_is_refused(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 20})

        assert response.status_code == 422
        assert "the smallest workable size is 52" in response.json()["detail"]

    def test_an_unknown_corpus_is_refused_before_any_fitting(self) -> None:
        response = client.post(
            FIT, json={"corpus": "shakespeare", "vocabulary_size": 100}
        )

        assert response.status_code == 422

    def test_an_extra_field_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={"corpus": "sentences", "vocabulary_size": 100, "seed": 3},
        )

        assert response.status_code == 422

    def test_a_size_beyond_the_ceiling_is_refused(self) -> None:
        response = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 5000}
        )

        assert response.status_code == 422


IDENTIFIER = re.compile(
    r"[A-Za-z]+_[A-Za-z_]+|\b[a-z]+[A-Z][A-Za-z]*\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b"
)
"""Anything shaped like a code name: an underscored word, or camel case."""


class TestNoIdentifierReachesThePage:
    def test_the_refusal_names_nothing_a_reader_has_no_use_for(self) -> None:
        detail = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 20}
        ).json()["detail"]

        assert IDENTIFIER.search(detail) is None, detail

    def test_no_refusal_this_page_can_reach_carries_an_identifier(self) -> None:
        for size in (2, 11, 20, 51):
            for corpus in ("four_words", "sentences"):
                response = client.post(
                    FIT, json={"corpus": corpus, "vocabulary_size": size}
                )
                if response.status_code != 422:
                    continue
                detail = response.json()["detail"]
                if not isinstance(detail, str):
                    continue
                assert IDENTIFIER.search(detail) is None, detail
