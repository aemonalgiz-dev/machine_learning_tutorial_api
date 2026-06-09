"""The Morfessor page's endpoints, on the three corpora the page carries.

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

BASE = "/concepts/morfessor"
COUNTS = f"{BASE}/counts"
COST = f"{BASE}/cost"
SEARCH = f"{BASE}/search"
STALL = f"{BASE}/stall"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."


def _rung(body: dict, size: int) -> dict:
    for rung in body["ladder"]:
        if rung["size"] == size:
            return rung
    raise AssertionError(f"no rung at {size}")


def _pair(body: dict, word: str) -> dict:
    for pair in body["pairs"]:
        if pair["word"] == word:
            return pair
    raise AssertionError(f"no comparison for {word}")


def _reading(body: dict, method: str, weight: float | None) -> dict:
    for reading in body["readings"]:
        if reading["method"] == method and reading["weight"] == weight:
            return reading
    raise AssertionError(f"no reading for {method} at {weight}")


def _repeat_row(body: dict, repeats: int) -> dict:
    for row in body["repeat_rows"]:
        if row["repeats"] == repeats:
            return row
    raise AssertionError(f"no row at {repeats} repeats")


def _split_set(body: dict, repeats: int) -> dict:
    for candidate in body["split_sets"]:
        if candidate["repeats"] == repeats:
            return candidate
    raise AssertionError(f"no splits at {repeats} repeats")


class TestCounts:
    def test_the_twelve_forms_are_spelled_in_thirteen_characters(self) -> None:
        body = client.get(COUNTS).json()

        assert len(body["words"]) == 12
        assert body["repeats"] == 3
        assert body["n_occurrences"] == 36
        assert body["n_characters"] == 13
        assert body["n_symbols"] == 15
        assert body["smallest_merged_size"] == 16

    def test_the_first_four_merges_all_cut_inside_a_stem(self) -> None:
        """al, alk, la and pla are the first four rows bought by counting."""
        body = client.get(COUNTS).json()
        learned = [rung["learned"] for rung in body["ladder"] if rung["learned"]]

        assert learned[:4] == ["al", "alk", "la", "pla"]

    def test_the_stems_arrive_only_after_nine_merges(self) -> None:
        body = client.get(COUNTS).json()
        learned = [rung["learned"] for rung in body["ladder"] if rung["learned"]]

        assert learned.index("play") == 7
        assert learned.index("talk") == 8
        assert learned.index("walk") == 9

    def test_a_vocabulary_of_nineteen_rows_reads_walking_in_five_pieces(self) -> None:
        body = client.get(COUNTS).json()
        cuts = {cut["word"]: cut["pieces"] for cut in _rung(body, 19)["cuts"]}

        assert cuts["walking"] == ["w", "alk", "i", "n", "g</w>"]
        assert cuts["plays"] == ["p", "la", "y", "s</w>"]

    def test_the_corpus_falls_from_198_pieces_to_63(self) -> None:
        body = client.get(COUNTS).json()

        assert _rung(body, 16)["corpus_pieces"] == 198
        assert _rung(body, 30)["corpus_pieces"] == 63

    def test_thirteen_rows_reach_the_length_thirty_merged_rows_reach(self) -> None:
        body = client.get(COUNTS).json()

        assert body["morph_types"] == 6
        assert body["morph_rows"] == 13
        assert body["morph_pieces"] == 63
        assert body["matched_size"] == 30

    def test_the_matched_merged_vocabulary_holds_whole_words(self) -> None:
        body = client.get(COUNTS).json()

        assert "playing</w>" in body["matched_learned"]
        assert "played</w>" in body["matched_learned"]

    def test_every_training_form_is_a_stem_or_a_stem_and_an_ending(self) -> None:
        body = client.get(COUNTS).json()
        cuts = {cut["word"]: cut["pieces"] for cut in body["morph_cuts"]}

        assert cuts["walking"] == ["walk", "ing"]
        assert cuts["plays"] == ["play", "s"]
        assert cuts["talk"] == ["talk"]

    def test_the_eighteen_sentences_hold_seventy_two_distinct_words(self) -> None:
        body = client.get(COUNTS).json()

        assert body["sentence"] == SENTENCE
        assert body["n_distinct_words"] == 72
        assert body["n_word_occurrences"] == 133
        assert body["merged_size"] == 135

    def test_merging_cuts_the_sentence_inside_two_of_its_words(self) -> None:
        body = client.get(COUNTS).json()

        assert body["merged_sentence_n"] == 25
        assert "analysi" in body["merged_sentence_pieces"]
        assert "s.</w>" in body["merged_sentence_pieces"]
        assert "lo" in body["merged_sentence_pieces"]

    def test_the_two_methods_disagree_about_where_lower_comes_apart(self) -> None:
        body = client.get(COUNTS).json()

        assert _pair(body, "lower")["morph_pieces"] == ["low", "er</w>"]
        assert _pair(body, "lower")["merged_pieces"] == ["lower</w>"]
        assert _pair(body, "lowest")["morph_pieces"] == ["low", "est</w>"]
        assert _pair(body, "lowest")["merged_pieces"] == ["lowe", "st</w>"]
        assert _pair(body, "costs")["morph_pieces"] == ["cost", "s</w>"]
        assert _pair(body, "costs")["merged_pieces"] == ["co", "sts</w>"]

    def test_the_shortest_description_reads_its_own_corpus_in_far_fewer_pieces(
        self,
    ) -> None:
        body = client.get(COUNTS).json()
        described = _reading(body, "shortest_description", 0.5)
        merged = _reading(body, "commonest_pair", None)

        assert described["n_rows"] == merged["n_rows"] == 135
        assert described["corpus_pieces"] == 167
        assert merged["corpus_pieces"] == 267

    def test_and_reads_a_sentence_it_never_saw_far_worse(self) -> None:
        body = client.get(COUNTS).json()
        described = _reading(body, "shortest_description", 0.5)
        merged = _reading(body, "commonest_pair", None)

        assert (described["n_pieces"], described["n_unknown"]) == (28, 18)
        assert (merged["n_pieces"], merged["n_unknown"]) == (25, 1)
        assert described["characters_held"] == 4
        assert merged["characters_held"] == 34

    def test_a_lighter_weight_keeps_the_characters_and_the_round_trip(self) -> None:
        body = client.get(COUNTS).json()
        readable = _reading(body, "shortest_description", 0.25)

        assert readable["characters_held"] == 34
        assert readable["n_characters"] == 35
        assert readable["n_unknown"] == 0
        assert readable["n_pieces"] == 31
        assert readable["round_trip"] is True

    def test_the_one_character_it_still_cannot_spell_alone_is_k(self) -> None:
        """Every word using it is spellable from longer pieces, so it never
        earns a place of its own."""
        body = client.get(COUNTS).json()

        assert _reading(body, "shortest_description", 0.25)["characters_missing"] == [
            "k"
        ]


class TestCost:
    def test_two_words_left_whole(self) -> None:
        body = client.get(COST).json()
        whole = body["two_word_whole"]["breakdown"]

        assert whole["n_types"] == 2
        assert whole["n_tokens"] == 2
        assert whole["lexicon_characters"] == 11
        assert whole["spelling_cost"] == pytest.approx(19.4454, abs=5e-5)
        assert whole["count_cost"] == pytest.approx(0.0, abs=1e-12)
        assert whole["corpus_cost"] == pytest.approx(1.3863, abs=5e-5)
        assert whole["total"] == pytest.approx(20.8317, abs=5e-5)

    def test_two_words_sharing_a_stem(self) -> None:
        body = client.get(COST).json()
        split = body["two_word_split"]["breakdown"]

        assert split["n_tokens"] == 3
        assert split["lexicon_characters"] == 7
        assert split["spelling_cost"] == pytest.approx(12.2351, abs=5e-5)
        assert split["count_cost"] == pytest.approx(0.6931, abs=5e-5)
        assert split["corpus_cost"] == pytest.approx(1.9095, abs=5e-5)
        assert split["total"] == pytest.approx(14.8378, abs=5e-5)

    def test_the_split_saves_six_nats_on_a_corpus_of_two_words(self) -> None:
        body = client.get(COST).json()

        assert body["two_word_saving"] == pytest.approx(5.9939, abs=5e-5)

    def test_a_nat_is_about_one_and_a_half_bits(self) -> None:
        body = client.get(COST).json()

        assert body["nat_in_bits"] == pytest.approx(1.4427, abs=5e-5)

    def test_the_list_column_the_page_prints_is_spelling_plus_counts(self) -> None:
        body = client.get(COST).json()
        lists = [row["breakdown"]["lexicon_cost"] for row in body["lexicons"]]
        texts = [row["breakdown"]["corpus_cost"] for row in body["lexicons"]]

        assert lists[0] == pytest.approx(211.6245, abs=5e-5)
        assert lists[1] == pytest.approx(73.2276, abs=5e-5)
        assert lists[2] == pytest.approx(94.4357, abs=5e-5)
        assert lists[0] - lists[1] == pytest.approx(138.3969, abs=5e-5)
        assert texts[1] - texts[0] == pytest.approx(22.7791, abs=5e-5)

    def test_how_many_symbols_each_list_is_written_in(self) -> None:
        body = client.get(COST).json()
        symbols = [row["breakdown"]["lexicon_characters"] for row in body["lexicons"]]
        types = [row["breakdown"]["n_types"] for row in body["lexicons"]]

        assert symbols == [78, 24, 26]
        assert types == [12, 6, 13]

    def test_the_three_lexicons_of_the_twelve_forms(self) -> None:
        body = client.get(COST).json()
        totals = {row["label"]: row["breakdown"]["total"] for row in body["lexicons"]}

        assert totals["one piece per word"] == pytest.approx(301.0812, abs=5e-5)
        assert totals["three stems and three endings"] == pytest.approx(
            185.4634, abs=5e-5
        )
        assert totals["one piece per character"] == pytest.approx(569.3002, abs=5e-5)

    def test_the_spelling_falls_all_the_way_and_the_text_rises_all_the_way(
        self,
    ) -> None:
        """Which is what makes the middle a minimum rather than an end point."""
        body = client.get(COST).json()
        spelling = [row["breakdown"]["spelling_cost"] for row in body["lexicons"]]
        corpus = [row["breakdown"]["corpus_cost"] for row in body["lexicons"]]

        assert spelling == sorted(spelling, reverse=True)
        assert corpus == sorted(corpus)
        assert spelling[0] == pytest.approx(191.7754, abs=5e-5)
        assert spelling[2] == pytest.approx(51.3662, abs=5e-5)
        assert corpus[0] == pytest.approx(89.4566, abs=5e-5)
        assert corpus[2] == pytest.approx(474.8646, abs=5e-5)

    def test_the_prefixed_corpus_is_dearer_split_than_whole(self) -> None:
        body = client.get(COST).json()

        assert body["prefix_whole"]["breakdown"]["total"] == pytest.approx(
            111.7806, abs=5e-5
        )
        assert body["prefix_split"]["breakdown"]["total"] == pytest.approx(
            148.5946, abs=5e-5
        )
        assert body["prefix_penalty"] == pytest.approx(36.8140, abs=5e-5)

    def test_which_part_pays_and_which_part_does_not(self) -> None:
        """Splitting doubles the references exactly, so the text cost doubles."""
        body = client.get(COST).json()
        whole = body["prefix_whole"]["breakdown"]
        split = body["prefix_split"]["breakdown"]

        assert (whole["n_tokens"], split["n_tokens"]) == (40, 80)
        assert whole["corpus_cost"] == pytest.approx(55.4518, abs=5e-5)
        assert split["corpus_cost"] == pytest.approx(110.9035, abs=5e-5)
        assert split["corpus_cost"] == pytest.approx(2 * whole["corpus_cost"], rel=1e-9)
        assert whole["spelling_cost"] == pytest.approx(47.2085, abs=5e-5)
        assert split["spelling_cost"] == pytest.approx(26.4129, abs=5e-5)
        assert (whole["lexicon_characters"], split["lexicon_characters"]) == (22, 13)
        assert split["count_cost"] - whole["count_cost"] == pytest.approx(
            2.1579, abs=5e-5
        )

    def test_so_the_prefixes_are_never_found(self) -> None:
        body = client.get(COST).json()

        assert [row["text"] for row in body["prefix_found"]] == [
            "redo",
            "retie",
            "undo",
            "untie",
        ]


class TestSearch:
    def test_the_walk_shortens_the_description_by_a_third(self) -> None:
        body = client.get(SEARCH).json()

        assert body["start_cost"] == pytest.approx(301.0812, abs=5e-5)
        assert body["final_cost"] == pytest.approx(185.4634, abs=5e-5)
        assert body["epochs_run"] == 2
        assert body["converged"] is True

    def test_the_six_pieces_and_their_counts(self) -> None:
        body = client.get(SEARCH).json()

        assert {row["text"]: row["count"] for row in body["morphs"]} == {
            "walk": 12,
            "talk": 12,
            "play": 12,
            "s": 9,
            "ed": 9,
            "ing": 9,
        }
        assert body["n_tokens"] == 63

    def test_the_first_pass_already_finds_all_six(self) -> None:
        """The second pass changes nothing and is what lets it stop."""
        body = client.get(SEARCH).json()
        epochs = body["inflection_epochs"]

        assert [row["epoch"] for row in epochs] == [1, 2]
        assert epochs[0]["n_morphs"] == 6
        assert epochs[0]["converged"] is False
        assert epochs[1]["converged"] is True
        assert epochs[1]["gained"] == []
        assert epochs[1]["lost"] == []

    def test_eighteen_sentences_take_eight_passes_and_lose_pieces_as_they_go(
        self,
    ) -> None:
        body = client.get(SEARCH).json()
        epochs = body["sentence_epochs"]

        assert [row["epoch"] for row in epochs] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert epochs[0]["n_morphs"] == 67
        assert epochs[-1]["n_morphs"] == 57
        assert epochs[-1]["converged"] is True
        assert body["sentence_final_morphs"] == 57

    def test_a_pass_gives_pieces_up_as_well_as_taking_them(self) -> None:
        body = client.get(SEARCH).json()
        second = body["sentence_epochs"][1]
        fourth = body["sentence_epochs"][3]

        for gone, kept in (
            ("carries", "carrie"),
            ("dropped", "dropp"),
            ("first", "fir"),
            ("greed", "gre"),
        ):
            assert gone in second["lost"]
            assert kept in second["gained"]

        for name in ("Dr", "Every", "Monday", "Nobody", "Tuesday"):
            assert name in fourth["lost"]

    def test_the_cost_falls_at_every_pass(self) -> None:
        body = client.get(SEARCH).json()
        costs = [row["cost"] for row in body["sentence_epochs"]]

        assert costs == sorted(costs, reverse=True)
        assert costs[0] == pytest.approx(1391.8737, abs=5e-4)
        assert costs[-1] == pytest.approx(976.7130, abs=5e-4)

    def test_an_unseen_word_is_cut_from_what_the_corpus_taught(self) -> None:
        body = client.get(SEARCH).json()
        cuts = {row["word"]: row["pieces"] for row in body["unseen"]}

        assert cuts["walkings"] == ["walk", "ing", "s"]
        assert cuts["stalking"] == ["s", "talk", "ing"]
        assert cuts["splay"] == ["s", "play"]

    def test_a_character_no_piece_covers_is_paid_for_dearly(self) -> None:
        body = client.get(SEARCH).json()
        rows = {row["word"]: row for row in body["unseen"]}

        assert rows["walkings"]["cost"] == pytest.approx(5.5500, abs=5e-5)
        assert rows["walkxed"]["n_unknown"] == 1
        assert rows["walkxed"]["cost"] == pytest.approx(33.6061, abs=5e-5)
        assert rows["jumping"]["n_unknown"] == 4
        assert rows["jumping"]["cost"] == pytest.approx(121.9537, abs=5e-5)


class TestStall:
    def test_the_same_words_more_often_are_left_whole(self) -> None:
        body = client.get(STALL).json()

        assert _repeat_row(body, 3)["n_morphs"] == 6
        assert _repeat_row(body, 4)["n_morphs"] == 6
        assert _repeat_row(body, 5)["n_morphs"] == 12
        assert _repeat_row(body, 6)["n_morphs"] == 12

    def test_the_best_first_move_goes_from_downhill_to_uphill(self) -> None:
        body = client.get(STALL).json()
        deltas = [row["best_delta"] for row in body["repeat_rows"]]

        assert deltas == sorted(deltas)
        assert deltas[2] == pytest.approx(-2.4630, abs=5e-5)
        assert deltas[3] == pytest.approx(-0.3771, abs=5e-5)
        assert deltas[4] == pytest.approx(1.7333, abs=5e-5)
        assert all(row["best_word"] == "played" for row in body["repeat_rows"])

    def test_the_stall_is_over_a_hundred_nats_from_a_shorter_description(self) -> None:
        body = client.get(STALL).json()

        assert body["stalled_cost"] == pytest.approx(367.2274, abs=5e-5)
        assert body["stem_and_suffix_cost"] == pytest.approx(262.9413, abs=5e-5)
        assert body["gap"] == pytest.approx(104.2861, abs=5e-5)

    def test_fifty_four_first_splits_and_nine_of_them_pay_at_three_repeats(
        self,
    ) -> None:
        body = client.get(STALL).json()
        three = _split_set(body, 3)

        assert three["n_splits"] == 54
        assert three["n_downhill"] == 9
        assert three["whole_cost"] == pytest.approx(301.0812, abs=5e-5)

    def test_and_none_of_them_pays_at_five(self) -> None:
        body = client.get(STALL).json()
        five = _split_set(body, 5)

        assert five["n_splits"] == 54
        assert five["n_downhill"] == 0
        assert five["rows"][0]["delta"] == pytest.approx(1.7333, abs=5e-5)
        assert (five["rows"][0]["left"], five["rows"][0]["right"]) == ("play", "ed")

    def test_the_dearest_split_cuts_a_stem_in_half(self) -> None:
        body = client.get(STALL).json()
        dearest = _split_set(body, 3)["rows"][-1]

        assert dearest["word"] == "walks"
        assert dearest["delta"] == pytest.approx(14.2842, abs=5e-5)

    def test_a_lighter_weight_lets_the_search_through(self) -> None:
        body = client.get(STALL).json()
        found = {row["weight"]: row["found_the_morphs"] for row in body["weights"]}

        assert found[0.5] is True
        assert found[0.83] is True
        assert found[0.9] is False
        assert found[1.0] is False
        assert body["escape_cost"] == pytest.approx(169.4115, abs=5e-5)

    def test_the_setting_it_turns_at(self) -> None:
        body = client.get(STALL).json()

        assert body["threshold_below"] == pytest.approx(0.8379, abs=5e-4)
        assert body["threshold_above"] == pytest.approx(0.8379, abs=5e-4)
        assert body["threshold_below"] < body["threshold_above"]


class TestFit:
    def test_the_default_request_reproduces_the_page_numbers(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["n_morphs"] == 6
        assert body["n_rows"] == 13
        assert body["breakdown"]["total"] == pytest.approx(185.4634, abs=5e-5)
        assert body["start_total"] == pytest.approx(301.0812, abs=5e-5)
        assert body["corpus_pieces"] == 63
        assert body["whole_word_pieces"] == 36
        assert body["epochs_run"] == 2
        assert body["converged"] is True

    def test_a_fifth_repeat_leaves_every_word_whole(self) -> None:
        body = client.post(FIT, json={"repeats": 5}).json()

        assert body["n_morphs"] == 12
        assert body["breakdown"]["total"] == pytest.approx(367.2274, abs=5e-5)

    def test_and_a_lighter_weight_recovers_the_six(self) -> None:
        body = client.post(FIT, json={"repeats": 5, "corpus_weight": 0.5}).json()

        assert body["n_morphs"] == 6
        assert body["breakdown"]["total"] == pytest.approx(169.4115, abs=5e-5)

    def test_the_text_is_read_and_glued_back(self) -> None:
        body = client.post(FIT, json={"text": "walking playing"}).json()

        assert body["cut"] == ["walk", "ing</w>", "play", "ing</w>"]
        assert body["n_pieces"] == 4
        assert body["n_unknown"] == 0
        assert body["decoded"] == "walking playing"
        assert body["round_trip_exact"] is True

    def test_a_character_the_pieces_do_not_cover_breaks_the_round_trip(self) -> None:
        body = client.post(FIT, json={"text": "walking and playing"}).json()

        assert body["n_unknown"] == 3
        assert body["round_trip_exact"] is False

    def test_one_word_is_cut_and_priced(self) -> None:
        body = client.post(FIT, json={"word": "stalking"}).json()

        assert body["word_pieces"] == ["s", "talk", "ing"]
        assert body["word_cost"] == pytest.approx(5.5500, abs=5e-5)

    def test_the_prefixed_corpus_keeps_its_words_whole(self) -> None:
        body = client.post(FIT, json={"corpus": "prefixes", "word": "undo"}).json()

        assert [row["text"] for row in body["morphs"]] == [
            "redo",
            "retie",
            "undo",
            "untie",
        ]
        assert body["word_pieces"] == ["undo"]

    def test_the_sentences_at_a_quarter_weight_keep_their_characters(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "sentences",
                "corpus_weight": 0.25,
                "text": SENTENCE,
                "word": "lowest",
            },
        ).json()

        assert body["alphabet_covered"] == 34
        assert body["alphabet_size"] == 35
        assert body["n_unknown"] == 0
        assert body["round_trip_exact"] is True


class TestRefusals:
    def test_an_empty_word_is_refused_in_the_page_s_own_words(self) -> None:
        response = client.post(FIT, json={"word": ""})

        assert response.status_code == 422
        assert response.json()["detail"] == "cannot segment an empty word"

    def test_an_unknown_keyword_is_refused(self) -> None:
        assert client.post(FIT, json={"weight": 2.0}).status_code == 422

    @pytest.mark.parametrize(
        "payload",
        [
            {"corpus_weight": 0.0},
            {"corpus_weight": 40.0},
            {"repeats": 0},
            {"repeats": 99},
            {"corpus": "poetry"},
            {"text": "x" * 500},
            {"word": "y" * 90},
        ],
    )
    def test_a_request_outside_the_bounds_is_refused(self, payload: dict) -> None:
        assert client.post(FIT, json=payload).status_code == 422

    def test_no_refusal_carries_an_identifier(self) -> None:
        """A message that grows one later fails here rather than reaching the page."""
        response = client.post(FIT, json={"word": ""})
        message = response.json()["detail"]

        assert not re.search(r"[a-z]+_[a-z]+", message)
        assert not re.search(r"\b[A-Z][a-z]+[A-Z]", message)
        assert "``" not in message
