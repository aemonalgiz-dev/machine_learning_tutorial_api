"""The word lattice page's endpoints, on the texts and the counts the page carries.

Every number the page quotes in prose is pinned here, so a change underneath that
moved one of them would fail a test rather than quietly rewrite the page.

Three other things are held. The ranking of readings is a second search over the
same candidates, so its winner is checked against the path the segmenter itself
answers with, on every text the page carries; a ranking that drifted from the
recurrence would fail there rather than reach the page as a plausible-looking
list. The count of readings is checked against a brute-force enumeration of every
cut of the short texts, written from the definition rather than from the dynamic
program that computes it. And the last class asserts that nothing
identifier-shaped survives into any string a widget renders, including into the
one refusal the page's own controls can reach.
"""

from __future__ import annotations

import math
import re
from itertools import product

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.word_lattice import (
    LIFE_FAVOURING,
    PARK_COUNTS,
    SPLIT_FAVOURING,
    STUDENT_FAVOURING,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/the-word-lattice"
SETUP = f"{BASE}/setup"
SCENARIOS = f"{BASE}/scenarios"
SCALE = f"{BASE}/scale"
GROWTH = f"{BASE}/growth"
SEGMENT = f"{BASE}/segment"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."
SPACELESS = "Dr.Alvarezdidn'texpectthelow-costre-analysis."
WATER = "thewaterunderthetable"
TABLE = "thetabledownthere"
GARDEN = "thetabledownthereinthegarden"
PARK = "我们在野生动物园玩"
RESEARCH = "研究生命起源"
NAME = "阿尔瓦雷斯研究生命起源"

LIFE_READING = ["研究", "生命", "起源"]
STUDENT_READING = ["研究生", "命", "起源"]
PARK_READING = ["我们", "在", "野生", "动物园", "玩"]

FREQUENCIES = {
    "research": LIFE_FAVOURING,
    "research-student": STUDENT_FAVOURING,
    "park-counted": PARK_COUNTS,
}


def pytest_approx(value: float, tolerance: float = 1e-4) -> object:
    """The page quotes four decimals, so four decimals is what is pinned."""
    return pytest.approx(value, abs=tolerance)


def _scenario(key: str) -> dict:
    body = client.get(SCENARIOS).json()
    return next(item for item in body["scenarios"] if item["key"] == key)


def _readings_by_enumeration(text: str, frequencies: dict[str, int]) -> int:
    """Every cut of the text, kept when each piece of more than one character is a word.

    Written from the definition of a path rather than from the recurrence that
    counts them, so the two agree only if both are right.
    """
    boundaries = len(text) - 1
    kept = 0
    for choices in product((False, True), repeat=boundaries):
        pieces: list[str] = []
        start = 0
        for position, cut in enumerate(choices, start=1):
            if cut:
                pieces.append(text[start:position])
                start = position
        pieces.append(text[start:])
        if all(len(piece) == 1 or piece in frequencies for piece in pieces):
            kept += 1
    return kept


def _score_from_the_definition(words: list[str], frequencies: dict[str, int]) -> float:
    total = sum(frequencies.values())
    return sum(math.log((frequencies.get(word) or 1) / total) for word in words)


class TestWhatTheGreedyScanLeftBehind:
    def test_the_sentence_keeps_its_forty_five_characters_and_seven_words(self) -> None:
        body = client.get(SETUP).json()

        assert body["sentence"] == SENTENCE
        assert body["spaceless_sentence"] == SPACELESS
        assert body["n_spaceless_characters"] == 45
        assert body["n_words_with_spaces"] == 7

    def test_the_english_list_is_the_previous_pages_own(self) -> None:
        body = client.get(SETUP).json()

        assert body["n_english_words"] == 810
        assert body["longest_english_entry"] == 10

    def test_the_park_sentence_needed_six_pieces_scanning_left_and_five_here(
        self,
    ) -> None:
        body = client.get(SETUP).json()

        assert body["park_greedy_left"] == ["我们", "在野", "生动", "物", "园", "玩"]
        assert body["park_greedy_right"] == PARK_READING
        assert len(body["park_best_path"]) == 5
        assert len(body["park_greedy_left"]) == 6

    def test_the_english_text_the_greedy_scan_breaks_is_read_whole_here(self) -> None:
        body = client.get(SETUP).json()

        assert body["water_text"] == WATER
        assert body["water_greedy_left"] == [
            "the",
            "water",
            "under",
            "theta",
            "b",
            "l",
            "e",
        ]
        assert body["water_best_path"] == ["the", "water", "under", "the", "table"]
        assert body["water_best_path"] == body["water_reader_reading"]


class TestTheLatticeOfTheResearchSentence:
    def test_ten_candidates_over_six_characters(self) -> None:
        scenario = _scenario("research")

        assert scenario["n_characters"] == 6
        assert scenario["n_edges"] == 10
        assert scenario["n_cuts"] == "32"
        assert scenario["n_readings"] == "10"
        assert scenario["n_lookups"] == 15

    def test_every_position_carries_a_candidate_and_the_spans_fit_the_words(
        self,
    ) -> None:
        scenario = _scenario("research")
        starts = {edge["start"] for edge in scenario["edges"]}

        assert starts == set(range(6))
        for edge in scenario["edges"]:
            assert edge["end"] - edge["start"] == len(edge["word"])
            assert RESEARCH[edge["start"] : edge["end"]] == edge["word"]
            assert edge["length"] == len(edge["word"])

    def test_five_candidates_are_entries_and_five_are_lone_characters(self) -> None:
        scenario = _scenario("research")
        entries = [edge for edge in scenario["edges"] if edge["frequency"] > 0]
        lone = [edge for edge in scenario["edges"] if edge["frequency"] == 0]

        assert sorted(edge["word"] for edge in entries) == sorted(LIFE_FAVOURING)
        assert len(lone) == 5
        assert {edge["log_score"] for edge in lone} == {math.log(1 / 33)}

    def test_each_candidate_scores_the_log_of_its_share(self) -> None:
        scenario = _scenario("research")

        for edge in scenario["edges"]:
            expected = math.log((edge["frequency"] or 1) / 33)
            assert edge["log_score"] == expected

    def test_the_research_reading_wins_by_the_log_of_ten_over_three(self) -> None:
        scenario = _scenario("research")

        assert scenario["best"]["words"] == LIFE_READING
        assert scenario["runner_up"]["words"] == STUDENT_READING
        assert scenario["best"]["total_log_score"] == pytest_approx(-4.4981)
        assert scenario["runner_up"]["total_log_score"] == pytest_approx(-5.7020)
        assert scenario["margin"] == pytest_approx(math.log(10 / 3))
        assert scenario["runner_up"]["share_of_best"] == pytest_approx(0.3)

    def test_the_other_counts_answer_the_other_way_on_the_same_candidates(self) -> None:
        life = _scenario("research")
        student = _scenario("research-student")

        assert student["n_edges"] == life["n_edges"] == 10
        assert student["best"]["words"] == STUDENT_READING
        assert student["runner_up"]["words"] == LIFE_READING
        assert student["best"]["total_log_score"] == pytest_approx(-4.9319)
        assert student["runner_up"]["total_log_score"] == pytest_approx(-6.0305)
        assert student["margin"] == pytest_approx(math.log(1200 / 400))

    def test_neither_greedy_scan_moves_when_the_counts_change(self) -> None:
        left = [scan for scan in _scenario("research")["greedy"]]
        student = [scan for scan in _scenario("research-student")["greedy"]]

        assert [scan["words"] for scan in left] == [
            STUDENT_READING,
            LIFE_READING,
        ]
        assert [scan["words"] for scan in student] == [scan["words"] for scan in left]


class TestWhereTheWholePathChangesTheAnswer:
    def test_the_greedy_scan_loses_three_characters_and_the_best_path_does_not(
        self,
    ) -> None:
        scenario = _scenario("water")
        left = scenario["greedy"][0]

        assert left["n_words"] == 7
        assert scenario["best"]["n_words"] == 5
        assert scenario["best"]["matches_reader"] is True
        assert scenario["best"]["total_log_score"] == pytest_approx(
            5 * math.log(1 / 810)
        )

    def test_the_greedy_answer_is_on_the_ranking_and_is_far_down_it(self) -> None:
        scenario = _scenario("water")
        greedy = scenario["greedy"][0]["words"]
        found = [
            reading for reading in scenario["readings"] if reading["words"] == greedy
        ]

        assert len(found) == 1
        assert found[0]["total_log_score"] == pytest_approx(7 * math.log(1 / 810))
        assert found[0]["share_of_best"] < 1e-5

    def test_a_bare_list_makes_the_score_a_count_of_words(self) -> None:
        scenario = _scenario("park-plain")

        assert scenario["total_frequency"] == 9
        for reading in scenario["readings"]:
            assert reading["total_log_score"] == pytest_approx(
                reading["n_words"] * math.log(1 / 9)
            )

    def test_a_bare_list_beats_the_greedy_scan_on_the_count_and_still_reads_wrong(
        self,
    ) -> None:
        scenario = _scenario("park-plain")

        assert scenario["greedy"][0]["n_words"] == 6
        assert scenario["best"]["n_words"] == 5
        assert scenario["best"]["words"] == ["我们", "在野", "生", "动物园", "玩"]
        assert scenario["best"]["matches_reader"] is False
        assert scenario["margin"] == 0.0
        assert scenario["runner_up"]["matches_reader"] is True

    def test_counts_separate_the_two_five_word_readings(self) -> None:
        scenario = _scenario("park-counted")

        assert scenario["total_frequency"] == 199
        assert scenario["best"]["words"] == PARK_READING
        assert scenario["best"]["matches_reader"] is True
        assert scenario["best"]["total_log_score"] == pytest_approx(-10.9001)
        assert scenario["runner_up"]["total_log_score"] == pytest_approx(-16.6684)
        assert scenario["margin"] == pytest_approx(5.7683)


class TestWhereItIsWorseThanTheGreedyScan:
    def test_two_readings_of_four_english_words_score_exactly_the_same(self) -> None:
        scenario = _scenario("table")
        first, second = scenario["readings"][0], scenario["readings"][1]

        assert first["total_log_score"] == second["total_log_score"]
        assert scenario["margin"] == 0.0
        assert first["share_of_best"] == 1.0
        assert second["share_of_best"] == 1.0

    def test_the_best_path_takes_the_wrong_one_and_the_right_to_left_scan_does_not(
        self,
    ) -> None:
        scenario = _scenario("table")

        assert scenario["best"]["words"] == ["theta", "bled", "own", "there"]
        assert scenario["best"]["matches_reader"] is False
        assert scenario["greedy"][1]["words"] == ["the", "table", "down", "there"]
        assert scenario["greedy"][1]["matches_reader"] is True

    def test_it_asks_the_list_about_far_more_substrings_than_the_greedy_scan(
        self,
    ) -> None:
        table = _scenario("table")
        spaceless = _scenario("spaceless")
        park = _scenario("park-plain")

        assert table["n_lookups"] == 125
        assert [scan["n_lookups"] for scan in table["greedy"]] == [20, 18]
        assert spaceless["n_lookups"] == 405
        assert [scan["n_lookups"] for scan in spaceless["greedy"]] == [191, 156]
        assert park["n_lookups"] == 24
        assert [scan["n_lookups"] for scan in park["greedy"]] == [9, 8]

    def test_on_our_own_sentence_it_answers_exactly_what_the_greedy_scan_did(
        self,
    ) -> None:
        scenario = _scenario("spaceless")

        assert scenario["best"]["n_words"] == 24
        assert all(scan["n_words"] == 24 for scan in scenario["greedy"])
        assert scenario["best"]["words"] == scenario["greedy"][0]["words"]


class TestAStretchTheListNeverSaw:
    def test_a_character_no_entry_covers_is_worth_one_count(self) -> None:
        scenario = _scenario("name")
        surname = [edge for edge in scenario["edges"] if edge["end"] <= 5]

        assert len(surname) == 5
        assert all(edge["frequency"] == 0 for edge in surname)
        assert all(
            edge["log_score"] == pytest_approx(math.log(1 / 33)) for edge in surname
        )

    def test_the_surname_costs_five_of_those_and_the_rest_reads_as_before(self) -> None:
        scenario = _scenario("name")
        research = _scenario("research")

        assert scenario["best"]["words"] == [
            "阿",
            "尔",
            "瓦",
            "雷",
            "斯",
            "研究",
            "生命",
            "起源",
        ]
        assert scenario["best"]["n_words"] == 8
        assert scenario["best"]["total_log_score"] == pytest_approx(
            research["best"]["total_log_score"] + 5 * math.log(1 / 33)
        )
        assert scenario["reader_reading"] is not None
        assert len(scenario["reader_reading"]) == 4

    def test_the_margin_over_the_next_reading_is_unchanged_by_the_surname(self) -> None:
        assert _scenario("name")["margin"] == pytest_approx(
            _scenario("research")["margin"]
        )


class TestFindingItWithoutEnumerating:
    def test_the_readings_grow_with_the_text_and_the_candidates_do_not(self) -> None:
        steps = client.get(SCALE).json()["steps"]

        assert [step["n_characters"] for step in steps] == [4, 8, 12, 16, 20, 24, 28]
        assert [step["n_readings"] for step in steps] == [
            "3",
            "7",
            "36",
            "144",
            "432",
            "1296",
            "2592",
        ]
        assert [step["n_edges"] for step in steps] == [6, 12, 20, 27, 34, 40, 45]

    def test_the_last_step_is_the_whole_text(self) -> None:
        body = client.get(SCALE).json()
        last = body["steps"][-1]

        assert body["text"] == GARDEN
        assert last["n_characters"] == len(GARDEN)
        assert last["n_cuts"] == str(2 ** (len(GARDEN) - 1))
        assert last["n_cuts"] == "134217728"

    def test_the_candidate_count_grows_no_faster_than_the_text(self) -> None:
        steps = client.get(SCALE).json()["steps"]

        for step in steps:
            assert step["n_edges"] <= step["n_characters"] * 10
            assert step["n_lookups"] <= step["n_characters"] * 10

    def test_the_seventeen_character_text_holds_far_fewer_readings_than_cuts(
        self,
    ) -> None:
        scenario = _scenario("table")

        assert scenario["n_cuts"] == "65536"
        assert scenario["n_readings"] == "216"
        assert scenario["n_edges"] == 30

    def test_the_park_sentence_holds_sixteen_readings_of_two_hundred_and_fifty_six(
        self,
    ) -> None:
        scenario = _scenario("park-plain")

        assert scenario["n_characters"] == 9
        assert scenario["n_cuts"] == "256"
        assert scenario["n_readings"] == "16"
        assert scenario["n_edges"] == 14

    def test_our_own_sentence_holds_two_hundred_and_forty_of_seventeen_trillion(
        self,
    ) -> None:
        scenario = _scenario("spaceless")

        assert scenario["n_cuts"] == "17592186044416"
        assert scenario["n_cuts"] == str(2**44)
        assert scenario["n_readings"] == "240"
        assert scenario["n_edges"] == 54


class TestTheCountsAgreeWithAnEnumeration:
    def test_the_research_sentence_has_ten_readings_by_both_routes(self) -> None:
        assert _scenario("research")["n_readings"] == str(
            _readings_by_enumeration(RESEARCH, LIFE_FAVOURING)
        )

    def test_the_park_sentence_has_sixteen_readings_by_both_routes(self) -> None:
        assert _scenario("park-plain")["n_readings"] == str(
            _readings_by_enumeration(PARK, {word: 1 for word in PARK_COUNTS})
        )

    def test_the_surname_sentence_agrees_too(self) -> None:
        assert _scenario("name")["n_readings"] == str(
            _readings_by_enumeration(NAME, LIFE_FAVOURING)
        )


class TestTheRankingAgreesWithTheSearch:
    def test_the_first_ranked_reading_is_the_one_the_best_path_answers(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            best = scenario["best"]
            on_path = [
                edge["word"] for edge in scenario["edges"] if edge["on_best_path"]
            ]

            assert best["words"] == on_path, scenario["key"]
            assert scenario["readings"][0] == best

    def test_the_ranking_falls(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            totals = [reading["total_log_score"] for reading in scenario["readings"]]

            assert totals == sorted(totals, reverse=True), scenario["key"]

    def test_no_reading_appears_twice(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            words = [tuple(reading["words"]) for reading in scenario["readings"]]

            assert len(set(words)) == len(words), scenario["key"]

    def test_every_reading_spells_the_text_back(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            for reading in scenario["readings"]:
                assert "".join(reading["words"]) == scenario["text"], scenario["key"]

    def test_a_total_is_the_sum_of_the_definitions_logs(self) -> None:
        for key, frequencies in FREQUENCIES.items():
            scenario = _scenario(key)
            for reading in scenario["readings"]:
                assert reading["total_log_score"] == pytest_approx(
                    _score_from_the_definition(reading["words"], frequencies)
                )


class TestAddingAWordTheTextDoesNotContain:
    def test_the_sentence_is_cut_one_way_below_the_threshold_and_another_above(
        self,
    ) -> None:
        body = client.get(GROWTH).json()
        steps = {step["added_frequency"]: step for step in body["steps"]}

        assert body["text"] == RESEARCH
        assert body["added_word"] == "很"
        assert body["added_word"] not in RESEARCH
        assert steps[0]["best"] == ["研究", "生", "命", "起源"]
        assert steps[1]["best"] == ["研究", "生", "命", "起源"]
        assert steps[2]["best"] == LIFE_READING
        assert steps[50]["best"] == LIFE_READING

    def test_the_crossing_is_where_the_total_passes_a_product_of_two_counts(
        self,
    ) -> None:
        body = client.get(GROWTH).json()

        assert body["base_total"] == 32
        assert body["crossing_total"] == 34
        assert body["exact_threshold"] == pytest_approx(10 * 10 / 3)
        assert body["exact_threshold"] == pytest_approx(33.3333, tolerance=1e-4)

    def test_the_two_readings_cross_where_the_best_path_changes(self) -> None:
        for step in client.get(GROWTH).json()["steps"]:
            assert step["whole_wins"] is (step["whole_score"] > step["split_score"])
            assert step["whole_wins"] is (step["total_frequency"] > 10 * 10 / 3), step[
                "total_frequency"
            ]

    def test_every_score_falls_as_the_list_grows(self) -> None:
        steps = client.get(GROWTH).json()["steps"]
        wholes = [step["whole_score"] for step in steps]
        splits = [step["split_score"] for step in steps]

        assert wholes == sorted(wholes, reverse=True)
        assert splits == sorted(splits, reverse=True)

    def test_counting_the_same_text_over_again_moves_nothing_at_all(self) -> None:
        scalings = client.get(GROWTH).json()["scalings"]

        assert [step["factor"] for step in scalings] == [1, 2, 5, 20, 100]
        assert [step["total_frequency"] for step in scalings] == [
            32,
            64,
            160,
            640,
            3200,
        ]
        for step in scalings:
            assert step["best"] == ["研究", "生", "命", "起源"]
            assert step["best_score"] == scalings[0]["best_score"]
            assert step["best_score"] == pytest_approx(-6.7729)

    def test_the_four_word_reading_falls_faster_than_the_three_word_one(self) -> None:
        steps = client.get(GROWTH).json()["steps"]
        first, last = steps[0], steps[-1]

        assert first["split_score"] - last["split_score"] == pytest_approx(
            4 * math.log(82 / 32)
        )
        assert first["whole_score"] - last["whole_score"] == pytest_approx(
            3 * math.log(82 / 32)
        )


class TestThePlayground:
    def test_it_answers_the_same_thing_the_fixed_texts_do(self) -> None:
        body = client.post(
            SEGMENT, json={"text": RESEARCH, "word_list": "research life"}
        ).json()

        assert body["analysis"]["best"]["words"] == LIFE_READING
        assert body["analysis"]["n_readings"] == "10"

    def test_counts_added_to_the_list_move_the_cut(self) -> None:
        without = client.post(
            SEGMENT, json={"text": RESEARCH, "word_list": "research split"}
        ).json()
        with_it = client.post(
            SEGMENT,
            json={
                "text": RESEARCH,
                "word_list": "research split",
                "unrelated_frequency": 2,
            },
        ).json()

        assert without["analysis"]["best"]["words"] == ["研究", "生", "命", "起源"]
        assert with_it["analysis"]["best"]["words"] == LIFE_READING
        assert with_it["unrelated_frequency"] == 2

    def test_striking_a_word_out_changes_the_candidates(self) -> None:
        body = client.post(
            SEGMENT,
            json={
                "text": RESEARCH,
                "word_list": "research life",
                "removed": ["生命"],
            },
        ).json()

        assert body["removed"] == ["生命"]
        assert body["analysis"]["best"]["words"] == STUDENT_READING
        assert "生命" not in [edge["word"] for edge in body["analysis"]["edges"]]

    def test_a_short_list_comes_back_with_its_counts_and_the_long_one_does_not(
        self,
    ) -> None:
        short = client.post(
            SEGMENT, json={"text": RESEARCH, "word_list": "research life"}
        ).json()
        long = client.post(SEGMENT, json={"text": TABLE, "word_list": "english"}).json()

        assert [entry["word"] for entry in short["words_in_list"]] == list(
            LIFE_FAVOURING
        )
        assert [entry["frequency"] for entry in short["words_in_list"]] == list(
            LIFE_FAVOURING.values()
        )
        assert long["words_in_list"] == []
        assert long["analysis"]["n_words_in_list"] == 810

    def test_a_word_not_in_the_chosen_list_is_ignored_rather_than_refused(self) -> None:
        body = client.post(
            SEGMENT,
            json={
                "text": RESEARCH,
                "word_list": "research life",
                "removed": ["table"],
            },
        ).json()

        assert body["removed"] == []
        assert body["analysis"]["best"]["words"] == LIFE_READING

    def test_a_blank_text_has_no_words_and_one_reading(self) -> None:
        body = client.post(
            SEGMENT, json={"text": "   ", "word_list": "research life"}
        ).json()

        assert body["analysis"]["best"]["words"] == []
        assert body["analysis"]["n_readings"] == "1"
        assert body["analysis"]["n_cuts"] == "1"

    def test_whitespace_ends_a_run_so_no_candidate_reaches_across_it(self) -> None:
        body = client.post(
            SEGMENT, json={"text": "研究 生命", "word_list": "research life"}
        ).json()
        words = [edge["word"] for edge in body["analysis"]["edges"]]

        assert body["analysis"]["best"]["words"] == ["研究", "生命"]
        assert all(" " not in word for word in words)


class TestTheFiguresThePageWritesOut:
    """Every four-decimal number the prose or a hand-worked table quotes."""

    def test_the_five_entries_score_what_the_table_of_shares_says(self) -> None:
        scenario = _scenario("research")
        scores = {edge["word"]: edge["log_score"] for edge in scenario["edges"]}

        assert scores["研究"] == pytest_approx(-1.1939)
        assert scores["生命"] == pytest_approx(-1.4171)
        assert scores["起源"] == pytest_approx(-1.8871)
        assert scores["研究生"] == pytest_approx(-1.7047)
        assert scores["命"] == pytest_approx(-2.1102)
        assert scores["研"] == pytest_approx(-3.4965)

    def test_the_recurrence_table_adds_up_to_the_ranked_readings(self) -> None:
        scenario = _scenario("research")
        scores = {edge["word"]: edge["log_score"] for edge in scenario["edges"]}
        totals = [reading["total_log_score"] for reading in scenario["readings"]]

        best_four = scores["起源"]
        best_three = scores["命"] + best_four
        best_two = scores["生命"] + best_four
        best_one = scores["究"] + best_two

        assert best_four == pytest_approx(-1.8871)
        assert best_three == pytest_approx(-3.9973)
        assert best_two == pytest_approx(-3.3042)
        assert best_one == pytest_approx(-6.8007)
        assert scores["研究"] + best_two == pytest_approx(totals[0])
        assert scores["研究生"] + best_three == pytest_approx(totals[1])
        assert scores["研"] + best_one == pytest_approx(-10.2972)

    def test_the_bare_list_scores_the_page_quotes(self) -> None:
        scenario = _scenario("park-plain")

        assert scenario["edges"][0]["log_score"] == pytest_approx(-2.1972)
        assert scenario["best"]["total_log_score"] == pytest_approx(-10.9861)
        assert scenario["readings"][2]["total_log_score"] == pytest_approx(-13.1833)

    def test_the_english_repair_scores_the_page_quotes(self) -> None:
        scenario = _scenario("water")
        greedy = scenario["greedy"][0]["words"]
        theirs = next(
            reading for reading in scenario["readings"] if reading["words"] == greedy
        )

        assert scenario["best"]["total_log_score"] == pytest_approx(-33.4852)
        assert theirs["total_log_score"] == pytest_approx(-46.8792)
        assert theirs["share_of_best"] == pytest_approx(1 / 810**2, tolerance=1e-9)
        assert 810**2 == 656100

    def test_the_tied_english_readings_score_what_the_page_quotes(self) -> None:
        scenario = _scenario("table")

        assert scenario["best"]["total_log_score"] == pytest_approx(-26.7881)

    def test_the_counted_park_words_score_what_the_page_quotes(self) -> None:
        scenario = _scenario("park-counted")

        assert scenario["best"]["total_log_score"] == pytest_approx(-10.9001)
        assert scenario["runner_up"]["total_log_score"] == pytest_approx(-16.6684)
        assert scenario["margin"] == pytest_approx(5.7683)
        assert scenario["runner_up"]["share_of_best"] == pytest_approx(1 / 320)

    def test_the_surname_costs_what_the_page_quotes(self) -> None:
        scenario = _scenario("name")

        assert scenario["best"]["total_log_score"] == pytest_approx(-21.9806)
        assert 5 * math.log(1 / 33) == pytest_approx(-17.4826)

    def test_the_growth_scores_the_page_quotes(self) -> None:
        steps = {
            step["total_frequency"]: step for step in client.get(GROWTH).json()["steps"]
        }

        assert steps[32]["split_score"] == pytest_approx(-6.7729)
        assert steps[32]["whole_score"] == pytest_approx(-6.8137)
        assert steps[32]["split_score"] - steps[32]["whole_score"] == pytest_approx(
            0.0408
        )
        assert steps[34]["whole_score"] == pytest_approx(-6.9956)
        assert steps[34]["split_score"] == pytest_approx(-7.0154)
        assert steps[34]["whole_score"] - steps[34]["split_score"] == pytest_approx(
            0.0198
        )


class TestRefusals:
    def test_striking_out_every_word_is_refused_in_the_pages_own_words(self) -> None:
        response = client.post(
            SEGMENT,
            json={
                "text": RESEARCH,
                "word_list": "research life",
                "removed": list(LIFE_FAVOURING),
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "a word list cannot be empty"

    def test_a_text_longer_than_the_bound_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "x" * 61})

        assert response.status_code == 422

    def test_an_unknown_word_list_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "abc", "word_list": "german"})

        assert response.status_code == 422

    def test_an_unknown_keyword_is_refused(self) -> None:
        response = client.post(SEGMENT, json={"text": "abc", "direction": "left"})

        assert response.status_code == 422

    def test_counts_beyond_the_bound_are_refused(self) -> None:
        response = client.post(
            SEGMENT, json={"text": RESEARCH, "unrelated_frequency": 501}
        )

        assert response.status_code == 422


IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z]")
"""Anything shaped like a parameter name or a class name."""

ALLOWED = frozenset({"Dr", "Alvarez"})
"""Words of the running sentence that look like nothing else."""

CARRIES_A_TEXT = frozenset(
    {
        "text",
        "sentence",
        "spaceless_sentence",
        "park_sentence",
        "water_text",
        "word",
        "words",
        "best",
        "added_word",
    }
)
"""The keys holding a text or a word rather than a sentence a reader is told."""


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a response body, bar the texts themselves."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in _strings(item)
            if key not in CARRIES_A_TEXT
        ]
    if isinstance(value, list):
        return [found for item in value for found in _strings(item)]
    return []


class TestNothingNamesTheCodeBehindThePage:
    def test_no_endpoint_returns_an_identifier(self) -> None:
        for path in (SETUP, SCENARIOS, SCALE, GROWTH):
            for sentence in _strings(client.get(path).json()):
                assert not IDENTIFIER.search(sentence), (path, sentence)

    def test_the_refusal_names_no_identifier(self) -> None:
        response = client.post(
            SEGMENT,
            json={
                "text": RESEARCH,
                "word_list": "research life",
                "removed": list(LIFE_FAVOURING),
            },
        )
        detail = response.json()["detail"]

        assert not IDENTIFIER.search(detail)
        for forbidden in ("dictionary", "Word", "Error", "segmenter", "lattice"):
            assert forbidden not in detail

    def test_the_labels_a_widget_shows_name_no_identifier(self) -> None:
        for scenario in client.get(SCENARIOS).json()["scenarios"]:
            assert not IDENTIFIER.search(scenario["label"])
            assert not IDENTIFIER.search(scenario["word_list_label"])
            for scan in scenario["greedy"]:
                assert not IDENTIFIER.search(scan["scan"])

    def test_the_allowed_words_are_the_running_sentences_own(self) -> None:
        assert ALLOWED == {"Dr", "Alvarez"}
        assert all(word in SENTENCE for word in ALLOWED)


class TestTheTablesThePageQuotes:
    def test_the_compound_is_rarer_than_either_of_its_characters(self) -> None:
        assert SPLIT_FAVOURING["生命"] == 3
        assert SPLIT_FAVOURING["生"] == 10
        assert SPLIT_FAVOURING["命"] == 10
        assert sum(SPLIT_FAVOURING.values()) == 32

    def test_the_two_spec_tables_differ_only_in_two_counts(self) -> None:
        moved = {
            word
            for word in LIFE_FAVOURING
            if LIFE_FAVOURING[word] != STUDENT_FAVOURING[word]
        }

        assert moved == {"研究生", "命"}
        assert sum(LIFE_FAVOURING.values()) == 33
        assert sum(STUDENT_FAVOURING.values()) == 55

    def test_the_park_counts_are_over_the_same_nine_words(self) -> None:
        assert len(PARK_COUNTS) == 9
        assert sum(PARK_COUNTS.values()) == 199
