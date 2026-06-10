"""The Markov chains page's endpoints, on the chapters of Alice the page counts.

The chain these endpoints call has its own spec in the library, which pins its
arithmetic on fixtures worked by hand. What is pinned here is every number the
page quotes, on the text the page counts, so a change under the chain that
moved one of them fails a test rather than quietly rewriting a paragraph.

The last class is the other half. Nothing a widget prints may carry an
identifier, a refusal included.
"""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient
from oop_ml.core.exceptions import (
    NonUniqueStationaryDistributionError,
    TooFewValuesError,
)
from oop_ml.numpy.sequences.markov_chain import MarkovChain

from oop_ml_api.concepts.markov_chains import (
    HELD_BACK,
    _class_chain,
    _class_sequences,
    _in_plain_words,
    _score,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/markov-chains"
VOWELS = f"{BASE}/vowels"
LETTERS = f"{BASE}/letters"
MEMORY = f"{BASE}/memory"
EDGES = f"{BASE}/edges"
FIT = f"{BASE}/fit"


def four(value: float) -> float:
    return round(value, 4)


def score_at(rows: list[dict], width: int, smoothing: float) -> dict:
    return next(
        row for row in rows if row["width"] == width and row["smoothing"] == smoothing
    )


def case_titled(body: dict, title: str) -> dict:
    return next(case for case in body["cases"] if case["title"] == title)


class TestOneSentenceByHand:
    def test_it_is_thirteen_letters_and_twelve_steps(self) -> None:
        hand = client.get(VOWELS).json()["hand"]

        assert hand["letters"] == "ohmydeardinah"
        assert hand["classes"] == "VCCCCVVCCVCVC"
        assert hand["tally"]["n_symbols"] == 13
        assert hand["tally"]["n_steps"] == 12

    def test_the_counts_are_the_pencil_count(self) -> None:
        """ea once; oh, ar, in, ah; de, di, na; hm, my, yd, rd."""
        tally = client.get(VOWELS).json()["hand"]["tally"]

        assert tally["counts"] == [[1, 4], [3, 4]]
        assert tally["table"][0] == pytest.approx([0.2, 0.8])
        assert tally["table"][1] == pytest.approx([3 / 7, 4 / 7])

    def test_it_settles_at_fifteen_forty_thirds_against_five_thirteenths(
        self,
    ) -> None:
        tally = client.get(VOWELS).json()["hand"]["tally"]

        assert tally["stationary"][0] == pytest.approx(15 / 43)
        assert tally["vowel_share"] == pytest.approx(5 / 13)


class TestTwoChapters:
    def test_the_counts_are_the_ones_the_page_quotes(self) -> None:
        body = client.get(VOWELS).json()

        assert body["counted"]["counts"] == [[933, 5451], [5423, 4868]]
        assert body["counted"]["n_symbols"] == 16803
        assert body["counted"]["n_steps"] == 16675
        assert body["n_sentences"] == 128

    def test_the_table_is_each_row_divided_by_its_total(self) -> None:
        table = client.get(VOWELS).json()["counted"]["table"]

        assert table[0][0] == pytest.approx(933 / 6384)
        assert table[1][0] == pytest.approx(5423 / 10291)
        assert four(table[0][0]) == 0.1461
        assert four(table[1][0]) == 0.5270

    def test_it_settles_where_the_flows_balance(self) -> None:
        counted = client.get(VOWELS).json()["counted"]
        to_consonant = counted["table"][0][1]
        to_vowel = counted["table"][1][0]

        assert counted["stationary"][0] == pytest.approx(
            to_vowel / (to_vowel + to_consonant)
        )
        assert four(counted["stationary"][0]) == 0.3816
        assert four(counted["vowel_share"]) == 0.3814

    def test_the_shares_of_three_texts_beside_the_settled_one(self) -> None:
        body = client.get(VOWELS).json()

        assert four(body["held_back_vowel_share"]) == 0.3916
        assert body["walk_vowel_share"] == pytest.approx(0.38325)
        assert body["walk_length"] == 20000

    def test_the_likelihood_peaks_at_the_count_ratio(self) -> None:
        body = client.get(VOWELS).json()

        assert body["best_probability"] == pytest.approx(933 / 6384)
        assert round(body["best_log_likelihood"], 2) == -2655.53
        assert all(
            point["log_likelihood"] < body["best_log_likelihood"]
            for point in body["likelihood"]
        )
        assert round(body["counted_log_probability"], 2) == -9773.73


class TestSeveralStepsAhead:
    def test_two_steps_is_the_table_used_twice(self) -> None:
        body = client.get(VOWELS).json()
        table = body["counted"]["table"]
        two = body["after"][2]

        assert two["vowel_from_vowel"] == pytest.approx(
            table[0][0] * table[0][0] + table[0][1] * table[1][0]
        )
        assert four(two["vowel_from_vowel"]) == 0.4713
        assert four(two["vowel_from_consonant"]) == 0.3263

    def test_the_distance_shrinks_by_the_second_eigenvalue_every_step(self) -> None:
        body = client.get(VOWELS).json()
        size = abs(body["second_eigenvalue"])

        assert four(body["second_eigenvalue"]) == -0.3808
        for earlier, later in zip(body["after"], body["after"][1:], strict=False):
            assert later["distance_from_vowel"] == pytest.approx(
                earlier["distance_from_vowel"] * size, rel=1e-6
            )

    def test_the_walks_cross_before_they_meet(self) -> None:
        after = client.get(VOWELS).json()["after"]

        assert [four(row["vowel_from_vowel"]) for row in after[1:4]] == [
            0.1461,
            0.4713,
            0.3475,
        ]
        assert four(after[1]["distance_from_vowel"]) == 0.2355
        assert four(after[2]["distance_from_vowel"]) == 0.0897
        assert after[10]["distance_from_vowel"] < 0.00005

    def test_the_letter_chain_forgets_q_last(self) -> None:
        mixing = client.get(LETTERS).json()["mixing"]

        assert [row["worst_start"] for row in mixing[:4]] == ["q", "q", "q", "q"]
        assert mixing[4]["worst_start"] == "v"
        assert [four(row["worst_distance"]) for row in mixing[:5]] == [
            0.9264,
            0.4918,
            0.158,
            0.0442,
            0.0096,
        ]
        assert mixing[7]["n_steps"] == 10
        assert mixing[7]["worst_distance"] < 0.00004


class TestWhatTheOneStepThrowsAway:
    def test_one_step_on_the_chain_and_the_text_agree_exactly(self) -> None:
        gap = client.get(VOWELS).json()["gaps"][0]

        assert gap["chain_vowel_after_vowel"] == pytest.approx(
            gap["text_vowel_after_vowel"]
        )

    def test_two_steps_on_the_chain_overshoots_the_text_by_a_tenth(self) -> None:
        gap = client.get(VOWELS).json()["gaps"][1]

        assert four(gap["chain_vowel_after_vowel"]) == 0.4713
        assert four(gap["text_vowel_after_vowel"]) == 0.3661
        assert four(gap["held_back_vowel_after_vowel"]) == 0.3727

    def test_three_steps_on_the_error_changes_sign(self) -> None:
        gap = client.get(VOWELS).json()["gaps"][2]

        assert four(gap["chain_vowel_after_vowel"]) == 0.3475
        assert four(gap["text_vowel_after_vowel"]) == 0.4036

    def test_grouping_letters_into_two_classes_accounts_for_about_half(self) -> None:
        """Text the letter chain wrote is one-step by construction as letters,
        and still misses two steps on once its letters are grouped."""
        gap = client.get(VOWELS).json()["gaps"][1]

        assert four(gap["written_chain_vowel_after_vowel"]) == 0.456
        assert four(gap["written_text_vowel_after_vowel"]) == 0.4084

    def test_each_chapter_has_its_own_vowel_row(self) -> None:
        chapters = client.get(VOWELS).json()["chapters"]

        assert [row["n_symbols"] for row in chapters] == [8601, 8202, 6874]
        assert [four(row["vowel_after_vowel"]) for row in chapters] == [
            0.1396,
            0.1529,
            0.1701,
        ]
        assert [four(row["vowel_after_consonant"]) for row in chapters] == [
            0.5256,
            0.5285,
            0.5341,
        ]


class TestTheLetterChain:
    def test_it_is_twenty_seven_states_and_half_its_cells_are_empty(self) -> None:
        body = client.get(LETTERS).json()

        assert len(body["states"]) == 27
        assert body["n_symbols"] == 21045
        assert body["n_steps"] == 20917
        assert body["n_nonzero"] == 363
        assert body["n_cells"] == 729

    def test_the_rows_the_page_quotes(self) -> None:
        followers = {
            row["state"]: row for row in client.get(LETTERS).json()["followers"]
        }

        assert followers["q"]["n_leaving"] == 20
        assert followers["q"]["top"][0] == {"state": "u", "probability": 0.95}
        assert followers["h"]["n_followers"] == 9
        assert four(followers["h"]["top"][0]["probability"]) == 0.49

    def test_the_settled_shares_against_three_texts(self) -> None:
        body = client.get(LETTERS).json()

        assert four(body["largest_gap_counted"]) == 0.0019
        assert four(body["largest_gap_held_back"]) == 0.0097
        assert four(body["largest_gap_walk"]) == 0.0054
        shares = {row["state"]: row for row in body["shares"]}
        assert four(shares["d"]["stationary"]) == 0.035
        assert four(shares["d"]["held_back"]) == 0.0447

    def test_the_samples_the_page_quotes(self) -> None:
        samples = client.get(LETTERS).json()["samples"]

        assert samples[1]["text"].startswith("thing rs bof te an")
        assert samples[2]["text"].startswith("then es ing pook all a")
        assert samples[3]["text"].startswith("the geotuy of thought")
        assert samples[4]["text"].startswith("thaqogsedke umuoumts")
        assert all(len(sample["text"]) == 150 for sample in samples)

    def test_what_each_letter_of_memory_costs(self) -> None:
        costs = client.get(LETTERS).json()["costs"]

        assert [row["n_possible_states"] for row in costs] == [27, 729, 19683]
        assert [row["n_states"] for row in costs] == [27, 363, 1982]
        assert [row["n_cells"] for row in costs] == [729, 131769, 3928324]
        assert [row["n_nonzero"] for row in costs] == [363, 1982, 5049]
        assert [round(row["steps_per_state"], 1) for row in costs] == [
            774.7,
            57.3,
            10.4,
        ]
        assert costs[2]["never_left"] == ["oud", "phy"]


class TestTheHeldBackChapter:
    def test_every_model_is_scored_on_the_same_positions(self) -> None:
        body = client.get(MEMORY).json()

        assert body["letter_positions"] == 8357
        assert body["letter_positions_set_aside"] == 71
        assert body["class_positions"] == 6506
        assert body["ahead_positions"] == 7940

    def test_one_letter_of_memory_against_none(self) -> None:
        rows = client.get(MEMORY).json()["letter_rows"]

        assert four(score_at(rows, 0, 0.0)["bits"]) == 4.067
        assert four(score_at(rows, 1, 0.0)["bits"]) == 3.2436

    def test_the_pair_chain_cannot_be_scored_unsmoothed(self) -> None:
        row = score_at(client.get(MEMORY).json()["letter_rows"], 2, 0.0)

        assert row["bits"] is None
        assert row["n_impossible"] == 261

    def test_the_pair_chain_wins_lightly_smoothed_and_loses_heavily(self) -> None:
        rows = client.get(MEMORY).json()["letter_rows"]

        assert four(score_at(rows, 2, 0.01)["bits"]) == 2.8355
        assert four(score_at(rows, 1, 0.01)["bits"]) == 3.2438
        assert four(score_at(rows, 2, 0.5)["bits"]) == 3.9628
        assert four(score_at(rows, 2, 1.0)["bits"]) == 4.4876
        assert four(score_at(rows, 0, 1.0)["bits"]) == 4.0677
        assert score_at(rows, 2, 1.0)["bits"] > score_at(rows, 0, 1.0)["bits"]
        assert four(score_at(rows, 1, 1.0)["bits"]) == 3.2645

    def test_the_current_letter_is_worthless_six_letters_on(self) -> None:
        body = client.get(MEMORY).json()

        assert [four(row["bits"]) for row in body["ahead"]] == [
            3.2906,
            3.957,
            4.0624,
            4.0764,
            4.0792,
            4.0792,
        ]
        assert body["ahead"][-1]["bits"] == pytest.approx(
            body["frequencies_bits"], abs=1e-6
        )

    def test_the_class_memory_sweep(self) -> None:
        rows = client.get(MEMORY).json()["class_rows"]

        assert [four(score_at(rows, width, 0.0)["bits"]) for width in range(6)] == [
            0.9651,
            0.8659,
            0.8402,
            0.8336,
            0.8309,
            0.8284,
        ]
        assert score_at(rows, 6, 0.0)["bits"] is None
        assert score_at(rows, 6, 0.0)["n_impossible"] == 1
        assert [four(score_at(rows, width, 0.5)["bits"]) for width in range(3, 7)] == [
            0.8359,
            0.8409,
            0.8673,
            0.9589,
        ]

    def test_scoring_off_the_table_is_the_chains_own_log_probability(self) -> None:
        """The endpoint reads each step off the table so every model can be
        scored on one set of positions; on every position it is the sum the
        chain itself returns."""
        chain = _class_chain()
        sequences = _class_sequences(HELD_BACK)
        positions = [
            (index, position)
            for index, sequence in enumerate(sequences)
            for position in range(1, len(sequence))
        ]

        bits, impossible = _score(chain, 1, sequences, positions)

        assert impossible == 0
        assert bits is not None
        assert bits * len(positions) == pytest.approx(
            -sum(chain.log_probability_of(sequence) for sequence in sequences)
            / math.log(2)
        )


class TestTheEdges:
    def test_banana_has_one_stationary_distribution_and_never_reaches_it(
        self,
    ) -> None:
        case = case_titled(client.get(EDGES).json(), "A word that alternates")

        assert case["kind"] == "never_settles"
        assert case["table"] == [[0.0, 1.0], [1.0, 0.0]]
        assert case["stationary"] == [0.5, 0.5]
        assert case["walk"] == [0.0, 1.0] * 4 + [0.0]
        assert case["long_run_share"] == pytest.approx([0.5, 0.5])
        assert four(case["smoothed_stationary"][0]) == 0.5122

    def test_two_words_that_never_meet_have_no_single_answer(self) -> None:
        case = case_titled(client.get(EDGES).json(), "Two words that never meet")

        assert case["kind"] == "several_groups"
        assert case["stationary"] is None
        assert case["table"] == [[1.0, 0.0], [0.0, 1.0]]
        assert four(case["smoothed_stationary"][0]) == 0.4545

    def test_a_vowel_left_for_good_holds_no_share(self) -> None:
        case = case_titled(
            client.get(EDGES).json(), "A vowel left once and never again"
        )

        assert case["kind"] == "settles"
        assert case["stationary"] == pytest.approx([0.0, 1.0])
        assert four(case["smoothed_stationary"][0]) == 0.1818

    def test_a_vowel_never_left_has_no_row(self) -> None:
        case = case_titled(client.get(EDGES).json(), "A vowel never left")

        assert case["kind"] == "never_left"
        assert case["table"] is None
        assert case["counts"] == [[0, 0], [1, 1]]
        assert case["smoothed_table"] == [[0.5, 0.5], [0.5, 0.5]]

    def test_the_real_text_strands_two_runs_of_three(self) -> None:
        body = client.get(EDGES).json()

        assert body["stranded_states"] == ["oud", "phy"]
        assert body["stranded_sentences"][0].endswith("she said aloud")
        assert body["stranded_sentences"][1].endswith("let s try geography")


class TestPlayground:
    def test_the_default_is_the_two_chapters_counted_in_classes(self) -> None:
        body = client.post(FIT, json={}).json()

        assert body["counts"] == [[933, 5451], [5423, 4868]]
        assert body["settles"] is True
        assert four(body["two_step_gap"]) == 0.0796

    def test_the_ideal_case_passes_the_two_step_check(self) -> None:
        body = client.post(FIT, json={"source": "written_by_the_chain"}).json()

        assert body["n_symbols"] == 1500
        assert body["two_step_gap"] < 0.01

    def test_a_random_run_is_six_sentences(self) -> None:
        body = client.post(FIT, json={"source": "alice_run", "alice_start": 40}).json()

        assert body["n_sequences"] == 6

    def test_every_letter_and_the_space(self) -> None:
        body = client.post(FIT, json={"alphabet": "letters"}).json()

        assert len(body["states"]) == 27
        for row in body["table"]:
            assert sum(row) == pytest.approx(1.0)

    def test_banana_is_reported_as_never_settling(self) -> None:
        body = client.post(FIT, json={"source": "typed", "text": "banana"}).json()

        assert body["settles"] is False
        assert body["stationary"] == [0.5, 0.5]
        assert body["stationary_note"]

    def test_spa_has_no_table_until_it_is_smoothed(self) -> None:
        bare = client.post(FIT, json={"source": "typed", "text": "spa"}).json()
        smoothed = client.post(
            FIT, json={"source": "typed", "text": "spa", "smoothing": 0.5}
        ).json()

        assert bare["table"] is None
        assert bare["never_left"] == ["V"]
        assert bare["table_note"]
        assert smoothed["table"] == [[0.5, 0.5], [0.5, 0.5]]

    def test_two_groups_have_no_stationary_distribution(self) -> None:
        body = client.post(
            FIT, json={"source": "typed", "text": "rhythm. aeiou"}
        ).json()

        assert body["stationary"] is None
        assert body["stationary_note"]

    def test_a_text_with_no_letters_is_refused_in_plain_words(self) -> None:
        response = client.post(FIT, json={"source": "typed", "text": "  12 ?"})

        assert response.status_code == 422
        assert response.json()["detail"] == "There are no letters here to count."

    @pytest.mark.parametrize(
        "request_body",
        [
            {"smoothing": -0.1},
            {"smoothing": 50.0},
            {"n_steps": 31},
            {"alice_start": 500},
            {"alphabet": "syllables"},
            {"smooting": 1.0},
        ],
        ids=["negative", "huge", "steps", "start", "alphabet", "misspelled"],
    )
    def test_a_request_out_of_bounds_is_refused(self, request_body: dict) -> None:
        assert client.post(FIT, json=request_body).status_code == 422


class TestNothingAWidgetPrintsNamesTheCode:
    """Every string a widget can render, checked for identifiers."""

    IDENTIFIER = re.compile(r"[a-z]+_[a-z]+|[A-Z][a-z]+[A-Z][A-Za-z]*")

    def test_no_note_or_title_on_the_edges_names_the_code(self) -> None:
        body = client.get(EDGES).json()
        texts = [
            *(case["title"] for case in body["cases"]),
            *(case["note"] for case in body["cases"]),
            *body["stranded_sentences"],
        ]

        for text in texts:
            assert self.IDENTIFIER.search(text) is None
            assert "Error" not in text

    def test_no_label_on_the_letters_or_chapters_names_the_code(self) -> None:
        labels = [
            *(sample["label"] for sample in client.get(LETTERS).json()["samples"]),
            *(row["chapter"] for row in client.get(VOWELS).json()["chapters"]),
        ]

        for label in labels:
            assert self.IDENTIFIER.search(label) is None

    @pytest.mark.parametrize("text", ["spa", "banana", "rhythm. aeiou", " 1 "])
    def test_no_playground_note_or_refusal_names_the_code(self, text: str) -> None:
        body = client.post(FIT, json={"source": "typed", "text": text}).json()
        notes = [
            body.get("detail"),
            body.get("table_note"),
            body.get("stationary_note"),
        ]

        for note in notes:
            if note is not None:
                assert self.IDENTIFIER.search(note) is None
                assert "Error" not in note

    def test_the_rewriting_leaves_no_identifier_in_either_chain_refusal(
        self,
    ) -> None:
        with pytest.raises(TooFewValuesError) as stranded:
            MarkovChain().fit([["C", "C", "V"]])
        with pytest.raises(NonUniqueStationaryDistributionError) as islands:
            MarkovChain().fit([["C", "C"], ["V", "V"]]).stationary_distribution()

        for refusal in (stranded.value, islands.value):
            sentence = _in_plain_words(refusal)
            assert self.IDENTIFIER.search(sentence) is None
            assert "Error" not in sentence
