"""The greedy coverage page's endpoints, on the corpora the page carries.

Every number the page quotes in prose is pinned here, so a change to the
library that moved one of them would fail a test rather than quietly rewrite
the page. Three other things are pinned as well: that reading a smaller
vocabulary off a longer fit gives what a genuine fit of that size gives, since
every static endpoint takes that shortcut; that the character-weighted scorer
this module adds agrees with the library wherever the two units coincide; and
that no identifier survives into a refusal a reader can see.
"""

from __future__ import annotations

import itertools
import re

from fastapi.testclient import TestClient
from oop_ml.core.natural_language_processing.tokenization.subword.greedy_coverage import (
    GreedyCoverageTokenizer,
)

from oop_ml_api.concepts.greedy_coverage import (
    SENTENCE_CORPUS,
    CorpusChoice,
    _prefix_at,
    _word_counts,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/greedy-coverage"
CHOOSING = f"{BASE}/choosing"
UNITS = f"{BASE}/units"
MATCHED = f"{BASE}/matched"
LIMITS = f"{BASE}/limits"
CAPPING = f"{BASE}/capping"
FIT = f"{BASE}/fit"

SENTENCE = "Dr. Alvarez didn't expect the low-cost re-analysis."


class TestChoosing:
    def test_the_four_words_hold_seventy_nine_positions(self) -> None:
        body = client.get(CHOOSING).json()

        assert body["total_occurrences"] == 16
        assert body["total_positions"] == 79
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

    def test_the_four_choices_come_in_the_order_coverage_gives(self) -> None:
        body = client.get(CHOOSING).json()

        chosen = [
            (step["piece"], step["coverage"])
            for step in body["steps"]
            if step["piece"] is not None
        ]

        assert chosen == [
            ("newest</w>", 36),
            ("widest</w>", 18),
            ("low</w>", 15),
            ("lower</w>", 10),
        ]

    def test_the_first_choice_beats_the_ending_it_is_usually_confused_with(
        self,
    ) -> None:
        body = client.get(CHOOSING).json()

        rivals = {
            rival["piece"]: (
                rival["n_symbols"],
                rival["occurrences"],
                rival["coverage"],
            )
            for rival in body["steps"][0]["rivals"]
        }

        assert rivals["newest</w>"] == (6, 6, 36)
        assert rivals["est</w>"] == (3, 9, 27)
        assert rivals["ewest</w>"] == (5, 6, 30)

    def test_every_position_ends_up_covered_and_the_fit_stops_there(self) -> None:
        body = client.get(CHOOSING).json()

        assert body["steps"][-1]["covered_positions"] == 79
        assert body["steps"][-1]["corpus_pieces"] == 16
        assert body["steps"][0]["corpus_pieces"] == 79
        assert body["stops_at"] == 16
        assert body["tokens"][-4:] == [
            "newest</w>",
            "widest</w>",
            "low</w>",
            "lower</w>",
        ]

    def test_an_unseen_word_may_be_covered_wholly_partly_or_not_at_all(self) -> None:
        body = client.get(CHOOSING).json()
        readings = {reading["word"]: reading for reading in body["readings"]}

        assert readings["newest"]["pieces"] == ["newest</w>"]
        assert readings["lowest"]["pieces"] == ["l", "o", "w", "e", "s", "t</w>"]
        assert readings["lowest"]["n_covered_symbols"] == 0
        assert readings["unwidest"]["pieces"] == ["u", "n", "widest</w>"]
        assert readings["unwidest"]["n_covered_symbols"] == 6
        assert readings["wide"]["n_covered_symbols"] == 0

    def test_merging_would_rank_a_different_candidate_first(self) -> None:
        body = client.get(CHOOSING).json()

        assert [(row["pair"], row["count"]) for row in body["top_pairs"]][:2] == [
            ("e + s", 9),
            ("s + t</w>", 9),
        ]


class TestUnits:
    def test_the_two_units_agree_on_the_first_pick_of_the_four_words(self) -> None:
        body = client.get(UNITS).json()
        four = next(
            corpus for corpus in body["corpora"] if corpus["key"] == "four_words"
        )

        assert four["same_winner"] is True
        assert four["winner_by_positions"] == "newest</w>"
        assert four["winner_positions_score"] == 36
        assert four["winner_by_characters"] == "newest</w>"
        assert four["winner_characters_score"] == 42

    def test_the_ending_reaches_thirty_six_in_characters_and_still_loses(self) -> None:
        body = client.get(UNITS).json()
        four = next(
            corpus for corpus in body["corpora"] if corpus["key"] == "four_words"
        )
        scores = {
            candidate["piece"]: (
                candidate["in_positions"],
                candidate["in_characters"],
            )
            for candidate in four["candidates"]
        }

        assert scores["est</w>"] == (27, 36)
        assert scores["ewest</w>"] == (30, 36)
        assert four["total_positions"] == 79
        assert four["total_characters"] == 95

    def test_the_same_four_pieces_come_out_under_either_unit(self) -> None:
        body = client.get(UNITS).json()
        four = next(
            corpus for corpus in body["corpora"] if corpus["key"] == "four_words"
        )

        assert [row["piece"] for row in four["run_by_positions"]] == [
            "newest</w>",
            "widest</w>",
            "low</w>",
            "lower</w>",
        ]
        assert [row["piece"] for row in four["run_by_characters"]] == [
            "newest</w>",
            "widest</w>",
            "low</w>",
            "lower</w>",
        ]
        assert [row["in_characters"] for row in four["run_by_characters"]] == [
            42,
            21,
            20,
            12,
        ]

    def test_on_six_words_the_unit_decides_the_first_pick(self) -> None:
        body = client.get(UNITS).json()
        six = next(corpus for corpus in body["corpora"] if corpus["key"] == "six_words")

        assert six["same_winner"] is False
        assert six["winner_by_positions"] == "st"
        assert six["winner_positions_score"] == 30
        assert six["winner_by_characters"] == "est</w>"
        assert six["winner_characters_score"] == 36
        assert [row["piece"] for row in six["run_by_positions"]][:2] == [
            "st",
            "est</w>",
        ]
        assert [row["piece"] for row in six["run_by_characters"]][:2] == [
            "est</w>",
            "st",
        ]

    def test_the_position_run_is_the_trainers_own_answer(self) -> None:
        """The scorer here is only trusted because it reproduces the library."""
        body = client.get(UNITS).json()
        six = next(corpus for corpus in body["corpora"] if corpus["key"] == "six_words")
        counts = {"test": 3, "rest": 3, "best": 3, "stem": 5, "stop": 5, "stub": 5}
        corpus = [" ".join(w for w, c in counts.items() for _ in range(c))]
        fitted = GreedyCoverageTokenizer(vocabulary_size=100).fit(corpus)

        assert [
            (row["piece"], row["in_positions"]) for row in six["run_by_positions"]
        ] == [
            (chosen.piece, chosen.coverage) for chosen in list(fitted.chosen_pieces)[:3]
        ]


class TestMatched:
    def test_the_corpus_is_the_one_the_merging_pages_carry(self) -> None:
        body = client.get(MATCHED).json()

        assert body["n_distinct_words"] == 72
        assert body["n_word_occurrences"] == 133
        assert body["alphabet_size"] == 51
        assert body["total_positions"] == 648

    def test_the_two_orderings_agree_at_the_top_and_part_immediately(self) -> None:
        body = client.get(MATCHED).json()

        assert body["n_candidates"] == 815
        assert [(row["pair"], row["count"]) for row in body["top_pairs"]] == [
            ("h + e</w>", 31),
            ("t + h", 20),
            ("r + e", 17),
            ("T + h", 12),
        ]

    def test_most_of_the_sentence_falls_back_to_bare_symbols(self) -> None:
        body = client.get(MATCHED).json()

        assert body["sentence_bare_by_coverage"] == 24
        assert body["sentence_bare_by_merging"] == 17

    def test_coverage_reads_the_training_corpus_shorter_at_every_size(self) -> None:
        body = client.get(MATCHED).json()

        for row in body["rows"]:
            assert row["coverage_corpus"] <= row["merging_corpus"], row["asked"]

    def test_the_held_out_sentence_crosses_over_at_seventy(self) -> None:
        body = client.get(MATCHED).json()
        rows = {row["asked"]: row for row in body["rows"]}

        assert body["crossover"] == 70
        assert (rows[60]["coverage_sentence"], rows[60]["merging_sentence"]) == (35, 39)
        assert (rows[70]["coverage_sentence"], rows[70]["merging_sentence"]) == (35, 32)
        assert (rows[100]["coverage_sentence"], rows[100]["merging_sentence"]) == (
            32,
            27,
        )
        assert (rows[100]["coverage_corpus"], rows[100]["merging_corpus"]) == (296, 344)

    def test_the_two_ceilings_differ(self) -> None:
        body = client.get(MATCHED).json()

        assert body["coverage_ceiling"] == 121
        assert body["merging_ceiling"] == 137
        assert body["rows"][-1]["asked"] == 121
        assert body["rows"][-1]["coverage_corpus"] == 267
        assert body["rows"][-1]["merging_corpus"] == 295

    def test_the_two_vocabularies_share_twelve_of_forty_eight_rows(self) -> None:
        body = client.get(MATCHED).json()
        compared = body["compared"]

        assert compared["size"] == 100
        assert len(compared["coverage_pieces"]) == 48
        assert len(compared["merging_pieces"]) == 48
        assert len(compared["shared"]) == 12
        assert compared["coverage_whole_words"] == 18
        assert compared["merging_whole_words"] == 8
        assert round(compared["coverage_mean_length"], 3) == 3.771
        assert round(compared["merging_mean_length"], 3) == 3.042

    def test_the_first_pieces_are_long_and_the_stem_is_the_cost(self) -> None:
        body = client.get(MATCHED).json()
        compared = body["compared"]
        readings = {reading["text"]: reading for reading in body["readings"]}

        assert compared["coverage_pieces"][:4] == [
            "he</w>",
            "analys",
            "expecte",
            "re",
        ]
        assert readings["expected"]["coverage"] == ["expecte", "d</w>"]
        assert readings["expect"]["coverage"] == ["e", "x", "p", "e", "c", "t</w>"]
        assert readings["expect"]["merging"] == ["exp", "ec", "t</w>"]

    def test_the_round_trip_is_exact_on_a_sentence_of_the_corpus(self) -> None:
        body = client.get(MATCHED).json()

        assert body["control_round_trip_exact"] is True
        assert body["control_pieces"] == 10
        assert body["sentence_round_trip_exact"] is False
        assert "[UNK]" in body["sentence_decoded"]


class TestLimits:
    def test_the_gains_can_only_fall(self) -> None:
        body = client.get(LIMITS).json()
        gains = [row["coverage"] for row in body["gains"]]

        assert gains[:8] == [62, 42, 35, 34, 24, 21, 21, 16]
        assert all(earlier >= later for earlier, later in itertools.pairwise(gains))
        assert len(gains) == 69
        assert body["total_positions"] == 648
        assert body["covered_positions"] == 576
        assert body["covered_positions"] == sum(gains)
        assert body["longest_tie_run"] == 13
        assert body["longest_tie_value"] == 2

    def test_a_long_piece_seen_seldom_outscores_a_short_one_seen_often(self) -> None:
        body = client.get(LIMITS).json()
        chosen = {row["piece"]: row for row in body["gains"][:4]}

        assert (chosen["analys"]["n_symbols"], chosen["analys"]["occurrences"]) == (
            6,
            7,
        )
        assert (chosen["re"]["n_symbols"], chosen["re"]["occurrences"]) == (2, 17)
        assert chosen["analys"]["coverage"] > chosen["re"]["coverage"]
        assert (chosen["he</w>"]["n_symbols"], chosen["he</w>"]["occurrences"]) == (
            2,
            31,
        )

    def test_the_best_pair_covers_everything_and_the_rule_does_not(self) -> None:
        body = client.get(LIMITS).json()
        rows = {row["budget"]: row for row in body["trap_rows"]}

        assert body["trap_positions"] == 39
        assert rows[1]["greedy_pieces"] == ["cost"]
        assert rows[1]["greedy_coverage"] == rows[1]["best_coverage"] == 24
        assert rows[2]["greedy_pieces"] == ["cost", "ing</w>"]
        assert rows[2]["greedy_coverage"] == 33
        assert rows[2]["best_pieces"] == ["costing</w>", "costly</w>"]
        assert rows[2]["best_coverage"] == 39
        assert round(rows[2]["ratio"], 4) == 0.8462

    def test_the_shortfall_stays_above_the_worst_case_guarantee(self) -> None:
        body = client.get(LIMITS).json()

        assert round(body["guarantee"], 4) == 0.6321
        for row in body["trap_rows"]:
            assert row["ratio"] >= body["guarantee"]

    def test_the_forfeited_pair_costs_pieces_as_well_as_coverage(self) -> None:
        body = client.get(LIMITS).json()

        assert body["trap_greedy_corpus"] == 15
        assert body["trap_best_corpus"] == 6

    def test_coverage_spends_rows_on_words_seen_once_and_merging_never_does(
        self,
    ) -> None:
        body = client.get(LIMITS).json()
        rows = {row["size"]: row for row in body["once_rows"]}

        assert body["n_words_seen_once"] == 54
        assert body["n_distinct_words"] == 72
        assert (rows[100]["coverage_rows"], rows[100]["coverage_once"]) == (48, 10)
        assert (rows[121]["coverage_rows"], rows[121]["coverage_once"]) == (69, 13)
        assert all(row["merging_once"] == 0 for row in body["once_rows"])
        assert "Tuesday.</w>" in rows[100]["examples"]


class TestCapping:
    def test_a_shorter_ceiling_reads_the_held_out_sentence_better(self) -> None:
        body = client.get(CAPPING).json()
        rows = {row["cap"]: row for row in body["rows"]}

        assert body["size"] == 100
        assert (rows[16]["corpus_pieces"], rows[16]["sentence_pieces"]) == (296, 32)
        assert (rows[5]["corpus_pieces"], rows[5]["sentence_pieces"]) == (324, 28)
        assert (rows[3]["corpus_pieces"], rows[3]["sentence_pieces"]) == (384, 29)

    def test_the_ceiling_stops_binding_once_it_is_long_enough(self) -> None:
        body = client.get(CAPPING).json()
        rows = {row["cap"]: row for row in body["rows"]}

        assert rows[8]["corpus_pieces"] == rows[16]["corpus_pieces"]
        assert rows[8]["first_pieces"] == rows[16]["first_pieces"]
        assert rows[5]["once_seen_rows"] == 2
        assert rows[16]["once_seen_rows"] == 10


class TestPrefixReading:
    """Every static endpoint reads a smaller vocabulary off one longer fit."""

    def test_a_prefix_gives_what_a_genuine_fit_of_that_size_gives(self) -> None:
        counts = _word_counts(SENTENCE_CORPUS)
        for size in (60, 100):
            genuine = GreedyCoverageTokenizer(vocabulary_size=size).fit(SENTENCE_CORPUS)
            prefix = _prefix_at(CorpusChoice.SENTENCES, size)

            assert list(prefix.chosen.pieces) == [
                chosen.piece for chosen in genuine.chosen_pieces
            ]
            assert prefix.n_tokens == genuine.vocabulary.n_tokens
            assert list(prefix.vocabulary) == list(genuine.vocabulary)
            assert prefix.pieces_of(SENTENCE) == list(genuine.encode(SENTENCE).texts)
            assert prefix.corpus_pieces(counts) == sum(
                genuine.encode(word).n_tokens * count for word, count in counts.items()
            )


class TestFit:
    def test_the_playground_reads_a_text_with_the_size_it_was_given(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "four_words",
                "vocabulary_size": 13,
                "text": "newest lowest",
            },
        ).json()

        assert body["smallest_size"] == 12
        assert body["learned"] == 13
        assert body["n_pieces"] == 1
        assert body["pieces"][0]["piece"] == "newest</w>"
        assert body["pieces"][0]["coverage"] == 36
        assert body["text_pieces"] == [
            "newest</w>",
            "l",
            "o",
            "w",
            "e",
            "s",
            "t</w>",
        ]
        assert body["covered_positions"] == 36
        assert body["total_positions"] == 79
        assert body["corpus_pieces"] == 49
        assert body["round_trip_exact"] is True

    def test_the_sentence_corpus_reads_its_own_sentence_back_exactly(self) -> None:
        body = client.post(
            FIT,
            json={
                "corpus": "sentences",
                "vocabulary_size": 100,
                "text": "The report was expected on Monday.",
            },
        ).json()

        assert body["n_text_pieces"] == 10
        assert body["unknown_pieces"] == 0
        assert body["round_trip_exact"] is True

    def test_a_vocabulary_below_the_alphabet_is_refused(self) -> None:
        response = client.post(FIT, json={"corpus": "sentences", "vocabulary_size": 20})

        assert response.status_code == 422
        assert "52" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            FIT,
            json={"corpus": "sentences", "vocabulary_size": 100, "vocabularySize": 60},
        )

        assert response.status_code == 422


IDENTIFIER = re.compile(
    r"[a-zA-Z_]+_[a-zA-Z_]+|[a-z]+[A-Z][a-zA-Z]*|\b[A-Z][a-z]+[A-Z]"
)
"""Anything shaped like a parameter name or a class name."""


class TestNothingIdentifierShapedReachesThePage:
    def test_the_refusal_a_reader_sees_names_no_parameter(self) -> None:
        detail = client.post(
            FIT, json={"corpus": "sentences", "vocabulary_size": 20}
        ).json()["detail"]

        assert IDENTIFIER.search(detail) is None, detail

    def test_no_endpoint_returns_a_string_shaped_like_an_identifier(self) -> None:
        for path in (CHOOSING, UNITS, MATCHED, LIMITS, CAPPING):
            for text in _every_string(client.get(path).json()):
                assert IDENTIFIER.search(text) is None, (path, text)


def _every_string(value: object) -> list[str]:
    """Every string in a response, except the pieces and texts a page shows.

    A vocabulary piece is text of the corpus rather than prose, and the corpus
    contains ``Dr.`` and ``Tuesday.``; the scan is on the labels and messages a
    reader reads as English.
    """
    if isinstance(value, str):
        return [value] if " " in value and len(value) > 12 else []
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            if key not in ("corpus", "sentence", "control")
            for found in _every_string(item)
        ]
    if isinstance(value, list):
        return [found for item in value for found in _every_string(item)]
    return []
