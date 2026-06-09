"""The pattern endpoints, on the running sentence and the six-sentence notebook.

Every number the page quotes is pinned here, so a change in the library or in
this module fails a test rather than quietly changing a sentence on the page.
The last class is the one that matters most: nothing these endpoints return may
carry an identifier, because a widget renders these strings.
"""

from __future__ import annotations

import re
from itertools import pairwise

from fastapi.testclient import TestClient

from oop_ml_api.concepts.pattern_words import (
    NOTEBOOK,
    RUNNING_SENTENCE,
    SPACED_SENTENCE,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/the-pattern-language-models-use"
SPLIT = f"{BASE}/split"
BRANCHES = f"{BASE}/branches"
CONTRACTIONS = f"{BASE}/contractions"
CORPUS = f"{BASE}/corpus"
DIALECT = f"{BASE}/dialect"


def split_one(text: str, pattern: str = "the-pattern") -> dict:
    """One text put to one pattern, unwrapped to the view the page quotes."""
    response = client.post(SPLIT, json={"texts": [text], "pattern": pattern})
    assert response.status_code == 200
    return response.json()["texts"][0]


class TestTheRunningSentence:
    def test_the_sentence_becomes_fourteen_pieces(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_pieces"] == 14
        assert [piece["text"] for piece in view["pieces"]] == [
            "Dr",
            ".",
            " Alvarez",
            " didn",
            "'t",
            " expect",
            " the",
            " low",
            "-",
            "cost",
            " re",
            "-",
            "analysis",
            ".",
        ]

    def test_six_of_the_fourteen_carry_a_leading_space(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["n_with_a_leading_space"] == 6
        assert sum(1 for piece in view["pieces"] if piece["has_a_leading_space"]) == 6

    def test_the_pieces_join_back_to_the_sentence_character_for_character(
        self,
    ) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["joins_back_exactly"] is True
        assert view["n_characters_in_pieces"] == view["n_characters"] == 51
        assert "".join(piece["text"] for piece in view["pieces"]) == RUNNING_SENTENCE

    def test_every_piece_is_the_slice_its_span_names(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        for piece in view["pieces"]:
            assert RUNNING_SENTENCE[piece["start"] : piece["end"]] == piece["text"]

    def test_the_spans_tile_the_sentence_with_no_gaps(self) -> None:
        view = split_one(RUNNING_SENTENCE)
        pieces = view["pieces"]

        assert pieces[0]["start"] == 0
        assert pieces[-1]["end"] == len(RUNNING_SENTENCE)
        for earlier, later in pairwise(pieces):
            assert earlier["end"] == later["start"]

    def test_none_of_the_four_earlier_rules_gives_the_sentence_back(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        for key in (
            "on_spaces",
            "on_boundaries",
            "on_annotation_rules",
            "on_translation_rules",
        ):
            assert view[key]["joins_back_exactly"] is False

    def test_the_four_earlier_rules_give_the_counts_the_page_quotes(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_spaces"]["n_pieces"] == 7
        assert view["on_boundaries"]["n_pieces"] == 9
        assert view["on_annotation_rules"]["n_pieces"] == 9
        assert view["on_translation_rules"]["n_pieces"] == 9

    def test_only_the_boundary_rules_lose_characters_as_well_as_spacing(self) -> None:
        view = split_one(RUNNING_SENTENCE)

        assert view["on_spaces"]["n_characters_in_pieces"] == 45
        assert view["on_annotation_rules"]["n_characters_in_pieces"] == 45
        assert view["on_translation_rules"]["n_characters_in_pieces"] == 45
        assert view["on_boundaries"]["n_characters_in_pieces"] == 41
        assert view["n_non_space_characters"] == 45


class TestTheOfferedPatterns:
    def test_deleting_the_optional_space_lengthens_the_sentence(self) -> None:
        whole = split_one(RUNNING_SENTENCE)
        without = split_one(RUNNING_SENTENCE, "without-the-optional-space")

        assert whole["n_pieces"] == 14
        assert without["n_pieces"] == 20
        assert without["joins_back_exactly"] is True
        assert without["n_with_a_leading_space"] == 6
        assert [
            piece["text"] for piece in without["pieces"] if piece["has_a_leading_space"]
        ] == [" "] * 6

    def test_deleting_the_contractions_costs_one_more_piece(self) -> None:
        without = split_one(RUNNING_SENTENCE, "without-the-contractions")

        assert without["n_pieces"] == 15
        assert without["pieces"][4]["text"] == "'"
        assert without["pieces"][5]["text"] == "t"

    def test_a_pattern_that_names_only_letters_drops_the_rest(self) -> None:
        letters = split_one(RUNNING_SENTENCE, "letters-only")

        assert letters["n_pieces"] == 10
        assert letters["joins_back_exactly"] is False
        assert letters["n_characters_in_pieces"] == 40

    def test_splitting_on_spaces_written_as_a_pattern_is_the_crude_rule(self) -> None:
        runs = split_one(RUNNING_SENTENCE, "runs-with-no-spaces")

        assert [piece["text"] for piece in runs["pieces"]] == runs["on_spaces"][
            "pieces"
        ]

    def test_only_the_page_s_own_pattern_credits_a_branch(self) -> None:
        whole = split_one(RUNNING_SENTENCE)
        other = split_one(RUNNING_SENTENCE, "words-and-marks")

        assert all(piece["branch"] is not None for piece in whole["pieces"])
        assert all(piece["branch"] is None for piece in other["pieces"])

    def test_a_pattern_outside_the_offered_seven_is_refused(self) -> None:
        response = client.post(
            SPLIT, json={"texts": ["one two"], "pattern": "something-else"}
        )

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(
            SPLIT, json={"texts": ["one two"], "patern": "the-pattern"}
        )

        assert response.status_code == 422

    def test_a_text_longer_than_the_limit_is_refused(self) -> None:
        response = client.post(SPLIT, json={"texts": ["a" * 401]})

        assert response.status_code == 422


class TestTheBranches:
    def test_there_are_twelve_of_them_and_six_sentences_reach_four(self) -> None:
        body = client.get(BRANCHES).json()

        assert body["n_branches"] == 12
        assert body["n_branches_used"] == 4
        assert body["n_pieces"] == 85
        assert body["n_texts"] == 6

    def test_the_letter_branch_claims_four_pieces_in_five(self) -> None:
        body = client.get(BRANCHES).json()
        letters = body["branches"][7]

        assert letters["source"] == r" ?[^\W\d_]+"
        assert letters["n_claimed"] == 68
        assert round(letters["share_claimed"], 4) == round(68 / 85, 4)

    def test_the_tallies_the_page_quotes(self) -> None:
        body = client.get(BRANCHES).json()
        claimed = [branch["n_claimed"] for branch in body["branches"]]

        assert claimed == [0, 1, 0, 0, 0, 0, 0, 68, 1, 15, 0, 0]
        assert sum(claimed) == body["n_pieces"]

    def test_every_piece_of_the_notebook_is_credited_to_some_branch(self) -> None:
        body = client.get(BRANCHES).json()

        assert sum(branch["n_claimed"] for branch in body["branches"]) == 85

    def test_the_held_back_space_changes_which_piece_owns_it(self) -> None:
        held = client.get(BRANCHES).json()["held_space"]

        assert held["text"] == SPACED_SENTENCE
        assert held["with_the_branch"] == [
            "Dr",
            ".",
            " ",
            " Alvarez",
            " ",
            " didn",
            "'t",
            " expect",
        ]
        assert held["without_the_branch"] == [
            "Dr",
            ".",
            "  ",
            "Alvarez",
            "  ",
            "didn",
            "'t",
            " expect",
        ]
        assert held["n_with"] == held["n_without"] == 8

    def test_four_thousand_drawn_texts_all_come_back_from_their_pieces(self) -> None:
        drawn = client.get(BRANCHES).json()["random_texts"]

        assert drawn["n_texts"] == 4000
        assert drawn["n_joining_back"] == 4000
        assert drawn["n_codepoints_drawn_from"] == 1972
        assert drawn["longest_text"] == 30

    def test_a_pattern_that_is_not_well_formed_names_no_rule(self) -> None:
        refusals = client.get(BRANCHES).json()["refusals"]

        assert len(refusals) == 3
        for refusal in refusals:
            assert "refused before it read anything" in refusal["reason"]


class TestTheContractions:
    def test_seven_spellings_and_half_the_probes_miss(self) -> None:
        body = client.get(CONTRACTIONS).json()

        assert body["n_clitics"] == 7
        assert body["clitics"] == ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        assert body["n_probes"] == 12
        assert body["n_firing"] == 6

    def test_capitals_and_a_curved_apostrophe_both_miss(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(CONTRACTIONS).json()["probes"]
        }

        assert probes["IT'S HERE"]["pieces"] == ["IT", "'", "S", " HERE"]
        assert probes["IT'S HERE"]["a_contraction_branch_fired"] is False
        assert probes["she didn’t go"]["pieces"] == [
            "she",
            " didn",
            "’",
            "t",
            " go",
        ]
        assert probes["she didn’t go"]["a_contraction_branch_fired"] is False

    def test_the_branch_fires_on_a_word_that_is_not_a_contraction(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(CONTRACTIONS).json()["probes"]
        }

        assert probes["'twas cold"]["pieces"] == ["'t", "was", " cold"]
        assert probes["'twas cold"]["a_contraction_branch_fired"] is True

    def test_the_three_negations_are_cut_in_the_same_place_every_time(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(CONTRACTIONS).json()["probes"]
        }
        three = probes["he can't won't shan't"]

        assert three["pieces"] == [
            "he",
            " can",
            "'t",
            " won",
            "'t",
            " shan",
            "'t",
        ]
        assert three["on_annotation_rules"] == [
            "he",
            "ca",
            "n't",
            "wo",
            "n't",
            "sha",
            "n't",
        ]

    def test_a_word_the_list_does_not_hold_comes_apart_into_three(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(CONTRACTIONS).json()["probes"]
        }

        assert probes["y'all came"]["pieces"] == ["y", "'", "all", " came"]
        assert probes["o'clock struck"]["pieces"] == ["o", "'", "clock", " struck"]

    def test_the_two_later_changes_repair_the_capitals_and_cap_the_digits(self) -> None:
        changes = client.get(CONTRACTIONS).json()["later_changes"]

        capitals, figures = changes
        assert capitals["before"] == [
            "THE",
            " DRIFT",
            " DIDN",
            "'",
            "T",
            " GO",
            " AWAY",
        ]
        assert capitals["after"] == [
            "THE",
            " DRIFT",
            " DIDN",
            "'T",
            " GO",
            " AWAY",
        ]
        assert figures["before"] == [
            "in",
            " 2019",
            " the",
            " drift",
            " was",
            " 1000000",
            " counts",
        ]
        assert figures["after"] == [
            "in",
            " 201",
            "9",
            " the",
            " drift",
            " was",
            " 100",
            "000",
            "0",
            " counts",
        ]


class TestTheCorpus:
    def test_the_five_rules_give_the_counts_the_page_quotes(self) -> None:
        readings = {
            reading["label"]: reading
            for reading in client.get(CORPUS).json()["readings"]
        }

        assert readings["every run of characters with no spacing"]["n_pieces"] == 63
        assert readings["the standard boundary rules"]["n_pieces"] == 69
        assert readings["the annotation rules"]["n_pieces"] == 72
        assert readings["the translation rules"]["n_pieces"] == 72
        assert readings["the pattern"]["n_pieces"] == 85

    def test_only_the_pattern_gives_all_six_sentences_back(self) -> None:
        body = client.get(CORPUS).json()
        readings = {reading["label"]: reading for reading in body["readings"]}

        assert readings["the pattern"]["n_joining_back_exactly"] == 6
        for label in (
            "every run of characters with no spacing",
            "the standard boundary rules",
            "the annotation rules",
            "the translation rules",
        ):
            assert readings[label]["n_joining_back_exactly"] == 0

    def test_one_space_between_each_pair_rescues_only_the_crudest_rule(self) -> None:
        readings = {
            reading["label"]: reading
            for reading in client.get(CORPUS).json()["readings"]
        }

        assert (
            readings["every run of characters with no spacing"][
                "n_joining_back_with_one_space"
            ]
            == 6
        )
        assert readings["the pattern"]["n_joining_back_with_one_space"] == 0

    def test_the_pattern_keeps_every_character_and_the_boundary_rules_do_not(
        self,
    ) -> None:
        body = client.get(CORPUS).json()
        readings = {reading["label"]: reading for reading in body["readings"]}

        assert body["n_characters"] == 375
        assert body["n_non_space_characters"] == 318
        assert readings["the pattern"]["n_characters_in_pieces"] == 375
        assert readings["the pattern"]["share_of_characters"] == 1.0
        assert readings["the standard boundary rules"]["n_characters_in_pieces"] == 303

    def test_the_distinct_counts_the_page_quotes(self) -> None:
        readings = {
            reading["label"]: reading
            for reading in client.get(CORPUS).json()["readings"]
        }

        assert readings["every run of characters with no spacing"]["n_distinct"] == 47
        assert readings["the standard boundary rules"]["n_distinct"] == 47
        assert readings["the annotation rules"]["n_distinct"] == 48
        assert readings["the translation rules"]["n_distinct"] == 49
        assert readings["the pattern"]["n_distinct"] == 53

    def test_the_optional_space_saves_pieces_and_costs_one_row(self) -> None:
        body = client.get(CORPUS).json()
        readings = {reading["label"]: reading for reading in body["readings"]}
        standing_alone = body["without_the_optional_space"]

        assert readings["the pattern"]["n_pieces"] == 85
        assert standing_alone["n_pieces"] == 142
        assert readings["the pattern"]["n_distinct"] == 53
        assert standing_alone["n_distinct"] == 52
        assert standing_alone["n_joining_back_exactly"] == 6

    def test_two_words_of_the_six_sentences_are_spelt_both_ways(self) -> None:
        body = client.get(CORPUS).json()

        assert body["n_distinct"] == 53
        assert body["n_distinct_with_a_leading_space"] == 41
        assert [row["word"] for row in body["doubled"]] == ["Alvarez", "analysis"]
        assert body["doubled"][1]["n_with_the_space"] == 2
        assert body["doubled"][1]["n_without_the_space"] == 2

    def test_only_the_pattern_tells_the_spacing_pairs_apart(self) -> None:
        pairs = client.get(CORPUS).json()["told_apart"]

        assert len(pairs) == 4
        for pair in pairs:
            assert pair["told_apart_here"] is True
            assert pair["told_apart_on_spaces"] is False
            assert pair["told_apart_on_boundaries"] is False
            assert pair["told_apart_on_annotation_rules"] is False
            assert pair["told_apart_on_translation_rules"] is False

    def test_a_script_that_writes_no_spaces_comes_back_as_one_piece(self) -> None:
        body = client.get(CORPUS).json()

        assert body["scriptless_pieces"] == [body["scriptless_text"]]
        assert len(body["scriptless_on_boundaries"]) == 6

    def test_the_corpus_is_the_one_the_page_names(self) -> None:
        body = client.get(CORPUS).json()

        assert body["notebook"] == list(NOTEBOOK)
        assert body["running"]["source"] == RUNNING_SENTENCE


class TestTheDialect:
    def test_the_two_classes_disagree_about_1151_codepoints(self) -> None:
        counts = client.get(DIALECT).json()["counts"]

        assert counts["n_number_codepoints"] == 1831
        assert counts["n_read_as_digits_here"] == 680
        assert counts["n_read_as_letters_here"] == 1151
        assert (
            counts["n_read_as_digits_here"] + counts["n_read_as_letters_here"]
            == counts["n_number_codepoints"]
        )

    def test_every_letter_of_unicode_is_a_letter_under_both_readings(self) -> None:
        counts = client.get(DIALECT).json()["counts"]

        assert counts["n_letter_codepoints"] == 136104
        assert counts["n_letters_agreeing"] == counts["n_letter_codepoints"]

    def test_the_three_texts_that_split_differently(self) -> None:
        probes = {
            probe["text"]: probe for probe in client.get(DIALECT).json()["probes"]
        }

        assert probes["the drift was 1½ per cent"]["as_published"] == [
            "the",
            " drift",
            " was",
            " 1½",
            " per",
            " cent",
        ]
        assert probes["the drift was 1½ per cent"]["here"] == [
            "the",
            " drift",
            " was",
            " 1",
            "½",
            " per",
            " cent",
        ]
        assert probes["page aⅫ of the report"]["as_published"] == [
            "page",
            " a",
            "Ⅻ",
            " of",
            " the",
            " report",
        ]
        assert probes["page aⅫ of the report"]["here"] == [
            "page",
            " aⅫ",
            " of",
            " the",
            " report",
        ]
        assert probes["a² and x₂"]["as_published"] == [
            "a",
            "²",
            " and",
            " x",
            "₂",
        ]
        assert probes["a² and x₂"]["here"] == ["a²", " and", " x₂"]

    def test_four_of_the_seven_texts_agree(self) -> None:
        body = client.get(DIALECT).json()

        assert body["n_probes"] == 7
        assert body["n_agreeing"] == 4

    def test_the_disagreement_needs_a_letter_or_a_digit_beside_it(self) -> None:
        rows = {row["context"]: row for row in client.get(DIALECT).json()["adjacency"]}

        assert rows["standing alone between two spaces"]["n_differing"] == 0
        assert rows["directly after a letter"]["n_differing"] == 1151
        assert rows["directly after a digit"]["n_differing"] == 1151
        assert rows["directly before a letter"]["n_differing"] == 1151
        for row in rows.values():
            assert row["n_characters"] == 1151

    def test_the_six_sentences_read_identically_under_both(self) -> None:
        body = client.get(DIALECT).json()

        assert body["n_notebook_texts_differing"] == 0

    def test_without_the_underscore_written_back_in_a_character_vanishes(self) -> None:
        underscore = client.get(DIALECT).json()["underscore"]

        assert underscore["with_the_repair"] == [
            "re",
            "-",
            "analysis",
            "_",
            "2019",
            ".",
            "csv",
        ]
        assert underscore["without_the_repair"] == [
            "re",
            "-",
            "analysis",
            "2019",
            ".",
            "csv",
        ]
        assert underscore["joins_back_with_the_repair"] is True
        assert underscore["joins_back_without_the_repair"] is False
        assert underscore["n_characters_lost"] == 1


IDENTIFIER = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
"""A snake_case or CamelCase word, which is what an identifier looks like in prose."""

FORBIDDEN = ("oop_ml", "the library", "the API", "the package", "the backend")


def _every_string(payload: object) -> list[str]:
    """Every string anywhere in a response body, keys excluded."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list):
        return [text for item in payload for text in _every_string(item)]
    if isinstance(payload, dict):
        return [text for item in payload.values() for text in _every_string(item)]
    return []


def _notation() -> set[str]:
    """The patterns and the writing given to the rules, which the page shows on purpose.

    A pattern is the notation this page is about, and the texts handed to the
    rules include a file name with an underscore in it, which is exactly the
    writing the third branch had to be repaired for. Both would read as
    identifiers to a check written for prose, so they are gathered from the
    responses themselves and excluded, and every label, note and verdict written
    here is checked.
    """
    branches = client.get(BRANCHES).json()
    corpus = client.get(CORPUS).json()
    dialect = client.get(DIALECT).json()
    contractions = client.get(CONTRACTIONS).json()

    written: set[str] = {branches["pattern_source"]}
    written.update(branch["source"] for branch in branches["branches"])
    written.update(refusal["source"] for refusal in branches["refusals"])
    written.update(change["was"] for change in contractions["later_changes"])
    written.update(change["became"] for change in contractions["later_changes"])
    written.update(corpus["notebook"])
    written.update(
        text
        for pair in corpus["told_apart"]
        for text in (pair["first"], pair["second"])
    )
    written.update(probe["text"] for probe in dialect["probes"])
    written.add(dialect["underscore"]["text"])
    written.update(dialect["underscore"]["with_the_repair"])
    written.update(dialect["underscore"]["without_the_repair"])
    written.update(
        dialect[key]
        for key in (
            "published_letters",
            "published_numbers",
            "letters_here",
            "numbers_here",
            "marks_here",
        )
    )
    return written


class TestNothingReadableNamesTheCode:
    def test_no_response_carries_an_identifier_or_credits_the_code(self) -> None:
        notation = _notation()
        bodies = [
            client.post(SPLIT, json={"texts": [RUNNING_SENTENCE]}).json(),
            client.get(BRANCHES).json(),
            client.get(CONTRACTIONS).json(),
            client.get(CORPUS).json(),
            client.get(DIALECT).json(),
        ]

        for body in bodies:
            for text in _every_string(body):
                if text in notation or "-" in text:
                    continue
                assert not IDENTIFIER.search(text), text
                for phrase in FORBIDDEN:
                    assert phrase not in text, text

    def test_no_response_carries_an_em_dash(self) -> None:
        for body in (
            client.get(BRANCHES).json(),
            client.get(CONTRACTIONS).json(),
            client.get(CORPUS).json(),
            client.get(DIALECT).json(),
        ):
            for text in _every_string(body):
                assert "—" not in text
