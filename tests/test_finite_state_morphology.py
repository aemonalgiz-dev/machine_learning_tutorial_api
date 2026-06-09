"""The finite-state morphology page's endpoints, on the three grammars it carries.

Every number the page quotes in prose is pinned here, so a change to the
library that moved one of them would fail a test rather than quietly rewrite
the page. Two other things are pinned. The walk the page draws is a second
implementation of the search, so a test asserts that the paths it says reach
the end spell exactly the readings the analyser returns; and the refusals a
reader can reach are asserted to carry nothing identifier-shaped, since a
widget renders them.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from oop_ml_api.concepts.finite_state_morphology import (
    CONSONANT_VERBS,
    NOUNS,
    SILENT_E_VERBS,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

BASE = "/concepts/finite-state-morphology"
MACHINE = f"{BASE}/machine"
SCENARIOS = f"{BASE}/scenarios"
COST = f"{BASE}/cost"
AMBIGUITY = f"{BASE}/ambiguity"
COVERAGE = f"{BASE}/coverage"
READ = f"{BASE}/read"


class TestMachine:
    def test_the_small_grammar_is_five_states_and_sixteen_lines(self) -> None:
        small = client.get(MACHINE).json()["grammars"][0]

        assert small["key"] == "small"
        assert small["n_states"] == 5
        assert small["n_written_lines"] == 16
        assert small["n_arcs"] == 16
        assert [state["name"] for state in small["states"]] == [
            "word",
            "verb ending",
            "ending that starts with a consonant",
            "ending that starts with a vowel",
            "noun ending",
        ]

    def test_the_small_grammar_accepts_twelve_words_and_reads_two_of_them_twice(
        self,
    ) -> None:
        small = client.get(MACHINE).json()["grammars"][0]

        assert small["n_forms"] == 12
        assert small["n_paths"] == 14
        assert small["n_ambiguous_forms"] == 2
        assert small["sample_forms"] == [
            "bake",
            "baked",
            "bakes",
            "baking",
            "talk",
            "talked",
            "talking",
            "talks",
            "walk",
            "walked",
            "walking",
            "walks",
        ]

    def test_the_grammar_before_the_spelling_change_accepts_the_wrong_words(
        self,
    ) -> None:
        before = client.get(MACHINE).json()["grammars"][1]

        assert before["key"] == "small before"
        assert before["n_states"] == 3
        assert before["n_written_lines"] == 12
        assert "bakeed" in before["sample_forms"]
        assert "bakeing" in before["sample_forms"]
        assert "baked" not in before["sample_forms"]

    def test_the_larger_grammar_is_a_hundred_and_seventy_eight_lines(self) -> None:
        larger = client.get(MACHINE).json()["grammars"][2]

        assert larger["key"] == "larger"
        assert larger["n_states"] == 6
        assert larger["n_written_lines"] == 178
        assert larger["n_arcs"] == 342
        assert larger["n_forms"] == 1074
        assert larger["n_paths"] == 1184
        assert larger["n_ambiguous_forms"] == 108

    def test_every_line_says_what_it_means_in_words(self) -> None:
        small = client.get(MACHINE).json()["grammars"][0]
        endings = next(
            state for state in small["states"] if state["name"] == "verb ending"
        )

        meanings = {line["surface"]: line["meaning"] for line in endings["lines"]}
        assert meanings == {
            "s": "he or she does it",
            "ed": "it already happened",
            "ing": "it is going on",
        }
        assert all(line["ends_the_word"] for line in endings["lines"])


class TestScenarios:
    def test_walked_is_the_verb_and_the_past(self) -> None:
        walked = client.get(SCENARIOS).json()["walked"]

        assert walked["is_readable"] is True
        assert walked["n_analyses"] == 1
        assert walked["analyses"][0]["tags"] == "walk+V +PAST"
        assert [morph["text"] for morph in walked["analyses"][0]["morphs"]] == [
            "walk",
            "ed",
        ]
        assert walked["cut"] == ["walk", "ed"]

    def test_the_walk_reports_the_state_each_character_was_read_in(self) -> None:
        walked = client.get(SCENARIOS).json()["walked"]

        assert [
            (row["character"], row["state"]) for row in walked["states_by_character"]
        ] == [
            ("w", "word"),
            ("a", "word"),
            ("l", "word"),
            ("k", "word"),
            ("e", "verb ending"),
            ("d", "verb ending"),
        ]

    def test_five_lines_are_tried_to_read_walked_and_four_lead_nowhere(self) -> None:
        walked = client.get(SCENARIOS).json()["walked"]

        assert walked["n_lines_tried"] == 5
        assert sum(1 for step in walked["trace"] if step["leads_anywhere"]) == 2
        assert walked["trace"][0]["outcome"] == "carries on in the verb ending"
        assert walked["trace"][1]["outcome"] == "the word ends here"

    def test_walks_is_read_two_ways_and_the_noun_reading_is_taken_for_the_cut(
        self,
    ) -> None:
        walks = client.get(SCENARIOS).json()["walks"]

        assert walks["n_analyses"] == 2
        assert [analysis["tags"] for analysis in walks["analyses"]] == [
            "walk+N +PL",
            "walk+V +3SG",
        ]
        assert walks["cut"] == ["walk", "s"]

    def test_the_spelling_change_reverses_which_two_words_can_be_read(self) -> None:
        body = client.get(SCENARIOS).json()

        assert body["baked_before"]["is_readable"] is False
        assert body["bakeed_before"]["analyses"][0]["tags"] == "bake+V +PAST"
        assert body["baked_after"]["analyses"][0]["tags"] == "bake+V +PAST"
        assert [
            morph["text"] for morph in body["baked_after"]["analyses"][0]["morphs"]
        ] == ["bak", "ed"]
        assert body["bakeed_after"]["is_readable"] is False

    def test_the_full_stem_still_takes_the_ending_that_starts_with_a_consonant(
        self,
    ) -> None:
        bakes = client.get(SCENARIOS).json()["bakes"]

        assert bakes["cut"] == ["bake", "s"]
        assert bakes["n_lines_tried"] == 4

    def test_recovers_is_read_three_ways_and_all_three_are_real_english(self) -> None:
        recovers = client.get(SCENARIOS).json()["recovers"]

        assert recovers["n_analyses"] == 3
        assert [analysis["tags"] for analysis in recovers["analyses"]] == [
            "recover+V +3SG",
            "+RE cover+N +PL",
            "+RE cover+V +3SG",
        ]

    def test_a_word_of_four_pieces_labels_every_one_of_them(self) -> None:
        reopeners = client.get(SCENARIOS).json()["reopeners"]

        assert reopeners["analyses"][0]["n_morphemes"] == 4
        assert [
            (morph["text"], morph["meaning"])
            for morph in reopeners["analyses"][0]["morphs"]
        ] == [
            ("re", "again"),
            ("open", "the verb open"),
            ("er", "the one who does it"),
            ("s", "more than one"),
        ]

    def test_a_word_outside_the_grammar_comes_back_whole_and_unlabelled(self) -> None:
        unreadable = client.get(SCENARIOS).json()["unreadable"]

        assert [view["word"] for view in unreadable] == [
            "alvarez",
            "analysis",
            "busiest",
            "thinned",
        ]
        for view in unreadable:
            assert view["is_readable"] is False
            assert view["n_analyses"] == 0
            assert view["analyses"] == []
            assert view["cut"] == [view["word"]]


class TestTheWalkAgreesWithTheSearch:
    """The trace is a second implementation, so it is held to the first one."""

    def _paths(self, view: dict) -> set[str]:
        """Every reading the trace claims, rebuilt from the lines it took."""
        stack: list[str] = []
        found: set[str] = set()
        for step in view["trace"]:
            del stack[step["depth"] :]
            stack.append(step["surface"])
            if step["leads_anywhere"] and step["goes_to"] == "the end of the word":
                found.add(" ".join(stack))
        return found

    def test_every_path_the_walk_found_is_a_reading_the_search_returned(self) -> None:
        body = client.get(SCENARIOS).json()

        for key in ("walked", "walks", "bakes", "recovers", "reopeners"):
            view = body[key]
            returned = {
                " ".join(morph["text"] for morph in analysis["morphs"])
                for analysis in view["analyses"]
            }
            assert self._paths(view) == returned, key

    def test_a_walk_that_found_nothing_matches_a_word_with_no_reading(self) -> None:
        for view in client.get(SCENARIOS).json()["unreadable"]:
            assert self._paths(view) == set()


class TestCost:
    def test_a_hundred_and_seventy_eight_lines_accept_a_thousand_forms(self) -> None:
        body = client.get(COST).json()

        assert body["n_written_lines"] == 178
        assert body["n_states"] == 6
        assert body["n_forms"] == 1074
        assert body["n_paths"] == 1184
        assert body["mean_morphs"] == 2.64804469273743
        assert body["max_morphs"] == 4

    def test_the_same_budget_learned_by_counting_gives_a_longer_sequence(self) -> None:
        first = client.get(COST).json()["budgets"][0]

        assert first["n_pieces"] == 178
        assert first["mean_pieces"] == 2.9432029795158288
        assert first["n_whole_words"] == 0

    def test_twice_the_budget_beats_the_grammar_on_length(self) -> None:
        second = client.get(COST).json()["budgets"][1]

        assert second["n_pieces"] == 356
        assert second["mean_pieces"] == 2.0707635009310986
        assert second["n_whole_words"] == 37
        assert second["mean_pieces"] < client.get(COST).json()["mean_morphs"]

    def test_the_learned_cuts_land_inside_a_piece_where_the_grammar_does_not(
        self,
    ) -> None:
        cuts = {
            cut["word"]: cut for cut in client.get(COST).json()["budgets"][0]["cuts"]
        }

        assert cuts["recovers"]["learned"] == ["recov", "ers"]
        assert cuts["recovers"]["written"] == ["recover", "s"]
        assert cuts["bakes"]["learned"] == ["bak", "es"]
        assert cuts["bakes"]["written"] == ["bake", "s"]
        assert cuts["measured"]["learned"] == ["measure", "d"]
        assert cuts["measured"]["written"] == ["measur", "ed"]
        assert cuts["walked"]["agree"] is True

    def test_the_learned_answer_changes_with_the_budget_and_the_grammar_does_not(
        self,
    ) -> None:
        budgets = client.get(COST).json()["budgets"]
        first = {cut["word"]: cut for cut in budgets[0]["cuts"]}
        second = {cut["word"]: cut for cut in budgets[1]["cuts"]}

        assert first["measured"]["learned"] == ["measure", "d"]
        assert second["measured"]["learned"] == ["measured"]
        assert first["unnamed"]["learned"] == ["un", "nam", "ed"]
        assert second["unnamed"]["learned"] == ["unnam", "ed"]
        assert first["measured"]["written"] == second["measured"]["written"]


class TestAmbiguity:
    def test_a_tenth_of_the_grammars_own_words_are_read_more_than_one_way(
        self,
    ) -> None:
        body = client.get(AMBIGUITY).json()

        assert body["n_forms"] == 1074
        assert body["n_ambiguous_forms"] == 108
        assert round(body["share_ambiguous"], 4) == 0.1006

    def test_the_most_read_words_are_the_two_that_go_three_ways(self) -> None:
        most = client.get(AMBIGUITY).json()["most_read_words"]

        assert [(view["word"], view["n_analyses"]) for view in most] == [
            ("recover", 3),
            ("recovers", 3),
            ("answer", 2),
        ]

    def test_the_readings_of_a_repeated_character_grow_as_the_fibonacci_numbers(
        self,
    ) -> None:
        growth = client.get(AMBIGUITY).json()["growth"]

        assert [(row["length"], row["n_readings"]) for row in growth] == [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 5),
            (5, 8),
            (6, 13),
            (8, 34),
            (10, 89),
            (12, 233),
        ]


class TestCoverage:
    def test_neither_way_of_finding_words_helps_our_sentence(self) -> None:
        splitters = client.get(COVERAGE).json()["splitters"]

        assert [splitter["n_readable"] for splitter in splitters] == [1, 1]
        assert [len(splitter["pieces"]) for splitter in splitters] == [7, 9]
        assert [
            piece["text"] for piece in splitters[0]["pieces"] if piece["is_readable"]
        ] == ["expect"]

    def test_the_grammar_reads_a_ninth_of_an_ordinary_paragraph(self) -> None:
        paragraph = client.get(COVERAGE).json()["coverages"][1]

        assert paragraph["n_written_lines"] == 178
        assert paragraph["n_tokens"] == 76
        assert paragraph["n_readable_tokens"] == 7
        assert round(paragraph["token_share"], 4) == 0.0921
        assert paragraph["n_types"] == 57
        assert paragraph["n_readable_types"] == 7
        assert round(paragraph["type_share"], 4) == 0.1228

    def test_fifty_more_lines_take_the_reading_past_half(self) -> None:
        listed = client.get(COVERAGE).json()["coverages"][2]

        assert listed["n_written_lines"] == 228
        assert listed["n_readable_tokens"] == 46
        assert round(listed["token_share"], 4) == 0.6053
        assert listed["n_readable_types"] == 27
        assert round(listed["type_share"], 4) == 0.4737
        assert len(listed["unreadable_types"]) == 30

    def test_the_sentence_gives_up_two_words_of_nine(self) -> None:
        sentence = client.get(COVERAGE).json()["coverages"][0]

        assert sentence["n_tokens"] == 9
        assert sentence["n_readable_tokens"] == 2
        assert sentence["unreadable_types"] == [
            "alvarez",
            "analysis",
            "didn't",
            "dr",
            "low",
            "re",
            "the",
        ]


class TestRead:
    def test_the_playground_reads_a_word_with_the_larger_grammar(self) -> None:
        body = client.post(READ, json={"word": "recovers"}).json()

        assert body["n_written_lines"] == 178
        assert body["view"]["n_analyses"] == 3
        assert body["added_lines"] == []

    def test_a_word_can_be_read_with_the_small_grammar_instead(self) -> None:
        body = client.post(READ, json={"word": "walked", "grammar": "small"}).json()

        assert body["n_written_lines"] == 16
        assert body["view"]["analyses"][0]["tags"] == "walk+V +PAST"

    def test_adding_a_noun_adds_two_lines_and_makes_a_word_readable(self) -> None:
        before = client.post(READ, json={"word": "busy"}).json()
        after = client.post(
            READ,
            json={"word": "busy", "added": [{"surface": "busy", "kind": "a noun"}]},
        ).json()

        assert before["view"]["is_readable"] is False
        assert after["view"]["is_readable"] is True
        assert after["n_written_lines"] == 180
        assert after["view"]["analyses"][0]["tags"] == "busy+N"

    def test_a_verb_ending_in_e_costs_three_lines_and_drops_the_e(self) -> None:
        body = client.post(
            READ,
            json={
                "word": "smiled",
                "added": [{"surface": "smile", "kind": "a verb ending in e"}],
            },
        ).json()

        assert body["n_written_lines"] == 181
        assert len(body["added_lines"]) == 3
        assert body["view"]["analyses"][0]["tags"] == "smile+V +PAST"
        assert [morph["text"] for morph in body["view"]["analyses"][0]["morphs"]] == [
            "smil",
            "ed",
        ]


class TestRefusals:
    def test_an_empty_word_is_refused(self) -> None:
        response = client.post(READ, json={"word": ""})

        assert response.status_code == 422
        assert response.json()["detail"] == "there is no word here to read"

    def test_a_line_with_no_spelling_is_refused(self) -> None:
        response = client.post(
            READ, json={"word": "walks", "added": [{"surface": "", "kind": "a noun"}]}
        )

        assert response.status_code == 422
        assert "at least one character of spelling" in response.json()["detail"]

    def test_a_line_already_in_the_grammar_is_refused(self) -> None:
        response = client.post(
            READ,
            json={
                "word": "walks",
                "grammar": "small",
                "added": [{"surface": "walk", "kind": "a noun"}],
            },
        )

        assert response.status_code == 422
        assert "twice" in response.json()["detail"]

    def test_a_line_leading_to_a_state_that_is_not_there_is_refused(self) -> None:
        response = client.post(
            READ,
            json={
                "word": "walks",
                "grammar": "small before",
                "added": [{"surface": "smile", "kind": "a verb ending in e"}],
            },
        )

        assert response.status_code == 422
        assert "neither a state of this grammar" in response.json()["detail"]

    def test_an_extra_field_is_refused(self) -> None:
        response = client.post(READ, json={"word": "walks", "root": "word"})

        assert response.status_code == 422

    def test_a_word_beyond_the_ceiling_is_refused(self) -> None:
        response = client.post(READ, json={"word": "walk" * 20})

        assert response.status_code == 422


class TestTheNumbersTheProseQuotes:
    """Figures the page states in words rather than reading off a widget."""

    def test_thirteen_of_the_larger_grammars_words_are_both_a_noun_and_a_verb(
        self,
    ) -> None:
        assert len(CONSONANT_VERBS) == 24
        assert len(SILENT_E_VERBS) == 12
        assert len(NOUNS) == 40
        assert len(set(NOUNS) & (set(CONSONANT_VERBS) | set(SILENT_E_VERBS))) == 13

    def test_the_two_words_the_small_grammar_reads_twice_are_walk_and_walks(
        self,
    ) -> None:
        body = client.post(READ, json={"word": "walk", "grammar": "small"}).json()
        also = client.post(READ, json={"word": "walks", "grammar": "small"}).json()

        assert body["view"]["n_analyses"] == 2
        assert also["view"]["n_analyses"] == 2
        for word in ("talk", "talks", "bake", "bakes", "walked", "baking"):
            once = client.post(READ, json={"word": word, "grammar": "small"}).json()
            assert once["view"]["n_analyses"] == 1, word

    def test_three_of_the_six_compared_words_are_cut_the_same_way_by_both(
        self,
    ) -> None:
        cuts = client.get(COST).json()["budgets"][0]["cuts"]

        assert len(cuts) == 6
        assert sum(1 for cut in cuts if cut["agree"]) == 3

    def test_the_repair_to_the_small_grammar_is_four_lines(self) -> None:
        grammars = client.get(MACHINE).json()["grammars"]

        assert grammars[0]["n_written_lines"] - grammars[1]["n_written_lines"] == 4

    def test_fifty_lines_bought_more_words_than_the_hundred_and_seventy_eight_did(
        self,
    ) -> None:
        coverages = client.get(COVERAGE).json()["coverages"]
        without = coverages[1]["n_readable_tokens"]
        with_them = coverages[2]["n_readable_tokens"]

        assert without == 7
        assert with_them - without == 39
        assert coverages[2]["n_written_lines"] - coverages[1]["n_written_lines"] == 50


IDENTIFIER = re.compile(
    r"[A-Za-z]+_[A-Za-z_]+|\b[a-z]+[A-Z][A-Za-z]*\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b"
)
"""Anything shaped like a code name: an underscored word, or camel case."""

REFUSABLE: tuple[dict, ...] = (
    {"word": ""},
    {"word": "walks", "added": [{"surface": "", "kind": "a noun"}]},
    {"word": "walks", "added": [{"surface": "walk", "kind": "a noun"}]},
    {
        "word": "walks",
        "grammar": "small",
        "added": [{"surface": "s", "kind": "a prefix"}],
    },
    {
        "word": "walks",
        "grammar": "small before",
        "added": [{"surface": "smile", "kind": "a verb ending in e"}],
    },
    {"word": "a", "added": [{"surface": "a", "kind": "a verb ending in e"}]},
)
"""Every refusal a reader working the playground can reach."""


def _strings(value: object) -> list[str]:
    """Every string anywhere inside a response body."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


class TestNoIdentifierReachesThePage:
    def test_no_refusal_this_page_can_reach_carries_an_identifier(self) -> None:
        for body in REFUSABLE:
            response = client.post(READ, json=body)
            if response.status_code != 422:
                continue
            detail = response.json()["detail"]
            if not isinstance(detail, str):
                continue
            assert IDENTIFIER.search(detail) is None, detail

    def test_no_string_any_endpoint_returns_carries_an_identifier(self) -> None:
        bodies = [
            client.get(path).json()
            for path in (MACHINE, SCENARIOS, COST, AMBIGUITY, COVERAGE)
        ]
        bodies.append(client.post(READ, json={"word": "recovers"}).json())

        for body in bodies:
            for text in _strings(body):
                assert IDENTIFIER.search(text) is None, text
