"""The paragraph-vectors page's endpoints, on the collection the page carries.

Every number the page prints is pinned here. The expensive endpoints are cached
in the module under test, so the first test that reaches one pays for it and the
rest are free.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.concepts.paragraph_vectors import (
    COOKING_WORDS,
    HELD_OUT_COOKING,
    SAILING_WORDS,
    TWO_TOPICS,
    _in_plain_words,
)
from oop_ml_api.main import create_app

client = TestClient(create_app())

FITS = "/concepts/paragraph-vectors/fits"
INFER = "/concepts/paragraph-vectors/infer"
CURVE = "/concepts/paragraph-vectors/inference-curve"
AVERAGING = "/concepts/paragraph-vectors/against-averaging"
ORDER = "/concepts/paragraph-vectors/order"
ONE_POSITION = "/concepts/paragraph-vectors/one-position"

#: Anything that reads as a class name, a parameter name or a dotted path.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z]*")


def architecture(body: dict, name: str) -> dict:
    return next(
        entry for entry in body["architectures"] if entry["architecture"] == name
    )


class TestTheCollection:
    def test_it_is_the_collection_the_spec_fits(self) -> None:
        """Reproduced rather than imported, so a test holds the two together.

        ``test`` is not part of the installed package, so this is skipped
        wherever the library's own suite is not on the path.
        """
        spec = pytest.importorskip(
            "test.core.natural_language_processing.embeddings.prediction."
            "test_paragraph_vectors"
        )

        assert TWO_TOPICS == spec.TWO_TOPICS
        assert COOKING_WORDS == spec.COOKING_WORDS
        assert SAILING_WORDS == spec.SAILING_WORDS
        assert HELD_OUT_COOKING == spec.HELD_OUT_COOKING

    def test_thirty_documents_of_eight_words_over_twenty_words(self) -> None:
        body = client.get(FITS).json()

        assert body["n_documents"] == 30
        assert body["n_words"] == 20
        assert body["n_occurrences"] == 240
        assert len(body["documents"]) == 30
        assert [entry["topic"] for entry in body["documents"][:15]] == ["cooking"] * 15
        assert [entry["topic"] for entry in body["documents"][15:]] == ["sailing"] * 15

    def test_the_two_lists_share_no_word(self) -> None:
        assert not set(COOKING_WORDS) & set(SAILING_WORDS)


class TestTheTwoFits:
    def test_distributed_memory_separates_the_two_topics(self) -> None:
        summary = architecture(client.get(FITS).json(), "distributed-memory")["summary"]

        assert summary["cooking"] == pytest.approx(0.8974, abs=5e-4)
        assert summary["sailing"] == pytest.approx(0.8696, abs=5e-4)
        assert summary["across"] == pytest.approx(0.2811, abs=5e-4)

    def test_the_bag_of_words_separates_them_too(self) -> None:
        summary = architecture(client.get(FITS).json(), "bag-of-words")["summary"]

        assert summary["cooking"] == pytest.approx(0.7763, abs=5e-4)
        assert summary["sailing"] == pytest.approx(0.8078, abs=5e-4)
        assert summary["across"] == pytest.approx(0.1445, abs=5e-4)

    def test_the_across_figure_is_positive_under_both(self) -> None:
        """Every document is trained against one output table, so they share a
        component, which is why the page's claim is a margin and not a sign."""
        body = client.get(FITS).json()

        for entry in body["architectures"]:
            assert entry["summary"]["across"] > 0.0

    def test_the_fourteen_nearest_are_the_other_cooking_documents(self) -> None:
        body = client.get(FITS).json()

        assert all(
            entry["fourteen_nearest_all_cooking"] for entry in body["architectures"]
        )

    def test_the_objective_falls_from_just_under_the_untrained_cost(self) -> None:
        body = client.get(FITS).json()
        memory = architecture(body, "distributed-memory")
        losses = [entry["mean_loss"] for entry in memory["epochs"]]

        assert body["untrained_cost"] == pytest.approx(4.1589, abs=5e-5)
        assert losses[0] == pytest.approx(4.1529, abs=5e-4)
        assert losses[-1] == pytest.approx(2.0891, abs=5e-4)
        assert len(losses) == 30

    def test_the_bag_of_words_falls_further(self) -> None:
        losses = [
            entry["mean_loss"]
            for entry in architecture(client.get(FITS).json(), "bag-of-words")["epochs"]
        ]

        assert losses[0] == pytest.approx(4.1538, abs=5e-4)
        assert losses[-1] == pytest.approx(1.8720, abs=5e-4)

    def test_the_word_rows_are_long_and_the_document_rows_short(self) -> None:
        """The mechanism behind everything Part 4 measures."""
        memory = architecture(client.get(FITS).json(), "distributed-memory")

        assert memory["word_lengths"]["smallest"] == pytest.approx(1.7429, abs=5e-4)
        assert memory["word_lengths"]["largest"] == pytest.approx(2.6753, abs=5e-4)
        assert memory["document_lengths"]["smallest"] == pytest.approx(0.2909, abs=5e-4)
        assert memory["document_lengths"]["largest"] == pytest.approx(0.5249, abs=5e-4)

    def test_the_bag_of_words_lengths_run_the_other_way(self) -> None:
        bag = architecture(client.get(FITS).json(), "bag-of-words")

        assert bag["word_lengths"]["largest"] < 0.15
        assert bag["document_lengths"]["smallest"] > 1.9

    def test_only_one_architecture_moves_the_word_rows(self) -> None:
        body = client.get(FITS).json()

        assert architecture(body, "distributed-memory")["word_vectors_moved"] is True
        assert architecture(body, "bag-of-words")["word_vectors_moved"] is False

    def test_the_word_pairs_say_the_same_thing(self) -> None:
        body = client.get(FITS).json()
        memory = architecture(body, "distributed-memory")
        bag = architecture(body, "bag-of-words")

        assert memory["near_pair"] == pytest.approx(0.994, abs=5e-4)
        assert memory["far_pair"] == pytest.approx(0.120, abs=5e-4)
        assert bag["near_pair"] == pytest.approx(0.4206, abs=5e-4)
        assert bag["far_pair"] == pytest.approx(-0.2104, abs=5e-4)
        assert body["bag_of_words_with_word_pairs_near"] == pytest.approx(
            0.9666, abs=5e-4
        )
        assert body["bag_of_words_with_word_pairs_far"] == pytest.approx(
            0.0965, abs=5e-4
        )

    def test_the_similarity_table_is_square_and_one_on_its_diagonal(self) -> None:
        memory = architecture(client.get(FITS).json(), "distributed-memory")
        table = memory["similarities"]

        assert len(table) == 30
        assert all(len(row) == 30 for row in table)
        assert all(table[position][position] == 1.0 for position in range(30))
        assert table[0][1] == pytest.approx(table[1][0], abs=1e-12)

    def test_the_tables_are_of_the_sizes_the_page_quotes(self) -> None:
        memory = architecture(client.get(FITS).json(), "distributed-memory")

        assert memory["word_numbers"] == 160
        assert memory["document_numbers"] == 240
        assert memory["output_numbers"] == 160

    def test_the_published_settings_separate_nothing(self) -> None:
        defaults = client.get(FITS).json()["reference_defaults"]

        assert defaults["positions"] == 2400
        assert defaults["first_loss"] == pytest.approx(4.1586, abs=5e-4)
        assert defaults["last_loss"] == pytest.approx(4.1155, abs=5e-4)
        assert defaults["summary"]["within"] == pytest.approx(0.0330, abs=5e-4)
        assert defaults["summary"]["across"] == pytest.approx(0.0437, abs=5e-4)
        assert defaults["summary"]["within"] < defaults["summary"]["across"]


class TestInference:
    def test_the_published_pass_count_lands_in_the_wrong_topic(self) -> None:
        body = client.post(
            INFER,
            json={
                "architecture": "distributed-memory",
                "text": HELD_OUT_COOKING,
                "passes": 20,
            },
        ).json()

        assert body["scores"]["cooking"] == pytest.approx(0.4786, abs=5e-4)
        assert body["scores"]["sailing"] == pytest.approx(0.7608, abs=5e-4)
        assert body["scores"]["leans_towards"] == "sailing"
        assert all(entry["topic"] == "sailing" for entry in body["nearest"])

    def test_a_hundred_passes_arrive(self) -> None:
        body = client.post(
            INFER,
            json={
                "architecture": "distributed-memory",
                "text": HELD_OUT_COOKING,
                "passes": 100,
            },
        ).json()

        assert body["scores"]["cooking"] == pytest.approx(0.9031, abs=5e-4)
        assert body["scores"]["sailing"] == pytest.approx(0.2116, abs=5e-4)
        assert all(entry["topic"] == "cooking" for entry in body["nearest"])

    def test_the_bag_of_words_arrives_in_one_pass(self) -> None:
        body = client.post(
            INFER,
            json={
                "architecture": "bag-of-words",
                "text": HELD_OUT_COOKING,
                "passes": 1,
            },
        ).json()

        assert body["scores"]["cooking"] == pytest.approx(0.8330, abs=5e-4)
        assert body["scores"]["sailing"] == pytest.approx(0.4047, abs=5e-4)
        assert body["scores"]["leans_towards"] == "cooking"

    def test_two_starting_draws_give_two_answers(self) -> None:
        first = client.post(
            INFER, json={"text": HELD_OUT_COOKING, "passes": 20, "random_seed": 1}
        ).json()
        second = client.post(
            INFER, json={"text": HELD_OUT_COOKING, "passes": 20, "random_seed": 2}
        ).json()

        assert first["scores"]["cooking"] != second["scores"]["cooking"]

    def test_the_same_draw_gives_the_same_answer(self) -> None:
        first = client.post(
            INFER, json={"text": HELD_OUT_COOKING, "passes": 20, "random_seed": 4}
        ).json()
        second = client.post(
            INFER, json={"text": HELD_OUT_COOKING, "passes": 20, "random_seed": 4}
        ).json()

        assert first["scores"] == second["scores"]
        assert first["length"] == second["length"]

    def test_a_word_the_collection_never_used_contributes_nothing(self) -> None:
        body = client.post(
            INFER, json={"text": "butter garlic quantum lattice", "passes": 100}
        ).json()

        assert body["known_words"] == ["butter", "garlic"]
        assert body["unknown_words"] == ["quantum", "lattice"]
        assert body["nothing_to_descend_on"] is False
        assert body["scores"]["leans_towards"] == "cooking"

    def test_a_text_of_nothing_known_keeps_its_starting_draw(self) -> None:
        body = client.post(
            INFER, json={"text": "quantum lattice photon", "passes": 200}
        ).json()

        assert body["known_words"] == []
        assert body["nothing_to_descend_on"] is True
        assert body["length"] == pytest.approx(0.11516, abs=5e-5)

    def test_the_length_grows_with_the_passes(self) -> None:
        short = client.post(INFER, json={"passes": 20}).json()["length"]
        long = client.post(INFER, json={"passes": 200}).json()["length"]

        assert long > short

    def test_more_passes_than_the_ceiling_are_refused(self) -> None:
        response = client.post(INFER, json={"passes": 4000})

        assert response.status_code == 422

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(INFER, json={"pases": 20})

        assert response.status_code == 422


class TestTheInferenceCurve:
    def test_distributed_memory_crosses_over_between_twenty_and_fifty(self) -> None:
        series = next(
            entry
            for entry in client.get(CURVE).json()["series"]
            if entry["architecture"] == "distributed-memory"
        )
        by_passes = {point["passes"]: point for point in series["points"]}

        assert by_passes[20]["leans_towards"] == "sailing"
        assert by_passes[50]["leans_towards"] == "cooking"
        assert by_passes[50]["cooking"] == pytest.approx(0.8347, abs=5e-4)
        assert by_passes[200]["cooking"] == pytest.approx(0.9081, abs=5e-4)

    def test_the_bag_of_words_leans_the_right_way_at_every_count(self) -> None:
        series = next(
            entry
            for entry in client.get(CURVE).json()["series"]
            if entry["architecture"] == "bag-of-words"
        )

        assert all(point["leans_towards"] == "cooking" for point in series["points"])

    def test_the_bag_of_words_drifts_away_after_twenty(self) -> None:
        series = next(
            entry
            for entry in client.get(CURVE).json()["series"]
            if entry["architecture"] == "bag-of-words"
        )
        by_passes = {point["passes"]: point for point in series["points"]}

        assert by_passes[20]["cooking"] == pytest.approx(0.8565, abs=5e-4)
        assert by_passes[200]["cooking"] == pytest.approx(0.7665, abs=5e-4)
        assert by_passes[200]["cooking"] < by_passes[20]["cooking"]

    def test_the_copy_really_takes_the_new_pass_count(self) -> None:
        """The one thing a silently ignored update would break."""
        series = client.get(CURVE).json()["series"][0]
        first, last = series["points"][0], series["points"][-1]

        assert first["length"] != last["length"]
        assert first["cooking"] != last["cooking"]

    def test_the_published_count_is_reported_as_twenty(self) -> None:
        assert client.get(CURVE).json()["published_passes"] == 20


class TestAgainstAveraging:
    def test_averaging_separates_the_topics_better_than_either_fit(self) -> None:
        rules = {
            entry["rule"]: entry for entry in client.get(AVERAGING).json()["rules"]
        }
        averaged = rules["the average of a document's word positions"]

        assert averaged["summary"]["worst_within"] == pytest.approx(0.9935, abs=5e-4)
        assert averaged["summary"]["best_across"] == pytest.approx(0.1552, abs=5e-4)
        assert averaged["summary"]["margin"] == pytest.approx(0.8383, abs=5e-4)
        for entry in rules.values():
            assert entry["summary"]["margin"] <= averaged["summary"]["margin"]

    def test_both_learned_rules_have_a_negative_margin(self) -> None:
        rules = client.get(AVERAGING).json()["rules"]
        learned = [
            entry
            for entry in rules
            if entry["rule"] != "the average of a document's word positions"
        ]

        assert len(learned) == 2
        for entry in learned:
            assert entry["summary"]["margin"] < 0.0

    def test_averaging_costs_a_thousandth_of_what_a_descent_costs(self) -> None:
        rules = {
            entry["rule"]: entry for entry in client.get(AVERAGING).json()["rules"]
        }
        averaged = rules["the average of a document's word positions"]
        memory = rules["distributed memory, 200 passes"]

        assert averaged["seconds_per_text"] * 100 < memory["seconds_per_text"]

    def test_every_rule_answers_the_same_width(self) -> None:
        for entry in client.get(AVERAGING).json()["rules"]:
            assert entry["numbers_per_document"] == 8


class TestOrder:
    def test_averaging_gives_two_orders_the_identical_direction(self) -> None:
        rows = {
            (entry["rule"], entry["passes"]): entry
            for entry in client.get(ORDER).json()["rows"]
        }
        averaged = rows[("the average of a document's word positions", 0)]

        assert averaged["two_orders"] == pytest.approx(1.0, abs=1e-12)
        assert averaged["reads_order"] is False

    def test_distributed_memory_reads_order_once_it_has_walked_far_enough(
        self,
    ) -> None:
        rows = {
            (entry["rule"], entry["passes"]): entry
            for entry in client.get(ORDER).json()["rows"]
        }
        long_walk = rows[("distributed memory", 200)]
        short_walk = rows[("distributed memory", 20)]

        assert long_walk["two_orders"] == pytest.approx(0.8548, abs=5e-4)
        assert long_walk["two_starts_first"] == pytest.approx(0.9823, abs=5e-4)
        assert long_walk["reads_order"] is True
        assert short_walk["reads_order"] is False

    def test_the_bag_of_words_reads_no_order_at_either_length(self) -> None:
        rows = [
            entry
            for entry in client.get(ORDER).json()["rows"]
            if entry["rule"] == "distributed bag of words"
        ]

        assert len(rows) == 2
        for entry in rows:
            assert entry["two_orders"] > 0.999
            assert entry["reads_order"] is False

    def test_the_two_texts_hold_the_same_words(self) -> None:
        body = client.get(ORDER).json()

        assert sorted(body["text"].split()) == sorted(body["shuffled"].split())
        assert body["text"] != body["shuffled"]


class TestOnePosition:
    def test_the_step_is_the_one_the_page_works_by_hand(self) -> None:
        body = client.get(ONE_POSITION).json()

        assert body["hidden"] == pytest.approx([0.1, 0.1, 0.1], abs=1e-12)
        assert body["probability"] == pytest.approx(0.524979, abs=5e-7)
        assert body["hidden_gradient"] == pytest.approx(
            [-0.475021, 0.524979, 0.0], abs=5e-7
        )
        assert body["step"] == pytest.approx([0.005938, -0.006562, 0.0], abs=5e-7)

    def test_the_step_is_the_hidden_gradient_shared_between_two_rows(self) -> None:
        body = client.get(ONE_POSITION).json()

        for gradient, step in zip(body["hidden_gradient"], body["step"], strict=True):
            assert step == pytest.approx(
                -body["learning_rate"] * gradient / body["n_averaged"], abs=1e-12
            )


class TestNothingNamesTheCodeBehindThePage:
    def test_a_rewritten_refusal_carries_no_identifier(self) -> None:
        from oop_ml.core.exceptions import InvalidValuesError

        sentences = [
            _in_plain_words(
                InvalidValuesError(
                    "a mean is over at least one vector, got n_averaged=0"
                )
            ),
            _in_plain_words(InvalidValuesError("n_results must be at least 1, got 0")),
        ]

        for sentence in sentences:
            assert not IDENTIFIER.search(sentence), sentence

    def test_no_string_a_widget_renders_names_a_class_or_a_parameter(self) -> None:
        strings: list[str] = []
        fits = client.get(FITS).json()
        strings.extend(entry["label"] for entry in fits["architectures"])
        strings.extend(entry["topic"] for entry in fits["documents"])
        strings.extend(entry["text"] for entry in fits["documents"])
        strings.extend(entry["rule"] for entry in client.get(AVERAGING).json()["rules"])
        strings.extend(entry["rule"] for entry in client.get(ORDER).json()["rows"])
        inferred = client.post(INFER, json={"passes": 20}).json()
        strings.append(inferred["label"])
        strings.append(inferred["scores"]["leans_towards"])
        strings.extend(entry["text"] for entry in inferred["nearest"])
        strings.extend(entry["topic"] for entry in inferred["nearest"])

        for sentence in strings:
            assert not IDENTIFIER.search(sentence), sentence
            for forbidden in ("oop_ml", "the library", "the API"):
                assert forbidden not in sentence
