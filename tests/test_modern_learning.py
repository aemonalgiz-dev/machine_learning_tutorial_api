"""Every new lesson has a finite response and bounded, explicit inputs."""

import math

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import app

client = TestClient(app)
SLUGS = [
    "transformer-blocks",
    "next-token-prediction",
    "autoregressive-generation",
    "sampling-and-temperature",
    "autoencoders",
    "variational-autoencoders",
    "generative-adversarial-networks",
    "diffusion-models",
    "supervised-fine-tuning",
    "low-rank-adaptation",
    "reinforcement-learning",
    "retrieval-augmented-generation",
    "evaluating-generative-models",
]


@pytest.mark.parametrize("slug", SLUGS)
def test_default_example_has_tables_with_finite_numeric_cells(slug):
    response = client.post(f"/concepts/modern/{slug}", json={})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["notice"] and view["tables"]
    for table in view["tables"]:
        assert table["title"] and table["columns"]
        for row in table["rows"]:
            assert len(row) == len(table["columns"])
            assert all(isinstance(cell, str) or math.isfinite(cell) for cell in row)


@pytest.mark.parametrize(
    "slug,body",
    [
        ("transformer-blocks", {"animal": "elephant"}),
        ("next-token-prediction", {"confidence": 5}),
        ("autoregressive-generation", {"max_new_tokens": 100000}),
        ("sampling-and-temperature", {"temperature": 0}),
        ("sampling-and-temperature", {"top_p": 1.1}),
        ("autoencoders", {"difference": 3}),
        ("variational-autoencoders", {"deviation": 0}),
        ("generative-adversarial-networks", {"steps": -1}),
        ("diffusion-models", {"step": 21}),
        ("supervised-fine-tuning", {"unexpected": 1}),
        ("low-rank-adaptation", {"rank": 4}),
        ("reinforcement-learning", {"episodes": 201}),
        ("reinforcement-learning", {"episodes": -1}),
        ("reinforcement-learning", {"learning_rate": 0}),
        ("reinforcement-learning", {"seed": 101}),
        ("retrieval-augmented-generation", {"query": "   "}),
        ("evaluating-generative-models", {"confidence": "nan"}),
    ],
)
def test_invalid_requests_return_validation_errors(slug, body):
    assert client.post(f"/concepts/modern/{slug}", json=body).status_code == 422


def test_response_only_loss_mask_excludes_prompt_predictions():
    view = client.post("/concepts/modern/supervised-fine-tuning", json={}).json()
    assert [row[-1] for row in view["tables"][0]["rows"]] == ["No", "No", "Yes", "Yes"]
    assert [row[-1] for row in view["tables"][2]["rows"]][:2] == [0, 0]


def test_unknown_retrieval_query_returns_an_empty_context():
    response = client.post(
        "/concepts/modern/retrieval-augmented-generation", json={"query": "elephant"}
    )
    assert response.status_code == 200
    assert response.json()["tables"][1]["rows"] == []


def test_reinforcement_zero_episodes_and_one_observed_reward():
    url = "/concepts/modern/reinforcement-learning"
    empty = client.post(url, json={"episodes": 0}).json()
    assert empty["tables"][1]["rows"] == []
    for row in empty["tables"][0]["rows"]:
        assert row[3] == row[4] == pytest.approx(1 / 3)
        assert row[5] == 0
    response = client.post(url, json={"episodes": 1, "seed": 0})
    assert response.status_code == 200
    table = response.json()["tables"]
    episode, action, reward, before, after = table[1]["rows"][0]
    assert episode == 1 and action == "Straight" and reward == 1
    assert after > before
    assert sum(row[5] for row in table[0]["rows"]) == 1
