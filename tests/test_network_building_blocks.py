"""HTTP examples preserve the SDK's numerical and sequence contracts."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import app

client = TestClient(app)


def worked(slug: str, payload: dict, action: str = "respond") -> dict:
    response = client.post(f"/concepts/{slug}/{action}", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_shares_are_normalized_and_obey_the_mask(heads, causal):
    body = worked("attention", {"heads": heads, "causal": causal})
    shares = np.array(body["shares"])
    assert shares.shape == (heads, 3, 3)
    assert np.allclose(shares.sum(axis=-1), 1)
    if causal:
        assert np.all(
            shares[:, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]] == 0
        )
        assert shares[0, 0, 0] == 1


def test_attention_has_a_hand_checkable_default_row():
    body = worked("attention", {})
    denominator = 2 * math.exp(0.5) + 1
    assert body["shares"][0][0] == pytest.approx(
        [math.exp(0.5) / denominator, 1 / denominator, math.exp(0.5) / denominator]
    )


def test_causal_outputs_cannot_depend_on_later_tokens():
    first = worked("attention", {"tokens": ["red", "blue", "red"], "causal": True})
    second = worked("attention", {"tokens": ["red", "green", "green"], "causal": True})
    assert first["outputs"][0] == second["outputs"][0]


def test_positions_distinguish_repeated_token_queries():
    plain = worked("attention", {"tokens": ["red", "red"]})
    positioned = worked("attention", {"tokens": ["red", "red"], "positions": True})
    assert plain["queries"][0][0] == plain["queries"][0][1]
    assert positioned["queries"][0][0] != positioned["queries"][0][1]


def test_embedding_repeats_accumulate_and_unused_rows_stay_put():
    body = worked("embedding-layers", {}, "step")
    assert body["identifiers"] == [0, 1, 0]
    assert body["gradient"][0] == pytest.approx([-1.6, -2])
    assert body["after"][0] == pytest.approx([0.36, 0.2])
    assert body["before"][2] == body["after"][2]
    assert body["loss_after"] < body["loss_before"]


def test_zero_learning_rate_keeps_the_embedding_table():
    body = worked("embedding-layers", {"learning_rate": 0}, "step")
    assert body["after"] == body["before"]
    assert body["loss_after"] == body["loss_before"]


def test_position_zero_and_one_show_the_sine_cosine_pairs():
    body = worked("positional-encoding", {})
    assert body["pattern"][0] == [0, 1, 0, 1]
    assert body["pattern"][1] == pytest.approx(
        [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)]
    )
    assert np.allclose(np.array(body["inputs"]) + body["pattern"], body["outputs"])


@pytest.mark.parametrize("kind", ["simple", "lstm", "gru"])
def test_recurrent_example_keeps_every_step_and_finite_gradients(kind):
    body = worked("recurrent-layers", {"kind": kind})
    assert body["inputs"] == [1, 0, 0, 0, 0, 0]
    assert len(body["states"]) == len(body["input_slopes"]) == 6
    assert len(body["gates"][0]) == len(body["gate_names"])
    assert np.isfinite(body["input_slopes"]).all()


def test_lstm_forget_control_changes_the_retained_cell():
    closed = worked("recurrent-layers", {"forget_bias": -2})
    open_gate = worked("recurrent-layers", {"forget_bias": 2})
    assert closed["cells"][-1] < open_gate["cells"][-1]


def test_radial_peak_and_one_width_distance():
    body = worked("radial-basis-networks", {"probe": 1, "width": 1})
    assert body["outputs"] == pytest.approx([math.exp(-2), math.exp(-0.5), 1])
    assert body["centre_slopes"][2] == body["width_slopes"][2] == 0


def test_residual_has_both_forward_and_backward_additions():
    body = worked("residual-connections", {"value": 1, "weight": 0.5})
    assert body["inner_output"] == pytest.approx(math.tanh(0.5))
    assert body["output"] == pytest.approx(1 + math.tanh(0.5))
    assert body["slope"] == pytest.approx(1 + body["inner_slope"])


@pytest.mark.parametrize(
    "slug,payload,action",
    [
        ("attention", {"heads": 3}, "respond"),
        ("attention", {"tokens": []}, "respond"),
        ("attention", {"tokens": ["red"] * 9}, "respond"),
        ("attention", {"tokens": ["unknown"]}, "respond"),
        ("embedding-layers", {"learning_rate": 2}, "step"),
        ("positional-encoding", {"steps": 1000}, "respond"),
        ("recurrent-layers", {"kind": "other"}, "respond"),
        ("recurrent-layers", {"steps": 1000}, "respond"),
        ("radial-basis-networks", {"width": 0}, "respond"),
        ("residual-connections", {"value": 1000}, "respond"),
    ],
)
def test_requests_refuse_out_of_bounds_work(slug, payload, action):
    response = client.post(f"/concepts/{slug}/{action}", json=payload)
    assert response.status_code == 422
