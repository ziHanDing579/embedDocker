"""Tests for the FastAPI app in main.py.

These cover request/response logic only -- routing, response shape, the
readiness gate, and error handling. The model is mocked (it's covered in
test_model.py), so these run in milliseconds and need no model weights.

main.py does `import model`, so patching `main.model.encode` patches the
name as the app actually uses it.
"""

import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import main

FAKE_EMBEDDING = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


@pytest.fixture
def mocked():
    """App with a stubbed model, started through the real lifespan."""
    with (
        patch.object(main.model, "load") as load,
        patch.object(main.model, "encode", return_value=FAKE_EMBEDDING) as encode,
        patch.object(main.model, "providers", return_value=["CPUExecutionProvider"]),
    ):
        with TestClient(main.app) as client:
            yield client, load, encode


# --- Probes -------------------------------------------------------------

def test_healthz_is_ok(mocked):
    client, _, _ = mocked
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_after_startup(mocked):
    client, load, _ = mocked
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    load.assert_called_once()


def test_readyz_is_503_before_startup():
    """Constructing TestClient without entering it skips the lifespan, which
    is the same state as a pod that hasn't finished loading the model."""
    client = TestClient(main.app)
    assert client.get("/readyz").status_code == 503


# --- Happy path ---------------------------------------------------------

def test_single_text_returns_200_and_embedding(mocked):
    client, _, encode = mocked
    resp = client.post("/embed", json={"text": "hello world"})

    assert resp.status_code == 200
    assert resp.json()["embedding"] == FAKE_EMBEDDING.tolist()
    encode.assert_called_once_with("hello world")


def test_list_of_texts_is_passed_through(mocked):
    client, _, encode = mocked
    texts = ["hello world", "how are you"]
    resp = client.post("/embed", json={"text": texts})

    assert resp.status_code == 200
    encode.assert_called_once_with(texts)


def test_missing_text_key_defaults_to_empty_string(mocked):
    client, _, encode = mocked
    resp = client.post("/embed", json={"foo": "bar"})

    assert resp.status_code == 200
    encode.assert_called_once_with("")


# --- Bad input ----------------------------------------------------------
# The RequestValidationError handler turns FastAPI's default 422 into the
# 400 the frontend and the Grafana dashboard expect.

@pytest.mark.parametrize(
    "body",
    [
        {"text": 123},
        {"text": {"nested": "object"}},
        {"text": [1, 2, 3]},
    ],
)
def test_wrong_text_type_returns_400(mocked, body):
    client, _, encode = mocked
    resp = client.post("/embed", json=body)

    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid request body"}
    encode.assert_not_called()


def test_malformed_json_returns_400(mocked):
    client, _, encode = mocked
    resp = client.post(
        "/embed",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400
    encode.assert_not_called()


def test_missing_body_returns_400(mocked):
    client, _, encode = mocked
    resp = client.post("/embed")

    assert resp.status_code == 400
    encode.assert_not_called()


# --- Failure inside the model ------------------------------------------

def test_model_failure_is_not_swallowed(mocked):
    """A model blow-up must surface as a 5xx, not a 200 with junk."""
    client, _, encode = mocked
    encode.side_effect = RuntimeError("session died")

    with pytest.raises(RuntimeError):
        client.post("/embed", json={"text": "hello"})