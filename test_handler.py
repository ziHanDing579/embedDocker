"""Unit tests for the Lambda app handler.

These test the handler's request/response logic only -- event parsing,
response shape, and error handling. The embedding model is mocked (it's
covered separately in test_model), which keeps these tests fast and
independent of the model weights.

Assumptions (adjust if your layout differs):
  * The handler lives in `app.py` as `handler`.
  * `app.py` brings `encode` into its namespace, e.g.
    `from model import encode`. We patch `app.encode` -- the name as it is
    *used* inside the handler -- not wherever encode is defined.
"""

import json

import numpy as np
import pytest
from unittest.mock import patch

import app


FAKE_EMBEDDING = np.array([0.1, 0.2, 0.3], dtype=np.float32)


@pytest.fixture
def mock_encode():
    """Replace the real model with a fast stub returning a fixed vector."""
    with patch("app.encode") as m:
        m.return_value = FAKE_EMBEDDING
        yield m


def make_event(body):
    """Minimal API-Gateway-style event. `body` is the raw string Lambda
    receives, or None to simulate a missing body."""
    return {"body": body}


# --- Happy path ---------------------------------------------------------

def test_happy_path_returns_200_and_embedding(mock_encode):
    event = make_event(json.dumps({"text": "hello world"}))

    resp = app.handler(event, None)

    assert resp["statusCode"] == 200
    payload = json.loads(resp["body"])
    assert payload["embedding"] == FAKE_EMBEDDING.tolist()


def test_encode_called_with_extracted_text(mock_encode):
    event = make_event(json.dumps({"text": "hello world"}))

    app.handler(event, None)

    mock_encode.assert_called_once_with("hello world")


def test_missing_text_key_defaults_to_empty_string(mock_encode):
    event = make_event(json.dumps({"foo": "bar"}))

    resp = app.handler(event, None)

    assert resp["statusCode"] == 200
    mock_encode.assert_called_once_with("")


# --- Missing body -------------------------------------------------------

def test_missing_body_returns_400(mock_encode):
    resp = app.handler(make_event(None), None)

    assert resp["statusCode"] == 400
    assert json.loads(resp["body"]) == {"error": "missing body"}
    mock_encode.assert_not_called()


def test_body_key_absent_returns_400(mock_encode):
    resp = app.handler({}, None)  # no "body" key at all

    assert resp["statusCode"] == 400
    mock_encode.assert_not_called()


# --- Malformed input ----------------------------------------------------
# The handler catches json.JSONDecodeError and returns a 400 rather than
# letting the invocation crash.

def test_malformed_json_returns_400(mock_encode):
    resp = app.handler(make_event("{not valid json"), None)

    assert resp["statusCode"] == 400
    mock_encode.assert_not_called()


def test_empty_string_body_returns_400(mock_encode):
    # "" is not None, so it reaches the parse, which fails -> 400.
    resp = app.handler(make_event(""), None)

    assert resp["statusCode"] == 400
    mock_encode.assert_not_called()