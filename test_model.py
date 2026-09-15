"""Tests for the real ONNX model: shape, normalization, batching, semantics.

These need the exported model on disk (MODEL_DIR, default "model"). Run
`python export.py` first. Locally they skip if the model is missing; in CI
(where $CI is set) a missing model is a failure, so a green check can never
mean "the model tests quietly didn't run".
"""

import os

import numpy as np
import pytest

import model

TARGET_DIMS = 384
MODEL_DIR = os.environ.get("MODEL_DIR", "model")


@pytest.fixture(scope="session", autouse=True)
def loaded_model():
    """Build the session once for the whole run, not at import time."""
    missing = [
        p
        for p in (f"{MODEL_DIR}/tokenizer.json", f"{MODEL_DIR}/onnx/model.onnx")
        if not os.path.exists(p)
    ]
    if missing:
        message = f"exported model not found: {', '.join(missing)} (run export.py)"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    model.load()


def _cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def _pair_similarity(a, b):
    """Scalar similarity between two single-sentence encodings.

    encode() returns shape (1, 384), so the matrix above is (1, 1). NumPy 2
    refuses float() on anything with ndim > 0 -- only genuine 0-d arrays --
    so unwrap with .item() instead.
    """
    return _cosine_similarity(a, b).item()


def test_providers_reported_after_load():
    assert model.providers(), "session reported no execution providers"


def test_shape():
    encoded = model.encode("Hello, world!")
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (1, TARGET_DIMS), f"got {encoded.shape}"


def test_normalization():
    encoded = model.encode("Hello, world!")
    norms = np.linalg.norm(encoded, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"expected unit vectors, got {norms}"


def test_batch_encoding():
    texts = ["Hello, world!", "How are you?", "This is a test."]
    encoded = model.encode(texts)
    assert encoded.shape == (len(texts), TARGET_DIMS), f"got {encoded.shape}"


def test_batching_matches_single_encoding():
    """Padding must not change a sentence's vector -- that's what the
    attention-mask-weighted mean pool in model.encode is there for."""
    texts = ["short", "a considerably longer sentence than the first one"]
    batched = model.encode(texts)
    singles = np.vstack([model.encode(t) for t in texts])
    assert np.allclose(batched, singles, atol=1e-5)


def test_empty_input():
    encoded = model.encode("")
    assert encoded.shape == (1, TARGET_DIMS), f"got {encoded.shape}"


def test_identical_inputs():
    a = model.encode("Hello, world!")
    b = model.encode("Hello, world!")
    similarity = _cosine_similarity(a, b)
    assert np.allclose(similarity, 1.0, atol=1e-6), f"got {similarity}"


def test_similar_inputs():
    a = model.encode("The man plays the guitar.")
    b = model.encode("The man is playing a string instrument.")
    similarity = _pair_similarity(a, b)
    assert similarity > 0.7, f"expected high similarity, got {similarity}"


def test_unrelated_inputs_are_further_apart():
    """Guards the axis maths the frontend depends on: unrelated sentences
    must land lower than paraphrases, not just 'somewhere in [0,1]'."""
    anchor = model.encode("The man plays the guitar.")
    near = model.encode("The man is playing a string instrument.")
    far = model.encode("Quarterly revenue exceeded analyst expectations.")
    far_sim = _pair_similarity(anchor, far)
    near_sim = _pair_similarity(anchor, near)
    assert far_sim < near_sim, f"unrelated {far_sim:.3f} >= paraphrase {near_sim:.3f}"