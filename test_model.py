from app import encode
import pytest
import numpy as np

target_dims = 384

def _cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def test_shape():
    input_data = "Hello, world!"
    encoded_data = encode(input_data)
    assert isinstance(encoded_data, list) or isinstance(encoded_data, np.ndarray)
    assert encoded_data.shape[1] == target_dims, f"Expected shape (N, {target_dims}), but got {encoded_data.shape}"

def test_normalization():
    input_data = "Hello, world!"
    encoded_data = encode(input_data)
    norms = np.linalg.norm(encoded_data, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"Expected normalized embeddings, but got norms: {norms}"

def test_batch_encoding():
    input_data = ["Hello, world!", "How are you?", "This is a test."]
    encoded_data = encode(input_data)
    assert encoded_data.shape[0] == len(input_data), f"Expected {len(input_data)} embeddings, but got {encoded_data.shape[0]}"
    assert encoded_data.shape[1] == target_dims, f"Expected shape (N, {target_dims}), but got {encoded_data.shape}"

def test_empty_input():
    input_data = ""
    encoded_data = encode(input_data)
    assert encoded_data.shape[0] == 1, f"Expected 1 embedding for empty input, but got {encoded_data.shape[0]}"
    assert encoded_data.shape[1] == target_dims, f"Expected shape (1, {target_dims}), but got {encoded_data.shape}"

def test_identical_inputs():
    input_data1 = "Hello, world!"
    input_data2 = "Hello, world!"
    encoded_data1 = encode(input_data1)
    encoded_data2 = encode(input_data2)
    similarity = _cosine_similarity(encoded_data1, encoded_data2)
    assert np.allclose(similarity, 1.0, atol=1e-6), f"Expected similarity of 1.0 for identical inputs, but got {similarity}"

def test_similar_inputs():
    input_data1 = "The man plays the guitar."
    input_data2 = "The man is playing a string instrument."
    encoded_data1 = encode(input_data1)
    encoded_data2 = encode(input_data2)
    similarity = _cosine_similarity(encoded_data1, encoded_data2)
    assert similarity > 0.7, f"Expected high similarity for similar inputs, but got {similarity}"