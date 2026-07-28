import json
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_file("model/tokenizer.json")
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(direction="right")
session = ort.InferenceSession("model/onnx/model_qint8_arm64.onnx")

input_names = [input.name for input in session.get_inputs()]

def encode(text):
    if isinstance(text, str):
        text = [text]

    enc = tokenizer.encode_batch(text)

    input_ids = np.array([e.ids for e in enc], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

    feed = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
        }

    if "token_type_ids" in input_names:
        token_type_ids = np.array([e.type_ids for e in enc], dtype=np.int64)
        feed["token_type_ids"] = token_type_ids

    output = session.run(None, feed)[0]

    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (output * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    embedding = summed / counts

    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding