# model.py
import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import otel_setup as otel

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MAX_LEN = int(os.environ.get("MAX_SEQ_LEN", "256"))
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "0") == "1"

_tokenizer = None
_session = None
_input_names = ()


def _provider_spec():
    """Providers in priority order. ORT walks this list and assigns each graph
    node to the first provider that can run it."""
    names = [
        n.strip()
        for n in os.environ.get(
            "ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
        ).split(",")
        if n.strip()
    ]
    spec = []
    for n in names:
        if n == "CUDAExecutionProvider":
            spec.append(
                (
                    n,
                    {
                        "device_id": 0,
                        # Default arena growth doubles each time it runs out.
                        # On a 4 GB card shared with the Windows desktop, ask
                        # for exactly what's needed instead.
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": int(
                            os.environ.get("GPU_MEM_LIMIT_BYTES", 1 << 30)
                        ),
                    },
                )
            )
        else:
            spec.append(n)
    return spec


def providers():
    """What the session actually got. Empty until load() runs."""
    return list(_session.get_providers()) if _session else []


def load():
    """Build the tokenizer + session and pay the first-inference cost.
    Called once at process start, never per request."""
    global _tokenizer, _session, _input_names
    if _session is not None:
        return

    # Points ORT at the CUDA/cuDNN libs that came from the [cuda,cudnn] extras.
    ort.preload_dlls()

    _tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    _tokenizer.enable_truncation(max_length=MAX_LEN)
    _tokenizer.enable_padding(direction="right")

    _session = ort.InferenceSession(
        f"{MODEL_DIR}/onnx/model.onnx", providers=_provider_spec()
    )
    _input_names = tuple(i.name for i in _session.get_inputs())

    active = _session.get_providers()
    print(f"ORT {ort.__version__} session providers: {active}", flush=True)
    if REQUIRE_GPU and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"REQUIRE_GPU=1 but session providers are {active}")

    t0 = time.perf_counter()
    encode("warmup")
    print(f"warm-up inference: {time.perf_counter() - t0:.3f}s", flush=True)


def encode(text):
    if isinstance(text, str):
        text = [text]

    enc = _tokenizer.encode_batch(text)
    input_ids = np.array([e.ids for e in enc], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in _input_names:
        feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)

    with otel.tracer.start_as_current_span("onnx.inference") as span:
        span.set_attribute("batch.size", len(enc))
        t0 = time.perf_counter()
        output = _session.run(None, feed)[0]
        otel.inference_hist.record(time.perf_counter() - t0)

    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (output * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    embedding = summed / counts
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)