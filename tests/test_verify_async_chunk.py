"""MTPLX_VERIFY_ASYNC_CHUNK_LAYERS: non-blocking submits inside the eager verify.

The switch only moves submit points (``mx.async_eval`` on the residual stream
every N layers); the verify output must stay bit-identical and prefill-sized
windows must not be touched.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen3_5

from mtplx import gdn_capture


def _tiny_model() -> qwen3_5.Model:
    text = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 64,
        "moe_intermediate_size": 32,
        "max_position_embeddings": 512,
    }
    mx.random.seed(7)
    model = qwen3_5.Model(
        qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text})
    )
    mx.eval(model.parameters())
    return model


def _run(model, monkeypatch, env: str | None, window: int) -> tuple[list, int]:
    if env is None:
        monkeypatch.delenv("MTPLX_VERIFY_ASYNC_CHUNK_LAYERS", raising=False)
    else:
        monkeypatch.setenv("MTPLX_VERIFY_ASYNC_CHUNK_LAYERS", env)
    for name in (
        "MTPLX_TARGET_LAYER_EVAL_SCHEDULE",
        "MTPLX_TARGET_LAYER_EVAL_EVERY",
        "MTPLX_TARGET_LAYER_EVAL_MAX_Q",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = {"n": 0}
    real_async_eval = mx.async_eval

    def counting_async_eval(*arrays):
        calls["n"] += 1
        return real_async_eval(*arrays)

    monkeypatch.setattr(gdn_capture.mx, "async_eval", counting_async_eval)
    cache = model.make_cache()
    prompt = mx.array([[3, 17, 42, 5, 9, 11, 64, 2, 8, 30]])
    model(prompt, cache=cache)
    mx.eval([c.state for c in cache])
    window_ids = mx.array([[21, 7, 99, 4, 55, 13, 70, 1, 6, 12][:window]])
    logits, hidden, captures = gdn_capture.forward_with_gdn_capture(
        model, window_ids, cache=cache, return_hidden=True
    )
    mx.eval(logits, hidden, captures)
    monkeypatch.setattr(gdn_capture.mx, "async_eval", real_async_eval)
    leaves = [np.array(logits), np.array(hidden)]
    for layer_idx in sorted(k for k in captures if isinstance(k, int)):
        for name in sorted(captures[layer_idx]):
            value = captures[layer_idx][name]
            if isinstance(value, mx.array):
                leaves.append(np.array(value))
    return leaves, calls["n"]


@pytest.mark.parametrize("chunk,expected_submits", [("2", 3), ("3", 2), ("8", 0)])
def test_async_chunk_is_bit_identical_and_submits_between_layers(
    monkeypatch, chunk, expected_submits
) -> None:
    model = _tiny_model()
    reference, ref_calls = _run(model, monkeypatch, None, window=3)
    candidate, calls = _run(model, monkeypatch, chunk, window=3)
    assert ref_calls == 0
    # 8 layers: submits after every `chunk` layers, never after the last one.
    assert calls == expected_submits
    assert len(reference) == len(candidate)
    for ref, cand in zip(reference, candidate):
        assert ref.dtype == cand.dtype and ref.shape == cand.shape
        assert np.array_equal(ref, cand)


def test_async_chunk_skips_prefill_sized_windows(monkeypatch) -> None:
    model = _tiny_model()
    _, calls = _run(model, monkeypatch, "2", window=10)
    assert calls == 0


@pytest.mark.parametrize(
    "raw,expected", [("", 0), ("0", 0), ("off", 0), ("x", 0), ("8", 8), ("-3", 0)]
)
def test_env_parsing(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("MTPLX_VERIFY_ASYNC_CHUNK_LAYERS", raw)
    assert gdn_capture._verify_async_chunk_layers() == expected
