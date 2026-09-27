"""The KV attention diagnostic line at the attention call site (issue #526).

#526 reported only "non-finite logits (nan=248320)": nothing said which KV
cache, route or offset served the attention. The split-attention hook now
emits one line per call under MTPLX_KV_ATTENTION_TRACE=1, only for
non-finite outputs under MTPLX_KV_ATTENTION_TRACE=nonfinite, and otherwise
keeps a host-only record that the server logs when the sampler reports
non-finite logits (no device read on the decode path).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import mtplx.attention_split as attention_split
from mtplx.attention_split import configure_split_full_attention, last_kv_attention_line
from mtplx.cache_state import TensorOffsetQuantizedPagedKVCache, VllmMetalPagedKVCache
from mtplx.kv_quant import PagedKVQuantConfig

HEADS, KV_HEADS, HEAD_DIM, IN_DIM = 2, 1, 64, 8


class _Proj:
    def __init__(self, out_dim: int, in_dim: int, seed: int) -> None:
        mx.random.seed(seed)
        self.weight = 0.2 * mx.random.normal((out_dim, in_dim))

    def __call__(self, x):
        return x @ self.weight.T


class _Norm:
    def __init__(self) -> None:
        self.weight = mx.ones((HEAD_DIM,))

    def __call__(self, x):
        return x


class _GatedAttention:
    """Qwen3Next-shaped gated attention the split hook accepts."""

    num_attention_heads = HEADS
    num_key_value_heads = KV_HEADS
    scale = HEAD_DIM**-0.5

    def __init__(self) -> None:
        self.q_proj = _Proj(2 * HEADS * HEAD_DIM, IN_DIM, 1)
        self.k_proj = _Proj(KV_HEADS * HEAD_DIM, IN_DIM, 2)
        self.v_proj = _Proj(KV_HEADS * HEAD_DIM, IN_DIM, 3)
        self.q_norm = _Norm()
        self.k_norm = _Norm()
        self.o_proj = lambda x: x

    def rope(self, x, offset=0):
        return x

    def __call__(self, x, mask=None, cache=None):
        raise AssertionError("split hook not installed")


class _Model:
    def __init__(self) -> None:
        layer = type("Layer", (), {"is_linear": False})()
        layer.self_attn = _GatedAttention()
        self.model = type("Inner", (), {"layers": [layer]})()


@pytest.fixture
def attn(monkeypatch):
    for name in ("MTPLX_SPLIT_FULL_ATTN", "MTPLX_SDPA_2PASS", "MTPLX_BLOCKWISE_ATTN", "MTPLX_GQA_PACKED_SDPA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setattr(attention_split, "_KV_ATTENTION_LAST", {})
    model = _Model()
    configure_split_full_attention(model)
    return model.model.layers[0].self_attn


def _q8_pages(rows: int = 5) -> VllmMetalPagedKVCache:
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4, kv_quant_config=PagedKVQuantConfig("q8"))
    mx.random.seed(11)
    cache.update_without_fetch(
        mx.random.normal((1, KV_HEADS, rows, HEAD_DIM)),
        mx.random.normal((1, KV_HEADS, rows, HEAD_DIM)),
    )
    return cache


def _x(rows: int = 2):
    mx.random.seed(13)
    return mx.random.normal((1, rows, IN_DIM))


def _lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("mtplx_kv_attention ")]


def test_trace_prints_one_line_per_call_with_every_field(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    attn(_x(), mask="causal", cache=_q8_pages())
    (line,) = _lines(capsys)
    assert line == (
        "mtplx_kv_attention layer=0 phase=unknown cache=VllmMetalPagedKVCache "
        "bits=8 route=paged_kv_quant_dequant q_dtype=float32 offset=7 capacity=64 "
        "q_len=2 mask=causal fallback=- finite=1"
    )


def test_nonfinite_mode_prints_only_the_poisoned_call(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "nonfinite")
    attn(_x(), mask="causal", cache=_q8_pages())
    assert _lines(capsys) == []

    poisoned = _q8_pages()
    scales = poisoned.value_scale_cache
    scales[0, 0] = float("nan")  # one bad fp32 row scale in the pages
    poisoned.value_scale_cache = scales
    attn(_x(), mask="causal", cache=poisoned)
    (line,) = _lines(capsys)
    assert "cache=VllmMetalPagedKVCache bits=8" in line
    assert line.endswith("finite=0")


def test_default_mode_records_host_facts_without_printing(attn, capsys, monkeypatch):
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask="causal", cache=_q8_pages())
    assert _lines(capsys) == []
    assert last_kv_attention_line() == (
        "mtplx_kv_attention layer=0 phase=unknown cache=VllmMetalPagedKVCache "
        "bits=8 route=paged_kv_quant_dequant q_dtype=float32 offset=7 capacity=64 "
        "q_len=2 mask=causal fallback=- finite=unchecked"
    )


def test_promoted_adapter_line_names_the_array_mask_decline(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_q8_pages())
    attn(_x(), mask=adapter.make_mask(2), cache=adapter)
    (line,) = _lines(capsys)
    assert "cache=TensorOffsetQuantizedPagedKVCache bits=8 route=dense_state_sdpa" in line
    assert "offset=7 capacity=64 q_len=2 mask=array_bool fallback=array_mask finite=1" in line

    # Default mode: the adapter's offset lives in an array, so the host-only
    # record says so instead of reading the device.
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask=adapter.make_mask(2), cache=adapter)
    assert "offset=array capacity=64" in last_kv_attention_line()


def test_server_non_finite_failure_logs_the_last_attention_line(attn, caplog, monkeypatch):
    import logging
    from types import SimpleNamespace

    from mtplx.sampling import NonFiniteLogitsError
    from mtplx.server.openai import _non_finite_logits_failure

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask="causal", cache=_q8_pages())
    state = SimpleNamespace(dashboard=SimpleNamespace(), sessions=None)
    error = NonFiniteLogitsError("non-finite logits in softmax", nan_count=5, inf_count=0, vocab_size=5)
    with caplog.at_level(logging.ERROR, logger="mtplx.server"):
        _non_finite_logits_failure(state, error, request_id="rid-1")
    expected = last_kv_attention_line()
    assert any(expected in record.getMessage() for record in caplog.records)
    (event,) = state.dashboard.memory_guard_events
    assert event["action"] == "non_finite_logits" and event["kv_attention"] == expected
