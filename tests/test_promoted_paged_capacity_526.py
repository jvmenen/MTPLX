"""Issue #526: a promoted paged KV cache must never write past its buffers.

The compiled verify bank promotes paged KV pages to fixed-shape tensor-offset
adapters. Before this fix nothing grew them: a verify window that crossed the
capacity fell back to an eager forward on the SAME adapters, whose write is a
dynamic ``mx.slice_update`` at the array offset. MLX does not clamp that
update, so on the head-major quantized banks (1, H_kv, rows, width) the rows
past the end of head h landed on the first rows of head h + 1 (the attention
sink) and the last head's rows went past the allocation, in the payloads and
in both fp32 scale planes, while the offset ran past the capacity. Under q4/q8
KV the 27B then produced all-NaN logits.

These tests build the exact shape of that failure: 31 rows in 32 (two 16-row
blocks), then a 4-row window. On the old code the eager write is accepted
silently (no ValueError, offset 35 > capacity 32, head 1's rows 0..2
overwritten); on the fixed code the window is reserved before the forward, or
refused before any row moves.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.graphbank as graphbank_module
from mtplx.cache_state import (
    TensorOffsetQuantizedPagedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
)
from mtplx.gdn_capture import commit_captured_prefix
from mtplx.graphbank import CompiledVerifyBank, ensure_eager_window_capacity
from mtplx.kv_quant import PagedKVQuantConfig, quantize_symmetric

HEADS = 2
HEAD_DIM = 64
BLOCK = 16
BLOCKS = 2  # 32 rows
MODES = ["q8", "q4", "plain"]


def _arrays_cache_cls() -> type:
    # Resolved per use, like production (see test_graphbank_compiled_verify).
    import mlx_lm.models.cache as cache_module

    return cache_module.ArraysCache


def _config(mode: str):
    return None if mode == "plain" else PagedKVQuantConfig(mode)


def _window(rows: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    values = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    mx.eval(keys, values)
    return keys, values


def _promoted(mode: str, *, rows: int = 31):
    paged = VllmMetalPagedKVCache(
        block_size=BLOCK, num_blocks=BLOCKS, kv_quant_config=_config(mode)
    )
    paged.update_without_fetch(*_window(rows, seed=5))
    if mode == "plain":
        return TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    return TensorOffsetQuantizedPagedKVCache.from_paged_cache(paged)


def _rows(adapter, start: int, end: int) -> list[np.ndarray]:
    """Every buffer leaf's rows [start, end), head-major, as numpy."""

    if isinstance(adapter, TensorOffsetQuantizedPagedKVCache):
        return [np.array(adapter.cache[slot][:, :, start:end, :]) for slot in (0, 1, 3, 4)]
    out = []
    for slot in (0, 1):
        pages = adapter.cache[slot]
        flat = pages.reshape(-1, int(pages.shape[2]), int(pages.shape[3]))
        rows = flat[start:end].transpose(1, 0, 2)[None, ...]
        out.append(np.array(rows.astype(mx.float32)))  # exact for bf16 and fp32
    return out


def _expected_rows(adapter, keys: mx.array, values: mx.array) -> list[np.ndarray]:
    """What a correct write of (keys, values) stores, in _rows() layout."""

    if isinstance(adapter, TensorOffsetQuantizedPagedKVCache):
        q_k, s_k = quantize_symmetric(keys, bits=adapter.kv_bits)
        q_v, s_v = quantize_symmetric(values, bits=adapter.kv_bits)
        return [np.array(x) for x in (q_k, q_v, s_k, s_v)]
    return [np.array(keys.astype(mx.float32)), np.array(values.astype(mx.float32))]


def _assert_same(got: list[np.ndarray], want: list[np.ndarray], what: str) -> None:
    assert len(got) == len(want)
    for leaf, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, f"{what}: leaf {leaf} shape {g.shape} != {w.shape}"
        assert np.array_equal(g, w), f"{what}: leaf {leaf} differs"


@pytest.mark.parametrize("mode", MODES)
def test_eager_write_past_capacity_is_refused_before_any_row_moves(mode, monkeypatch):
    # Old code: "DID NOT RAISE" — the write went through, offset 35 of 32.
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    adapter = _promoted(mode)
    assert adapter.size() == 31 and adapter.capacity == 32
    before = _rows(adapter, 0, 32)

    with pytest.raises(ValueError, match="capacity exceeded"):
        adapter.update_without_fetch(*_window(4, seed=9))

    assert adapter.size() == 31
    _assert_same(_rows(adapter, 0, 32), before, "refused write")


@pytest.mark.parametrize("mode", MODES)
def test_ensure_capacity_appends_zero_blocks_and_keeps_every_row(mode, monkeypatch):
    # Old code: AttributeError — promoted paged adapters could not grow.
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)

    assert adapter.ensure_capacity(35) is True
    # The eager pages' growth policy: 1.5x the blocks, at least the need.
    assert adapter.num_blocks == 3 and adapter.capacity == 48
    _assert_same(_rows(adapter, 0, 32), before, "grown buffers, old rows")
    assert all(not leaf.any() for leaf in _rows(adapter, 32, 48)), "new rows are zero"

    keys, values = _window(4, seed=9)
    adapter.update_without_fetch(keys, values)
    assert adapter.size() == 35 <= adapter.capacity
    _assert_same(_rows(adapter, 0, 31), [x[:, :, :31, :] for x in before], "rows below the window")
    _assert_same(_rows(adapter, 31, 35), _expected_rows(adapter, keys, values), "written window")


@pytest.mark.parametrize("mode", MODES)
def test_ensure_capacity_refuses_when_dynamic_growth_is_off(mode, monkeypatch):
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)
    assert adapter.ensure_capacity(35) is False
    assert adapter.capacity == 32 and adapter.num_blocks == BLOCKS
    _assert_same(_rows(adapter, 0, 32), before, "refused growth")


@pytest.mark.parametrize("mode", MODES)
def test_eager_window_reservation_covers_promoted_paged_adapters(mode, monkeypatch):
    # The copy-block route and every bank fallback reserve through this
    # helper before their eager forward. Old code grew only dense adapters:
    # grown == 0 and the paged adapter stayed at 32 rows.
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)
    adapter = _promoted(mode)
    assert ensure_eager_window_capacity([None, adapter], 4) == 1
    assert adapter.capacity == 48
    assert ensure_eager_window_capacity([None, adapter], 4) == 0  # fits now

    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    full = _promoted(mode)
    before = _rows(full, 0, 32)
    with pytest.raises(ValueError, match="refusing before any row is written"):
        ensure_eager_window_capacity([full], 4)
    assert full.capacity == 32 and full.size() == 31
    _assert_same(_rows(full, 0, 32), before, "refused reservation")


class TwoHeadPagedRuntime:
    """A GDN-like layer plus a TWO-KV-head attention layer over paged KV.

    Same cache-mutation pattern as the graph-bank toys: fresh-array slot
    assignment in the recurrent layer, ``update_and_fetch`` on the attention
    entry and an offset-masked readout. Two KV heads make the old overflow
    visible as data (head 0's extra rows landing on head 1's first rows), not
    only as an offset past the capacity. The readout is pure MLX math, so no
    Metal attention kernel runs; the bank, promotion, eager fallback and
    in-graph quantized writes are the real code.
    """

    D = 4
    K = 3
    V = 5

    def __init__(self, mode: str, seed: int = 7) -> None:
        self.mode = mode
        mx.random.seed(seed)
        scale = 0.3
        width = HEADS * HEAD_DIM
        self.embed = mx.random.normal((self.V, self.D))
        self.w_conv = scale * mx.random.normal((self.K * self.D, self.D))
        self.w_out = scale * mx.random.normal((self.D, self.V))
        self.w_kp = 0.4 * mx.random.normal((self.D, width))
        self.w_vp = 0.4 * mx.random.normal((self.D, width))
        self.w_qp = 0.4 * mx.random.normal((self.D, width))
        self.w_ao = 0.05 * mx.random.normal((width, self.D))
        self.last_kv: tuple[mx.array, mx.array] | None = None

    def make_cache(self) -> list:
        gdn = _arrays_cache_cls()(2)
        gdn[0] = mx.zeros((1, self.K, self.D))
        gdn[1] = mx.zeros((1, 1, self.D, self.D))
        paged = VllmMetalPagedKVCache(
            block_size=BLOCK, num_blocks=BLOCKS, kv_quant_config=_config(self.mode)
        )
        return [gdn, paged]

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        del hidden_variant, capture_backend
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        gdn_entry, attn_entry = cache
        h = self.embed[input_ids]

        conv = gdn_entry.cache[0]
        state = gdn_entry.cache[1]
        conv_steps, state_steps, outs = [], [], []
        for t in range(S):
            conv = mx.concatenate([conv[:, 1:, :], h[:, t : t + 1, :]], axis=1)
            mixed = mx.tanh(conv.reshape(B, -1) @ self.w_conv)
            state = mx.tanh(state + mixed[:, None, :, None] * mixed[:, None, None, :])
            conv_steps.append(conv)
            state_steps.append(state)
            outs.append(mx.sum(state, axis=-1))
        gdn_entry[0] = conv
        gdn_entry[1] = state
        gdn_entry.advance(S)
        h = h + mx.concatenate(outs, axis=1)

        def heads(w):
            return (h @ w).reshape(B, S, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        keys, values = heads(self.w_kp), heads(self.w_vp)
        if self.mode == "plain":
            # Plain pages hold the activation dtype; the compiled paged lane
            # takes bf16/fp16 pages only (graphbank._paged_kernel_bucket_eligible).
            keys, values = keys.astype(mx.bfloat16), values.astype(mx.bfloat16)
        self.last_kv = (keys, values)
        k_buf, v_buf = attn_entry.update_and_fetch(keys, values)
        offset = attn_entry.offset  # int (stock) or mx.array (adapter)
        capacity = int(k_buf.shape[2])
        scores = heads(self.w_qp) @ mx.swapaxes(k_buf, 2, 3)  # (B, H, S, T)
        limit = offset - S + 1 + mx.arange(S)
        mask = (mx.arange(capacity)[None, :] < limit[:, None]).astype(mx.float32)
        attn = (scores * mask) @ v_buf  # (B, H, S, HEAD_DIM)
        h = h + attn.transpose(0, 2, 1, 3).reshape(B, S, -1) @ self.w_ao

        captures = {
            0: {
                "conv_states": mx.stack(conv_steps, axis=1),
                "states": mx.stack(state_steps, axis=1),
            }
        }
        if return_hidden:
            return h @ self.w_out, h, captures
        return h @ self.w_out, captures


@pytest.mark.parametrize("keep", [4, 2], ids=["accept", "trim_then_rewrite"])
@pytest.mark.parametrize("mode", MODES)
def test_capacity_overflow_round_reserves_before_mutation(mode, keep, monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)
    monkeypatch.delenv("MTPLX_GRAPHBANK_QUANTIZED_PAGED", raising=False)
    monkeypatch.setattr(graphbank_module, "_PREWARM_DONE", True)
    adapter_cls = (
        TensorOffsetVllmMetalPagedKVCache
        if mode == "plain"
        else TensorOffsetQuantizedPagedKVCache
    )
    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    rt.forward_ar_capture(mx.array([[i % 5 for i in range(27)]]), cache=cache, return_hidden=True)

    # Compiled round: promotes the pages and fills them to 31 of 32 rows.
    _l, _h, captures = bank.forward_ar_capture(mx.array([[1, 2, 3, 4]]), cache=cache)
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    assert type(adapter) is adapter_cls
    assert bank.stats["compiled_calls"] == 1
    assert adapter.size() == 31 and adapter.capacity == 32
    committed = _rows(adapter, 0, 31)

    # 31 + 4 > 32: the bank falls back eager on the promoted adapter.
    logits, _h, captures = bank.forward_ar_capture(mx.array([[4, 3, 2, 1]]), cache=cache)
    assert bank.stats["fallback_reasons"].get("capacity_overflow") == 1
    assert cache[1] is adapter
    # Old code fails here: head 0's rows 32..34 were written onto head 1's
    # rows 0..2 (and head 1's past the allocation).
    _assert_same(_rows(adapter, 0, 31), committed, "committed rows after the overflow round")
    # ... and here: offset 35 in 32 rows.
    assert adapter.size() == 35 <= adapter.capacity
    _assert_same(_rows(adapter, 31, 35), _expected_rows(adapter, *rt.last_kv), "overflow window")
    assert np.isfinite(np.array(logits)).all()
    assert bank.stats["eager_window_growths"] == 1

    # Accept all four rows, or reject two and let the next window rewrite them.
    assert commit_captured_prefix(cache, captures, keep_tokens=keep, verified_tokens=4)
    kept = 31 + keep
    assert adapter.size() == kept
    committed = _rows(adapter, 0, kept)
    _l, _h, _c = bank.forward_ar_capture(mx.array([[2, 2, 2, 2]]), cache=cache)
    # Back on the compiled route, over the grown buffers.
    assert bank.stats["compiled_calls"] == 2
    assert adapter.size() == kept + 4 <= adapter.capacity
    _assert_same(_rows(adapter, 0, kept), committed, "committed rows after the next window")


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
def test_packed_quant_kernel_refuses_an_eager_array_offset_past_its_buffers(bits):
    # The integer offset branch always bailed past the buffers; the array
    # branch (the promoted adapter's) passed any value to a walk that reads
    # offset rows of every leaf. Old code: "DID NOT RAISE".
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant

    head_dim, capacity = 256, 40
    mx.random.seed(bits)
    queries = mx.random.normal((1, 4, 2, head_dim)).astype(mx.bfloat16)
    k_q, k_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    v_q, v_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    args = dict(
        queries=queries, k_q=k_q, k_scale=k_s, v_q=v_q, v_scale=v_s,
        scale=head_dim**-0.5, bits=bits,
    )

    with pytest.raises(ValueError, match="past the KV buffers"):
        sdpa_gqa_packed_tail_quant(offset=mx.array(capacity + 3, dtype=mx.int32), **args)

    # An in-range array offset still runs and matches the integer branch.
    by_array = sdpa_gqa_packed_tail_quant(offset=mx.array(33, dtype=mx.int32), **args)
    by_int = sdpa_gqa_packed_tail_quant(offset=33, **args)
    assert by_array is not None and by_int is not None
    assert np.array_equal(
        np.array(by_array.astype(mx.float32)), np.array(by_int.astype(mx.float32))
    )
