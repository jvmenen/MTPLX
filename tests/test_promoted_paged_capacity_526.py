"""Issue #526: a promoted paged KV cache must never write past its buffers.

The verify banks promote paged KV pages to fixed-shape tensor-offset adapters.
Before this fix nothing grew them: a window that crossed the capacity was
written by a dynamic ``mx.slice_update`` at the array offset, and MLX does not
clamp that update. On the head-major quantized banks (1, H_kv, rows, width)
the rows past the end of head h landed on the first rows of head h + 1 (its
attention sink) and the last head's rows went past the allocation, in the
payloads and in both fp32 scale planes, while the offset ran past the
capacity. Under q4/q8 KV the 27B then produced all-NaN logits.

Every write path is reserved now: an eager forward reserves in the adapter's
``make_mask`` (the mask is built once per forward from the capacity, before
any layer writes), ``SpecDecodeGraphBank`` reserves in its preflight, and
``CompiledVerifyBank`` falls back eager on a bucket that does not fit. These
tests build the failure's shape (31 or 32 rows in 32, two 16-row blocks, then
a window past the end) on each path, and compare the rounds after a growth
with a run whose buffers were large enough from the start.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.graphbank as graphbank_module
import mtplx.system_memory as system_memory
from mtplx.cache_state import (
    PagedKVGrowthRefused,
    TensorOffsetQuantizedPagedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
    _concrete_offset,
    is_compile_trace_error,
    link_paged_window_group,
    reserve_paged_window,
)
from mtplx.gdn_capture import commit_captured_prefix
from mtplx.graphbank import CompiledVerifyBank, SpecDecodeGraphBank
from mtplx.kv_quant import PagedKVQuantConfig, quantize_symmetric

HEADS = 2
HEAD_DIM = 64
BLOCK = 16
BLOCKS = 2  # 32 rows
MODES = ["q8", "q4", "plain"]


@pytest.fixture(autouse=True)
def _growth_env(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    for name in (
        "MTPLX_CONTEXT_WINDOW_TOKENS",
        "MTPLX_GRAPHBANK_QUANTIZED_PAGED",
        "MTPLX_GRAPHBANK_PRESERVE_PAGED_KV",
        "MTPLX_ALLOW_SWAP",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(graphbank_module, "_PREWARM_DONE", True)


def _arrays_cache_cls() -> type:
    # Resolved per use, like production (see test_graphbank_compiled_verify).
    import mlx_lm.models.cache as cache_module

    return cache_module.ArraysCache


def _config(mode: str):
    return None if mode == "plain" else PagedKVQuantConfig(mode)


def _adapter_cls(mode: str) -> type:
    if mode == "plain":
        return TensorOffsetVllmMetalPagedKVCache
    return TensorOffsetQuantizedPagedKVCache


def _window(rows: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    values = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    mx.eval(keys, values)
    return keys, values


def _promoted(mode: str, *, rows: int = 31, blocks: int = BLOCKS, seed: int = 5):
    paged = VllmMetalPagedKVCache(
        block_size=BLOCK, num_blocks=blocks, kv_quant_config=_config(mode)
    )
    paged.update_without_fetch(*_window(rows, seed=seed))
    return _adapter_cls(mode).from_paged_cache(paged)


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


# -- the adapters' own write path -------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_eager_write_past_capacity_is_refused_before_any_row_moves(mode, monkeypatch):
    # Old code: "DID NOT RAISE". The write went through, offset 35 of 32.
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    adapter = _promoted(mode)
    assert adapter.size() == 31 and adapter.capacity == 32
    before = _rows(adapter, 0, 32)

    with pytest.raises(ValueError, match="capacity exceeded"):
        adapter.update_without_fetch(*_window(4, seed=9))

    assert adapter.size() == 31
    _assert_same(_rows(adapter, 0, 32), before, "refused write")


@pytest.mark.parametrize("mode", MODES)
def test_ensure_capacity_appends_zero_blocks_and_keeps_every_row(mode):
    # Old code: AttributeError. Promoted paged adapters could not grow.
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
def test_reserve_paged_window_grows_every_sibling_or_refuses_before_any_row(mode, monkeypatch):
    first, second = _promoted(mode, seed=5), _promoted(mode, seed=6)
    link_paged_window_group([None, first, second])
    assert {id(m) for m in first._window_members()} == {id(first), id(second)}

    assert reserve_paged_window(first._window_members(), 4) == 2
    assert first.capacity == second.capacity == 48
    assert reserve_paged_window([first, second], 4) == 0  # fits now

    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    full = _promoted(mode)
    before = _rows(full, 0, 32)
    with pytest.raises(ValueError, match="refusing before any row is written"):
        reserve_paged_window([full], 4)
    assert full.capacity == 32 and full.size() == 31
    _assert_same(_rows(full, 0, 32), before, "refused reservation")


# -- the trace exemption -----------------------------------------------------


def test_trace_refusal_is_recognised_inside_mx_compile():
    # Pins MLX's refusal text: a reworded message must fail here, not turn
    # every traced write into an error.
    seen: dict[str, object] = {}

    def body(offset):
        try:
            offset.item()
        except ValueError as exc:
            seen["recognised"] = is_compile_trace_error(exc)
        seen["concrete"] = _concrete_offset(offset)
        return offset + 1

    mx.eval(mx.compile(body)(mx.array(3, dtype=mx.int32)))
    assert seen == {"recognised": True, "concrete": None}
    assert _concrete_offset(mx.array(7, dtype=mx.int32)) == 7


def test_concrete_offset_raises_every_other_value_error():
    # Old code: returned None, so a malformed offset read as "traced" and the
    # write went ahead unchecked.
    with pytest.raises(ValueError, match="length-1 arrays"):
        _concrete_offset(mx.array([1, 2], dtype=mx.int32))


# -- the offset read follows the bucket walk's batching rules ---------------


def test_reservation_reads_offsets_through_the_batched_offset_rules(monkeypatch):
    members = [_promoted("q8", rows=20, seed=s) for s in (1, 2, 3)]
    for member in members:
        member.cache[2] = member.cache[2] + 0  # pending, like after a trim
    calls: list[int] = []
    real_eval = mx.eval

    def counting_eval(*arrays):
        calls.append(len(arrays))
        return real_eval(*arrays)

    monkeypatch.setattr(mx, "eval", counting_eval)
    token = graphbank_module.set_paged_offsets_context_ok(True)
    try:
        assert reserve_paged_window(members, 4) == 0
        assert calls == [3], "one batched eval for every offset"

        calls.clear()
        graphbank_module.set_paged_offsets_context_ok(False)
        assert reserve_paged_window(members, 4) == 0
        assert calls == [], "past the long-context fence each offset syncs alone"

        calls.clear()
        graphbank_module.set_paged_offsets_context_ok(True)
        monkeypatch.setattr(graphbank_module, "_BATCH_PAGED_OFFSETS", False)
        assert reserve_paged_window(members, 4) == 0
        assert calls == [], "MTPLX_BATCH_PAGED_OFFSETS=0 opts out"
    finally:
        graphbank_module._PAGED_OFFSETS_CONTEXT_OK.reset(token)


# -- growth admission against the memory guard's reading --------------------


def _install_available(monkeypatch, available_bytes: int) -> None:
    total = 64 * 1024**3
    monkeypatch.setattr(
        system_memory,
        "_reader",
        lambda: system_memory.SystemMemory(
            available_bytes=int(available_bytes),
            total_bytes=total,
            level_percent=int(available_bytes * 100 // total),
        ),
    )


@pytest.mark.parametrize("mode", MODES)
def test_growth_that_would_cross_the_desktop_floor_is_refused_cleanly(mode, monkeypatch):
    # 64 GB Mac with 1 GiB left: under the 3.2 GiB shed floor already.
    _install_available(monkeypatch, 1 * 1024**3)
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)
    with pytest.raises(PagedKVGrowthRefused, match="insufficient memory") as refused:
        reserve_paged_window([adapter], 4)
    assert isinstance(refused.value, MemoryError)  # the server answers 507
    assert adapter.capacity == 32 and adapter.size() == 31
    _assert_same(_rows(adapter, 0, 32), before, "refused growth")

    # The operator's --allow-swap (env form) accepts the swap, as for prompts.
    monkeypatch.setenv("MTPLX_ALLOW_SWAP", "1")
    assert reserve_paged_window([adapter], 4) == 1
    assert adapter.capacity == 48


@pytest.mark.parametrize("old_leaves", ["released", "still_referenced"])
def test_growth_peak_memory_stays_inside_the_admitted_transient(old_leaves, monkeypatch):
    """The admission bound is a true upper bound on what a growth allocates.

    Four linked q8 adapters of 2 heads x 256 dims grow from 8192 to 12288
    rows. Each adapter is evaluated as it grows, so an old leaf nobody else
    holds is released before the next adapter grows ("released"); the bound
    covers the case where every old leaf is still referenced, by a banked
    snapshot for example ("still_referenced"), and there the peak meets it.
    """

    _install_available(monkeypatch, 60 * 1024**3)
    width = 256
    members = []
    for seed in range(4):
        paged = VllmMetalPagedKVCache(
            block_size=BLOCK, num_blocks=512, kv_quant_config=PagedKVQuantConfig("q8")
        )
        mx.random.seed(seed)
        paged.update_without_fetch(
            mx.random.normal((1, HEADS, 8190, width)).astype(mx.bfloat16),
            mx.random.normal((1, HEADS, 8190, width)).astype(mx.bfloat16),
        )
        members.append(TensorOffsetQuantizedPagedKVCache.from_paged_cache(paged))
    del paged
    link_paged_window_group(members)
    mx.eval(*[leaf for m in members for leaf in m.cache if leaf is not None])
    held = (
        [m.cache[s] for m in members for s in m._leaf_slots()]
        if old_leaves == "still_referenced"
        else []
    )
    old_bytes = sum(int(m.cache[s].nbytes) for m in members for s in m._leaf_slots())
    bound = 0
    tail = 0
    for m in members:
        leaves, zero_tail = m._growth_bytes(768)  # 8192 -> 12288 rows
        bound += leaves
        tail = max(tail, zero_tail)
    bound += tail

    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    assert reserve_paged_window(members, 4) == 4
    peak_delta = mx.get_peak_memory() - active_before
    assert all(m.capacity == 12288 for m in members)
    print(
        f"growth peak ({old_leaves}): old leaves {old_bytes} B, admitted bound "
        f"{bound} B, measured peak above the pre-growth active memory {peak_delta} B"
    )
    # The bound counts buffers; the only other allocations are the scalar
    # fill values of the zero tails (1 byte per int8 leaf, 4 per fp32 leaf).
    assert 0 < peak_delta <= bound + 64
    del held


# -- the forward paths ------------------------------------------------------


class TwoHeadPagedRuntime:
    """A GDN-like layer plus a TWO-KV-head attention layer over paged KV.

    Same cache-mutation pattern as the graph-bank toys: fresh-array slot
    assignment in the recurrent layer, ``update_and_fetch`` on the attention
    entry and a masked readout. Like every model forward it builds its
    attention mask from the cache (``create_attention_mask`` ->
    ``make_mask``) before any write, and reads with that mask when it is an
    array, so a mask narrower than the buffers fails loudly. Two KV heads
    make the old overflow visible as data (head 0's extra rows landing on
    head 1's first rows), not only as an offset past the capacity. The
    readout is pure MLX math: no Metal attention kernel runs; the banks,
    promotion, fallbacks and in-graph quantized writes are the real code.
    ``last_kv`` is the last EAGER forward's window: after a compiled call it
    holds that call's tracers, which must never be evaluated.
    """

    D = 4
    K = 3
    V = 5

    def __init__(self, mode: str, *, blocks: int = BLOCKS, seed: int = 7) -> None:
        self.mode = mode
        self.blocks = blocks
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
        # Materialized now, as a loaded model's weights are: a lazy weight
        # captured by a compiled trace is recomputed inside the graph, and the
        # traces then disagree with the eager forward by far more than kernel
        # rounding, depending on what happened to be evaluated before them.
        mx.eval(self.embed, self.w_conv, self.w_out, self.w_kp, self.w_vp, self.w_qp, self.w_ao)
        self.last_kv: tuple[mx.array, mx.array] | None = None

    def make_cache(self) -> list:
        gdn = _arrays_cache_cls()(2)
        gdn[0] = mx.zeros((1, self.K, self.D))
        gdn[1] = mx.zeros((1, 1, self.D, self.D))
        paged = VllmMetalPagedKVCache(
            block_size=BLOCK, num_blocks=self.blocks, kv_quant_config=_config(self.mode)
        )
        return [gdn, paged]

    def _forward(self, input_ids, cache):
        from mlx_lm.models.base import create_attention_mask

        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        gdn_entry, attn_entry = cache
        h = self.embed[input_ids]
        built_mask = create_attention_mask(h, attn_entry)

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
        capacity = int(k_buf.shape[2])
        if isinstance(built_mask, mx.array):
            mask = built_mask.astype(mx.float32)
        else:
            offset = attn_entry.offset  # the stock cache's int, after the write
            limit = offset - S + 1 + mx.arange(S)
            mask = (mx.arange(capacity)[None, :] < limit[:, None]).astype(mx.float32)
        scores = heads(self.w_qp) @ mx.swapaxes(k_buf, 2, 3).astype(mx.float32)
        attn = (scores * mask) @ v_buf.astype(mx.float32)  # (B, H, S, HEAD_DIM)
        h = h + attn.transpose(0, 2, 1, 3).reshape(B, S, -1) @ self.w_ao
        captures = {
            0: {
                "conv_states": mx.stack(conv_steps, axis=1),
                "states": mx.stack(state_steps, axis=1),
            }
        }
        return h @ self.w_out, h, captures

    def forward_ar(self, input_ids, cache=None, return_hidden: bool = False, hidden_variant=None):
        del hidden_variant
        logits, h, _captures = self._forward(input_ids, cache)
        return (logits, h) if return_hidden else logits

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        del hidden_variant, capture_backend
        logits, h, captures = self._forward(input_ids, cache)
        if return_hidden:
            return logits, h, captures
        return logits, captures


def _prefill(rt, cache, rows: int) -> None:
    rt.forward_ar_capture(mx.array([[i % 5 for i in range(rows)]]), cache=cache, return_hidden=True)


def _verify(bank, rt, cache, ids, *, eager: bool = False):
    if eager:
        return rt.forward_ar_capture(mx.array([ids]), cache=cache, return_hidden=True)
    return bank.forward_ar_capture(mx.array([ids]), cache=cache)


@pytest.mark.parametrize("keep", [4, 2], ids=["accept", "trim_then_rewrite"])
@pytest.mark.parametrize("mode", MODES)
def test_compiled_bank_overflow_round_reserves_before_mutation(mode, keep):
    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)

    # Compiled round: promotes the pages and fills them to 31 of 32 rows.
    _l, _h, captures = _verify(bank, rt, cache, [1, 2, 3, 4])
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    assert type(adapter) is _adapter_cls(mode)
    assert bank.stats["compiled_calls"] == 1
    assert adapter.size() == 31 and adapter.capacity == 32
    committed = _rows(adapter, 0, 31)

    # 31 + 4 > 32: the bank falls back eager, and the forward's make_mask
    # grows the adapter before the first write.
    logits, _h, captures = _verify(bank, rt, cache, [4, 3, 2, 1])
    assert bank.stats["fallback_reasons"].get("capacity_overflow") == 1
    assert cache[1] is adapter
    # 2.12.0 fails here: head 0's rows 32..34 were written onto head 1's
    # rows 0..2 (and head 1's past the allocation).
    _assert_same(_rows(adapter, 0, 31), committed, "committed rows after the overflow round")
    # ... and here: offset 35 in 32 rows.
    assert adapter.size() == 35 <= adapter.capacity == 48
    assert adapter.grow_events == 1
    _assert_same(_rows(adapter, 31, 35), _expected_rows(adapter, *rt.last_kv), "overflow window")
    assert np.isfinite(np.array(logits)).all()

    # Accept all four rows, or reject two and let the next window rewrite them.
    assert commit_captured_prefix(cache, captures, keep_tokens=keep, verified_tokens=4)
    kept = 31 + keep
    assert adapter.size() == kept
    committed = _rows(adapter, 0, kept)
    resumed, _h, _c = _verify(bank, rt, cache, [2, 2, 2, 2])
    # Back on the compiled route, over the grown buffers.
    assert bank.stats["compiled_calls"] == 2
    assert adapter.size() == kept + 4 <= adapter.capacity
    _assert_same(_rows(adapter, 0, kept), committed, "committed rows after the next window")

    # The same rounds on buffers that were 48 rows from the start, with the
    # same route per round, give the same logits and the same rows.
    ref_rt = TwoHeadPagedRuntime(mode, blocks=3)
    ref_bank = CompiledVerifyBank(ref_rt)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)
    _l, _h, c = _verify(ref_bank, ref_rt, ref, [1, 2, 3, 4])
    assert commit_captured_prefix(ref, c, keep_tokens=4, verified_tokens=4)
    ref_logits, _h, c = _verify(ref_bank, ref_rt, ref, [4, 3, 2, 1], eager=True)
    assert commit_captured_prefix(ref, c, keep_tokens=keep, verified_tokens=4)
    ref_resumed, _h, _c = _verify(ref_bank, ref_rt, ref, [2, 2, 2, 2])
    assert ref_bank.stats["compiled_calls"] == 2 and ref[1].capacity == 48
    assert np.array_equal(np.array(logits), np.array(ref_logits))
    assert np.array_equal(np.array(resumed), np.array(ref_resumed))
    _assert_same(_rows(adapter, 0, 48), _rows(ref[1], 0, 48), "grown run vs preallocated run")


@pytest.mark.parametrize("mode", MODES)
def test_several_growth_events_keep_every_row_and_logit(mode):
    """Three growths (2 -> 3 -> 5 -> 8 blocks: 32 -> 48 -> 80 -> 128 rows),
    each on an eager fallback round, against a run preallocated at 128 rows
    that takes the same route in every round."""

    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)
    ref_rt = TwoHeadPagedRuntime(mode, blocks=8)
    ref_bank = CompiledVerifyBank(ref_rt)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)

    rounds = 0
    capacities = []
    while cache[1].capacity < 100:
        ids = [(rounds + j) % 5 for j in range(4)]
        fallbacks = bank.stats["fallback_calls"]
        logits, _h, captures = _verify(bank, rt, cache, ids)
        eager = bank.stats["fallback_calls"] > fallbacks
        ref_logits, _h, ref_captures = _verify(ref_bank, ref_rt, ref, ids, eager=eager)
        assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
        assert commit_captured_prefix(ref, ref_captures, keep_tokens=4, verified_tokens=4)
        adapter = cache[1]
        rows = adapter.size()
        assert rows == ref[1].size() <= adapter.capacity
        _assert_same(_rows(adapter, 0, rows), _rows(ref[1], 0, rows), f"round {rounds} rows")
        np.testing.assert_allclose(
            np.array(logits), np.array(ref_logits), rtol=1e-6, atol=1e-6,
            err_msg=f"round {rounds} logits",
        )
        capacities.append(adapter.capacity)
        rounds += 1
    assert cache[1].grow_events == 3
    assert sorted(set(capacities)) == [32, 48, 80, 128]
    assert bank.stats["fallback_reasons"].get("capacity_overflow") == 3


@pytest.mark.parametrize("path", ["final_pending_commit", "lazy_bonus_commit"])
@pytest.mark.parametrize("mode", MODES)
def test_direct_one_token_forward_on_a_full_promoted_cache_reserves_its_row(mode, path):
    """generation.py's final pending-token commit and its lazy-bonus commit
    call ``rt.forward_ar`` with one token on the promoted cache, outside any
    bank. With the buffers exactly full, the old code raised the capacity
    error there even with growth on; the forward's own mask now reserves."""

    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 28)
    # Every row of the window accepted: the buffers are exactly full.
    _l, _h, captures = _verify(bank, rt, cache, [1, 2, 3, 4])
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    assert adapter.size() == adapter.capacity == 32
    committed = _rows(adapter, 0, 32)

    token = 3 if path == "final_pending_commit" else 4
    logits, _hidden = rt.forward_ar(mx.array([[token]]), cache=cache, return_hidden=True)
    mx.eval(logits)
    assert adapter.size() == 33 and adapter.capacity == 48
    _assert_same(_rows(adapter, 0, 32), committed, "committed rows")
    _assert_same(_rows(adapter, 32, 33), _expected_rows(adapter, *rt.last_kv), "committed token")
    assert np.isfinite(np.array(logits)).all()


@pytest.mark.parametrize("mode", MODES)
def test_direct_forward_with_growth_off_refuses_before_any_row(mode, monkeypatch):
    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 28)
    _l, _h, captures = _verify(bank, rt, cache, [1, 2, 3, 4])
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    committed = _rows(adapter, 0, 32)
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    with pytest.raises(ValueError, match="refusing before any row is written"):
        rt.forward_ar(mx.array([[3]]), cache=cache, return_hidden=True)
    assert adapter.size() == 32 and adapter.capacity == 32
    _assert_same(_rows(adapter, 0, 32), committed, "refused forward")


@pytest.mark.parametrize("mode", MODES)
def test_spec_decode_graph_bank_reserves_before_its_compiled_replay(mode, monkeypatch):
    """SpecDecodeGraphBank (graphbank selection with preserved paged KV)
    replays a traced graph that writes the adapters at their traced offset;
    no Python runs on that path. Old code: the second call replayed the
    31-row graph into 32 rows and head 0's rows landed on head 1's."""

    monkeypatch.setenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", "1")
    rt = TwoHeadPagedRuntime(mode)
    bank = SpecDecodeGraphBank(rt, max_verify_len=4)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)

    bank.forward_ar(mx.array([[1, 2, 3, 4]]), cache=cache)
    adapter = cache[1]
    assert type(adapter) is _adapter_cls(mode)
    assert bank.stats.compiled_calls == 1
    assert adapter.size() == 31 and adapter.capacity == 32
    committed = _rows(adapter, 0, 31)

    logits, _hidden = bank.forward_ar(mx.array([[4, 3, 2, 1]]), cache=cache)
    mx.eval(logits)
    assert bank.stats.compiled_calls == 2 and bank.stats.fallback_calls == 0
    _assert_same(_rows(adapter, 0, 31), committed, "committed rows after the replay")
    assert adapter.size() == 35 <= adapter.capacity == 48
    assert np.isfinite(np.array(logits)).all()

    # The same two calls on buffers that were 48 rows from the start.
    ref_rt = TwoHeadPagedRuntime(mode, blocks=3)
    ref_bank = SpecDecodeGraphBank(ref_rt, max_verify_len=4)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)
    ref_bank.forward_ar(mx.array([[1, 2, 3, 4]]), cache=ref)
    ref_logits, _hidden = ref_bank.forward_ar(mx.array([[4, 3, 2, 1]]), cache=ref)
    assert ref_bank.stats.compiled_calls == 2
    assert np.array_equal(np.array(logits), np.array(ref_logits))
    _assert_same(_rows(adapter, 0, 48), _rows(ref[1], 0, 48), "grown run vs preallocated run")


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
