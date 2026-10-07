"""Segmented KV only above the compiled-verify boundary (MTPLX_SEGMENTED_KV_MIN_TOKENS)."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import (
    configure_tail_owned_attention_kv_cache,
    restore_cache,
    snapshot_cache,
)
from mtplx.segmented_kv import (
    SegmentedKVCache,
    SegmentedKVState,
    segmented_kv_min_tokens,
)

HEADS = 2
DIM = 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")
    monkeypatch.delenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", raising=False)


def _rows(seed: int, n: int):
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed + 50_000)).astype(mx.bfloat16)
    return k, v


def _same(a, b) -> bool:
    ka, va = a
    kb, vb = b
    return bool(mx.array_equal(ka, kb).item() and mx.array_equal(va, vb).item())


def _target_cache(n_layers: int = 2) -> list:
    cache = [KVCache() for _ in range(n_layers)]
    configure_tail_owned_attention_kv_cache(cache)
    return cache


def test_boundary_follows_the_compiled_verify_settings(monkeypatch) -> None:
    assert segmented_kv_min_tokens() == 0  # compiled verify off: segments from the first row
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    assert segmented_kv_min_tokens() == 6144  # compiled verify's own default ceiling
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "32768")
    assert segmented_kv_min_tokens() == 32768
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "0")
    assert segmented_kv_min_tokens() > 10**12  # compiled at every length: never segmented
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "123")
    assert segmented_kv_min_tokens() == 123
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "0")
    assert segmented_kv_min_tokens() == 0


def test_boundary_zero_installs_segments_at_creation() -> None:
    cache = _target_cache()
    assert all(isinstance(c, SegmentedKVCache) for c in cache)


def test_below_the_boundary_the_cache_stays_stock(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    cache = _target_cache()
    assert all(type(c) is KVCache for c in cache)
    cache[0].update_and_fetch(*_rows(1, 60))
    configure_tail_owned_attention_kv_cache(cache)  # what repage runs after a prefill
    assert type(cache[0]) is KVCache


def test_a_prefill_above_the_boundary_becomes_one_sealed_segment(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    cache = _target_cache()
    control = KVCache()
    for seed, n in ((1, 70), (2, 70)):
        cache[0].update_and_fetch(*_rows(seed, n))
        control.update_and_fetch(*_rows(seed, n))
    stats_before = type(cache[0])
    configure_tail_owned_attention_kv_cache(cache)
    assert stats_before is KVCache and isinstance(cache[0], SegmentedKVCache)
    assert type(cache[1]) is KVCache  # empty layer: nothing to convert
    assert cache[0].offset == 140 and cache[0].segment_count == 1
    assert _same(cache[0].gather(), control.state)
    # growth after the conversion goes to a new tail; contents equal the stock cache
    cache[0].update_and_fetch(*_rows(3, 33))
    control.update_and_fetch(*_rows(3, 33))
    assert cache[0].segment_count == 2 and _same(cache[0].gather(), control.state)


def test_restore_picks_the_layout_by_rows(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    long_src = _target_cache(1)
    long_src[0].update_and_fetch(*_rows(1, 150))
    configure_tail_owned_attention_kv_cache(long_src)
    assert isinstance(long_src[0], SegmentedKVCache)
    long_snap = snapshot_cache(long_src)
    assert isinstance(long_snap.states[0], SegmentedKVState)

    short_src = _target_cache(1)
    short_src[0].update_and_fetch(*_rows(2, 80))
    short_snap = snapshot_cache(short_src)
    assert type(short_src[0]) is KVCache

    # segmented snapshot, long -> segments (a reference, no copy of the rows)
    dest = _target_cache(1)
    restore_cache(dest, long_snap, clone_states=False)
    assert isinstance(dest[0], SegmentedKVCache)
    assert dest[0].attention_segments()[0][0] is long_src[0].attention_segments()[0][0]
    # stock snapshot, short -> stays stock
    dest = _target_cache(1)
    restore_cache(dest, short_snap)
    assert type(dest[0]) is KVCache and dest[0].offset == 80
    # stock snapshot above the boundary -> one sealed segment
    stock_long = _target_cache(1)
    stock_long[0].update_and_fetch(*_rows(5, 130))
    dest = _target_cache(1)
    restore_cache(dest, snapshot_cache(stock_long))
    assert isinstance(dest[0], SegmentedKVCache) and dest[0].offset == 130
    assert _same(dest[0].gather(), stock_long[0].state)
    # segmented snapshot of a short conversation (boundary raised) -> stock rows
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "500")
    dest = _target_cache(1)
    restore_cache(dest, long_snap)
    assert type(dest[0]) is KVCache and dest[0].offset == 150
    assert _same(dest[0].state, long_src[0].gather())


def test_restored_segments_extend_like_a_stock_cache(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    control = KVCache()
    src = _target_cache(1)
    for seed, n in ((1, 90), (2, 90)):
        src[0].update_and_fetch(*_rows(seed, n))
        control.update_and_fetch(*_rows(seed, n))
    snap = snapshot_cache(src)  # stock, 180 rows
    dest = _target_cache(1)
    restore_cache(dest, snap)
    assert isinstance(dest[0], SegmentedKVCache)
    dest[0].update_and_fetch(*_rows(9, 17))
    control.update_and_fetch(*_rows(9, 17))
    assert _same(dest[0].gather(), control.state)
    dest[0].trim(5)
    control.trim(5)
    assert _same(dest[0].gather(), control.state)


def test_untagged_caches_are_never_converted(monkeypatch) -> None:
    # An MTP history cache is a list of stock KVCache that never went through the install.
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    mtp = [KVCache()]
    mtp[0].update_and_fetch(*_rows(1, 150))
    snap = snapshot_cache(mtp)
    dest = [KVCache()]
    restore_cache(dest, snap)
    assert type(dest[0]) is KVCache
    configure_tail_owned_attention_kv_cache(mtp)
    assert type(mtp[0]) is KVCache


def test_conversion_and_snapshot_do_not_copy_the_history(monkeypatch) -> None:
    rows = 4096
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    cache = _target_cache(1)
    cache[0].update_and_fetch(*_rows(1, rows))
    mx.eval(cache[0].keys, cache[0].values)
    history = 2 * HEADS * rows * DIM * 2
    mx.synchronize()
    before = mx.get_active_memory()
    configure_tail_owned_attention_kv_cache(cache)
    snap = snapshot_cache(cache)
    cache[0].append_rows(*_rows(2, 64))
    mx.eval(cache[0]._tail.keys, cache[0]._tail.values)
    grew = mx.get_active_memory() - before
    assert isinstance(snap.states[0], SegmentedKVState)
    assert grew < history // 4
