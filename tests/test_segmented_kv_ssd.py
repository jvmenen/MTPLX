"""SSD tier for segmented snapshots: the codec writes the stock format of the same rows."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_bank.codec import decode_payload, encode_payload
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot, restore_cache
from mtplx.segmented_kv import (
    SegmentedKVCache,
    SegmentedKVState,
    SegmentedRows,
    segmented_kv_enabled,
)

HEADS, DIM = 2, 16


def _rand(seed, n):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, HEADS, n, DIM)).astype(np.float32)).astype(mx.bfloat16)


def _segmented_and_stock(lens=(700, 311, 90), layers=2):
    """Same rows in a SegmentedKVCache with one sealed segment per turn and in a stock cache."""
    seg, stock = [], []
    for layer in range(layers):
        s, c = SegmentedKVCache(), KVCache()
        for i, n in enumerate(lens):
            k, v = _rand(10 * layer + i, n), _rand(100 + 10 * layer + i, n)
            s.update_and_fetch(k, v)
            c.update_and_fetch(k, v)
            if i < len(lens) - 1:
                s.seal()
        seg.append(s)
        stock.append(c)
    return seg, stock


def _snap(cache):
    return CacheSnapshot(
        states=tuple(c.state for c in cache) + ((mx.ones((1, 4, 8), dtype=mx.float32),),),
        meta_states=tuple(("",) for _ in cache) + (("g",),),
    )


def _encode(snapshot, reuse=None):
    return encode_payload(
        cache_snapshot=snapshot, logits=mx.zeros((1, 8)), hidden=None, mtp_history_snapshot=None,
        block_size=256, reuse=reuse,
    )


def test_segmented_snapshot_encodes_to_the_stock_spec_and_bytes() -> None:
    seg, stock = _segmented_and_stock()
    a, b = _snap(seg), _snap(stock)
    assert any(isinstance(x, SegmentedKVState) for x in a.states)
    ea, eb = _encode(a), _encode(b)
    assert json.dumps(ea.spec, sort_keys=True) == json.dumps(eb.spec, sort_keys=True)
    assert ea.tensors == eb.tensors


def test_incremental_encode_of_a_segmented_snapshot_matches_the_stock_encode() -> None:
    seg, stock = _segmented_and_stock()
    first = _encode(_snap(stock), reuse={})
    reuse = {key: {"sha256": name, "nbytes": 1} for name, key in first.fingerprints.items()}
    ea, eb = _encode(_snap(seg), reuse=dict(reuse)), _encode(_snap(stock), reuse=dict(reuse))
    assert ea.fingerprints == eb.fingerprints
    assert set(ea.reused) == set(eb.reused) and ea.reused
    assert ea.tensors == eb.tensors


def test_a_segmented_entry_written_to_disk_restores_as_the_stock_rows(tmp_path: Path) -> None:
    seg, stock = _segmented_and_stock()
    snap = _snap(seg)
    decoded = decode_payload(_encode(snap).spec, _encode(snap).tensors.__getitem__)
    for layer in range(2):
        k, v = decoded.cache_snapshot.states[layer]
        ck, cv = stock[layer].state
        assert mx.array_equal(k, ck).item() and mx.array_equal(v, cv).item()


def test_the_codec_never_gathers_the_history() -> None:
    seg, _ = _segmented_and_stock(lens=(2000, 40))
    rows = SegmentedRows(seg[0].state.refs, values=False)
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.eval(rows.rows_slice(256, 512))  # inside the first segment: a view, nothing gathered
    assert mx.get_active_memory() - base < 100_000
    straddle = rows.rows_slice(1900, 2040)
    assert straddle.shape[2] == 140


def test_rows_slice_matches_the_concatenation() -> None:
    seg, stock = _segmented_and_stock()
    rows = SegmentedRows(seg[0].state.refs, values=True)
    full = stock[0].state[1]
    for lo, hi in [(0, 1), (0, 256), (690, 710), (255, 1101), (1000, 1101), (0, 1101)]:
        assert mx.array_equal(rows[:, :, lo:hi, :], full[:, :, lo:hi, :]).item()
    assert rows.shape == tuple(full.shape) and rows.nbytes == full.nbytes


class _Entry:
    def __init__(self, cache, tokens, session="s1"):
        self.token_ids = tuple(tokens)
        self.cache_snapshot = _snap(cache)
        self.logits = mx.zeros((1, 8))
        self.hidden = mx.ones((1, 4), dtype=mx.bfloat16)
        self.mtp_history_snapshot = None
        self.gdn_boundaries = ()
        self.has_recurrent = True
        self.session_id = session
        self.model_path = "models/example"
        self.mtp_enabled = True
        self.snapshot_epoch = len(tokens)
        self.mtp_snapshot_epoch = len(tokens)
        self.nbytes = 4_000_000


def _tier(path: Path, **kw) -> SessionBankColdTier:
    return SessionBankColdTier(base_dir=path, mode="on", min_prefix_tokens=2, block_size=256, **kw)


def _blobs(tier):
    return {p.name: p.read_bytes() for p in (tier.base_dir / "blobs").rglob("*.bin")}


def _entry_json(tier, entry):
    eid = tier._metadata_for_entry(entry, capabilities=(), payload_nbytes=0)["entry_id"]
    payload = json.loads((tier.base_dir / "entries" / eid[:2] / eid / "payload.json").read_text())
    payload["metadata"].pop("created_at_s", None)
    return payload


@pytest.mark.parametrize("path", ["put", "spill"])
def test_tier_files_of_a_segmented_entry_equal_those_of_the_stock_entry(tmp_path, path) -> None:
    seg, stock = _segmented_and_stock()
    toks = range(1101)
    out = {}
    for name, cache in (("seg", seg), ("stock", stock)):
        tier = _tier(tmp_path / name)
        entry = _Entry(cache, toks)
        if path == "put":
            assert tier.put_entry(entry) and tier.flush(timeout_s=20.0)
        else:
            assert tier.spill_entry(entry)
        out[name] = (_entry_json(tier, entry), _blobs(tier))
    assert out["seg"][0] == out["stock"][0]
    assert out["seg"][1] == out["stock"][1] and out["seg"][1]


def test_forked_entries_share_blobs(tmp_path) -> None:
    """Two snapshots that share their first segments store the shared blocks once."""
    seg, _ = _segmented_and_stock(lens=(1024, 512))
    tier = _tier(tmp_path / "t")
    first = _Entry(seg, range(1536), session="a")
    assert tier.put_entry(first) and tier.flush(timeout_s=20.0)
    one = sum(len(b) for b in _blobs(tier).values())
    fork = []
    for layer in seg:
        c = SegmentedKVCache()
        c.state = SegmentedKVState([layer.state.refs[0]])  # the first segment only
        c.update_and_fetch(_rand(900, 256), _rand(901, 256))
        fork.append(c)
    second = _Entry(fork, range(1280), session="b")
    assert tier.put_entry(second) and tier.flush(timeout_s=20.0)
    two = sum(len(b) for b in _blobs(tier).values())
    assert two - one < 0.6 * one  # the 1024 shared rows (4 blocks per tensor) are not stored again


def test_flag_off_leaves_the_stock_path_untouched(monkeypatch, tmp_path) -> None:
    """With the switch off no segmented object exists; a stock entry writes the same files as before the change."""
    monkeypatch.delenv("MTPLX_SEGMENTED_KV", raising=False)
    assert not segmented_kv_enabled()
    _, stock = _segmented_and_stock()
    tier = _tier(tmp_path / "a")
    entry = _Entry(stock, range(1101))
    assert tier.put_entry(entry) and tier.flush(timeout_s=20.0)
    payload, blobs = _entry_json(tier, entry), _blobs(tier)
    ref = encode_payload(
        cache_snapshot=entry.cache_snapshot, logits=entry.logits, hidden=entry.hidden, mtp_history_snapshot=None,
        has_recurrent=True, block_size=256,
    )
    assert sorted(blobs.values()) == sorted(ref.tensors.values())
    assert json.dumps(payload["payload_spec"], sort_keys=True) == json.dumps(ref.spec, sort_keys=True)
    # and the segmented branch of the codec is never entered for stock tensors
    import mtplx.segmented_kv as module

    def boom(*a, **k):
        raise AssertionError("segmented branch taken")

    monkeypatch.setattr(module, "segmented_state_tensors", boom)
    tier2 = _tier(tmp_path / "b")
    assert tier2.put_entry(entry) and tier2.flush(timeout_s=20.0)
    assert _blobs(tier2) == blobs
    assert not module.segmented_ssd_enabled()


def test_a_stock_entry_on_disk_restores_with_the_switch_on_as_segments(monkeypatch, tmp_path) -> None:
    _, stock = _segmented_and_stock()
    tier = _tier(tmp_path / "t")
    entry = _Entry(stock, range(1101))
    assert tier.put_entry(entry) and tier.flush(timeout_s=20.0)
    eid = tier._metadata_for_entry(entry, capabilities=(), payload_nbytes=0)["entry_id"]
    payload = json.loads((tier.base_dir / "entries" / eid[:2] / eid / "payload.json").read_text())
    names = payload["tensor_blobs"]
    blobs = {n: (tier.base_dir / "blobs" / b["sha256"][:2] / (b["sha256"] + ".bin")).read_bytes() for n, b in names.items()} \
        if (tier.base_dir / "blobs" / next(iter(names.values()))["sha256"][:2]).exists() else None
    decoded = decode_payload(payload["payload_spec"], (lambda n: blobs[n]) if blobs else None)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MIN_TOKENS", "100")
    from mtplx.cache_state import configure_tail_owned_attention_kv_cache
    cache = [KVCache(), KVCache(), KVCache()]
    cache = cache[:2]
    configure_tail_owned_attention_kv_cache(cache)
    snap = CacheSnapshot(states=decoded.cache_snapshot.states[:2], meta_states=decoded.cache_snapshot.meta_states[:2])
    restore_cache(cache, snap)
    assert all(isinstance(c, SegmentedKVCache) and c.offset == 1101 for c in cache)
    assert mx.array_equal(cache[0].gather()[0], stock[0].state[0]).item()
