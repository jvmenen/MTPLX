"""Incremental SSD encode (MTPLX_SSD_INCREMENTAL_ENCODE, opt-in).

The switch must leave the store byte-identical to a full encode: the same
payload spec, the same tensor names, the same blob digests and the same blob
bytes. Only the work changes: KV blocks (and large whole tensors) whose
content key matches the session's previous completed write are referenced
instead of evaluated, copied to the host and hashed again.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.cache_bank.codec import content_fingerprints, encode_payload
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot


SESSION = "sess-incremental"
MODEL = "models/example"


def _rand(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


class _Entry:
    def __init__(self, *, tokens, kv, gdn, mtp, session_id=SESSION):
        self.token_ids = tuple(tokens)
        self.cache_snapshot = CacheSnapshot(
            states=((kv[0], kv[1]), (gdn,)),
            meta_states=({"offset": len(tokens)}, {"kind": "gdn"}),
        )
        self.logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        self.hidden = mx.ones((1, 8), dtype=mx.bfloat16)
        self.mtp_history_snapshot = CacheSnapshot(
            states=((mtp,),), meta_states=({"offset": len(tokens)},)
        )
        self.gdn_boundaries = ()
        self.has_recurrent = True
        self.session_id = session_id
        self.model_path = MODEL
        self.mtp_enabled = True
        self.snapshot_epoch = len(tokens)
        self.mtp_snapshot_epoch = len(tokens)
        self.nbytes = sum(
            int(a.nbytes) for a in (kv[0], kv[1], gdn, mtp, self.logits, self.hidden)
        )


def _turns():
    """Turn 1: 1,024 tokens. Turn 2: the same KV bits plus 300 new tokens
    (4 full blocks shared, the 5th block and a partial 6th are new) and the
    same 1 MiB recurrent state, as after a restore from turn 1."""
    k1 = _rand((1, 2, 1024, 64), mx.bfloat16, 1)
    v1 = _rand((1, 2, 1024, 64), mx.bfloat16, 2)
    m1 = _rand((1, 1, 1024, 32), mx.bfloat16, 3)
    gdn = _rand((1, 8, 128, 256), mx.float32, 4)  # exactly 1 MiB
    turn1 = _Entry(tokens=range(1024), kv=(k1, v1), gdn=gdn, mtp=m1)
    k2 = mx.concatenate([k1, _rand((1, 2, 300, 64), mx.bfloat16, 5)], axis=2)
    v2 = mx.concatenate([v1, _rand((1, 2, 300, 64), mx.bfloat16, 6)], axis=2)
    m2 = mx.concatenate([m1, _rand((1, 1, 300, 32), mx.bfloat16, 7)], axis=2)
    turn2 = _Entry(tokens=range(1324), kv=(k2, v2), gdn=gdn, mtp=m2)
    return turn1, turn2


def _tier(path: Path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=path, mode="on", min_prefix_tokens=2, block_size=256
    )


def _payload(tier: SessionBankColdTier, entry_id: str) -> dict:
    path = tier.base_dir / "entries" / entry_id[:2] / entry_id / "payload.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("created_at_s")
    return payload


def _blobs(tier: SessionBankColdTier) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in (tier.base_dir / "blobs").rglob("*.bin")
    }


def _write(tier: SessionBankColdTier, entry) -> str:
    assert tier.put_entry(entry) is True
    assert tier.flush(timeout_s=10.0) is True
    return tier._metadata_for_entry(entry, capabilities=(), payload_nbytes=0)["entry_id"]


def _bits_equal(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def test_fingerprint_sees_every_bit():
    base = mx.zeros((1, 1, 512, 4), dtype=mx.bfloat16)
    negative_zero = mx.array(np.array([-0.0], dtype=np.float32)).astype(mx.bfloat16)
    flipped = mx.concatenate(
        [base[:, :, :300], mx.broadcast_to(negative_zero, (1, 1, 1, 4)), base[:, :, 301:]],
        axis=2,
    )
    fp_base = content_fingerprints(base, rows=256)
    fp_flip = content_fingerprints(flipped, rows=256)
    assert len(fp_base) == 2
    assert fp_base[0] == fp_flip[0]
    assert fp_base[1] != fp_flip[1]  # -0.0 == +0.0 as a value, not as bits
    assert content_fingerprints(base, rows=256) == fp_base  # deterministic
    # Swapped elements inside a block change the fingerprint (positional).
    a = mx.arange(512 * 4, dtype=mx.float32).reshape(1, 1, 512, 4)
    b = mx.concatenate([a[:, :, 1:2], a[:, :, 0:1], a[:, :, 2:]], axis=2)
    assert content_fingerprints(a, rows=256)[0] != content_fingerprints(b, rows=256)[0]


def test_switch_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MTPLX_SSD_INCREMENTAL_ENCODE", raising=False)
    tier = _tier(tmp_path / "off")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        _write(tier, turn2)
        stats = tier.stats()
        assert stats["incremental_encode"] is False
        assert stats["incremental_encodes"] == 0
        assert stats["incremental_reused_blobs"] == 0
        assert tier._block_memos == {}
    finally:
        tier.close()


def test_incremental_store_is_identical_to_full_encode(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        assert incremental.stats()["incremental_encode"] is True
        ids = []
        for entry in _turns():
            ids.append(_write(full, entry))
            assert _write(incremental, entry) == ids[-1]
        for entry_id in ids:
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
        assert _blobs(incremental) == _blobs(full)

        stats = incremental.stats()
        # Turn 2 reused 4 full blocks each of K, V and the MTP history, plus
        # the unchanged 1 MiB recurrent state.
        assert stats["incremental_encodes"] == 1
        assert stats["incremental_reused_blobs"] == 4 * 3 + 1
        assert stats["incremental_reused_bytes"] == (
            4 * (2 * 2 * 256 * 64 * 2) + 4 * (256 * 32 * 2) + 1024 * 1024
        )
        assert stats["incremental_missing_blobs"] == 0
    finally:
        full.close()
        incremental.close()


def test_changed_prefix_block_is_captured_not_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        turn1, turn2 = _turns()
        # One element of block 1 of K differs from turn 1 (e.g. a re-prefill
        # with other numerics): that block must be captured, never borrowed.
        k2 = turn2.cache_snapshot.states[0][0]
        bumped = k2[:, :, 300:301] + mx.array(1.0, dtype=mx.bfloat16)
        k2 = mx.concatenate([k2[:, :, :300], bumped, k2[:, :, 301:]], axis=2)
        turn2.cache_snapshot = CacheSnapshot(
            states=((k2, turn2.cache_snapshot.states[0][1]), turn2.cache_snapshot.states[1]),
            meta_states=turn2.cache_snapshot.meta_states,
        )
        ids = []
        for entry in (turn1, turn2):
            ids.append(_write(full, entry))
            _write(incremental, entry)
        for entry_id in ids:
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
        assert _blobs(incremental) == _blobs(full)
        assert incremental.stats()["incremental_reused_blobs"] == 3 + 4 + 4 + 1
    finally:
        full.close()
        incremental.close()


def test_restore_from_incremental_write_is_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    base = tmp_path / "store"
    tier = _tier(base)
    turn1, turn2 = _turns()
    try:
        _write(tier, turn1)
        _write(tier, turn2)
        assert tier.stats()["incremental_reused_blobs"] > 0
    finally:
        tier.close()
    # After a restart, with the switch off: the store is an ordinary store.
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    reopened = _tier(base)
    try:
        record = reopened.lookup(
            list(turn2.token_ids) + [7, 7],
            model_path=MODEL,
            mtp_enabled=True,
        )
        assert record is not None
        assert record.token_ids == turn2.token_ids
        (k, v), (gdn,) = record.cache_snapshot.states
        (k2, v2), (gdn2,) = turn2.cache_snapshot.states
        assert _bits_equal(k, k2)
        assert _bits_equal(v, v2)
        assert _bits_equal(gdn, gdn2)
        assert _bits_equal(
            record.mtp_history_snapshot.states[0][0],
            turn2.mtp_history_snapshot.states[0][0],
        )
        assert _bits_equal(record.logits, turn2.logits)
        assert _bits_equal(record.hidden, turn2.hidden)
    finally:
        reopened.close()


def test_missing_reused_blob_skips_write_then_full_encode_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    tier = _tier(tmp_path / "store")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        # A blob the memo points at disappears (its entries were evicted).
        memo = tier._block_memo(SESSION)
        victim = next(iter(memo.values()))["sha256"]
        tier._blob_path(victim).unlink()

        assert tier.put_entry(turn2) is True
        assert tier.flush(timeout_s=10.0) is True
        stats = tier.stats()
        assert stats["incremental_missing_blobs"] == 1
        assert stats["writes_completed"] == 1
        assert tier._block_memo(SESSION) == {}

        # Memo dropped: the retry encodes in full and lands.
        _write(tier, turn2)
        stats = tier.stats()
        assert stats["writes_completed"] == 2
        assert stats["incremental_encodes"] == 1
        record = tier.lookup(list(turn2.token_ids), model_path=MODEL, mtp_enabled=True)
        assert record is not None
        assert _bits_equal(
            record.cache_snapshot.states[0][0], turn2.cache_snapshot.states[0][0]
        )
    finally:
        tier.close()


def test_other_session_does_not_borrow(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    tier = _tier(tmp_path / "store")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        turn2.session_id = "another-session"
        _write(tier, turn2)
        assert tier.stats()["incremental_reused_blobs"] == 0
    finally:
        tier.close()


def test_encode_payload_without_reuse_is_unchanged():
    turn1, _ = _turns()
    kwargs = dict(
        cache_snapshot=turn1.cache_snapshot,
        logits=turn1.logits,
        hidden=turn1.hidden,
        mtp_history_snapshot=turn1.mtp_history_snapshot,
        has_recurrent=True,
        block_size=256,
    )
    legacy = encode_payload(**kwargs)
    empty_memo = encode_payload(**kwargs, reuse={})
    assert legacy.spec == empty_memo.spec
    assert legacy.tensors == empty_memo.tensors
    assert legacy.reused == {} and legacy.fingerprints == {}
    assert empty_memo.reused == {} and empty_memo.fingerprints
