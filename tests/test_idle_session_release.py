"""Releasing idle conversations before a prompt is refused.

The 2026-09-26 field report (M5 Max 128 GB, Flash-Next, the pi coding agent):
pi's compaction request arrives as a new anonymous session while the
114k-token conversation it summarizes waits for the answer. The admission
shed could not reach that conversation: the superseded clear targets the
incoming session id, the LRU pass protects sessions touched in the last 600 s,
and the chain walk never evicts a snapshot that also holds a live cache. 13
refusals in a row, and only a restart cleared them.

These tests pin the release step on the real SessionBank and
EngineSessionManager (synthetic entries, no MLX arrays): which sessions and
entries it takes, in what order, and what it never touches.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

from mtplx.engine_session import EngineSessionManager
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.session_bank import SessionBank, cold_persistence_key

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=32, max_bytes=10_000, per_session_max_bytes=10_000)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _put(bank: SessionBank, tokens, *, session_id, nbytes, live_cache=False, on_ssd=False):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=nbytes,
    )
    assert entry is not None
    if live_cache:
        # A generation-final commit of a coding-agent turn keeps the live
        # cache next to its snapshot (keep_live_ref).
        entry.cache_ref = object()
    if on_ssd:
        entry.cold_encode_completed_at = time.monotonic()
    return entry


def _keys(bank: SessionBank):
    return set(bank._entries)


def _conversation(bank: SessionBank, session_id="anon-conv"):
    """Julian's shape: generation-final and postcommit siblings that diverge
    at the assistant turn, each holding a live cache, plus an older sibling
    from an earlier turn (retained: not a strict prefix of either)."""

    older = _put(bank, (1, 2, 9), session_id=session_id, nbytes=300)
    final = _put(bank, (1, 2, 3, 4, 5, 60), session_id=session_id, nbytes=400, live_cache=True)
    post = _put(bank, (1, 2, 3, 4, 5, 70), session_id=session_id, nbytes=400, live_cache=True)
    return older, final, post


class TestBankRelease:
    def test_an_idle_conversation_is_released_whole(self):
        bank = _bank()
        _conversation(bank)
        receipt = bank.release_sessions(
            None, keep_session_ids={"anon-compaction"}, protect_tokens=(9, 9, 9)
        )
        assert _keys(bank) == set()
        assert receipt["entries"] == 3
        assert receipt["held_bytes"] == 1_100
        [row] = receipt["sessions"]
        assert row["session_id"] == "anon-conv"
        assert row["live_cache_refs"] == 2
        assert row["dropped_entries"] == 3
        # Every eviction dropped its live reference (release_live_refs).
        assert all(
            record["reason"] == "idle_session_release"
            for record in list(bank.eviction_log)[-3:]
        )

    def test_the_chain_walk_alone_never_reaches_it(self):
        """The defect this step exists for: nothing in the old escalation
        takes a snapshot that also holds a live cache, so the walk stops
        with every byte of the conversation still resident."""

        bank = _bank()
        _conversation(bank)
        bank.shrink_for_admission(0, protect_tokens=(9, 9, 9))
        assert {(1, 2, 3, 4, 5, 60), (1, 2, 3, 4, 5, 70)} <= _keys(bank)

    def test_kept_sessions_are_never_touched(self):
        bank = _bank()
        _conversation(bank)
        receipt = bank.release_sessions(None, keep_session_ids={"anon-conv"})
        assert receipt["entries"] == 0
        assert len(bank._entries) == 3

    def test_the_restore_source_survives_and_its_siblings_go(self):
        bank = _bank()
        _older, final, _post = _conversation(bank)
        # The prompt extends the generation-final entry exactly.
        prompt = final.token_ids + (80, 81)
        receipt = bank.release_sessions(None, protect_tokens=prompt)
        assert _keys(bank) == {final.token_ids}
        assert receipt["protected_restore_source_tokens"] == len(final.token_ids)

    def test_a_shared_first_token_is_not_a_restore_source(self):
        bank = _bank()
        _conversation(bank)
        assert bank.restore_source_key((1, 99, 98, 97)) is None
        receipt = bank.release_sessions(None, protect_tokens=(1, 99, 98, 97))
        assert receipt["protected_restore_source_tokens"] is None
        assert _keys(bank) == set()

    def test_a_block_restorable_prefix_is_a_restore_source(self):
        bank = _bank(max_bytes=10**9, per_session_max_bytes=10**9)
        shared = tuple(range(1_000))
        entry = _put(bank, shared + (5, 5), session_id="conv", nbytes=100)
        # 1,000 shared tokens round down to 768 on 256-token blocks: over the
        # 512-token restore floor, so the entry serves this prompt.
        assert bank.restore_source_key(shared + (7, 7)) == entry.token_ids
        # 300 shared tokens round down to 256: under the floor.
        assert bank.restore_source_key(shared[:300] + (7, 7)) is None

    def test_the_admission_walk_no_longer_protects_a_one_token_overlap(self):
        """Old rule: the entry with the greatest common prefix was protected
        even when that prefix was one token; shrink_for_admission(0) then
        kept it. Now it is walked like any other terminal."""

        bank = _bank()
        _put(bank, (1, 50, 51), session_id="a", nbytes=100)
        assert bank.shrink_for_admission(0, protect_tokens=(1, 99, 98)) == (0, 1)
        assert _keys(bank) == set()

    def test_entries_already_on_ssd_go_first(self):
        bank = _bank()
        not_on_disk = _put(bank, (1, 1, 1), session_id="older", nbytes=100)
        on_disk = _put(bank, (2, 2, 2), session_id="newer", nbytes=100, on_ssd=True)
        not_on_disk.last_access_s = 1.0
        on_disk.last_access_s = 2.0
        receipt = bank.release_sessions(100)
        assert _keys(bank) == {not_on_disk.token_ids}
        [row] = receipt["sessions"]
        assert row["session_id"] == "newer"
        assert row["on_ssd_entries"] == 1
        assert receipt["dropped_entries"] == 0

    def test_least_recently_used_session_goes_first(self):
        bank = _bank()
        stale = _put(bank, (1, 1, 1), session_id="stale", nbytes=100)
        recent = _put(bank, (2, 2, 2), session_id="recent", nbytes=100)
        stale.last_access_s = 10.0
        recent.last_access_s = 20.0
        bank.release_sessions(100)
        assert _keys(bank) == {recent.token_ids}

    def test_dropping_an_entry_not_on_disk_cancels_its_queued_encode(self):
        bank = _bank()
        cancelled: list[str] = []
        bank.cold_enqueue_cancel = lambda key: cancelled.append(key) or 1
        _put(bank, (1, 1, 1), session_id="conv", nbytes=100)
        _put(bank, (1, 1, 9), session_id="conv", nbytes=100, on_ssd=True)
        _put(bank, (3, 3), session_id="other", nbytes=100, on_ssd=True)
        receipt = bank.release_sessions(None)
        # One cancel per session key, and only for the session with an entry
        # that is not on disk yet.
        assert cancelled == ["ssd_cold:conv"]
        assert receipt["persistence_cancelled"] == 1

    def test_a_released_session_loses_its_activity_pin(self):
        bank = _bank()
        _conversation(bank)
        assert "anon-conv" in bank._active_session_ids()
        bank.release_sessions(None)
        assert "anon-conv" not in bank._active_session_ids()


class TestSchedulerCancel:
    def test_cancel_drops_the_pending_job_and_the_arrays_it_pins(self):
        scheduler = ModelWorkScheduler(name="test-release", idle_grace_s=3600)
        try:
            ran = threading.Event()

            class Snapshot:
                pass

            pinned = Snapshot()
            ref = weakref.ref(pinned)
            entry = SimpleNamespace(session_id="conv", token_hash="h")

            def job(_held=pinned):
                ran.set()

            key = cold_persistence_key(entry)
            future = scheduler.submit_idle_persistence(job, coalesce_key=key)
            del job, pinned
            assert scheduler.cancel_idle_persistence("ssd_cold:other") == 0
            assert scheduler.cancel_idle_persistence(key) == 1
            assert future.cancelled()
            gc.collect()
            assert ref() is None
            assert not ran.is_set()
            assert scheduler.stats()["persistence_cancelled"] == 1
        finally:
            scheduler.shutdown(wait=True, cancel_futures=True)

    def test_the_key_constructor_matches_every_dispatch_shape(self):
        assert cold_persistence_key(SimpleNamespace(session_id="s", token_hash="h")) == "ssd_cold:s"
        assert cold_persistence_key(SimpleNamespace(session_id=None, token_hash="h")) == "ssd_cold:hash:h"


class TestManagerRelease:
    def _manager(self) -> EngineSessionManager:
        return EngineSessionManager(bank=_bank(), idle_ttl_s=3600)

    def test_an_in_flight_session_is_never_touched(self):
        manager = self._manager()
        busy = manager.get_or_create("busy")
        idle = manager.get_or_create("idle")
        _put(manager.bank, (1, 1, 1), session_id="busy", nbytes=100)
        _put(manager.bank, (2, 2, 2), session_id="idle", nbytes=100)
        assert busy.try_begin_generation()
        try:
            receipt = manager.release_idle_sessions(None)
        finally:
            busy.end_generation()
        assert manager.bank.has_session_entries("busy")
        assert not manager.bank.has_session_entries("idle")
        assert "busy" in receipt["kept_sessions"]
        assert manager.peek("busy") is busy
        assert idle.session_id in receipt["session_records_dropped"]

    def test_the_callers_sessions_are_kept(self):
        manager = self._manager()
        manager.get_or_create("incoming")
        _put(manager.bank, (1, 1, 1), session_id="incoming", nbytes=100)
        receipt = manager.release_idle_sessions(None, keep_session_ids={"incoming"})
        assert receipt["entries"] == 0
        assert manager.bank.has_session_entries("incoming")

    def test_a_released_session_stops_advertising_its_prefix(self):
        """committed_token_ids outliving the cache would let the admission
        estimate read a cold prefill as warm."""

        manager = self._manager()
        session = manager.get_or_create("conv")
        session.commit(prompt_ids=[1, 2, 3], generated_ids=[4, 5], finish_reason="stop")
        _put(manager.bank, (1, 2, 3, 4, 5), session_id="conv", nbytes=100)
        assert manager.longest_prefix_session([1, 2, 3, 4, 5, 6]) is session
        manager.release_idle_sessions(None, keep_session_ids={"anon-compaction"})
        assert manager.peek("conv") is None
        assert manager.longest_prefix_session([1, 2, 3, 4, 5, 6]) is None

    def test_a_released_sessions_pending_postcommit_is_aborted(self):
        from concurrent.futures import Future

        manager = self._manager()
        session = manager.get_or_create("conv")
        _put(manager.bank, (1, 2, 3), session_id="conv", nbytes=100)
        record = session.set_pending_postcommit(Future())
        receipt = manager.release_idle_sessions(None)
        assert receipt["postcommits_aborted"] == 1
        assert record.abort_event.is_set()

    def test_a_session_with_entries_left_keeps_its_record(self):
        manager = self._manager()
        manager.get_or_create("conv")
        kept_entry = _put(manager.bank, (1, 2, 3), session_id="conv", nbytes=100)
        _put(manager.bank, (1, 2, 9), session_id="conv", nbytes=100)
        # The prompt restores from (1, 2, 3): it stays, so does the record.
        manager.release_idle_sessions(None, protect_tokens=kept_entry.token_ids + (4,))
        assert manager.bank.has_session_entries("conv")
        assert manager.peek("conv") is not None
