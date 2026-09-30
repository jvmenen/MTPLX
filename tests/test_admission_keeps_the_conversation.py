"""The prefill admission never deletes the conversation it is serving.

On 2026-09-29 (Pi, Flash-Next, 128 GB, 90 GiB limit) every screenshot turn
after the first image diverged from the session's only bank entry inside the
newest turn. The admission's lookup skips partial image matches, read "no
reusable tokens and a large miss" as a rewritten history, cleared the
session's entries as "superseded" and cancelled their SSD write: the bank
went from 3.9-4.7 GB to 0 four times and each next turn re-read 123K-138K
tokens cold (126-138 s).

The contract now, on the real SessionBank and EngineSessionManager (MLX's
allocator and the process footprint mocked, no model):

* the session's entry that holds the most of the prompt beyond what the
  restore reuses survives every reclamation step, priced where it is, and
  its queued SSD write is never cancelled (text and image divergences);
* once the SSD cache has published it, it may leave RAM as the last step
  before a refusal (a later restore reads it from disk);
* when it and the request still do not fit, the request is refused before
  any work, saying so and whether a retry can succeed;
* a rewritten history (an entry sharing under half of itself with the
  prompt) and a duplicate sibling stay reclaimable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_idle_session_release import _Lane, _Tier
from test_memguard_admission import (
    CONV,
    FN_ROW,
    GIB,
    _flash_next_state,
    _install,
    _Machine,
    _manager,
    _put,
)

import mtplx.server.openai as srv

LIMIT = 96 * GIB
PAD = 7_000_000


@pytest.fixture(autouse=True)
def _chunked_prefill(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")


def _flash_next_entry(bank, tokens, *, session_id="pi"):
    """A Flash-Next conversation entry with no recurrent checkpoint: the
    restore can reuse it only as an exact prefix (the 09-29 shape)."""

    entry = _put(bank, tokens, session_id=session_id, row_bytes=FN_ROW, live_cache=True)
    entry.has_recurrent = True
    return entry


def _world(*, ssd: bool):
    manager = _manager(cold_tier=_Tier() if ssd else None)
    lane = _Lane()
    if ssd:
        manager.bank.cold_enqueue_dispatch = lane.dispatch
        manager.bank.cold_enqueue_cancel = lane.cancel
    return manager, lane


def _admit(monkeypatch, manager, lane, prompt, *, base_gib, vision_splice=None):
    machine = _Machine(manager.bank, base_gib=base_gib, cache_gib=0.0, host_gib=0.0, lane=lane)
    _install(monkeypatch, machine)
    state = _flash_next_state(manager)
    # The request holds its session's slot while it is admitted.
    conversation = manager.get_or_create("pi")
    assert conversation.try_begin_generation()
    try:
        return srv._prefill_admission_shed(
            state,
            prompt_ids=prompt,
            session_bank=manager.bank,
            session_id="pi",
            vision_splice=vision_splice,
        ), state
    finally:
        conversation.end_generation()


# The request's bill (Flash-Next, cold 123K prompt, 2,048 rows) once the
# admission skips the banked prompt copy: read off the real admission.
GROWTH = 6_138_587_648
TEXT_PROMPT = list(CONV[:110_000]) + list(range(2_000_000, 2_013_000))


def _base_for(projected_over_limit_gib: float, entry_bytes: int) -> float:
    """The engine's base so the request, with the kept entry resident,
    projects ``projected_over_limit_gib`` past the 96 GiB limit."""

    return (LIMIT + projected_over_limit_gib * GIB - GROWTH - entry_bytes) / GIB


class TestSameConversationSurvives:
    def test_a_text_divergence_keeps_the_entry_and_its_ssd_write(self, monkeypatch):
        manager, lane = _world(ssd=True)
        entry = _flash_next_entry(manager.bank, CONV)
        stranger = _put(manager.bank, range(9_000_000, 9_060_000), session_id="other", row_bytes=FN_ROW)
        assert set(lane.pending) == {"ssd_cold:pi", "ssd_cold:other"}

        # Over the 0.97 line with the stranger resident, under the limit once
        # it is gone.
        receipt, _state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(-1.0, entry.nbytes),
        )

        # The conversation's entry survives, and so does its SSD write.
        assert manager.bank._entries.get(entry.token_ids) is entry
        assert "ssd_cold:pi" in lane.pending
        assert receipt["reusable_prefix_tokens"] == 0
        assert "refused" not in receipt
        assert "superseded_session_entries_evicted" not in receipt
        assert receipt["kept_session_entries"]["longest_shared_prefix_tokens"] == 110_000
        # Reclamation still took the idle stranger.
        assert stranger.token_ids not in manager.bank._entries

    def test_an_image_divergence_keeps_the_entry_and_its_ssd_write(self, monkeypatch):
        from mtplx.vision.splice import vision_bank_key_ids

        splice = SimpleNamespace(image_pad_token_id=PAD, image_digests=[123], pad_counts=[300])
        before = list(range(60_000)) + [PAD] * 300 + list(range(100_000, 150_000))
        # The next turn re-encodes the newest assistant turn differently
        # (an edit's tool call): it diverges 5K tokens before the entry ends.
        prompt = before[:105_300] + list(range(3_000_000, 3_017_700))
        manager, lane = _world(ssd=True)
        entry = _flash_next_entry(manager.bank, vision_bank_key_ids(before, splice))
        stranger = _put(manager.bank, range(9_000_000, 9_060_000), session_id="other", row_bytes=FN_ROW)

        receipt, _state = _admit(
            monkeypatch, manager, lane, prompt,
            base_gib=_base_for(-1.0, entry.nbytes), vision_splice=splice,
        )

        assert manager.bank._entries.get(entry.token_ids) is entry
        assert "ssd_cold:pi" in lane.pending
        assert receipt["reusable_prefix_tokens"] == 0
        assert "refused" not in receipt
        assert receipt["kept_session_entries"]["entries"] == 1
        assert stranger.token_ids not in manager.bank._entries


class TestWhenBothCannotFit:
    def test_refused_before_work_and_the_retry_waits_for_the_ssd_write(self, monkeypatch):
        manager, lane = _world(ssd=True)
        entry = _flash_next_entry(manager.bank, CONV)

        receipt, state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(1.0, entry.nbytes),
        )

        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["retry_can_succeed"] is True
        assert receipt["retry_when"] == "after_the_conversation_cache_reaches_ssd"
        kept = receipt["kept_session_entries_resident_after"]
        assert kept["ssd_write_pending"] is True
        assert kept["held_bytes"] == entry.nbytes
        # Nothing of the conversation was lost to the refusal.
        assert manager.bank._entries.get(entry.token_ids) is entry
        assert "ssd_cold:pi" in lane.pending
        error = srv._prefill_admission_refusal(state, receipt)
        assert error.status_code == 507
        message = error.detail["message"]
        assert "kept this conversation's cached context" in message
        assert "once it is on disk the engine moves it out of memory" in message
        assert error.detail["memory"]["retry_when"] == "after_the_conversation_cache_reaches_ssd"

        # The write lands while the engine is idle; the retry moves the entry
        # to the SSD cache instead of deleting it, and is admitted.
        lane.run_all()
        manager.bank.cold_tier.published.add(entry.token_ids)
        retry, _state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(1.0, entry.nbytes),
        )

        assert "refused" not in retry
        assert retry["kept_session_entries_moved_to_ssd"]["entries"] == 1
        assert "same_conversation_to_ssd" in retry["reclamation_steps"]
        assert entry.token_ids not in manager.bank._entries
        assert manager.bank.eviction_log[-1]["reason"] == "prefill_admission_moved_to_ssd"
        assert manager.bank.eviction_log[-1]["persistence_cancelled"] == 0

    def test_without_an_ssd_copy_the_entry_gives_way_and_the_request_runs(self, monkeypatch):
        """Nothing will put the entry on disk (the SSD cache is off), so
        keeping it would refuse this conversation on every turn that does not
        fit beside it. It is released instead and the request re-reads what
        it held; the receipt says why."""

        manager, lane = _world(ssd=False)
        entry = _flash_next_entry(manager.bank, CONV)

        receipt, _state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(1.0, entry.nbytes),
        )

        assert "refused" not in receipt
        assert receipt["kept_session_entries_released_no_ssd"]["entries"] == 1
        assert "same_conversation_released_no_ssd" in receipt["reclamation_steps"]
        assert entry.token_ids not in manager.bank._entries
        assert manager.bank.eviction_log[-1]["reason"] == "prefill_admission_released_no_ssd"


class TestStillReclaimable:
    def test_a_rewritten_history_is_released_before_a_refusal(self, monkeypatch):
        # The client compacted: the session's entry shares only its first
        # 4,000 tokens (under half of itself) with the prompt.
        manager, lane = _world(ssd=True)
        stale = _flash_next_entry(
            manager.bank, list(CONV[:4_000]) + list(range(5_000_000, 5_110_191))
        )

        receipt, _state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(1.0, stale.nbytes),
        )

        assert "kept_session_entries" not in receipt
        assert "refused" not in receipt
        assert stale.token_ids not in manager.bank._entries

    def test_only_the_entry_that_holds_the_most_is_kept(self, monkeypatch):
        # A generation-final and a postcommit twin of the same conversation:
        # the one that holds more of the prompt is kept, the twin goes.
        manager, lane = _world(ssd=True)
        best = _flash_next_entry(manager.bank, CONV)
        twin = _flash_next_entry(manager.bank, list(CONV[:108_000]) + [8_888_888] * 6_191)

        # With the twin resident the request is over the limit; without it,
        # it fits beside the kept entry.
        receipt, _state = _admit(
            monkeypatch, manager, lane, TEXT_PROMPT,
            base_gib=_base_for(-1.0, best.nbytes),
        )

        assert receipt["kept_session_entries"]["entries"] == 1
        assert "refused" not in receipt
        assert manager.bank._entries.get(best.token_ids) is best
        assert twin.token_ids not in manager.bank._entries


class TestBank:
    def test_same_conversation_entries_by_share_and_residency(self):
        manager, lane = _world(ssd=True)
        bank = manager.bank
        prompt = tuple(range(1_000))
        compacted = _put(bank, prompt[:10] + (6,) * 90, session_id="pi", row_bytes=1)
        # Put last, so the session's newest queued SSD write is this entry's.
        near = _put(bank, prompt[:900] + (5,) * 100, session_id="pi", row_bytes=1)
        other = _put(bank, prompt[:900], session_id="other", row_bytes=1)

        rows = {row["key"]: row for row in bank.same_conversation_entries("pi", prompt)}

        assert set(rows) == {near.token_ids}
        assert rows[near.token_ids]["shared_tokens"] == 900
        assert rows[near.token_ids]["resident"] is True
        assert rows[near.token_ids]["ssd_write_pending"] is True
        assert compacted.token_ids not in rows and other.token_ids not in rows

    def test_an_entry_out_of_ram_with_its_write_queued_counts(self):
        manager, lane = _world(ssd=True)
        bank = manager.bank
        entry = _put(bank, tuple(range(100)), session_id="pi", row_bytes=1)
        bank._evict_entry(entry, reason="test", cancel_queued_persistence=False)

        [row] = bank.same_conversation_entries("pi", tuple(range(120)))

        assert row["resident"] is False
        assert row["ssd_write_pending"] is True

    def test_only_published_entries_move_to_ssd_and_no_write_is_cancelled(self):
        manager, lane = _world(ssd=True)
        bank = manager.bank
        published = _put(bank, tuple(range(100)), session_id="a", row_bytes=1)
        pending = _put(bank, tuple(range(200, 300)), session_id="b", row_bytes=1)
        bank.cold_tier.published.add(published.token_ids)

        moved = bank.move_durable_entries_to_ssd(
            [published.token_ids, pending.token_ids], reason="test_move"
        )

        assert moved["entries"] == 1
        assert published.token_ids not in bank._entries
        assert bank._entries.get(pending.token_ids) is pending
        assert set(lane.pending) == {"ssd_cold:a", "ssd_cold:b"}

    def test_release_spares_protected_keys_and_their_writes(self):
        manager, lane = _world(ssd=True)
        bank = manager.bank
        sibling = _put(bank, (1, 2, 9), session_id="conv", row_bytes=1)
        kept = _put(bank, (1, 2, 3, 4), session_id="conv", row_bytes=1)

        receipt = bank.release_sessions(
            None, only_session_ids={"conv"}, protect_keys={kept.token_ids}
        )

        assert receipt["entries"] == 1
        assert bank._entries.get(kept.token_ids) is kept
        assert sibling.token_ids not in bank._entries
        assert "ssd_cold:conv" in lane.pending
