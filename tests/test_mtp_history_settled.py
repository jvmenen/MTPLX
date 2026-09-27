"""The draft history holds at most one unevaluated append, on every path (#544).

With ``MTPLX_LAZY_MTP_HISTORY_APPEND`` each draft-history append is left lazy
for the draft forward that reads the history next. Many supported paths append
without such a reader: a draft on a fresh per-step cache, a proposal served
from the correction cache (its draft logits are never read), a target-prefix
draft substitution, a context-copy round, and the final pending-token commit,
whose state rebase installs a new history with a lazy prefill append. Stacked
appends keep every round's rows alive, and on a QSA draft head the Metal shared
event of each round that computed them; the session bank snapshots whatever the
generation hands over. ``generate_mtpk`` therefore schedules the history before
every lazy append and once more when it ends.

Pinned here on the tiny Flash-Next pack, path by path:
- the history holds no unevaluated work when an append starts, so at most one
  append is ever pending;
- the history the generation hands over holds none;
- scheduling changes no bit: a bank snapshot taken while the prompt's history
  append was still lazy keeps its bits through the settle and the writes after
  it, and a warm turn restored from it decodes the same tokens as with eager
  appends.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

import mtplx.context_copy as context_copy
import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.session_bank import SessionBank
from tests.test_qsa_index_block_writes_evaluated import (  # noqa: F401 - tiny_pack is a fixture
    _pending_ops,
    _qsa_entries,
    _qsa_leaves,
    _same_bits,
    tiny_pack,
)

_PROMPT = [3, 5, 7, 9, 11, 13] + list(range(20, 40))
_APPEND = generation._append_mtp_history
_RESET = {
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_STATE_REBASE_EVERY",
    "MTPLX_FAMILY_CAPTURE_COMMIT",
    "MTPLX_SUSTAINED_PREFILL",
    "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD",
    "MTPLX_MTP_HISTORY_MATERIALIZE_EVERY",
    "MTPLX_CONTEXT_COPY_TARGET_PREFIX",
    "MTPLX_LAZY_MTP_HISTORY_APPEND",
}


def _backlog(cache) -> int:
    """Unevaluated operations anywhere in a draft history cache."""

    return sum(_pending_ops(leaf) for leaf in _qsa_leaves(_qsa_entries(cache)))


def _generate(
    tiny_pack,
    monkeypatch,
    *,
    max_tokens: int,
    prompt: list[int] = _PROMPT,
    env: dict[str, str] | None = None,
    lazy: bool = True,
    copy_streak: bool = False,
    **kwargs,
):
    """One generation, recording the history's backlog as each append starts.

    Returns the output, the backlog at every append, and the history cache the
    last append wrote, which is the cache the generation hands over.
    """

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.sampling import SamplerConfig

    smoke, model = tiny_pack
    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name in _RESET:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_LAZY_MTP_HISTORY_APPEND", "1" if lazy else "0")
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    if copy_streak:
        # Propose the prompt's first block every round: the tiny random model
        # rejects it, so the lane runs its probation rounds back to back.
        monkeypatch.setattr(
            context_copy.NgramIndex, "find", lambda self, history, max_pos=None: (0, 64)
        )
    at_append: list[int] = []
    written: list[list] = []

    def recording(rt, mtp_cache, *args, **append_kwargs):
        at_append.append(_backlog(mtp_cache))
        if written:
            written[0] = mtp_cache
        else:
            written.append(mtp_cache)
        return _APPEND(rt, mtp_cache, *args, **append_kwargs)

    monkeypatch.setattr(generation, "_append_mtp_history", recording)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    options = dict(
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=3,
        seed=1234,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )
    options.update(kwargs)
    out = generation.generate_mtpk(rt, list(prompt), **options)
    assert len(out.tokens) == max_tokens
    return out, at_append, (written[0] if written else None)


def test_drafts_on_fresh_caches_never_leave_the_committed_history_pending(
    tiny_pack, monkeypatch
):
    out, at_append, _ = _generate(
        tiny_pack, monkeypatch, max_tokens=40, mtp_cache_policy="fresh"
    )
    # Every draft ran on a cache of its own, so no draft read the history.
    assert out.stats.drafted_tokens > 0 and len(at_append) > 3
    handed_over = _backlog(out.final_state.final_committed_mtp_cache)
    assert (max(at_append), handed_over) == (0, 0), at_append


def test_correction_cache_proposals_never_leave_the_history_pending(tiny_pack, monkeypatch):
    # Every token of the tiny vocabulary is followed by its successor in this
    # prompt, so each depth-one proposal comes from the seeded correction cache
    # and no draft logit is read.
    prompt = [token % 128 for token in range(130)]
    out, at_append, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        prompt=prompt,
        speculative_depth=1,
        prompt_correction_cache=True,
        prompt_correction_cache_min_depth=1,
    )
    assert out.stats.online_correction_cache["hits"] > 3, out.stats.online_correction_cache
    assert len(at_append) > 3
    handed_over = _backlog(out.final_state.final_committed_mtp_cache)
    assert (max(at_append), handed_over) == (0, 0), at_append


def test_a_live_reset_leaves_no_history_pending(tiny_pack, monkeypatch):
    out, at_append, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=60,
        env={"MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "8"},
    )
    assert out.stats.mtp_history_live_resets > 0
    handed_over = _backlog(out.final_state.final_committed_mtp_cache)
    assert (max(at_append), handed_over) == (0, 0), at_append


def test_a_rebase_inside_the_final_commit_hands_over_no_pending_history(
    tiny_pack, monkeypatch
):
    # The final pending-token commit rebases the state, and a prefill without
    # the sustained path gives the new history a lazy append of the prompt.
    out, _, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=1,
        env={"MTPLX_SUSTAINED_PREFILL": "0", "MTPLX_STATE_REBASE_EVERY": "1"},
    )
    assert out.stats.state_rebase_events > 0
    assert out.final_state.safe_to_commit
    assert _backlog(out.final_state.final_committed_mtp_cache) == 0


def test_target_prefix_draft_substitution_leaves_no_history_pending(tiny_pack, monkeypatch):
    out, at_append, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        copy_streak=True,
        verify_strategy="target_prefix",
        speculative_depth=1,
        env={"MTPLX_CONTEXT_COPY_TARGET_PREFIX": "1"},
    )
    # The copy match served the depth-one draft; the draft head never ran.
    assert out.stats.context_copy_drafted_tokens > 0
    handed_over = _backlog(out.final_state.final_committed_mtp_cache)
    assert (max(at_append), handed_over) == (0, 0), at_append


def test_a_generation_without_a_final_state_leaves_no_pending_history(tiny_pack, monkeypatch):
    out, at_append, history = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=40,
        copy_streak=True,
        capture_final_state=False,
        env={"MTPLX_FAMILY_CAPTURE_COMMIT": "1"},
    )
    assert out.final_state is None and history is not None
    assert (max(at_append), _backlog(history)) == (0, 0), at_append


def _saved_arrays(entry) -> list[mx.array]:
    return [
        leaf
        for leaf in generation._tree_mx_arrays(
            [entry.cache_snapshot, entry.mtp_history_snapshot, entry.logits, entry.hidden]
        )
    ]


def _bank_then_restore(tiny_pack, monkeypatch, *, lazy: bool):
    bank = SessionBank()
    first, _, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=40,
        lazy=lazy,
        copy_streak=True,
        env={"MTPLX_FAMILY_CAPTURE_COMMIT": "1"},
        session_bank=bank,
        session_id="snapshot-before-settle",
        commit_prompt_state_to_bank=True,
    )
    entries = list(bank._entries.values())
    assert len(entries) == 1 and entries[0].mtp_history_snapshot is not None
    # Read after the generation settled the history and wrote past it.
    saved = _saved_arrays(entries[0])
    mx.eval(saved)
    warm, warm_appends, _ = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        lazy=lazy,
        env={"MTPLX_FAMILY_CAPTURE_COMMIT": "1"},
        session_bank=bank,
        session_id="snapshot-before-settle",
    )
    return first, saved, warm, warm_appends


def test_a_bank_snapshot_taken_before_the_settle_keeps_its_bits_and_restores(
    tiny_pack, monkeypatch
):
    lazy_first, lazy_saved, lazy_warm, warm_appends = _bank_then_restore(
        tiny_pack, monkeypatch, lazy=True
    )
    eager_first, eager_saved, eager_warm, _ = _bank_then_restore(
        tiny_pack, monkeypatch, lazy=False
    )
    assert lazy_warm.stats.cached_tokens > 0, "the warm turn restored from the bank"
    assert max(warm_appends) == 0, warm_appends
    assert list(lazy_first.tokens) == list(eager_first.tokens)
    assert len(lazy_saved) == len(eager_saved)
    assert all(_same_bits(a, b) for a, b in zip(lazy_saved, eager_saved))
    assert list(lazy_warm.tokens) == list(eager_warm.tokens)
    assert _same_bits(lazy_warm.final_state.final_logits, eager_warm.final_state.final_logits)
    lazy_head = _qsa_leaves(_qsa_entries(lazy_warm.final_state.final_committed_mtp_cache))
    eager_head = _qsa_leaves(_qsa_entries(eager_warm.final_state.final_committed_mtp_cache))
    assert len(lazy_head) == len(eager_head)
    assert all(_same_bits(a, b) for a, b in zip(lazy_head, eager_head))
