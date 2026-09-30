"""One copy of the conversation's attention state (Flash-Next).

On 2026-09-29 a 138K-token Pi session held the same conversation two or three
times: the session bank's snapshot views, the live cache the next request
wrote into (a view blocks MLX's in-place write, so the first write copied the
whole buffer), and the fixed-M4 verifier's padded bank (promotion
concatenated the history into new arrays). The duplicates switched the
compiled decode route off from 98K tokens and caused both mid-answer 507s.

The one-copy store keeps a single set of buffers per conversation:

- the prefill sizes the QSA buffers once to the rows the verifier's bank will
  have (``qwen4_exp.qsa_rows_target``), so the bank adopts them instead of
  copying (``graphbank.TensorOffsetQSACache.adoptable_rows``);
- the bank hands the same buffers back at the end of the request
  (``demote(keep_capacity=True)``);
- the session bank keeps the conversation as a reference lease with its
  recurrent anchors and never as views of the live buffers.

Numerics do not change: every reader still reads the rows it read before,
at the widths it read them.
"""

from __future__ import annotations

import os
from typing import Any

_DISABLE_VALUES = {"0", "false", "no", "off"}


def one_copy_enabled() -> bool:
    """The one-copy conversation store (MTPLX_ONE_COPY, on unless set off)."""

    return str(os.environ.get("MTPLX_ONE_COPY", "1")).strip().lower() not in _DISABLE_VALUES


def one_copy_runtime(rt: Any) -> bool:
    """True for a runtime whose verifier keeps the conversation in fixed QSA banks."""

    return one_copy_enabled() and bool(getattr(rt, "qwen4_fixed_m4_compiled_verify", False))


def qsa_entries(cache: Any) -> list[Any]:
    """The QSA layers of a trunk cache, stock or promoted."""

    from .graphbank import TensorOffsetQSACache
    from .models.qwen4_exp import QSACache

    return [
        entry for entry in (cache or ())
        if isinstance(entry, (QSACache, TensorOffsetQSACache))
    ]


def held_qsa_rows(cache: Any) -> int | None:
    """Rows every stock QSA layer of ``cache`` already holds, or None.

    None when the cache has no QSA layer or its layers disagree (then the
    verifier's promotion pads and copies, as it always did).
    """

    from .graphbank import TensorOffsetQSACache

    rows: set[int] = set()
    entries = qsa_entries(cache)
    if not entries:
        return None
    for entry in entries:
        held = (
            entry.capacity
            if isinstance(entry, TensorOffsetQSACache)
            else TensorOffsetQSACache.held_rows(entry)
        )
        if held is None:
            return None
        rows.add(int(held))
    return rows.pop() if len(rows) == 1 else None


def prefill_rows_target(rt: Any, prompt_tokens: int, plan: Any) -> int:
    """Rows a request's prefill should size its QSA buffers to, or 0.

    The rows the fixed-M4 bank will be built with for this prompt
    (``FixedM4CapacityPlan.rows``), when the prompt starts on the rows-gather
    lane with the capacity bucket in force: there the capacity is a whole
    number of pages (in-place writes stay in place) and a bucket-larger
    buffer changes no value. The dense lane keeps its exact capacity and the
    stock step growth (its buffers are small and its width is arithmetic), so
    it gets 0.
    """

    if plan is None or not one_copy_runtime(rt):
        return 0
    from .graphbank import TensorOffsetQSACache
    from .models.qwen4_exp import _qsa_gather_enabled, _qsa_gather_min_context

    prompt_tokens = max(0, int(prompt_tokens))
    if not (_qsa_gather_enabled() and prompt_tokens >= _qsa_gather_min_context()):
        return 0
    rows = int(plan.rows(prompt_tokens, _qsa_ratio(rt), TensorOffsetQSACache.step))
    if int(getattr(plan, "bucket", 0) or 0) <= 0:
        return 0
    return rows


def prompt_lease_fields(
    cache: Any,
    *,
    committed_mtp_cache: Any,
    hidden: Any,
    prompt_len: int,
    boundaries: Any,
    runtime: Any = None,
) -> dict[str, Any]:
    """``SessionBank.put`` arguments that bank a finished prompt as a lease.

    The prompt's prefill used to be banked as a snapshot of views before the
    answer decoded into the same buffers, which made the first decode write
    copy the whole history. Here the bank takes a lease on the live cache
    and a recurrent anchor at the prompt's end: if the answer is committed,
    its entry replaces this lease and inherits the anchor; if it is not
    (cancelled, refused), a restore rewinds the lease to the anchor. The
    anchor is evaluated now, so it holds its own 115 MB and never pins the
    live recurrent buffers the verifier writes in place. Its size is recorded
    on ``runtime`` (``anchor_nbytes``): it is what the server's admission
    charges for publishing a prompt.
    """

    import mlx.core as mx

    from .cache_state import snapshot_untrimmable_cache

    prompt_len = int(prompt_len)
    snapshot = snapshot_untrimmable_cache(cache)
    leaves = [
        leaf
        for state in snapshot.states
        if state is not None
        for leaf in (state if isinstance(state, (list, tuple)) else [state])
        if isinstance(leaf, mx.array)
    ]
    if leaves:
        mx.eval(*leaves)
    if runtime is not None:
        _record_anchor_nbytes(runtime, sum(int(leaf.nbytes) for leaf in leaves))
    kept = [
        record for record in (boundaries or ())
        if int(record[0]) != prompt_len
    ]
    return {
        "keep_live_ref": True,
        "mtp_history_snapshot": None,
        "mtp_history_cache_ref": committed_mtp_cache,
        "gdn_boundaries": [*kept, (prompt_len, snapshot, hidden)],
    }


def anchor_nbytes(rt: Any) -> int | None:
    """Bytes one prompt anchor holds on this runtime, once one was taken.

    The recurrent state is fixed-size (115,642,384 bytes on Flash-Next), so
    the first prompt lease measures it for every later admission. None before
    that: the admission then prices the publication as the copy it used to
    be, which only ever over-charges.
    """

    value = getattr(rt, "one_copy_anchor_nbytes", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def prefill_slack_rows(rt: Any, prompt_tokens: int, max_tokens: int) -> int:
    """Rows a one-copy prefill allocates past the prompt, or 0.

    The prefill sizes its QSA buffers once to the verifier bank's rows
    (``prefill_rows_target``): the answer's reserve rounded up to the
    capacity bucket, allocated when the prompt is written instead of when
    the bank is built.
    """

    if not one_copy_runtime(rt):
        return 0
    from .graphbank import FixedM4CapacityPlan

    plan = FixedM4CapacityPlan.for_request(max(1, int(max_tokens)), runtime=rt)
    target = prefill_rows_target(rt, prompt_tokens, plan)
    return max(0, int(target) - max(0, int(prompt_tokens)))


def _record_anchor_nbytes(rt: Any, nbytes: int) -> None:
    if int(nbytes) <= 0:
        return
    try:
        rt.one_copy_anchor_nbytes = int(nbytes)
    except Exception:
        # A runtime that takes no attributes keeps the copy price.
        pass


class LeaseReturn:
    """What a warm prefill needs to give its lease back if it stops early.

    Taken right after the restore handed the prefill a one-copy lease, before
    anything is written: the recurrent state at the restore point (a real
    copy, 115.6 MB on Flash-Next, so the prefill's in-place writes cannot
    touch it) and the attention and draft-history offsets there. ``give_back``
    banks the lease again at that point (``SessionBank.return_lease``); a
    later restore rewinds whatever the prefill wrote past it.
    """

    def __init__(self, bank, runtime, *, cache, mtp_history_cache, token_ids,
                 hidden, source, boundaries=()):
        import mlx.core as mx

        from .cache_state import snapshot_untrimmable_cache
        from .session_bank import _cache_kv_offset

        self.bank = bank
        self.runtime = runtime
        self.cache = cache
        self.mtp_history_cache = mtp_history_cache
        self.token_ids = tuple(int(token) for token in token_ids)
        self.source = source
        self.boundaries = list(boundaries or ())
        self.kv_offset = _cache_kv_offset(cache)
        self.mtp_offset = (
            _cache_kv_offset(mtp_history_cache) if mtp_history_cache is not None else None
        )
        snapshot = snapshot_untrimmable_cache(cache)
        leaves = [
            leaf
            for state in snapshot.states
            if state is not None
            for leaf in (state if isinstance(state, (list, tuple)) else [state])
            if isinstance(leaf, mx.array)
        ]
        if leaves:
            mx.eval(*leaves)
        self.anchor = (len(self.token_ids), snapshot, hidden)

    def give_back(self) -> Any:
        if self.kv_offset is None:
            return None
        return self.bank.return_lease(
            self.runtime,
            token_ids=self.token_ids,
            cache=self.cache,
            mtp_history_cache=self.mtp_history_cache,
            anchor=self.anchor,
            kv_offset=self.kv_offset,
            mtp_offset=self.mtp_offset,
            source=self.source,
            boundaries=self.boundaries,
        )


def lease_return(bank, runtime, *, restore_mode, cache, mtp_history_cache, token_ids,
                 hidden, source, boundaries=()) -> LeaseReturn | None:
    """A ``LeaseReturn`` when the restore handed over a one-copy lease, else None."""

    if (
        str(restore_mode) != "reference_lease"
        or token_ids is None
        or not callable(getattr(bank, "return_lease", None))
    ):
        return None
    from .session_bank import _one_copy_cache

    if not _one_copy_cache(cache):
        return None
    return LeaseReturn(
        bank, runtime, cache=cache, mtp_history_cache=mtp_history_cache,
        token_ids=token_ids, hidden=hidden, source=source, boundaries=boundaries,
    )


def prefill_rows_scope(rows: int):
    """``qwen4_exp.qsa_rows_target(rows)``, or a no-op for 0 (no model import)."""

    import contextlib

    if int(rows or 0) <= 0:
        return contextlib.nullcontext()
    from .models.qwen4_exp import qsa_rows_target

    return qsa_rows_target(int(rows))


def _qsa_ratio(rt: Any) -> int:
    model = getattr(rt, "model", None)
    text = getattr(model, "language_model", model)
    args = getattr(text, "args", None) or getattr(getattr(text, "model", None), "args", None)
    return max(1, int(getattr(args, "indexer_compress_ratio", 4) or 4))
