"""Shared serving prefill guard construction and prompt-scoring admission."""

from __future__ import annotations

import json
from typing import Any, Mapping


def make_prefill_system_guard(
    state: Any,
    *,
    prompt_tokens: int,
    chunk_tokens: int | None,
    priced: Mapping[str, Any] | None,
):
    from mtplx.server import openai as srv

    after_forward: dict[str, Any] = {}
    try:
        reserve = srv._prefill_chunk_reserve_bytes(
            state, prompt_tokens=prompt_tokens, chunk_tokens=chunk_tokens, priced=priced
        )
        after_forward = srv._prefill_after_forward_plan(
            state, prompt_tokens=prompt_tokens, chunk_tokens=chunk_tokens, priced=priced
        )
    except Exception as exc:  # noqa: BLE001
        # Keep generation's conservative reservation and visible degraded
        # health if pricing fails. The live guard still checks every chunk.
        from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

        reserve = int(RUNTIME_TRANSIENTS_BYTES)
        srv._note_guard_health(state, where="prefill_chunk_reserve", error=exc)
        event = {
            "action": "prefill_chunk_reserve_error",
            "error": repr(exc),
            "guard_degraded": True,
        }
        srv._record_guard_event(state, event)
        try:
            print("[mtplx] memory guard " + json.dumps(event), flush=True)
        except Exception:
            pass
    else:
        srv._note_guard_health(state, where="prefill_chunk_reserve", error=None)
    return srv._PrefillSystemGuard(
        state, chunk_reserve_bytes=reserve, **after_forward
    )


def score_prompt_with_memory_policy(
    state: Any,
    prompt_ids: list[int],
    *,
    top_k: int,
    request_observability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score under the serving lock with generation's prefill safety policy."""
    from mtplx.generation import prefill_chunk_size_override
    from mtplx.server import openai as srv

    width = getattr(state.args, "prefill_chunk_tokens", None)
    pricing: dict[str, Any] = {}
    admission = srv._prefill_admission_shed(
        state,
        prompt_ids=prompt_ids,
        session_bank=None,
        session_id=None,
        max_new_tokens=0,
        mtp_depth=0,
        prefill_chunk_tokens=width,
        pricing=pricing,
    )
    if admission is not None:
        if request_observability is not None:
            request_observability["prefill_admission_shed"] = admission
        if admission.get("refused"):
            raise srv._prefill_admission_refusal(state, admission)
        if admission.get("prefill_chunk_tokens") is not None:
            width = int(admission["prefill_chunk_tokens"])
    guard = make_prefill_system_guard(
        state, prompt_tokens=len(prompt_ids), chunk_tokens=width,
        priced=pricing.get("growth"),
    )

    def abort_check() -> bool:
        return srv._pressure_abort_requested(state) or guard()

    try:
        with prefill_chunk_size_override(width):
            return srv.score_prompt_logprobs(
                state.runtime, prompt_ids, top_k=top_k,
                abort_check=abort_check, prefill_callback=guard.note_prefill_progress,
            )
    except srv.PostcommitAbort:
        if guard.tripped is not None:
            if request_observability is not None:
                request_observability["prefill_system_abort"] = dict(guard.tripped)
            raise srv._prefill_system_abort_exception(state, guard.tripped)
        if srv._pressure_abort_requested(state):
            raise srv._allocation_failure_http_exception(
                state, RuntimeError("sustained critical memory pressure during prompt scoring")
            )
        raise
