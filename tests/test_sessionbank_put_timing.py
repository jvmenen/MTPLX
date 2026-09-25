"""The serial lane's final session-bank put is timed and published.

The generation-final ``session_bank.put`` in ``_run_generation`` runs on the
model-owner thread before the response's terminal frame, so its wall time is
part of every banked request's tail. The batched lanes already publish their
puts (``ar_batch_row_bank_put_s``, ``mtp_batch_prompt_boundary_bank_put_s``);
these tests pin the serial lane's ``sessionbank_put_s`` in the request stats,
the public ``mtplx_stats`` and the metrics row behind ``/v1/mtplx/snapshot``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from test_server_openai import (
    _fake_final_state,
    _fake_streaming_session_state,
)

from mtplx.server import openai

PUT_DELAY_S = 0.02


class _SlowBank:
    """Records puts like RecordingBank, but each put takes PUT_DELAY_S."""

    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put(self, **kwargs):
        time.sleep(PUT_DELAY_S)
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]), nbytes=123, token_hash="h"
        )


def _run(monkeypatch, *, final_state_factory):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.sessions = openai.EngineSessionManager(bank=_SlowBank())
    tokens = [ord("O"), ord("K")]

    def fake_generate_mtpk(*_args, **_kwargs):
        return SimpleNamespace(
            tokens=tokens,
            text="OK",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 2,
                    "elapsed_s": 0.1,
                    "tok_s": 20.0,
                }
            ),
            final_state=final_state_factory(tokens),
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)
    generated = openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=16,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=42,
        generation_mode="mtp",
        depth=3,
        session_id="sess-put-timing",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
    )
    return state, generated


def test_final_put_wall_time_is_published(monkeypatch):
    state, generated = _run(monkeypatch, final_state_factory=_fake_final_state)

    assert len(state.sessions.bank.puts) == 1
    put_s = generated["stats"]["sessionbank_put_s"]
    assert isinstance(put_s, float)
    # The put's own work, not the request around it.
    assert PUT_DELAY_S <= put_s < PUT_DELAY_S + 1.0

    public = openai._public_mtplx_stats(generated)
    assert public["sessionbank_put_s"] == put_s

    # The metrics row that /v1/mtplx/snapshot serves as `latest`.
    assert state.last_metrics[-1]["sessionbank_put_s"] == put_s


def test_no_put_means_no_put_timing(monkeypatch):
    state, generated = _run(monkeypatch, final_state_factory=lambda _tokens: None)

    assert state.sessions.bank.puts == []
    assert "sessionbank_put_s" not in generated["stats"]
    assert "sessionbank_put_s" not in openai._public_mtplx_stats(generated)
    assert "sessionbank_put_s" not in state.last_metrics[-1]
