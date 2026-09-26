"""/v1/completions and the session bank (``MTPLX_COMPLETIONS_SESSION_BANK``).

Chat generations restore shared prompt prefixes from the session bank and
bank their final state; completions generations prefilled every prompt cold.
With the switch on, a completion gets the restore and commit an anonymous
chat request gets, under its own session and policy fingerprint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from tests.test_server_openai import _fake_state


def _post_completion(monkeypatch, *, stream: bool = False) -> tuple[dict, dict]:
    captured: dict = {}

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured.update(kwargs)
        if kwargs.get("token_callback") is not None:
            kwargs["token_callback"]([4])
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {"completion_tokens": 1},
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    state = _fake_state()
    state.sessions.bank = object()
    client = TestClient(openai.create_app(state))
    monkeypatch.setattr(openai, "_encode_prompt", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 1, "stream": stream},
    )
    assert response.status_code == 200
    return captured, {"state": state}


def test_completions_skip_the_bank_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPLETIONS_SESSION_BANK", raising=False)

    captured, _ = _post_completion(monkeypatch)

    assert captured.get("session_bank") is None
    assert captured.get("session_id") is None


@pytest.mark.parametrize("stream", [False, True])
def test_completions_use_the_bank_when_switched_on(monkeypatch, stream):
    monkeypatch.setenv("MTPLX_COMPLETIONS_SESSION_BANK", "1")

    captured, context = _post_completion(monkeypatch, stream=stream)

    state = context["state"]
    assert captured["session_bank"] is state.sessions.bank
    assert captured["session_id"] == openai.COMPLETIONS_SESSION_ID
    assert captured["session_policy_fingerprint"] == openai.COMPLETIONS_POLICY_FINGERPRINT
    assert captured["session_template_hash"] == state.template_hash
    assert captured["session_draft_head_identity"] == state.draft_head_identity
    # Anonymous traffic never pins live paged-KV buffers.
    assert captured["session_keep_live_ref"] is False


def test_completions_bank_needs_a_bank(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPLETIONS_SESSION_BANK", "1")
    state = _fake_state()
    state.sessions.bank = None

    assert openai._completions_session_bank_kwargs(state) == {}
