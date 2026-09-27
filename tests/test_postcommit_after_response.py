"""T7: MTPLX_POSTCOMMIT_AFTER_RESPONSE finishes the chat response before the
session postcommit, and the next request waits for that commit to land."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_server_openai import (
    _fake_final_state,
    _fake_streaming_session_state,
)

from mtplx.server import openai
from mtplx.server.openai import create_app

SESSION = "t7-session"


# --- the barrier itself -----------------------------------------------------


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE", raising=False)
    assert openai._postcommit_after_response_enabled() is False
    for raw in ("1", "true", "on", "YES"):
        monkeypatch.setenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE", raw)
        assert openai._postcommit_after_response_enabled() is True
    for raw in ("0", "", "off", "nee"):
        monkeypatch.setenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE", raw)
        assert openai._postcommit_after_response_enabled() is False


def test_wait_bound_defaults_and_override(monkeypatch):
    monkeypatch.delenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE_WAIT_S", raising=False)
    assert openai._post_response_tail_wait_s() >= 60.0
    monkeypatch.setenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE_WAIT_S", "2.5")
    assert openai._post_response_tail_wait_s() == 2.5
    monkeypatch.setenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE_WAIT_S", "rubbish")
    assert openai._post_response_tail_wait_s() >= 60.0


def test_tails_wait_idle_blocks_until_leave():
    tails = openai._PostResponseTails()
    assert tails.wait_idle(0.01) is True
    tails.enter()
    assert tails.active == 1
    assert tails.wait_idle(0.02) is False
    threading.Timer(0.05, tails.leave).start()
    started = time.monotonic()
    assert tails.wait_idle(2.0) is True
    assert time.monotonic() - started >= 0.03
    assert tails.to_admin_dict() == {
        "active": 0,
        "entered": 1,
        "completed": 1,
        "waits": 3,
        "wait_timeouts": 1,
    }


def test_admission_barrier_receipt():
    class State:
        pass

    state = State()
    assert asyncio.run(openai._await_post_response_tails(state)) is None
    tails = openai._post_response_tails(state)
    assert openai._post_response_tails(state) is tails
    assert asyncio.run(openai._await_post_response_tails(state)) is None
    tails.enter()
    threading.Timer(0.05, tails.leave).start()
    receipt = asyncio.run(openai._await_post_response_tails(state))
    assert receipt is not None
    assert receipt["idle"] is True
    assert receipt["waited_s"] > 0.0


def test_merge_only_touches_own_metrics_row():
    class State:
        def __init__(self) -> None:
            self.last_metrics = [{"request_id": "other"}]

    state = State()
    openai._merge_post_response_stats(
        state, "mine", {"session_postcommit_snapshot": {"stored": True}}
    )
    assert "session_postcommit_snapshot" not in state.last_metrics[-1]
    state.last_metrics.append({"request_id": "mine"})
    openai._merge_post_response_stats(
        state,
        "mine",
        {
            "session_postcommit_snapshot": {"stored": True},
            "session_prompt_prefix_commit": {"committed": True},
            "unrelated": 1,
        },
    )
    assert state.last_metrics[-1] == {
        "request_id": "mine",
        "session_postcommit_snapshot": {"stored": True},
        "session_prompt_prefix_commit": {"committed": True},
    }


# --- end to end through the chat endpoint -------------------------------------


class _Scheduler:
    """Serial stand-in: runs each job in the submitting thread."""

    def __init__(self, log) -> None:
        self.log = log

    def is_owner_thread(self) -> bool:
        return False

    def submit_foreground(self, fn, *args, batch_key=None, **kwargs):
        self.log("job:" + str(batch_key).split(":")[0])
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller
            future.set_exception(exc)
        return future

    def shutdown(self, **_kwargs):
        return None


def _scenario(monkeypatch, *, flag: bool, stream: bool, release_after_s: float):
    """Two same-session turns. The first turn's postcommit blocks until
    ``release`` fires; returns the event log, both responses and the state."""
    if flag:
        monkeypatch.setenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE", "1")
    else:
        monkeypatch.delenv("MTPLX_POSTCOMMIT_AFTER_RESPONSE", raising=False)
    state = _fake_streaming_session_state()
    events: list[str] = []
    lock = threading.Lock()

    def log(event: str) -> None:
        with lock:
            events.append(event)

    state.model_scheduler = _Scheduler(log)
    release = threading.Event()
    runs = {"n": 0}

    def blocking_store(*_args, **_kwargs):
        log("store_start")
        release.wait(5.0)
        log("store_end")
        return {
            "stored": True,
            "mode": "generation_final_exact",
            "reason": "compatible",
            "prefix_len": 3,
            "nbytes": 123,
        }

    real_compat = openai._generation_final_postcommit_compatibility

    def blocking_compat(*args, **kwargs):
        log("store_start")
        release.wait(5.0)
        outcome = real_compat(*args, **kwargs)
        log("store_end")
        return outcome

    def fake_run_generation(_state, prompt_ids, **kwargs):
        runs["n"] += 1
        session = state.sessions.peek(SESSION)
        committed = len(getattr(session, "committed_token_ids", ()) or ())
        log(f"run{runs['n']}:committed={committed}")
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens[:1])
            token_callback(tokens[1:])
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", blocking_store
    )
    monkeypatch.setattr(
        openai, "_generation_final_postcommit_compatibility", blocking_compat
    )
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    body = {
        "messages": [{"role": "user", "content": "Say OK"}],
        "enable_thinking": False,
        "stream": stream,
        "max_tokens": 4,
    }
    with TestClient(create_app(state)) as client:
        timer = threading.Timer(release_after_s, release.set)
        timer.start()
        first = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": SESSION},
            json=body,
        )
        log("response1_done")
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": SESSION},
            json={
                **body,
                "messages": [
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "Again"},
                ],
            },
        )
        log("response2_done")
        release.set()
        timer.cancel()
        tails = getattr(state, "post_response_tails", None)
        if isinstance(tails, openai._PostResponseTails):
            assert tails.wait_idle(5.0)
    return events, first, second, state


def _stream_content(text: str) -> str:
    parts = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: ") :])
        for choice in payload.get("choices") or []:
            parts.append(str((choice.get("delta") or {}).get("content") or ""))
    return "".join(parts)


def _final_stats(text: str) -> dict:
    stats = {}
    for line in text.splitlines():
        if line.startswith("data: {") and '"mtplx_stats"' in line:
            stats = json.loads(line[len("data: ") :])["mtplx_stats"]
    return stats


def test_stream_off_waits_for_postcommit_before_terminal_frame(monkeypatch):
    events, first, second, _state = _scenario(
        monkeypatch, flag=False, stream=True, release_after_s=0.3
    )
    assert first.status_code == 200 and second.status_code == 200
    # Baseline: the first response only ends after its postcommit.
    assert events.index("store_end") < events.index("response1_done")
    assert _final_stats(first.text)["session_postcommit_snapshot"]["stored"] is True


def test_stream_on_ends_response_before_postcommit_and_next_turn_waits(monkeypatch):
    events, first, second, state = _scenario(
        monkeypatch, flag=True, stream=True, release_after_s=0.3
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert "already in flight" not in second.text
    # The response ended while the postcommit was still blocked ...
    assert events.index("response1_done") < events.index("store_end")
    # ... and the second turn only ran once the commit had fully landed,
    # with the first turn's frontier committed.
    run2 = next(e for e in events if e.startswith("run2:"))
    assert events.index("store_end") < events.index(run2)
    assert run2 != "run2:committed=0"
    stats = _final_stats(first.text)
    assert stats["session_postcommit_snapshot"] == openai._POST_RESPONSE_SNAPSHOT_MARKER
    assert state.post_response_tails.to_admin_dict()["active"] == 0
    assert state.post_response_tails.completed >= 1


def test_stream_text_is_identical_with_and_without_flag(monkeypatch):
    _e_off, first_off, second_off, _s = _scenario(
        monkeypatch, flag=False, stream=True, release_after_s=0.05
    )
    _e_on, first_on, second_on, _s = _scenario(
        monkeypatch, flag=True, stream=True, release_after_s=0.05
    )
    assert _stream_content(first_off.text) == _stream_content(first_on.text) == "OK"
    assert _stream_content(second_off.text) == _stream_content(second_on.text)


def test_nonstream_on_returns_body_before_postcommit_and_next_turn_waits(
    monkeypatch,
):
    events, first, second, state = _scenario(
        monkeypatch, flag=True, stream=False, release_after_s=0.3
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert events.index("response1_done") < events.index("store_end")
    assert "job:postcommit.after_response" in events
    run2 = next(e for e in events if e.startswith("run2:"))
    assert events.index("store_end") < events.index(run2)
    body = first.json()
    assert (
        body["mtplx_stats"]["session_postcommit_snapshot"]
        == openai._POST_RESPONSE_SNAPSHOT_MARKER
    )
    assert state.post_response_tails.to_admin_dict()["active"] == 0


def test_nonstream_text_is_identical_with_and_without_flag(monkeypatch):
    _e_off, first_off, second_off, _s = _scenario(
        monkeypatch, flag=False, stream=False, release_after_s=0.05
    )
    _e_on, first_on, second_on, _s = _scenario(
        monkeypatch, flag=True, stream=False, release_after_s=0.05
    )
    off = first_off.json()["choices"][0]["message"]
    on = first_on.json()["choices"][0]["message"]
    assert off == on
    assert (
        second_off.json()["choices"][0]["message"]
        == second_on.json()["choices"][0]["message"]
    )
