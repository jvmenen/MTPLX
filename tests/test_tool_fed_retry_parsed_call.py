"""A well-formed bare tool call after tool results is a call, not orphan markup.

Before the fix, a call whose arguments strip to one short token
(``count_lines`` with ``part=2``) was classified ``orphan_tool_control_markup``
and the stream worker re-ran the turn with a nudge message, re-prefilling the
history. These tests pin the exemption, its kill switch, and that genuine
orphan tails still retry.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from tests.test_server_openai import _fake_streaming_session_state, _stream_payloads

BARE_CALL = (
    "\n</think>\n\n<tool_call>\n<function=count_lines>\n<parameter=part>\n2\n"
    "</parameter>\n</function>\n</tool_call>"
)
ORPHAN_TAIL = "parameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"
ANSWER = "</think>\n\nPart 2 has 93 lines."


def _count_lines_tool():
    return {
        "type": "function",
        "function": {
            "name": "count_lines",
            "description": "Count the lines in one part of the notes.",
            "parameters": {
                "type": "object",
                "properties": {"part": {"type": "integer"}},
                "required": ["part"],
            },
        },
    }


def _run(monkeypatch, texts):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    calls: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 22.0,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {"role": "user", "content": "Count the lines of each part."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "count_lines",
                                "arguments": '{"part": 1}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Part 1 has 93 lines."},
            ],
            "tools": [_count_lines_tool()],
            "stream": True,
            "max_tokens": 128,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    payloads = _stream_payloads(body)
    final = [p for p in payloads if p["choices"][0]["finish_reason"]]
    return calls, payloads, final[-1]


def _tool_call_names(payloads):
    return [
        (call.get("function") or {}).get("name")
        for payload in payloads
        for choice in payload.get("choices", [])
        for call in (choice.get("delta") or {}).get("tool_calls") or []
        if (call.get("function") or {}).get("name")
    ]


def test_heuristic_alone_flags_bare_call_as_orphan():
    # The pre-existing heuristic is unchanged; the exemption sits beside it.
    assert (
        openai._tool_fed_degenerate_completion_reason(BARE_CALL)
        == "orphan_tool_control_markup"
    )


def test_bare_well_formed_call_is_not_retried(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    calls, payloads, final = _run(monkeypatch, [BARE_CALL, ANSWER])

    assert calls == [BARE_CALL]
    assert _tool_call_names(payloads) == ["count_lines"]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert not final["mtplx_stats"].get("tool_fed_empty_retry_attempted")


def test_kill_switch_restores_retry(monkeypatch):
    monkeypatch.setenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", "0")
    calls, _payloads, final = _run(monkeypatch, [BARE_CALL, ANSWER])

    assert calls == [BARE_CALL, ANSWER]
    assert final["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
    assert final["mtplx_stats"]["tool_fed_empty_retry_reason"] == (
        "orphan_tool_control_markup"
    )


def test_orphan_tail_without_call_still_retries(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    calls, _payloads, final = _run(monkeypatch, [ORPHAN_TAIL, ANSWER])

    assert calls == [ORPHAN_TAIL, ANSWER]
    assert final["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
