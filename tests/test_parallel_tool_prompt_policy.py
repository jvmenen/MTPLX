"""Declarative parallel-tool routing is shared by generation and counting."""

import pytest
from test_server_openai import _fake_state, _named_tool_schema

from mtplx.server.openai import ChatCompletionRequest, ChatMessage
from mtplx.server.request_policy import resolve_request_policy


@pytest.mark.parametrize("endpoint", ["chat", "count_tokens"])
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
def test_parallel_policy_keeps_generation_and_postcommit_in_the_same_mode(
    monkeypatch, endpoint, tool_choice
):
    monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)
    state = _fake_state()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Read both files.")],
        tools=[_named_tool_schema("read")],
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )
    policy = resolve_request_policy(
        state,
        request,
        headers={"x-mtplx-client": "pi"},
        metadata={},
        endpoint=endpoint,
    )
    assert policy.template_tool_prompt_mode == "native"
    assert policy.postcommit_tool_prompt_mode == "native"
