"""The window clients configure: the one this server executes.

The app wrote 262,144 into Pi's contextWindow and maxTokens from settings on
2026-09-29 while the engine could not serve that. /health now publishes
``execution_window``, from one function, and the app configures Pi and
OpenCode from it.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

from mtplx.memory_plan import MemoryPlan
from mtplx.server.served_window import answer_share_tokens, served_execution_window


def _state(*, window, fit=None, available=True, allow_swap=False):
    plan = None
    if fit is not None or not available:
        plan = MemoryPlan(available=available, context_window_fit=int(fit or 0))
    return SimpleNamespace(context_window=window, memory_plan=plan, allow_swap=allow_swap)


def test_an_explicit_window_above_the_machine_fit_is_not_what_clients_get():
    served = served_execution_window(_state(window=262_144, fit=98_304))
    assert served["tokens"] == 98_304
    assert served["answer_tokens"] == 49_152
    assert served["basis"] == "machine_fit"
    assert served["configured_tokens"] == 262_144
    assert served["machine_fit_tokens"] == 98_304


def test_a_window_inside_the_fit_is_served_as_configured():
    served = served_execution_window(_state(window=131_072, fit=262_144))
    assert served["tokens"] == 131_072
    assert served["answer_tokens"] == 65_536
    assert served["basis"] == "configured_window"


def test_allow_swap_keeps_the_operators_window():
    served = served_execution_window(_state(window=262_144, fit=98_304, allow_swap=True))
    assert served["tokens"] == 262_144
    assert served["allow_swap"] is True


def test_an_unavailable_plan_leaves_the_resolved_window():
    served = served_execution_window(_state(window=65_536, available=False))
    assert served["tokens"] == 65_536
    assert served["machine_fit_tokens"] is None
    served = served_execution_window(SimpleNamespace(context_window=32_768))
    assert served["tokens"] == 32_768 and served["answer_tokens"] == 16_384


def test_answer_share_is_half_the_window_at_the_boundaries():
    assert answer_share_tokens(4_096) == 2_048
    assert answer_share_tokens(32_768) == 16_384
    assert answer_share_tokens(262_144) == 131_072
    assert answer_share_tokens(1) == 1


def test_health_publishes_the_execution_window():
    from fastapi.testclient import TestClient
    from test_server_openai import _fake_state
    from mtplx.server.openai import create_app

    state = _fake_state()
    state.context_window = 262_144
    state.memory_plan = MemoryPlan(available=True, context_window_fit=98_304)
    payload = TestClient(create_app(state)).get("/health").json()
    assert payload["context_window"] == 262_144
    assert payload["execution_window"]["tokens"] == 98_304
    assert payload["execution_window"]["answer_tokens"] == 49_152
    assert payload["execution_window"]["basis"] == "machine_fit"
