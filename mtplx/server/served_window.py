"""The context window this server can execute, as clients should see it.

The app used to write the window from its settings (262,144) into Pi and
OpenCode even when the engine could not serve a conversation that long
(2026-09-29). ``served_execution_window`` is the one place that answers two
questions for every client: how long a conversation, prompt plus answer, this
server executes, and how much of it one answer may claim. ``/health``
publishes the answer as ``execution_window`` and the app configures Pi and
OpenCode from it.

Today the answer comes from what the server already computes: the resolved
serving window, bounded by the memory planner's machine fit
(``memory_plan.plan_memory``) unless the operator chose ``--allow-swap``. A
qualified per-machine profile can replace the body of this function later
without touching any client.
"""

from __future__ import annotations

from typing import Any


def answer_share_tokens(window_tokens: int) -> int:
    """The longest answer a client should plan for inside ``window_tokens``.

    Half the window. A client that advertises the whole window as its output
    ceiling asks the engine to plan an answer as long as the conversation
    itself while the history is still short, and leaves no room for the
    history once the answer arrives. Half keeps answers far longer than real
    coding answers on large windows (131,072 tokens of 262,144; the longest
    answers measured on the founder's workload are 35,000 to 47,000 tokens)
    and matches the 16,384 Pi itself assumes on a 32,768 window. Clients
    still clamp each request to the room its prompt leaves (Pi: window minus
    its prompt estimate minus 4,096), so prompt plus answer stays inside the
    window as the history grows.

    The app writes this pair into Pi and recognises its own pair by it
    (``PiIntegration.windowFieldsWereWrittenByMTPLX`` and
    ``ClientContextBudget.answerTokens`` in the Swift package): change both
    sides together.
    """

    return max(1, int(window_tokens) // 2)


def served_execution_window(state: Any) -> dict[str, Any]:
    """The conversation length this server executes, and the answer share.

    ``tokens`` is the resolved window (``--context-window``, else the plan's
    default) bounded by the memory plan's machine fit, the largest window
    whose state fits this Mac's engine budget. An explicit window above the
    fit is still accepted by the server (it warns and sheds caches), but it
    is not a length clients should plan to use. ``--allow-swap`` means the
    operator accepts swap past the fit, so the resolved window stands.
    """

    configured = max(0, int(getattr(state, "context_window", 0) or 0))
    allow_swap = bool(getattr(state, "allow_swap", False))
    plan = getattr(state, "memory_plan", None)
    fit = 0
    if plan is not None and bool(getattr(plan, "available", False)):
        fit = max(0, int(getattr(plan, "context_window_fit", 0) or 0))
    tokens = configured
    basis = "configured_window"
    if configured > 0 and fit > 0 and fit < configured and not allow_swap:
        tokens = fit
        basis = "machine_fit"
    return {
        "tokens": int(tokens),
        "answer_tokens": answer_share_tokens(tokens) if tokens > 0 else 0,
        "basis": basis,
        "configured_tokens": int(configured),
        "machine_fit_tokens": int(fit) if fit > 0 else None,
        "allow_swap": allow_swap,
    }
