"""Short-request priority: admission order between whole foreground items."""

from __future__ import annotations

import time
from threading import Event, Lock, Thread
from types import SimpleNamespace

from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai
from mtplx.server.openai import parse_args


def _block_owner(scheduler: ModelWorkScheduler, order: list[str]):
    started = Event()
    release = Event()

    def running() -> None:
        order.append("running")
        started.set()
        assert release.wait(timeout=5)

    future = scheduler.submit_foreground(running)
    assert started.wait(timeout=5)
    return future, release


def test_priority_item_runs_before_queued_plain_foreground():
    scheduler = ModelWorkScheduler(name="test-priority", idle_grace_s=0.0)
    order: list[str] = []
    try:
        running, release = _block_owner(scheduler, order)
        long_turn = scheduler.submit_foreground(lambda: order.append("long"))
        short = scheduler.submit_priority_foreground(lambda: order.append("short"))
        assert scheduler.stats()["priority_pending"] == 1
        release.set()
        for future in (running, long_turn, short):
            future.result(timeout=5)
        # The running item is never interrupted; the short one goes next.
        assert order == ["running", "short", "long"]
        assert scheduler.stats()["priority_passed"] == 1
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_priority_items_keep_fifo_among_themselves():
    scheduler = ModelWorkScheduler(name="test-priority", idle_grace_s=0.0)
    order: list[str] = []
    try:
        running, release = _block_owner(scheduler, order)
        futures = [
            scheduler.submit_priority_foreground(lambda v=v: order.append(v))
            for v in ("s1", "s2", "s3")
        ]
        release.set()
        for future in [running, *futures]:
            future.result(timeout=5)
        assert order == ["running", "s1", "s2", "s3"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_priority_streak_cannot_starve_plain_foreground():
    scheduler = ModelWorkScheduler(
        name="test-priority", idle_grace_s=0.0, priority_max_streak=2
    )
    order: list[str] = []
    try:
        running, release = _block_owner(scheduler, order)
        long_turn = scheduler.submit_foreground(lambda: order.append("long"))
        shorts = [
            scheduler.submit_priority_foreground(lambda v=v: order.append(v))
            for v in ("s1", "s2", "s3", "s4")
        ]
        release.set()
        for future in [running, long_turn, *shorts]:
            future.result(timeout=5)
        assert order == ["running", "s1", "s2", "long", "s3", "s4"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_priority_counts_as_foreground_for_idle_work():
    scheduler = ModelWorkScheduler(name="test-priority", idle_grace_s=0.0)
    order: list[str] = []
    try:
        running, release = _block_owner(scheduler, order)
        idle = scheduler.submit_idle_postcommit(lambda: order.append("idle"))
        short = scheduler.submit_priority_foreground(lambda: order.append("short"))
        assert scheduler.foreground_pending() == 1
        assert scheduler.foreground_busy()
        release.set()
        for future in (running, short, idle):
            future.result(timeout=5)
        assert order == ["running", "short", "idle"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_priority_max_streak_env(monkeypatch):
    monkeypatch.setenv("MTPLX_SCHEDULER_PRIORITY_MAX_STREAK", "7")
    scheduler = ModelWorkScheduler(name="test-priority")
    try:
        assert scheduler.stats()["priority_max_streak"] == 7
    finally:
        scheduler.shutdown(wait=True)


def test_is_short_request_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SHORT_REQUEST_PRIORITY", raising=False)
    assert openai._is_short_request([1, 2, 3], 1) is False


def test_is_short_request_limits(monkeypatch):
    monkeypatch.setenv("MTPLX_SHORT_REQUEST_PRIORITY", "1")
    monkeypatch.setenv("MTPLX_SHORT_REQUEST_MAX_TOKENS", "40")
    monkeypatch.setenv("MTPLX_SHORT_REQUEST_MAX_PROMPT_TOKENS", "8")
    assert openai._is_short_request([1] * 8, 40) is True
    assert openai._is_short_request([1] * 8, 41) is False
    assert openai._is_short_request([1] * 9, 1) is False
    assert openai._is_short_request([1] * 8, None) is False


def _serial_state():
    return SimpleNamespace(
        args=parse_args(["--warmup-tokens", "0"]),
        model_scheduler=ModelWorkScheduler(name="test-serial-priority"),
        runtime=SimpleNamespace(mtp_enabled=True),
    )


def _wait_until_admitted(scheduler: ModelWorkScheduler, *, queued: int) -> None:
    """The busy item is running and ``queued`` items wait behind it."""
    deadline = time.monotonic() + 5
    while not (
        scheduler.stats()["active_batch_key"] == "test.busy"
        and scheduler.foreground_pending() == queued
    ):
        assert time.monotonic() < deadline
        time.sleep(0.005)


def _dispatch_concurrently(state, requests, monkeypatch):
    """Dispatch ``requests`` (name, prompt_ids, max_tokens) while the owner
    is busy, in a fixed arrival order; return the execution order and the
    per-request results."""
    gate = Event()
    order: list[str] = []
    order_lock = Lock()

    def fake_generation(_state, prompt_ids, **kwargs):
        name = kwargs["request_observability"]["request_id"]
        if name == "busy":
            assert gate.wait(timeout=5)
        with order_lock:
            order.append(name)
        # Output depends only on the request itself, as in real serial mode.
        return {"text": f"{name}:{sum(prompt_ids)}:{kwargs['max_tokens']}"}

    monkeypatch.setattr(openai, "_run_generation", fake_generation)
    results: dict[str, dict] = {}

    def dispatch(name, prompt_ids, max_tokens):
        results[name] = openai._run_generation_dispatched(
            state,
            prompt_ids,
            batch_key=f"test.{name}",
            response_id=name,
            generation_mode="mtp",
            max_tokens=max_tokens,
            request_observability={},
        )

    threads = []
    try:
        for name, prompt_ids, max_tokens in [("busy", [1], 512), *requests]:
            thread = Thread(target=dispatch, args=(name, prompt_ids, max_tokens))
            thread.start()
            threads.append(thread)
            _wait_until_admitted(state.model_scheduler, queued=len(threads) - 1)
        gate.set()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        gate.set()
        state.model_scheduler.shutdown(wait=False, cancel_futures=True)
    return order, results


_REQUESTS = [
    ("agent-1", list(range(100)), 512),
    ("agent-2", list(range(200)), 512),
    ("title", [5, 6, 7], 40),
]


def test_serial_dispatch_is_fifo_when_switch_is_off(monkeypatch):
    monkeypatch.delenv("MTPLX_SHORT_REQUEST_PRIORITY", raising=False)
    order, _ = _dispatch_concurrently(_serial_state(), _REQUESTS, monkeypatch)
    assert order == ["busy", "agent-1", "agent-2", "title"]


def test_serial_dispatch_runs_short_request_first_without_changing_output(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SHORT_REQUEST_PRIORITY", raising=False)
    _, baseline = _dispatch_concurrently(_serial_state(), _REQUESTS, monkeypatch)
    monkeypatch.setenv("MTPLX_SHORT_REQUEST_PRIORITY", "1")
    order, results = _dispatch_concurrently(_serial_state(), _REQUESTS, monkeypatch)
    assert order == ["busy", "title", "agent-1", "agent-2"]
    assert results == baseline
