"""The OS footprint floor charges only what a seat does not normally hold.

PR #500 (Maikel Vos) added the probe and floored every memory guard with
``max(active + cache, phys_footprint)``. The probe is right. The comparison
gave the process no room at all outside MLX's own account, and the allocator
limit is not the process's budget: the memory plan fits weights, KV and the
session cache inside the limit (75% of RAM by default) and leaves the rest
to macOS and to what the process holds outside Metal.

The 2026-09-16 review worked the seats: a 48 GB Mac with the 27B has a 36
GiB limit, a session the plan sized to fit sits near it, and 2 to 3 GiB of
ordinary host memory on top reads 1.06 to 1.08 of the limit. That is
CRITICAL at rest, which empties the warm session cache and arms the prefill
abort. These tests pin the re-based floor and keep the strict one reachable.

2026-09-27: the allowance is the fixed measured floor (8 GiB). It used to be
the larger of that and RAM - system reserve - limit, which kept the guard's
process ceiling at RAM minus the reserve whatever the limit (112 GiB on a
128 GB Mac at 96, 90 or 88 GiB): lowering MTPLX_MEMORY_LIMIT_BYTES did not
lower it, and 14 GiB of leaked host memory (#546) was forgiven up to 16 GiB.
The seats that motivated the old rule (48 GB / 27B, 96 GB / Flash-Next) stay
quiet on the floor alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv

GIB = 1024**3


def _state(*, total_gib: float, limit_gib: float, budget_gib: float | None = None):
    return SimpleNamespace(
        metal_memory_caps={
            "memory_limit_bytes": int(limit_gib * GIB),
            "total_ram_bytes": int(total_gib * GIB),
        },
        memory_budget_bytes=None if budget_gib is None else int(budget_gib * GIB),
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=24576,
            aux_bytes_per_token=7872,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
    )


def _pin(monkeypatch, *, allocator_gib: float, footprint_gib: float | None):
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(allocator_gib * GIB),
            "cache_memory_bytes": 0,
        },
    )
    monkeypatch.setattr(
        srv,
        "phys_footprint_bytes",
        lambda *a, **k: None if footprint_gib is None else int(footprint_gib * GIB),
    )


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)


# --------------------------------------------------------------------------
# The allowance is one measured floor, whatever the seat
# --------------------------------------------------------------------------


def test_the_allowance_is_the_measured_floor():
    assert srv._host_memory_allowance_bytes() == 8 * GIB


def test_an_explicit_allowance_wins(monkeypatch):
    monkeypatch.setenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", "12G")
    assert srv._host_memory_allowance_bytes() == 12 * GIB


@pytest.mark.parametrize("limit_gib", [96, 90, 88])
def test_the_process_ceiling_follows_the_limit(monkeypatch, limit_gib):
    """The footprint that first reads CRITICAL is 1.02 x limit + 8 GiB.

    Before, the allowance grew by exactly what the limit shrank, so that
    footprint stayed at 1.02 x limit + (128 - 16 - limit) GiB: 113.9 GiB at
    96, 90 and 88 alike.
    """

    state = _state(total_gib=128, limit_gib=limit_gib)
    ceiling = int(1.02 * limit_gib * GIB) + 8 * GIB
    _pin(monkeypatch, allocator_gib=60, footprint_gib=(ceiling - GIB) / GIB)
    assert srv._allocator_pressure_level(state)[0] < 4
    _pin(monkeypatch, allocator_gib=60, footprint_gib=(ceiling + GIB) / GIB)
    assert srv._allocator_pressure_level(state)[0] == 4


def test_lowering_the_limit_makes_the_same_process_read_higher(monkeypatch):
    """One process, 80 GiB in MLX's account and a 102 GiB footprint.

    Old rule: the allowance was 16 GiB at a 96 GiB limit and 22 GiB at 90,
    so both limits read level 1 (86/96 and 80/90). Now the 14 GiB charged
    reads WARNING at 96 and CRITICAL at 90.
    """

    _pin(monkeypatch, allocator_gib=80, footprint_gib=102)
    at_96 = srv._allocator_pressure_level(_state(total_gib=128, limit_gib=96))
    at_90 = srv._allocator_pressure_level(_state(total_gib=128, limit_gib=90))
    assert at_96 == (2, pytest.approx(94 / 96))
    assert at_90 == (4, pytest.approx(94 / 90))


# --------------------------------------------------------------------------
# The pressure level
# --------------------------------------------------------------------------


def test_a_plan_sized_session_on_a_48gb_mac_is_not_critical(monkeypatch):
    """The review's seat. 34 GiB in MLX's account (0.944 of the 36 GiB
    limit) and 3 GiB of ordinary host memory on top."""

    state = _state(total_gib=48, limit_gib=36)
    _pin(monkeypatch, allocator_gib=34, footprint_gib=37)

    level, fraction = srv._allocator_pressure_level(state)
    assert level == 1
    assert fraction == pytest.approx(34 / 36)

    # The strict floor reads the same healthy session as CRITICAL.
    monkeypatch.setenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", "0")
    level, fraction = srv._allocator_pressure_level(state)
    assert level == 4
    assert fraction == pytest.approx(37 / 36)


def test_flash_next_on_96gb_does_not_sit_in_warning(monkeypatch):
    state = _state(total_gib=96, limit_gib=84)
    _pin(monkeypatch, allocator_gib=80, footprint_gib=85)  # 5 GiB of host memory

    level, fraction = srv._allocator_pressure_level(state)
    assert level == 1
    assert fraction == pytest.approx(80 / 84)


def test_footprint_beyond_the_allowance_is_charged(monkeypatch):
    state = _state(total_gib=128, limit_gib=96)  # allowance 8 GiB

    _pin(monkeypatch, allocator_gib=80, footprint_gib=90)  # overhang 10, charged 2
    level, fraction = srv._allocator_pressure_level(state)
    assert level == 1
    assert fraction == pytest.approx(82 / 96)

    # Overhang 26, charged 18. Under the old 16 GiB allowance only 10 was
    # charged and this read level 1 (90/96): the leak was forgiven.
    _pin(monkeypatch, allocator_gib=80, footprint_gib=106)
    level, fraction = srv._allocator_pressure_level(state)
    assert level == 4
    assert fraction == pytest.approx(98 / 96)


def test_a_failed_probe_changes_nothing(monkeypatch):
    state = _state(total_gib=128, limit_gib=96)
    _pin(monkeypatch, allocator_gib=94, footprint_gib=None)

    level, fraction = srv._allocator_pressure_level(state)
    assert level == 2
    assert fraction == pytest.approx(94 / 96)


def test_a_footprint_below_the_allocators_account_changes_nothing(monkeypatch):
    # File-backed weights that were never touched are in active, not resident.
    state = _state(total_gib=128, limit_gib=96)
    _pin(monkeypatch, allocator_gib=94, footprint_gib=60)

    live, fields = srv._footprint_floor(state, limit=96 * GIB, allocator_bytes=94 * GIB)
    assert live == 94 * GIB
    assert fields["host_overhang_bytes"] == 0
    assert fields["host_overhang_charged_bytes"] == 0


# --------------------------------------------------------------------------
# The admission shed
# --------------------------------------------------------------------------


class _EmptyBank:
    total_nbytes = 0

    def longest_prefix(self, token_ids):
        return None


def test_admission_does_not_refuse_a_request_the_plan_sized_to_fit(monkeypatch):
    state = _state(total_gib=48, limit_gib=36)
    _pin(monkeypatch, allocator_gib=28, footprint_gib=38)  # overhang 10, charged 2

    # 8,192 new tokens: 0.25 GiB of KV plus 3 GiB of transients on 30 GiB.
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=list(range(8192)), session_bank=_EmptyBank(), session_id="pi"
    )
    assert receipt is None


def test_the_admission_receipt_explains_what_it_charged(monkeypatch):
    state = _state(total_gib=128, limit_gib=96)
    _pin(monkeypatch, allocator_gib=70, footprint_gib=110)  # overhang 40, charged 32

    receipt = srv._prefill_admission_shed(
        state, prompt_ids=list(range(40_000)), session_bank=_EmptyBank(), session_id="pi"
    )
    assert receipt is not None
    assert receipt["phys_footprint_bytes"] == 110 * GIB
    assert receipt["host_overhang_bytes"] == 40 * GIB
    assert receipt["host_allowance_bytes"] == 8 * GIB
    assert receipt["host_overhang_charged_bytes"] == 32 * GIB
    assert receipt["active_bytes"] == 70 * GIB
    # Still over the limit after reclamation, on the footprint alone.
    assert receipt["refused"] is True
    assert receipt["host_overhang_charged_bytes_after"] == 32 * GIB
