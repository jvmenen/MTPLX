"""What the rest of the Mac has left: memory the kernel can hand out without compressing.

The allocator guard measures the engine against its own Metal limit. That limit
is sized from total RAM when the daemon starts, so it says nothing about the
other apps on the desktop. A long cold prefill adds 10 to 15 GiB on top of the
resident weights. When an editor and two browsers already hold the rest, macOS
compresses and swaps the desktop until the UI stops answering, and nothing in
the engine notices: the engine itself is still under its limit.

Receipt (2026-09-19): a 129,050-token Pi compaction prompt, fresh daemon, 128 GB
Mac with a game editor open. The Mac froze about a minute into the prefill and
needed a hard power-off. The same prefill on a quiet desktop wires about 100 GB.

The supply figure is read from ``host_statistics64(HOST_VM_INFO64)``: free pages
(the kernel's free count already includes speculative pages), purgeable pages,
and the file-backed pages that are not speculative. Speculative pages are
counted in both the free and the file-backed counters, so they are taken
once. File-backed pages are credited in full: the kernel reclaims clean file
pages without compressing anything, and ``vm_statistics64`` has no counter for
the dirty ones, which must be written back first (the unified buffer cache
flushes them within about 30 s, and the pages MTPLX itself maps, weights and
the n-gram table, are read-only and clean). The floors below are the margin
for that dirty or busy share, and the death signature catches a Mac whose
file cache is large while its reclaim stalls anyway.
``kern.memorystatus_level`` is kept only as a coarse, reported signal: on macOS
it is ``(total - wired - compressor) / total`` (this Mac, 2026-09-27: level 96
with 3.85 GiB wired and 0.02 GiB compressed; the 2026-09-26 field report: 28
percent at 88 GiB wired + 3.7 GiB compressed, 17 percent at 92.2 + 14.7), so it
counts every app's anonymous memory as available. In that report it read 17
percent while free pages sat at 0.1 to 0.5 GiB and the compressor grew 10 GiB
in 30 s; the Macs that froze had the same shape.

The floor the Mac needs scales with what is wired: wired pages cannot be
compressed or paged, so the more the GPU holds, the less room the kernel has to
absorb a burst. A 128 GB Mac with 83 GiB wired did not recover from 3 GiB free
(the field report's freezes), and its last recorded state before freeze 3 had
5.7 GiB free two seconds before the machine stopped.

Everything here degrades to "unknown" and never to a refusal: a guard that
cannot read the machine takes no action.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

GIB = 1024**3
MIB = 1024**2

# Below the abort floor the desktop is one allocation away from a swap storm.
# Three terms, the largest wins: 1 GiB (a small Mac's absolute minimum), 2.5
# percent of RAM (the shipped 2026-09-20 rule), and a sixteenth of what is
# wired system-wide. The wired term is the new one: 83 GiB wired did not
# recover from 3 GiB free, and a sixteenth of it is 5.2 GiB. 16 GB Mac: 1 GiB;
# 48 GB Mac with the 27B (about 21 GiB wired): 1.3 GiB; 128 GB Mac with
# Flash-Next (about 88 GiB wired): 5.5 GiB. The shed floor is twice that: the
# engine gives back its own reusable memory (allocator pool, idle session
# snapshots) before anyone is refused.
_ABORT_FLOOR_FRACTION = 0.025
_ABORT_FLOOR_MIN_BYTES = 1 * GIB
_ABORT_FLOOR_WIRED_DIVISOR = 16
_SHED_FLOOR_MULTIPLE = 2

# The death signature from the machine's own crash receipts (2026-09-03 and
# 2026-09-23): free pages near zero while the compressor or swap grows, with
# the pressure level still reading normal. Between two readings, free pages
# under the abort floor together with compressor growth at 256 MiB/s or swap
# growth at 64 MiB/s is treated as critical. Growth under 256 MiB between two
# readings (compressor or swap) is never counted, so a single page-out burst
# cannot trip it. The compaction the report's Mac survived compressed about
# 10 GiB in 30 to 60 s (170 to 340 MiB/s) at free pages of 0.1 to 0.5 GiB: at
# this line. It reads net occupancy: churn that compresses and decompresses at
# a steady total is not seen here, and falls to the supply floors instead.
# The rates come from that one Mac; other sizes are not calibrated.
_THRASH_COMPRESSOR_BYTES_PER_S = 256 * MIB
_THRASH_SWAP_BYTES_PER_S = 64 * MIB
_THRASH_MIN_GROWTH_BYTES = 256 * MIB
# Compressor growth alone is not the signature while the kernel still has a
# large clean file cache to drop (supply at or above the shed floor) and its
# free pages are not starved. macOS 27 compresses other apps' idle pages
# early and runs with fewer free pages: on 2026-09-29 (E1, 128 GB, Flash-Next,
# macOS 27.0.1) a healthy 65K prefill was aborted twice as "death_signature"
# at 2.4 and 1.45 GB free with 25.0 and 23.2 GB of file-backed pages, the
# compressor growing 0.8 GB in 2.8 s and swap flat. Every recorded death
# shape stays caught: the freezes and the 09-23 panic had free pages at 0.0
# to 0.5 GiB (under this line) or little file cache (supply under the shed
# floor, where compressor growth counts as before), and swap growth counts at
# the abort floor whatever the file cache. 512 MiB is eight times the
# kernel's own free-page target on a 16 KiB-page Mac (vm_page_free_target
# 4,000 pages), where reclaim is not yet under way.
_THRASH_STARVED_FREE_BYTES = 512 * MIB
# Growth is measured from every reading of the last ten seconds (and the
# newest one before them), not only from the previous reading: the per-chunk
# check reads every 0.2 s, and 320 MiB/s arriving in 80 MiB steps never grew
# 256 MiB between two adjacent readings (the 2026-09-27 review of this
# change). Ten seconds holds the guard loop's own 10 s tick.
_THRASH_WINDOW_S = 10.0


@dataclass(frozen=True)
class SystemMemory:
    """One reading of the kernel's memory accounting.

    ``available_bytes`` is the supply: what the kernel can hand out without
    compressing or swapping (free, purgeable and file-backed pages).
    ``free_bytes`` is the part that needs no reclaim at all (free, which
    includes speculative, plus purgeable). ``level_percent`` is
    ``kern.memorystatus_level``, reported for comparison only.
    """

    available_bytes: int
    total_bytes: int
    level_percent: int
    free_bytes: int | None = None
    file_backed_bytes: int | None = None
    wired_bytes: int = 0
    compressor_bytes: int | None = None
    swap_used_bytes: int | None = None
    monotonic_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_bytes": int(self.available_bytes),
            "free_bytes": None if self.free_bytes is None else int(self.free_bytes),
            "file_backed_bytes": (
                None if self.file_backed_bytes is None else int(self.file_backed_bytes)
            ),
            "wired_bytes": int(self.wired_bytes),
            "compressor_bytes": (
                None if self.compressor_bytes is None else int(self.compressor_bytes)
            ),
            "swap_used_bytes": (
                None if self.swap_used_bytes is None else int(self.swap_used_bytes)
            ),
            "total_bytes": int(self.total_bytes),
            "memorystatus_level": int(self.level_percent),
        }


def system_memory_guard_enabled() -> bool:
    raw = os.environ.get("MTPLX_SYSTEM_MEMORY_GUARD", "1").strip().lower()
    return raw not in {"0", "off", "false", "no"}


@functools.cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.mach_host_self.restype = ctypes.c_uint32
    return libc


_HOST_PORT_LOCK = threading.Lock()
_HOST_PORT: int | None = None


def _host_port() -> int:
    # One send right for the process's lifetime: every mach_host_self() call
    # adds a user reference to the host port, and the guard reads the machine
    # every few seconds for as long as the daemon runs. The lock keeps two
    # first calls on different threads from taking two.
    global _HOST_PORT
    with _HOST_PORT_LOCK:
        if _HOST_PORT is None:
            _HOST_PORT = int(_libc().mach_host_self())
        return _HOST_PORT


def _sysctl_int(name: bytes, width: int) -> int | None:
    value = ctypes.c_uint64(0) if width == 8 else ctypes.c_int32(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = _libc().sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0)
    if rc != 0:
        return None
    return int(value.value)


class _VMStatistics64(ctypes.Structure):
    """``struct vm_statistics64`` from ``<mach/vm_statistics.h>`` (rev2, 160 bytes).

    ``host_statistics64`` is told the buffer size in 32-bit words, so a
    shorter struct is filled only as far as it reaches, never past its end.
    """

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ("swapped_count", ctypes.c_uint64),
    ]


class _XswUsage(ctypes.Structure):
    """``struct xsw_usage`` behind ``sysctl vm.swapusage``."""

    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]


_HOST_VM_INFO64 = 4


def _read_vm_statistics() -> tuple[_VMStatistics64, int] | None:
    libc = _libc()
    host = _host_port()
    info = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(_VMStatistics64) // 4)
    rc = libc.host_statistics64(
        ctypes.c_uint32(host),
        _HOST_VM_INFO64,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if rc != 0:
        return None
    page = ctypes.c_size_t(0)
    if libc.host_page_size(ctypes.c_uint32(host), ctypes.byref(page)) != 0:
        return None
    if int(page.value) <= 0:
        return None
    return info, int(page.value)


def _read_swap_used_bytes() -> int | None:
    usage = _XswUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    rc = _libc().sysctlbyname(
        b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0
    )
    if rc != 0:
        return None
    return int(usage.xsu_used)


def _supply_from_statistics(info: _VMStatistics64, page: int) -> tuple[int, int]:
    """(free_bytes, file_backed_bytes) from one ``vm_statistics64`` reading.

    ``free_count`` already includes the speculative pages, and so does
    ``external_page_count`` (a speculative page is a file page read ahead):
    the file-backed credit leaves them out so they count once.
    """

    speculative = int(info.speculative_count)
    free = (int(info.free_count) + int(info.purgeable_count)) * int(page)
    file_backed = max(0, int(info.external_page_count) - speculative) * int(page)
    return free, file_backed


def _read_kernel() -> SystemMemory | None:
    total = _sysctl_int(b"hw.memsize", 8)
    if total is None or total <= 0:
        return None
    stats = _read_vm_statistics()
    if stats is None:
        return None
    info, page = stats
    free, file_backed = _supply_from_statistics(info, page)
    level = _sysctl_int(b"kern.memorystatus_level", 4)
    return SystemMemory(
        available_bytes=min(int(total), free + file_backed),
        total_bytes=int(total),
        level_percent=int(level) if level is not None and 0 <= level <= 100 else -1,
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=int(info.wire_count) * page,
        compressor_bytes=int(info.compressor_page_count) * page,
        swap_used_bytes=_read_swap_used_bytes(),
        monotonic_s=time.monotonic(),
    )


# Swappable for tests and for the rehearsal switch below; production reads the
# kernel.
_reader: Callable[[], SystemMemory | None] = _read_kernel


def _parse_bytes(raw: str) -> int | None:
    text = raw.strip().upper()
    if not text:
        return None
    scale = 1
    for suffix, factor in (("K", 1024), ("M", 1024**2), ("G", GIB)):
        if text.endswith(suffix + "IB"):
            text, scale = text[:-3], factor
            break
        if text.endswith(suffix + "B"):
            text, scale = text[:-2], factor
            break
        if text.endswith(suffix):
            text, scale = text[:-1], factor
            break
    try:
        return int(float(text) * scale)
    except ValueError:
        return None


def read_system_memory() -> SystemMemory | None:
    """The kernel's reading, or None when it cannot be read or the guard is off.

    ``MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES`` replaces the supply
    figure with a fixed value (free pages are capped at it too) so the shed,
    refuse and abort paths can be exercised on a machine that is not actually
    short of memory.
    """

    if not system_memory_guard_enabled():
        return None
    try:
        reading = _reader()
    except Exception:
        return None
    if reading is None:
        return None
    rehearsal = _parse_bytes(
        os.environ.get("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", "")
    )
    if rehearsal is not None:
        available = max(0, min(int(rehearsal), reading.total_bytes))
        free = reading.free_bytes
        return SystemMemory(
            available_bytes=available,
            total_bytes=reading.total_bytes,
            level_percent=int(available * 100 // reading.total_bytes),
            free_bytes=None if free is None else min(int(free), available),
            file_backed_bytes=reading.file_backed_bytes,
            wired_bytes=reading.wired_bytes,
            compressor_bytes=reading.compressor_bytes,
            swap_used_bytes=reading.swap_used_bytes,
            monotonic_s=reading.monotonic_s,
        )
    return reading


def system_memory_floors(total_bytes: int, wired_bytes: int = 0) -> tuple[int, int]:
    """(shed_floor_bytes, abort_floor_bytes) for a machine of this size and wiring."""

    abort = _parse_bytes(os.environ.get("MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES", ""))
    if abort is None or abort <= 0:
        abort = max(
            _ABORT_FLOOR_MIN_BYTES,
            int(total_bytes * _ABORT_FLOOR_FRACTION),
            max(0, int(wired_bytes)) // _ABORT_FLOOR_WIRED_DIVISOR,
        )
    shed = _parse_bytes(os.environ.get("MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES", ""))
    if shed is None or shed <= 0:
        shed = abort * _SHED_FLOOR_MULTIPLE
    return int(max(shed, abort)), int(abort)


def reading_floors(reading: SystemMemory) -> tuple[int, int]:
    """The floors that apply to one reading (its RAM and its wired memory)."""

    return system_memory_floors(reading.total_bytes, reading.wired_bytes)


def admission_floors(reading: SystemMemory, growth_bytes: int) -> tuple[int, int]:
    """The floors a request is admitted against: the reading's, with the
    request's own growth counted as wired.

    What the engine allocates is wired memory in the kernel's accounting
    (2026-09-27 validation, 128 GB, Flash-Next: wired went from 4.4 GB to
    83.6 GB with the model loaded, swung between 85.9 and 93.2 GB across the
    turns, and fell back to 4.4 GB when the server stopped), and the floors
    grow with what is wired. By the end of a prefill the floor the per-chunk
    check reads has risen by about the growth over 16, so the admission
    holds the same line: a request admitted within that distance of the
    abort floor would otherwise trip late in its own prefill. A reading
    with no wired figure keeps its RAM-share floors.
    """

    wired = max(0, int(reading.wired_bytes or 0))
    if wired > 0:
        wired += max(0, int(growth_bytes))
    return system_memory_floors(reading.total_bytes, wired)


def _grew_fast(now: int | None, then: int | None, elapsed: float, rate: float) -> bool:
    if now is None or then is None or elapsed <= 0:
        return False
    growth = int(now) - int(then)
    return growth >= _THRASH_MIN_GROWTH_BYTES and growth / elapsed >= rate


def thrashing_base(
    reading: SystemMemory | None,
    previous: SystemMemory | Sequence[SystemMemory | None] | None,
) -> SystemMemory | None:
    """The earlier reading the death signature is measured from, or None.

    The signature: free pages under the abort floor while swap grew fast
    since one of the ``previous`` readings (one reading, or several from a
    ``ReadingWindow``), or while the compressor grew fast and either the
    supply is under the shed floor or free pages are starved. Needs the page
    counters; anything missing reads as not thrashing.
    """

    if reading is None or previous is None:
        return None
    if reading.free_bytes is None:
        return None
    shed, abort = reading_floors(reading)
    free = int(reading.free_bytes)
    if free >= abort:
        return None
    # Compression counts when the kernel has little else to reclaim or its
    # free pages are starved; with a large clean file cache it is the
    # kernel's own choice (see _THRASH_STARVED_FREE_BYTES). Swap always does.
    compression_counts = (
        int(reading.available_bytes) < shed or free < _THRASH_STARVED_FREE_BYTES
    )
    earlier = [previous] if isinstance(previous, SystemMemory) else list(previous)
    for base in earlier:
        if base is None or base is reading:
            continue
        elapsed = float(reading.monotonic_s) - float(base.monotonic_s)
        if (
            compression_counts
            and _grew_fast(
                reading.compressor_bytes,
                base.compressor_bytes,
                elapsed,
                _THRASH_COMPRESSOR_BYTES_PER_S,
            )
        ) or _grew_fast(
            reading.swap_used_bytes,
            base.swap_used_bytes,
            elapsed,
            _THRASH_SWAP_BYTES_PER_S,
        ):
            return base
    return None


def memory_thrashing(
    reading: SystemMemory | None,
    previous: SystemMemory | Sequence[SystemMemory | None] | None,
) -> bool:
    """Whether ``reading`` shows the death signature (``thrashing_base``)."""

    return thrashing_base(reading, previous) is not None


class ReadingWindow:
    """The readings of the last ``window_s`` seconds, plus the newest one
    before them, for ``memory_thrashing``. One per reader: the guard loop
    keeps one, and each prefill's per-chunk check keeps its own."""

    def __init__(self, window_s: float = _THRASH_WINDOW_S) -> None:
        self.window_s = float(window_s)
        self._readings: list[SystemMemory] = []

    def readings(self) -> list[SystemMemory]:
        return list(self._readings)

    def add(self, reading: SystemMemory | None) -> None:
        if reading is None:
            return
        self._readings.append(reading)
        now_s = float(reading.monotonic_s)
        # Keep the newest reading older than the window: at the loop's 10 s
        # tick it is the only earlier one.
        while (
            len(self._readings) > 2
            and now_s - float(self._readings[1].monotonic_s) >= self.window_s
        ):
            self._readings.pop(0)


def system_pressure_level(
    reading: SystemMemory | None,
    previous: SystemMemory | Sequence[SystemMemory | None] | None = None,
) -> int:
    """Map a reading onto the guard's scale: 1 normal, 2 warning, 4 critical."""

    if reading is None:
        return 1
    shed_floor, abort_floor = reading_floors(reading)
    if memory_thrashing(reading, previous):
        return 4
    if reading.available_bytes < abort_floor:
        return 4
    if reading.available_bytes < shed_floor:
        return 2
    return 1


def admission_shortfall_bytes(
    reading: SystemMemory | None, *, growth_bytes: int, floor: str
) -> int:
    """How far a request's growth would take the Mac's supply under a floor.

    ``floor="shed"``: a request that would leave less than the shed floor
    runs only after the engine gives back its own reusable memory (the
    allocator pool, then idle session state) and the Mac is read again.
    ``floor="abort"``: one that would still leave less than the abort floor
    after that is refused. The request is charged once, for its own growth:
    the per-chunk supply check and the death signature are what catch an
    estimate that was wrong, not a second copy of the request held in
    reserve (2026-09-27 validation, 128 GB, Flash-Next, 12 GB of apps open:
    a request-sized margin refused an 18K turn that left 9.4 GB free after
    its growth, over a 6.0 GB abort floor). The floors are the ones the Mac
    will have once the growth is wired (``admission_floors``). Zero means it
    fits, and an unreadable machine never reports a shortfall.
    """

    if floor not in {"shed", "abort"}:
        raise ValueError(f"floor must be 'shed' or 'abort', not {floor!r}")
    if reading is None:
        return 0
    growth = max(0, int(growth_bytes))
    shed_floor, abort_floor = admission_floors(reading, growth)
    line = shed_floor if floor == "shed" else abort_floor
    return max(0, growth + int(line) - int(reading.available_bytes))


__all__ = [
    "ReadingWindow",
    "SystemMemory",
    "admission_floors",
    "admission_shortfall_bytes",
    "memory_thrashing",
    "read_system_memory",
    "thrashing_base",
    "reading_floors",
    "system_memory_floors",
    "system_memory_guard_enabled",
    "system_pressure_level",
]
