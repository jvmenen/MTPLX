"""The death signature on macOS 27: early compression of idle pages with a
large clean file cache is not thrash, while every recorded death shape still
trips. Readings are replayed byte for byte where the receipt has them."""

from __future__ import annotations

import pytest

from mtplx import system_memory as sm

GB = 1_000_000_000
GIB = 1024**3
MIB = 1024**2
RAM = 137_438_953_472  # 128 GB


def _reading(
    *,
    free: int,
    file_backed: int,
    wired: int,
    compressor: int,
    swap: int,
    level: int = 20,
    at_s: float = 0.0,
    total: int = RAM,
) -> sm.SystemMemory:
    return sm.SystemMemory(
        available_bytes=free + file_backed,
        total_bytes=total,
        level_percent=level,
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=wired,
        compressor_bytes=compressor,
        swap_used_bytes=swap,
        monotonic_s=at_s,
    )


# E1, 2026-09-29, macOS 27.0.1, 50de43bb serving Flash-Next Optimized Speed:
# a healthy 65,536-token cold prefill aborted as "death_signature" twice
# (bench/e1-base-long-0929-0422/00-base0927/server.log), swap flat both times.
E1_FIRST = (
    _reading(
        free=6_469_861_376,
        file_backed=20_122_599_424,
        wired=96_834_748_416,
        compressor=5_389_058_048,
        swap=274_268_160,
        level=23,
        at_s=0.0,
    ),
    _reading(
        free=2_414_084_096,
        file_backed=24_996_462_592,
        wired=94_539_022_336,
        compressor=6_238_814_208,
        swap=274_268_160,
        level=24,
        at_s=2.915,
    ),
)
E1_SECOND = (
    _reading(
        free=2_058_387_456,
        file_backed=24_076_009_472,
        wired=95_925_157_888,
        compressor=5_661_065_216,
        swap=240_713_728,
        level=24,
        at_s=0.0,
    ),
    _reading(
        free=1_445_068_800,
        file_backed=23_204_462_592,
        wired=95_912_755_200,
        compressor=6_464_536_576,
        swap=240_713_728,
        level=23,
        at_s=2.827,
    ),
)


@pytest.mark.parametrize("pair", [E1_FIRST, E1_SECOND], ids=["e1-2.4GB-free", "e1-1.45GB-free"])
def test_macos27_early_compression_with_a_big_file_cache_is_not_death(pair):
    before, after = pair
    # The readings are what tripped on 50de43bb: free under the abort floor
    # while the compressor grew faster than 256 MiB/s.
    _shed, abort = sm.reading_floors(after)
    assert after.free_bytes < abort
    growth = after.compressor_bytes - before.compressor_bytes
    assert growth / (after.monotonic_s - before.monotonic_s) >= 256 * MIB

    assert not sm.memory_thrashing(after, before)
    assert sm.system_pressure_level(after, previous=before) == 1


def test_macos27_reading_with_swap_growth_still_trips():
    before, after = E1_FIRST
    swapping = _reading(
        free=after.free_bytes,
        file_backed=after.file_backed_bytes,
        wired=after.wired_bytes,
        compressor=after.compressor_bytes,
        swap=before.swap_used_bytes + 512 * MIB,
        at_s=after.monotonic_s,
    )
    assert sm.memory_thrashing(swapping, before)


def test_the_0923_panic_trips_whatever_the_file_cache():
    # Panic memoryStatus 2026-09-23: 878 free pages (14 MB), 57.7 GB
    # compressed, one process at 103 GiB. The file cache was not recorded:
    # both a small and a large one must trip.
    for file_backed in (1 * GIB, 30 * GIB):
        before = _reading(
            free=900 * MIB,
            file_backed=file_backed,
            wired=90 * GIB,
            compressor=int(55.0 * GB),
            swap=2 * GIB,
            at_s=0.0,
        )
        after = _reading(
            free=878 * 16384,
            file_backed=file_backed,
            wired=90 * GIB,
            compressor=int(57.7 * GB),
            swap=2 * GIB,
            at_s=2.0,
        )
        assert sm.memory_thrashing(after, before), file_backed


def test_the_0903_panic_with_the_ngram_pre_read_in_the_file_cache_trips():
    # 2026-09-03: 110 GiB engine budget plus a 23.4 GiB n-gram pre-read (clean
    # file pages) plus a 206K prefill; free RAM 0.0 GB before the watchdog.
    before = _reading(
        free=400 * MIB, file_backed=int(23.4 * GIB), wired=100 * GIB,
        compressor=8 * GIB, swap=0, at_s=0.0,
    )
    after = _reading(
        free=16 * MIB, file_backed=int(23.4 * GIB), wired=100 * GIB,
        compressor=9 * GIB, swap=0, at_s=2.0,
    )
    assert sm.memory_thrashing(after, before)


def test_the_field_report_freeze_shape_trips():
    # 2026-09-26 field report: free 0.1 to 0.5 GiB, 2 GiB of file cache,
    # 92.2 GiB wired, the compressor growing 10 GiB in 30 s.
    before = _reading(
        free=int(0.4 * GIB), file_backed=2 * GIB, wired=int(92.2 * GIB),
        compressor=int(13.4 * GIB), swap=0, at_s=0.0,
    )
    after = _reading(
        free=int(0.3 * GIB), file_backed=2 * GIB, wired=int(92.2 * GIB),
        compressor=int(14.7 * GIB), swap=0, at_s=4.0,
    )
    assert sm.memory_thrashing(after, before)


def test_a_thin_supply_keeps_the_abort_floor_sensitivity():
    # Freeze 3 of the field report had 5.7 GiB free two seconds before the
    # Mac stopped, with little file cache. Once free pages fall under the
    # abort floor with the supply under the shed floor, compressor growth
    # trips exactly as before, well above the starved line.
    before = _reading(
        free=int(5.6 * GIB), file_backed=2 * GIB, wired=88 * GIB,
        compressor=4 * GIB, swap=0, at_s=0.0,
    )
    after = _reading(
        free=5 * GIB, file_backed=2 * GIB, wired=88 * GIB,
        compressor=5 * GIB, swap=0, at_s=2.0,
    )
    shed, abort = sm.reading_floors(after)
    assert after.free_bytes < abort and after.available_bytes < shed
    assert after.free_bytes > 512 * MIB
    assert sm.memory_thrashing(after, before)
