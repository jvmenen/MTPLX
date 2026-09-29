"""The one entry point for MLX's expert-sorted quantized gather, with its row guard.

Every MoE in MTPLX that sorts its routed rows by expert multiplies them with
``mx.gather_qmm(..., sorted_indices=True)`` over a ``[routed_rows, 1, K]``
activation.  Those calls go through :func:`gather_qmm` (a direct call) or
:func:`switch_linear` (a switch-linear module call), and mlx-lm's own switch
layers get the same guard from :func:`install_switch_linear_guard`.

Why the guard exists
--------------------
On a tensor-unit (NAX) GPU, MLX 0.32.2 runs that call on its sorted-rows
tensor-unit kernel for affine weights.  The kernel turns "rows left from this
simdgroup's first row" into a 16-bit integer before clamping it to the tile.
When a call routes more than 32,767 rows and the row count is not a multiple of
the kernel's row tile (32 or 64 rows, picked from rows per expert), the leading
simdgroups read a negative or wrapped count, skip their matmul and never store:
their output rows keep whatever the allocator handed back.  An aligned count
takes a branch that never computes that number, which is why 32,768 and 40,960
rows are fine.  Measured on an M5 Max (applegpu_g17s, MLX 0.32.2): 32,770 rows
leave rows 0-31 unwritten, 34,040 rows 0-1,279, 40,950 rows 0-8,191.  MLX 0.32.3
schedules the kernel per expert segment and is correct.

Flash-Next routes ten rows per token, so any prefill forward of 3,277 to 4,095
tokens whose width is not a multiple of 32 crosses the bound twice per layer
(the gate/up gather and the down gather).

The guard
---------
Such a call is padded to the next multiple of 64 rows (zero activations, the
last expert index repeated so the order stays sorted), run once, and sliced
back to its real rows.  Every tile is then full, so the kernel takes its
aligned branch.  Each output row depends only on its own input row and its
expert's weights, so the real rows are bit-identical to an unpadded call
wherever that call is correct (``tests/test_moe_sorted_gather.py`` pins it).

It engages only where the defect can occur: MLX's own tensor-unit test
(``nax_detect.nax_hardware_available``: the M1 to M4 rehearsal architecture
turns it off exactly as it turns off MLX's kernel, and the route switch
``MTPLX_FORCE_GPU_FAMILY_FALLBACK`` does not, because MLX does not read it) and
an MLX release before 0.32.3.  Every other call passes through unchanged.
"""

from __future__ import annotations

import functools
import re
import sys
from functools import lru_cache
from typing import Any

import mlx.core as mx

from mtplx import nax_detect

#: Row counts up to this bound cannot wrap the kernel's 16-bit row count.
INT16_ROW_BOUND = 32767
#: Padding to a multiple of 64 fills every row tile MLX may pick (32 or 64).
ROW_ALIGN = 64
#: The first MLX release whose sorted tensor-unit gather counts rows in 32 bits.
FIXED_MLX_RELEASE = (0, 32, 3)

_PRE_RELEASE = re.compile(r"^[.\-_]?(dev|a|b|c|rc|alpha|beta|pre|preview)\d*", re.I)
_GUARD_MARK = "_mtplx_sorted_rows_guard"

_STATS: dict[str, int] = {"padded_calls": 0, "padded_rows": 0}
_ANNOUNCED = False


def mlx_release_affected(version: str) -> bool:
    """Whether an MLX version string predates the sorted-gather row fix.

    A pre-release of the fixing release counts as affected (it may predate the
    fix); an unreadable version counts as affected (the guard is exact either
    way, so the safe reading costs nothing but a pad).
    """

    match = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)(.*)$", str(version))
    if match is None:
        return True
    release = tuple(int(part) for part in match.groups()[:3])
    if release < FIXED_MLX_RELEASE:
        return True
    if release == FIXED_MLX_RELEASE and _PRE_RELEASE.match(match.group(4) or ""):
        return True
    return False


@lru_cache(maxsize=1)
def guard_active() -> bool:
    """MLX will run its sorted tensor-unit kernel here, and that kernel has the
    16-bit row count.  Both halves are fixed for the life of the process."""

    return mlx_release_affected(mx.__version__) and nax_detect.nax_hardware_available()


def pad_rows(rows: int) -> int:
    """Rows to append to a sorted call of ``rows`` routed rows (0 = none)."""

    rows = int(rows)
    if rows <= INT16_ROW_BOUND or rows % ROW_ALIGN == 0 or not guard_active():
        return 0
    return ROW_ALIGN - rows % ROW_ALIGN


def stats() -> dict[str, int]:
    """How many sorted gathers this process padded, and by how many rows."""

    return dict(_STATS)


def _call_pad(
    x: mx.array,
    rhs_indices: Any,
    lhs_indices: Any,
    *,
    sorted_indices: bool,
    transpose: bool,
    mode: Any,
) -> int:
    """The pad for one call, 0 unless it is an affine, transposed, expert-sorted
    call in the ``[rows, 1, K]`` layout MLX sends to the affected kernel."""

    if not sorted_indices or not transpose or mode != "affine" or rhs_indices is None:
        return 0
    if rhs_indices.ndim != 1 or x.ndim < 2 or x.shape[-2] != 1:
        return 0
    rows = int(rhs_indices.shape[0])
    if int(x.shape[0]) != rows:
        return 0
    if lhs_indices is not None and (
        lhs_indices.ndim != 1 or int(lhs_indices.shape[0]) != rows
    ):
        return 0
    return pad_rows(rows)


def _pad(x: mx.array, indices: mx.array, pad: int) -> tuple[mx.array, mx.array]:
    """``x`` with ``pad`` zero rows and ``indices`` with its last expert repeated."""

    global _ANNOUNCED
    rows = int(indices.shape[0])
    _STATS["padded_calls"] += 1
    _STATS["padded_rows"] += pad
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            f"[moe-sorted-gather] padding a {rows:,}-row sorted expert gather to "
            f"{rows + pad:,} rows: MLX {mx.__version__} leaves rows unwritten past "
            f"{INT16_ROW_BOUND:,} unaligned rows on tensor-unit GPUs (fixed in "
            "0.32.3); the real rows are unchanged",
            file=sys.stderr,
            flush=True,
        )
    x = mx.concatenate([x, mx.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)
    indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0)
    return x, indices


def gather_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None = None,
    lhs_indices: mx.array | None = None,
    rhs_indices: mx.array | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    *,
    sorted_indices: bool = False,
) -> mx.array:
    """``mx.gather_qmm`` with the sorted-rows guard: same arguments, same result."""

    pad = (
        _call_pad(
            x,
            rhs_indices,
            lhs_indices,
            sorted_indices=True,
            transpose=transpose,
            mode=mode,
        )
        if sorted_indices
        else 0
    )
    if pad:
        rows = int(x.shape[0])
        x, rhs_indices = _pad(x, rhs_indices, pad)
        if lhs_indices is not None:
            lhs_indices = mx.concatenate(
                [lhs_indices, mx.arange(rows, rows + pad, dtype=lhs_indices.dtype)]
            )
    y = mx.gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=sorted_indices,
    )
    return y[: x.shape[0] - pad] if pad else y


def _module_mode(module: Any) -> Any:
    """A switch linear's quantization mode, None when it holds dense weights."""

    if "scales" not in module:
        return None
    return getattr(module, "mode", "affine")


def switch_linear(
    module: Any, x: mx.array, indices: mx.array, *, sorted_indices: bool
) -> mx.array:
    """``module(x, indices, sorted_indices=...)`` for a switch linear, guarded."""

    pad = _call_pad(
        x,
        indices,
        None,
        sorted_indices=sorted_indices,
        transpose=True,
        mode=_module_mode(module),
    )
    if not pad:
        return module(x, indices, sorted_indices=sorted_indices)
    rows = int(indices.shape[0])
    x, indices = _pad(x, indices, pad)
    return module(x, indices, sorted_indices=True)[:rows]


def install_switch_linear_guard() -> bool:
    """Guard mlx-lm's quantized switch linear, which every mlx-lm MoE family
    (and any MTPLX block that keeps a stock projection) calls with sorted rows.

    Wraps the class's current ``__call__`` rather than replacing its body, so
    whatever it does (bias, a later patch) runs unchanged on the padded rows.
    Idempotent; a no-op wherever :func:`guard_active` is false.  Returns
    whether the guard is installed.
    """

    if not guard_active():
        return False
    from mlx_lm.models import switch_layers

    cls = switch_layers.QuantizedSwitchLinear
    current = cls.__call__
    if getattr(current, _GUARD_MARK, False):
        return True

    @functools.wraps(current)
    def guarded_call(self, x, indices, sorted_indices=False):
        if sorted_indices:
            pad = _call_pad(
                x,
                indices,
                None,
                sorted_indices=True,
                transpose=True,
                mode=getattr(self, "mode", "affine"),
            )
            if pad:
                rows = int(indices.shape[0])
                x, indices = _pad(x, indices, pad)
                return current(self, x, indices, sorted_indices=True)[:rows]
        return current(self, x, indices, sorted_indices=sorted_indices)

    setattr(guarded_call, _GUARD_MARK, True)
    cls.__call__ = guarded_call
    return True
