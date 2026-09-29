"""Flash-Next's routed gate/up gather on the tensor units, reading token rows in place.

A prefill-width MoE sorts its routed rows by expert and multiplies them with
``mx.gather_qmm(..., sorted_indices=True)``.  The sort first copies every token
row once per expert it is routed to (``x[order // top_k]``: at a 4,096-token
Flash-Next chunk, 40,960 rows of 2,560 values, 210 MB written and read back in
every layer), and MLX's kernel then streams that copy.

This kernel skips the copy.  Each lane reads the token rows its fragments need
through the sort's row map, so the ten routed copies of a token row are one
row in memory and come from cache.  Threadgroups own 64 sorted rows of ONE
expert: each expert's run is cut into its own tiles, and a threadgroup finds
its expert by binary search over a small prefix table of tiles per expert.
Streaming rows that are already sorted (the down projection) is not faster
this way (0.88x to 0.94x at 4,096 and 8,192-token chunks), so that stays on
MLX's kernel.

With ``swiglu=True`` the weight's rows are Flash-Next's fused ``[gate | up]``
and the kernel writes ``silu(gate) * up`` directly: each threadgroup holds 32
gate rows and the 32 matching up rows, so a lane's gate and up values meet in
one threadgroup, and the 1,280-wide gate/up output, the split and the two
elementwise passes over it are never materialized.  The epilogue rounds the
accumulators to the activation dtype and applies MLX's own ``Sigmoid`` and
``Multiply`` functors (read from the installed headers) with the same rounding
after each op as ``nn.silu(gate) * up``, which a probe over all 65,536 bf16 and
fp16 inputs matched exactly.

Measured on an M5 Max (MLX 0.32.2, 2026-09-29; E 512, top-10, 4-bit gate/up
2,560 -> 1,280, median of 15 synchronized calls) against the chain it
replaces (row copy, stock gather, split, ``nn.silu(gate) * up``): 1.47x /
1.31x / 1.23x at 2,048 / 4,096 / 8,192-token chunks with group 32 and 1.43x /
1.30x / 1.23x with group 64; 10.0 ms -> 7.65 ms per layer at 4,096 tokens.

The arithmetic of every output row is MLX's: the kernel is compiled from the
installed MLX's own tensor-unit tile, multiply-accumulate and quantized
weight-loader templates (read from the installed package's headers at first
use; nothing of MLX is copied into this tree), with the tile shape, the 32-wide
inner steps and the float accumulation of MLX's sorted kernel, so the result is
bit-identical to ``mx.gather_qmm(tokens[row_map], ..., sorted_indices=True)``.
Row positions are 32-bit throughout, so there is no 32,767-row bound
(``mtplx.moe_sorted_gather``).  A first-use canary per compiled instantiation
compares a sample of the real call against that stock result, bit for bit; a
mismatch or a compile failure turns the kernel off for that instantiation for
the rest of the process with a printed reason and a counter, and the caller
runs the stock op (for ``swiglu=True``: the stock gather, split and
``nn.silu(gate) * up``).

Only on tensor-unit GPUs (``nax_detect.nax_available()``: the M1 to M4
rehearsal switch keeps the stock path), for affine weights, bf16 or fp16
activations, 4 or 8 bits, K and N multiples of 64, and at least ``MIN_ROWS``
routed rows (decode and verify widths keep MLX's kernel).
``MTPLX_MOE_SORTED_GATHER_KERNEL=0`` keeps the stock path.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mtplx import nax_detect

__all__ = ["applies", "gather_rows_qmm", "stats"]

BM = 64
BN = 64
BK = 64
THREADS = 128
#: Routed rows below this keep MLX's kernel (decode and verify widths).
MIN_ROWS = 4096
_BITS = (4, 8)
_GROUPS = (32, 64, 128)
_DTYPES = (mx.bfloat16, mx.float16)
_ENV = "MTPLX_MOE_SORTED_GATHER_KERNEL"
_CANARY_ROWS = 4096

# The set MLX's own JIT concatenates for its sorted tensor-unit gather.
_KERNEL_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h",
    "mlx/backend/metal/kernels/quantized_utils.h",
    "mlx/backend/metal/kernels/quantized_nax.h",
)
# MLX prepends utils.h (and what it includes) to every custom kernel.
_PREAMBLE_HEADER = "mlx/backend/metal/kernels/utils.h"
_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")

_STATS: dict[str, int] = {"calls": 0, "fallbacks": 0, "canaries": 0, "canary_failures": 0}
_CANARY: dict[tuple, bool] = {}


def stats() -> dict[str, int]:
    return dict(_STATS)


def _include_root() -> Path | None:
    root = Path(mx.__file__).resolve().parent / "include"
    for rel in _KERNEL_HEADERS + (_PREAMBLE_HEADER,):
        if not (root / rel).is_file():
            return None
    return root


def _closure(root: Path, rel: str, seen: set[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _closure(root, match.group(1), seen)


def _inline(root: Path, rel: str, seen: set[str], out: list[str]) -> None:
    """Append ``rel`` with its quoted includes expanded once (in include order)."""

    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _inline(root, match.group(1), seen, out)
        elif not _PRAGMA_ONCE.match(line):
            out.append(line)


def _functor(root: Path, rel: str, name: str) -> str | None:
    """One elementwise functor struct from an installed MLX header, verbatim
    from the installed package (so the epilogue computes what MLX's own
    elementwise kernels compute)."""

    text = (root / rel).read_text()
    start = text.find(f"struct {name} {{")
    if start < 0:
        return None
    end = text.find("\n};", start)
    return None if end < 0 else text[start : end + 3]


@lru_cache(maxsize=1)
def _mlx_headers() -> str | None:
    """The installed MLX's tile, MMA and quantized-loader templates and its
    Sigmoid and Multiply functors as one header string, or None when this MLX
    install ships no kernel headers."""

    root = _include_root()
    if root is None:
        return None
    seen: set[str] = set()
    _closure(root, _PREAMBLE_HEADER, seen)  # already in every custom kernel
    out: list[str] = []
    for rel in _KERNEL_HEADERS:
        _inline(root, rel, seen, out)
    sigmoid = _functor(root, "mlx/backend/metal/kernels/unary_ops.h", "Sigmoid")
    multiply = _functor(root, "mlx/backend/metal/kernels/binary_ops.h", "Multiply")
    if sigmoid is None or multiply is None:
        return None
    out += [sigmoid, multiply]
    return "\n".join(out) + "\nusing namespace mlx::steel;\n"


# Our kernel body.  Template constants: T (activation dtype), GS, BITS, KD, ND,
# ED (reduction width, weight rows, experts) and SWIGLU: the weight's rows are
# [gate | up] and the kernel writes silu(gate) * up (ND / 2 columns), each
# threadgroup holding 32 gate rows and the 32 matching up rows.
_SOURCE = r"""
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WN = 2;
    constexpr short SM = 32;
    constexpr short SN = 32;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    constexpr int PACK = get_pack_factor<BITS, 8>();
    constexpr int PACK_BYTES = get_bytes_per_pack<BITS>();
    constexpr int BK_PAD = BK + 16 / sizeof(T);
    constexpr int ROW_BYTES = KD * PACK_BYTES / PACK;
    constexpr int ROW_GROUPS = KD / GS;
    constexpr int OUT_COLS = SWIGLU ? ND / 2 : ND;
    // SWIGLU: the tile's first 32 weight rows are gate rows, the last 32 the
    // matching up rows, each half loaded by 64 threads from its own row range.
    using WeightLoader = metal::conditional_t<
        SWIGLU,
        QuantizedBlockLoader<T, BN / 2, BK, BK_PAD, 1, 64, GS, BITS>,
        QuantizedBlockLoader<T, BN, BK, BK_PAD, 1, 128, GS, BITS>>;

    threadgroup T w_tile[BN * BK_PAD];

    // The tile this threadgroup owns, and the one expert whose rows it holds:
    // the last expert whose first tile is at or before it.
    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tile_start[ED]) {
        return;
    }
    int lo = 0;
    int hi = ED - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int expert = lo;
    const int row0 = row_start[expert] + (tile - tile_start[expert]) * BM;
    const int tile_rows = min(BM, row_start[expert + 1] - row0);
    const int col0 = int(threadgroup_position_in_grid.x) * BN;

    const ushort sg = ushort(simdgroup_index_in_threadgroup);
    const ushort lane = ushort(thread_index_in_simdgroup);
    const short tm = SM * short(sg / WN);
    const short tn = SN * short(sg % WN);
    // Rows of this tile that fall in this simdgroup's 32-row half (0 to 32).
    const short rows = short(clamp(tile_rows - int(tm), 0, int(SM)));

    size_t w_row = size_t(expert) * ND + size_t(col0);
    threadgroup T* w_dst = w_tile;
    ushort load_sg = sg;
    if (SWIGLU) {
        const int up_half = int(sg / 2);
        w_row = size_t(expert) * ND + size_t(up_half * (ND / 2)) + size_t(col0 / 2);
        w_dst = w_tile + up_half * (BN / 2) * BK_PAD;
        load_sg = sg % 2;
    }
    thread WeightLoader loader(
        (const device uint8_t*)w + w_row * ROW_BYTES,
        scales + w_row * ROW_GROUPS,
        biases + w_row * ROW_GROUPS,
        KD,
        w_dst,
        load_sg,
        lane);

    NAXTile<float, TM, TN> acc;
    acc.clear();

    // A lane holds rows (y, y + 8) and four consecutive columns from x of every
    // 16x16 fragment (MLX's fragment coordinates).  Its four token rows are
    // fixed for the whole K loop; rows past the tile read as zero, as MLX's
    // bounded load does.
    const short2 coord = BaseNAXFrag::get_coord();
    const device T* lane_rows[TM][2];
    bool lane_live[TM][2];
    for (short i = 0; i < TM; i++) {
        for (short r = 0; r < 2; r++) {
            const short local = i * 16 + coord.y + r * 8;
            lane_live[i][r] = local < rows;
            const int token = lane_live[i][r] ? int(row_map[row0 + tm + local]) : 0;
            lane_rows[i][r] = x + size_t(token) * KD + coord.x;
        }
    }

    // The K loop, compiled twice: for a simdgroup whose 32 rows are all live
    // (no bounds) and for a partial or empty half.
    dispatch_bool(rows == SM, [&](auto full) {
        for (int k = 0; k < KD / BK; k++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);

            STEEL_PRAGMA_NO_UNROLL
            for (short kk = 0; kk < BK; kk += SK) {
                if (decltype(full)::value || rows > 0) {
                    NAXTile<T, TM, TK> a;
                    NAXTile<T, TN, TK> b;
                    volatile int keep_order;
                    const int col = k * BK + kk;
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < TM; i++) {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < TK; j++) {
                            thread auto& frag = a.frag_at(i, j);
                            STEEL_PRAGMA_UNROLL
                            for (short r = 0; r < 2; r++) {
                                const device T* src = lane_rows[i][r] + col + j * 16;
                                STEEL_PRAGMA_UNROLL
                                for (short c = 0; c < 4; c++) {
                                    if constexpr (decltype(full)::value) {
                                        frag[r * 4 + c] = src[c];
                                    } else {
                                        frag[r * 4 + c] = lane_live[i][r] ? src[c] : T(0);
                                    }
                                }
                            }
                        }
                    }
                    b.template load<T, BK_PAD, 1>(w_tile + tn * BK_PAD + kk);
                    tile_matmad_nax(
                        acc,
                        a,
                        metal::bool_constant<false>{},
                        b,
                        metal::bool_constant<true>{});
                    (void)keep_order;
                }
            }
            loader.next();
        }
    });

    if constexpr (SWIGLU) {
        // The up simdgroup of each 32-row half hands its values, rounded to T
        // as the stock gather's output is, to the gate simdgroup of the same
        // rows through the (now free) weight tile, lane to lane.  The gate
        // simdgroup writes silu(gate) * up with MLX's own functors, rounding to
        // T after each op as the stock chain does.
        constexpr short VALS = TM * TN * 8;
        static_assert(2 * 32 * VALS <= BN * BK_PAD, "the hand-off must fit the weight tile");
        threadgroup T* handoff = w_tile + (sg / 2) * (32 * VALS) + lane * VALS;
        const thread float* vals = acc.elems();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tn != 0) {
            for (short e = 0; e < VALS; e++) {
                handoff[e] = static_cast<T>(vals[e]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tn == 0 && rows > 0) {
            device T* out = y + size_t(row0 + tm) * OUT_COLS + size_t(col0 / 2);
            for (short i = 0; i < TM; i++) {
                for (short j = 0; j < TN; j++) {
                    for (short r = 0; r < 2; r++) {
                        const short row = i * 16 + coord.y + r * 8;
                        if (row >= rows) {
                            continue;
                        }
                        for (short c = 0; c < 4; c++) {
                            const short e = (i * TN + j) * 8 + r * 4 + c;
                            const T gate = static_cast<T>(vals[e]);
                            const T act = Multiply()(gate, Sigmoid()(gate));
                            out[size_t(row) * OUT_COLS + j * 16 + coord.x + c] =
                                Multiply()(act, handoff[e]);
                        }
                    }
                }
            }
        }
    } else if (rows > 0) {
        device T* out = y + size_t(row0 + tm) * ND + size_t(col0 + tn);
        if (rows == SM) {
            acc.store(out, ND);
        } else {
            acc.store_safe(out, ND, short2(SN, rows));
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    headers = _mlx_headers()
    if headers is None:
        return None
    return mx.fast.metal_kernel(
        name="mtplx_moe_gather_rows",
        input_names=["x", "w", "scales", "biases", "tile_start", "row_start", "row_map"],
        output_names=["y"],
        source=_SOURCE,
        header=headers,
        ensure_row_contiguous=True,
    )


def _enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def applies(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    rhs_indices: mx.array,
    *,
    group_size,
    bits,
    mode,
) -> bool:
    """Whether this gather can take the kernel (shape, dtype and device).

    ``tokens`` is ``[n_tokens, 1, K]``; ``row_map`` and ``rhs_indices`` are the
    ``[rows]`` token index and expert of every sorted row.
    """

    if mode != "affine" or bits not in _BITS or group_size not in _GROUPS:
        return False
    if tokens.dtype not in _DTYPES or tokens.ndim != 3 or tokens.shape[1] != 1:
        return False
    if scales.dtype != tokens.dtype or biases is None or biases.dtype != tokens.dtype:
        return False
    if rhs_indices is None or rhs_indices.ndim != 1 or row_map is None or row_map.ndim != 1:
        return False
    rows = int(rhs_indices.shape[0])
    if rows < MIN_ROWS or int(row_map.shape[0]) != rows or row_map.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    n = int(w.shape[1])
    k = int(w.shape[2]) * 32 // int(bits)
    if n % BN or k % BK or int(tokens.shape[-1]) != k or k % group_size:
        return False
    if not _enabled() or not nax_detect.nax_available():
        return False
    return _mlx_headers() is not None


def _schedule(rhs_indices: mx.array, experts: int) -> tuple[mx.array, mx.array]:
    """Exclusive prefix sums of 64-row tiles and of rows per expert, [E + 1] each."""

    counts = mx.zeros((experts,), dtype=mx.int32).at[rhs_indices].add(1)
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(counts)])
    tiles = (counts + (BM - 1)) // BM
    tile_start = mx.concatenate([zero, mx.cumsum(tiles)])
    return tile_start, row_start


def _launch(tokens, row_map, w, scales, biases, rhs_indices, *, group_size, bits, swiglu=False):
    rows = int(rhs_indices.shape[0])
    experts = int(w.shape[0])
    n = int(w.shape[1])
    k = int(w.shape[2]) * 32 // int(bits)
    out_cols = n // 2 if swiglu else n
    tile_start, row_start = _schedule(rhs_indices, experts)
    (y,) = _kernel()(
        inputs=[tokens.reshape(-1, k), w, scales, biases, tile_start, row_start, row_map],
        template=[
            ("T", tokens.dtype),
            ("GS", int(group_size)),
            ("BITS", int(bits)),
            ("KD", k),
            ("ND", n),
            ("ED", experts),
            ("SWIGLU", bool(swiglu)),
        ],
        # Every expert adds at most one partial tile.
        grid=((n // BN) * THREADS, (rows + BM - 1) // BM + experts, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, out_cols)],
        output_dtypes=[tokens.dtype],
    )
    return y.reshape(rows, 1, out_cols)


def _fail(key: tuple, reason: str) -> None:
    _CANARY[key] = False
    _STATS["canary_failures"] += 1
    print(
        f"[moe-sorted-gather] tensor-unit kernel off for {key}: {reason}; "
        "the stock sorted gather runs instead",
        file=sys.stderr,
        flush=True,
    )


def _stock(tokens, rows, w, scales, biases, idx, *, group_size, bits, swiglu):
    """What the kernel replaces: the copy, the stock sorted gather and, for
    ``swiglu``, Flash-Next's split and ``nn.silu(gate) * up``."""

    y = mx.gather_qmm(
        tokens[rows], w, scales, biases, rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )
    if swiglu:
        import mlx.nn as nn

        gate, up = mx.split(y, 2, axis=-1)
        y = nn.silu(gate) * up
    return y


def _canary(key, tokens, row_map, w, scales, biases, rhs_indices, *, group_size, bits, swiglu) -> bool:
    """First use of an instantiation: the kernel against the stock chain on
    the copied rows, bit for bit, over a sample of the real call's rows."""

    passed = _CANARY.get(key)
    if passed is not None:
        return passed
    _STATS["canaries"] += 1
    # Every step-th sorted row: still sorted, and every expert is sampled.
    total = int(rhs_indices.shape[0])
    step = max(1, total // _CANARY_ROWS)
    sample = mx.arange(0, total, step, dtype=mx.uint32)[:_CANARY_ROWS]
    idx = rhs_indices[sample]
    rows = row_map[sample]
    kw = dict(group_size=group_size, bits=bits, swiglu=swiglu)
    try:
        ours = _launch(tokens, rows, w, scales, biases, idx, **kw)
        stock = _stock(tokens, rows, w, scales, biases, idx, **kw)
        mx.eval(ours, stock)
        same = bool(mx.array_equal(ours, stock).item())
    except Exception as exc:  # a compile or dispatch failure on this GPU
        _fail(key, f"{type(exc).__name__}: {exc}")
        return False
    if not same:
        _fail(key, "output differs from the stock sorted gather on the canary rows")
        return False
    _CANARY[key] = True
    return True


def gather_rows_qmm(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    rhs_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    swiglu: bool = False,
) -> mx.array | None:
    """``gather_qmm(tokens[row_map], w, ..., sorted_indices=True)`` as
    ``[rows, 1, N]`` without the copy, or with ``swiglu`` Flash-Next's
    ``silu(gate) * up`` of that ``[gate | up]`` output as ``[rows, 1, N / 2]``;
    None when the stock path must run."""

    key = (tokens.dtype, int(group_size), int(bits), tuple(w.shape), bool(swiglu))
    kw = dict(group_size=int(group_size), bits=int(bits), swiglu=bool(swiglu))
    if not _canary(key, tokens, row_map, w, scales, biases, rhs_indices, **kw):
        _STATS["fallbacks"] += 1
        return None
    _STATS["calls"] += 1
    return _launch(tokens, row_map, w, scales, biases, rhs_indices, **kw)
