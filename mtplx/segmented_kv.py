"""Segmented KV cache for the full-attention layers (MTPLX_SEGMENTED_KV=1, default off).

A conversation's attention KV is a list of immutable SEGMENTS plus one growing TAIL, instead
of one contiguous buffer that the session bank aliases with a lazy snapshot:

* a sealed segment has its exact capacity and is never written again;
* the tail is the only buffer a decode, a verify forward or a prefill chunk writes into;
  verify rollback (``trim``) only moves the tail's row count;
* ``seal()`` closes the tail at the end of a turn (a copy of the tail only, never of the
  history) and a new tail starts empty at the next write;
* a snapshot of the cache (``state``) IS the list of segment references: it seals the tail
  and hands out ``SegmentedKVState`` (segment, rows) references. Nothing a later turn writes
  can alias them, so MLX's copy-on-write never copies the history. A fork or a return into the
  middle of a segment is a reference ``(segment, n')`` with n' below the segment's rows:
  no copy, and the rows past n' belong to whoever else holds the segment.

Why (receipts, 2026-10-06/07, Qwen3.8-27B, 16 full-attention layers, 64 KiB of KV per token):
the session bank's snapshot views alias the per-layer KV buffers, so the first write of the
next turn copies every buffer: +4.95 GiB at 80K (+3.1 GiB at 50K), whether the restore is a
clone or a reference lease (docs/onderzoek/2026-10-07-lease-vs-clone.md). With segments the
same follow-up turn costs the new segment only (+0.125 GiB for 1,024 tokens plus slack,
2026-10-07 phase 1 measurement) and a branch from the same history costs 0.

Attention over the segments: ``mtplx/kernels/sdpa_segmented.py`` (exact split-softmax merge,
one launch for up to 12 segments) for the verify/decode window; everything else gathers the
rows into one contiguous array per layer and call (``gather``, correct and slower for long
histories) and runs the stock route on it. The prefill over existing segments has its own
interface, ``attend_segments_lse`` (per segment ``(out, lse)`` plus one merge), so that an
MLX kernel which returns the logsumexp can replace ``segment_sdpa_lse``.

GatedDeltaNet (recurrent) layers keep their state objects as they are.

Reference counts. A segment counts its holders (live caches and ``SegmentedKVState``
snapshots) in a weak set: ``refcount`` is the number of holders still alive, an evicted or
dropped bank entry releases its segments by going away, and the merge policy only merges
segments that no snapshot holds.
"""

from __future__ import annotations

import itertools
import os
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

_SEGMENT_IDS = itertools.count(1)


def segmented_kv_enabled() -> bool:
    """MTPLX_SEGMENTED_KV (default off), read on every call."""
    return os.environ.get("MTPLX_SEGMENTED_KV", "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


#: Receipts: which route served segmented prefill chunks and which per-segment attention ran.
route_counts: dict[str, int] = {}


def _count(route: str) -> None:
    route_counts[route] = route_counts.get(route, 0) + 1


_SDPA_LSE: bool | None = None


def _probe_sdpa_lse() -> bool:
    """Whether this MLX build's fused SDPA can return the logsumexp (fork branch sdpa-lse).

    The one place that knows the API: ``mx.fast.scaled_dot_product_attention(..., return_lse=True)``
    returning ``(out, lse)``. Stock released MLX raises TypeError on the keyword, which is the
    "absent" answer; any other failure also counts as absent.
    """
    try:
        q = mx.zeros((1, 1, 1, 64), dtype=mx.bfloat16)
        result = mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, return_lse=True)
        mx.eval(result)
        return isinstance(result, (tuple, list)) and len(result) == 2
    except Exception:  # noqa: BLE001
        return False


def sdpa_lse_available() -> bool:
    """Feature detection, once per process (first use)."""
    global _SDPA_LSE
    if _SDPA_LSE is None:
        _SDPA_LSE = _probe_sdpa_lse()
    return _SDPA_LSE


def prefill_route() -> str:
    """How a prefill chunk attends over existing segments.

    ``lse``: per segment (out, lse) plus one merge. Default when the MLX kernel with LSE output
    is present; ``MTPLX_SEGMENTED_KV_PREFILL=lse`` forces it (exact reference route without the
    kernel). ``gather`` (default on stock MLX): one contiguous copy per layer and call, then the
    stock SDPA.
    """
    raw = os.environ.get("MTPLX_SEGMENTED_KV_PREFILL", "").strip().lower()
    if raw in {"lse", "gather"}:
        return raw
    return "lse" if sdpa_lse_available() else "gather"


class KVSegment:
    """One K/V buffer pair, (1, Hk, capacity, D), shared by reference.

    ``sealed`` segments hold exactly ``rows == capacity`` live rows and are never written.
    The tail of a cache is an unsealed segment: only its owner writes it.
    """

    __slots__ = ("__weakref__", "_holders", "id", "keys", "sealed", "values")

    def __init__(self, keys: mx.array, values: mx.array, *, sealed: bool = False) -> None:
        self.id = next(_SEGMENT_IDS)
        self.keys = keys
        self.values = values
        self.sealed = bool(sealed)
        self._holders: weakref.WeakSet[Any] = weakref.WeakSet()

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes) + int(self.values.nbytes)

    @property
    def refcount(self) -> int:
        """Holders alive right now: live caches and snapshot states."""
        return len(self._holders)

    def hold(self, holder: Any) -> None:
        self._holders.add(holder)

    def release(self, holder: Any) -> None:
        self._holders.discard(holder)


@dataclass(frozen=True)
class SegRef:
    """The first ``n`` rows of a segment."""

    segment: KVSegment
    n: int


class SegmentedKVState:
    """An immutable snapshot of a segmented cache: references, no buffers of its own.

    This is what the session bank stores as the cache snapshot of a segmented layer. It holds
    its segments (reference counts) for as long as it lives.
    """

    def __init__(self, refs: Iterable[SegRef]) -> None:
        self.refs: tuple[SegRef, ...] = tuple(refs)
        for ref in self.refs:
            ref.segment.hold(self)

    @property
    def rows(self) -> int:
        return sum(int(ref.n) for ref in self.refs)

    @property
    def nbytes(self) -> int:
        """Bytes of the segments this state references, counted as if it were alone."""
        return sum(ref.segment.nbytes for ref in _unique_refs(self.refs))

    def segments(self) -> list[KVSegment]:
        return [ref.segment for ref in self.refs]

    def prefix(self, rows: int) -> SegmentedKVState:
        """The first ``rows`` rows as a new state (shortens the last reference, no copy)."""
        return SegmentedKVState(_prefix_refs(self.refs, rows))

    def to_arrays(self) -> tuple[mx.array, mx.array] | None:
        """One contiguous (keys, values) pair, the state a stock KV cache would give."""
        return _gather_refs(self.refs)

    def __len__(self) -> int:
        return len(self.refs)


def _unique_refs(refs: Iterable[SegRef]) -> list[SegRef]:
    seen: set[int] = set()
    out: list[SegRef] = []
    for ref in refs:
        if ref.segment.id not in seen:
            seen.add(ref.segment.id)
            out.append(ref)
    return out


def _prefix_refs(refs: Sequence[SegRef], rows: int) -> list[SegRef]:
    rows = max(0, int(rows))
    out: list[SegRef] = []
    left = rows
    for ref in refs:
        if left <= 0:
            break
        take = min(int(ref.n), left)
        out.append(SegRef(ref.segment, take) if take != ref.n else ref)
        left -= take
    return out


def _rows_view(array: mx.array, n: int) -> mx.array:
    return array if int(array.shape[2]) == int(n) else array[..., : int(n), :]


def _gather_refs(refs: Sequence[SegRef]) -> tuple[mx.array, mx.array] | None:
    if not refs:
        return None
    keys = [_rows_view(ref.segment.keys, ref.n) for ref in refs]
    values = [_rows_view(ref.segment.values, ref.n) for ref in refs]
    if len(refs) == 1:
        return keys[0], values[0]
    return mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)


def _own_copy(array: mx.array) -> mx.array:
    """An evaluated array that owns its buffer (a slice of a larger buffer is copied)."""
    owned = mx.contiguous(array)
    mx.eval(owned)
    return owned


class SegmentedKVCache:
    """KV cache of one full-attention layer as sealed segments plus a growing tail.

    Drop-in for the stock ``KVCache`` where the model code needs it: ``update_and_fetch``,
    ``offset`` (the TOTAL number of rows, which is also the rotary offset and what every
    route gate must read instead of a buffer capacity), ``make_mask``, ``trim``, ``state``.
    """

    step = 256

    def __init__(self, *, step: int = 256) -> None:
        self.step = int(step)
        self._sealed: list[SegRef] = []
        self._tail: KVSegment | None = None
        self._tail_n = 0
        self.seal_copies = 0
        self.seal_copy_rows = 0
        self.merges = 0
        self.merge_rows = 0

    # ---- shape ----------------------------------------------------------------------------

    @property
    def offset(self) -> int:
        return sum(int(ref.n) for ref in self._sealed) + self._tail_n

    @offset.setter
    def offset(self, value: int) -> None:
        # Moving the offset is a trim (the stock cache has no other use for the setter here).
        value = int(value)
        current = self.offset
        if value > current:
            raise ValueError("a segmented KV cache cannot grow its offset without rows")
        self.trim(current - value)

    @property
    def tail_rows(self) -> int:
        return self._tail_n

    @property
    def segment_count(self) -> int:
        return len(self._sealed) + (1 if self._tail_n else 0)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        """Bytes this cache keeps allocated: its unique segments plus the tail buffer."""
        total = sum(ref.segment.nbytes for ref in _unique_refs(self._sealed))
        if self._tail is not None:
            total += self._tail.nbytes
        return total

    @property
    def meta_state(self) -> str:
        return ""

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        del value

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    # ---- writing --------------------------------------------------------------------------

    def append_rows(self, keys: mx.array, values: mx.array) -> None:
        """Write new rows into the tail (the only buffer this cache ever writes)."""
        if self._tail is None:
            self.compact()
        prev = self._tail_n
        steps = int(keys.shape[2])
        tail = self._tail
        if tail is None or prev + steps > tail.capacity:
            b, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + steps - 1) // self.step
            new_k = mx.zeros((b, n_kv_heads, n_steps * self.step, k_head_dim), keys.dtype)
            new_v = mx.zeros((b, n_kv_heads, n_steps * self.step, v_head_dim), values.dtype)
            if tail is not None:
                old_k, old_v = tail.keys, tail.values
                if prev % self.step != 0:
                    old_k, old_v = old_k[..., :prev, :], old_v[..., :prev, :]
                new_k = mx.concatenate([old_k, new_k], axis=2)
                new_v = mx.concatenate([old_v, new_v], axis=2)
            tail = KVSegment(new_k, new_v)
            self._tail = tail
        tail.keys[..., prev : prev + steps, :] = keys
        tail.values[..., prev : prev + steps, :] = values
        self._tail_n = prev + steps

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.append_rows(keys, values)
        gathered = self.gather()
        assert gathered is not None
        return gathered

    # ---- reading --------------------------------------------------------------------------

    def attention_segments(self) -> list[tuple[mx.array, mx.array, int]]:
        """(keys, values, rows) per segment, oldest first; the last is the tail when it has rows."""
        out = [(ref.segment.keys, ref.segment.values, int(ref.n)) for ref in self._sealed]
        if self._tail is not None and self._tail_n:
            out.append((self._tail.keys, self._tail.values, self._tail_n))
        return out

    def gather(self) -> tuple[mx.array, mx.array] | None:
        """One contiguous (keys, values) over all rows (lazy; a view when there is one segment)."""
        parts = self.attention_segments()
        if not parts:
            return None
        keys = [_rows_view(k, n) for k, _, n in parts]
        values = [_rows_view(v, n) for _, v, n in parts]
        if len(parts) == 1:
            return keys[0], values[0]
        return mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)

    @property
    def keys(self) -> mx.array | None:
        gathered = self.gather()
        return None if gathered is None else gathered[0]

    @property
    def values(self) -> mx.array | None:
        gathered = self.gather()
        return None if gathered is None else gathered[1]

    # ---- rollback -------------------------------------------------------------------------

    def trim(self, n: int) -> int:
        """Drop the last ``n`` rows. Verify rollback only ever moves the tail's row count."""
        n = min(self.offset, max(0, int(n)))
        left = n
        take = min(self._tail_n, left)
        self._tail_n -= take
        left -= take
        if self._tail_n == 0 and self._tail is not None and left > 0:
            self._tail = None
        while left > 0 and self._sealed:
            ref = self._sealed[-1]
            if ref.n <= left:
                self._sealed.pop()
                if not any(r.segment is ref.segment for r in self._sealed):
                    ref.segment.release(self)
                left -= int(ref.n)
            else:
                self._sealed[-1] = SegRef(ref.segment, int(ref.n) - left)
                left = 0
        return n

    # ---- sealing and state ----------------------------------------------------------------

    def seal(self) -> bool:
        """Close the tail: a segment of exactly its rows, then a fresh empty tail at the next write.

        Copies the tail's rows only (never a sealed segment); a tail that already has exactly
        its capacity is adopted without a copy. Returns True when a segment was added.
        """
        tail, rows = self._tail, self._tail_n
        if tail is None:
            return False
        if rows == 0:
            self._tail = None
            return False
        _count("seal")
        if rows == tail.capacity:
            keys, values = tail.keys, tail.values
            mx.eval(keys, values)
        else:
            _count("seal_copy")
            keys = _own_copy(tail.keys[..., :rows, :])
            values = _own_copy(tail.values[..., :rows, :])
            self.seal_copies += 1
            self.seal_copy_rows += rows
        self._tail = None
        self._tail_n = 0
        segment = KVSegment(keys, values, sealed=True)
        segment.hold(self)
        self._sealed.append(SegRef(segment, rows))
        return True

    @property
    def state(self) -> SegmentedKVState:
        """The snapshot of this cache: seals the tail, then the list of references."""
        self.seal()
        return SegmentedKVState(self._sealed)

    @state.setter
    def state(self, value: Any) -> None:
        for ref in self._sealed:
            ref.segment.release(self)
        self._sealed = []
        self._tail = None
        self._tail_n = 0
        if value is None:
            return
        if isinstance(value, SegmentedKVState):
            refs = list(value.refs)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and value[0] is not None:
            # A contiguous pair (an SSD restore, a stock snapshot): one sealed segment.
            # Rows of a stock buffer are a strided view of it (capacity beyond the rows). The
            # fused kernel takes row-contiguous inputs and MLX would copy such a view on every
            # launch (the whole history, per layer, per verify: +34 ms at 50K), so the segment
            # owns exact rows (a no-op when the pair is already row-contiguous).
            keys, values = (_own_copy(a) for a in value)
            segment = KVSegment(keys, values, sealed=True)
            refs = [SegRef(segment, int(keys.shape[2]))]
        else:
            raise ValueError(f"unsupported segmented KV state: {type(value).__name__}")
        for ref in refs:
            ref.segment.hold(self)
        self._sealed = refs

    # ---- merge policy ---------------------------------------------------------------------

    def compact(self, *, max_segments: int | None = None) -> int:
        """Merge small recent sealed segments among themselves (geometric tiers).

        Two adjacent sealed segments merge when no snapshot holds either (reference count 1:
        this cache), the older is at most ``MTPLX_SEGMENTED_KV_TIER`` times (default 2) the
        newer one and the result stays within ``MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS`` (default
        16384) rows. The big history segment is never merged here: a full copy would bring the
        duplicate back while a snapshot, a subagent or a lease references it. Copying two small
        segments costs 0.8 ms (512 + 512 rows) to 4.1 ms (4096 + 4096 rows) for 16 layers (phase 1
        measurement). Returns the number of merges.
        """
        tier = max(1, _int_env("MTPLX_SEGMENTED_KV_TIER", 2))
        cap_rows = _int_env("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", 16384)
        limit = max_segments if max_segments is not None else _int_env("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", 12)
        merged = 0
        while len(self._sealed) >= 2:
            older, newer = self._sealed[-2], self._sealed[-1]
            alone = older.segment.refcount <= 1 and newer.segment.refcount <= 1
            small = int(older.n) + int(newer.n) <= cap_rows
            geometric = int(older.n) <= tier * int(newer.n)
            over = len(self._sealed) > limit and alone
            if not (alone and ((small and geometric) or over)):
                break
            self._merge_last_two()
            merged += 1
        return merged

    def _merge_last_two(self) -> None:
        older, newer = self._sealed[-2], self._sealed[-1]
        keys = mx.concatenate(
            [_rows_view(older.segment.keys, older.n), _rows_view(newer.segment.keys, newer.n)], axis=2
        )
        values = mx.concatenate(
            [_rows_view(older.segment.values, older.n), _rows_view(newer.segment.values, newer.n)], axis=2
        )
        mx.eval(keys, values)
        segment = KVSegment(keys, values, sealed=True)
        segment.hold(self)
        for ref in (older, newer):
            ref.segment.release(self)
        self._sealed[-2:] = [SegRef(segment, int(older.n) + int(newer.n))]
        self.merges += 1
        _count("merge")
        self.merge_rows += int(older.n) + int(newer.n)

    def stats(self) -> dict[str, int]:
        return {
            "segments": self.segment_count,
            "sealed": len(self._sealed),
            "tail_rows": self._tail_n,
            "rows": self.offset,
            "bytes": self.nbytes,
            "seal_copies": self.seal_copies,
            "seal_copy_rows": self.seal_copy_rows,
            "merges": self.merges,
            "merge_rows": self.merge_rows,
        }


class GatheredKVView:
    """The contiguous (keys, values, offset) of a segmented cache, for the stock route ladder.

    The attention ladder in ``attention_split`` reads ``cache.keys``/``values``/``offset``;
    when no segment route takes a call it runs on this view of one gathered pair, so the
    fallback is the stock route on the same numbers.
    """

    def __init__(self, keys: mx.array, values: mx.array, offset: int) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)


def gathered_view(cache: SegmentedKVCache) -> GatheredKVView:
    gathered = cache.gather()
    if gathered is None:
        raise ValueError("empty segmented KV cache")
    return GatheredKVView(gathered[0], gathered[1], cache.offset)


# ---- prefill over segments: per segment (out, lse) plus one merge ------------------------


def _segment_sdpa_lse_reference(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    n: int,
    *,
    scale: float,
    causal: bool,
    q_chunk_bytes: int = 256 * 1024 * 1024,
) -> tuple[mx.array, mx.array]:
    """Exact attention of ``queries`` over the first ``n`` rows of one segment, with its logsumexp.

    ``queries`` (1, Hq, Q, D); returns (out in the query dtype, lse float32 (1, Hq, Q)).
    ``causal=True`` marks the segment holding the Q new rows as its last Q rows (query row i
    sees keys up to ``n - Q + i``); otherwise every key is visible. Scores run in float32 and
    are bounded by chunking the query rows: this is the slow, exact reference route (QK^T plus
    logsumexp; phase 1 estimated about 3 times the contiguous prefill at 80K). An MLX attention
    kernel that returns the logsumexp replaces this function (``segment_sdpa_lse``).
    """
    b, hq, q_len, d = (int(x) for x in queries.shape)
    hk = int(keys.shape[1])
    gqa = hq // hk
    k = _rows_view(keys, n).astype(mx.float32)
    v = _rows_view(values, n).astype(mx.float32)
    per_row = hk * gqa * int(n) * 4
    chunk = max(1, min(q_len, q_chunk_bytes // max(1, per_row)))
    outs, lses = [], []
    for start in range(0, q_len, chunk):
        stop = min(q_len, start + chunk)
        qc = queries[:, :, start:stop, :].astype(mx.float32)
        rows = stop - start
        qg = qc.reshape(b, hk, gqa * rows, d)
        scores = (qg @ k.transpose(0, 1, 3, 2)) * scale
        if causal:
            row = mx.arange(gqa * rows) % rows + start
            limit = n - q_len + row
            visible = mx.arange(int(n))[None, :] <= limit[:, None]
            scores = mx.where(visible[None, None], scores, -mx.inf)
        lse = mx.logsumexp(scores, axis=-1)
        p = mx.exp(scores - lse[..., None])
        out = (p @ v).reshape(b, hq, rows, d)
        outs.append(out.astype(queries.dtype))
        lses.append(lse.reshape(b, hq, rows))
    if len(outs) == 1:
        return outs[0], lses[0]
    return mx.concatenate(outs, axis=2), mx.concatenate(lses, axis=2)


def _segment_sdpa_lse_fast(queries, keys, values, n, *, scale, causal):
    """Per-segment attention on the MLX kernel that returns the logsumexp (fork only)."""
    k, v = _rows_view(keys, n), _rows_view(values, n)
    kwargs = {"mask": "causal"} if causal else {}
    out, lse = mx.fast.scaled_dot_product_attention(
        queries, k, v, scale=scale, return_lse=True, **kwargs
    )
    # The kernel's logsumexp is (B, Hq, Q, 1); the merge works on (B, Hq, Q).
    return out, lse.astype(mx.float32).reshape(out.shape[:-1])


def segment_sdpa_lse(queries, keys, values, n, *, scale, causal):
    """Per segment (out, lse): the MLX kernel when present, else the exact reference route."""
    if sdpa_lse_available():
        _count("sdpa_lse_kernel")
        return _segment_sdpa_lse_fast(queries, keys, values, n, scale=scale, causal=causal)
    _count("sdpa_lse_reference")
    return _segment_sdpa_lse_reference(queries, keys, values, n, scale=scale, causal=causal)


def merge_segment_outputs(outs: Sequence[mx.array], lses: Sequence[mx.array]) -> mx.array:
    """Exact merge of per-segment (out, lse): sum_s exp(lse_s - L) * out_s with L = logsumexp(lse)."""
    if len(outs) == 1:
        return outs[0]
    stacked = mx.stack([l.astype(mx.float32) for l in lses], axis=0)
    total = mx.logsumexp(stacked, axis=0)
    merged = None
    for out, lse in zip(outs, lses):
        weight = mx.exp(lse.astype(mx.float32) - total)[..., None]
        term = out.astype(mx.float32) * weight
        merged = term if merged is None else merged + term
    return merged.astype(outs[0].dtype)


def attend_segments_lse(queries: mx.array, cache: SegmentedKVCache, *, scale: float) -> mx.array:
    """Attention of the cache's last Q rows (``queries``) over all of its segments, via (out, lse)."""
    segments = cache.attention_segments()
    outs, lses = [], []
    for index, (keys, values, n) in enumerate(segments):
        out, lse = segment_sdpa_lse(
            queries, keys, values, n, scale=scale, causal=(index == len(segments) - 1)
        )
        outs.append(out)
        lses.append(lse)
    return merge_segment_outputs(outs, lses)


# ---- the decode/verify route used by attention_split -----------------------------------


#: Verify windows (rows per call) the segmented decode route serves; bigger calls are prefill chunks.
VERIFY_WINDOW_MAX = 32


def decode_segments_attention(
    queries: mx.array,
    cache: SegmentedKVCache,
    *,
    scale: float,
    mask: Any,
    packed_threshold: int,
) -> mx.array | None:
    """Verify/decode attention straight over the segments, or None for the gather fallback.

    Engages under the same conditions as the packed route over a dense cache, with the total
    length where that route reads the buffer capacity: MTPLX_NAX_FLASH_ROUTE on, a verify
    window of 2 to 8 rows, total length at least ``packed_threshold``, a causal or no mask.
    The head-dim-split kernel serves windows of up to 5 rows at GQA 6 (wide and dsplit4 are not
    in this tree); a longer window, an unsupported shape or a kernel bail returns None and the
    call runs on the gathered rows through the stock ladder.
    """
    from .kernel_selfcheck import lane_disabled
    from .kernels.sdpa_segmented import (
        sdpa_nax_flash_dsplit_segments,
        segments_supported,
    )

    if os.environ.get("MTPLX_NAX_FLASH_ROUTE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if lane_disabled("nax_flash_dsplit_sdpa"):
        return None
    q_len = int(queries.shape[2])
    if not 1 <= q_len <= VERIFY_WINDOW_MAX:
        return None
    if cache.offset < int(packed_threshold):
        return None
    if not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return None
    hq, d = int(queries.shape[1]), int(queries.shape[3])
    segments = cache.attention_segments()
    if not segments:
        return None
    hk = int(segments[0][0].shape[1])
    if hk <= 0 or hq % hk:
        return None
    gqa = hq // hk
    if segments_supported(q_len, gqa, d):
        out = sdpa_nax_flash_dsplit_segments(queries=queries, segments=segments, scale=scale)
        if out is not None:
            _count(f"fused_q{q_len}")
        return out
    # A window the kernel does not take whole (6 to 32 rows without a wide route): sub-windows of
    # the rows it does take, each over the keys up to its own last row, so the history is read
    # once per sub-window and never gathered. Row i of the window sees keys up to n - Q + i.
    sub = max(1, 32 // gqa)
    if not segments_supported(sub, gqa, d):
        return None
    tail_keys, tail_values, tail_n = segments[-1]
    if tail_n < q_len:
        return None
    outs = []
    for r0 in range(0, q_len, sub):
        r1 = min(q_len, r0 + sub)
        trimmed = segments[:-1] + [(tail_keys, tail_values, tail_n - (q_len - r1))]
        part = sdpa_nax_flash_dsplit_segments(
            queries=queries[:, :, r0:r1, :], segments=trimmed, scale=scale
        )
        if part is None:
            return None
        outs.append(part)
    _count(f"chunked_q{q_len}")
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


# ---- install --------------------------------------------------------------------------------


def is_segmented(entry: Any) -> bool:
    return isinstance(entry, SegmentedKVCache)


def segmented_kv_min_tokens() -> int:
    """Context length above which a layer is kept as segments (MTPLX_SEGMENTED_KV).

    The segment cache cannot be promoted to the compiled verify (graphbank:
    ``promotion_failure:segmented_kv_cache``), so below the compiled-verify context ceiling the
    stock cache stays: production runs compiled there and the switch would only slow it down.
    ``MTPLX_SEGMENTED_KV_MIN_TOKENS`` sets the boundary explicitly (0 = always segmented).
    Otherwise it is the compiled-verify ceiling itself (``MTPLX_COMPILED_VERIFY_MAX_CONTEXT``,
    32768 in the turbo profile, 6144 without a profile), read per call; with the compiled verify
    off the boundary is 0, and with the compiled verify unlimited (ceiling 0) it is infinite,
    because a compiled verify that runs at every length leaves the segments nothing to win.
    A layer is segmented when its rows are strictly above the boundary, the same comparison the
    compiled verify uses to fall back to eager.
    """
    raw = os.environ.get("MTPLX_SEGMENTED_KV_MIN_TOKENS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    from .graphbank import _compiled_verify_max_context, compiled_verify_mode

    if compiled_verify_mode() == "off":
        return 0
    ceiling = _compiled_verify_max_context()
    return ceiling if ceiling > 0 else (1 << 62)


def _is_stock_plain_kv(entry: Any) -> bool:
    return type(entry).__name__ == "KVCache" and getattr(entry, "_idx", None) is None


def _segmentable(entry: Any) -> bool:
    """A stock full-attention layer of the target cache (tagged at install; MTP caches never are)."""
    return bool(getattr(entry, "_mtplx_segmentable", False)) and _is_stock_plain_kv(entry)


def _segmented_from_stock(entry: Any) -> SegmentedKVCache:
    """The stock cache's rows as one sealed segment, without a copy.

    The segment views the stock buffers and is never written again; the stock object is dropped
    by the caller, so nothing writes into those buffers any more.
    """
    new = SegmentedKVCache(step=int(getattr(entry, "step", 256)))
    rows = int(getattr(entry, "offset", 0) or 0)
    if rows and entry.keys is not None:
        new.state = (entry.keys[..., :rows, :], entry.values[..., :rows, :])
    return new


def adapt_layer_for_restore(cache: list[Any], idx: int, state: Any) -> tuple[Any, Any]:
    """Pick the layout of layer ``idx`` for a restore of ``state``; returns (entry, state).

    Above ``segmented_kv_min_tokens()`` the layer is a segment cache (a stock snapshot becomes
    one sealed segment, no copy); at or below it the layer is a stock cache (a segmented
    snapshot is gathered into contiguous rows), so a conversation that crosses the boundary
    changes layout at its next restore. Layers that were never tagged (MTP caches, recurrent
    layers) are left alone. The entry is replaced in ``cache`` when its class changes.
    """
    entry = cache[idx]
    if not (isinstance(entry, SegmentedKVCache) or _segmentable(entry)):
        return entry, state
    if isinstance(state, SegmentedKVState):
        rows = state.rows
    elif isinstance(state, (tuple, list)) and len(state) == 2 and state[0] is not None:
        rows = int(state[0].shape[2])
    else:
        return entry, state
    floor = segmented_kv_min_tokens()
    want = rows > floor if floor > 0 else isinstance(entry, SegmentedKVCache)
    if want and not isinstance(entry, SegmentedKVCache):
        entry = SegmentedKVCache(step=int(getattr(entry, "step", 256)))
        entry._mtplx_segmentable = True
        cache[idx] = entry
    elif not want and isinstance(entry, SegmentedKVCache):
        from mlx_lm.models.cache import KVCache

        entry = KVCache()
        entry._mtplx_segmentable = True
        cache[idx] = entry
    if isinstance(state, SegmentedKVState) and not isinstance(entry, SegmentedKVCache):
        state = state.to_arrays()
    return entry, state


def install_segmented_attention_kv_cache(cache: list[Any], *, step: int | None = None) -> dict[str, int | str]:
    """Choose the layout of the full-attention KV layers of a target cache list.

    Plain dense ``KVCache`` layers are tagged as segmentable. A layer without rows becomes a
    segment cache straight away when the boundary is 0 (``segmented_kv_min_tokens``); otherwise
    it stays stock (compiled verify below the boundary) and is converted after its prefill when
    its rows are above the boundary (this function runs again from ``repage_target_prefill_cache``)
    or at the next restore (``adapt_layer_for_restore``). A stock layer with rows above the
    boundary becomes one sealed segment (no copy). Recurrent layers, rotating/indexed caches and
    caches of other lists (MTP) are left alone.
    """
    stats: dict[str, int | str] = {"enabled": 1, "mode": "segmented", "entries": 0, "skipped": 0, "converted": 0}
    from .cache_state import _is_trimmable

    floor = segmented_kv_min_tokens()
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, SegmentedKVCache):
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if not _is_trimmable(entry) or not _is_stock_plain_kv(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        rows = int(getattr(entry, "offset", 0) or 0)
        if rows == 0 and getattr(entry, "keys", None) is None:
            if floor == 0:
                new = SegmentedKVCache(step=int(step or getattr(entry, "step", 256)))
                new._mtplx_segmentable = True
                cache[idx] = new
                stats["entries"] = int(stats["entries"]) + 1
            else:
                entry._mtplx_segmentable = True
                stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_mtplx_segmentable", False) and rows > floor > 0:
            cache[idx] = _segmented_from_stock(entry)
            cache[idx]._mtplx_segmentable = True
            stats["entries"] = int(stats["entries"]) + 1
            stats["converted"] = int(stats["converted"]) + 1
            continue
        stats["skipped"] = int(stats["skipped"]) + 1
    return stats


# ---- accounting -----------------------------------------------------------------------------


def unique_state_bytes(states: Iterable[Any]) -> int:
    """Bytes of the segments referenced by ``states`` (SegmentedKVState objects), each counted once."""
    seen: dict[int, int] = {}
    for state in states:
        if isinstance(state, SegmentedKVState):
            for ref in state.refs:
                seen.setdefault(ref.segment.id, ref.segment.nbytes)
    return sum(seen.values())


def walk_segmented_states(value: Any) -> Iterable[SegmentedKVState]:
    """Every SegmentedKVState inside a snapshot tree (tuples, lists, dicts)."""
    if isinstance(value, SegmentedKVState):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_segmented_states(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_segmented_states(item)
