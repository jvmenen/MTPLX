"""Batch-invariant prefill matmuls (opt-in, MTPLX_BATCH_INVARIANT_PREFILL=1).

Problem: MLX picks its quantized-matmul kernel on the row count M of each
forward, and those kernels round differently. On MLX 0.32 (quantized.cpp):

- ``quantized_matmul`` with 2-D weights runs ``qmm_splitk`` whenever the
  32x32 output tiles number fewer than ~512; the split count follows
  ``512 / (m_tiles * n_tiles)``, so for narrow outputs (MoE router N=256,
  shared expert N=512, GDN a/b N=32, k/v N=512) a row's result depends on how
  many rows share the forward, up to M=1024 and beyond. Batched calls skip
  split-K and run the NAX ``qmm`` whose rows are independent once each batch
  holds more than 32 rows (``bm`` 64).
- ``gather_qmm`` (routed experts) runs the tiled ``gather_qmm_rhs`` only with
  at least 4 token-expert rows per expert; smaller forwards fall back to the
  per-row ``gather_qmv``, which rounds differently.
- ``scaled_dot_product_attention`` with head dim 256 runs the fused causal
  kernel from 1024 query rows, an unfused matmul route from 9 to 1023 and the
  vector kernel below 9.

On Qwen3.6-35B-A3B the 8th and 9th expert often score within one bf16 step,
so those rounding differences change the routing and move scores ~0.2 nats
per token between block layouts (256 against 2048 rows, token by token, a
64-row tail forward).

Fix, prefill phase only (decode and verify keep the stock kernels): every
affine ``QuantizedLinear`` runs as a two-batch matmul of at least 33 rows
per batch (zero padded), every ``SwitchGLU`` pads its tokens so the sorted
tiled kernel always runs, and Qwen3-Next full attention always runs the fused
causal kernel (zero query rows in front below 9 rows). All three reuse stock
kernels; padding costs work only on forwards narrower than the minimums. The
GDN layers needed nothing. The switch is read once at construction, like the
other MoE knobs.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.switch_layers import SwitchGLU

from .attention_context import current_attention_phase

BATCH_INVARIANT_PREFILL_ENV = "MTPLX_BATCH_INVARIANT_PREFILL"

# Two batches of more than 32 rows each: MLX's batched NAX qmm then uses the
# same 64-row tiles and no split-K at every row count.
_LINEAR_BATCHES = 2
_MIN_LINEAR_ROWS = 2 * 33
# gather_qmm_rhs needs at least this many token-expert rows per expert.
_MIN_ROWS_PER_EXPERT = 4
# Fused SDPA: more than 8 query rows selects the full (not the vector) kernel.
_MIN_FUSED_QUERY_ROWS = 9
_FUSED_HEAD_DIMS = frozenset({64, 72, 80, 96, 128, 192, 256})
_STOCK_SDPA: dict[str, Any] = {"sdpa": None}
_STATE = {"installed": False}


def batch_invariant_prefill_enabled() -> bool:
    """Whether construction installs the batch-invariant prefill lane."""

    value = os.environ.get(BATCH_INVARIANT_PREFILL_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def batch_invariant_prefill_installed() -> bool:
    """Whether this process runs its prefill through the invariant lane, so
    callers may choose forward widths freely without changing results."""

    return _STATE["installed"]


def _in_prefill() -> bool:
    return current_attention_phase() == "prefill"


def _pad_rows(rows: mx.array, target: int) -> mx.array:
    missing = target - int(rows.shape[0])
    if missing <= 0:
        return rows
    padding = mx.zeros((missing,) + tuple(rows.shape[1:]), dtype=rows.dtype)
    return mx.concatenate([rows, padding], axis=0)


def _broadcast_batches(array: mx.array | None) -> mx.array | None:
    if array is None:
        return None
    return mx.broadcast_to(array, (_LINEAR_BATCHES,) + tuple(array.shape))


def split_k_free_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> mx.array:
    """``x @ dequantize(weight).T`` with a result per row that does not depend
    on how many rows ``x`` holds."""

    in_features = int(x.shape[-1])
    rows = x.reshape(-1, in_features)
    row_count = int(rows.shape[0])
    padded = max(_MIN_LINEAR_ROWS, row_count + row_count % _LINEAR_BATCHES)
    batched = _pad_rows(rows, padded).reshape(
        _LINEAR_BATCHES, padded // _LINEAR_BATCHES, in_features
    )
    out = mx.quantized_matmul(
        batched,
        _broadcast_batches(weight),
        _broadcast_batches(scales),
        _broadcast_batches(biases),
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    out = out.reshape(padded, -1)[:row_count]
    return out.reshape(*x.shape[:-1], int(out.shape[-1]))


class BatchInvariantQuantizedLinear(nn.QuantizedLinear):
    """``QuantizedLinear`` whose prefill rows are independent of the row count."""

    def __call__(self, x: mx.array) -> mx.array:
        if not _in_prefill():
            return super().__call__(x)
        out = split_k_free_quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            out = out + self["bias"]
        return out


class BatchInvariantSwitchGLU(SwitchGLU):
    """``SwitchGLU`` that always runs the sorted tiled expert kernel in prefill."""

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if not _in_prefill():
            return super().__call__(x, indices)
        experts = int(self.gate_proj["weight"].shape[0])
        top_k = int(indices.shape[-1])
        min_tokens = -(-_MIN_ROWS_PER_EXPERT * experts // top_k)
        tokens = x.reshape(-1, int(x.shape[-1]))
        token_count = int(tokens.shape[0])
        if token_count >= min_tokens:
            return super().__call__(x, indices)
        # Padding tokens use the last top_k experts: valid ids, and their
        # rows only extend the tail of the sorted expert list.
        pad_ids = mx.broadcast_to(
            mx.arange(experts - top_k, experts, dtype=indices.dtype),
            (min_tokens - token_count, top_k),
        )
        routes = mx.concatenate([indices.reshape(-1, top_k), pad_ids], axis=0)
        out = super().__call__(_pad_rows(tokens, min_tokens), routes)
        return out[:token_count].reshape(*indices.shape, int(out.shape[-1]))


def _fused_attention_rows(
    queries: mx.array, keys: mx.array, mask: Any, cache: Any
) -> int | None:
    """Padded query rows for the fused causal kernel, or None when this call
    must keep the stock route (array mask, quantized KV, too few keys)."""

    if hasattr(cache, "bits") or not (mask is None or mask == "causal"):
        return None
    query_rows = int(queries.shape[2])
    padded = max(query_rows, _MIN_FUSED_QUERY_ROWS)
    if padded > int(keys.shape[2]) or int(queries.shape[-1]) not in _FUSED_HEAD_DIMS:
        return None
    return padded


def batch_invariant_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: Any,
    sinks: mx.array | None = None,
) -> mx.array:
    """mlx-lm's ``scaled_dot_product_attention`` with, in prefill, one fused
    causal kernel for every query count.

    Stock MLX runs the fused kernel only from 1024 query rows (head dim 256)
    and an unfused matmul-softmax-matmul below that, so a row's attention
    depends on the forward's width. Fewer than 9 query rows would pick the
    vector kernel; zero query rows in front of them keep the causal diagonal
    of the real rows and are dropped after the call."""

    padded = _fused_attention_rows(queries, keys, mask, cache) if _in_prefill() else None
    if padded is None:
        return _STOCK_SDPA["sdpa"](
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
        )
    lead = padded - int(queries.shape[2])
    if lead:
        shape = list(queries.shape)
        shape[2] = lead
        queries = mx.concatenate([mx.zeros(shape, dtype=queries.dtype), queries], axis=2)
    out = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask="causal",
        sinks=sinks,
        force_fused=True,
    )
    return out[:, :, lead:]


def _install_attention_route() -> bool:
    from mlx_lm.models import qwen3_next

    if qwen3_next.scaled_dot_product_attention is batch_invariant_sdpa:
        return False
    _STOCK_SDPA["sdpa"] = qwen3_next.scaled_dot_product_attention
    qwen3_next.scaled_dot_product_attention = batch_invariant_sdpa
    return True


def _swap_class(module: Any, new_class: type) -> None:
    module.__class__ = new_class


def install_batch_invariant_prefill(model: Any) -> dict[str, int]:
    """Route every affine ``QuantizedLinear`` and stock ``SwitchGLU`` of
    ``model``, and the Qwen3-Next attention, through the batch-invariant
    prefill lane. Class swaps and one function hook: the parameter tree and
    the decode path stay exactly as loaded."""

    report = {
        "linears": 0,
        "switch_glus": 0,
        "skipped_linears": 0,
        "attention_hooked": int(_install_attention_route()),
    }
    for _name, module in model.named_modules():
        if type(module) is nn.QuantizedLinear:
            if module.mode == "affine":
                _swap_class(module, BatchInvariantQuantizedLinear)
                report["linears"] += 1
            else:
                report["skipped_linears"] += 1
        elif type(module) is SwitchGLU:
            _swap_class(module, BatchInvariantSwitchGLU)
            report["switch_glus"] += 1
    _STATE["installed"] = True
    return report
