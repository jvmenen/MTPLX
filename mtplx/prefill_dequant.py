"""Dequantize-then-bf16-matmul for wide prefill chunks (opt-in).

``MTPLX_PREFILL_DEQUANT_MIN_ROWS=N`` (N >= 1, default off) makes every affine
``nn.QuantizedLinear`` call in the prefill phase with at least N input rows run
``x @ mx.dequantize(...).T`` instead of ``mx.quantized_matmul``. Decode, verify
and short prefill forwards keep the stock kernel. The bf16 weight is a
per-call transient (one projection, e.g. 17408x5120 = 178 MB); nothing is
cached across calls or layers.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from mlx import nn

PREFILL_DEQUANT_ENV = "MTPLX_PREFILL_DEQUANT_MIN_ROWS"
_PATCH: dict[str, Any] = {"installed": False, "original": None}
_COUNTS = {"dequant_calls": 0}


def prefill_dequant_min_rows() -> int:
    """The row threshold from the environment; 0 means the lane is off."""

    try:
        return max(0, int(os.environ.get(PREFILL_DEQUANT_ENV, "0") or 0))
    except ValueError:
        return 0


def prefill_dequant_counts() -> dict[str, int]:
    return dict(_COUNTS)


def dequantized_linear(layer: nn.QuantizedLinear, x: mx.array) -> mx.array:
    weight = mx.dequantize(
        layer["weight"],
        layer["scales"],
        layer["biases"],
        group_size=layer.group_size,
        bits=layer.bits,
        mode=layer.mode,
        dtype=x.dtype,
    )
    y = x @ weight.T
    if "bias" in layer:
        y = y + layer["bias"]
    return y


def install_prefill_dequant_patch() -> dict[str, object]:
    """Wrap ``nn.QuantizedLinear.__call__``. Idempotent; call after any other
    QuantizedLinear patch so this one sits outermost and falls through to it."""

    if _PATCH["installed"]:
        return {"installed": True, "already": True}
    from .attention_context import current_attention_phase

    min_rows = prefill_dequant_min_rows()
    if min_rows <= 0:
        return {"installed": False, "reason": "disabled"}
    original = nn.QuantizedLinear.__call__

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        if (
            self.mode == "affine"
            and self.get("biases") is not None
            and x.ndim >= 2
            and x.size // int(x.shape[-1]) >= min_rows
            and current_attention_phase() == "prefill"
        ):
            _COUNTS["dequant_calls"] += 1
            return dequantized_linear(self, x)
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _PATCH.update(installed=True, original=original)
    return {"installed": True, "already": False, "min_rows": min_rows}


def uninstall_prefill_dequant_patch() -> None:
    if _PATCH["installed"]:
        nn.QuantizedLinear.__call__ = _PATCH["original"]
        _PATCH.update(installed=False, original=None)
