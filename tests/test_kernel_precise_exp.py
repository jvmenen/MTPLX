"""Kernel copies of stock Sigmoid and compute_g use the precise exp (source scan, no GPU).

An ``mx.fast.metal_kernel`` does not get the metallib's ``-fno-fast-math``:
a plain ``metal::exp`` there is the fast approximation. Stock ``Exp`` has
always spelled ``metal::precise::exp``, and since MLX 0.32.3
(ml-explore/mlx#4461) stock ``Sigmoid`` does too, so eager and compiled
sigmoids agree. Two kinds of kernel code copy those ops to match stock bit
for bit, and must spell the precise exp as well:

* the ``unary_ops.h`` sigmoid, ``1 / (1 + exp(metal::abs(x)))``;
* the GDN decay of a kernel that reproduces ``compute_g`` with stock's
  ``LogAddExp`` softplus: ``exp(A_log)`` and ``exp(-exp(A_log) * softplus)``.

The softmax and logsumexp exps stay fast, as in stock ``softmax.h`` and
``logsumexp.h``, and so does the exp inside the ``LogAddExp`` softplus, which
stock spells as a plain ``metal::exp``. Kernels with their own sigmoid form,
such as ``x / (1 + exp(-x))``, do not reproduce stock bits with either exp and
are not scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = (
    "mtplx/**/*.py",
    "native_extensions/**/*.metal",
    "native_extensions/**/*.h",
)

_NOT_PRECISE = r"(?:metal::fast::|metal::|fast::)exp\s*\("
_STOCK_SIGMOID = re.compile(rf"\b1(?:\.0f?)?\s*\+\s*{_NOT_PRECISE}\s*metal::abs\(")
_STOCK_SOFTPLUS = "log1p(metal::exp(minval - maxval))"
_DECAY = re.compile(rf"\b{_NOT_PRECISE}([^;]*)")
_DECAY_ARGUMENT = re.compile(r"\bA_log\b|\bdecay_a\b")


def fast_exp_sites(text: str) -> list[tuple[int, str]]:
    """Line number and line of each stock-op copy that uses a non-precise exp."""
    starts = [match.start() for match in _STOCK_SIGMOID.finditer(text)]
    if _STOCK_SOFTPLUS in text:
        starts += [
            match.start()
            for match in _DECAY.finditer(text)
            if _DECAY_ARGUMENT.search(match.group(1))
        ]
    found = []
    for start in sorted(starts):
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        line = text.count("\n", 0, start) + 1
        found.append((line, text[line_start:line_end].strip()))
    return found


def test_scan_flags_fast_exp_in_stock_sigmoid_copies():
    sources = """
        auto y = 1 / (1 + metal::exp(metal::abs(x)));
        float y = 1.0f / (1.0f + metal::fast::exp(metal::abs(x)));
        auto sigmoid_y = 1 / (
            1 + metal::exp(metal::abs(gate_value)));
    """
    assert [line for line, _ in fast_exp_sites(sources)] == [2, 3, 5]


def test_scan_flags_fast_exp_in_a_compute_g_copy():
    sources = """
        InT softplus_val = (minval == -inf || maxval == inf)
          ? maxval
          : (maxval + log1p(metal::exp(minval - maxval)));
        float decay_a = metal::exp(float(A_log[hv_idx]));
        g_shared = metal::exp(-decay_a * float(softplus_val));
    """
    assert [line for line, _ in fast_exp_sites(sources)] == [5, 6]


def test_scan_accepts_precise_exp_softmax_softplus_and_own_forms():
    sources = """
        auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        InT softplus_val = (minval == -inf || maxval == inf)
          ? maxval
          : (maxval + log1p(metal::exp(minval - maxval)));
        float decay_a = metal::precise::exp(float(A_log[hv_idx]));
        g_shared = metal::precise::exp(-decay_a * float(softplus_val));
        float probability = metal::exp(local_logit - simd_values[0]);
        float exp_score = fast::exp(score - new_max);
        const float sw = gv / (1.0f + metal::exp(-gv));
        const float sp = metal::max(a_v, 0.0f)
                       + metal::log(1.0f + metal::exp(-metal::abs(a_v)));
    """
    assert fast_exp_sites(sources) == []
    own_decay = """
        const float sp = metal::max(a_v, 0.0f)
                       + metal::log(1.0f + metal::exp(-metal::abs(a_v)));
        const float g_dec = metal::exp(-metal::exp((float)A_log[hv]) * sp);
    """
    assert fast_exp_sites(own_decay) == []


def test_stock_op_copies_use_precise_exp():
    offenders = []
    scanned = set()
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            scanned.add(path.relative_to(ROOT).as_posix())
            text = path.read_text(encoding="utf-8")
            for line, statement in fast_exp_sites(text):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {statement}")
    assert "mtplx/kernels/verify_mlp_fused.py" in scanned
    assert not offenders, (
        "spell metal::precise::exp where a kernel copies stock Sigmoid or "
        "compute_g (as stock Sigmoid and Exp do, ml-explore/mlx#4461):\n"
        + "\n".join(offenders)
    )
