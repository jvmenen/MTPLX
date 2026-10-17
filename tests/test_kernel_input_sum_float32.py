"""Kernel input sums accumulate in float (source scan, no GPU).

The affine qmv kernels fold the quantization bias in as ``bias * sum(x)``.
Since MLX 0.32.4 (ml-explore/mlx#4516) stock qmv casts every activation to
float before it adds it to that sum. Adding two bf16 or fp16 values first
rounds the partial sum to 16 bits, and a kernel that does so differs from
stock qmv in the last bit. This scan fails on any accumulation that adds two
or more raw 16-bit values, so a new copy of the loader cannot bring the old
form back.
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

# Template parameters and 16-bit Metal types. ``{t}`` covers kernel sources
# that splice the dtype in with an f-string.
_NARROW_TYPE = r"(?:T|bfloat|bfloat16_t|half|\{\w+\})"
_NARROW_POINTER = re.compile(
    rf"\bdevice\s+(?:const\s+)?{_NARROW_TYPE}\s*\*\s*(?:const\s+)?(\w+)"
)
_NARROW_SCALAR = re.compile(rf"(?<![\w*])(?:const\s+)?{_NARROW_TYPE}\s+(\w+)\s*=")
_ACCUMULATION = re.compile(r"(?:\w+\s*\+=|\bfloat\s+\w+\s*=)([^;{}]+);")
_SUBSCRIPT = re.compile(r"^(\w+)\s*\[[^\[\]]+\]$")
_IDENTIFIER = re.compile(r"^\w+$")


def _terms(expression: str) -> list[str]:
    terms: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "+" and depth == 0:
            terms.append(expression[start:index].strip())
            start = index + 1
    terms.append(expression[start:].strip())
    return terms


def narrow_input_sums(text: str) -> list[tuple[int, str]]:
    """Line number and statement of each sum that adds raw 16-bit values."""
    pointers = set(_NARROW_POINTER.findall(text))
    scalars = set(_NARROW_SCALAR.findall(text))
    found = []
    for match in _ACCUMULATION.finditer(text):
        narrow = 0
        for term in _terms(match.group(1)):
            subscript = _SUBSCRIPT.match(term)
            if (subscript and subscript.group(1) in pointers) or (
                _IDENTIFIER.match(term) and term in scalars
            ):
                narrow += 1
        if narrow >= 2:
            line = text.count("\n", 0, match.start()) + 1
            found.append((line, " ".join(match.group(0).split())))
    return found


def test_scan_flags_the_bf16_input_sum():
    pointer_loader = """
        inline float load(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < 16; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          }
          return sum;
        }
    """
    scalar_loader = """
        inline float load(const device T* gate, const device T* up) {
          float sum = 0.0f;
          T x0 = swiglu<T>(gate[0], up[0]);
          T x1 = swiglu<T>(gate[1], up[1]);
          sum += x0 + x1;
          return sum;
        }
    """
    assert [line for line, _ in narrow_input_sums(pointer_loader)] == [5]
    assert [line for line, _ in narrow_input_sums(scalar_loader)] == [6]


def test_scan_accepts_float_input_sums():
    loaders = """
        inline float load(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < 16; i += 4) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
            sum += (float)x[i] + static_cast<float>(x[i + 1]);
          }
          float x0 = float(x[0]);
          float x1 = float(x[1]);
          sum += x0 + x1;
          thread float* partial;
          float total = partial[0] + partial[1];
          return sum + total;
        }
    """
    assert narrow_input_sums(loaders) == []


def test_no_kernel_adds_raw_16_bit_inputs():
    offenders = []
    scanned = set()
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            scanned.add(path.relative_to(ROOT).as_posix())
            text = path.read_text(encoding="utf-8")
            for line, statement in narrow_input_sums(text):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {statement}")
    assert "mtplx/kernels/qmv4_matmul.py" in scanned
    assert not offenders, (
        "cast each 16-bit value to float before adding it to the input sum "
        "(as MLX 0.32.4 qmv does, ml-explore/mlx#4516):\n" + "\n".join(offenders)
    )
