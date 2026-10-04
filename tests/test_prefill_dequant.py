import mlx.core as mx
import pytest
from mlx import nn

from mtplx import prefill_dequant as pd
from mtplx.attention_context import attention_phase


def _layer(bits=4, group=32, bias=False):
    mx.random.seed(0)
    lin = nn.Linear(256, 192, bias=bias)
    q = nn.QuantizedLinear.from_linear(lin, group_size=group, bits=bits)
    q.set_dtype(mx.bfloat16)
    return q


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setenv(pd.PREFILL_DEQUANT_ENV, "64")
    original = nn.QuantizedLinear.__call__
    pd._COUNTS["dequant_calls"] = 0
    assert pd.install_prefill_dequant_patch()["installed"]
    yield
    pd.uninstall_prefill_dequant_patch()
    assert nn.QuantizedLinear.__call__ is original


@pytest.mark.parametrize("bits,bias", [(4, False), (4, True), (8, False)])
def test_matches_quantized_matmul(bits, bias):
    layer = _layer(bits=bits, bias=bias)
    x = mx.random.normal((100, 256)).astype(mx.bfloat16)
    ref = nn.QuantizedLinear.__call__(layer, x)
    out = pd.dequantized_linear(layer, x)
    assert out.dtype == ref.dtype
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=0.1, rtol=0.05).item()


def test_routes_only_wide_prefill(patched):
    layer = _layer()
    wide = mx.random.normal((1, 128, 256)).astype(mx.bfloat16)
    narrow = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)
    with attention_phase("prefill"):
        layer(narrow)
        assert pd.prefill_dequant_counts()["dequant_calls"] == 0
        layer(wide)
        assert pd.prefill_dequant_counts()["dequant_calls"] == 1
    with attention_phase("decode_verify"):
        layer(wide)
    assert pd.prefill_dequant_counts()["dequant_calls"] == 1


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pd.PREFILL_DEQUANT_ENV, raising=False)
    assert pd.install_prefill_dequant_patch() == {"installed": False, "reason": "disabled"}
