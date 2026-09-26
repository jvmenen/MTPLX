"""Batch-invariant prefill lane (MTPLX_BATCH_INVARIANT_PREFILL).

The claim is about which Metal kernels MLX picks per row count, so the parity
tests run on the GPU and skip without Metal. Shapes follow Qwen3.6-35B-A3B
where the stock kernels are known to differ (router 256 x 2048 in 8 bit).
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm.models.qwen3_next")

from mlx_lm.models import qwen3_next
from mlx_lm.models.switch_layers import SwitchGLU

from mtplx import batch_invariant_prefill as bip
from mtplx.attention_context import attention_phase

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _router(bits: int = 8) -> nn.QuantizedLinear:
    mx.random.seed(7)
    linear = nn.Linear(2048, 256, bias=False)
    linear.weight = (mx.random.normal((256, 2048)) * 0.02).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=bits)


def _rows(count: int, width: int = 2048) -> mx.array:
    mx.random.seed(11)
    return mx.random.normal((count, width)).astype(mx.bfloat16)


def _prefix_equal(small: mx.array, big: mx.array) -> bool:
    return bool(mx.array_equal(small, big[: small.shape[0]]).item())


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("on", True), ("TRUE", True), ("", False), ("0", False), ("no", False)],
)
def test_switch_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(bip.BATCH_INVARIANT_PREFILL_ENV, value)
    assert bip.batch_invariant_prefill_enabled() is expected


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(bip.BATCH_INVARIANT_PREFILL_ENV, raising=False)
    assert bip.batch_invariant_prefill_enabled() is False


@needs_metal
def test_stock_router_matmul_depends_on_row_count():
    """The problem this lane exists for: split-K follows the row count."""
    router = _router()
    rows = _rows(2048)
    full = router(rows)
    assert not _prefix_equal(router(rows[:256]), full)


@needs_metal
@pytest.mark.parametrize("count", [1, 3, 17, 64, 100, 255, 331, 1000])
def test_split_k_free_matmul_rows_do_not_depend_on_row_count(count):
    router = _router()
    rows = _rows(2048)

    def route(x):
        return bip.split_k_free_quantized_matmul(
            x,
            router.weight,
            router.scales,
            router.biases,
            group_size=64,
            bits=8,
        )

    assert _prefix_equal(route(rows[:count]), route(rows))


@needs_metal
def test_split_k_free_matmul_matches_stock_numerics_closely():
    router = _router()
    rows = _rows(300)
    stock = router(rows).astype(mx.float32)
    invariant = bip.split_k_free_quantized_matmul(
        rows, router.weight, router.scales, router.biases, group_size=64, bits=8
    ).astype(mx.float32)
    assert invariant.shape == stock.shape
    assert mx.allclose(invariant, stock, atol=0.05, rtol=0.02).item()


@needs_metal
def test_linear_keeps_stock_route_outside_prefill():
    router = _router()
    stock = router(_rows(40))
    router.__class__ = bip.BatchInvariantQuantizedLinear
    with attention_phase("decode_verify"):
        assert mx.array_equal(router(_rows(40)), stock).item()


@needs_metal
def test_linear_is_row_invariant_in_prefill_and_keeps_leading_dims():
    router = _router()
    router.__class__ = bip.BatchInvariantQuantizedLinear
    rows = _rows(1024)
    with attention_phase("prefill"):
        full = router(rows[None])
        part = router(rows[None, :77])
    assert full.shape == (1, 1024, 256)
    assert _prefix_equal(part[0], full[0])


def _switch_glu(experts: int = 64, top_k: int = 8) -> SwitchGLU:
    mx.random.seed(3)
    glu = SwitchGLU(256, 128, experts)
    nn.quantize(glu, group_size=64, bits=4)
    return glu


def _routes(count: int, experts: int = 64, top_k: int = 8) -> mx.array:
    mx.random.seed(5)
    scores = mx.random.uniform(shape=(count, experts))
    return mx.argpartition(scores, kth=-top_k, axis=-1)[:, -top_k:].astype(mx.uint32)


@needs_metal
@pytest.mark.parametrize("count", [1, 5, 20, 31])
def test_switch_glu_small_prefill_matches_wide_prefill(count):
    glu = _switch_glu()
    glu.__class__ = bip.BatchInvariantSwitchGLU
    x = _rows(256, width=256)[None]
    routes = _routes(256)[None]
    with attention_phase("prefill"):
        wide = glu(x, routes)
        narrow = glu(x[:, :count], routes[:, :count])
    assert narrow.shape == (1, count, 8, 256)
    assert _prefix_equal(narrow[0], wide[0])


@needs_metal
def test_switch_glu_keeps_stock_route_outside_prefill():
    glu = _switch_glu()
    x = _rows(3, width=256)[None]
    routes = _routes(3)[None]
    stock = glu(x, routes)
    glu.__class__ = bip.BatchInvariantSwitchGLU
    with attention_phase("ar_decode"):
        assert mx.array_equal(glu(x, routes), stock).item()


def _attention_inputs(tokens: int = 1200):
    mx.random.seed(9)
    queries = mx.random.normal((1, 16, tokens, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
    return queries, keys, values


def _chunked_attention(queries, keys, values, chunk):
    outs = []
    total = int(queries.shape[2])
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        outs.append(
            bip.batch_invariant_sdpa(
                queries[:, :, start:end],
                keys[:, :, :end],
                values[:, :, :end],
                cache=None,
                scale=256**-0.5,
                mask="causal" if end - start > 1 else None,
            )
        )
    return mx.concatenate(outs, axis=2)


@needs_metal
@pytest.mark.parametrize("chunk", [1, 5, 17, 256, 333])
def test_attention_rows_do_not_depend_on_chunking_in_prefill(monkeypatch, chunk):
    monkeypatch.setitem(bip._STOCK_SDPA, "sdpa", qwen3_next.scaled_dot_product_attention)
    queries, keys, values = _attention_inputs()
    with attention_phase("prefill"):
        whole = _chunked_attention(queries, keys, values, 1200)
        parts = _chunked_attention(queries, keys, values, chunk)
    # Rows before the 9th have too few keys to pad in front of them.
    assert mx.array_equal(parts[:, :, 9:], whole[:, :, 9:]).item()


@needs_metal
def test_attention_keeps_stock_route_outside_prefill_and_for_array_masks(monkeypatch):
    stock_sdpa = qwen3_next.scaled_dot_product_attention
    monkeypatch.setitem(bip._STOCK_SDPA, "sdpa", stock_sdpa)
    queries, keys, values = _attention_inputs(64)
    stock = stock_sdpa(queries, keys, values, cache=None, scale=0.0625, mask="causal")
    with attention_phase("decode_verify"):
        routed = bip.batch_invariant_sdpa(
            queries, keys, values, cache=None, scale=0.0625, mask="causal"
        )
    assert mx.array_equal(routed, stock).item()
    bool_mask = mx.tril(mx.ones((64, 64), dtype=mx.bool_))
    stock_masked = stock_sdpa(queries, keys, values, cache=None, scale=0.0625, mask=bool_mask)
    with attention_phase("prefill"):
        routed_masked = bip.batch_invariant_sdpa(
            queries, keys, values, cache=None, scale=0.0625, mask=bool_mask
        )
    assert mx.array_equal(routed_masked, stock_masked).item()


def test_install_swaps_classes_without_touching_parameters(monkeypatch):
    monkeypatch.setattr(qwen3_next, "scaled_dot_product_attention", qwen3_next.scaled_dot_product_attention)
    monkeypatch.setitem(bip._STOCK_SDPA, "sdpa", None)
    monkeypatch.setitem(bip._STATE, "installed", False)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _router(bits=8)
            self.dense = nn.Linear(8, 8)
            self.experts = _switch_glu()

    model = Tiny()
    assert bip.batch_invariant_prefill_installed() is False
    before = sorted(key for key, _ in tree_flatten(model.parameters()))
    report = bip.install_batch_invariant_prefill(model)
    after = sorted(key for key, _ in tree_flatten(model.parameters()))
    assert before == after
    assert report == {
        "linears": 1,
        "switch_glus": 1,
        "skipped_linears": 0,
        "attention_hooked": 1,
    }
    assert type(model.router) is bip.BatchInvariantQuantizedLinear
    assert type(model.dense) is nn.Linear
    assert type(model.experts) is bip.BatchInvariantSwitchGLU
    assert qwen3_next.scaled_dot_product_attention is bip.batch_invariant_sdpa
    assert bip.batch_invariant_prefill_installed() is True
    assert bip.install_batch_invariant_prefill(model)["attention_hooked"] == 0
