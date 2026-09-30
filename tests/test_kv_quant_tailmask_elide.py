"""MTPLX_KV_QUANT_TAILMASK_ELIDE: the quantized paged adapter serves its own mask.

``TensorOffsetQuantizedPagedKVCache`` inherits ``make_mask`` from the bf16
paged adapter, which always emits the capacity-wide tail-causal bool mask.
Its ``paged_attention`` refused every array mask, so each verify round fell
back to ``cache.state`` (a full-capacity dequantization per layer). With the
switch on, that exact mask is dropped because the packed-quant kernel's
built-in visibility is the same; any other mask still declines.
"""

import mlx.core as mx
import pytest

from mtplx.cache_state import TensorOffsetQuantizedPagedKVCache, VllmMetalPagedKVCache
from mtplx.kv_quant import PagedKVQuantConfig

METAL = mx.metal.is_available()
HK, GQA, D, BLOCK, BLOCKS = 4, 6, 256, 16, 8
HQ = HK * GQA
SCALE = D**-0.5


def _adapter(mode: str, prefix: int) -> TensorOffsetQuantizedPagedKVCache:
    paged = VllmMetalPagedKVCache(
        block_size=BLOCK,
        num_blocks=BLOCKS,
        kv_quant_config=PagedKVQuantConfig(mode),
    )
    paged.update_without_fetch(
        mx.random.normal((1, HK, prefix, D)).astype(mx.bfloat16),
        mx.random.normal((1, HK, prefix, D)).astype(mx.bfloat16),
    )
    return TensorOffsetQuantizedPagedKVCache.from_paged_cache(paged)


def _verify_step(adapter, q_len: int):
    """One verify-shaped call: mask first (pre-write offset), then the write."""
    mask = adapter.make_mask(q_len)
    adapter.update_without_fetch(
        mx.random.normal((1, HK, q_len, D)).astype(mx.bfloat16),
        mx.random.normal((1, HK, q_len, D)).astype(mx.bfloat16),
    )
    queries = mx.random.normal((1, HQ, q_len, D)).astype(mx.bfloat16)
    return queries, mask


def _dense_fallback(adapter, queries, mask):
    """What attention_split does when paged_attention returns None."""
    keys, values = adapter.state
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=SCALE, mask=mask
    )


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_switch_off_keeps_declining_the_array_mask(monkeypatch):
    monkeypatch.delenv("MTPLX_KV_QUANT_TAILMASK_ELIDE", raising=False)
    mx.random.seed(1)
    adapter = _adapter("q8", 37)
    queries, mask = _verify_step(adapter, 4)
    assert adapter.paged_attention(queries, scale=SCALE, mask=mask) is None


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("mode", ["q8", "q4"])
@pytest.mark.parametrize("prefix,q_len", [(37, 4), (60, 3), (93, 5)])
def test_elided_kernel_matches_dense_fallback(monkeypatch, mode, prefix, q_len):
    monkeypatch.setenv("MTPLX_KV_QUANT_TAILMASK_ELIDE", "1")
    mx.random.seed(prefix * 10 + q_len)
    adapter = _adapter(mode, prefix)
    queries, mask = _verify_step(adapter, q_len)

    out = adapter.paged_attention(queries, scale=SCALE, mask=mask)
    assert out is not None, "packed-quant kernel declined the adapter's own mask"
    assert adapter.paged_attention_calls == 1

    ref = _dense_fallback(adapter, queries, mask)
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert err <= 2e-2, f"elide changed the attention result: max err {err}"


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_foreign_mask_still_declines(monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_TAILMASK_ELIDE", "1")
    mx.random.seed(3)
    adapter = _adapter("q8", 37)
    queries, mask = _verify_step(adapter, 4)
    narrower = mask[:, : int(adapter.capacity) - 1]
    assert adapter.paged_attention(queries, scale=SCALE, mask=narrower) is None
    additive = mx.where(mask, 0.0, -1e9).astype(mx.bfloat16)
    assert adapter.paged_attention(queries, scale=SCALE, mask=additive) is None
