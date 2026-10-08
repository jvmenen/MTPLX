"""Models the segment kernel cannot serve keep the stock cache (and say why), never the gather route."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import mtplx.segmented_kv as module
from mtplx.cache_state import configure_tail_owned_attention_kv_cache
from mtplx.segmented_kv import (
    evaluate_model_support,
    model_support,
    reset_model_support,
    segmented_kv_enabled,
    segmented_kv_health,
    segmented_ssd_enabled,
)


@pytest.fixture(autouse=True)
def _cpu_and_clean(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    reset_model_support()
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setattr("mtplx.nax_verify.nax_available", lambda: True)
    yield
    reset_model_support()
    mx.set_default_device(previous)


def _attn(head_dim=256, q=24, kv=4, packed=True):
    return SimpleNamespace(
        head_dim=head_dim, num_attention_heads=q, num_key_value_heads=kv, _mtplx_gqa_packed_sdpa_enabled=packed
    )


def test_a_supported_model_stays_enabled() -> None:
    verdict = evaluate_model_support([_attn(), _attn()])
    assert verdict["supported"] and verdict["reasons"] == []
    assert verdict["head_dims"] == [256] and verdict["gqa"] == [6]
    assert segmented_kv_enabled() is True


@pytest.mark.parametrize(
    "layers, hooked, reason",
    [
        ([_attn(head_dim=64, q=16, kv=4)], None, "head_dim_64_unsupported"),
        ([_attn(head_dim=256, q=40, kv=1)], None, "gqa_40_1_unsupported"),
        ([_attn()], [False], "attention_not_hooked"),
        ([_attn(packed=False)], None, "gqa_packed_sdpa_off"),
        ([], None, "no_full_attention_layers"),
        ([SimpleNamespace(head_dim=None)], None, "attention_shape_unknown"),
    ],
)
def test_an_unsupported_model_switches_the_feature_off_with_a_reason(layers, hooked, reason, capsys) -> None:
    verdict = evaluate_model_support(layers, hooked=hooked)
    assert not verdict["supported"] and reason in verdict["reasons"]
    assert segmented_kv_enabled() is False
    assert segmented_ssd_enabled() is False
    out = capsys.readouterr().out
    assert out.count("MTPLX_SEGMENTED_KV is on but this model cannot use it") == 1 and reason in out


def test_the_log_line_appears_once(capsys) -> None:
    evaluate_model_support([_attn(head_dim=64)])
    evaluate_model_support([_attn(head_dim=64)])
    assert capsys.readouterr().out.count("keeping the stock KV cache") == 1


def test_missing_flash_route_or_nax_is_a_reason(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_NAX_FLASH_ROUTE")
    assert "nax_flash_route_off" in evaluate_model_support([_attn()])["reasons"]
    monkeypatch.setenv("MTPLX_NAX_FLASH_ROUTE", "1")
    monkeypatch.setattr("mtplx.nax_verify.nax_available", lambda: False)
    assert "nax_unavailable" in evaluate_model_support([_attn()])["reasons"]


def test_an_unsupported_model_keeps_stock_caches_and_never_builds_a_segmented_one() -> None:
    evaluate_model_support([_attn(head_dim=64)])
    cache = [KVCache(), KVCache()]
    configure_tail_owned_attention_kv_cache(cache)
    assert all(type(c) is KVCache for c in cache)
    assert not any(getattr(c, "_mtplx_segmentable", False) for c in cache)
    for c in cache:
        c.update_and_fetch(mx.zeros((1, 2, 9000, 8)), mx.zeros((1, 2, 9000, 8)))
    configure_tail_owned_attention_kv_cache(cache)  # what the repage after a prefill runs
    assert all(type(c) is KVCache for c in cache)


def test_health_reports_the_reason_and_is_absent_when_not_requested(monkeypatch) -> None:
    evaluate_model_support([_attn(head_dim=64)])
    block = segmented_kv_health([])
    assert block["enabled"] is False and block["requested"] is True
    assert "head_dim_64_unsupported" in block["model_support"]["reasons"]
    monkeypatch.delenv("MTPLX_SEGMENTED_KV")
    assert segmented_kv_health([]) is None


def test_health_of_a_supported_model_carries_the_verdict() -> None:
    evaluate_model_support([_attn()])
    block = segmented_kv_health([])
    assert block["enabled"] is True and block["model_support"]["supported"] is True


def test_nothing_is_decided_before_a_model_is_checked() -> None:
    assert model_support() is None and segmented_kv_enabled() is True
    assert module._MODEL_SUPPORT is None


def test_configure_split_full_attention_decides_from_the_hooked_layers(monkeypatch) -> None:
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    from mtplx.attention_split import configure_split_full_attention

    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")

    def model(head_dim):
        args = SimpleNamespace(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim, attention_bias=False,
            rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=10000.0, rope_scaling=None,
            max_position_embeddings=1024,
        )
        attn = Qwen3NextAttention(args)
        layer = SimpleNamespace(is_linear=False, self_attn=attn)
        return SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    stats = configure_split_full_attention(model(32))
    assert stats["segmented_kv_supported"] is False
    assert "head_dim_32_unsupported" in model_support()["reasons"]
    assert segmented_kv_enabled() is False
    reset_model_support()
    stats = configure_split_full_attention(model(256))
    assert stats["segmented_kv_supported"] is True
    assert segmented_kv_enabled() is True


def test_head_dim_is_read_from_the_scale_when_the_module_has_no_attribute() -> None:
    """mlx-lm's Qwen3 attention (head_dim 128) keeps n_heads, n_kv_heads and scale only."""
    attn = SimpleNamespace(n_heads=32, n_kv_heads=8, scale=128**-0.5, _mtplx_gqa_packed_sdpa_enabled=True)
    verdict = evaluate_model_support([attn])
    assert verdict["supported"], verdict
    assert verdict["head_dims"] == [128] and verdict["gqa"] == [4]


def _qwen3_attention(head_dim=32, q=4, kv=2, hidden=64):
    from mlx_lm.models.qwen3 import Attention, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=hidden, num_hidden_layers=1, intermediate_size=64, num_attention_heads=q,
        rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=kv, max_position_embeddings=4096, rope_theta=10000.0,
        head_dim=head_dim, tie_word_embeddings=False,
    )
    mx.random.seed(0)
    return Attention(args)


def test_plain_qwen3_attention_is_hooked_for_segmented_caches_only_and_matches_the_stock_forward() -> None:
    """A plain q/k-norm attention (no gate) goes through the hook body only on a segmented cache; with a
    stock cache it is the unchanged mlx-lm forward, so models without the switch keep their code."""
    from mlx_lm.models.cache import KVCache

    from mtplx.attention_split import _attention_has_gated_q_proj, _attention_has_plain_q_proj, _install_split_attention_hook
    from mtplx.segmented_kv import SegmentedKVCache

    attn = _qwen3_attention()
    assert not _attention_has_gated_q_proj(attn) and _attention_has_plain_q_proj(attn)

    def run(cache):
        outs = []
        for seed, n in ((1, 40), (2, 9), (3, 1)):
            x = mx.random.normal((1, n, 64), key=mx.random.key(seed))
            outs.append(attn(x, mask="causal" if n > 1 else None, cache=cache))
        mx.eval(outs)
        return outs

    before = run(KVCache())
    _install_split_attention_hook(attn)
    attn._mtplx_split_full_attention_enabled = True
    after_stock = run(KVCache())
    for a, b in zip(before, after_stock):
        assert mx.array_equal(a, b).item()  # stock cache: the unchanged forward
    cache = SegmentedKVCache()
    seg = run(cache)
    cache.seal()
    for a, b in zip(before, seg):
        assert float(mx.abs(a - b).max().item()) < 1e-4  # same math on the gathered rows (CPU: no kernel)
    assert cache.offset == 50


def test_the_hook_covers_plain_attention_in_the_support_check() -> None:
    from types import SimpleNamespace

    from mtplx.attention_split import configure_split_full_attention

    attn = _qwen3_attention(head_dim=128, q=8, kv=2, hidden=128)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False, self_attn=attn)]))
    import os

    os.environ["MTPLX_GQA_PACKED_SDPA"] = "1"
    try:
        stats = configure_split_full_attention(model)
    finally:
        os.environ.pop("MTPLX_GQA_PACKED_SDPA")
    assert stats["segmented_kv_supported"] is True
    assert model_support()["head_dims"] == [128] and model_support()["gqa"] == [4]
