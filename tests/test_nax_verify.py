"""Tests for the m4/NAX verify kernel module."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.nax_verify import (
    install_nax_qlinear_patch,
    m4_ksplit_eligible,
    m16_nax_eligible,
    nax_available,
    nax_qmm_m4,
    nax_qmm_m16,
    uninstall_nax_qlinear_patch,
)


def test_eligibility_shape_policy() -> None:
    dt = mx.bfloat16
    # m4: exact 4 rows only, no NAX hardware requirement
    assert m4_ksplit_eligible(4, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(5, 5120, 17408, 4, 64, dt)
    assert not m4_ksplit_eligible(4, 5120, 17408, 8, 64, dt)
    # m16: K % 256, N % 32, 4-bit, M in 1..16 (and NAX hardware)
    expect = nax_available()
    assert m16_nax_eligible(5, 5120, 17408, 4, 64, dt) == expect
    assert m16_nax_eligible(16, 17408, 5120, 4, 64, dt) == expect
    assert not m16_nax_eligible(17, 5120, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120 + 64, 17408, 4, 64, dt)
    assert not m16_nax_eligible(5, 5120, 17408 + 8, 4, 64, dt)


def _quantized_fixture(K: int, N: int):
    mx.random.seed(3)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _stock(x, w_q, scales, biases):
    return mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
    )


def test_m4_kernel_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    x = (mx.random.normal((4, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    y = nax_qmm_m4(x, w_q, scales, biases, group_size=64)
    ref = _stock(x, w_q, scales, biases)
    diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    assert y.shape == (4, N)
    assert diff < 0.25, f"m4 kernel drift too large: {diff}"


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_m16_nax_kernel_pads_and_matches_stock_within_tolerance() -> None:
    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 16):
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m16(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"nax16 kernel drift too large at M={m}: {diff}"


def test_qlinear_patch_routes_only_verify_shapes() -> None:
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        for m in (1, 3, 4, 8, 17, 64):
            x = (mx.random.normal((m, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
            y = layer(x)
            mx.eval(y)
            assert y.shape == (m, 256)
    finally:
        uninstall_nax_qlinear_patch()


def test_turbo_profile_carries_nax_env() -> None:
    from mtplx.profiles import PROFILES, PROFILE_CHOICES, apply_profile_env, restore_profile_env
    import os

    assert "turbo" in PROFILE_CHOICES
    profile = PROFILES["turbo"]
    assert profile.env_dict().get("MTPLX_NAX_VERIFY") == "1"
    assert profile.product_claim_eligible is False
    # Sustained env must be a subset (turbo = sustained + kernels).
    sustained = PROFILES["sustained"].env_dict()
    turbo = profile.env_dict()
    missing = {k: v for k, v in sustained.items() if turbo.get(k) != v}
    assert not missing, f"turbo drops sustained env keys: {missing}"
    previous = apply_profile_env("turbo")
    try:
        assert os.environ.get("MTPLX_NAX_VERIFY") == "1"
    finally:
        restore_profile_env(previous)
        assert os.environ.get("MTPLX_NAX_VERIFY") != "1"


def test_qlinear_patch_never_routes_in_prefill_phase() -> None:
    """Regression guard: prefill must stay on stock kernels byte-for-byte."""
    import mlx.core as mx
    from mtplx.attention_context import attention_phase
    from mtplx import nax_verify

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0, "m16": 0}
    orig_m4, orig_m16 = nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16

    def count_m4(*a, **k):
        calls["m4"] += 1
        return orig_m4(*a, **k)

    def count_m16(*a, **k):
        calls["m16"] += 1
        return orig_m16(*a, **k)

    nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = count_m4, count_m16
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("prefill"):
            mx.eval(layer(x))
        assert calls == {"m4": 0, "m16": 0}, f"kernels routed during prefill: {calls}"
        with attention_phase("decode_verify"):
            mx.eval(layer(x))
        assert calls["m4"] == 1, f"m4 kernel did not engage outside prefill: {calls}"
    finally:
        nax_verify.nax_qmm_m4, nax_verify.nax_qmm_m16 = orig_m4, orig_m16
        uninstall_nax_qlinear_patch()


def test_m6_kernel_matches_stock_within_tolerance() -> None:
    from mtplx.nax_verify import m6_ksplit_eligible, nax_qmm_m6

    K, N = 5120, 6144
    w_q, scales, biases = _quantized_fixture(K, N)
    for m in (5, 6):
        assert m6_ksplit_eligible(m, K, N, 4, 64, mx.bfloat16)
        x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        y = nax_qmm_m6(x, w_q, scales, biases, group_size=64)
        ref = _stock(x, w_q, scales, biases)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert y.shape == (m, N)
        assert diff < 0.25, f"m6 kernel drift too large at M={m}: {diff}"
    assert not m6_ksplit_eligible(4, K, N, 4, 64, mx.bfloat16)
    assert not m6_ksplit_eligible(7, K, N, 4, 64, mx.bfloat16)


def test_vk_6bit_hexpack_ksplit_matches_stock() -> None:
    """The 9B-tier 6-bit lane (2026-07-07): MLX packs 6-bit values
    bit-contiguously little-endian; the hexpack kernels must agree with
    stock quantized_matmul within the accumulation-order ULP band."""
    from mtplx.verify_kernels import (
        vk_eligible_ksplit,
        vk_qmm_m4_ksplit,
        vk_qmm_m6_ksplit,
    )

    K, N = 4096, 1024
    for dtype in (mx.bfloat16, mx.float16):
        for gs in (32, 64, 128):
            mx.random.seed(5)
            w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
            w_q, scales, biases = mx.quantize(w, group_size=gs, bits=6)
            mx.eval(w_q, scales, biases)
            for m, fn in ((4, vk_qmm_m4_ksplit), (5, vk_qmm_m6_ksplit), (6, vk_qmm_m6_ksplit)):
                assert vk_eligible_ksplit(m, K, N, 6, gs, dtype)
                x = (mx.random.normal((m, K), dtype=mx.float32) * 0.5).astype(dtype)
                y = fn(x, w_q, scales, biases, bits=6, group_size=gs)
                ref = mx.quantized_matmul(
                    x, w_q, scales=scales, biases=biases,
                    transpose=True, group_size=gs, bits=6,
                )
                diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
                assert y.shape == (m, N)
                assert diff < 0.05, f"6-bit drift {dtype} gs={gs} M={m}: {diff}"


def test_qlinear_patch_routes_6bit_verify_shapes() -> None:
    """The patch routes 6-bit verify shapes (N >= 2048 floor) through the
    hexpack kernels and leaves small-N projections on stock."""
    from mtplx import verify_kernels

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    calls = {"m4": 0}
    orig = verify_kernels.vk_qmm_m4_ksplit

    def counting(*a, **k):
        calls["m4"] += 1
        return orig(*a, **k)

    from mtplx.attention_context import attention_phase

    import mtplx.nax_verify  # noqa: F401  (patch reads through the module)

    verify_kernels.vk_qmm_m4_ksplit = counting
    try:
        big = nn.QuantizedLinear(512, 2048, bias=False, group_size=64, bits=6)
        small = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=6)
        x = (mx.random.normal((4, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "6-bit verify shape did not route the hexpack kernel"
            mx.eval(small(x))
            assert calls["m4"] == 1, "small-N 6-bit projection must stay stock"
        with attention_phase("prefill"):
            mx.eval(big(x))
            assert calls["m4"] == 1, "prefill must stay stock"
    finally:
        verify_kernels.vk_qmm_m4_ksplit = orig
        uninstall_nax_qlinear_patch()


def test_m5_falls_through_to_stock_by_default(monkeypatch) -> None:
    """2026-08-22 routing fix: M=5 4-bit skips the padded m6/m16 lanes
    (three-session micro: padded-m6 +3..13% vs stock, m16 worst) unless
    MTPLX_M5_PADDED_LANE opts back in. Receipt = the b4_m5 fallback counter
    (entered the verify window, took no custom lane)."""
    from mtplx.nax_verify import _m5_padded_lane, nax_qlinear_fallback_counts

    monkeypatch.delenv("MTPLX_M5_PADDED_LANE", raising=False)
    assert _m5_padded_lane() is False
    monkeypatch.setenv("MTPLX_M5_PADDED_LANE", "1")
    assert _m5_padded_lane() is True
    monkeypatch.delenv("MTPLX_M5_PADDED_LANE", raising=False)

    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        # K=512 %256==0, N=256 %32==0: m6-ksplit and m16 would both accept
        # this shape at M=5 — only the new guard keeps it on stock.
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x = (mx.random.normal((5, 512), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        before = nax_qlinear_fallback_counts.get("b4_m5", 0)
        y = layer(x)
        mx.eval(y)
        assert y.shape == (5, 256)
        assert nax_qlinear_fallback_counts.get("b4_m5", 0) == before + 1
        ref = _stock(x, layer["weight"], layer["scales"], layer["biases"])
        assert mx.array_equal(y, ref), "M=5 default must be the stock result"
    finally:
        uninstall_nax_qlinear_patch()


def test_q8_nax_pad_rows_follow_the_mlx_dispatch() -> None:
    """The padded 8-bit lane pads only where stock MLX misses its NAX tile
    (Qwen3.8-27B 8-bit shapes, group size 64)."""
    from mtplx.nax_verify import mlx_qmm_takes_nax, q8_nax_pad_rows

    gdn_out, mlp_down, gate_up, lm_head = (6144, 5120), (17408, 5120), (5120, 17408), (5120, 248320)
    # Narrow N: split-K tile from 13 rows; 33 rows (two 32-row tile rows) reach NAX.
    for k, n in (gdn_out, mlp_down):
        for m in (13, 17, 25, 32):
            assert not mlx_qmm_takes_nax(m, k, n, 64)
            assert q8_nax_pad_rows(m, k, n, 64) == 33
        assert mlx_qmm_takes_nax(33, k, n, 64)
        # Two tiles cost more than qmv_wide below 9 rows: stock there.
        assert q8_nax_pad_rows(9, k, n, 64) == 33
        assert q8_nax_pad_rows(8, k, n, 64) is None
    # Wide N: stock already takes NAX from 13 rows, qmv_wide below.
    for k, n in (gate_up, lm_head):
        for m in (13, 17, 25, 32):
            assert mlx_qmm_takes_nax(m, k, n, 64)
            assert q8_nax_pad_rows(m, k, n, 64) is None
        for m in (7, 9, 12):
            assert q8_nax_pad_rows(m, k, n, 64) == 13
    # Outside 7..32 rows and K not a multiple of 64: never padded.
    for m in (4, 6, 33, 64):
        assert q8_nax_pad_rows(m, *gdn_out, 64) is None
    assert q8_nax_pad_rows(13, 6144 + 32, 5120, 32) is None


def _q8_layer(K: int, N: int):
    mx.random.seed(5)
    layer = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=8)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16), group_size=64, bits=8
    )
    mx.eval(layer.parameters())
    return layer


def _q8_stock(layer, x):
    return mx.quantized_matmul(
        x, layer["weight"], scales=layer["scales"], biases=layer["biases"],
        transpose=True, group_size=64, bits=8,
    )


def test_q8_nax_pad_is_off_by_default(monkeypatch) -> None:
    from mtplx import nax_verify
    from mtplx.attention_context import attention_phase

    monkeypatch.delenv("MTPLX_Q8_NAX_PAD", raising=False)
    calls = {"n": 0}
    original = nax_verify.nax_qmm_q8_padded

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(nax_verify, "nax_qmm_q8_padded", counting)
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = _q8_layer(1024, 8192)
        x = (mx.random.normal((13, 1024), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        with attention_phase("decode_verify"):
            y = layer(x)
        assert calls["n"] == 0
        assert mx.array_equal(y, _q8_stock(layer, x)), "switch off must give the stock result"
    finally:
        uninstall_nax_qlinear_patch()


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_q8_nax_pad_routes_wide_verify_rows_only(monkeypatch) -> None:
    """With the switch on, 13 verify rows of a split-K shape take the padded
    lane and stay within the qmm tolerance of stock; prefill, rows below the
    floor and shapes that already reach NAX stay stock bit for bit."""
    from mtplx import nax_verify
    from mtplx.attention_context import attention_phase

    monkeypatch.setenv("MTPLX_Q8_NAX_PAD", "1")
    calls = {"n": 0}
    original = nax_verify.nax_qmm_q8_padded

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(nax_verify, "nax_qmm_q8_padded", counting)
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    try:
        layer = _q8_layer(1024, 8192)
        x = (mx.random.normal((1, 13, 1024), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
        ref = _q8_stock(layer, x)
        with attention_phase("decode_verify"):
            y = layer(x)
        assert calls["n"] == 1
        assert y.shape == (1, 13, 8192)
        diff = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
        assert diff < 0.1, f"padded 8-bit lane drift too large: {diff}"
        with attention_phase("prefill"):
            assert mx.array_equal(layer(x), ref)
        assert calls["n"] == 1, "prefill took the padded lane"
        wide = _q8_layer(1024, 16384)  # 13 rows already reach NAX here
        with attention_phase("decode_verify"):
            assert mx.array_equal(wide(x), _q8_stock(wide, x))
        assert calls["n"] == 1
        monkeypatch.setenv("MTPLX_Q8_NAX_PAD_MIN_ROWS", "14")
        with attention_phase("decode_verify"):
            assert mx.array_equal(layer(x), ref)
        assert calls["n"] == 1, "rows below MTPLX_Q8_NAX_PAD_MIN_ROWS took the padded lane"
    finally:
        uninstall_nax_qlinear_patch()
