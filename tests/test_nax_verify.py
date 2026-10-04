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


def _wide_rows_layer(K: int, N: int, bits: int, group_size: int):
    mx.random.seed(11)
    layer = nn.QuantizedLinear(K, N, bias=False, group_size=group_size, bits=bits)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(mx.bfloat16)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        w, group_size=group_size, bits=bits
    )
    mx.eval(layer.parameters())
    return layer


def test_pad_wide_rows_switch_and_eligibility(monkeypatch) -> None:
    """MTPLX_QMM_PAD_ROWS is read per call and off by default; the lane covers
    9..12 rows of 4/8-bit affine layers only (and needs NAX hardware, since
    qmm_nax is what the padding buys)."""
    from mtplx.nax_verify import _pad_wide_rows_eligible, _pad_wide_rows_enabled

    monkeypatch.delenv("MTPLX_QMM_PAD_ROWS", raising=False)
    assert _pad_wide_rows_enabled() is False
    monkeypatch.setenv("MTPLX_QMM_PAD_ROWS", "1")
    assert _pad_wide_rows_enabled() is True

    layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=8)
    dt = mx.bfloat16
    expect = nax_available()
    for m in (9, 10, 11, 12):
        assert _pad_wide_rows_eligible(m, 5120, 248320, 8, 64, dt, layer) == expect
        assert _pad_wide_rows_eligible(m, 5120, 17408, 8, 64, dt, layer) == expect
        assert _pad_wide_rows_eligible(m, 5120, 17408, 4, 32, dt, layer) == expect
    for m in (1, 4, 5, 6, 7, 8, 13, 16, 24):
        assert not _pad_wide_rows_eligible(m, 5120, 248320, 8, 64, dt, layer)
    assert not _pad_wide_rows_eligible(8, 5120, 248320, 6, 64, dt, layer)
    assert not _pad_wide_rows_eligible(8, 5120, 248320, 8, 64, mx.float32, layer)
    # Shapes where padding measured slower stay stock: q8 down (K=17408),
    # q8 out_proj (N=5120) and the N=48 gate projections.
    assert not _pad_wide_rows_eligible(8, 17408, 5120, 8, 64, dt, layer)
    assert not _pad_wide_rows_eligible(8, 6144, 5120, 8, 64, dt, layer)
    assert not _pad_wide_rows_eligible(8, 5120, 48, 4, 32, dt, layer)


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
@pytest.mark.parametrize(
    "bits,group_size,K,N",
    [(8, 64, 5120, 16384), (8, 32, 5120, 16384), (4, 32, 544, 16384), (4, 64, 576, 16384)],
)
def test_pad_wide_rows_matches_stock_rowwise(monkeypatch, bits, group_size, K, N) -> None:
    """Padded 9..12-row calls: each row equals the stock row to bf16 rounding
    (qmm_nax and qmv_wide accumulate in different orders), is bit-identical to
    the same row of the 13-row stock call, and does not depend on how many
    rows were padded. Rows outside 9..12 and the switch off are stock bits."""
    layer = _wide_rows_layer(K, N, bits, group_size)
    x_all = (mx.random.normal((16, K), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    ref13 = layer(x_all[:13])
    mx.eval(x_all, ref13)

    monkeypatch.setenv("MTPLX_QMM_PAD_ROWS", "1")
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    from mtplx.attention_context import attention_phase

    try:
        with attention_phase("decode_verify"):
            first_rows = []
            for m in range(9, 13):
                y = layer(x_all[:m])
                mx.eval(y)
                assert y.shape == (m, N)
                assert mx.array_equal(y, ref13[:m]), f"M={m} differs from 13-row rows"
                first_rows.append(y[:9])
            for rows in first_rows[1:]:
                assert mx.array_equal(rows, first_rows[0])
            # 3-D input (batch 1, M rows) keeps its shape.
            y3 = layer(x_all[None, :12, :])
            assert y3.shape == (1, 12, N)
            assert mx.array_equal(y3[0], ref13[:12])
            # Outside 9..12 the call is the untouched stock result (M=4..6
            # belong to the 8-bit verify_kernels lanes, so they are not here).
            for m in (1, 2, 3, 13, 16):
                assert mx.array_equal(layer(x_all[:m]), _stock_layer(layer, x_all[:m]))
        # Prefill phase never pads.
        with attention_phase("prefill"):
            assert mx.array_equal(layer(x_all[:12]), _stock_layer(layer, x_all[:12]))
        # Switch off: stock bits.
        monkeypatch.setenv("MTPLX_QMM_PAD_ROWS", "0")
        with attention_phase("decode_verify"):
            assert mx.array_equal(layer(x_all[:12]), _stock_layer(layer, x_all[:12]))
    finally:
        uninstall_nax_qlinear_patch()
    # Against the stock 9..12-row kernel only the accumulation order differs.
    for m in (9, 12):
        diff = float(
            mx.abs(ref13[:m].astype(mx.float32) - _stock_layer(layer, x_all[:m]).astype(mx.float32)).max()
        )
        assert diff < 0.25, f"padded rows drift too large at M={m}: {diff}"


def _stock_layer(layer, x):
    return mx.quantized_matmul(
        x, layer["weight"], scales=layer["scales"], biases=layer["biases"],
        transpose=True, group_size=layer.group_size, bits=layer.bits,
    )


@pytest.mark.skipif(not nax_available(), reason="requires Apple G17 + macOS >= 26.2")
def test_pad_wide_rows_leaves_m16_nax_lane_alone(monkeypatch) -> None:
    """A 4-bit shape the m16 NAX lane serves (K % 256 == 0) keeps that lane:
    the padding only picks up calls that fell through to stock."""
    from mtplx.attention_context import attention_phase

    layer = _wide_rows_layer(5120, 17408, 4, 32)
    x = (mx.random.normal((12, 5120), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    results = []
    for value in ("0", "1"):
        monkeypatch.setenv("MTPLX_QMM_PAD_ROWS", value)
        install_nax_qlinear_patch()
        try:
            with attention_phase("decode_verify"):
                y = layer(x)
                mx.eval(y)
            results.append(y)
        finally:
            uninstall_nax_qlinear_patch()
    assert mx.array_equal(results[0], results[1])
