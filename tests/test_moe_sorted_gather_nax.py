"""Flash-Next's routed gate/up gather reading token rows in place
(mtplx/kernels/moe_sorted_gather_nax.py).

Class A: every output row must be bit-identical to ``mx.gather_qmm(tokens[row_map],
..., sorted_indices=True)`` (the copy plus stock gather the model ran before),
and with the SwiGLU epilogue to that gather's split and ``nn.silu(gate) * up``,
wherever the stock kernel is correct, and right where it is not (past 32,767
unaligned rows on MLX 0.32.2).  The kernel runs only on tensor-unit GPUs;
everywhere else these tests check that the stock path runs.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx import moe_sorted_gather as msg
from mtplx import nax_detect
from mtplx.kernels import moe_sorted_gather_nax as nax_gather

needs_tensor_units = pytest.mark.skipif(
    not nax_detect.nax_available() or nax_gather._mlx_headers() is None,
    reason="the kernel runs only on tensor-unit GPUs with MLX's kernel headers installed",
)


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _weights(experts, n, k, group_size, bits, dtype, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((experts, n, k)) * 0.05).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b


def _routed(tokens, experts, top_k, k, dtype, seed):
    """Tokens and their expert-sorted routing, the way the model sorts them."""

    mx.random.seed(seed)
    x = (mx.random.normal((tokens, k)) * 0.5).astype(dtype)
    inds = mx.argsort(mx.random.uniform(shape=(tokens, experts)), axis=-1)[:, :top_k]
    tok, row_map, idx, _inv = msg.sort_rows(x, inds.astype(mx.uint32))
    mx.eval(tok, row_map, idx)
    return tok, row_map, idx


def _stock(tokens, row_map, wq, s, b, idx, group_size, bits):
    return mx.gather_qmm(
        tokens[row_map], wq, s, b, rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )


def _stock_swiglu(tokens, row_map, wq, s, b, idx, group_size, bits):
    """The chain Flash-Next's expert module ran before the epilogue."""

    gate, up = mx.split(_stock(tokens, row_map, wq, s, b, idx, group_size, bits), 2, axis=-1)
    return nn.silu(gate) * up


@pytest.fixture(autouse=True)
def _fresh_canaries(monkeypatch):
    monkeypatch.setattr(nax_gather, "_CANARY", {})
    monkeypatch.setattr(
        nax_gather, "_STATS", {"calls": 0, "fallbacks": 0, "canaries": 0, "canary_failures": 0}
    )


@needs_tensor_units
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(32, 4), (64, 4), (64, 8)])
@pytest.mark.parametrize("n,k", [(128, 256), (256, 128)])
def test_bit_identical_to_copy_plus_stock_gather(dtype, group_size, bits, n, k):
    tok, row_map, idx = _routed(500, 64, 10, k, dtype, seed=3)
    wq, s, b = _weights(64, n, k, group_size, bits, dtype)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=group_size, bits=bits)
    assert ours is not None, nax_gather.stats()
    stock = _stock(tok, row_map, wq, s, b, idx, group_size, bits)
    assert tuple(ours.shape) == tuple(stock.shape) == (5000, 1, n)
    assert np.array_equal(_bits(ours), _bits(stock))
    assert nax_gather.stats()["canary_failures"] == 0


@needs_tensor_units
def test_ragged_runs_empty_experts_and_tile_edges():
    """Runs of 0, 1, 63, 64, 65, 127, 128 and 129 rows: tiles end inside and at
    the edge of an expert's run, and empty experts own no tile."""

    counts = [0, 1, 63, 64, 65, 0, 127, 128, 129, 0, 2000, 1, 0, 3000]
    counts += [0] * (64 - len(counts))
    idx = mx.array(np.repeat(np.arange(64, dtype=np.uint32), counts))
    rows = int(idx.shape[0])
    mx.random.seed(6)
    tok = (mx.random.normal((900, 1, 256)) * 0.5).astype(mx.bfloat16)
    row_map = mx.random.randint(0, 900, (rows,)).astype(mx.uint32)
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=5)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert ours is not None
    assert np.array_equal(_bits(ours), _bits(_stock(tok, row_map, wq, s, b, idx, 32, 4)))


@needs_tensor_units
@pytest.mark.parametrize("tokens", [3404, 4095])
def test_no_row_bound(tokens):
    """Past 32,767 unaligned rows: equal, bit for bit, to the padded stock call
    (correct, and equal to the stock call wherever that is correct)."""

    tok, row_map, idx = _routed(tokens, 64, 10, 256, mx.bfloat16, seed=9)
    rows = int(idx.shape[0])
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=7)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert ours is not None
    pad = 64 - rows % 64
    xp = mx.concatenate([tok[row_map], mx.zeros((pad, 1, 256), dtype=tok.dtype)])
    ip = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (pad,))])
    padded = mx.gather_qmm(
        xp, wq, s, b, rhs_indices=ip, transpose=True, group_size=32, bits=4, sorted_indices=True
    )[:rows]
    assert np.array_equal(_bits(ours), _bits(padded))


@needs_tensor_units
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(32, 4), (64, 4), (64, 8)])
@pytest.mark.parametrize("n,k", [(128, 256), (256, 128)])
def test_swiglu_bit_identical_to_the_stock_chain(dtype, group_size, bits, n, k):
    tok, row_map, idx = _routed(500, 64, 10, k, dtype, seed=4)
    wq, s, b = _weights(64, n, k, group_size, bits, dtype, seed=2)
    ours = nax_gather.gather_rows_qmm(
        tok, row_map, wq, s, b, idx, group_size=group_size, bits=bits, swiglu=True
    )
    assert ours is not None, nax_gather.stats()
    stock = _stock_swiglu(tok, row_map, wq, s, b, idx, group_size, bits)
    assert tuple(ours.shape) == tuple(stock.shape) == (5000, 1, n // 2)
    assert np.array_equal(_bits(ours), _bits(stock))
    assert nax_gather.stats()["canary_failures"] == 0


@needs_tensor_units
def test_swiglu_ragged_runs_empty_experts_and_tile_edges():
    counts = [0, 1, 63, 64, 65, 0, 127, 128, 129, 0, 2000, 1, 0, 3000]
    counts += [0] * (64 - len(counts))
    idx = mx.array(np.repeat(np.arange(64, dtype=np.uint32), counts))
    rows = int(idx.shape[0])
    mx.random.seed(16)
    tok = (mx.random.normal((900, 1, 256)) * 0.5).astype(mx.bfloat16)
    row_map = mx.random.randint(0, 900, (rows,)).astype(mx.uint32)
    wq, s, b = _weights(64, 192, 256, 32, 4, mx.bfloat16, seed=15)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert ours is not None
    stock = _stock_swiglu(tok, row_map, wq, s, b, idx, 32, 4)
    assert np.array_equal(_bits(ours), _bits(stock))


@needs_tensor_units
@pytest.mark.parametrize("tokens", [3404, 4095])
def test_swiglu_no_row_bound(tokens):
    tok, row_map, idx = _routed(tokens, 64, 10, 256, mx.bfloat16, seed=19)
    rows = int(idx.shape[0])
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=17)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert ours is not None
    pad = 64 - rows % 64
    xp = mx.concatenate([tok[row_map], mx.zeros((pad, 1, 256), dtype=tok.dtype)])
    ip = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (pad,))])
    gu = mx.gather_qmm(
        xp, wq, s, b, rhs_indices=ip, transpose=True, group_size=32, bits=4, sorted_indices=True
    )[:rows]
    gate, up = mx.split(gu, 2, axis=-1)
    assert np.array_equal(_bits(ours), _bits(nn.silu(gate) * up))


def test_short_chunks_follow_the_installed_mlx():
    """MLX before 0.32.3 tiles rows across experts and ours wins from 4,096
    rows; MLX 0.32.3 tiles per expert and keeps short chunks."""

    assert nax_gather.min_rows("0.32.2") == 4096
    assert nax_gather.min_rows("0.32.3.dev20260920") == 4096
    assert nax_gather.min_rows("0.32.3") == 16384
    assert nax_gather.min_rows("0.32.4") == 16384
    assert nax_gather.min_rows() == nax_gather.min_rows(mx.__version__)


def test_small_widths_and_other_layouts_keep_the_stock_kernel():
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    rows = nax_gather.min_rows()
    tok = mx.zeros((rows, 1, 256), dtype=mx.bfloat16)
    idx = mx.zeros((rows,), dtype=mx.uint32)
    kw = dict(group_size=32, bits=4, mode="affine")
    small = mx.zeros((rows - 1,), dtype=mx.uint32)
    assert not nax_gather.applies(tok, small, wq, s, b, small, **kw)
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, group_size=32, bits=4, mode="mxfp4")
    assert not nax_gather.applies(tok.astype(mx.float32), idx, wq, s, b, idx, **kw)
    assert not nax_gather.applies(tok, idx.astype(mx.int32), wq, s, b, idx, **kw)
    assert not nax_gather.applies(tok, idx, wq, s, None, idx, **kw)
    assert not nax_gather.applies(tok, idx, wq, s.astype(mx.float16), b, idx, **kw)
    assert not nax_gather.applies(tok.reshape(rows, 256), idx, wq, s, b, idx, **kw)


def test_rehearsal_switch_and_kill_switch_keep_the_stock_kernel(monkeypatch):
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    rows = nax_gather.min_rows()
    tok = mx.zeros((rows, 1, 256), dtype=mx.bfloat16)
    idx = mx.zeros((rows,), dtype=mx.uint32)
    kw = dict(group_size=32, bits=4, mode="affine")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, **kw)
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
    monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", "0")
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, **kw)


@needs_tensor_units
def test_a_failed_canary_falls_back_to_the_stock_op(monkeypatch):
    real_launch = nax_gather._launch

    def wrong(*args, **kwargs):
        return real_launch(*args, **kwargs) + 1

    monkeypatch.setattr(nax_gather, "_launch", wrong)
    # Enough rows for the entry point to take the kernel under any MLX.
    tok, row_map, idx = _routed(2000, 64, 10, 256, mx.bfloat16, seed=13)
    assert int(idx.shape[0]) >= nax_gather.min_rows()
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    assert nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4) is None
    assert nax_gather.stats()["canary_failures"] == 1
    # The entry point then copies the rows and runs the stock op.
    y = msg.gather_qmm_rows(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert np.array_equal(_bits(y), _bits(_stock(tok, row_map, wq, s, b, idx, 32, 4)))
    assert nax_gather.stats()["fallbacks"] == 2


@needs_tensor_units
def test_a_failed_swiglu_canary_falls_back_to_the_stock_chain(monkeypatch):
    real_launch = nax_gather._launch

    def wrong(*args, **kwargs):
        return real_launch(*args, **kwargs) + 1

    monkeypatch.setattr(nax_gather, "_launch", wrong)
    tok, row_map, idx = _routed(2000, 64, 10, 256, mx.bfloat16, seed=23)
    assert int(idx.shape[0]) >= nax_gather.min_rows()
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    y = msg.swiglu_rows(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert np.array_equal(_bits(y), _bits(_stock_swiglu(tok, row_map, wq, s, b, idx, 32, 4)))
    # The fused instantiation failed, then the plain one, then the stock chain ran.
    assert nax_gather.stats()["canary_failures"] == 2
    assert nax_gather.stats()["calls"] == 0


def test_flash_next_expert_module_equals_the_code_it_replaced(monkeypatch):
    """Flash-Next's own gate/up + down module at a 4,096-token chunk against
    the code it ran before (mlx-lm's gather-sort copy, the stock gate/up gather,
    split, ``nn.silu(gate) * up``, the down gather, the unsort), bit for bit, in
    both the block-forward and the prefill-combine entries.  On a tensor-unit
    GPU the module runs the fused kernel; elsewhere (and with the kill switch)
    the stock chain."""

    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    from mtplx.models import qwen4_exp

    experts, hidden, inter, top_k, tokens = 64, 256, 128, 10, 4096
    mx.random.seed(14)
    gu = (mx.random.normal((experts, 2 * inter, hidden)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gu, group_size=32, bits=4)
    down = QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=32, bits=4)
    dn = (mx.random.normal((experts, hidden, inter)) * 0.05).astype(mx.bfloat16)
    down.weight, down.scales, down.biases = mx.quantize(dn, group_size=32, bits=4)
    switch = qwen4_exp._FusedGateUpSwitchGLU(down, gu_w, gu_s, gu_b, 32, 4, "affine")
    x = (mx.random.normal((1, tokens, hidden)) * 0.5).astype(mx.bfloat16)
    inds = mx.argsort(mx.random.uniform(shape=(tokens, experts)), axis=-1)[:, :top_k]
    inds = inds.astype(mx.uint32).reshape(1, tokens, top_k)

    xs, idx, inv_ref = _gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    gate, up = mx.split(
        mx.gather_qmm(
            xs, gu_w, gu_s, gu_b, rhs_indices=idx, transpose=True,
            group_size=32, bits=4, sorted_indices=True,
        ),
        2,
        axis=-1,
    )
    y_ref = down(nn.silu(gate) * up, idx, sorted_indices=True)
    block_ref = _scatter_unsort(y_ref, inv_ref, inds.shape).squeeze(-2)

    rows = mx.zeros((tokens * top_k,), dtype=mx.uint32)
    on_tensor_units = nax_gather.applies(
        x.reshape(-1, 1, hidden), rows, gu_w, gu_s, gu_b, rows,
        group_size=32, bits=4, mode="affine",
    )
    for label, env in (("default", None), ("kill switch", "0")):
        if env is not None:
            monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", env)
        calls = nax_gather.stats()["calls"]
        y, inv = switch.sorted_experts(x, inds)
        block = switch(x, inds)
        mx.eval(y, inv, block)
        kernel_calls = nax_gather.stats()["calls"] - calls
        assert kernel_calls == (2 if on_tensor_units and env is None else 0), label
        assert np.array_equal(np.array(inv), np.array(inv_ref)), label
        assert np.array_equal(_bits(y), _bits(y_ref.reshape(y_ref.shape[0], -1))), label
        assert np.array_equal(_bits(block), _bits(block_ref)), label
