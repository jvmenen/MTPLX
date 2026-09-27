"""The fixed-M4 verify bank's capacity, and what the rows-gather lane's outputs depend on.

A compiled function keeps one trace per input-shape signature for as long as
it lives (MLX's compile cache), and the fixed-M4 bank's capacity is part of
that signature. The bank is sized prompt + reserve rounded to 256 tokens, so
every agent turn at 16K or more arrives with a capacity the process has not
seen and pays a fresh trace. Rounding the capacity up to a coarser bucket
would let a growing session keep one trace, but only if the verify outputs do
not depend on the capacity.

On the rows-gather lane (16,384 tokens of KV and more) they must not: each
row attends over its own gathered rows and the index scores are per block, so
the padded tail never enters a value. This file proves it on the compiled
verifier itself, with the tiny random pack of
``scripts/qwen4exp_mtp_tiny_smoke.py`` in bfloat16 on the GPU: the same
sampled request at two capacities, logits, hidden states, captures and every
state leaf bit for bit, round by round.

The dense lane is not capacity-invariant in general. It reduces over the
whole bank through MLX's vector SDPA, and MLX 0.32.2 picks that kernel and its
number of key blocks from the key length, which is the capacity
(``mlx/backend/metal/scaled_dot_product_attention.cpp`` 875-877 and 487-517).
One Flash-Next-geometry attention layer, capacity 3,524 against 8,192, differs
in 53 to 55% of its bfloat16 outputs per verify step under the base/Pro
dispatch class; capacity 9,024 against 16,384 differs in 58 to 59% under the
Ultra class (and in neither case under the Max class this suite runs on).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions
from mtplx.sampling import SamplerConfig

_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
MAX_TOKENS = 40
SEED = 1234
PROMPT = [3, 5, 7, 9, 11, 13] + list(range(20, 54))  # 40 tokens: 10 pooled blocks


def _smoke():
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", _SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack():
    """The tiny pack in bf16 on the GPU, shaped for the fixed-M4 lane (one PLE layer)."""

    if not mx.metal.is_available():
        pytest.skip("bfloat16 expert gathers need the GPU; the CPU has no exact full-model lane")
    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp
    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    smoke = _smoke()
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    # The verify bank looks the ArraysCache class up when it is called; build
    # the model with the same class a runtime load earlier in the session
    # may have swapped in.
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.random.seed(0)
    args = dataclasses.replace(
        smoke._tiny_text_args(),
        head_dim=32,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[1],
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    yield smoke, model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


@pytest.fixture()
def lane(monkeypatch):
    # One request's route must not depend on what the shell exported.
    import os

    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name.startswith("MTPLX_QSA_") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    demotions.reset()
    yield monkeypatch
    demotions.reset()


def _bits(value) -> np.ndarray:
    """Exact bit patterns, so equality is equality and NaN compares too."""

    if value.dtype in (mx.bfloat16, mx.float16):
        return np.array(value.view(mx.uint16))
    if value.dtype == mx.float32:
        return np.array(value.view(mx.uint32))
    return np.array(value)


def _recorded_rounds(monkeypatch) -> list[dict]:
    """Everything one installed fixed-M4 replay produced, round by round."""

    rounds: list[dict] = []
    real = graphbank.CompiledVerifyBank._forward_installed_fixed_m4

    def recording(self, input_ids, host_input_ids, completion_tokens, committed_count, cache):
        logits, hidden, extra = real(
            self, input_ids, host_input_ids, completion_tokens, committed_count, cache
        )
        dispatch = self._fixed_m4_dispatch
        record: dict[str, object] = {
            "capacity": int(dispatch["capacity"]),
            "logits": _bits(logits),
            "hidden": _bits(hidden),
        }
        for n, (kind, entry, n_leaves) in enumerate(dispatch["state_plan"]):
            if kind == graphbank.VERIFY_SPEC_KIND_QSA:
                end = int(entry.size())
                record[f"qsa{n}.rows_gather"] = bool(entry.fixed_rows_gather)
                record[f"qsa{n}.end"] = end
                record[f"qsa{n}.keys"] = _bits(entry.kv.cache[0][..., :end, :])
                record[f"qsa{n}.values"] = _bits(entry.kv.cache[1][..., :end, :])
                record[f"qsa{n}.raw"] = _bits(entry.raw_keys[:, :end])
                record[f"qsa{n}.pooled"] = _bits(entry.pooled[:, : end // entry.ratio])
            else:
                for slot in range(n_leaves):
                    record[f"state{n}.{slot}"] = _bits(entry.cache[slot])
        for n, (entry, _start, _count) in enumerate(dispatch["capture_plan"]):
            for slot, leaf in enumerate(getattr(entry, "_mtplx_verify_rows", ()) or ()):
                record[f"capture{n}.rows{slot}"] = _bits(leaf)
            for slot, leaf in enumerate(getattr(entry, "_mtplx_verify_ple", ()) or ()):
                record[f"capture{n}.ple{slot}"] = _bits(leaf)
        rounds.append(record)
        return logits, hidden, extra

    monkeypatch.setattr(graphbank.CompiledVerifyBank, "_forward_installed_fixed_m4", recording)
    return rounds


def _run(pack, monkeypatch, *, extra_rows: int):
    """One sampled request on the compiled fixed-M4 lane, the bank ``extra_rows`` larger."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    real_rule = graphbank.TensorOffsetQSACache._bank_capacity

    def rule(needed, ratio, kv_step, **kwargs):
        return real_rule(needed, ratio, kv_step, **kwargs) + extra_rows

    with monkeypatch.context() as patch:
        patch.setattr(graphbank.TensorOffsetQSACache, "_bank_capacity", staticmethod(rule))
        rounds = _recorded_rounds(patch)
        patch.setenv("MTPLX_COMPILED_VERIFY", "1")
        rt = smoke._tiny_runtime(model)
        install_qwen4_fixed_verify_route(rt)
        result = generation.generate_mtpk(
            rt,
            list(PROMPT),
            max_tokens=MAX_TOKENS,
            sampler=NATIVE,
            draft_sampler=NATIVE,
            speculative_depth=3,
            seed=SEED,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )
    return result, rounds


def _assert_same_rounds(narrow: list[dict], wide: list[dict], *, bucket: int) -> None:
    assert len(narrow) == len(wide) >= 8
    for index, (a, b) in enumerate(zip(narrow, wide)):
        assert b["capacity"] == a["capacity"] + bucket, index
        assert a.keys() == b.keys(), index
        for name in a:
            if name == "capacity":
                continue
            left, right = a[name], b[name]
            if isinstance(left, np.ndarray):
                assert left.shape == right.shape, (index, name)
                assert np.array_equal(left, right), (index, name)
            else:
                assert left == right, (index, name)


@pytest.mark.parametrize("extra_rows", [256, 16_384])
def test_rows_gather_verify_does_not_depend_on_the_capacity(pack, lane, extra_rows):
    """The invariant a capacity bucket rests on, on the compiled fixed-M4 verifier.

    256 rows is one K/V step; 16,384 puts the wide bank past the rows-gather
    switch length while the narrow one stays below it.
    """

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")  # the tiny prompt is past it
    narrow, narrow_rounds = _run(pack, lane, extra_rows=0)
    wide, wide_rounds = _run(pack, lane, extra_rows=extra_rows)

    if extra_rows == 16_384:
        assert narrow_rounds[0]["capacity"] < 16_384 < wide_rounds[0]["capacity"]
    assert all(r["qsa1.rows_gather"] for r in narrow_rounds + wide_rounds)
    _assert_same_rounds(narrow_rounds, wide_rounds, bucket=extra_rows)
    assert list(wide.tokens) == list(narrow.tokens)
    report = (wide.stats.graphbank or {}).get("compiled_verify") or {}
    assert report["compiled_calls"] == len(wide_rounds) and report["fallback_calls"] == 0

