"""The compiled verifier and the eager verifier read the same evaluated weights.

``mx.compile`` treats an evaluated array it meets while tracing as a
constant, but it walks into an array that is still a lazy graph and replays
that graph on every call of the compiled function, inside fused JIT kernels.
A fused kernel does not reproduce the precompiled kernels bit for bit: MLX
0.32.2 prints its scalar constants with 7 significant digits and resolves the
unqualified ``metal::exp``/``metal::log``/``metal::pow`` in its sigmoid,
erfinv, power and logaddexp ops to the fast variants. A weight left lazy
would therefore reach the compiled verifier with different low bits than the
eager verifier, and be recomputed every round. A checkpoint load evaluates
the parameters (mlx_lm loads with ``lazy=False``) and the eager prefill
evaluates every array the forward reads, so the product's first verify trace
must find nothing lazy. The compiled-verify toy runtime once missed exactly
this; float32 GEMM on a Metal 4 tensor-unit GPU reads TF32 inputs and hid it
on M5, while M1 to M4 read all 23 mantissa bits.

The synthetic four-layer ``qwen3_5`` model of ``tests/dense_mrope_synth.py``
(the real mlx_lm GatedDeltaNet, gated attention and MLP code, MTPLX's split
attention route as the turbo profile installs it, the draft head, random
evaluated weights, no pack) runs ``generate_mtpk`` with the compiled verifier
on:

* every array reachable from the model when the verify step is built is an
  evaluated array;
* parity mode (eager authoritative, abort on the first bit mismatch) passes
  every compiled round in float32, bfloat16, float16 and the 4-bit affine
  layout of the shipping packs, with and without the turbo profile's
  verify-shaped quantized matmul kernels, greedy and sampled.

Run natively for the M5 kernels and with ``MLX_METAL_GPU_ARCH=applegpu_g16s
MTPLX_FORCE_GPU_FAMILY_FALLBACK=1`` for the M1 to M4 kernels.
"""

from __future__ import annotations

import io

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import demotions
from mtplx.attention_split import configure_split_full_attention
from mtplx.generation import generate_mtpk
from mtplx.graphbank import CompiledVerifyBank
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from tests import dense_mrope_synth as synth
from tests.test_dense_mrope_generation import _Tokenizer

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)
# The 27B pack's own settings, with a draft sampler at temperature 1.0.
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
NATIVE_DRAFT = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
TEXT = [(7 * i + 3) % 97 for i in range(16)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The attention route of the turbo profile, the profile that turns the
    # compiled verifier on (configure_split_full_attention reads it).
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    for name in (
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_POSITION_MODE",
        "MTPLX_STATE_REBASE_EVERY",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


def _unevaluated_arrays(root) -> list[str]:
    """Paths of the arrays reachable from ``root`` that are still lazy graphs.

    Walks module and container entries (private keys included, where
    ``parameters()`` stops) and the plain attribute objects hung on modules,
    such as rotary adapters. MLX has no public "is evaluated" query;
    ``mx.export_to_dot`` writes a primitive node (``shape=rectangle``) for
    every unevaluated array it reaches and only source nodes for an
    evaluated one.
    """

    found: list[str] = []
    seen: set[int] = set()

    def visit(value, path: str) -> None:
        if isinstance(value, mx.array):
            dot = io.StringIO()
            mx.export_to_dot(dot, value)
            if "shape=rectangle" in dot.getvalue():
                found.append(path)
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        if isinstance(value, nn.Module) or (
            hasattr(value, "__dict__") and type(value).__module__.startswith("mtplx")
        ):
            for key, item in vars(value).items():
                visit(item, f"{path}.{key}")

    visit(root, "model")
    return found


def _runtime(tmp_path, variant: str) -> MTPLXRuntime:
    model = synth.model_with_draft_head(tmp_path, seed=8, tie=False)
    if variant in ("bfloat16", "q4-bfloat16"):
        model.set_dtype(mx.bfloat16)
    elif variant in ("float16", "q4-float16"):
        model.set_dtype(mx.float16)
    if variant.startswith("q4"):
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda _path, module: isinstance(module, nn.Linear),
        )
    mx.eval(model.parameters())
    # What runtime.load installs on a gated-attention dense model.
    configure_split_full_attention(model)
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=tmp_path,
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _generate(rt, *, sampler=GREEDY, draft_sampler=None, max_tokens=16):
    return generate_mtpk(
        rt,
        list(TEXT),
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=draft_sampler,
        seed=11,
        speculative_depth=3,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        mtp_history_policy="committed",
    )


def _bank(out) -> dict:
    return out.stats.graphbank["compiled_verify"]


def test_the_lazy_array_detector_sees_private_and_nested_lazy_arrays():
    # The check below is only as good as its detector: a lazy array under a
    # private key, inside a list, or on an attribute object must be reported,
    # and nothing once evaluated.
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((2, 2))
            self._table = mx.arange(4) * 0.5
            self.parts = [mx.zeros((3,)) + 1.0]

    holder = Holder()
    mx.eval(holder.weight)
    assert sorted(_unevaluated_arrays(holder)) == ["model._table", "model.parts[0]"]
    mx.eval(holder._table, holder.parts)
    assert _unevaluated_arrays(holder) == []


def test_the_first_verify_trace_finds_every_model_array_evaluated(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, "q4-bfloat16")
    snapshots: list[list[str]] = []
    real = CompiledVerifyBank._make_verify_step

    def spying_make_verify_step(self, *args, **kwargs):
        snapshots.append(_unevaluated_arrays(self.runtime.model))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CompiledVerifyBank, "_make_verify_step", spying_make_verify_step)
    eager = _generate(rt)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    compiled = _generate(rt)

    assert _bank(compiled)["compiled_calls"] >= 1
    assert snapshots, "no compiled verify step was built"
    assert all(paths == [] for paths in snapshots), snapshots
    assert compiled.tokens == eager.tokens


@pytest.mark.parametrize(
    "variant",
    [
        "float32",
        "bfloat16",
        "float16",
        "q4-bfloat16",
        "q4-float16",
        "q4-bfloat16+turbo-kernels",
        "q4-float16+turbo-kernels",
    ],
)
@pytest.mark.parametrize(
    ("sampler", "draft_sampler"),
    [(GREEDY, None), (NATIVE, NATIVE_DRAFT)],
    ids=["greedy", "native"],
)
def test_parity_mode_passes_every_round_on_the_product_architecture(
    tmp_path, monkeypatch, variant, sampler, draft_sampler
):
    from mtplx import nax_verify

    dtype_variant, _, kernels = variant.partition("+")
    routed: list[int] = []
    if kernels:
        # The turbo profile's verify-shaped quantized matmul route for sampled
        # rounds: here the plain-SIMD K-split kernel for the 4-row windows
        # (the synthetic K is too small for the M5-only 16-row NAX tile).
        monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
        monkeypatch.setenv("MTPLX_NAX_M4_IMPL", "vk_k")
        real_m4 = nax_verify.nax_qmm_m4

        def counting_m4(*args, **kwargs):
            routed.append(1)
            return real_m4(*args, **kwargs)

        monkeypatch.setattr(nax_verify, "nax_qmm_m4", counting_m4)
        assert nax_verify.install_nax_qlinear_patch()["installed"]
    try:
        rt = _runtime(tmp_path, dtype_variant)
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
        on = _generate(rt, sampler=sampler, draft_sampler=draft_sampler)
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
        # A mismatch raises CompiledVerifyParityError out of the stream.
        checked = _generate(rt, sampler=sampler, draft_sampler=draft_sampler)
    finally:
        if kernels:
            # A process-global class patch: never leak it into other tests.
            nax_verify.uninstall_nax_qlinear_patch()

    bank = _bank(checked)
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert bank["parity_checks"] == bank["compiled_calls"]
    assert bank["parity_failures"] == 0
    assert checked.tokens == on.tokens
    if kernels and sampler is NATIVE:
        assert routed, "the K-split verify kernel never served a sampled round"
