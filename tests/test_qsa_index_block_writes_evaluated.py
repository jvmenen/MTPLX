"""A stock QSACache's index-block writes are evaluated with the forward (#544).

``QSACache.write_pooled`` stores each completed index block in ``pooled`` and
in its fp32 mirror with lazy slice updates. A forward reads those buffers only
past the selection budget (2,048 tokens on Flash-Next), and then only one of
the two; a draft-head history append reads neither. Before the fix an unread
buffer stacked one update per block into a single unevaluated graph for the
life of the cache. The graph kept every written block alive, and with it the
Metal shared event of the ``mx.async_eval`` that computed the block. The
session bank snapshots lazily, so each banked final state held one event per
index write of its request. A long-running non-stream Flash-Next server then
failed with ``[Event::Event] Failed to create Metal shared event`` after 140K
to 260K generated tokens.

Pinned here:
- after every evaluated forward the pooled buffer and its mirror hold no
  unevaluated work, below and above the budget;
- a forward whose output is dropped leaves only its own writes pending;
- the dependency changes no value, on the CPU and on the GPU;
- live Metal shared events stay flat over many rounds, and a lazy bank
  snapshot of the cache pins none of them;
- a tiny Flash-Next generation hands the bank index state whose pending work
  does not grow with the length of the generation.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import io
import os
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.graphbank as graphbank
import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.cache_state import snapshot_cache, snapshot_cache_lazy_hybrid
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

_GPU = mx.metal.is_available()


def _tiny_args() -> TextArgs:
    # budget 8 / ratio 2: selection engages once more than 4 blocks complete,
    # so a short run covers the unread regime and the selecting one.
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


def _layer(device, dtype=mx.float32) -> Attention:
    mx.set_default_device(device)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    if dtype != mx.float32:
        layer.update(mlx.utils.tree_map(lambda p: p.astype(dtype), layer.parameters()))
    mx.eval(layer.parameters())
    return layer


@pytest.fixture()
def restore_device():
    previous = mx.default_device()
    yield
    mx.set_default_device(previous)


def _hidden(tokens: int, seed: int, dtype=mx.float32) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(dtype)


def _pending_ops(array: mx.array | None) -> int:
    """Primitives not yet evaluated in ``array``'s graph (0 once evaluated)."""

    if array is None:
        return 0
    out = io.StringIO()
    mx.export_to_dot(out, array)
    return out.getvalue().count("shape=rectangle")


def _pending_writes(array: mx.array | None) -> int:
    """Slice updates not yet evaluated in ``array``'s graph."""

    if array is None:
        return 0
    out = io.StringIO()
    mx.export_to_dot(out, array)
    return out.getvalue().count('label ="SliceUpdate"')


def _live_metal_shared_events() -> int | None:
    """Live ``MTLSharedEvent`` objects in this process, from ``heap(1)``."""

    tool = shutil.which("heap")
    if tool is None:
        return None
    result = subprocess.run(
        [tool, str(os.getpid())], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "_MTLSharedEvent" in line:
            return int(line.split()[0])
    return 0


def test_every_evaluated_forward_leaves_the_index_buffers_evaluated(restore_device):
    layer = _layer(mx.cpu)
    cache = QSACache()
    mx.eval(layer(_hidden(2, seed=1), cache))
    pooled_seen = mirror_seen = False
    for step in range(30):
        out = layer(_hidden(1, seed=100 + step), cache)
        mx.eval(out)
        pooled_seen |= cache.pooled is not None
        mirror_seen |= cache.pooled_f32_t is not None
        assert _pending_ops(cache.pooled) == 0, f"pooled backlog after forward {step}"
        assert _pending_ops(cache.pooled_f32_t) == 0, f"mirror backlog after forward {step}"
    # The run crossed the budget (2 + 30 tokens, 16 blocks against a top-k
    # of 4), so both the unread regime and the selecting one were exercised.
    assert cache.offset == 32
    assert pooled_seen and mirror_seen


def test_a_dropped_forward_leaves_only_its_own_writes_pending(restore_device):
    layer = _layer(mx.cpu)
    cache = QSACache()
    mx.eval(layer(_hidden(4, seed=2), cache))
    # Two rows complete one block. The output is dropped, the way a lazy
    # draft-head history append drops its own.
    layer(_hidden(2, seed=3), cache)
    one_forward = _pending_ops(cache.pooled)
    assert one_forward > 0
    mx.eval(layer(_hidden(1, seed=4), cache))
    assert _pending_ops(cache.pooled) == 0
    assert _pending_ops(cache.pooled_f32_t) == 0


def _run_sequence(layer: Attention, dtype) -> list[mx.array]:
    """Prefill, decode across the budget, a rolled-back verify, a restore."""

    produced: list[mx.array] = []
    cache = QSACache()
    produced.append(layer(_hidden(6, seed=10, dtype=dtype), cache))
    for step in range(8):
        produced.append(layer(_hidden(1, seed=20 + step, dtype=dtype), cache))
        mx.eval(produced[-1])
    produced.append(layer(_hidden(4, seed=40, dtype=dtype), cache))
    mx.eval(produced[-1])
    assert cache.trim(3) == 3
    for step in range(6):
        produced.append(layer(_hidden(1, seed=50 + step, dtype=dtype), cache))
    resumed = QSACache()
    resumed.state = cache.state
    for step in range(4):
        produced.append(layer(_hidden(1, seed=70 + step, dtype=dtype), resumed))
    produced.extend(leaf for leaf in resumed.state if leaf is not None)
    produced.append(resumed.pooled_f32_view(resumed.pooled_len))
    mx.eval(produced)
    return produced


@pytest.mark.parametrize(
    "device,dtype",
    [
        pytest.param(mx.cpu, mx.float32, id="cpu-fp32"),
        pytest.param(
            mx.gpu,
            mx.bfloat16,
            id="gpu-bf16",
            marks=pytest.mark.skipif(not _GPU, reason="needs the Metal GPU"),
        ),
    ],
)
def test_the_dependency_changes_no_value(restore_device, monkeypatch, device, dtype):
    layer = _layer(device, dtype)
    tied = _run_sequence(layer, dtype)
    monkeypatch.setattr(qwen4_exp, "_after_index_block_writes", lambda rows, _cache: rows)
    untied = _run_sequence(layer, dtype)
    assert len(tied) == len(untied)
    for index, (a, b) in enumerate(zip(tied, untied)):
        assert a.shape == b.shape and a.dtype == b.dtype, index
        assert mx.array_equal(a, b).item(), f"value {index} differs"


@pytest.mark.skipif(not _GPU, reason="Metal shared events exist only on the GPU")
def test_metal_shared_events_stay_flat_across_rounds(restore_device):
    before = _live_metal_shared_events()
    if before is None:
        pytest.skip("heap(1) is not available to count Metal shared events")
    layer = _layer(mx.gpu, mx.bfloat16)
    cache = QSACache()
    mx.eval(layer(_hidden(2, seed=5, dtype=mx.bfloat16), cache))
    for step in range(96):
        out = layer(_hidden(1, seed=200 + step, dtype=mx.bfloat16), cache)
        # The decode loop's shape: one pipelined evaluation per round, then
        # a host read of its result.
        mx.async_eval(out)
        out[0, -1, 0].item()
    live = _live_metal_shared_events()
    # The bank keeps a lazy snapshot of the final state.
    snapshots = [snapshot_cache([cache]), snapshot_cache_lazy_hybrid([cache])]
    del cache, out
    gc.collect()
    banked = _live_metal_shared_events()
    del snapshots
    gc.collect()
    # 96 rounds complete 48 index blocks; before the fix each one pinned the
    # event of the round that wrote it, live and in the snapshots.
    assert live - before <= 4, f"{live - before} Metal shared events pinned after 96 rounds"
    assert banked - before <= 4, f"{banked - before} Metal shared events pinned by the snapshots"


_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"


@pytest.fixture(scope="module")
def tiny_pack():
    """The tiny random Flash-Next pack on the compiled fixed-M4 lane."""

    if not _GPU:
        pytest.skip("bfloat16 expert gathers need the GPU")
    import mlx_lm.models.cache as cache_module

    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", _SMOKE)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
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
    mx.set_default_device(previous_device)


def _generate(tiny_pack, monkeypatch, max_tokens: int):
    import mtplx.generation as generation
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.sampling import SamplerConfig

    smoke, model = tiny_pack
    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_LAZY_MTP_HISTORY_APPEND", "1")
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    out = generation.generate_mtpk(
        rt,
        [3, 5, 7, 9, 11, 13] + list(range(20, 40)),
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=3,
        seed=1234,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )
    assert out.final_state is not None and len(out.tokens) == max_tokens
    return out


def _final_index_backlog(tiny_pack, monkeypatch, max_tokens: int) -> list[int]:
    final = _generate(tiny_pack, monkeypatch, max_tokens).final_state
    entries = list(final.final_trunk_cache) + list(final.final_committed_mtp_cache or [])
    qsa = [entry for entry in entries if isinstance(entry, QSACache)]
    assert qsa, "the tiny pack has QSA layers in the trunk and the draft head"
    return [_pending_writes(entry.pooled) for entry in qsa]


def test_a_generation_hands_the_bank_index_state_that_does_not_grow(tiny_pack, monkeypatch):
    short = _final_index_backlog(tiny_pack, monkeypatch, max_tokens=120)
    long = _final_index_backlog(tiny_pack, monkeypatch, max_tokens=360)
    # What is left is the last few unevaluated draft-head history appends,
    # whatever the length. Before the fix this grew with every index write of
    # the request (hundreds of slice updates at these lengths).
    assert max(long) <= max(short) + 4, (short, long)
    assert max(long) <= 16, long


def test_a_generation_decodes_the_same_tokens_without_the_dependency(tiny_pack, monkeypatch):
    tied = _generate(tiny_pack, monkeypatch, max_tokens=160)
    monkeypatch.setattr(qwen4_exp, "_after_index_block_writes", lambda rows, _cache: rows)
    untied = _generate(tiny_pack, monkeypatch, max_tokens=160)
    assert list(tied.tokens) == list(untied.tokens)
    assert mx.array_equal(tied.final_state.final_logits, untied.final_state.final_logits).item()
