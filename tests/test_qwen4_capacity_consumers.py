"""Capacity buckets preserve every consumer of a fixed QSA bank."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.cache_state import restore_cache, snapshot_cache, snapshot_untrimmable_cache_lazy
from mtplx.models.qwen4_exp import QSACache
from test_qwen4_fixed_m4_capacity_bucket import (
    NATIVE, PROMPT, SEED, _bits, _prefilled_entry, _prompt, cpu_layer, lane, pack,
)


def _leaves(value):
    if isinstance(value, mx.array):
        return [_bits(value)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    return []


def _state(cache):
    result = []
    for entry in cache:
        if isinstance(entry, graphbank.TensorOffsetQSACache):
            end = entry.size()
            result.extend(_leaves((entry.kv.keys[:, :, :end], entry.kv.values[:, :, :end],
                                   entry.raw_keys[:, :end], entry.pooled[:, :end // entry.ratio])))
        else:
            result.extend(_leaves(entry.state))
    return result


def _equal(left, right):
    assert len(left) == len(right)
    for i, (a, b) in enumerate(zip(left, right)):
        assert a.shape == b.shape, (i, a.shape, b.shape)
        assert np.array_equal(a, b), i


@pytest.mark.parametrize("tile", [0, 1, 2, 4])
def test_dense_consumers_keep_the_parent_sdpa_width(cpu_layer, lane, tile):
    import mtplx.models.qwen4_exp as qwen4

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    lane.setenv("MTPLX_QSA_SCORE_TILE_ROWS", str(tile))
    banks = [graphbank.TensorOffsetQSACache.from_qsa_cache(
        _prefilled_entry(cpu_layer), reserve_tokens=32, capacity_bucket=bucket,
    ) for bucket in (0, 1024)]
    if tile:
        assert banks[1].capacity == banks[0].capacity == 256
    else:
        assert [bank.capacity for bank in banks] == [256, 1024]
    widths = []
    sdpa = qwen4._verify_sdpa

    def observe(q, k, v, **kwargs):
        widths.append(k.shape[2])
        return sdpa(q, k, v, **kwargs)

    lane.setattr(qwen4, "_verify_sdpa", observe)
    for rows in (4, 1, 3, 9, 1):
        mx.random.seed(91 + rows)
        x = mx.random.normal((1, rows, 64)).astype(mx.bfloat16)
        outputs = [cpu_layer(x, bank) for bank in banks]
        _equal(_leaves(outputs[0]), _leaves(outputs[1]))
        if rows == 1 or 0 < tile < rows:
            # A behavioral check on the actual SDPA arguments, independent
            # of whether this device happens to round both widths equally.
            assert widths[-2:] == [256, 256]
        _equal(_state([banks[0]]), _state([banks[1]]))


def test_short_request_admission_prices_the_constructed_bank(lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "8192")
    args = SimpleNamespace(layer_types=["full_attention"], num_key_value_heads=2,
                           head_dim=16, indexer_head_dim=16, indexer_compress_ratio=4)
    rt = SimpleNamespace(model=SimpleNamespace(args=args), qwen4_fixed_m4_compiled_verify=True)
    lane.setattr(generation, "_mlx_live_memory_bytes", lambda: 0)
    lane.setattr(generation, "_metal_memory_limit_bytes", lambda rt: 2**30)
    plan = graphbank.FixedM4CapacityPlan.for_request(32)
    receipt = {}
    assert generation._qwen4_fixed_m4_compiled_verify_requested(
        rt, verify_strategy="batched", compiled_mode="on", max_tokens=32,
        cached_tokens=0, prompt_tokens=24_000, receipt=receipt, capacity_plan=plan,
    )
    entry = QSACache(4)
    entry.kv.keys = mx.zeros((1, 2, 24_000, 16), dtype=mx.bfloat16)
    entry.kv.values = mx.zeros_like(entry.kv.keys)
    entry.kv.offset = 24_000
    entry.raw_keys = mx.zeros((1, 24_000, 16), dtype=mx.bfloat16)
    entry.pooled = mx.zeros((1, 6_000, 16), dtype=mx.bfloat16)
    entry.pooled_len = 6_000
    bank = graphbank.TensorOffsetQSACache.from_qsa_cache(
        entry, reserve_tokens=plan.reserve_tokens, capacity_plan=plan,
    )
    assert receipt["reserve_tokens"] == 36
    assert bank.capacity == receipt["promotion_rows"] == 24_576
    assert receipt["promotion_bytes"] == bank.nbytes - bank.offset.nbytes


def test_short_request_gate_passes_its_real_budget_to_admission(lane):
    """Fails behaviorally on 50de43bb: its gate charges the 1,024-row reserve."""
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "0")
    lane.setattr(generation, "_mlx_live_memory_bytes", lambda: 0)
    lane.setattr(generation, "_metal_memory_limit_bytes", lambda rt: 10**9)
    lane.setattr(generation, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: 100)
    receipt = {}
    assert generation._qwen4_fixed_m4_compiled_verify_requested(
        SimpleNamespace(qwen4_fixed_m4_compiled_verify=True),
        verify_strategy="batched", compiled_mode="on", max_tokens=32,
        cached_tokens=0, prompt_tokens=24_000, receipt=receipt,
    )
    assert receipt["promotion_bytes"] == 24_064 * 100


def _partial_session(pack, patch, bucket):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    patch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    cache = model.make_cache()
    rt.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)
    bank = graphbank.CompiledVerifyBank(rt, max_verify_len=4, request_max_tokens=96)
    bank.install_fixed_m4(cache, prompt_ids=PROMPT, hidden_variant=None)
    records, completion, capacities = [], [], []
    for i in range(20):
        ids = [(i * 13 + j * 7) % 128 for j in range(4)]
        snap = snapshot_untrimmable_cache_lazy(cache)
        logits, hidden, _ = bank.forward_fixed_m4(
            mx.array([ids]), host_input_ids=ids, completion_tokens=completion,
            committed_count=len(completion), cache=cache,
        )
        records.extend(_leaves((logits, hidden)))
        for entry in cache:
            records.extend(_leaves(getattr(entry, "_mtplx_verify_rows", ())))
            records.extend(_leaves(getattr(entry, "_mtplx_verify_ple", ())))
        keep = 1 + i % 4
        assert model.language_model.model.commit_verified_window(
            cache, snap.states, keep_tokens=keep, verified_tokens=4,
        )
        completion.extend(ids[:keep])
        records.extend(_state(cache))
        qsa = next(entry for entry in cache if isinstance(entry, graphbank.TensorOffsetQSACache))
        capacities.append((qsa.capacity, qsa.dense_capacity, qsa.fixed_rows_gather))
    # Same one-row call as generation's final-pending capture, before demotion.
    bank.reserve_fixed_m4_window(cache, committed_count=len(completion), window_tokens=1)
    records.extend(_leaves(rt.forward_ar(mx.array([[17]]), cache=cache, return_hidden=True)))
    records.extend(_state(cache))
    bank.demote(cache)
    snapshot = snapshot_cache(cache)
    restored = model.make_cache()
    restore_cache(restored, snapshot)
    records.extend(_state(restored))
    records.extend(_leaves(rt.forward_ar(mx.array([[21, 23, 25]]), cache=restored, return_hidden=True)))
    records.extend(_state(restored))
    return records, capacities


def test_partial_acceptance_restore_and_dense_to_gather_growth_are_identical(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "48")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    narrow, narrow_caps = _partial_session(pack, lane, 0)
    wide, wide_caps = _partial_session(pack, lane, 1024)
    assert not narrow_caps[0][2] and narrow_caps[-1][2]
    assert any(a[0] != b[0] for a, b in zip(narrow_caps, wide_caps))
    assert [a[1:] for a in narrow_caps] == [b[1:] for b in wide_caps]
    _equal(narrow, wide)


def _captured_session(pack, patch, bucket):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.session_bank import SessionBank

    smoke, model = pack
    patch.setenv("MTPLX_COMPILED_VERIFY", "1")
    patch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    bank = SessionBank(max_bytes=2**28, per_session_max_bytes=2**28)
    pending_calls = []
    forward = rt.forward_ar

    def observe(ids, **kwargs):
        if ids.shape[1] == 1 and any(
            isinstance(entry, graphbank.TensorOffsetQSACache)
            for entry in kwargs.get("cache", ())
        ):
            pending_calls.append(True)
        return forward(ids, **kwargs)

    patch.setattr(rt, "forward_ar", observe)
    prompt = list(PROMPT)
    records, tokens, restored_counts = [], [], []
    for turn in range(2):
        result = generation.generate_mtpk(
            rt, prompt, max_tokens=40, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched", stop_token_ids=set(),
            capture_final_state=True, session_bank=bank,
        )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        records.extend(_leaves((final.final_logits, final.final_hidden)))
        records.extend(_state(final.final_trunk_cache))
        records.extend(_state(final.final_committed_mtp_cache))
        tokens.append(list(result.tokens))
        restored_counts.append(result.stats.cached_tokens)
        prefix = prompt + list(result.tokens)
        assert bank.put(
            runtime=rt, token_ids=prefix, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden,
            mtp_history_policy="committed",
            mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
        ) is not None
        prompt = prefix + [31, 37, 41, 43]
    assert restored_counts[0] == 0 and restored_counts[1] > 0
    assert pending_calls, "must exercise the one-row forward before bank demotion"
    return records, tokens


def test_final_pending_capture_and_next_session_bank_turn_are_identical(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    narrow, narrow_tokens = _captured_session(pack, lane, 0)
    wide, wide_tokens = _captured_session(pack, lane, 1024)
    assert narrow_tokens == wide_tokens
    _equal(narrow, wide)


@pytest.mark.parametrize("allow_step", [False, True])
def test_growth_is_admitted_before_any_leaf_changes(pack, lane, allow_step):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    cache = model.make_cache()
    rt.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)
    plan = graphbank.FixedM4CapacityPlan.for_request(1000)
    bank = graphbank.CompiledVerifyBank(rt, max_verify_len=4, request_max_tokens=1000,
                                        capacity_plan=plan)
    bank.install_fixed_m4(cache, prompt_ids=PROMPT, hidden_variant=None)
    qsa = next(entry for entry in cache if isinstance(entry, graphbank.TensorOffsetQSACache))
    refs = qsa.state_leaves
    requests = []

    def admit(rows):
        assert qsa.capacity == 512 and qsa.dense_capacity == 256
        assert all(a is b for a, b in zip(refs, qsa.state_leaves))
        requests.append(rows)
        return allow_step and rows == 768

    plan.admit_growth = admit
    if allow_step:
        bank.reserve_fixed_m4_window(cache, committed_count=469)
        assert qsa.capacity == qsa.dense_capacity == 768
        assert plan.bucket == qsa.capacity_bucket == 0
    else:
        with pytest.raises(MemoryError, match="growth exceeds memory admission"):
            bank.reserve_fixed_m4_window(cache, committed_count=469)
        assert qsa.capacity == 512 and qsa.dense_capacity == 256
        assert all(a is b for a, b in zip(refs, qsa.state_leaves))
    assert requests == [1024, 768]
