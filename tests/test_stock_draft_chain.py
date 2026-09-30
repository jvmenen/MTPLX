"""MTPLX_STOCK_SAMPLED_DRAFT_CHAIN: the sampled draft chain on the stock full-row reader.

The lane's claim is identity: drafted tokens, proposal distributions and the
request generator's stream must be exactly those of the stock serial reader
(``sparse_distribution_from_mlx_logits`` + ``Generator.choice``), for every
seed, including rounds where the host does not confirm the device prediction.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx import stock_draft_chain as chain
from mtplx.fast_sampling import (
    _device_serial_support_arrays,
    serial_support_device_arrays,
    serial_support_host_rows,
    sparse_distribution_from_mlx_logits,
)
from mtplx.sampling import SamplerConfig
from tests.test_qwen4_draft_device_chain import (  # noqa: F401  (shared stub runtime)
    TOP_K,
    VOCAB,
    _events_are_recorded,
    _fingerprint,
    _runtime,
)


def _rows(seed: int, vocab: int = 512, ties: bool = False) -> mx.array:
    rng = np.random.default_rng(seed)
    row = rng.normal(0, 2.5, size=(vocab,)).astype(np.float32)
    if ties:
        # Many exact ties around the top-k cutoff force the spill fallback.
        row = np.round(row).astype(np.float32)
    return mx.array(row).astype(mx.bfloat16)


@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("ties", [False, True])
def test_split_serial_support_is_the_serial_reader(top_p, ties):
    config = SamplerConfig(temperature=1.0, top_p=top_p, top_k=20)
    for seed in range(20):
        row = _rows(seed, ties=ties).reshape(-1).astype(mx.float32)
        want_ids, want_probs, want_vocab = _device_serial_support_arrays(row, config)
        got_ids, got_probs, got_vocab = serial_support_host_rows(
            serial_support_device_arrays(row, config), config
        )
        assert got_vocab == want_vocab
        np.testing.assert_array_equal(got_ids, want_ids)
        np.testing.assert_array_equal(got_probs, want_probs)


@pytest.mark.parametrize("top_p", [0.95, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("ties", [False, True])
def test_host_read_is_the_stock_reader(top_p, temperature, ties):
    config = SamplerConfig(temperature=temperature, top_p=top_p, top_k=20)
    plan = chain.StockChainPlan(config=config)
    for seed in range(20):
        logits = _rows(seed, ties=ties).reshape(1, 1, -1)
        want = sparse_distribution_from_mlx_logits(logits[:, -1, :][0], config)
        got = chain.host_read(plan, chain.device_support(plan, logits))
        np.testing.assert_array_equal(got.token_ids, want.token_ids)
        np.testing.assert_array_equal(got.probs, want.probs)
        assert got.vocab_size == want.vocab_size


def test_host_pick_is_generator_choice_with_the_same_stream():
    config = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    for trial in range(200):
        logits = _rows(trial).reshape(1, 1, -1)
        dist = sparse_distribution_from_mlx_logits(logits[:, -1, :][0], config)
        a = np.random.default_rng(trial)
        b = np.random.default_rng(trial)
        assert chain.host_pick(dist, float(b.random())) == int(
            a.choice(dist.token_ids, p=dist.probs)
        )
        assert a.random() == b.random()


@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_device_prediction_matches_the_host_pick(top_p):
    config = SamplerConfig(temperature=1.0, top_p=top_p, top_k=20)
    plan = chain.StockChainPlan(config=config)
    rng = np.random.default_rng(3)
    misses = 0
    trials = 300
    for seed in range(trials):
        logits = _rows(seed).reshape(1, 1, -1)
        support = chain.device_support(plan, logits)
        uniform = float(rng.random())
        predicted = chain.predict(plan, support, mx.array(uniform, dtype=mx.float32))
        mx.eval(*chain.eval_arrays(support), predicted)
        host = chain.host_pick(chain.host_read(plan, support), uniform)
        misses += int(int(predicted.item()) != host)
    # A prediction may only miss when the uniform sits at float32 distance
    # from a cdf edge.
    assert misses <= 2


def test_claim_declines_greedy_and_penalised_drafts():
    assert chain.claim(None, SamplerConfig(temperature=0.0, top_k=20))[0] is None
    assert chain.claim(None, SamplerConfig(temperature=1.0, top_k=0))[0] is None
    assert (
        chain.claim(None, SamplerConfig(temperature=1.0, top_k=20, presence_penalty=0.5))[0]
        is None
    )
    plan, reason = chain.claim(None, SamplerConfig(temperature=1.0, top_k=20))
    assert plan is not None and reason is None


# ---------------------------------------------------------------------------
# The generation loop (stub runtime shared with the Flash-Next chain tests)
# ---------------------------------------------------------------------------


def _stock_runtime():
    rt = _runtime()
    rt.qwen4_relaxed_draft_ties = False  # the default serial reader
    return rt


def _generate(seed: int, max_tokens: int = 48):
    from mtplx.generation import generate_mtpk

    return generate_mtpk(
        _stock_runtime(),
        [3],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=TOP_K),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
        seed=seed,
    )


@pytest.fixture(autouse=True)
def _no_chain_env(monkeypatch):
    monkeypatch.delenv("MTPLX_STOCK_SAMPLED_DRAFT_CHAIN", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN", raising=False)


@pytest.mark.parametrize("pipeline", ["1", "0"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_chain_is_token_identical_to_the_stock_serial_lane(monkeypatch, seed, pipeline):
    baseline = _generate(seed)
    assert baseline.stats.sampled_draft_chain_status == "off"
    monkeypatch.setenv("MTPLX_STOCK_SAMPLED_DRAFT_CHAIN", "1")
    monkeypatch.setenv("MTPLX_QWEN4_DRAFT_CHAIN_PIPELINE", pipeline)
    on = _generate(seed)
    assert on.stats.sampled_draft_chain_status == "installed"
    assert on.stats.sampled_draft_chain_rounds > 0
    assert _fingerprint(on) == _fingerprint(baseline)
    assert any(
        draft.get("draft_core") == "sampled-chain"
        for event in (on.stats.events or [])
        for draft in event.get("drafts", [])
    )


@pytest.mark.parametrize("seed", [0, 5])
def test_an_unconfirmed_prediction_finishes_serially_with_the_same_tokens(monkeypatch, seed):
    baseline = _generate(seed)

    def always_wrong(plan, support, uniform):
        return mx.array(VOCAB - 1, dtype=mx.int32) + 0 * mx.sum(support.cand_idx).astype(
            mx.int32
        )

    monkeypatch.setenv("MTPLX_STOCK_SAMPLED_DRAFT_CHAIN", "1")
    monkeypatch.setattr(chain, "device_predict", always_wrong)
    on = _generate(seed)
    assert on.stats.sampled_draft_chain_cuts > 0
    assert _fingerprint(on) == _fingerprint(baseline)


def test_relaxed_tie_reader_declines_the_stock_chain(monkeypatch):
    from mtplx.generation import generate_mtpk

    monkeypatch.setenv("MTPLX_STOCK_SAMPLED_DRAFT_CHAIN", "1")
    rt = _runtime()  # qwen4_relaxed_draft_ties = True
    out = generate_mtpk(
        rt,
        [3],
        max_tokens=16,
        sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=TOP_K),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
        seed=0,
    )
    assert out.stats.sampled_draft_chain_status == "declined:draft_reader_not_default"
