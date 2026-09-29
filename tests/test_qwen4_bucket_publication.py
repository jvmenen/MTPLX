"""Published bucketed QSA state owns only the bytes the session bank charges."""

import gc
import time

import mlx.core as mx
import numpy as np
import pytest

from mtplx.graphbank import TensorOffsetQSACache
from mtplx.models.qwen4_exp import QSACache
from mtplx.session_bank import SessionBank
from test_qwen4_fixed_m4_capacity_bucket import _bits, lane, pack


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_bucket_demotion_releases_slack_before_lazy_session_publication(pack, lane, dtype):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    smoke, model = pack
    runtime = smoke._tiny_runtime(model)
    # One production-shaped attention layer. A 16K prompt grants 17,408
    # rows inside a 24,576-row bucket; the bank must not retain its slack.
    offset = 16_384
    logical_bytes = offset * (2 * 2 * 256 + 128 + 128 // 4) * 2
    session = SessionBank(max_bytes=logical_bytes * 3 // 2,
                          per_session_max_bytes=logical_bytes * 3 // 2)
    mx.synchronize()
    gc.collect()
    before = mx.get_active_memory()

    def publish(token):
        entry = QSACache(4)
        entry.indexer_budget = 2048
        entry.kv.keys = mx.full((1, 2, offset, 256), 0.25, dtype=dtype)
        entry.kv.values = mx.full((1, 2, offset, 256), -0.5, dtype=dtype)
        entry.kv.offset = offset
        entry.raw_keys = mx.full((1, offset, 128), 0.75, dtype=dtype)
        entry.pooled = mx.full((1, offset // 4, 128), -0.25, dtype=dtype)
        entry.pooled_len = offset // 4
        promoted = TensorOffsetQSACache.from_qsa_cache(
            entry, reserve_tokens=1024, capacity_bucket=8192,
        )
        assert promoted.capacity == 24_576 and promoted.dense_capacity == 17_408
        mx.eval(*promoted.state_leaves)
        started = time.perf_counter()
        demoted = promoted.demote()
        mx.eval(*demoted.state)
        elapsed = time.perf_counter() - started
        for original, published in zip(entry.state, demoted.state):
            assert np.array_equal(_bits(original), _bits(published))
        saved = session.put(
            runtime=runtime, token_ids=[token] * offset, cache=[demoted],
            logits=mx.zeros((1, 128)), hidden=mx.zeros((1, 1, 64)),
        )
        assert saved is not None
        mx.eval(*saved.cache_snapshot.states[0])
        return elapsed

    for token in (31, 37):
        elapsed = publish(token)
        gc.collect()
        mx.synchronize()
        retained = mx.get_active_memory() - before
        print(f"publication dtype={dtype} rows={offset} time_ms={elapsed * 1000:.3f} "
              f"retained={retained} charged={session.total_nbytes}")
        # A second independent session must evict the first on the real
        # budget, and leave at most allocator page rounding unaccounted.
        assert len(session) == 1
        assert retained <= session.total_nbytes + 1024 * 1024, (
            "published views retained uncharged bucket storage", retained, session.total_nbytes,
        )
