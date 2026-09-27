"""Quantized KV is snapshotted at full width, and the MTP head's history counts.

The review of 9c96dd9c (finding 3): the decode-start bill priced the copy of a
banked prompt at the paged width, but ``snapshot_cache_lazy_hybrid`` reads each
cache's ``state``, and a quantized paged cache's ``state`` dequantizes
(``VllmMetalPagedKVCache._dequant_active_arrays``): q4 returns fresh
full-width arrays, q8 returns views of its bf16 mirror, which decode's first
write then copies. A 250K-token 27B snapshot is 15.3 GiB, not the 4.9 GiB of
its q4 pages. Decode also keeps a working copy beside the pages (the q8
mirror at full width, the q4 head-major bank), the per-session cap was
checked at the paged width, and the 27B's committed MTP-history cache
(4,096 B a token) was in no term at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.memory_plan as memory_plan
import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.memory_plan import plan_memory
from tests.test_memguard_admission import (
    FN_KV,
    GIB,
    Q27_KV,
    Q27_TEXT_CONFIG,
    Q27_WEIGHTS,
    _install,
    _Machine,
    _manager,
    _q27_runtime,
    _state,
)

Q4 = int(Q27_KV * 0.30)
Q8 = int(Q27_KV * 0.55)
MTP_HISTORY = 1 * 2 * 4 * 256 * 2


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _geometry(quant: str):
    """What the admission reads off a real plan with this KV quantization."""

    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=Q27_WEIGHTS,
        kv_bytes_per_token=Q27_KV,
        kv_quantization=quant,
        model_max_context=262_144,
    )
    assert plan.kv_bytes_per_token_effective == {"off": Q27_KV, "q8": Q8, "q4": Q4}[quant]
    return srv._admission_geometry(SimpleNamespace(memory_plan=plan, runtime=None))


def _cold(geometry, prompt: int, *, publish: bool = True):
    return srv._admission_growth(
        geometry,
        prompt_tokens=prompt,
        reused_tokens=0,
        restore_copies_prefix=True,
        layout="contiguous_then_repage",
        source_layout=None,
        output_tokens=16_386,
        publish=publish,
        scratch_bytes=3 * GIB,
    )


class TestQuantizedSnapshot:
    def test_a_q4_snapshot_is_the_prompt_at_full_width(self):
        growth = _cold(_geometry("q4"), 250_000)
        assert growth["publish_copy_bytes"] == 250_000 * Q27_KV
        assert round(growth["publish_copy_bytes"] / GIB, 1) == 15.3
        # Not the 4.9 GiB of its q4 pages.
        assert growth["publish_copy_bytes"] > 3 * (250_000 * Q4)

    def test_q4_decode_keeps_its_head_major_bank_beside_the_pages(self):
        growth = _cold(_geometry("q4"), 250_000, publish=False)
        assert growth["quant_working_bytes"] == 250_000 * Q4
        assert growth["decode_start_bytes"] == (
            (250_000 + 16_386) * Q4 + 250_000 * Q4
        )

    def test_q8_decode_keeps_a_full_width_mirror(self):
        growth = _cold(_geometry("q8"), 100_000, publish=False)
        assert growth["quant_working_bytes"] == 100_000 * Q27_KV
        growth = _cold(_geometry("q8"), 100_000)
        assert growth["publish_copy_bytes"] == 100_000 * Q27_KV

    def test_plain_pages_are_copied_at_their_own_width(self):
        growth = _cold(_geometry("off"), 150_000)
        assert growth["quant_working_bytes"] == 0
        assert growth["publish_copy_bytes"] == (150_000 + 16_386) * Q27_KV

    def test_the_per_session_cap_is_checked_at_the_snapshots_width(self, monkeypatch):
        """200K tokens under q4: 3.9 GB of pages, 13.1 GB of snapshot. A
        10 GiB cap refuses that snapshot at the put, so nothing is banked
        and nothing is copied; the paged width said it fitted."""

        monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q4")
        plan = plan_memory(
            total_ram_bytes=128 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_bytes_per_token=Q27_KV,
            kv_quantization="q4",
            model_max_context=262_144,
        )
        manager = _manager(max_bytes=60 * GIB, per_session_max_bytes=10 * GIB)
        state = _state(manager, plan=plan, runtime=_q27_runtime(), limit_gib=96, total_gib=128)
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: sm.SystemMemory(
                available_bytes=80 * GIB,
                total_bytes=128 * GIB,
                level_percent=60,
                free_bytes=70 * GIB,
                file_backed_bytes=10 * GIB,
                wired_bytes=30 * GIB,
                compressor_bytes=GIB,
                swap_used_bytes=0,
            ),
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=22.0, cache_gib=0.0, host_gib=1.0))
        pricing: dict = {}
        srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(200_000)),
            session_bank=manager.bank,
            session_id="deep",
            max_new_tokens=16_384,
            prefill_chunk_tokens=None,
            pricing=pricing,
        )
        assert 200_000 * Q4 < 10 * GIB < 200_000 * Q27_KV
        assert pricing["growth"]["publish_copy_bytes"] == 0


class TestMtpHistory:
    def test_the_27b_head_keeps_4096_bytes_a_token(self):
        config = {"text_config": dict(Q27_TEXT_CONFIG, mtp_num_hidden_layers=1)}
        assert (
            memory_plan.mtp_history_bytes_per_token_from_config(config)
            == MTP_HISTORY
            == 4_096
        )

    def test_families_whose_aux_already_counts_it_or_that_have_none(self):
        mtp_history_bytes_per_token_from_config = (
            memory_plan.mtp_history_bytes_per_token_from_config
        )
        flash_next = {
            "text_config": {
                "indexer_n_heads": 4,
                "layer_types": ["full_attention"] * 12,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "mtp_num_hidden_layers": 1,
            }
        }
        assert mtp_history_bytes_per_token_from_config(flash_next) == 0
        assert mtp_history_bytes_per_token_from_config({"text_config": Q27_TEXT_CONFIG}) == 0
        assert mtp_history_bytes_per_token_from_config(None) == 0

    def test_the_plan_carries_it_and_its_fit_is_unchanged(self):
        common = dict(
            total_ram_bytes=48 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_bytes_per_token=Q27_KV,
            model_max_context=262_144,
        )
        without = plan_memory(**common)
        with_history = plan_memory(**common, mtp_history_bytes_per_token=MTP_HISTORY)
        assert with_history.mtp_history_bytes_per_token == MTP_HISTORY
        assert with_history.to_dict()["mtp_history_bytes_per_token"] == MTP_HISTORY
        assert with_history.context_window_fit == without.context_window_fit

    def test_the_admission_prices_every_row_with_it(self):
        plan = SimpleNamespace(
            available=True,
            kv_bytes_per_token=Q27_KV,
            kv_bytes_per_token_effective=Q27_KV,
            aux_bytes_per_token=0,
            mtp_history_bytes_per_token=MTP_HISTORY,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_quantization="off",
        )
        geometry = srv._admission_geometry(SimpleNamespace(memory_plan=plan, runtime=None))
        assert geometry.live_bytes_per_token == Q27_KV + MTP_HISTORY
        assert geometry.paged_bytes_per_token == Q27_KV + MTP_HISTORY
        assert geometry.aux_bytes_per_token == MTP_HISTORY
        # A QSA family's aux already has its MTP head; nothing is added.
        qsa = SimpleNamespace(**{**vars(plan), "aux_bytes_per_token": 7_872,
                                 "mtp_history_bytes_per_token": 0,
                                 "kv_bytes_per_token": FN_KV,
                                 "kv_bytes_per_token_effective": FN_KV})
        assert srv._admission_geometry(
            SimpleNamespace(memory_plan=qsa, runtime=None)
        ).live_bytes_per_token == FN_KV + 7_872

    def test_the_server_hands_it_to_the_plan(self):
        import inspect

        src = inspect.getsource(srv.ServerState.__init__)
        assert '"mtp_history_bytes_per_token": (' in src
        assert "_plan_mtp_history_from_config(_plan_model_config)" in src
