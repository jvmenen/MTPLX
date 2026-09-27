"""The admission prices what each backend runs, not what the chunk settings say.

The review of 9c96dd9c (finding 1): Gemma 4 was priced as a chunked prefill,
but ``_gemma4_prefill_prompt`` forwards every uncached token in one call; its
runtime has no ``model.args``, so its scratch fell to the flat per-row bill;
and neither Gemma branch handed the request's abort check on, so the
per-chunk memory check never ran there. Reading the backend's code gives
three more ways the generic model misread it:

* after a prefill forward every sliding layer holds all of the new rows
  (``Gemma4RollbackRotatingKVCache`` trims on the next update), but once
  decode runs a token keeps only the full-attention layers' KV: 81,920 B on
  the 31B against the planner's 983,040 for all 60 layers, which the restore
  and the pre-decode clone were charged at;
* the sliding layers build a rows x (cached window + rows) boolean mask for
  one forward (4.3 GB for a 65,536-token cold prompt);
* the prompt cache is cloned before decode whenever a session bank is
  present, whatever store-on-prefill says.

Plus the audit of the 4B: the generic loop, its own geometry, the profile's
chunk. No model is loaded: the runtimes carry the real configs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.backends.gemma4_assistant as gemma4
import mtplx.generation as generation
import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.memory_plan import dense_kv_bytes_per_token_from_config
from tests.test_memguard_admission import (
    GIB,
    _install,
    _Machine,
    _manager,
    _state,
)

# google/gemma-4-31B-it's text_config (the Gemma 4 Optimized Speed target).
GEMMA31B_TEXT = dict(
    model_type="gemma4_text",
    hidden_size=5376,
    num_hidden_layers=60,
    intermediate_size=21504,
    num_attention_heads=32,
    num_key_value_heads=16,
    head_dim=256,
    global_head_dim=512,
    num_global_key_value_heads=4,
    sliding_window=1024,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    attention_k_eq_v=True,
    hidden_size_per_layer_input=0,
    enable_moe_block=False,
    max_position_embeddings=262_144,
    vocab_size=262_144,
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
)
GEMMA_PLANNED_KV = dense_kv_bytes_per_token_from_config({"text_config": GEMMA31B_TEXT})
GEMMA_RESIDENT = 10 * 2 * 4 * 512 * 2
GEMMA_WINDOWS = 50 * 2 * 16 * 256 * 2 * 1024
GEMMA_ROW = 2 * (5376 + 3 * 21504)  # the MLP, the widest layer
GEMMA_WEIGHTS = int(17.5 * GIB)

# Qwen3.5-4B's text_config (the 4B Optimized Speed pack's config.json).
Q4B_TEXT = dict(
    model_type="qwen3_5_text",
    hidden_size=2560,
    intermediate_size=9216,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=32,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262_144,
    vocab_size=248_320,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    # The served profiles prefill in 2,048-row chunks with the auto layout.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _gemma_text_args():
    from mlx_lm.models.gemma4_text import ModelArgs

    return ModelArgs.from_dict(GEMMA31B_TEXT)


def _gemma_runtime():
    """The backend's runtime class with the target's text model reduced to
    its config: every admission question goes through the real methods."""

    runtime = object.__new__(gemma4.Gemma4AssistantRuntime)
    runtime.target = SimpleNamespace(
        text_model=SimpleNamespace(config=_gemma_text_args())
    )
    runtime.model_path = Path("models/gemma4-31b/target")
    runtime.mtp_enabled = True
    runtime.backend_id = gemma4.BACKEND_NAME
    return runtime


def _gemma_state(manager, *, limit_gib: float = 96):
    plan = SimpleNamespace(
        available=True,
        kv_bytes_per_token=GEMMA_PLANNED_KV,
        kv_bytes_per_token_effective=GEMMA_PLANNED_KV,
        aux_bytes_per_token=0,
        prefill_transient_bytes_per_token=0,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=GEMMA_WEIGHTS,
    )
    return _state(
        manager, plan=plan, runtime=_gemma_runtime(), limit_gib=limit_gib, total_gib=128
    )


def _roomy(monkeypatch, manager):
    monkeypatch.setattr(
        sm,
        "_reader",
        lambda: sm.SystemMemory(
            available_bytes=90 * GIB,
            total_bytes=128 * GIB,
            level_percent=70,
            free_bytes=80 * GIB,
            file_backed_bytes=10 * GIB,
            wired_bytes=20 * GIB,
            compressor_bytes=GIB,
            swap_used_bytes=0,
        ),
    )
    _install(monkeypatch, _Machine(manager.bank, base_gib=18.0, cache_gib=0.0, host_gib=1.0))


def _gemma_scratch(rows: int, cached: int) -> int:
    per_row = srv._ADMISSION_LIVE_LAYERS * GEMMA_ROW
    fixed = max(srv._ADMISSION_FIXED_FLOOR_BYTES, 3 * GIB - per_row * 2048)
    flat_share = 3 * GIB * rows // 2048
    window = min(1023, cached)
    mask = rows * (window + rows) if window + rows > 1024 else 0
    return max(fixed + per_row * rows, flat_share) + 2 * mask


class TestGemmaGeometry:
    def test_its_text_config_is_read_through_its_runtime(self):
        """The runtime has no ``model``: its adapter holds the text model.
        The widest layer on the 31B is the MLP (the full-attention layers:
        512-wide heads, 4 KV heads, one array for K and V)."""

        runtime = _gemma_runtime()
        args = srv._runtime_text_args(runtime)
        assert args is not None and args.hidden_size == 5376
        assert srv._forward_row_bytes(args) == GEMMA_ROW == 139_776
        assert not srv._runtime_has_qsa_indexer(runtime)

    def test_the_full_attention_layers_have_their_own_width(self):
        # With a narrow MLP the full-attention layers are the widest:
        # 2 x (5,376 + 2 x 16,384 + 4 x 512 + 16,384 + 5,376).
        args = _gemma_text_args()
        args.intermediate_size = 1024
        assert srv._forward_row_bytes(args) == 2 * (5376 + 2 * 16384 + 4 * 512 + 16384 + 5376)

    def test_what_its_caches_keep(self):
        args = _gemma_text_args()
        assert GEMMA_PLANNED_KV == 983_040
        assert gemma4.gemma4_resident_kv_bytes_per_token(args) == GEMMA_RESIDENT == 81_920
        assert gemma4.gemma4_window_cache_bytes(args) == GEMMA_WINDOWS == 838_860_800
        assert gemma4.gemma4_prefill_mask_bytes(args, 65_536, 0) == 65_536 * 65_536
        # Inside the window no array mask is built.
        assert gemma4.gemma4_prefill_mask_bytes(args, 1024, 0) == 0
        # A warm suffix sees at most the window's worth of cached keys.
        assert gemma4.gemma4_prefill_mask_bytes(args, 600, 30_000) == 600 * (1023 + 600)


class TestGemmaAdmission:
    def test_a_cold_prompt_is_one_forward_over_every_uncached_token(self, monkeypatch):
        """16,384 cold tokens: priced before as a 2,048-row chunk with a
        3 GiB flat scratch; the backend runs one 16,384-row forward, with
        its sliding mask, and never repages."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(16_384)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["prefill_chunk_tokens"] is None
        assert growth["scratch_rows"] == 16_384
        assert growth["scratch_source"] == "geometry+mask"
        assert growth["scratch_bytes"] == _gemma_scratch(16_384, 0)
        assert growth["layout"] == "contiguous_dense_decode"
        assert growth["repage_copy_bytes"] == 0
        assert growth["chunk_bytes"] == 16_384 * GEMMA_PLANNED_KV + _gemma_scratch(16_384, 0)
        # The pre-decode clone copies what decode keeps, not every layer.
        assert growth["publish_copy_bytes"] == 16_384 * GEMMA_RESIDENT + GEMMA_WINDOWS

    def test_a_warm_turn_is_charged_what_the_restored_cache_keeps(self):
        """A 30,000-token conversation restored by clone for a 600-token
        turn: the restore copies the full-attention KV and the windows
        (3.3 GB), not 30,000 rows of every layer (29.5 GB)."""

        geometry = srv._admission_geometry(_gemma_state(_manager()))
        assert geometry.resident_width == GEMMA_RESIDENT
        growth = srv._admission_growth(
            geometry,
            prompt_tokens=30_600,
            reused_tokens=30_000,
            restore_copies_prefix=True,
            layout="contiguous_dense_decode",
            source_layout=None,
            output_tokens=512,
            publish=True,
            scratch_bytes=_gemma_scratch(600, 30_000),
        )
        assert growth["restore_copy_bytes"] == 30_000 * GEMMA_RESIDENT + GEMMA_WINDOWS
        assert growth["live_prefill_bytes"] == (
            30_000 * GEMMA_RESIDENT + GEMMA_WINDOWS + 600 * GEMMA_PLANNED_KV
        )
        assert growth["publish_copy_bytes"] == 30_600 * GEMMA_RESIDENT + GEMMA_WINDOWS
        assert growth["growth_bytes"] < 10 * GIB

    def test_skipping_store_on_prefill_saves_nothing_here(self, monkeypatch):
        """The clone does not follow store-on-prefill: turning it off must
        not read as room."""

        monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL", "0")
        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(4_096)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        assert pricing["growth"]["publish_copy_bytes"] == 4_096 * GEMMA_RESIDENT + GEMMA_WINDOWS

    def test_a_prompt_the_backend_cannot_hold_is_refused_before_it_starts(
        self, monkeypatch
    ):
        """65,536 cold tokens: every layer holds all of them when the forward
        ends (64 GB at 983,040 B a token) plus the forward's scratch and its
        4.3 GB mask. Past a 96 GiB limit on any Mac: refused up front
        instead of running into the wall."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        receipt = srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(65_536)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
        )
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["prefill_chunk_tokens"] is None
        assert receipt["growth"]["scratch_rows"] == 65_536


class TestGemmaAbortSite:
    def _runtime(self):
        return SimpleNamespace(
            model_path=Path("models/gemma4"),
            mtp_enabled=True,
            make_cache=lambda: [SimpleNamespace(offset=0)],
        )

    def _counting_check(self, trip_on: int):
        calls = []

        def check() -> bool:
            calls.append(1)
            return len(calls) >= trip_on

        return check, calls

    def _fake_forward(self, monkeypatch):
        forwards = []

        class _Rows:
            def __getitem__(self, _key):
                return self

        def fake_prefill(_runtime, prompt_ids, *, cache, phase):
            forwards.append(len(prompt_ids))
            return (
                SimpleNamespace(
                    logits=_Rows(),
                    hidden=_Rows(),
                    shared_kv_states={},
                    cache_offset=len(prompt_ids),
                ),
                0.0,
            )

        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", fake_prefill)
        return forwards

    def test_it_stops_before_the_forward_allocates(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=2)
        with pytest.raises(generation.PostcommitAbort):
            gemma4._restore_or_prefill_gemma4_prompt(
                self._runtime(),
                list(range(64)),
                require_shared_kv=True,
                abort_check=check,
            )
        assert forwards == []
        assert len(calls) == 2

    def test_it_runs_when_nothing_trips(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=99)
        state = gemma4._restore_or_prefill_gemma4_prompt(
            self._runtime(),
            list(range(64)),
            require_shared_kv=True,
            abort_check=check,
        )
        assert forwards == [64]
        assert len(calls) == 3
        assert state.suffix_tokens == 64

    def test_it_stops_after_the_forward_before_decode(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=3)
        with pytest.raises(generation.PostcommitAbort):
            gemma4._restore_or_prefill_gemma4_prompt(
                self._runtime(),
                list(range(64)),
                require_shared_kv=True,
                abort_check=check,
            )
        assert forwards == [64]
        assert len(calls) == 3

    def test_both_gemma_branches_hand_the_check_on(self, monkeypatch):
        """generate_ar and generate_mtpk route a Gemma runtime to the
        backend's own loops; both used to drop ``abort_check``."""

        seen: dict[str, object] = {}

        def fake_ar(rt, prompt_ids, **kwargs):
            seen["ar"] = kwargs.get("abort_check")
            return "ar"

        def fake_assistant(rt, prompt_ids, **kwargs):
            seen["mtp"] = kwargs.get("abort_check")
            return "mtp"

        monkeypatch.setattr(gemma4, "generate_gemma4_ar", fake_ar)
        monkeypatch.setattr(gemma4, "generate_gemma4_assistant", fake_assistant)
        runtime = SimpleNamespace(
            backend_id="gemma4_assistant",
            config=SimpleNamespace(draft_block_size=4),
        )

        def check() -> bool:
            return False

        sampler = generation.SamplerConfig(temperature=0.0)
        assert (
            generation.generate_ar(
                runtime, [1, 2, 3], max_tokens=4, sampler=sampler, abort_check=check
            )
            == "ar"
        )
        assert (
            generation.generate_mtpk(
                runtime,
                [1, 2, 3],
                max_tokens=4,
                sampler=sampler,
                speculative_depth=3,
                abort_check=check,
            )
            == "mtp"
        )
        assert seen == {"ar": check, "mtp": check}


class TestTheGenericLoop:
    def test_it_forwards_the_whole_prompt_without_sustained_prefill(self, monkeypatch):
        monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
        assert generation.prefill_forward_widths(SimpleNamespace(), 30_000, 4096) == [None]

    def test_it_runs_the_request_chunk_then_the_profiles(self):
        runtime = SimpleNamespace()
        assert generation.prefill_forward_widths(runtime, 30_000, 4096) == [4096, 2048]
        assert generation.prefill_forward_widths(runtime, 30_000, None) == [2048]
        assert generation.prefill_forward_widths(runtime, 30_000, 1024) == [1024]
        assert generation.prefill_cache_layout(runtime, 30_000) == "contiguous_dense_decode"
        assert generation.prefill_cache_layout(runtime, 200_000) == "contiguous_then_repage"

    def test_a_backend_answers_for_itself(self):
        runtime = _gemma_runtime()
        assert generation.prefill_forward_widths(runtime, 30_000, 4096) == [None]
        assert generation.prefill_cache_layout(runtime, 200_000) == "contiguous_dense_decode"


class TestThe4B:
    """Qwen3.5-4B runs the generic loop: the profile's 2,048-row chunk,
    its own geometry (the gated-delta layers are its widest), the planner's
    KV width, no backend terms."""

    def _runtime(self):
        from mlx_lm.models.qwen3_5 import TextModelArgs

        return SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(args=TextModelArgs.from_dict(Q4B_TEXT))
            ),
            mtp_enabled=True,
            model_path=Path("models/qwen3.5-4b"),
        )

    def test_its_row_is_read_from_its_own_config(self):
        args = srv._runtime_text_args(self._runtime())
        # Gated delta: 2 x (2,560 + 4,096 + 8,192 + 4,096 + 4,096 + 2,560)
        # + 4 x (4,096 + 8,192); wider than its MLP (60,416) and attention.
        assert srv._forward_row_bytes(args) == 100_352

    def test_it_is_priced_at_the_profile_chunk(self, monkeypatch):
        kv = dense_kv_bytes_per_token_from_config({"text_config": Q4B_TEXT})
        assert kv == 8 * 2 * 4 * 256 * 2
        plan = SimpleNamespace(
            available=True,
            kv_bytes_per_token=kv,
            kv_bytes_per_token_effective=kv,
            aux_bytes_per_token=0,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=int(2.6 * GIB),
        )
        manager = _manager()
        _roomy(monkeypatch, manager)
        state = _state(manager, plan=plan, runtime=self._runtime(), limit_gib=12, total_gib=16)
        pricing: dict = {}
        srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(20_000)),
            session_bank=manager.bank,
            session_id="small",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        per_row = srv._ADMISSION_LIVE_LAYERS * 100_352
        fixed = max(srv._ADMISSION_FIXED_FLOOR_BYTES, 3 * GIB - per_row * 2048)
        assert growth["prefill_chunk_tokens"] == 2048
        assert growth["scratch_rows"] == 2048
        assert growth["scratch_source"] == "geometry"
        assert growth["scratch_bytes"] == max(fixed + per_row * 2048, 3 * GIB)
        assert growth["publish_copy_bytes"] == 20_000 * kv
