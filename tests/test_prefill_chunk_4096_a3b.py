"""Prefill chunks of 4,096 rows stay exact on the A3B expert layout.

``MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096`` and ``..._REPAGE=4096`` are pure
settings; these tests pin why 4,096 is the widest safe value on MLX 0.32.2:

* ``gather_qmm`` on M5 (NAX) returns wrong rows once a forward holds more
  than 32,768 token-expert rows that are not a multiple of 64 (MLX #3856,
  fixed after 0.32.2). A3B routes top-8, so 4,096 tokens are exactly 32,768
  rows: on the invariant lane one forward of 4,096 tokens equals two of
  2,048 bit for bit, and so does every narrower tail;
* the prefill planner never forwards more than the chunk, with or without
  the tail ladder and mandatory edges, and the chunk knobs resolve to 4,096
  for both layouts;
* the cold streaming loop with 4,096-row chunks never runs a trunk forward
  or a draft-history append wider than 4,096.
"""

from __future__ import annotations

from itertools import pairwise

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx_lm.models.switch_layers import SwitchGLU

from mtplx import batch_invariant_prefill as bip
from mtplx import generation
from mtplx.attention_context import attention_phase
from tests.a3b_tiny_synth import prompt, tiny_model_with_draft_head

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

CHUNK = 4096
A3B_EXPERTS, A3B_TOP_K = 256, 8


def _lane_switch_glu(hidden: int = 128, intermediate: int = 64):
    mx.random.seed(0)
    glu = SwitchGLU(hidden, intermediate, A3B_EXPERTS)
    glu.set_dtype(mx.bfloat16)
    nn.quantize(glu, group_size=64, bits=6, class_predicate=lambda _p, m: hasattr(m, "to_quantized"))
    glu.__class__ = bip.BatchInvariantSwitchGLU
    mx.eval(glu.parameters())
    return glu


def _experts(glu, x, routes):
    with attention_phase("prefill"):
        out = glu(x, routes)
    mx.eval(out)
    return np.array(out.view(mx.uint16))


@needs_metal
@pytest.mark.parametrize("tokens", [4096, 4095, 2049, 1000])
def test_one_forward_up_to_4096_tokens_equals_chunks_of_2048(tokens):
    glu = _lane_switch_glu()
    mx.random.seed(tokens)
    x = (mx.random.normal((1, tokens, 128)) * 0.5).astype(mx.bfloat16)
    routes = mx.argpartition(
        mx.random.normal((1, tokens, A3B_EXPERTS)), kth=-A3B_TOP_K, axis=-1
    )[..., -A3B_TOP_K:].astype(mx.uint32)
    whole = _experts(glu, x, routes)
    parts = np.concatenate(
        [
            _experts(glu, x[:, start : start + 2048], routes[:, start : start + 2048])
            for start in range(0, tokens, 2048)
        ],
        axis=1,
    )
    assert tokens * A3B_TOP_K <= 32768
    assert np.array_equal(whole, parts)


@pytest.mark.parametrize("layout", ["contiguous_dense_decode", "contiguous_then_repage"])
def test_the_chunk_knobs_resolve_to_4096_for_both_layouts(monkeypatch, layout):
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", str(CHUNK))
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", str(CHUNK))
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", layout)
    assert generation._prefill_chunk_size() == CHUNK


@pytest.mark.parametrize("tokens", [1, 511, 4095, 4096, 4097, 4159, 8192, 8250, 12345, 32768])
@pytest.mark.parametrize("capture, inforward", [(False, False), (True, False), (True, True)])
def test_the_planner_never_forwards_more_than_the_chunk(tokens, capture, inforward):
    edges = tuple(edge for edge in (tokens // 3, tokens - 5) if 0 < edge < tokens)
    spans, _interior = generation._prefill_boundary_plan(
        tokens,
        capture_boundaries=capture,
        inforward=inforward,
        tail_interval=256,
        mandatory_edges=edges,
        chunk_size=CHUNK,
    )
    assert spans[0][0] == 0 and spans[-1][1] == tokens
    assert all(end - start <= CHUNK for start, end in spans)
    assert all(left[1] == right[0] for left, right in pairwise(spans))


class _WidthRuntime:
    """The cold loop's runtime surface, recording every forward width."""

    mtp_enabled = True

    def __init__(self, model, tmp_path):
        self.model = model
        self.model_path = tmp_path
        self.diagnostic_counters: dict[str, int] = {}
        self.forwards: list[int] = []
        self.appends: list[int] = []

    def make_cache(self):
        return self.model.make_cache()

    def make_mtp_cache(self):
        return self.model.make_mtp_cache()

    @property
    def embed_tokens(self):
        return self.model.language_model.model.embed_tokens

    def forward_ar(self, tokens, *, cache, return_hidden=False, hidden_variant=None,
                   emit_logits=True, logits_keep=None, input_embeddings=None):
        self.forwards.append(int(tokens.shape[1]))
        return self.model(tokens, cache=cache, return_hidden=return_hidden,
                          hidden_variant=hidden_variant, emit_logits=emit_logits,
                          logits_keep=logits_keep, input_embeddings=input_embeddings)

    def update_mtp_cache(self, hidden_states, token_ids, *, mtp_cache, **kwargs):
        self.appends.append(int(token_ids.shape[1]))
        return self.model.mtp_update_cache(hidden_states, token_ids, mtp_cache=mtp_cache, **kwargs)


def test_the_cold_loop_never_runs_wider_than_4096(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", str(CHUNK))
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", str(CHUNK))
    rt = _WidthRuntime(tiny_model_with_draft_head(tmp_path), tmp_path)
    tokens = prompt(9000, seed=5)
    out = generation._prefill_committed_mtp_history_streaming(
        rt, tokens, gdn_boundary_sink=[]
    )
    mx.eval(out[1], out[2])
    assert max(rt.forwards) == CHUNK and max(rt.appends) <= CHUNK
    assert sum(rt.forwards) == len(tokens)
