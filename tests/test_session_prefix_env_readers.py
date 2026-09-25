"""One reader per session-prefix setting, shared by every caller.

The near-prefix gap and minimum match, the prefix block size and the
store-on-prefill minimum were each parsed in two or three modules with their
own literal defaults. ``mtplx.runtime_options`` now owns each parse; the
decode loop, the engine session and the server read through it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import engine_session, generation, runtime_options
from mtplx.server import openai

READERS = [
    (runtime_options.near_prefix_max_token_gap, runtime_options.NEAR_PREFIX_MAX_TOKEN_GAP_ENV, 8, 0),
    (runtime_options.near_prefix_min_match_tokens, runtime_options.NEAR_PREFIX_MIN_MATCH_ENV, 64, 1),
    (runtime_options.prefix_block_size, runtime_options.PREFIX_BLOCK_SIZE_ENV, 256, 1),
    (
        runtime_options.store_on_prefill_min_suffix,
        runtime_options.STORE_ON_PREFILL_MIN_SUFFIX_ENV,
        1024,
        1,
    ),
    (
        runtime_options.block_prefix_min_match_tokens,
        runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV,
        512,
        1,
    ),
]


@pytest.fixture
def clean_env(monkeypatch):
    for _reader, name, _default, _floor in READERS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(("reader", "name", "default", "floor"), READERS)
def test_reader_defaults_parses_and_floors(clean_env, reader, name, default, floor):
    assert reader() == default
    for raw in ("", "   ", "junk", "8.5"):
        clean_env.setenv(name, raw)
        assert reader() == default
    clean_env.setenv(name, " 300 ")
    assert reader() == 300
    clean_env.setenv(name, "-5")
    assert reader() == floor


def test_engine_session_and_generation_read_the_same_values(clean_env):
    clean_env.setenv(runtime_options.NEAR_PREFIX_MAX_TOKEN_GAP_ENV, "3")
    clean_env.setenv(runtime_options.NEAR_PREFIX_MIN_MATCH_ENV, "40")
    clean_env.setenv(runtime_options.PREFIX_BLOCK_SIZE_ENV, "128")
    clean_env.setenv(runtime_options.STORE_ON_PREFILL_MIN_SUFFIX_ENV, "700")

    assert engine_session._near_prefix_max_token_gap() == 3
    assert engine_session._near_prefix_min_match_tokens() == 40
    assert engine_session._prefix_block_size() == 128
    assert generation._store_on_prefill_min_suffix() == 700


def test_block_restorable_estimate_defaults_unchanged(clean_env):
    assert openai._block_restorable_prefix_tokens(511) == 0
    assert openai._block_restorable_prefix_tokens(767) == 512
    assert openai._block_restorable_prefix_tokens(1030) == 1024


def test_block_restorable_estimate_follows_the_restore_threshold(clean_env):
    # The restore itself serves a 300-token shared prefix at its 256-token
    # block edge once the threshold is lowered; the admission estimate used
    # to read 0 here.
    clean_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, "128")
    assert openai._block_restorable_prefix_tokens(300) == 256
    assert openai._block_restorable_prefix_tokens(200) == 0

    clean_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, "2048")
    assert openai._block_restorable_prefix_tokens(1030) == 0


def test_block_restorable_estimate_follows_the_block_size(clean_env):
    clean_env.setenv(runtime_options.PREFIX_BLOCK_SIZE_ENV, "128")
    assert openai._block_restorable_prefix_tokens(700) == 640


def _cold_tier_state(min_prefix_tokens):
    return SimpleNamespace(
        session_bank_cold_tier=SimpleNamespace(enabled=True, min_prefix_tokens=min_prefix_tokens)
    )


@pytest.mark.parametrize("tier_min", [None, 0, 128, 512])
def test_prompt_prefix_commit_never_below_512(tier_min):
    state = _cold_tier_state(tier_min)
    commit = openai._commit_prompt_prefix_for_request
    assert openai._PROMPT_PREFIX_COMMIT_MIN_TOKENS == 512
    assert commit(state, prompt_ids=list(range(512)), tools_active=False)
    assert not commit(state, prompt_ids=list(range(511)), tools_active=False)


def test_prompt_prefix_commit_follows_a_higher_tier_minimum():
    state = _cold_tier_state(2048)
    commit = openai._commit_prompt_prefix_for_request
    assert not commit(state, prompt_ids=list(range(2047)), tools_active=False)
    assert commit(state, prompt_ids=list(range(2048)), tools_active=False)
