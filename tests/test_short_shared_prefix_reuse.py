"""Short shared prompt prefixes (a fixed preamble, a varying tail).

A classifier sends ~220 fixed tokens followed by a varying message. Three
things kept that preamble from ever being restored on a hybrid model:

* the block-prefix threshold was floored at the block size (256), so
  ``MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS=128`` still meant 256;
* the prefill records recurrent snapshots on a grid near the prompt end,
  never where the preamble ends;
* prompts shorter than the store-on-prefill minimum were banked without
  their snapshots.

``--ram-session-prefix-min-match-tokens N`` addresses all three; unset,
every default is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import runtime_options
from mtplx.config import _apply_runtime_defaults, load_user_config
from mtplx.generation import _prefill_spans_with_tail_grid, _shared_prefix_edge
from mtplx.session_bank import SessionBank

PREAMBLE = list(range(1, 221))
RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _bank_with(*token_lists: list[int]) -> SessionBank:
    bank = SessionBank(max_entries=16, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    for index, token_ids in enumerate(token_lists):
        bank.put(
            runtime=RUNTIME,
            token_ids=token_ids,
            cache=[],
            logits=None,
            hidden=None,
            session_id=f"s{index}",
            nbytes_override=16,
        )
    return bank


def _classifier_prompt(*tail: int) -> list[int]:
    return PREAMBLE + list(tail)


def _banked_request(first_tail_token: int) -> list[int]:
    """A stored classifier request: preamble, a 30-token message, one answer
    token. Its tail is longer than the near-prefix tiny-gap window (8), so
    only the block-prefix lane can serve its preamble."""

    return _classifier_prompt(*range(first_tail_token, first_tail_token + 31))


@pytest.fixture
def clean_prefix_env(monkeypatch):
    for name in (
        runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV,
        runtime_options.SHARED_PREFIX_EDGE_ENV,
        runtime_options.STORE_ON_PREFILL_MIN_SUFFIX_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --- the threshold ----------------------------------------------------------


def test_min_match_threshold_defaults_to_512(clean_prefix_env):
    assert runtime_options.block_prefix_min_match_tokens() == 512


@pytest.mark.parametrize(("raw", "expected"), [("128", 128), ("0", 1), ("junk", 512), ("", 512)])
def test_min_match_threshold_parse(clean_prefix_env, raw, expected):
    clean_prefix_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, raw)
    assert runtime_options.block_prefix_min_match_tokens() == expected


def test_block_candidate_below_block_size_when_threshold_allows_it():
    bank = _bank_with(_banked_request(9001))
    prompt = _classifier_prompt(7001, 7002)

    candidates = bank.near_prefix_candidates(
        prompt,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=128,
        allow_block_prefix=True,
    )

    assert [matched for _entry, matched in candidates] == [220]


def test_block_candidate_below_default_threshold_is_still_rejected():
    bank = _bank_with(_banked_request(9001))

    candidates = bank.near_prefix_candidates(
        _classifier_prompt(7001, 7002),
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    assert candidates == []


# --- where the preamble ends ------------------------------------------------


def test_dominant_shared_prefix_ignores_tails_that_start_alike():
    bank = _bank_with(
        _classifier_prompt(9001),
        _classifier_prompt(9002),
        _classifier_prompt(7001, 5),  # shares one tail token with the prompt
        [42] * 300,  # unrelated traffic
    )

    assert bank.dominant_shared_prefix_tokens(
        _classifier_prompt(7001, 6), min_tokens=128
    ) == 220


def test_dominant_shared_prefix_prefers_the_shorter_length_on_a_tie():
    bank = _bank_with(_classifier_prompt(9001), _classifier_prompt(7001, 5))

    assert bank.dominant_shared_prefix_tokens(
        _classifier_prompt(7001, 6), min_tokens=128
    ) == 220


def test_dominant_shared_prefix_is_zero_below_the_threshold():
    bank = _bank_with(_classifier_prompt(9001))

    assert bank.dominant_shared_prefix_tokens(PREAMBLE[:100] + [5], min_tokens=128) == 0
    assert bank.dominant_shared_prefix_tokens(_classifier_prompt(1), min_tokens=512) == 0


def test_shared_prefix_edge_is_off_by_default(clean_prefix_env):
    bank = _bank_with(_classifier_prompt(9001))

    assert _shared_prefix_edge(bank, _classifier_prompt(7001)) is None


def test_shared_prefix_edge_names_the_preamble_end(clean_prefix_env):
    clean_prefix_env.setenv(runtime_options.SHARED_PREFIX_EDGE_ENV, "1")
    clean_prefix_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, "128")
    bank = _bank_with(_classifier_prompt(9001), _classifier_prompt(9002))

    assert _shared_prefix_edge(bank, _classifier_prompt(7001, 7002)) == 220
    # A prompt the bank already holds whole has no edge to add.
    assert _shared_prefix_edge(bank, PREAMBLE) is None


def test_the_edge_becomes_a_prefill_span_end():
    """Span ends are where the prefill records recurrent snapshots."""

    without = _prefill_spans_with_tail_grid(420, tail_interval=256)
    with_edge = _prefill_spans_with_tail_grid(
        420, tail_interval=256, mandatory_edges=(220,)
    )

    assert 220 not in [end for _start, end in without]
    assert 220 in [end for _start, end in with_edge]


class _LaneProbe(Exception):
    pass


def _stable_prefix_len_seen_by_restore(monkeypatch, bank, prompt, **kwargs):
    from mtplx import generation

    seen: dict[str, object] = {}

    def recorder(rt, prompt_ids, **lane_kwargs):
        seen.update(lane_kwargs)
        raise _LaneProbe()

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", recorder)
    runtime = SimpleNamespace(
        model_path=Path("models/example"), mtp_enabled=False, contract=SimpleNamespace()
    )
    with pytest.raises(_LaneProbe):
        generation.restore_or_prefill_prompt_state(
            runtime,
            prompt,
            mtp_history_policy="cycle",
            session_bank=bank,
            session_id="s",
            **kwargs,
        )
    return seen.get("stable_prefix_len")


def test_restore_passes_the_shared_prefix_edge_to_the_prefill(clean_prefix_env):
    clean_prefix_env.setenv(runtime_options.SHARED_PREFIX_EDGE_ENV, "1")
    clean_prefix_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, "128")
    bank = _bank_with(_classifier_prompt(9001, 9002))

    assert _stable_prefix_len_seen_by_restore(
        clean_prefix_env, bank, _classifier_prompt(7001, 7002)
    ) == 220


def test_restore_keeps_the_callers_stable_edge(clean_prefix_env):
    clean_prefix_env.setenv(runtime_options.SHARED_PREFIX_EDGE_ENV, "1")
    clean_prefix_env.setenv(runtime_options.BLOCK_PREFIX_MIN_MATCH_ENV, "128")
    bank = _bank_with(_classifier_prompt(9001, 9002))

    assert _stable_prefix_len_seen_by_restore(
        clean_prefix_env, bank, _classifier_prompt(7001, 7002), stable_prefix_len=150
    ) == 150


def test_restore_adds_no_edge_by_default(clean_prefix_env):
    bank = _bank_with(_classifier_prompt(9001, 9002))

    assert _stable_prefix_len_seen_by_restore(
        clean_prefix_env, bank, _classifier_prompt(7001, 7002)
    ) is None


# --- the one setting --------------------------------------------------------


def test_setting_writes_threshold_edge_and_store_minimum():
    env: dict[str, str] = {}

    written = runtime_options.apply_session_prefix_min_match_env(128, env)

    assert env == written == {
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS": "128",
        "MTPLX_SESSION_SHARED_PREFIX_EDGE": "1",
        "MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX": "128",
    }


def test_setting_keeps_explicit_implied_env_but_owns_the_threshold():
    env = {
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS": "512",
        "MTPLX_SESSION_SHARED_PREFIX_EDGE": "0",
        "MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX": "300",
    }

    runtime_options.apply_session_prefix_min_match_env(128, env)

    assert env == {
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS": "128",
        "MTPLX_SESSION_SHARED_PREFIX_EDGE": "0",
        "MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX": "300",
    }


def test_setting_never_raises_the_store_minimum():
    env: dict[str, str] = {}

    runtime_options.apply_session_prefix_min_match_env(4096, env)

    assert env["MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX"] == "1024"


def test_unset_setting_changes_nothing():
    env: dict[str, str] = {}

    assert runtime_options.apply_session_prefix_min_match_env(None, env) == {}
    assert env == {}


def test_config_key_reaches_the_launch_args(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("ram_session_prefix_min_match_tokens = 128\n", encoding="utf-8")
    config = load_user_config(path)
    args = SimpleNamespace(_cli_flags=set())

    _apply_runtime_defaults(args, config)

    assert config.ram_session_prefix_min_match_tokens == 128
    assert args.ram_session_prefix_min_match_tokens == 128


def test_cli_flag_wins_over_the_config_key(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("ram_session_prefix_min_match_tokens = 128\n", encoding="utf-8")
    args = SimpleNamespace(
        _cli_flags={"ram-session-prefix-min-match-tokens"},
        ram_session_prefix_min_match_tokens=64,
    )

    _apply_runtime_defaults(args, load_user_config(path))

    assert args.ram_session_prefix_min_match_tokens == 64


def test_server_parser_accepts_the_setting():
    from mtplx.server import openai as oa

    parser_args = oa.parse_args(
        ["--model", "models/example", "--ram-session-prefix-min-match-tokens", "128"]
    )

    assert parser_args.ram_session_prefix_min_match_tokens == 128


def test_launcher_forwards_the_setting():
    from mtplx.commands.public import _batching_command_suffix

    args = SimpleNamespace(ram_session_prefix_min_match_tokens=128)

    assert "--ram-session-prefix-min-match-tokens 128" in _batching_command_suffix(args)
