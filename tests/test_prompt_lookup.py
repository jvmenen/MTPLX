import mlx.core as mx

from mtplx.prompt_lookup import (
    NgramIndex,
    PromptLookupConfig,
    PromptLookupDrafter,
    accepted_prefix_length,
    greedy_pld_eligible,
    prompt_lookup_config_from_env,
    truncate_at_stop,
    verify_draft,
)


def test_index_proposes_tokens_after_latest_earlier_match():
    index = NgramIndex(3, 4)
    index.extend([1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3])
    assert index.lookup(8) == [4, 5, 6, 9, 9, 1, 2, 3]
    assert index.lookup(2) == [4, 5]


def test_index_never_matches_its_own_suffix():
    index = NgramIndex(3, 4)
    index.extend([7, 8, 9])
    assert index.lookup(4) == []


def test_index_prefers_longest_ngram_then_most_recent():
    index = NgramIndex(2, 3)
    index.extend([5, 1, 2, 3, 10, 6, 1, 2, 3, 11, 0, 1, 2, 3])
    assert index.lookup(1) == [11]
    index2 = NgramIndex(2, 3)
    index2.extend([4, 1, 2, 50, 9, 1, 2, 60, 8, 1, 2])
    assert index2.lookup(1) == [60]


def test_index_is_incremental_with_generated_tokens():
    drafter = PromptLookupDrafter(PromptLookupConfig(enabled=True), [1, 2, 3, 4])
    assert drafter.propose(8) == []
    for token in (9, 1, 2, 3):
        drafter.append(token)
    assert drafter.propose(8) == [4, 9, 1, 2, 3]
    assert drafter.propose(0) == []


def test_adaptive_draft_length():
    drafter = PromptLookupDrafter(PromptLookupConfig(enabled=True, max_draft=8), [])
    drafter.record(8, 3)
    assert drafter.draft_len == 4
    drafter.record(4, 0)
    assert drafter.draft_len == 2
    drafter.record(2, 2)
    assert drafter.draft_len == 4
    drafter.record(4, 4)
    assert drafter.draft_len == 8
    drafter.record(8, 8)
    assert drafter.draft_len == 8
    summary = drafter.summary()
    assert summary["verify_rounds"] == 5
    assert summary["drafted_tokens"] == 26
    assert summary["accepted_tokens"] == 17


def test_accepted_prefix_and_stop_truncation():
    assert accepted_prefix_length([4, 5, 6], [4, 5, 7, 1]) == 2
    assert accepted_prefix_length([4, 5, 6], [4, 5, 6, 1]) == 3
    assert accepted_prefix_length([4], [9, 1]) == 0
    assert truncate_at_stop([4, 68, 6], 3, {68}) == (2, True)
    assert truncate_at_stop([4, 5, 6], 3, {68}) == (3, False)
    assert truncate_at_stop([4, 68, 6], 1, {68}) == (1, False)


def _logits_for(choices, vocab=16):
    rows = mx.zeros((1, len(choices), vocab))
    for i, token in enumerate(choices):
        rows[0, i, token] = 1.0
    return rows


def test_verify_draft_commits_prefix_and_rolls_back_cache():
    trimmed = []
    accepted, stopped, next_logits = verify_draft(
        lambda ids: _logits_for([4, 5, 9, 2]),
        lambda keep: trimmed.append(keep) or True,
        cache_offset=10,
        token=3,
        draft=[4, 5, 6],
        stop_token_ids={15},
    )
    assert (accepted, stopped) == (2, False)
    assert trimmed == [13]
    assert int(mx.argmax(next_logits[0])) == 9
    assert next_logits.shape == (1, 16)


def test_verify_draft_stop_token_stays_out_of_cache():
    trimmed = []
    accepted, stopped, _ = verify_draft(
        lambda ids: _logits_for([4, 15, 6, 2]),
        lambda keep: trimmed.append(keep) or True,
        cache_offset=10,
        token=3,
        draft=[4, 15, 6],
        stop_token_ids={15},
    )
    assert (accepted, stopped) == (2, True)
    assert trimmed == [12]


def test_verify_draft_full_accept_gives_bonus_logits():
    trimmed = []
    accepted, _, next_logits = verify_draft(
        lambda ids: _logits_for([4, 5, 7]),
        lambda keep: trimmed.append(keep) or True,
        cache_offset=0,
        token=3,
        draft=[4, 5],
        stop_token_ids=set(),
    )
    assert accepted == 2
    assert int(mx.argmax(next_logits[0])) == 7


def test_config_from_env(monkeypatch):
    monkeypatch.delenv("MTPLX_PROMPT_LOOKUP", raising=False)
    assert not prompt_lookup_config_from_env().enabled
    monkeypatch.setenv("MTPLX_PROMPT_LOOKUP", "1")
    monkeypatch.setenv("MTPLX_PROMPT_LOOKUP_MAX_DRAFT", "6")
    monkeypatch.setenv("MTPLX_PROMPT_LOOKUP_MIN_MATCH", "2")
    config = prompt_lookup_config_from_env()
    assert (config.enabled, config.max_draft, config.min_match) == (True, 6, 2)
    assert config.max_ngram >= config.min_match


def test_eligibility_requires_exact_greedy_conditions():
    on = PromptLookupConfig(enabled=True)
    kwargs = {
        "temperature": 0.0,
        "penalized": False,
        "guarded": False,
        "sync_eval": True,
        "cache_trimmable": True,
    }
    assert greedy_pld_eligible(on, **kwargs)
    assert not greedy_pld_eligible(PromptLookupConfig(), **kwargs)
    for key, value in (
        ("temperature", 0.6), ("penalized", True), ("guarded", True),
        ("sync_eval", False), ("cache_trimmable", False),
    ):
        assert not greedy_pld_eligible(on, **{**kwargs, key: value})
