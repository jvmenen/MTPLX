"""Prompt-lookup (n-gram) speculative drafting.

A model-free draft source: when the last few committed tokens also occurred
earlier in the prompt or in the generated text, the tokens that followed that
earlier occurrence are proposed as a draft and verified in one target forward
pass. Output that copies from its context (extraction, code edits, quoting)
accepts long drafts; free writing finds few matches and falls back to the
normal one-token step.

First iteration: greedy decoding only, autoregressive models (no MTP head).
The drafter is independent of the model and of the verify path, so a later
change can ask it for a draft when a long match exists and use the MTP head
otherwise.

Switches (all read once per request):
  MTPLX_PROMPT_LOOKUP=1              enable
  MTPLX_PROMPT_LOOKUP_MAX_DRAFT=8    longest draft per verify
  MTPLX_PROMPT_LOOKUP_MIN_MATCH=3    shortest n-gram that may trigger a draft
  MTPLX_PROMPT_LOOKUP_MAX_NGRAM=4    longest n-gram tried first
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_MIN_ADAPTIVE_DRAFT = 2


@dataclass(frozen=True)
class PromptLookupConfig:
    enabled: bool = False
    max_draft: int = 8
    min_match: int = 3
    max_ngram: int = 4


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


def prompt_lookup_config_from_env() -> PromptLookupConfig:
    enabled = str(os.environ.get("MTPLX_PROMPT_LOOKUP", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    min_match = max(1, _env_int("MTPLX_PROMPT_LOOKUP_MIN_MATCH", 3))
    return PromptLookupConfig(
        enabled=enabled,
        max_draft=max(1, _env_int("MTPLX_PROMPT_LOOKUP_MAX_DRAFT", 8)),
        min_match=min_match,
        max_ngram=max(min_match, _env_int("MTPLX_PROMPT_LOOKUP_MAX_NGRAM", 4)),
    )


class NgramIndex:
    """Incremental n-gram index over a growing token sequence.

    For each n in [min_n, max_n] it maps the n-gram to the position of the
    token that followed its most recent earlier occurrence. The n-gram that
    ends the sequence is only registered once a next token arrives, so a
    lookup of the current suffix never finds itself.
    """

    def __init__(self, min_n: int, max_n: int) -> None:
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.sequence: list[int] = []
        self._tables: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in range(self.min_n, self.max_n + 1)
        }

    def append(self, token: int) -> None:
        seq = self.sequence
        seq.append(int(token))
        follower = len(seq) - 1
        for n, table in self._tables.items():
            if follower >= n:
                table[tuple(seq[follower - n : follower])] = follower

    def extend(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            self.append(token)

    def lookup(self, max_draft: int) -> list[int]:
        """Tokens that followed the latest earlier match of the longest suffix."""
        seq = self.sequence
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(seq) < n:
                continue
            start = self._tables[n].get(tuple(seq[-n:]))
            if start is not None:
                return seq[start : start + max_draft]
        return []


class PromptLookupDrafter:
    """N-gram index plus an adaptive draft length."""

    def __init__(self, config: PromptLookupConfig, prompt_ids: Sequence[int]) -> None:
        self.config = config
        self.index = NgramIndex(config.min_match, config.max_ngram)
        self.index.extend(prompt_ids)
        self.draft_len = config.max_draft
        self.verify_rounds = 0
        self.drafted = 0
        self.accepted = 0

    def append(self, token: int) -> None:
        self.index.append(token)

    def propose(self, cap: int) -> list[int]:
        limit = min(self.draft_len, int(cap))
        if limit <= 0:
            return []
        return self.index.lookup(limit)

    def record(self, proposed: int, accepted: int) -> None:
        """Adapt the draft length: grow after a full accept, shrink after a miss."""
        self.verify_rounds += 1
        self.drafted += proposed
        self.accepted += accepted
        if accepted >= proposed:
            self.draft_len = min(self.config.max_draft, self.draft_len * 2)
        else:
            self.draft_len = max(_MIN_ADAPTIVE_DRAFT, accepted + 1)

    def summary(self) -> dict[str, Any]:
        rounds = self.verify_rounds
        return {
            "verify_rounds": rounds,
            "drafted_tokens": self.drafted,
            "accepted_tokens": self.accepted,
            "acceptance_rate": self.accepted / self.drafted if self.drafted else 0.0,
            "mean_accepted_length": self.accepted / rounds if rounds else 0.0,
            "draft_len": self.draft_len,
        }


def accepted_prefix_length(draft: Sequence[int], target_ids: Sequence[int]) -> int:
    """Greedy acceptance: draft[i] must equal the target's choice at row i.

    `target_ids[i]` is the target's argmax after the committed token plus
    draft[:i], so `len(target_ids) == len(draft) + 1`.
    """
    accepted = 0
    for proposed, chosen in zip(draft, target_ids):
        if int(proposed) != int(chosen):
            break
        accepted += 1
    return accepted


def truncate_at_stop(
    draft: Sequence[int], accepted: int, stop_token_ids: set[int]
) -> tuple[int, bool]:
    """Cut the accepted prefix after its first stop token, if any."""
    for i in range(accepted):
        if int(draft[i]) in stop_token_ids:
            return i + 1, True
    return accepted, False


def greedy_pld_eligible(
    config: PromptLookupConfig,
    *,
    temperature: float,
    penalized: bool,
    guarded: bool,
    sync_eval: bool,
    cache_trimmable: bool,
) -> bool:
    """PLD runs only where exact greedy acceptance is valid."""
    return (
        config.enabled
        and temperature <= 0
        and not penalized
        and not guarded
        and sync_eval
        and cache_trimmable
    )


def verify_draft(
    forward: Callable[[Any], Any],
    trim_cache: Callable[[int], bool],
    cache_offset: int,
    token: int,
    draft: Sequence[int],
    stop_token_ids: set[int],
) -> tuple[int, bool, Any]:
    """One target pass over `[token] + draft`; commit the accepted prefix.

    `forward(ids)` runs the target on a (1, n) int array and returns logits of
    shape (1, n, vocab). `trim_cache(offset)` rolls the KV cache back to that
    length and reports success. Returns `(accepted, stopped, next_logits)`
    where `next_logits` has shape (1, vocab) for the position after the last
    committed token. The cache ends holding every committed token except the
    last one, as after a normal AR step.
    """
    import mlx.core as mx

    ids = mx.array([[int(token), *[int(d) for d in draft]]])
    logits = forward(ids)
    chosen = mx.argmax(logits[0], axis=-1)
    mx.eval(chosen)
    target_ids = chosen.tolist()
    accepted = accepted_prefix_length(draft, target_ids)
    accepted, stopped = truncate_at_stop(draft, accepted, stop_token_ids)
    # Cache holds token + draft; keep token + the accepted drafts. A stop
    # token is the last committed token, so it stays out of the cache.
    keep = cache_offset + 1 + accepted - (1 if stopped else 0)
    if not trim_cache(keep):
        raise RuntimeError("prompt lookup: KV cache rollback failed")
    return accepted, stopped, logits[:, accepted, :]
