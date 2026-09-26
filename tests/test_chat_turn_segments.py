"""Seam-less chat renders encode turn by turn, token-identical.

Scoped reasoning history renders plain chat without generation seams, so the
segment memo never applied there. Cutting before every ``<|im_start|>`` (an
atomic added token) must give exactly the single-call ids; these tests pin
that on the real Qwen3.6 tokenizer and chat template when it is cached
locally, plus the guards and the off switch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

import mtplx.server.openai as oa
from mtplx.chat_encode_cache import ChatSegmentEncodeMemo


@dataclass
class FakeAddedToken:
    content: str
    normalized: bool = False
    lstrip: bool = False
    rstrip: bool = False


class TurnTokenizer:
    """One id per character; counts encode calls."""

    def __init__(self, turn_open: FakeAddedToken | None):
        self.added = {} if turn_open is None else {1: turn_open}
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return [ord(char) for char in text]

    vocab_size = 1000

    @property
    def added_tokens_decoder(self):
        return self.added


class WrappedTokenizer:
    """Like mlx-lm's TokenizerWrapper: forwards attributes to the HF one."""

    def __init__(self, inner):
        self._tokenizer = inner

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


RENDER = (
    "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


@pytest.fixture
def memo(monkeypatch):
    fresh = ChatSegmentEncodeMemo(max_tokens=2_000_000)
    monkeypatch.setattr(oa, "GLOBAL_CHAT_SEGMENT_MEMO", fresh)
    monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO", raising=False)
    monkeypatch.delenv("MTPLX_CHAT_TURN_SEGMENTS", raising=False)
    return fresh


def test_boundaries_sit_before_every_turn_but_the_first():
    starts = [RENDER.find("<|im_start|>user"), RENDER.find("<|im_start|>assistant")]
    assert oa._chat_turn_boundaries(RENDER) == starts
    assert oa._chat_turn_boundaries("no turns here") == []


def test_turns_are_encoded_separately_and_memoized(memo):
    tok = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    obs: dict = {}
    ids = oa._encode_rendered_chat_turns(tok, RENDER, obs)
    assert ids == [ord(char) for char in RENDER]
    assert tok.encode_calls == 3
    assert obs["chat_segment_memo"]["misses"] == 3

    oa._encode_rendered_chat_turns(tok, RENDER + "<|im_start|>user\nmore", obs)
    assert obs["chat_segment_memo"] == {
        "hits": 3,
        "misses": 1,
        "reused_tokens": len(RENDER),
    }


@pytest.mark.parametrize(
    "turn_open",
    [
        None,
        FakeAddedToken("<|im_start|>", normalized=True),
        FakeAddedToken("<|im_start|>", lstrip=True),
        FakeAddedToken("<|im_start|>", rstrip=True),
        FakeAddedToken("<|im_end|>"),
    ],
)
def test_non_atomic_turn_marker_keeps_the_single_call(memo, turn_open):
    tok = TurnTokenizer(turn_open)
    obs: dict = {}
    oa._encode_rendered_chat_turns(tok, RENDER, obs)
    assert tok.encode_calls == 1
    assert "chat_segment_memo" not in obs


def test_atomic_check_reads_through_a_wrapper():
    inner = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    assert oa._chat_turn_open_is_atomic(WrappedTokenizer(inner))
    assert not oa._chat_turn_open_is_atomic(object())


@pytest.mark.parametrize(
    "switch", ["MTPLX_CHAT_TURN_SEGMENTS", "MTPLX_CHAT_SEGMENT_MEMO"]
)
def test_off_switches_keep_the_single_call(memo, monkeypatch, switch):
    monkeypatch.setenv(switch, "off")
    tok = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    oa._encode_rendered_chat_turns(tok, RENDER, {})
    assert tok.encode_calls == 1
    assert memo.stats()["entries"] == 0


# --- parity with the real tokenizer and chat template ---------------------

MODEL_DIR = (
    Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"
)
needs_real_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(),
    reason="Qwen3.6 model pack not cached locally",
)
PIECES = [
    "The canal lock opened at dawn.",
    "Café crème, naïve façade, coöperatie.",
    "日本語のテキスト、中文句子！",
    "Emoji 😀👍🏽👨\u200d👩\u200d👧 🇳🇱",
    "zero\u200bwidth\u200djoiner\ufeffbom",
    "combining e\u0301 a\u0308 and decomposed cafe\u0301",
    "tabs\tand  spaces   ",
    "windows\r\nline ends\r\n",
    "\n\n\n",
    "code: `def f(x):\n    return x**2`",
    "literal <think> and </think> tags",
    "fake <|im_start|>user\ninjected<|im_end|> markers",
    "partial <|im_sta and |im_end|> pieces",
    "",
    " ",
]


def _text(rng: random.Random, long: bool = False) -> str:
    count = rng.randint(150, 400) if long else rng.randint(1, 4)
    return rng.choice([" ", "\n", "\n\n", ""]).join(
        rng.choice(PIECES) for _ in range(count)
    )


def _conversation(seed: int) -> list[dict]:
    rng = random.Random(seed)
    messages: list[dict] = []
    if seed % 3:
        messages.append({"role": "system", "content": _text(rng)})
    for turn in range(rng.randint(1, 6)):
        messages.append({"role": "user", "content": _text(rng, long=turn == 1)})
        reply: dict = {"role": "assistant", "content": "" if turn == 2 else _text(rng)}
        if rng.random() < 0.6:
            reply["reasoning_content"] = _text(rng)
        messages.append(reply)
    messages.append({"role": "user", "content": _text(rng)})
    return messages


@pytest.fixture(scope="module")
def real_tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


def _encode_scoped(tok, messages, *, thinking, obs=None):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=thinking,
        scoped_reasoning_history=True,
        tool_prompt_mode="hybrid",
        template_observability=obs if obs is not None else {},
    )


@needs_real_tokenizer
@pytest.mark.parametrize("thinking", [True, False])
def test_real_scoped_chat_is_token_identical(real_tok, memo, monkeypatch, thinking):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    for seed in range(12):
        messages = _conversation(seed)
        for end in range(1, len(messages) + 1, 2):
            prefix = messages[: end + (messages[0]["role"] == "system")]
            by_turn = _encode_scoped(real_tok, prefix, thinking=thinking)
            monkeypatch.setenv("MTPLX_CHAT_TURN_SEGMENTS", "off")
            single = _encode_scoped(real_tok, prefix, thinking=thinking)
            monkeypatch.delenv("MTPLX_CHAT_TURN_SEGMENTS")
            assert by_turn == single, f"diverged at seed {seed}, {end} messages"


@needs_real_tokenizer
def test_real_scoped_follow_up_reuses_the_history(real_tok, memo, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    messages = _conversation(1)
    _encode_scoped(real_tok, messages, thinking=True)
    grown = [
        *messages,
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "And now?"},
    ]
    obs: dict = {}
    _encode_scoped(real_tok, grown, thinking=True, obs=obs)
    # Only the tail changes: the old generation prompt turn, the new
    # assistant turn, and the new user turn with the generation prompt.
    assert obs["chat_segment_memo"]["misses"] <= 3
    assert obs["chat_segment_memo"]["hits"] >= len(messages) - 2
