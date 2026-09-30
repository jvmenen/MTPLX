"""Server default: FR-Spec on for dense Qwen3.8 packs, every other pack unchanged."""

from __future__ import annotations

import argparse
import json

import pytest


def _dense_config(**overrides) -> dict:
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248_320,
            "tie_word_embeddings": False,
        },
        "quantization": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "language_model.lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
        },
    }
    config.update(overrides)
    return config


def _args(model, **overrides) -> argparse.Namespace:
    values = {
        "model": str(model),
        "verify_strategy": "capture_commit",
        "generation_mode": "mtp",
        "scheduler_mode": "serial",
        "draft_lm_head_bits": 4,
        "draft_lm_head_mode": "affine",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _pack(tmp_path, name: str, config: dict):
    model = tmp_path / name
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    return model


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("MTPLX_FRSPEC_DRAFT", "MTPLX_FRSPEC_VOCAB", "MTPLX_FRSPEC_LEGACY"):
        monkeypatch.delenv(key, raising=False)


def _frspec_keys(overrides: dict) -> dict:
    return {key: value for key, value in overrides.items() if "FRSPEC" in key}


def test_qwen38_dense_pack_gets_frspec_by_default(tmp_path, clean_env) -> None:
    from mtplx.profiles import normalize_runtime_env_overrides
    from mtplx.server.openai import _server_runtime_env_overrides

    model = _pack(tmp_path, "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config())
    overrides = _server_runtime_env_overrides(_args(model), {})

    assert _frspec_keys(overrides) == {
        "MTPLX_FRSPEC_DRAFT": "1",
        "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
    }
    assert normalize_runtime_env_overrides(overrides) == overrides


def test_explicit_off_switch_wins(tmp_path, clean_env, monkeypatch) -> None:
    from mtplx.server.openai import _server_runtime_env_overrides

    model = _pack(tmp_path, "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config())
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")

    assert _frspec_keys(_server_runtime_env_overrides(_args(model), {})) == {}


def test_operator_vocab_wins(tmp_path, clean_env, monkeypatch) -> None:
    from mtplx.server.openai import _server_runtime_env_overrides

    model = _pack(tmp_path, "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config())
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(tmp_path / "ranked.npy"))

    assert _frspec_keys(_server_runtime_env_overrides(_args(model), {})) == {
        "MTPLX_FRSPEC_DRAFT": "1"
    }


@pytest.mark.parametrize(
    ("name", "config", "arg_overrides"),
    [
        # Other families on the same qwen3_5 layout keep their defaults.
        ("Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed", _dense_config(), {}),
        ("Qwen-Qwen3.5-9B-MTPLX-Speed", _dense_config(), {}),
        # Rotated derivative of the 3.8 family (Bonsai): different layout.
        (
            "Youssofal--Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
            _dense_config(model_type="prism_hadamard_qwen35"),
            {},
        ),
        # No configured affine draft head to prune.
        ("Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config(), {"draft_lm_head_bits": 0}),
        ("Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config(), {"draft_lm_head_mode": "mxfp4"}),
        # Not MTP.
        ("Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", _dense_config(), {"generation_mode": "ar"}),
        # A vocabulary the builtin table was not ranked over.
        (
            "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
            _dense_config(text_config={"vocab_size": 151_936, "tie_word_embeddings": False}),
            {},
        ),
        # Tied output projection.
        (
            "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
            _dense_config(text_config={"vocab_size": 248_320, "tie_word_embeddings": True}),
            {},
        ),
    ],
)
def test_other_packs_are_unchanged(tmp_path, clean_env, name, config, arg_overrides) -> None:
    from mtplx.server.openai import _server_runtime_env_overrides

    model = _pack(tmp_path, name, config)
    overrides = _server_runtime_env_overrides(_args(model, **arg_overrides), {})

    assert _frspec_keys(overrides) == {}
