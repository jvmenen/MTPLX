# Qwen3.8-27B: speed options measured on 2 October 2026

Hardware: M5 Pro 64 GB, MLX 0.32.2, mlx-lm 0.31.3. Model: Youssofal Qwen3.8-27B MTPLX Optimized-Speed (4-bit g32 body, MTP head 4-bit g64). Code: `perf/definitief` unless stated otherwise. All runs in-process (no server), depth 3, thinking off, 600 output tokens, unless stated otherwise.

Workloads (local, `~/Dev/laya-nl/walnoot/pld-bench/`): a synthetic Dutch e-mail with an extraction-to-JSON instruction (~1,700 prompt tokens), a 150-line Python file with "add type hints and return the full file" (~1,400 tokens), a short Dutch blog post (free writing), and an agent-style Edit (return `old_string`/`new_string` for a 154-line file).

## 1. Prompt lookup already exists: context-copy

MTPLX ships prompt-lookup drafting as **context-copy** (`mtplx/context_copy.py`), on by default (`MTPLX_CONTEXT_COPY=0` turns it off) on the `capture_commit` verify lanes. An n-gram match (6 to 10 tokens) over the prompt replaces the MTP draft with a copy block of 8 to 32 tokens, verified in one capture-commit forward; sampling uses point-mass acceptance with residual resampling, and an acceptance EMA suspends it with exponential backoff.

Measured on Qwen3.8 (`origin/main` base, 3 runs, median decode tok/s):

| Workload | Temp | MTP only | MTP + context-copy | Min n-gram 3 |
|---|---|---|---|---|
| Extraction | 0 | 39.4 | 39.1 | 37.8 |
| Extraction | 1.0 | 34.7 | 35.0 | 34.4 |
| Code edit | 0 | 38.3 | **47.2** (+23%) | 40.3 |
| Code edit | 1.0 | 36.8 | **50.8** (+38%) | 29.2 |
| Blog | 0 | 21.4 | 21.6 | 21.6 |
| Agent Edit | 0 | 40.8 | **84.7** (2.1x) | 74.7 |
| Agent Edit | 1.0 | 39.5 | **85.0** (2.15x) | 71.8 |

- On the agent Edit half of all verify rounds are copy rounds, with 20 accepted tokens per copy round.
- The default minimum of 6 tokens beats 3 everywhere; 3 proposes wrong blocks (at temperature 1.0 it is slower than MTP only).
- Extraction to JSON gains nothing: the output does not repeat 6-token runs of the prompt.
- Greedy output is identical to MTP only on extraction, code edit and blog (600 of 600 tokens); on the agent Edit it diverges after 214 tokens (batched verify numerics, see section 5).

**Production logs** (request log since 29 September, numeric fields only): context-copy was active on every request. Qwen3.6: 15% of all generated tokens came from copy rounds (1,205 of 3,220 requests); Qwen3.8: 7% (572 of 4,066). Copy rounds draft 12 to 13 tokens and accept 5.5 to 6.7 on average (45 to 52%). Estimated effect: about 9% fewer forward passes on Qwen3.6, about 5% on Qwen3.8 (estimate, not measured).

Not measured: `MTPLX_RAMP_ENABLED` (fuzzy re-anchor), other block lengths, min n-gram 4 or 5. Rebuilding the n-gram index per request costs 13 ms at 90K prompt tokens (measured), so it is not worth caching.

## 2. Prompt lookup for models without an MTP head

Branch `feat/prompt-lookup-draft` (local, not pushed) adds a model-independent drafter (`mtplx/prompt_lookup.py`, 11 unit tests) and hooks it into `generate_ar` behind `MTPLX_PROMPT_LOOKUP=1`: greedy only, AR runtimes only, verify via a multi-token `forward_ar` and cache trim. Tested on a dense 8B Apertus-based model without MTP head (6-bit), with another model loaded on the GPU at the same time, so absolute numbers are noisy:

| Workload | Off | On | Speed-up | Identical |
|---|---|---|---|---|
| Code edit | 20.3 | 60.6 | 2.99x | yes |
| Extraction | 20.3 | 26.0 | 1.28x | no (token 8) |
| Blog | ~38.7 | 40.2 | ~1.0x | no (token 86) |

The divergence comes from batched rows giving slightly different logits than one-row steps in bf16 (max logit difference 1.2 to 5); forcing a per-row recompute makes the output identical, so the accept and rollback logic is correct. For MTP models this branch is redundant (context-copy covers it).

Side finding on that model: plain 4-bit (g64) and `mixed_4_6` conversions with `mlx_lm.convert` echo the prompt instead of answering; 6-bit and 8-bit work. Its config uses the transformers 5 `rope_parameters` key, which mlx-lm 0.31.3 does not read; adding `rope_theta` and `rope_scaling` fixes the conversion.

## 3. DFlash 2 drafter (stopped early, indicative)

Drafter `incoai/Qwen3.8-27B-DFlash2` (revision 015e795) with `z-lab/dflash` at 07ebd93, quantized to 4-bit g64 at load, on the same production body (so the target weights are identical). Greedy, 60 s cool-down and thermal pressure 0 before each run.

| Workload | AR | DFlash2 block 8 | DFlash2 block 5 | MTPLX MTP only (section 1) | MTPLX + context-copy |
|---|---|---|---|---|---|
| Extraction | 14.2 | 35.9 (5.25 per verify) | 38.9 (3.82) | 39.4 (3.68) | 39.1 |
| Code edit | 14.4 | 39.0 (5.70) | 42.7 (4.16) | 38.3 (3.64) | 47.2 |
| Blog | 14.5 | 11.7 (1 run) | 17.1 (1 run) | 21.4 | 21.6 |

- DFlash2 at block 5 is on par with MTP only on extraction and code edit, clearly worse on free prose, and behind MTP plus context-copy on edit-shaped work. The drafter adds about 1.2 GB.
- Missing: the same-session MTPLX comparison with cool-down, the agent Edit, Dutch prose, sampling. Stopped on purpose: the interim result does not justify integration (block-diffusion drafter, five tapped layers, GDN rollback on partial accept).

## 4. BF16 MTP head on Qwen3.8 does not help

The Optimized-Speed pack labels its head `keep_bf16`, but `mtp.safetensors` holds a 4-bit g64 head (239 MB, `U32` payload with scales; `mtplx_mtp_quantization.prequantized: true` in `config.json`). A combination pack with the unchanged body and the BF16 head from `Qwen/Qwen3.8-27B` (shard 18, 15 tensors; norms stored with +1 as in the MLX pack; `mtplx_mtp_quantization` removed) loads and runs bit-identical in greedy mode.

A/B, 2 rounds interleaved, 2 runs per workload per round, 60 s rest plus thermal pressure 0 before each process (pressure rose to 2 during each run for both heads):

| Workload | Temp | int4 head | BF16 head | Difference |
|---|---|---|---|---|
| Extraction | 0 | 39.2 | 36.5 | -6.8% |
| Extraction | 1.0 | 34.5 | 34.0 | -1.6% |
| Code edit | 0 | 38.4 | 36.4 | -5.2% |
| Code edit | 1.0 | 36.4 | 30.3 | -16.7% (sampled text differs; acceptance happened to be lower) |
| Blog | 0 | 22.2 | 20.8 | -6.3% |
| Blog | 1.0 | 19.2 | 18.8 | -1.7% |
| Agent Edit | 0 | 41.0 | 37.8 | -7.9% |
| Agent Edit | 1.0 | 38.8 | 36.6 | -5.7% |

Acceptance per depth is equal within noise (greedy blog 0.69/0.29/0.13 against 0.67/0.29/0.15), so the int4 head loses nothing and the BF16 head only adds bytes per draft step. This differs from Qwen3.6-35B-A3B, where the BF16 head gained 12 to 32%. Dutch prose stays the weak spot for either head; a gain there has to come from training, not precision.

## 5. Prefill and session reuse

- Matmul at the prefill shape (4096 x 5120 x 17408): BF16 30.9 TFLOPS, 4-bit `quantized_matmul` 28.3 to 28.5, dequantize plus BF16 29.7. Kernels leave at most ~10%; prefill (~400 tok/s, ~21 TFLOPS effective) loses the rest in attention over long context and the GDN layers.
- Request-log analysis: 43% of Qwen3.8 server time went to prefill, but most of it came from short standalone prompts (classification, titles) and from our own benchmark runs. In real agent sessions follow-up turns get their first token after 0.4 to 1.2 s at 20K to 90K context, and a new session reuses the shared system/tool prefix of an earlier one (11.5K tokens observed). Only the first session after a server restart is cold (31 to 34 s for 13K to 14K tokens).

## Conclusions

1. Keep the int4 head and context-copy as they are.
2. No DFlash2 integration for Qwen3.8 on this hardware.
3. The remaining lever for Dutch prose is the MTP head itself (acceptance 0.69/0.29/0.13 on free Dutch writing against 0.94/0.87/0.82 on code): fine-tuning on own text is being tried as an offline experiment.
4. Tree verification (several candidates per position) is a candidate for the same weak spot; first an offline estimate of top-2/top-3 hit rates from cached hidden states.

Scripts and raw data are local (`~/Dev/laya-nl/pld-38/`, `dflash38/`, `q38bf/`, `prefill38/`).
