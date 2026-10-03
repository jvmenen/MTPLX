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

## 6. Fine-tuning the MTP head on own text (cheap trial, 3 October): no gain

Setup: LoRA (rank 32, 3.96M trainable parameters) on fc, q/k/v/o and gate/up/down of the 4-bit head. Loss: KL against the trunk's top-64 distribution, restricted to the production FR-Spec table, depth weights 0.5/0.3/0.2, recursive rollout with the head's own hidden state. Data: 2.15M off-policy tokens (61% Dutch documents and assistant answers written by other models or people, 20% code, 20% English), split by source document. A teacher-forced trunk pass cached the post-norm hidden state and top-64 logits per position (2.4M tokens, 24 GB). Training: 1.5 epochs, learning rate 2e-4 cosine, about 770 tokens/s. Total GPU time 4.3 hours.

The offline evaluator was checked against the real MTPLX head: identical drafts in 100%, 99.2% and 95.8% of 120 positions per depth. On Dutch prose it reproduces the in-process acceptance (0.673/0.295/0.101 against 0.69/0.29/0.13).

Greedy acceptance per depth (cumulative), on-policy sets are greedy answers by Qwen3.8 itself on held-out prompts:

| Set | Current head | After LoRA, merged and requantized to 4-bit g64 |
|---|---|---|
| Dutch, on-policy (12.5K positions) | 0.757 / 0.481 / 0.295 | 0.761 / 0.487 / 0.302 |
| Dutch documents, off-policy | 0.799 / 0.576 / 0.450 | 0.819 / 0.596 / 0.471 |
| Code, on-policy | 0.866 / 0.689 / 0.549 | 0.861 / 0.689 / 0.549 |
| English, on-policy | 0.770 / 0.503 / 0.302 | 0.767 / 0.497 / 0.287 |

On Dutch text written by Qwen itself the gain is +0.004 to +0.007 per depth, about +0.6% tokens per verify; the decision rule was +0.04. On agent-style text written by another model the acceptance rose sharply in BF16 (0.535 to 0.755 at depth 1), but that is the head learning the other model's style: it does not carry over to Qwen's own output, and requantization removes part of it. The missing ingredient is on-policy data (self-distillation), which costs about 20 hours of generation per 1M tokens on this machine; this trial gives no reason to expect the earlier estimate of +9 to +13%.

## 7. Tree verification: offline estimate

From the same evaluator, base head, FR-Spec restricted, how often the target's greedy token is among the head's top-k (given that earlier depths were hit):

| Set | Top-1 d1/d2/d3 | Top-2 | Top-3 |
|---|---|---|---|
| Dutch, on-policy | 0.757 / 0.626 / 0.578 | 0.856 / 0.724 / 0.674 | 0.892 / 0.767 / 0.717 |
| Code, on-policy | 0.866 / 0.792 / 0.788 | 0.927 / 0.864 / 0.858 | 0.948 / 0.891 / 0.883 |

Expected accepted tokens per verify for the current 4-row chain against two small trees (t211: 2 candidates at depth 1, 1 below each, 6 nodes; t221: 2 at depth 1 and 2, 1 at depth 3, 10 nodes; each branch continues the head's own rollout):

| Set | Chain | t211 | t221 |
|---|---|---|---|
| Dutch, on-policy | 2.53 | 2.71 (+7%) | 2.83 (+12%) |
| Dutch blog (hardest) | 2.07 | 2.24 (+8%) | 2.37 (+15%) |
| Code, on-policy | 3.10 | 3.23 (+4%) | 3.33 (+7%) |
| English, on-policy | 2.57 | 2.80 (+9%) | 2.98 (+16%) |

These are token counts only: the extra cost of verifying 6 or 10 rows instead of 4 and of drafting a tree is not included, and the GatedDeltaNet layers cannot process a tree in one sequence (each branch needs its own recurrent state, so in practice parallel chains with a shared prefix). A measurement of verify cost against row count is the next step.

## Conclusions

1. Keep the int4 head and context-copy as they are.
2. No DFlash2 integration for Qwen3.8 on this hardware.
3. Fine-tuning the MTP head on off-policy text does not move acceptance on Qwen's own Dutch output; no further work without on-policy data.
4. Tree verification is the most promising remaining lever for prose (+12 to +16% tokens per verify for a 10-node tree in the offline estimate), provided the extra verify cost on this dense hybrid model stays small; to be measured.

Scripts and raw data are local (`~/Dev/laya-nl/pld-38/`, `dflash38/`, `q38bf/`, `prefill38/`, `mtp-head-ft/`); the training data and cache contain private text and are not published.
