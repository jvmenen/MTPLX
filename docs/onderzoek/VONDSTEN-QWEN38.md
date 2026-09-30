# Findings Qwen3.8-27B

Running list for the dense model Qwen3.8-27B (Youssofal Optimized-Speed: 4-bit g32 body, lm_head Q8/g64, MTP head bf16). Started 30 September 2026. Same ground rules as [VONDSTEN.md](VONDSTEN.md); report: [2026-09-30-qwen38](2026-09-30-qwen38.md).

Measurement setup (local, `~/Dev/laya-nl/qwen38/`): `run38.zsh <variant>` starts a fresh server per variant on port 8000 with the final-test-3 arguments for Qwen3.8 and the Bink production env, on `perf/definitief` (a5df61d0). Then `bench38.py` (4 prompts with reasoning on, server sampling, 1,200 tokens, 2 rounds; plus cold prefill of ~7K and ~22K tokens) and 2x `lus2.py` (real agent task). Hardware: M5 Pro 64 GB, turbo profile, depth 3.

## Baseline (30 Sep)

| Measurement | Value |
|---|---|
| `mtplx tune` (code suite, thinking off, 512 tokens): AR / D1 / D2 / D3 | 13.1 / 29.9 / 40.5 / 48.2 tok/s |
| tune D3: acceptance per position; verify-forward + eval per round | 0.98 / 0.91 / 0.82; 72 + 11 ms |
| bench38 thinking (reasoning on, temp 1.0): median / total | 34.1 / 33.9 tok/s; acceptance 0.88 / 0.71 / 0.57; 3.18 tokens per verify |
| Cold prefill 7,182 / 21,626 tokens | 17.6 / 55.2 s (~400 tok/s) |
| lus2 agent task (2x) | green, 81 and 57 s |

## Open

Noise: `frspec` (FR-Spec not active) came in at 32.4 tok/s against a baseline of 34.1; differences under ~5% count as not demonstrated. Variant `basis2` measures repeatability.


| # | Finding | Source | Next step |
|---|---|---|---|
| Q2 | FR-Spec (pruned draft head with the bundled 64K Qwen3.8 vocab) is only enabled by default for Flash-Next. On the 27B pack the head gets installed without `MTPLX_FRSPEC_LEGACY=1` but is not used: the binding `_mtplx_bind_draft_lm_head` only exists on Flash-Next's native MTP route (`frspec_draft.py:236-248`). With the legacy lane (`MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k MTPLX_FRSPEC_LEGACY=1`, draft temp 0.6): **36.4 tok/s against 34.1 and 33.6 for two baseline runs (+7 to 8%)**, acceptance equal (0.86 / 0.71 / 0.58), prefill equal, agent task 2x green. Prescatter and the sampled chain refuse on this route (`DraftK20PrescatterIneligible`) | code analysis; variants `frspec`, `frchain`, `frlegacy`, `basis2` (30 Sep) | Recommended setting for 3.8. Still to do: quality on non-code text (the vocab is ranked for code; output stays exact, only acceptance may drop) and the sampled chain on the legacy route (agent, see Q6) |
| Q3 | Prefill ~400 tok/s on 27B dense: compute-bound (~20 TFLOPS effective, comparable to Qwen3.6). Gains must come from less cold prefill (session bank, SSD), not from the kernel. Chunk 2048: equal to 4096 (18.4 / 57.0 s against 17.6 / 55.2 s; decode 34.2 tok/s). Chunk 8192: prefill 41.5 / 136.5 s and everything slow after that; the Mac started swapping (9.4 GB swap, 8.5 million swapouts) and kept doing so all night. So 4096 is the right value; don't use 8192 on 64 GB | bench38, variants `ch2k` and `ch8k` (30 Sep) | Measure bank behavior during long agent sessions (how often cold prefill occurs) |
| Q4 | KV at 64 KB per token (16 full-attention layers) causes memory pressure and bank trimming at ~65K context. `--paged-kv-quantization q8`: decode 30.8 tok/s (-9% against baseline), prefill equal; the compiled-verify paths drop out as the help text indicates. The memory effect was not measured in this workload | earlier model test (20 Sep), variant `kvq8` (30 Sep) | Only worthwhile if long sessions cold-prefill due to memory pressure; measure that bank behavior first |

| Q7 | Decode and acceptance drop at long context: in the QBR task (30 Sep) 35 tok/s at 20K, 22 tok/s at 60K, 19.7 tok/s at 75K; acceptance from 0.85 / 0.67 / 0.58 to 0.72 / 0.48 / 0.31. MTPLX has a depth policy for long context (`long_context_mtp_depth_policy`, threshold 98,304), but it is disabled for this model (`reason: disabled`) | request log QBR task 30 Sep | Measure whether D2 is faster than D3 above ~50K (similar expected tokens per round, smaller verify window and one fewer draft step); if so, propose a threshold and policy. Needs a server on port 8000 without Bink traffic |
| Q9 | The QBR task on 3.8 took ~60 min: the dispatcher aborted at the one-hour limit (`dispatch_timeout_seconds` 3600) after the report was already finished, and the automatic retry started overwriting it | QBR task 30 Sep | Platform issue: draft task for Daan (per-backend or per-task hour limit, no blind restart) |

## Handled

| # | Finding | Outcome |
|---|---|---|
| Q1 | Our measurement arguments set `--draft-temperature 1.0`; the maker calibrated 0.6 as the family default (+9% in his measurement, commit b8d6e785) | 30 Sep: applied (resolved to 0.6), no difference on our workload: 34.15 against 34.12 tok/s, acceptance 0.87 / 0.72 / 0.58 against 0.88 / 0.71 / 0.57. Doesn't matter for Bink; with `mtplx serve` without the flag, 0.6 already applies |
| Q6 | ~11 ms eval per verify round besides the forward | 30 Sep (Opus agent, report `~/Dev/laya-nl/qwen38/verify-analyse.md`): no extra work, just the tail of the same GPU verify (`generation.py:2345`); per round 87.7 ms, of which 73.5 ms verify and 11.6 ms drafts on the GPU, at most 1.5 to 2 ms host with an idle GPU. M=4 matmuls reach 250 to 280 GB/s, close to the bandwidth limit. Bit-identical there's at most ~2% to gain. Built on `perf/qwen38` (befe8933): `MTPLX_STOCK_SAMPLED_DRAFT_CHAIN=1`, same tokens, +0.5 to 1% (within noise), off by default. `MLX_MAX_MB_PER_BUFFER=1000`: +1.7%, within noise. Conclusion: gains now only from reading fewer bytes (FR-Spec, smaller draft lm_head) |
| Q5 | Verify-core variants | 30 Sep: `linear-gdn-from-conv` 33.6 tok/s (equal to baseline); `linear-gdn-from-conv-stream-skip0` broken on this model (no acceptance figures in `latest`, 22.8 tok/s). Current choice `-conv-tape` stays |
| Q8 | "Stutters" in the stream (`producer_gaps_over_200ms` 129 to 258 per long request) | 30 Sep: not a stall. The metric is the time between emitted tokens; with MTP, multiple tokens arrive per round, so the gaps are round times. At 75K context a round is 130 to 145 ms (p95), 2 to 6% of rounds reach 200 ms, peaks up to ~400 ms: attention over long context, not a verify or cache stall |
| Q10 | Thinking spiral at ~65K context (model test 20 Sep: thinking block that never ended) | 30 Sep: QBR task completes. `MTPLX_THINKING_BUDGET=6144` kicked in twice (09:50 and 09:57, at 53K and 62K context: 6,144 thinking tokens, force-closed, then continued normally). Report quality approaches the Opus reference (speakers cautious, process points complete), in 60 minutes |
| Q0 | Depth 4 or more | Not retested: the maker measured D6 -27% against D3 and D4 made the daemon die silently (`backends/descriptors.py:536-555`); cap stays at 3 |
