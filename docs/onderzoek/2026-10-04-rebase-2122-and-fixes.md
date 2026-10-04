# Rebase onto v2.12.2, prefill chunk plan, SDPA threshold, SSD fingerprint, upstream issues, AR context-copy, Bonsai 2 (3 to 4 October 2026)

Hardware for every measurement: Apple M5 Pro, 64 GB. Models: Qwen3.8-27B Optimized-Speed (dense), Qwen3.6-35B-A3B (MoE, production Balance/Speed packs), Llama-3.1-8B-Instruct-4bit, Ternary-Bonsai-2-27B. Findings 99 to 104 in [VONDSTEN](VONDSTEN.md). All numbers are measured on the real model (on the server or in process) unless marked as estimated or code read only.

## 1. Rebase of `perf/definitief` onto v2.12.2

Branch `perf/definitief-2122` (fork, 71288758) sits on tag `v2.12.2` (9882703f). `perf/definitief` was upstream 2.12.0 plus 41 non-merge commits. Classification of those commits (local list `~/Dev/laya-nl/rebase2122/commits.md`):

| Class | Count | Meaning |
|---|---|---|
| Already upstream | 18 | Our PRs, re-written upstream as separate commits (first-token logprobs, scoring top-K, chat-encode memo, scoped chat, test isolation, stream recovery, incremental SSD encode and others) |
| Superseded | 2 | `MTPLX_POSTCOMMIT_AFTER_RESPONSE` (upstream does it by default under `MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT`) and `MTPLX_PERSIST_QUEUE_MAX_GB` (upstream `MTPLX_PERSISTENCE_MAX_PENDING_BYTES`, default RAM/32) |
| Carried | 21 | Batch-invariant lane (#549), scoring trunk, in-forward GDN boundaries (#550), one-kernel MoE combine (#558), session head anchor (#559), scoped reasoning (#570), verify async chunk (#579), plus 8 opt-in commits that are not in the production command |

Notes:

- The head anchor (#559) had to be re-implemented: upstream replaced the GDN boundary thinning with `checkpoint_anchors.py` (8,192 grid, `CheckpointSink`), 14 conflicts in `generation.py`. The anchor is now `AnchorPlan.head_anchors`; 43 tests adapted and green (bank simulation, bit-identical on the tiny MoE, SSD).
- Env flags of the production command: `MTPLX_MTP_HISTORY_CACHE_ONLY` is now default-on upstream (no-op). `MTPLX_POSTCOMMIT_AFTER_RESPONSE` and `MTPLX_PERSIST_QUEUE_MAX_GB` are obsolete (superseded). `MTPLX_SSD_INCREMENTAL_ENCODE=1` is still needed: upstream first enabled it, then turned it off again (57fb2105, collision risk, see section 4). `MTPLX_BATCH_INVARIANT_PREFILL`, `MTPLX_A3B_MOE_PREFILL_COMBINE`, `MTPLX_SESSION_HEAD_ANCHOR` and `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS` remain ours. The lane is rejected on dense Qwen3.8 (`dense_model`).
- Tests (CPU, `MTPLX_CONFIG=/nonexistent`): tag v2.12.2 and the branch both 3,723 ruff findings; one test failure on the branch (a stub without `abort_check`), fixed in dd28b920; otherwise 0 failures. Two non-reproducible aborts (g++ link clash in the MLX CPU cache, Metal `Aborted` with a shared GPU) disappeared when run again.

### Four-variant measurement

| Variant | Tree | Environment |
|---|---|---|
| A | `perf/definitief` (current production) | full production env |
| B | bare v2.12.2 | none |
| C | v2.12.2 | only the 8 flags that exist upstream (cache-only, two chunk sizes, incremental SSD, thinking budget, three FR-Spec flags) |
| D | `perf/definitief-2122` | full production env plus `MTPLX_PERSISTENCE_MAX_PENDING_BYTES=4 GiB` |

Each run: fresh server on port 8000 with its own SSD cache directory, 145 agent-style requests (decode per text type), 12 prose requests, 8 long-prompt requests (cold and warm prefill), two agent loops (tests green in every run, 0 tracebacks).

**Qwen3.8-27B, round 1 (3 Oct, thermal pressure 1 to 2, `run-ronde1.log`).** Decode in tok/s, cold prefill in seconds, warm prefill is the follow-up on the same prompt.

| Variant | Decode all | code | en | nl | nl-uit | Cold 8K | Cold 22K | Warm 8K | Warm 22K | 145-request run |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 29.9 | 36.6 | 28.0 | 28.2 | 24.6 | 19.6 | 56.1 | 0.43 | 0.60 | 242 s |
| B | 28.0 | 33.1 | 26.2 | 27.2 | 22.8 | 19.3 | 55.2 | 0.44 | 0.74 | 247 s |
| C | 29.3 | 35.0 | 27.7 | 28.0 | 24.1 | 19.3 | 55.3 | 0.48 | 0.60 | 243 s |
| D | 30.1 | 37.2 | 27.5 | 28.9 | 24.4 | 23.5 | 66.2 | 0.58 | 0.73 | 244 s |

- A repeat of A in round 2 gave decode 30.3, cold 19.7 / 56.6 s: repeatability about 1.4% on decode, about 1% on cold prefill.
- The D cold-prefill outlier (66.2 s against 55.3 s for C) was machine interference: during that run another MLX job and a GPU-heavy browser process were active (sampler log: browser at 84% CPU, device utilization 50 to 84%). In-process re-measurement of the chunked prefill without a server (`~/Dev/laya-nl/rebase2122/inproc/out/`, cold prefill 8K / 22K, median of runs):

| Run | Variant | 8K (s) | 22K (s) |
|---|---|---|---|
| first pass (disturbed, one run at 31.7 / 85.4 s) | C | 18.9 | 55.4 |
| | D | 20.8 | 78.4 |
| D with only C's flags (`Dc`) | D | 19.4 | |
| second pass, 2 runs each, machine quiet | C | 18.9 | 53.0 |
| | D | 17.2 | 52.5 |
| third pass, 8 runs of 8K | C | 20.2 | |
| | D | 19.2 | |

  Conclusion: on dense Qwen3.8, D equals C within noise; no regression from the rebase. The reported 66 s is not a property of D.
- Round 2 (B, C, D repeats) did not run: the guard stopped the series because swap grew to 6,744 MB against a limit of 4,556 + 2,048 MB, after the second A run (whose last step, the prefill check, failed with a refused connection). Round 1 therefore has one run per variant, round 2 only A.

**Qwen3.6-35B-A3B, variants A, C, D, two rounds (4 Oct, thermal pressure 0, `run.log`).** Mean of both rounds.

| Variant | Decode all | code | en | nl | nl-uit | Cold 8K | Cold 22K | Warm 22K | 145-request run |
|---|---|---|---|---|---|---|---|---|---|
| A | 101.3 | 111.4 | 100.0 | 98.2 | 88.4 | 3.47 | 10.16 | 0.29 | 65 s |
| C | 92.0 | 101.4 | 90.8 | 89.2 | 80.2 | 3.91 | 10.62 | 0.30 | 74 s |
| D | 100.0 | 111.2 | 98.1 | 95.6 | 87.8 | 3.42 | 10.01 | 0.29 | 65 s |

Per round the spread inside a variant is 0.5 to 3% (D nl: 99.7 and 91.5, the largest). Relative to C: A +10.1%, D +8.7% decode; cold 22K prefill A -4.3%, D -5.7%; the 145-request run 12% shorter. Draft acceptance is equal across variants (0.80 to 0.81 first position), so the gain is compute, not acceptance.

**Conclusion.** On the MoE model our carried work (batch-invariant lane, async verify chunk, MoE combine, head anchor) gives D about equal to A: about +9 to +10% decode and about 5% faster cold prefill against v2.12.2 with only the upstream flags. On dense Qwen3.8, D is equal to C (and A is within +2% of C); the carried work only matters for the MoE pack. Whether to switch production is open (finding 99); the switch needs a restart of the production server and Jeroen's approval.

## 2. Prefill chunk plan: `MTPLX_PREFILL_MIN_CHUNK_ROWS`

Branch `perf/prefill-chunk-plan` (fork, 9650a3d3), default off. MLX runs causal SDPA with head_dim 256 on the fused kernel only from 1,024 query rows; below that it takes the unfused matmul-softmax route. With chunk size 4,096 a 5,000-token append plans 4,096 + 839 + 64 rows, so the 839 rows pay the slow route. The switch merges a final chunk narrower than the value (1024 matches the kernel threshold) into the chunk before it; the tail ladder then re-cuts the merged chunk (3,911 + 1,024 + 64 for 5,000 tokens, 4,096 + 3,840 + 1,024 + 39 for 9,000).

Measured in process on Qwen3.8-27B, old and new plans interleaved, 3 repetitions, 60 s rest between runs (`~/Dev/laya-nl/chunkplan/measure.json`). Time in seconds for the whole append, median:

| Prefix | New tokens | Old plan | New plan | Saved |
|---|---|---|---|---|
| 20K | 5,000 | 12.60 | 12.54 | 0.06 (noise: one old run 14.04) |
| 20K | 9,000 | 22.92 | 22.61 | 0.31 |
| 80K | 5,000 | 20.69 | 19.25 | 1.4 (paired per repetition: 0.12 to 1.56) |
| 80K | 9,000 | 35.45 | 34.60 | 0.85 |

The commit message states -0.4 to -0.95 s per append; in the table above the gain is 0.3 to 1.4 s where it is visible (80K, and 9,000 tokens at 20K), and as large as the run-to-run spread at 20K with 5,000 tokens. Only appends whose last chunk is under 1,024 rows are affected; a 6,000-token append (4,096 + 1,839) gets the same plan on both sides and shows no change.

Why the 64-row tail stays: it is the tail ladder's last rung (per the code comment, it exists for families without in-forward boundary capture) and the switch only merges the remainder in front of it. The tail costs 0.24 s at 20K and 0.44 to 0.47 s at 80K (measured). Removing it would change the session-bank boundary grid, which was outside this change.

Correctness (teacher-forced 24 tokens after the append, `correct2.json`, 20K prefix plus 5,000 tokens): the new plan is not bit-identical to the old one because the chunk shapes differ: max absolute logit difference 1.26, mean KL 0.00024, max KL 0.0019, top-1 agreement 96% (23 of 24). A control with the old plan and chunk size 2,048 differs by the same amount (1.39, mean KL 0.00032, 96%). The old plan against itself is exactly equal. So the change is of the same size as any change of chunk shape, not larger.

Qwen3.6 Balance pack (`measure_balance.json`, different chunk configuration, the new plan cut 5,000 tokens as 2,500 + 1,411 + 1,024 + 64): mixed. 5,000 new tokens at 80K 22.3 to 19.1 s (-3.0 s), 9,000 at 20K -1.0 to -3.3 s, but 9,000 at 80K got slower in 2 of 3 pairs (+2.3 and +2.6 s). The cause of the extra cuts on that pack was not investigated. Therefore: Qwen3.8 only, off by default; no PR yet (finding 100).

## 3. MLX SDPA head_dim 256

Local fork build of MLX in `~/Dev/mlx-sdpa` (not in the MTPLX fork). Kernel benchmark, bf16, 24 query / 4 KV heads, D=256, causal, ms per call, median of three runs (`bench/out`, `agg.py`).

**New MLX release v0.32.3 against the installed wheel.** In the second, quieter series the difference is 0 to 2% slower and the output is bit-identical (the first series, with load on the machine, showed up to 12% in single cells):

| Query rows + prefix | wheel | v0.32.3 |
|---|---|---|
| 2048 + 20K | 49.7 | 50.9 |
| 6144 + 0 | 21.4 | 21.8 |
| 6144 + 20K | 164.8 | 168.3 |
| 6144 + 50K | 439.1 | 447.3 |
| 6144 + 80K | 626.1 | 640.5 |
| 1023 + 50K (unfused) | 106.8 | 107.0 |

No gain from the upgrade for this shape.

**Experimental kernel variant with TQ=2 (two query fragments per simdgroup, query block 128).** Local experimental kernel (`patches/wip-experimental.diff`) with env overrides for the tile shape: query block 128 gave 6.2 to 7.5 TFLOPS against 17 to 21 for the default (2.8 to 3.2x slower), key block 64 gave 12 to 15 TFLOPS (1.2 to 1.6x slower), warp-m 2 gave 5.3 to 5.8 TFLOPS (about 3.4x slower). The default tile stays. Example, 6,144 rows + 50K: default 467 ms, bq128 1,252 ms, bk64 658 ms, wm2 1,521 ms.

**Fused-route threshold (MINQ).** `use_fallback` in `mlx/backend/metal/scaled_dot_product_attention.cpp` sends head_dim 256 causal SDPA to the fused kernel only from 1,024 query rows. With a patch that makes this threshold an env value (`MLX_SDPA_D256_MINQ`, default unchanged at 1024), the fused kernel at 256, 512 and 1,024 as the threshold gives (ms, median; the 1,024 column is the unfused route for rows below 1,024, with a wide spread):

| Query rows + prefix | threshold 256 (fused) | threshold 1024 (unfused) |
|---|---|---|
| 512 + 20K | 12.4 | 111 (best 18) |
| 512 + 80K | 48.7 | 888 (best 80) |
| 839 + 50K | 60.2 | 214 (best 88) |
| 839 + 80K | 97.7 | 290 (best 150) |
| 1023 + 80K | 115.5 | 181 (best 180) |

The unfused route is slow and noisy at large prefixes because it materializes the score matrix. Fused output is slightly closer to a float32 reference (max abs error 8e-6 against 1.1e-5, 839 rows after a 20K prefix). The patch file is `~/Dev/mlx-sdpa/patches/minq.patch` with a drafted upstream description; it is local and not upstreamed.

**End to end (Qwen3.8-27B, in process, prefix in the bank, append of N tokens; `~/Dev/laya-nl/mlx-e2e/`).** Wheel MLX against the patched build with the threshold at 64 (`P64`), time for the whole append in seconds, median of 3 runs:

| Prefix | Append | Wheel | Patched | Saved |
|---|---|---|---|---|
| 50K | 100 | 0.54 | 0.54 | 0 |
| 50K | 350 | 1.47 | 1.27 | 0.20 (14%) |
| 50K | 730 | 3.02 | 2.61 | 0.41 (14%) |
| 50K | 5,000 | 15.82 | 15.80 | 0 |
| 80K | 100 | 0.69 | 0.68 | 0.01 |
| 80K | 350 | 1.91 | 1.54 | 0.37 (19%) |
| 80K | 730 | 3.92 | 3.16 | 0.76 (19%) |
| 80K | 5,000 | 19.08 | 19.29 | within noise |

The 80K and 50K wheel baselines come from the quieter of two wheel series (the other wheel series was 10% slower and would show gains up to 23%); the table is the conservative reading. At 20K (one wheel series) 350 and 730 tokens save 0.19 and 0.33 s (14 to 16%). A 100-token append and a 5,000-token append show no effect (the 5,000-token case is dominated by 4,096-row chunks that already use the fused kernel; for 100 tokens the saving is at most 0.04 s).

Correctness: greedy decoding of 24 tokens after a 350-token append at 20K, 50K and 80K gives the same tokens (3 of 3). Logits are not bit-identical (max absolute difference 1.7 to 4.6 on the logits, argmax equal at all positions), as expected from a different kernel for the tail rows.

Decision open: use the patched MLX in production or wait for upstream (finding 101). Gains matter for agent turns that append a few hundred tokens after a long prefix.

## 4. SSD incremental encode: fingerprint collision

Analysis from code reading, a NumPy simulation and our request log (`~/Dev/laya-nl/ssd-risico/rapport.md`).

- The fingerprint is two 32-bit sums per KV block (256 positions) or per whole tensor from 1 MiB (GDN state): a linear sum with odd per-position weights and a weighted sum of squares of the raw bits.
- Collision (measured in simulation): an even number of float32 sign flips leaves both sums unchanged (the linear change is 2^31 times an even number, and the square is blind to bit 31). 2 and 4 sign flips collided in 100% of the trials; one flip and a swap of two values did not. bf16 KV blocks do not collide on sign-flip pairs (0 of 300).
- Risk in production (estimated, from the mechanism): reuse only happens within the same session, same position, newest previous write. Natural changes (other tokens, other chunk shape, MoE routing) change nearly all elements, so a collision needs about 2^-64 chance per block (about 10^5 blocks per turn, below 1e-13 per turn). Impact if it happened: wrong KV served silently. Logs (26,353 requests): 55 SSD restores, no error, no hash mismatch; the log has no reuse counters and a silent collision would not log anything, so this is no proof of absence.
- Switching the feature off costs (measured earlier, findings 65 and 72): +160% encode work (68k to 161k units per run), about 3 GB higher peak (51.7 against 48.0 GB), tail after the last token 20 to 40 ms heavier; speed otherwise equal.

Fix: branch `fix/ssd-fingerprint-mixer` (fork, 1f032eac). The second sum goes through a non-linear uint32 mixer (xorshift-multiply with `0x7FEB352D` and `0x846CA68B`) before the position weight. Output stays 64 bit, nothing is written to disk, no migration (fingerprints live in the in-process memo). Measured: 0 of 400 collisions in simulation (2 and 4 sign flips, 200 each); the strict xfail in `tests/test_cold_tier_incremental_encode.py` becomes a passing test; all 130 cold-tier tests pass.

GPU cost of `content_fingerprints`, old against new (M5 Pro, MLX 0.32.2, 20 warm-up + 300 timed calls, two interleaved passes, median):

| Input | Old | New |
|---|---|---|
| bf16 KV `[1,4,2048,256]`, 8 blocks, 4 MiB | 474-527 us | 845-853 us |
| bf16 KV `[1,8,2048,128]`, 8 blocks, 4 MiB | 472 us | 852-858 us |
| bf16 KV `[1,4,256,256]`, one block | 245 us | 260 us |
| float32 GDN `[1,48,128,128]`, 3 MiB | 361-367 us | 487 us |
| float32 GDN `[1,8,128,256]`, 1 MiB | 237-239 us | 260 us |

About +0.35 ms per 4 MiB call; small inputs +6 to +9%. Fingerprinting runs on SSD writes, not on the decode path. PR text: `~/Dev/laya-nl/pr-teksten/pr-ssd-fingerprint-mixer.md` (English, not opened). Advice: keep the feature on and take the fix with the next build (finding 102).

## 5. Upstream issues

**#583, empty content after a tool result (reproduced).** On requests that declare tools, the model sometimes answers inside the template-opened `<think>` block without `</think>`; the whole answer is filed as reasoning and `content` is empty. The existing recovery is gated on `not tools_active`. Reproduction (`~/Dev/laya-nl/issue583/`): one tool, six growing turns (+8K characters each), 3 runs, chat completions and messages, non-streaming; 18 turns per protocol per repeat.

| Build and setting | Empty turns per protocol, per repeat |
|---|---|
| v2.12.2 default, Qwen3.6-35B-A3B | chat 1/18, 1/18, 1/18; messages 0/18, 1/18, 1/18 |
| v2.12.2 with `MTPLX_THINKING_BUDGET=6144` | 2/18 and 2/18 (one repeat): the budget does not help |
| Our production configuration | 1/18 chat, 0/18 messages: affected |
| v2.12.2, Qwen3.8-27B | 0/18, 0/18: not affected |
| Fix branch | 0/18 in all three repeats, both protocols |

The failing turns are the same each time, so the repeats are close to one sample; the comparison is before and after on identical requests. Fix: PR [#588](https://github.com/youssofal/MTPLX/pull/588), branch `fix/unclosed-reasoning-with-tools` (5340f51e): helper `_unclosed_reasoning_recovery_allowed` allows the recovery with tools active when the finish reason is `stop`, no tool call was parsed and the text has no tool markup. Tests in `tests/test_unclosed_reasoning_with_tools.py`.

**#584, broken tool arguments with an OpenCode user agent (reproduced and analysed).** Qwen3.8-27B Optimized-Speed on v2.12.2, one screenshot in the history, thinking off, 10 requests per row (`~/Dev/laya-nl/issue584/`). "Malformed" means a value that is not a number.

| User-Agent | Tool prompt mode | Stream | Malformed |
|---|---|---|---|
| Python-urllib | default | no / yes | 0/10, 0/10 |
| `opencode/1.18.32` | default | no / yes | 6/10, 4/10 |
| `opencode/1.18.32` | hybrid | no / yes | 0/10, 0/10 |
| `opencode/1.18.32` | native | no / yes | 1/10, 1/10 |

Cause (analysis): the OpenCode user agent selects a compact tool-prompt mode; the same body renders as 1,245 prompt tokens against 1,553 in the default mode. In the compact rendering the model emits leftover `<parameter=...>` markup as values. `X-MTPLX-Tool-Prompt-Mode: hybrid` restores the default rendering and gave 0 of 20 malformed. Comment posted on the issue with the table and the cause: [issuecomment-5974656822](https://github.com/youssofal/MTPLX/issues/584#issuecomment-5974656822). No code change from our side.

## 6. Context-copy for AR-only models

Branch `feat/context-copy-ar` (fork, ee63be4d), switch `MTPLX_CONTEXT_COPY_AR=1`, default off. The context-copy lane (n-gram match against the prompt, copy a block, verify in one pass) exists only in the MTP loops; this adds it to `generate_ar` for models without an MTP head. The acceptance policy is extracted as `CopyGovernor`; the MTP lanes are untouched; the new lane is not used with hybrid recurrent models (no cache trim), penalties, constrained decoding or guards.

Measured on `mlx-community/Llama-3.1-8B-Instruct-4bit`, in process, off and on interleaved, 5 runs per cell, commit f1d7d1e1, 4 Oct (`~/Dev/laya-nl/pld-pr/results-summary.txt`), temperature 0, decode tok/s from verify time:

| Workload | Off | On | Decode ratio | End to end | Acceptance | Identical streams |
|---|---|---|---|---|---|---|
| code edit, 600 tokens | 55.3 | 289.3 | 4.9x to 5.2x | 3.4x | 0.78 | 5/5 |
| code edit, full file (1,301 tokens) | 56.0 | 275.2 | 4.9x | 4.3x | 0.86 | 5/5 |
| extraction (4,075-token report) | 52.4 | 69.2 | 1.3x | 1.25x | 0.56 | 5/5 |
| prose (72-token prompt) | 59.5 | 59.3 | 1.0x | 1.0x | no rounds | 5/5 |

Greedy: 20 of 20 streams identical to the plain loop (not guaranteed in general, since a multi-row forward can differ in the last bits). Sampled runs (temperature 1.0): same ratios (5.3x, 1.25x, 1.0x), law preserved by point-mass acceptance (chi-square test on a toy model; 150-seed check on the code prompt). Overhead on prose: none visible (59.3 against 59.5 tok/s). Peak memory equal. Not measured: hybrid models (lane stays off). PR text ready at `~/Dev/laya-nl/pr-teksten/pr-context-copy-ar.md`, not opened (finding 103). This replaces the earlier `feat/prompt-lookup-draft` branch for AR-only models.

## 7. Ternary-Bonsai-2-27B through MTPLX: Dutch quality

Public summary only; no prompt or answer content is recorded here. Pack: `Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`, production sampling settings and reasoning on, same short system prompt for all models, 12 task types (summarizing, extracting action items, rewriting, drafting a short mail and a short post, explaining a method, small date arithmetic, others) in Dutch, 2 rounds, 24 answers per model. Compared blind (labels shuffled per task and round) against Qwen3.8-27B in its production setup and a frontier reference model. Scored 1 to 5 by an agent reviewer on correctness, completeness and Dutch, plus counts of invented or garbled words and English insertions; not checked by a human.

| Model | Correct | Complete | Dutch | Invented or garbled words (total) | Best of three (of 24) |
|---|---|---|---|---|---|
| Frontier reference | 4.71 | 4.96 | 4.88 | 0 | 23 |
| Qwen3.8-27B | 3.17 | 3.21 | 3.17 | 22 | 0 |
| Bonsai 2 27B | 2.83 | 3.08 | 2.54 | 58 | 1 |

Speed (end to end including reasoning, same machine): Bonsai median 36.8 tok/s, mean 1,745 completion tokens and 46.6 s per answer; Qwen3.8 median 28.5 tok/s, mean 533 tokens and 19.3 s per answer. Bonsai decodes faster but reasons much longer, so it is slower per answer. 18 of 24 Bonsai answers contained at least one invented or garbled Dutch word, 6 of 24 contained five or more.

Conclusion: not suitable for Dutch writing tasks in this setup; Qwen3.8 stays. Earlier tests on classification and agent tasks are in [modeltest](2026-09-26-modeltest.md). Single run set, one reviewer: treat the gap with Qwen as indicative, the gap with the reference as clear.

## 8. Faster 4-bit prefill matmuls: two routes (4 October)

Goal: close the gap between the 4-bit `quantized_matmul` (28.3 TFLOPS) and the bf16 matmul (30.9 TFLOPS) at prefill shapes on Qwen3.8-27B (4-bit g32 body). In-process, production path, 60 s rest and thermal pressure 0 per series, 3 repetitions, medians.

**Route A, in MTPLX: dequantize to bf16 for wide prefill chunks** (branch `perf/dequant-prefill`, switch `MTPLX_PREFILL_DEQUANT_MIN_ROWS`, default off). Each projection's weight is dequantized per call (transient, about 178 MB) and multiplied in bf16.

| Case | Off | N=1024 | N=2048 | Peak memory |
|---|---|---|---|---|
| 6K appended, cold | 12.59 s | 12.32 s | | +1.0 GB |
| 6K at 20K prefix | 14.93 s | 14.66 s | 14.61 s | +1.0 GB |
| 6K at 80K prefix | 22.50 s | 22.25 s | 22.16 s | +1.0 GB |
| 730 at 50K prefix | 3.02 s | 3.02 s (not routed) | | |
| 730 at 50K, N=512 | | 3.20 s (+6%, slower) | | +0.4 GB |

Only the 4096-row chunk gains; an 1839-row chunk does not, and a 665-row chunk gets slower. Greedy output identical at 20K and 80K; teacher-forced KL about 4e-4, the same order as a chunk-size change. Verdict: about 0.3 s per 6K append for 1 GB more peak memory; not used.

**Route B, in MLX: a larger tile for large-M qmm** (patch `~/Dev/mlx-sdpa/patches/qmm-large-m-tile.patch` against v0.32.2, local only). For affine transposed qmm with M > 64 and K % 128 == 0 the NAX kernel uses BM=128, BN=64, BK=128 with 2x2 simdgroups instead of 64x64x64, plus a weight loader without the old thread-count constraint. Cause of the gap: per-step dequantization into threadgroup memory with barriers on a small tile; the bf16 NAX GEMM loads straight from device memory.

| Shape (K to N), M=4096 | Stock | Patched | bf16 |
|---|---|---|---|
| 5120 to 17408 | 25.85 ms | 24.25 ms (-6.2%) | 23.72 ms |
| 17408 to 5120 | 27.54 ms | 24.98 ms (-9.3%) | 26.02 ms |
| 5120 to 6144 | 9.19 ms | 8.64 ms (-6.0%) | 8.44 ms |

Bit-identical to stock in 178 cases (bits 2 to 8, group sizes 32 to 128, M 16 to 4096, unaligned N and K); M <= 64 (decode, verify) unchanged. End to end: 6K appended cold 12.59 to 12.06 s (-4.2%), at 20K 14.94 to 14.37 s (-3.8%). Tried and slower: 128x128 tiles, 256x64, double-buffered threadgroup memory; BN=32 or 16 tiles looked faster but produced wrong output (NAX tile needs an N-width of 32 per simdgroup). A similar upstream proposal for the non-NAX path (ml-explore/mlx#4204) was closed without review.

**Combined MLX build** (v0.32.2 + `minq.patch` + the tile patch, `~/Dev/mlx-sdpa/pkg-combo`, `MLX_SDPA_D256_MINQ=64`), end to end on Qwen3.8-27B, production MTPLX tree and env, medians of 3, interleaved, thermal pressure 0:

| Prefix | Append | Stock | Combined | Saved |
|---|---|---|---|---|
| 0 | 6000 | 12.58 s | 12.00 s | 4.6% |
| 20K | 350 | 1.07 s | 0.96 s | 10% |
| 20K | 730 | 2.19 s | 2.10 s | 4.0% |
| 20K | 6000 | 14.95 s | 14.35 s | 4.0% |
| 80K | 350 | 1.93 s | 1.50 s | 22% |
| 80K | 730 | 3.93 s | 3.15 s | 20% |
| 80K | 6000 | 22.50 s | 21.80 s | 3.1% |

Decode unchanged (MTP depth 3: 34.4 vs 34.5 tok/s; AR 13.75 vs 13.76), generated tokens identical. qmm bit-identical in 178 cases; the fused SDPA path adds a small logit drift (KL mean about 5e-4, top-1 agreement 100%, greedy continuations identical at 20K, 50K, 80K). Without `MLX_SDPA_D256_MINQ` the build is bit-identical to stock apart from the qmm speed-up.

## 9. Decode-side kernel profile on Qwen3.8-27B (4 October)

Production path in-process (turbo profile, dense KV cache, eager `forward_ar_capture`), MLX 0.32.2 wheel, thermal pressure 0 at the start of each series. Data and scripts in `~/Dev/laya-nl/decode-prof/` (local).

**Verify forward (M=4), ms:** 74.0 at 2K, 79.2 at 20K, 89.3 at 50K, 101.8 at 80K. Matmuls are about 75% of the forward at 2K and 60% at 80K (weights about 17 GB per forward, about 230 GB/s overall). Only the attention op grows with context: 2.3 ms at 2K to 27.9 ms at 80K.

**Attention route and efficiency.** Minimal KV read at 80K is 5.24 GB (64 KiB per position over 16 layers; an earlier estimate of 2.6 GB counted K or V only). Measured read roof 288 GB/s. At M=4 the MTPLX `sdpa_nax_flash_dsplit` route reaches 181 to 188 GB/s (63% of the roof), in-model equal to an isolated microbenchmark; M=1 (MLX vector 2-pass) reaches 238 GB/s (83%). The dsplit route bails when 6 x q_len > 32, so M=6 to 8 fall to `sdpa_nax_flash` (129 GB/s at 80K) and M>=9 to MLX's unfused path (112 ms at M=16, 131 ms at M=24, 80K). `MTPLX_GQA_PACKED_WIDE=1` with `MTPLX_NAX_TILE_ROUTE=1` does not help (M=16 worse).

**4-bit matmul bandwidth vs rows (production route, GB/s, gate/up 5120 to 17408 / down 17408 to 5120 / lm_head q8 5120 to 248320):** M=1: 283/282/280; M=4: 266/188/281; M=5: 241/184/284; M=6: 146/118/221; M=8: 199/165/121; M=12: 199/165/83; M=16: 146/174/190; M=24: 169/67/185. Cause: MLX `qmv_wide` handles M=2 to 12 in tiles of at most 5 rows and re-reads the weights per tile; MTPLX overlays kernels at M=4, 6 and 7 to 16 (4-bit only), M=5 falls back to stock; the q8 lm_head has no overlay at M=7 to 12 (10.6 to 16.2 ms instead of 4.8); `qmm_nax` above 16 rows drops to 60 to 67 GB/s on K=17408 and K=5120 shapes.

**Whole verify forward by rows, ms (2K / 80K):** M=4 74.0/101.8, M=5 83.6/111.9, M=6 82.4/122.2, M=8 105.5/149.1, M=12 124.4/299.4, M=16 122.6/237.3, M=24 192.8/325.5.

**Estimated payoff (model, not measured end to end):** M=4 attention at 80% of the roof: +1.7%/+3.1%/+5.5% tok/s at 20K/50K/80K. Matmuls at M=5 to 16 at M=4 bandwidth: depth 4 goes from about +2 to +5% to about +12% (assuming cumulative depth-4 acceptance 0.45), and the break-even for 24-row context-copy blocks at 80K drops from 7.3 to 5.1 accepted tokens. Small trees only pay with both that and wide-row attention.

**Ranking:** (1) lm_head at M=7 to 12: pad to 13 rows or route q8 through NAX, 3 to 9 ms per affected verify, trivial; (2) multi-row 4-bit matmul for M=5 to 16 without per-tile weight re-reads; (3) attention for q=6 to 24 at long context; (4) qmm above 16 rows on K=17408/K=5120 (split into 16-row calls); (5) M=4 attention efficiency; quick check: pad M=9 to 12 to 16 rows.

## 10. MLX patches in upstream form; correction on the SDPA threshold (4 October)

Both local MLX patches were ported to MLX main (65be04707, 128 commits after v0.32.2), formatted with MLX's pre-commit hooks, and covered by tests in `python/tests/test_fast_sdpa.py` and `python/tests/test_quantized.py` (both files pass with `MLX_ENABLE_TF32=0 MLX_ENABLE_CACHE_THRASHING_CHECK=0`, the settings MLX's own test runner uses; without them about 1,250 fp32 assertions fail on M5 because of TF32, which explains the earlier failures).

**Correction.** A clean 8x8 grid (qL 16 to 1023 against keys 0 to 80K, three head configurations, GPU guarded against other jobs) shows that the earlier "1.3 to 18x" for the fused SDPA route below 1024 query rows was inflated by allocator effects and GPU contention in the unfused runs. Clean ratios (unfused / fused): 512 rows 1.04 to 2.2 (growing with key length), 1023 rows 1.09 to 1.83, 256 rows 0.86 to 1.53 (mostly 1.1 or more from 8K keys), 64 to 128 rows about 1.0, 16 to 32 rows 0.5 to 0.8 (fused slower). The upstream form therefore uses the fused kernel for head_dim 256 causal fp16/bf16 when qL >= 512, or qL >= 256 and kL >= 8192. The end-to-end gains measured on Qwen3.8 for 350 and 730-token appends (section 8) remain valid; the production setting `MLX_SDPA_D256_MINQ=64` is about neutral for 64 to 255-row chunks.

**qmm tile, upstream form:** same kernel and loader; 1.02 to 1.33x (median 1.10x) on 21 large-M shapes, bit-identical; 144 new kernel instantiations, metallib +0.8%.

MLX's contribution rules require the PR description to be written by a person and suggest an issue first for larger changes; the SDPA part fits as input to the open draft ml-explore/mlx#4476.
