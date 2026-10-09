# Classifier prefix reuse and short requests behind a running generation (findings 117 and 116)

Date: 9 Oct 2026. Qwen3.8-27B 4-bit Speed, M5 Pro 64 GB, MTPLX-prod (`perf/definitief-2122`, read only), production flags, port 8010. Scripts and data: `~/Dev/laya-nl/classifier-prefix/`. Nothing pushed, config untouched.

Legend: **[measured]** = seen in a run in this report, **[read]** = from the code, not run, **[assumed]** = reasoning only.

## Part A: shared prefix reuse for classifier calls (finding 117)

### What the mechanism is

A logprobs request (`max_tokens: 1`) goes through the same session bank as any chat request [read]. `solo_logprobs` is only the scheduler lane label (`openai.py` ~30842): it means "not batchable, run serially". Nothing there blocks restoring or storing a prefix. In production the bank is not used for these calls because of three defaults [read]:

1. Restoring needs a shared prefix of at least 512 tokens (`MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS`).
2. A prompt is only banked when at least 1024 new tokens were prefilled (`MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX`).
3. The prompt-prefix commit before decode only happens at 512 tokens or more.

`--ram-session-prefix-min-match-tokens N` sets the threshold to N, switches on `MTPLX_SESSION_SHARED_PREFIX_EDGE` (the prefill records the recurrent state exactly where the shared head ends, which a hybrid model needs) and lowers the store minimum to N. Calls 1 and 2 only fill the bank; after that the head is restored.

### The fixed head is short [measured on the tokenizer]

`build_user_message` puts the text first and the question and labels after it. The shared head of two prompts is therefore only the system prompt plus `user\nTekst:\n<<<\n`: **41 tokens** of 218 to 289 (production 231 to 242). The assumption in finding 117 ("fixed instruction head") holds only for those 41 tokens. The 70 to 80 tokens of question and labels come after the varying text and can never be reused with the current prompt.

### Setup

40 synthetic Dutch meeting windows (80 to 100 words, starting mid-sentence), built with the real `LogitClassifier` system prompt and `build_user_message`; 4 of 6 question slots use the live-meeting question of `live_stt_bench.py`, the rest two other 3-label questions. 3 warm-up calls, then the 40 calls in fixed order, `temperature 0`, `max_tokens 1`, `top_logprobs 20`, thinking off. Server per run: production launch command with only the port (8010), a throwaway SSD cache dir, and for B `--ram-session-prefix-min-match-tokens 32` (below the 41-token head, so it matches). Order A1 B1 B2 A2, one server process each, all through `gate.sh`; thermal 0 at start.

Phase R (extra, not the production prompt): the same content with question and labels first and the text last, to see what a restructured prompt would give.

### Results, original prompt (phase P) [measured]

| run | wall p50 (s) | wall p95 (s) | prompt_eval p50 (s) | cached tokens | peak memory (GiB) |
|---|---|---|---|---|---|
| A1 (production flags) | 0.764 | 0.871 | 0.679 | 0 | 27.07 |
| B1 (prefix reuse) | 0.640 | 0.702 | 0.547 | 41 | 25.46 |
| B2 (prefix reuse) | 0.632 | 0.680 | 0.540 | 41 | 25.46 |
| A2 (production flags) | 0.864 | 1.007 | 0.771 | 0 | 27.07 |

- B restores 41 tokens on all 40 calls (the warm-up already seeded the bank, so calls 0 to 39 are all warm). Wall time p50 about 0.64 s against 0.76 to 0.86 s: **15 to 27% faster**, a gain of about 0.12 to 0.23 s per call. The A1 to A2 spread (0.76 against 0.86, same flags) is as big as part of that gain, so the order effect (heat) is not negligible; B1 and B2 agree with each other (0.640, 0.632). The saving per cached token is about 3 ms, as expected from 41 of about 264 tokens.
- Not "more than 2x" as finding 117 hoped. A large part of the 0.6 to 0.7 s is a floor that does not scale with the reusable head (assumed: fixed cost of the first-token path and the logits; not separated here).
- Peak memory (from the request log) is not higher with B: 25.46 against 27.07 GiB. I do not read this as a saving (peak varies with heat and allocator state between processes); the bank cost of 40 stored prompts did not show up in the peak. RSS of the server at the end: A 19.8 GB, B 19.7 GB. Bank growth over a long production run, and displacement of agent entries, was not measured.

### Correctness [measured]

- **Chosen token: identical in all 40 of 40 calls**, A against B (every pair of A1/A2/B1/B2).
- A against itself (A1 vs A2) and B against itself (B1 vs B2) are bit-identical in all top-20 logprobs (difference 0.0000). So the run-to-run noise is zero, and any A-B difference comes from the restore path (prefill in two pieces instead of one), not from noise.
- A against B: the largest absolute difference in any top-20 logprob is **0.62** (call 5, a low-probability tail token), median over calls 0.25. On the label tokens A, B, C: max 0.39, median 0.13. On the chosen token: max 0.08, median 0.007. Translated to the label probabilities the classifier computes (softmax over A/B/C): **max 0.054, median 0.008**. In 42 cases a token is in the top-20 of one run and not of the other (tail tokens at the cut-off).
- So the answer does not change here, but the numbers are not equal "within bf16 noise" in the sense of zero: they differ by up to 5 percentage points of label probability. A caller that thresholds probabilities near a boundary (for example a calibrated router) can flip on such a difference. Other prompts and the 80 to 100 word range only were tested.

### Phase R: question first [measured]

| run | wall p50 (s) | prompt_eval p50 (s) | cached tokens |
|---|---|---|---|
| A1 / A2 (no reuse) | 0.831 / 0.889 | 0.738 / 0.791 | 0 |
| B1 / B2 (after phase P in the same server) | 0.644 / 0.672 | 0.556 / 0.573 | 38 |
| B3R (fresh server, only phase R) | 0.612 | 0.519 | 124 (calls 3 onward; 32 on calls 1 to 2) |

- In B1/B2 phase R reused only 38 tokens, because the 40 phase-P entries dominated the "dominant shared prefix" vote in the bank. Only a fresh server shows the effect of the new layout: **124 tokens cached, wall 0.612 s against 0.83 to 0.89 s (-26 to -31%)**.
- Labels: 40 of 40 identical to A1's phase R labels in B3R; max top-20 logprob difference 0.50. Against the production prompt (phase P) the question-first layout gives a **different label in 11 of 40 calls** (29 of 40 agree). Moving the question is therefore a change of the classifier itself, not a pure speed-up, and would need re-evaluation of accuracy (it is not tested here which layout is more accurate).
- Even 124 of about 264 tokens cached leaves about 0.5 s of prompt_eval: the saving is about 0.2 s, not a halving [measured]. Cached-token gain is roughly 3 ms per token, also here.

### Conclusion A

- Prefix reuse with the current prompt works without code changes: `--ram-session-prefix-min-match-tokens 32`. Gain about 0.12 to 0.23 s of 0.76 to 0.86 s per call (15 to 27%), same chosen token in 40 of 40, label probabilities within 0.054.
- A larger gain needs the question and labels in front of the text (about 0.2 s, -26 to -31%), but that changes 11 of 40 answers and is a client decision.
- Not measured: the effect on agent traffic of banking every prompt of 32 or more new tokens (bank churn and displacement, SSD writes), a long-running memory curve, and behaviour with real transcripts instead of synthetic windows. In production the prompt_eval was 0.86 s at 231 to 242 tokens; in this test A was 0.68 to 0.77 s at about 264 tokens (production was under thermal pressure "heavy" in the log; here the gate required level 0). Absolute numbers are therefore lower than in the production log.

## Part B: a short request behind a running generation (finding 116)

### What exists [read]

- **Default `serial`**: the `ModelWorkScheduler` (`model_scheduler.py`) runs one foreground item at a time on one owner thread. A whole generation is one item. A short request waits for all of it (51 s observed).
- **`MTPLX_SHORT_REQUEST_PRIORITY=1`** (`_is_short_request`, `submit_priority_foreground`): requests with `max_tokens <= 64` and prompts up to 4096 tokens jump ahead of queued plain items, with an anti-starvation streak limit. It changes the order in the waiting queue only; the docstring says "never a running generation". It does not interrupt a running generation, so it does not help finding 116 (confirms the reasoning in 116). Earlier measured on titles in finding 54.
- **`cooperative`**: only a policy label. `MTPContinuousScheduler` (`mtplx/batching/scheduler.py`, "for future stepable generation") is not used by the server. In practice `cooperative` behaves like `serial`.
- **`ar_batch` / `mtp_cohort_experimental`**: a real pump (mlx-lm BatchGenerator) on the owner thread that newly arrived jobs join, so batchable requests do run concurrently. Costs: decode is target-only (no MTP). And logprobs requests are explicitly sent out of it (`first_token_logprobs` bypass, lane `solo_logprobs`): they run as a separate item and wait for the whole pump. So it does not solve this case either.
- **`mtp_batch`**: Qwen3.6-35B-A3B fixed B8/K1 only; it rejects logprobs outright. Not available for Qwen3.8-27B.
- **`hyper`**: FIFO like serial.

So there is **no existing mode** that interleaves a short logprobs request between decode rounds, preempts, or runs it concurrently.

### Smallest change that would work [design, not built, not measured]

1. **Hook point.** The decode loop of `generate_mtpk` (`while len(tokens) < max_tokens`, `generation.py` ~13152). After each round, `emit_new_tokens()` runs (~12760): tokens are committed, and housekeeping that may already block on `mx.synchronize` follows. Cancellation already works as an exception thrown from the callback at this spot. A "between rounds" hook right after it is the natural place.
2. **Work to run there.** A one-token logprobs request is a plain prefill-shaped pass with no decode: `score_prompt_logprobs` (`generation.py` 9349) already does this statelessly with its own fresh cache (`_make_target_prefill_cache`), no bank, "nothing here touches generation". It drops the last position (it predicts a token outside the prompt), so a variant that returns the last position's top-k is needed.
3. **Wiring.** `_run_generation_dispatched` puts logprobs requests with `max_tokens == 1` in a small interlude queue instead of the scheduler queue (while a generation runs); the hook drains that queue on the owner thread (MLX streams are thread-affine, so it must be the owner thread) and completes the futures. If no generation runs, the request takes the normal path.

Expected effect [assumed]: the wait drops from up to 51 s to at most one decode round (about 20 to 120 ms per verify at long context, per findings 97/98/113) plus the request's own prefill (about 0.5 to 0.8 s here). The long generation is delayed by that same prefill per interlude.

### Risks [assumed from reading, none tested]

- **Cache state.** The interlude must not touch the running generation's caches or the bank (no restore, no store, no eviction, no `put`). The stateless scoring route does this, but any shared object on the runtime would break it: a persistent MTP cache, the paged-KV pool, kernel scratch buffers, leases.
- **Compiled / graph state.** The compiled verify bank traces per shape and cache layout. A 240-row prefill is a different shape, so it should not retrace the verify graph, but the interlude must not run inside a compiled region, and the compile cache must not be invalidated (dsplit/wide kernels pick routes per query rows).
- **Pending async work.** `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS` and `async_eval` mean outstanding submits at some points; the hook must sit where they are settled (after commit it should be).
- **Memory.** A prefill of 240 to 290 rows adds transient activations on top of the running generation (the log shows peak 31.1 GiB against 24.2 GiB active for such a call; this machine has 64 GB, so it fits at 27B but not with a long context in the generation). The admission/memory guard would need to price the interlude.
- **Stats and observability.** `_runtime_counter_snapshot` deltas, round timings and the request log of the long generation would include the interlude's work unless separated.
- **Determinism.** The interlude changes timing but not the state of the generation, if the isolation holds; this needs a test: output of the long generation with and without interludes must be token-identical.
- **Starvation and fairness.** Many interludes can slow a long generation without bound; a cap per round is needed.

### Recommended next step

A GPU test that runs a hand-made hook (a patch in a scratch copy, not in the prod tree) with the scoring pass between rounds of a 900-token generation and checks (a) the generation is token-identical, (b) the interlude latency, (c) peak memory, (d) the compiled-verify trace counters stay unchanged. Alternatively accept the 51 s worst case for this traffic and run the classifier on a second small model/process, which was not evaluated here.

## What is measured and what is assumed

Measured: all tables in Part A (4 server runs ABBA plus one extra fresh-server run), the 41-token head, label equality, the logprob differences. Read from the code: the whole Part B mechanism description and the bank rules. Assumed: the fixed-cost explanation of the 0.5 s floor, all Part B effects and risks, anything about long-run bank behaviour.
