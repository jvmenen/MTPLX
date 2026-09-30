# Host overhead per decode round on Qwen3.6-35B-A3B (30 September 2026)

Question: during decode on Qwen3.6-35B-A3B, how long per MTP round does the GPU sit idle waiting for host (CPU) work, and is there a realistic gain from overlapping or removing it? On Qwen3.8-27B this was 1.5 to 2 ms of an ~88 ms round (~2%, `~/Dev/laya-nl/qwen38/verify-analyse.md`). The MoE rounds are about three times shorter, so the same host milliseconds weigh more.

## Summary

- **Host time with an idle GPU is 2.7 to 3.2 ms of a 26 to 30 ms round: 9.4 to 11.2%** (measured directly, all workloads, with and without FR-Spec). On top of that, about 1 ms more sits hidden inside the blocking verify eval (inferred, see §3). Realistic total: ~14%.
- **The largest piece is building the verify graph: ~1.95 ms per round** (2.1 ms at 14K context). The Balance pack is 6-bit, so the compiled verify is gated off (`quant_bits_gate:bits=6`) and every round rebuilds the 40-layer graph in Python while the GPU waits. The rest: ~0.5 ms host work between the target-row evals, ~0.4 ms around the two draft steps.
- **Fix built and measured: `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8`** (branch `perf/host-overlap-36`, off by default). It submits the verify graph to the GPU every 8 layers with `mx.async_eval`, so the GPU works on the first layers while the host builds the rest. Tokens are identical in every run (all workloads, with and without FR-Spec). In-process decode: **+10 to +14%** (reasoning 94.0 to 103.7 tok/s, code 107 to 121, 14K context 82.3 to 90.7). **On the real server (production arguments, port 8001, bench36 workload): +4.7%** (round 25.95 to 24.68 ms, 2 runs each); the server keeps only half the in-process gain, cause not yet found (§5).
- For comparison, the existing `MTPLX_COMPILED_VERIFY_FORCE=1` gives +3.6% in the same setup and is not bit-identical at long context (report 26 Sep). The async chunk gives three times as much, stays bit-identical, and needs no compiled graph.
- **Recommendation:** put `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` in the Bink server env after a longer agent-workload check (bit-identical, +5% on the server now), and find out why the server keeps only half of the in-process gain; closing that gap is worth another ~5%. It is also a good upstream PR: small, bit-identical, and it helps every model whose verify runs eager (6-bit packs, and every model beyond the compiled-verify context limit of 32K).

## 1. Setup

- **Hardware:** M5 Pro, 64 GB. Swap stable at ~5.6 GB during all runs (no growth).
- **Model and code:** pack `Qwen3.6-35B-A3B-MTPLX-Balance-bf16mtp-yb`, depth 2, turbo profile, worktree `~/Dev/MTPLX-definitief` (a5df61d0). Bink production env: `MTPLX_BATCH_INVARIANT_PREFILL=1 MTPLX_MTP_HISTORY_CACHE_ONLY=1 MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096 MTPLX_PREFILL_CHUNK_SIZE_REPAGE=4096 MTPLX_A3B_MOE_PREFILL_COMBINE=1 MTPLX_SESSION_HEAD_ANCHOR=1 MTPLX_THINKING_BUDGET=6144`. FR-Spec as in production (`MTPLX_FRSPEC_DRAFT=1`, `bink-64k.npy`, `MTPLX_FRSPEC_LEGACY=1`) and without FR-Spec.
- **Method:** in-process tune candidate (`mtplx tune --_candidate 2`, the same approach as the Qwen3.8 analysis), sampled decoding at temperature 0.6 / top-p 0.95 / top-k 20 for target and draft, seed 0. The tune path runs the same verify strategy as the server (`capture_commit`, `linear-gdn-from-conv-tape`) and installs the 4-bit draft head plus FR-Spec the same way. A profiling hook (`prof_hook.py`, extended from the 3.8 tools) times every `mx.eval` / `mx.async_eval` / `np.asarray(mx.array)` per call site, plus the verify dispatch and each draft step, and optionally records a timeline. `tl36.py` cuts the timeline into rounds.
- **Workloads:**
  - `code`: the tune default suite (183 prompt tokens, 512 tokens, thinking off);
  - `reasoning`: one English explanation prompt with thinking on (41 prompt tokens, 1,024 tokens of reasoning-style output);
  - `reasoning 2.8K` and `reasoning 14.2K`: the same kind of question after a Dutch knowledge-base text (2,812 and 14,215 prompt tokens, 768 tokens, thinking on).
- **What "host with idle GPU" means here:** the round time minus the time the host spends blocked in an eval. MLX runs lazily: nothing reaches the GPU until an eval, so time outside evals is time with an empty GPU queue (there are no `async_eval` calls in the eager round). This is an upper bound on idle time outside evals, and a lower bound on total idle, because each eval also contains a host-side preamble (graph sort, encode) before the GPU starts (§3).
- **Every run under the GPU slot**, with a check on port 8000 before each model load and every 5 s during the run (no server appeared). Model loads take ~10 s; one tune run 15 to 30 s.
- **Noise:** the baseline ran 91.1 to 94.9 tok/s over the same reasoning workload (5 runs); differences under ~3% are not demonstrated.

## 2. Timeline per round (baseline, measured)

Mean ms per MTP round (depth 2: one verify of 3 tokens, then 2 draft steps). 175 to 421 rounds per row. Tokens per round: 2.36 to 2.81.

| Workload | Verify build (host, GPU idle) | Verify eval (GPU) | Target-row evals (GPU) | Host between row evals | Draft evals (GPU) | Host around drafts | Round | Host with GPU idle | Share |
|---|---|---|---|---|---|---|---|---|---|
| code, FR-Spec | 1.98 | 18.30 | 2.79 | 0.56 | 2.72 | 0.45 | 26.81 | 2.99 | 11.2% |
| code, no FR-Spec | 1.96 | 18.27 | 2.96 | 0.55 | 4.16 | 0.39 | 28.29 | 2.91 | 10.3% |
| reasoning, FR-Spec | 1.96 | 18.26 | 2.47 | 0.50 | 2.63 | 0.40 | 26.23 | 2.86 | 10.9% |
| reasoning, no FR-Spec | 1.94 | 18.12 | 2.44 | 0.47 | 3.98 | 0.35 | 27.29 | 2.76 | 10.1% |
| reasoning 2.8K, FR-Spec | 1.82 | 18.88 | 3.00 | 0.48 | 2.64 | 0.36 | 27.19 | 2.66 | 9.8% |
| reasoning 2.8K, no FR-Spec | 1.86 | 18.87 | 2.82 | 0.46 | 4.05 | 0.33 | 28.40 | 2.66 | 9.4% |
| reasoning 14.2K, FR-Spec | 2.16 | 20.05 | 2.51 | 0.53 | 2.96 | 0.47 | 28.68 | 3.17 | 11.0% |
| reasoning 14.2K, no FR-Spec | 2.13 | 19.97 | 2.50 | 0.53 | 4.30 | 0.39 | 29.82 | 3.05 | 10.2% |

Where each segment lives (`MTPLX-definitief`):

| Segment | Code | Per round |
|---|---|---|
| Verify build | `graphbank.py:3308` `forward_ar_capture` → `graphbank.py:4723` `_fallback` (permanent eager, reason `quant_bits_gate:bits=6`, gate at `graphbank.py:1878-1889`) → `runtime.py:280` → `gdn_capture.py:2998` `forward_with_gdn_capture` (layer loop `:3031-3092`) | 1 call, ~1.95 ms, no eval inside |
| Verify eval | `generation.py:2344` `_eval_verify_outputs`, eval of the hidden state at `:2355` (lazy logits, `MTPLX_LAZY_VERIFY_LOGITS=1`) | 1 eval, ~18.3 ms |
| Target-row evals | `fast_sampling.py:421` (`mx.eval(cand_idx, cand_vals, cand_probs)` in `_device_serial_support_arrays`) from `sparse_distribution_from_mlx_logits` `:598`, via `generation.py:6807` / `:6865`. The first row also runs the lazy verify lm_head (~2 ms); later rows ~0.35 ms each | 2.4 to 2.8 evals |
| Host between row evals | acceptance, commit of the captured prefix (`gdn_capture.py:3107`), MTP history append (`generation.py:11279`), host reads at `fast_sampling.py:422-427` | ~0.5 ms |
| Draft evals | `runtime.py:339` `draft_mtp` (graph, ~0.08 ms per step) and the same `fast_sampling.py:421` eval via the draft reader (`generation.py:7073` / `:7057`) | 2 evals: 1.3 ms each with FR-Spec, 2.0 ms without |
| Host around drafts | draft graph build plus host sampling | ~0.4 ms |

Observations:

- FR-Spec changes only the draft evals (2.6 to 3.0 ms instead of 4.0 to 4.3 ms per round). Host time is the same with and without; its share rises slightly with FR-Spec because the round gets shorter.
- Context length barely moves the picture: at 14.2K the verify eval grows by ~1.8 ms and the build by ~0.2 ms.
- The verify build dominates the host time: ~2/3 of the measured idle time.

### What the 1.95 ms verify build consists of (cProfile, profiled times scaled by 0.49)

cProfile over one reasoning run (profiled build 4.0 ms against 1.95 ms without profiler), per round:

| Part | Calls per round | Share of build | Estimate unprofiled |
|---|---|---|---|
| 30 GDN layers (`gdn_capture.py:2582`, of which input projections `:1632`, conv capture `:1673`, delta-from-tape capture `:1851`) | 30 | 41% | ~0.8 ms |
| 40 MoE blocks (`a3b_moe_prefill_combine.py:76` → mlx-lm `qwen3_next.py:327`, router, shared expert, `switch_layers`) | 40 | 40% | ~0.8 ms |
| 10 attention layers (`attention_split.py:200`) | 10 | 12% | ~0.25 ms |
| Of the above: the quantized-linear wrapper chain `batch_invariant_prefill.py:236` → `_in_prefill` `:181` → `nax_verify.py:754` → `QuantizedLinear` | 393 | ~20% wrapper overhead | ~0.4 ms |
| Of the above: `os.environ` reads on the hot path (`gdn_capture.py:22`, `:2582`, `:2998`, `:1851`, `attention_split.py:18`, `a3b_moe_prefill_combine.py:45`) | ~410 | ~8% | ~0.15 ms |

No single piece stands out; the cost is spread over ~400 module calls and a few thousand MLX op calls. Shaving the Python costs directly (caching env reads, flattening wrappers) would save perhaps 0.5 ms (~2%). Overlapping the build with GPU work hides almost all of it.

## 3. Hidden host time inside the verify eval (inferred)

The async-chunk experiment (§4) shortened the round by 2.4 to 2.9 ms, more than the 1.95 ms Python build. The simplest explanation: `mx.eval` on a 40-layer graph first sorts and starts encoding thousands of nodes on the host before the first command buffer reaches the GPU. With chunked submits that preamble also overlaps with GPU work. A two-point fit (baseline verify 22.3 ms = build 1.9 + preamble P + GPU time; chunked 19.9 ms ≈ GPU time + first chunk build ~0.4 + P/5) gives **P ≈ 1.1 ms and a pure GPU verify of ~19.3 ms**. Not measured directly (MLX exposes no GPU start timestamps), so treat it as an estimate. With it, total GPU idle time per round is ~4 ms (~14 to 15%).

## 4. Candidates

| # | Candidate | Estimated gain | Measured | Exact? | Status |
|---|---|---|---|---|---|
| A | `mx.async_eval` on the residual stream every N layers in the eager verify (`MTPLX_VERIFY_ASYNC_CHUNK_LAYERS`) | build 1.95 ms + part of the preamble → ~2.5 ms (~9%) | **+10 to +14% in-process**, all workloads; **+4.7% on the server** (§5) | bit-identical (tokens identical in every run) | built, `perf/host-overlap-36` (7467558b) |
| B | `MTPLX_COMPILED_VERIFY_FORCE=1` (existing switch) | build → compiled dispatch | +3.6% (96.1 against 92.7 tok/s); round 26.2 → 25.3 ms | not bit-identical at long context (report 26 Sep) | exists; A is better |
| C | Trim Python overhead in the verify build (cache env reads, flatten the `batch_invariant_prefill` → `nax_verify` → `QuantizedLinear` chain) | ~0.5 ms (~2%); with A mostly hidden anyway | not built | bit-identical | not worth it after A |
| D | One eval for all target rows instead of 2.4 to 2.8 per round (`fast_sampling.py:421`) | saves ~1.5 eval round-trips, ~0.2 to 0.3 ms (~1%) | not built | bit-identical if the host half is unchanged | small |
| E | Device-side draft chain (sample the draft on the GPU, build the next graph while it runs), as candidate A of the 3.8 analysis | ~0.4 to 0.8 ms (2 to 3%) | not built for 3.6 | token-identical by design | larger job; 3.8 gave 0.5 to 1% |

The blocking variant of A (`MTPLX_TARGET_LAYER_EVAL_EVERY=10`, existing, uses `mx.eval`) gives nothing (91.9 against 92.7 tok/s): the host then waits for each chunk and builds the next one on an idle GPU again. The overlap only works with `async_eval`.

### Chunk size (in-process via the hook, `definitief`, reasoning workload with FR-Spec, 3 runs each)

| Layers per submit | tok/s (3 runs) | Mean | vs baseline |
|---|---|---|---|
| baseline (off) | 94.87 / 91.07 / 91.27 | 92.4 | |
| 20 (2 chunks) | 99.36 / 98.36 / 98.84 | 98.9 | +7.0% |
| 10 (4 chunks) | 104.30 / 104.24 / 101.03 | 103.2 | +11.7% |
| 8 (5 chunks) | 104.81 / 104.71 / 99.79 | 103.1 | +11.6% |
| 4 (10 chunks) | 104.50 / 105.23 / 100.96 | 103.6 | +12.1% |

4 to 10 layers per submit are equivalent; 8 is the chosen default for the switch (5 submits for 40 layers).

### Across workloads (async every 8 layers via the hook, `definitief`, 2 runs each)

| Workload | Baseline tok/s | Async 8 tok/s | Gain | Tokens |
|---|---|---|---|---|
| code, FR-Spec | 104.1 / 109.8 | 121.2 / 121.5 | +13.6% | identical |
| reasoning, no FR-Spec | 89.9 / 89.8 | 98.9 / 98.0 | +9.6% | identical |
| reasoning 2.8K, FR-Spec | 91.8 / 91.9 | 95.9 / 101.0 | +7.3% | identical |
| reasoning 14.2K, FR-Spec | 81.8 / 84.7 | 83.9 / 93.0 | +6.3% | identical |
| reasoning 14.2K, no FR-Spec | 75.7 / 81.3 | 81.3 / 89.8 | +9.0% | identical |

The long-context rows are noisier (one run of each pair was slow in both variants).

## 5. Prototype and measurement

**Code** (worktree `~/Dev/MTPLX-host36`, branch `perf/host-overlap-36` from `origin/main` 1de2b1c0, commit 7467558b, local):

- `mtplx/gdn_capture.py`: `_verify_async_chunk_layers()` reads `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS` (default off; `0`/`off`/invalid = off). In `forward_with_gdn_capture` the layer loop calls `mx.async_eval(hidden_states)` after every N layers, never after the last one, and only for decode-sized windows (T ≤ `MTPLX_TARGET_LAYER_EVAL_MAX_Q`, default 8; prefill is untouched). When the existing blocking layer-eval schedule applies it wins.
- `tests/test_verify_async_chunk.py`: a tiny Qwen3.5-MoE model (8 layers, GDN plus attention, 8 experts); logits, hidden state and all GDN captures bit-identical with the switch on for N = 2, 3, 8, the expected number of submits (3, 2, 0), no submits for a prefill-sized window, env parsing. All 269 tests in the 14 test files that touch `gdn_capture` pass; ruff reports no new findings.

**In-process, prototype on `main`** (the hook not active, only the env switch; 3 runs each for reasoning):

| Workload | Off | `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` | Gain |
|---|---|---|---|
| reasoning, FR-Spec | 94.36 / 93.77 / 93.94 | 103.93 / 103.53 / 103.58 | +10.4% |
| reasoning 14.2K, FR-Spec | 82.34 | 90.72 | +10.2% |
| reasoning, no FR-Spec | 88.28 | 97.25 | +10.2% |

Verify per round 22.3 → 19.9 ms; draft and acceptance unchanged; tokens identical in every pair.

**Server (production path, port 8001):** fresh server per run from `~/Dev/MTPLX-host36` with the full production arguments (`mtplx-originele-args.txt`, Balance-yb pack, `--port 8001`), Bink env plus FR-Spec, own SSD cache directory; workload `bench36.py` from the FR-Spec measurement (11 requests: code, English and Dutch reasoning, Dutch without thinking; 800 to 1,200 tokens). Server sampling is not seeded, so the texts differ between runs; ms per round (decode time / verify calls) is the fair comparison. All 11/11 requests, no tracebacks, swap flat.

| Run | code | en | nl | nl, thinking off | All: ms per round | All: tok/s |
|---|---|---|---|---|---|---|
| base 1 | 25.65 | 25.66 | 25.88 | 25.40 | 25.69 | 94.0 |
| base 2 | 25.69 | 25.86 | 26.98 | 26.33 | 26.21 | 90.7 |
| async 8, run 1 | 23.19 | 24.72 | 25.55 | 25.04 | 24.58 | 96.7 |
| async 8, run 2 | 24.87 | 24.58 | 25.00 | 24.53 | 24.78 | 96.8 |

On the server the round drops from 25.95 to 24.68 ms (-4.9%) and decode rises from 92.4 to 96.8 tok/s (**+4.7%**). That is half the in-process gain (-1.3 ms per round against -2.4 ms). The baseline round is the same in both setups (~26 ms), so the server loses part of the overlap with the switch on. Not investigated; the most likely cause is the server's other threads (asyncio event loop, SSE per token at `--stream-interval 1`) taking the GIL while the decode thread sits in `mx.async_eval`, which delays the build of the next chunk. Next step to confirm: the same A/B with a larger `--stream-interval`, or a timeline inside the server process. Base 1 overlapped briefly with a unit-test run (tiny model); base 2 is clean and the slower of the two, so that does not flatter the result.

## 6. Conclusion and recommendation

- On Qwen3.6-35B-A3B the host share is five times larger than on Qwen3.8-27B: ~11% measured, ~14% including the eval preamble, against ~2%. The cause is the eager verify of the 6-bit pack: ~2 ms of Python graph building per round, with nothing on the GPU.
- Hiding that build behind the GPU with chunked `async_eval` gives ~10% decode in-process and ~5% on the real server, bit-identical, with a 30-line change. That clears the 3% bar.
- Next steps:
  1. Find the server gap (larger `--stream-interval` as a test, or a timeline inside the server).
  2. Longer check on the Bink workload (agent turns at 20K to 60K context, where the verify is longer and the gain in relative terms smaller), then `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` in the server env.
  3. Upstream PR (small, off by default, bit-identical). Also relevant for other eager-verify cases: 6-bit packs of other models and contexts above `MTPLX_COMPILED_VERIFY_MAX_CONTEXT` (32K), where Qwen3.8 and the 4-bit packs also fall back to eager.
  4. Remaining host time after A: ~0.9 ms around acceptance and drafts plus the first chunk's build (~0.4 ms), together ~5%. Candidates D and E attack that part; each gives at most 1 to 3%.
- Side finding: `MTPLX_TARGET_LAYER_EVAL_MODE` is in `EXTERNAL_RUNTIME_ENV_KEYS` (`commands/public.py:238`) but no code reads it. A dead switch, probably left over from an earlier async variant.

## Material

`~/Dev/laya-nl/host36/`: `run1.zsh` (one tune run), `job.zsh` (GPU slot, port 8000 and swap checks), `prof_hook.py` (profiling hook, including the `PROF_LAYER_ASYNC` experiment and `PROF_THINKING`), `tl36.py` (per-round timeline), `summ.py`, `srv.zsh` + `bench36_8001.py` (server A/B on port 8001), prompt suites `reason*.jsonl`, raw results in `res/` (`cand-*`, `prof-*`, `*.tl.json`, `*.pstats`, `srv-*.json`) and logs in `logs/`.
