# Async prefill rungs with GDN states (3a) and spike reserve (3c): server measurements, 9 Oct 2026

Code: MTPLX prod tree, tag prod-2026-10-09b (read-only). Server: the production launch command from config.json, port 8010, throwaway SSD cache dir, MLX build pkg-combo-0323. Scripts and raw data: `~/Dev/laya-nl/opt-3a-3c/` (`run_one.py`, `chain.sh`, `analyze_a.py`, `analyze_b.py`, `raw-*.jsonl`, `poll-*.jsonl`). One server process per run, Qwen3.8-27B Optimized-Speed (4-bit, 64 layers).

## 3a: `MTPLX_PREFILL_ASYNC_RUNGS_STATES=1`

### What it does

MLX builds a whole prefill forward lazily. With `MTPLX_PREFILL_ASYNC_RUNGS=<stride>` the hidden state is dispatched every `stride` layers, so the GPU starts earlier. A GDN layer however also leaves its conv tail (a copy of the last rows of its pre-conv stream), its delta state and any boundary captures on its cache entry. Those stay lazy until the end of the forward, so every GDN layer's pre-conv stream stays alive until then. With `STATES=1` each rung also names the recurrent states of the layers since the previous rung, so each stream is freed with its layer. Same graph, same kernels: only the moment of dispatch changes.

### Setup

- Stride 4. Reason: it is the stride Flash-Next uses for its mid-loop eval, the one the earlier K4 measurements (28 Sep) used, and the PR text proposes; stride 1 would add a dispatch per layer, stride 4 gives 16 rungs over 64 layers.
- Variants: X = production flags; Y = X + `MTPLX_PREFILL_ASYNC_RUNGS=4`; Z = Y + `MTPLX_PREFILL_ASYNC_RUNGS_STATES=1`. Order X Y Z Z Y X, six server processes.
- Per process: a 3K warm-up, then four cold prompts (16K, 50K, a second distinct 16K, a second distinct 50K; 16,047 and ~50,000 tokens), greedy, 200 new tokens, thinking off. Before each prompt `/admin/cache/clear` (it also resets MLX's process-wide peak counter), so `peak_memory_bytes` is a true per-request peak.
- Process memory: sampled `phys_footprint_bytes` from `/v1/mtplx/snapshot` (every 2 s, so a sampled maximum, not exact).

### Results (Qwen3.8-27B, mean of 4 requests per cell, per-request values in brackets)

| Prompt | Variant | Peak (`peak_memory_bytes`, GiB) | Sampled process footprint (GiB) | Prefill time (s) | TTFT (s) |
|---|---|---|---|---|---|
| 16K | X | 25.14 (all four) | 29.4 | 34.9 (33.7 to 35.6) | 35.0 |
| 16K | Y | 25.33 (all four) | 28.7 | 34.6 (33.5 to 36.8) | 34.7 |
| 16K | Z | 22.90 / 23.91 (a/b prompt, repeatable) | 28.8 | 34.2 (33.1 to 36.5) | 34.3 |
| 50K | X | 28.28, 28.23 | 34.6 | 130.2 (127.4 to 135.2) | 130.3 |
| 50K | Y | 28.36, 28.31 | 34.8 | 131.8 (129.4 to 135.2) | 131.9 |
| 50K | Z | 27.42 (all four) | 33.7 | 128.0 (126.5 to 128.7) | 128.2 |

- Peak: Z is 0.84 GiB lower than X at 50K (-3.0%) and 1.2 to 2.2 GiB lower at 16K (-4.8% and -8.9%, depending on the prompt). Hidden-only rungs (Y) do not lower the peak; they are 0.1 to 0.2 GiB higher. The effect of Z is deterministic (identical across the repeats).
- The 0.84 GiB at 50K is close to the claimed 0.87 GiB per 2,048-row forward; here forwards are 4,096 rows, so the saving was not doubled. The larger 16K saving depends on the prompt and is not explained; I did not investigate it.
- Prefill time: Z is 2% (16K) and 1.7% (50K) faster than X, Y within 1% of X; all inside the run-to-run spread (range of one cell is about 5%). No speed claim is supportable. Decode speed is unchanged (28.8 to 29.1 and 25.8 to 26.0 tok/s).
- Greedy output: identical for X, Y and Z on all four prompts (one distinct output per prompt across six processes).
- Sampled footprint: Z about 0.9 GiB lower at 50K, but the sampling is coarse and the 16K cells do not show it clearly (29.4 vs 28.8 vs 28.7). Treat the request-log peak as the number; the footprint only confirms nothing grew.

### Measured vs assumed

- Measured: peak, prefill time, TTFT, decode speed, output identity, on Qwen3.8-27B, cold 16K and 50K.
- Not proven directly: that the rungs ran in Y/Z. The exit receipt counters (`[prefill-rungs] exit receipt`) were not captured because the server is stopped with SIGTERM and the atexit handler did not write. Engagement is inferred from the peak difference between Y and Z and from the unit tests that cover it.
- Not measured: Qwen3.6-35B-A3B (the shapes the commit quotes). Local models were available (Optimized-Balance, 28 GB, 6-bit), the runs were queued after the 27B work in the chain, but the coordinator stopped the chain before they started. The 0.87 GiB claim therefore remains a synthetic number for A3B; the 27B result is the only server evidence.
- Not measured: the effect on long contexts beyond 50K, and with segmented KV on.

### Conclusion 3a

Does it help on a real server: yes for memory, not for speed. About 0.8 GiB (3%) lower peak at 50K, 1.2 to 2.2 GiB at 16K on a dense Qwen3.8-27B, bit-identical output, no measurable change in time. Risk: low (same graph, same outputs on four prompts; a different dispatch order only). The earlier K4b measurement (28 Sep) had cold +4.4% and agent +3.3% slower; this run does not reproduce a slowdown (Z is 1 to 2% faster, within noise), but that earlier test was a different workload (agent runs), so a mixed agent workload was not retested here.

Recommendation: do not turn it on in production for speed. Turning it on for memory headroom is optional and cheap (two env variables) if the peak matters; production currently has room (peak 28 GiB at 50K against a 48 GiB limit). Filing the PR is reasonable, framed as a memory option, not a speed option.

Numbers for "Results" in the PR (Qwen3.8-27B, M-series 64 GB, MLX 0.32.3 build, prod flags, cold prompts, greedy, per-request peak after `/admin/cache/clear`, stride 4, 4 requests per cell):

- Peak with `STATES=1`: 22.9 / 23.9 GiB vs 25.1 GiB (16K), 27.4 vs 28.3 GiB (50K).
- Hidden-only rungs: 25.3 and 28.3 to 28.4 GiB (no reduction).
- Prefill time: 34.2 vs 34.9 s (16K), 128.0 vs 130.2 s (50K), within spread; output identical to no rungs on all four prompts.
- Qwen3.6-35B-A3B not measured on a server.

## 3c: `MTPLX_SESSION_BANK_SPIKE_BURSTS` (dropped)

The option was dropped by Jeroen after these measurements; this section only records the result.

Workload (Qwen3.8-27B, bank cap at production default 25.7 GiB, no segmented KV): a cold 68K turn, a 40 s pause, 24 different sessions of 4 to 8K tokens (first turn, 15 s gaps so each is its own burst), a follow-up turn on each, then a follow-up on the 68K session and one more on a small session. Runs: off, N=2, N=2, off (order A B B A). The N=4 run was cancelled.

| Run | Ceiling after the deep turn, mean (GiB) | Ceiling at idle samples, mean / min | Bank bytes max (GiB) | Small follow-ups | Follow-up TTFT mean | Deep follow-up |
|---|---|---|---|---|---|---|
| off 1 | 20.3 | 20.5 / 14.0 | 19.7 | 24 of 24 from SSD | 3.06 s | refused (error) |
| N=2 a | 20.8 | 20.9 / 12.5 | 22.7 | 24 of 24 from SSD | 3.08 s | SSD restore, 18.7 s TTFT |
| N=2 b | 20.4 | 20.6 / 12.7 | 23.8 | 24 of 24 from SSD | 3.06 s | refused (error) |
| off 2 | 19.4 | 19.6 / 13.6 | 21.9 | 24 of 24 from SSD | 3.07 s | SSD restore, 17.9 s TTFT |

- No difference in ceiling, cache source (all small follow-ups were SSD restores, no RAM hits, no cold re-prefills in any run) or TTFT (3.1 s). The refusal of the 68K follow-up occurred in one off run and one N=2 run: it depends on machine memory pressure at that moment, not on the option. No HTTP 507 was seen; the refusal arrives as a stream with `finish_reason: error` after a `prefill_admission_shed` guard action.
- The implied reserve (derived from the ceiling, limit, weights and working set, so an estimate) was lower with N=2 (median about 4 GiB vs 8 to 10 GiB with the option off), so the mechanism does forget the deep spike. It did not translate into a higher ceiling or more retained entries here, because the working set during requests and system memory pressure (guard actions `memory_pressure_critical`, `prefill_shed_before_abort` occurred in one N=2 run and one off run) dominate the ceiling on this 64 GB machine.
- Side effect to note: with the option on, `peak_memory_bytes` means "since the engine was last idle".
- Conclusion: no measurable benefit; do not file; do not enable.

## 3a on Qwen3.6-35B-A3B (added later on 9 Oct)

Model: Youssofal Qwen3.6-35B-A3B Optimized-Balance (6-bit, 40 layers, 28 GB). Same harness, prompts and method as the Qwen3.8 runs: X = production flags without rungs, Z = X + `MTPLX_PREFILL_ASYNC_RUNGS=4` + `MTPLX_PREFILL_ASYNC_RUNGS_STATES=1`; order X Z Z X, one server process per run, four cold prompts per run (two 16K, two 50K), greedy, 200 tokens, `/admin/cache/clear` before each prompt. Raw data: `~/Dev/laya-nl/opt-3a-3c/a3b-final/` (and `raw-B3-a3b-*.jsonl`).

Note on the baseline: `config.json` had both rung variables switched on at 16:51 on 9 Oct, so a first A3B series (labels A-a3b-*, kept in `first-config-had-flags/`) ran with the flags on in both X and Z and is void as a comparison. The scripts now strip both variables from the launch command for X. The void series agrees with the final Z (same peaks), which doubles as a repeat of Z.

### Proof that the rungs ran

A small sitecustomize in my own directory (`opt-3a-3c/sc/`, not in the prod tree) wrote `mtplx.prefill_rungs.COUNTERS` to a file every 3 s. Final values per Z process: installed 1, wide_layer_calls 1,320, rungs_dispatched 330, state_arrays_dispatched 2,520 (both Z runs). Both X processes: installed 0, all counters 0. So the hook was installed and fired in Z, and not in X. (Not a change to /health; this is a measurement-side probe.)

### Results (mean of 4 requests per cell; each value identical across repeats)

| Prompt | Variant | Peak (`peak_memory_bytes`, GiB) | Sampled footprint (GiB) | Prefill time (s) | TTFT (s) | Decode (tok/s) |
|---|---|---|---|---|---|---|
| 16K | X | 30.30 | 34.0 | 6.0 | 6.1 | 75.7 |
| 16K | Z | 28.83 | 33.5 | 6.0 | 6.0 | 75.7 |
| 50K | X | 31.35 | 35.7 | 25.4 | 25.5 | 68.8 |
| 50K | Z | 30.08 / 29.98 (a/b prompt) | 35.6 | 25.3 | 25.4 | 68.9 |

- Peak: -1.47 GiB (-4.9%) at 16K, -1.27 and -1.37 GiB (-4.1% and -4.4%) at 50K. Larger than the 0.87 GiB of the synthetic 2,048-row check, because production prefill chunks are 4,096 rows.
- Speed: no difference (prefill 6.0 vs 6.0 s and 25.4 vs 25.3 s, spread under 1%; decode equal).
- Greedy output identical in X and Z on all four prompts (one distinct output per prompt across four processes).

### Does the K4b slowdown (+3 to 4% on an agent workload) apply?

From the data in hand: not on cold prefill. With states on, Qwen3.8-27B prefill was 2% (16K) and 1.7% (50K) faster than without, within the spread, and A3B showed 0 to 0.4%; neither shows a slowdown. An agent workload (many short, partly cached turns, where rungs fire on small forwards too) was not run here, so the earlier +3 to 4% neither is reproduced nor refuted there. Production now has the variables on, so agent TTFT in real use is the check to watch.

### Numbers for "Results" in the PR

Qwen3.6-35B-A3B Optimized-Balance (6-bit), 64 GB Mac, MLX 0.32.3 build, prod flags, cold prompts, greedy, stride 4, per-request peak after `/admin/cache/clear`, 4 requests per cell:

- Peak 28.83 vs 30.30 GiB (16K, -4.9%) and 30.08 / 29.98 vs 31.35 GiB (50K, -4.1% / -4.4%).
- Prefill 6.0 vs 6.0 s (16K) and 25.3 vs 25.4 s (50K); decode 75.7 vs 75.7 and 68.9 vs 68.8 tok/s.
- Output identical to no rungs on all four prompts.
- Engagement receipt: 330 rungs and 2,520 state arrays dispatched over 1,320 wide layer calls per process (0 in the baseline).

Updated conclusion for 3a: on the model it targets the saving is 1.3 to 1.5 GiB per request at no measurable time cost, with identical output. The PR is justified as a memory option.
