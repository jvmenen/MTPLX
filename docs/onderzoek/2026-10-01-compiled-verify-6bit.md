# Compiled verify on the 6-bit Balance pack, stacked with the async chunk (1 October 2026)

Follow-up to [compiled-verify-6bit](2026-09-26-compiled-verify-6bit.md) and [host-overhead-qwen36](2026-09-30-host-overhead-qwen36.md). Question: now that `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` exists, does admitting 6-bit to the compiled verify add anything, and is it correct?

## Why 6-bit is gated off

Caution, not a defect. The gate was added in 2.0.0 (f2b27c03, 4 July) with the comment "unmeasured quantizations stay eager"; 2.0.1 added the 6-bit split-K kernels and repeated that compiled verify stays off for 6-bit, with no measurement or error behind it. No test, changelog or mistakes entry names a kernel or numerical limit. `MTPLX_COMPILED_VERIFY_FORCE=1` already bypassed it.

## Change

Branch `perf/compiled-verify-6bit` (fork, based on upstream main 1de2b1c0): opt-in `MTPLX_COMPILED_VERIFY_ALLOW_BITS` (comma-separated bit widths) extends only the bits allowlist. Default unchanged. Nothing else broke: the compiled path ran 3115 of 3115 calls with no fallback and no exception streak. Commits 3d4faf10, b56b450b.

## Correctness

Greedy (temperature 0), 32 streams: 24 short (Dutch and code included), 6 at 1.7K to 5.8K, 2 at 7.8K and 15.7K prompt tokens. Eager against compiled, and against compiled with the async chunk: 32 of 32 token-identical, acceptance equal (0.7328 short, 0.8495 mid, 0.7968 long). Async chunk alone: 32 of 32 identical. `MTPLX_COMPILED_VERIFY=parity2` (eager clone compared bit for bit each call): 3115 calls, 0 divergent. The 26 September divergence at ~16K was a multi-turn chat, not reproduced by the single 15.7K prompt here.

Control, 4-bit Speed-yb (compiled by default) against `MTPLX_COMPILED_VERIFY=0`: 31 of 32 identical; one Dutch short prompt diverges at token 116 of 254 (acceptance 0.7448 against 0.7422). So the 6-bit compiled path is not noisier than the admitted 4-bit one; it was bit-exact here.

## Measurements

Qwen3.6-35B-A3B Balance-yb, depth 2, FR-Spec, thinking, 11 requests of 800 to 1200 tokens (harness `~/Dev/laya-nl/host36/bench36_8001.py`, runner `~/Dev/laya-nl/cv6/run.zsh`), tree = branch plus `perf/verify-async-chunk`. Round = decode time / verify calls.

| Variant | Run 1 tok/s | Run 2 tok/s | Round ms (1 / 2) | Route | Thermal level seen |
|---|---|---|---|---|---|
| eager | 91.6 | 92.5 | 25.83 / 25.74 | 0 compiled | 0 / 1-2 |
| eager + async chunk 8 | 101.3 | 101.8 | 24.04 / 23.46 | 0 compiled | 1-2 / 1-2 |
| compiled (`ALLOW_BITS=6`) | 94.3 | 94.9 | 25.14 / 25.15 | 3115/3115 | 0 / 2 |
| compiled + async chunk 8 | 96.2 | 96.8 | 24.62 / 24.40 | 3115/3115 | 1-2 / 2 |

4-bit Speed-yb control, one run each: eager 111.9, compiled 115.8 tok/s (+3.5%).

- Compiled gives +3%; the async chunk gives +10% and wins. They do not stack: the switch is a no-op in the traced graph, and the +2% of compiled + async over compiled alone is within thermal noise (those runs saw level 1-2). Reason for the small compiled gain not established; the per-call verify time is unchanged (20.0 against 19.7 ms), the saving is only the host graph build.
- Peak memory identical: 33.63 GiB with the 16K prefill, 29.91 GiB in the bench-only runs; compiled bank adds no visible memory. Swap flat (one run +0.9 GB, an async-2 run, unrelated to the variant; abort threshold 2 GB).

## Recommendation

Do not enable compiled verify for 6-bit in production; keep the async chunk. The gate stays as it is, the opt-in lives in the fork branch only.
