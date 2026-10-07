# Segmented KV cache, milestone 3: boundary, model runs, integration with prod, server run

Date: 2026-10-07. Status: measurement and implementation on local branches; nothing pushed. Production tree, port 8000, `config/config.json`, the `~/Dev/MTPLX` venv and `pkg-combo` were not touched.
Data and scripts: `/Users/joonix/Dev/laya-nl/segment-kv/m3/` (`modelrun.py`, `margin.py`, `cmp.py`, `out/`, `logs/`, `server/` with `tables.md`, `raw-*.jsonl`, `admission-*.jsonl`, `server-*.out`). Branches: `feat/segmented-kv` (worktree `laya-nl/mtplx-segkv`), `prod-segkv` (worktree `laya-nl/mtplx-prod-segkv`, `fork/prod` plus the feature commits). Earlier reports: `2026-10-07-segmented-kv-prototype.md`, `2026-10-07-sdpa-lse-prefill.md`, `2026-10-07-lease-vs-clone.md`.

## 1. Boundary for the compiled verify (step 1)

The segment cache cannot be promoted to the compiled verify (`promotion_failure:segmented_kv_cache`), so with the switch on a verify runs eager. Decision (VONDSTEN 113): segments stay on above the compiled-verify fence, verify above the fence runs eager, compiled-verify compatibility is not worked on now.

Design (`mtplx/segmented_kv.py`):
- `segmented_kv_min_tokens()` returns `MTPLX_SEGMENTED_KV_MIN_TOKENS` when set, otherwise the compiled-verify ceiling `MTPLX_COMPILED_VERIFY_MAX_CONTEXT` (32768 in the turbo profile, 6144 without), read per call; 0 when the compiled verify is off, effectively infinite when its ceiling is 0. A layer is segmented when its rows are strictly above the boundary (the comparison the compiled verify uses to fall back).
- Cache creation: the target cache's plain `KVCache` layers are tagged segmentable and stay stock (boundary above 0). MTP caches are never tagged, so they are never converted.
- After a prefill (`repage_target_prefill_cache` re-runs the install) a stock layer above the boundary becomes one sealed segment. At a bank restore (`restore_cache`) the layer takes the layout its rows call for: segmented snapshot above the boundary stays references, a stock snapshot above it becomes one sealed segment, a segmented snapshot below it is gathered into a stock cache. A conversation that crosses the boundary therefore changes layout at its next restore.
- Segments made from stock rows own exact row-contiguous storage (see section 3, a strided view cost 34 ms per verify).
- Tests: `tests/test_segmented_kv_boundary.py` (boundary resolution, install, conversion bit-identical to the stock cache, restore in all four directions, extension and trim after restore, untagged caches untouched, no history copy by conversion plus snapshot, exact-row storage). All segmented-KV tests pass on stock MLX and on `pkg-lse`.

Established while doing this: the "32K fence" does not hold for the dense adapter. `_last_context_estimate` stays 0 for entries without `capacity`, so in production (and in `lease-meting/raw-A1`) the compiled verify ran for every call at 50K and 80K (`compiled_calls` equal to all calls). The fence therefore marks where segments start, not where compiled verify stops. The cost of verify going eager: stock eager is 116.4 ms per verify against 112.9 compiled at 50K (3 runs each, about 3 %).

## 2. Model run, in-process (step 2)

Setup: Qwen3.8-27B Optimized-Speed, production launch env, `apply_profile_env("turbo")`, `pkg-lse` (`mx.__file__` checked), real `SessionBank` (clone restore, `commit` after every turn), `generate_mtpk` (depth 3, capture_commit, `linear-gdn-from-conv-tape`, context copy on, greedy), prefix of 50K or 80K corpus tokens plus 4 follow-up turns (1024, 2048, 3072, 4096 new tokens, 200 decoded tokens each; each turn continues from the run's own output, so every bank hit is exact). One process per run, order off on on off off on (three per variant), gate before each run (no process above 2 GB, thermal 0, 60 s rest). Switch on = segments, prefill through `return_lse`. Medians of 3. Route counters confirm `sdpa_lse_kernel` was used (864 calls per run), no verify window fell back to a gather.

Feature tree (`feat/segmented-kv` at 365810aa, which has no wide and no dsplit4 kernels; those envs are inert there). Turn 0 is the cold prefill. Per turn, off then on:

| ctx | turn | prefill s | decode tok/s | ms per verify | peak over turn start GiB | peak GiB |
|---|---|---|---|---|---|---|
| 50K | 1 | 3.4 / 3.5 | 26.7 / 25.5 | 109 / 105 | 4.8 / 4.9 | 27.8 / 27.9 |
| 50K | 2 | 6.3 / 6.5 | 22.9 / 22.0 | 108 / 106 | 5.9 / 3.1 | 29.0 / 26.1 |
| 50K | 3 | 9.7 / 10.6 | 22.8 / 21.5 | 122 / 120 | 6.1 / 3.2 | 29.3 / 26.3 |
| 50K | 4 | 13.5 / 15.2 | 19.9 / 19.5 | 104 / 103 | 7.5 / 4.6 | 30.9 / 28.0 |
| 80K | 1 | 4.1 / 4.1 | 17.6 / 17.9 | 115 / 106 | 6.7 / 6.9 | 31.7 / 31.8 |
| 80K | 2 | 7.7 / 7.6 | 24.3 / 25.0 | 113 / 105 | 7.9 / 3.2 | 32.9 / 28.2 |
| 80K | 3 | 11.9 / 11.7 | 19.2 / 18.0 | 111 / 104 | 8.0 / 3.3 | 33.2 / 28.4 |
| 80K | 4 | 23.2 / 16.6 | 15.9 / 16.5 | 121 / 115 | 9.4 / 4.7 | 34.8 / 30.0 |

Cold prefill (turn 0): 125.8 / 134.6 s at 50K (thermal 2 in the on runs), 241.9 / 239.7 s at 80K; identical peak (the prefill path is the stock one until the repage). Active memory after each turn is identical (23.0 to 23.7 GiB at 50K, 24.9 to 25.6 at 80K), so the saving is a transient: peak over the turn start drops by 2.8 to 4.7 GiB from turn 2 on (47 to 63 % at 80K, 36 to 59 % at 50K). The 80K turn 4 prefill difference (23.2 s off) came with thermal 1 in an off run (supposition: thermal, not the mechanism).

Conversion copy on turn 1: the snapshot of the cold prefill is a stock snapshot; restoring it above the boundary makes it one sealed segment with exact rows. That is 64 copies (16 layers, K and V, per restore) in 0.18 s (80K), and it costs one history copy once (+6.9 GiB peak at 80K, +4.9 at 50K, the same as the off variant's copy-on-write duplicate that turn). From turn 2 on there is no history copy. It cannot be avoided without a segmented prefill; open item.

Decode and prefill: 80K, per verify 6 to 8 % faster (supposition: fewer cache buffers resident; not isolated), tok/s within trajectory noise; 50K: ms per verify equal within 1 to 4 %, tok/s -2 to -6 % because the greedy trajectories differ (different acceptance). Prefill with the lse route is equal to +12 % over 50K (more segments, six per conversation by turn 4), equal at 80K.

Output: both variants are deterministic run to run. On and off differ at the first differing token at positions 5, 82, 176, 72 (50K turns 1 to 4) and 158, 53, 57 (80K turns 2 to 4). Margins were measured on the 4000-token smoke run only: the three differing positions there are exact bf16 ties (top-1/top-2 margin 0.0, 0.125, 0.0), as in the phase 1 report. Margins at 50K and 80K were not measured (`margin.py` exists; each case is a cold prefill); that these are ties as well is a supposition. A tie flips with one bf16 ulp, which the merge and the lse route produce (phase 1: one ulp from the contiguous kernel).

### Findings during the runs (before and after)

1. First 50K ABBA with the feature commit as it was (medians of 3): decode 12 to 19 % slower, 127 to 131 ms per verify against 109. Diagnosis: not a host sync (`MTPLX_VERIFY_ASYNC_CHUNK_LAYERS` unset gave the same gap; the Python-side time of the route is 35 ms per turn; no `.item()` or eval on the route) and not a gather (counters); the timed 4-row forward was 96 ms stock and 130 ms segmented. Cause: a segment made from a stock buffer was a strided view (capacity beyond the rows); MLX copies a non-contiguous kernel input at every launch, 34 ms per verify. Fix: the segment owns exact row-contiguous storage (`365810aa`). After: 105 against 109 ms. Microbenchmarks (`kbench.py`, `cbench.py`, `fwd.py`) of kernel, cache management and whole forward are equal to stock for contiguous segments.
2. Windows of 6 to 32 rows (context-copy blocks) used a per-call gather of the whole history. They now run as sub-windows of the rows the kernel takes, over the segments (`chunked_qN` in the route counters); the integration tree fuses the wide windows in one launch.
3. Bank hits that looked like `block_prefix_boundary_clone` with 200 extra prefill tokens came from the harness (prompts built from reference output); with each variant continuing from its own output every hit is exact (`clone`).
4. A bug the first real model run exposed: the kernel's logsumexp is `(B, Hq, Q, 1)`, the merge expects `(B, Hq, Q)` (fixed, test with the real kernel added).

## 3. Integration with prod (step 3a)

`prod-segkv` = `fork/prod` (7eab5dd4) plus the feature commits, cherry-picked. Conflicts: `CHANGELOG.md` (kept the prod text, added the segmented-KV bullet), `session_bank.py` (both import blocks), `kernels/sdpa_nax_flash_dsplit.py` (the template list now has TG_M, NDH and NOMASK; the source line `(NOMASK != 0 || gp <= row_limit)` merged; the existing call sites pass NOMASK 0 and are unchanged). `kernels/sdpa_segmented.py` was adapted to the new dispatcher: `_route_shape` mirrors `_dsplit_dispatch` (halves or quarters by `MTPLX_NAX_FLASH_DSPLIT4` and the `nax_flash_dsplit4_sdpa` lane, TG_M row groups, the wide route for 9 to 32 rows with `MTPLX_NAX_FLASH_WIDE`), the block counts follow the dsplit4 and wide tables, a single segment calls the contiguous entry point and is bit-identical (tests for q 1 to 5 in halves and quarters, and wide q 9 to 32). Multi-segment outputs are within 3 bf16 ulps of the contiguous kernel and of the fp32 reference.

Tests on `prod-segkv`: the segmented-KV suites (kernel, attention, bank, boundary) pass on stock MLX and on `pkg-lse`; the full suite (`tests/`, 610 files) ran green, no failures, with `pkg-lse` and the integration tree (`m3/logs/suite-prod.log`). Nothing had to be fixed after the cherry-picks beyond the conflicts above.

## 4. Server run (step 3b)

Setup as in `2026-10-07-lease-vs-clone.md`: launch command from `config.json` verbatim except port 8010, a private SSD cache directory per start, `PYTHONPATH` = hook directory, `pkg-lse`, the integration tree; A = switch off, B = `MTPLX_SEGMENTED_KV=1`; order A1 B1 B2 A2, each its own server start, GPU lock and gate per start, stopped after each. A hook (`server/sc/sitecustomize.py`) logs the admission pricing and the route counters on every admission call. Scenarios `c50` (54K prompt, 5 follow-ups, fork with the same head, return to turn 2, one follow-up) and `c80` (84K), session header `x-mtplx-session-id`, temperature 0, max_tokens 400, thinking off. Thermal level was 1 to 2 throughout (long cold prefills), for all four variants alike; absolute speeds are therefore below the in-process numbers. Full per-request table: `server/tables.md`.

Per variant, means over follow-up turns t1 to t5:

| scenario | variant | TTFT s | prefill s | decode tok/s | ms per verify | peak GB (max) |
|---|---|---|---|---|---|---|
| c50 | A1 / A2 | 3.20 / 3.10 | 3.12 / 3.02 | 26.2 / 27.4 | 97 / 93 | 33.9 / 33.9 |
| c50 | B1 / B2 | 3.24 / 3.15 | 3.14 / 3.05 | 26.1 / 27.4 | 91 / 86 | 32.3 / 32.3 |
| c80 | A1 / A2 | 4.14 / 3.95 | 3.92 / 3.78 | 20.8 / 23.6 | 120 / 105 | 41.7 / 39.0 |
| c80 | B1 / B2 | 3.99 / 3.68 | 3.89 / 3.58 | 22.3 / 25.5 | 110 / 95 | 37.8 / 37.8 |

Admission, memory and refusals (all requests, per variant): A1 had one 507 (c50 fork) and 8 admission sheds, A2 had 8 sheds, B1 and B2 had no 507 and no shed. The admission gate priced `restore_copy_bytes` 3.2 to 5.75 GB on 12 of the A requests and 0 on every B request; `growth_bytes` for the c80 follow-ups 11 to 13 GB (A) against 6.6 to 7.2 GB (B). Peak process memory over the whole run: 41.7 / 39.0 GB (A) against 37.8 GB (B). Active memory after follow-ups at c50: 31.7 to 32.8 GB (A) against 25.3 to 28.3 GB (B); at c80: 36 to 38 (A) against 33.4 to 34.6 (B).

Fork and return (the cases the bank change is for):
- c50 fork: A1 refused with 507; A2 restored 49152 of 54047 tokens by reference lease with a 3.19 GB copy; B1 and B2 restored the same 49152 tokens by lease with no copy (TTFT 17.1 and 16.1 s, dominated by the 5K token suffix and a shed-free admission).
- c80 fork: A1 and A2 re-prefilled the whole 84049-token prompt cold (304.7 and 257.8 s TTFT; the older entry was gone); B1 and B2 restored 82958 tokens (TTFT 6.1 and 4.8 s).
- c80 return to turn 2: B 15.2 to 15.3 s (82958 cached, boundary clone), A1 11.7 s (lease), A2 21.7 s (SSD restore, 8.6 tok/s, 254 ms per verify).
- Restore modes in B for ordinary follow-ups are `clone` (exact); the cold tier is not used for segmented entries (section 5), which is visible in the fork and return rows (no `ssd_*` modes in B).

Output: the c50 and c80 texts of the first turn are identical for A and B; from turn 1 on, B differs from A in every turn, and B1 equals B2, A1 equals A2 token for token (deterministic). I did not measure the margins for the server texts; given section 2 these are supposed to be bf16 ties, not established. Routes in B: all 16 layers' verify windows served by the segment kernel (`fused_q4` 24,976, `fused_q9` 592, further wide windows `fused_q21` to `fused_q30`, `fused_q1` 272, `prefill_lse` 736, `sdpa_lse_kernel` 2480, no `gather_qN`, no kernel bails).

Limits of the comparison: one pair of repeats per variant, one scenario family, thermal 2, a different history in the bank per start (A hit evictions the B starts did not), so A's 507 and sheds and the cold c80 fork also reflect the bank budget at that moment. The mechanism measured in section 2 (no history copy from turn 2 on, a transient of 3 to 5 GB less) is what the admission pricing and the peak figures show here.

## 5. Open items for the rest of milestone 3

- SSD tier: a segmented snapshot is not persisted (`skip_reason segmented_kv`), so after eviction a conversation above the boundary re-prefills; the 256-token block hashing maps onto segments but the encoder needs a segment-aware path and the restore must load into sealed buffers.
- Merge of the large history in rest: the tiered merge only merges small unshared segments; folding the history and the recent tier in the background when idle (reference count 1) is not built. Seal copies of the tail (512 per full run) and the one-time conversion copy at the first restore remain.
- `/health`: report the switch, the boundary, route counters, segment count and bytes, and the conversion events.
- Not covered by the measurements: compiled verify on segments (decided not now), head anchors taken mid-request on a stock live cache above the boundary (supposition: the old copy-on-write duplicate returns for that case), margins of the 50K, 80K and server divergences, 128K and above, the 20-segment two-launch case in a model run.
