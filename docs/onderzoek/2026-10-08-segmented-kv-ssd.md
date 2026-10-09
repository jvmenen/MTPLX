# Segmented KV cache: SSD tier for segmented bank entries

Date: 2026-10-07/08. Status: implemented on local branches (`feat/segmented-kv` 598127b5, cherry-picked onto `prod-segkv`), measured; nothing pushed. Production tree, port 8000, `config/config.json`, `~/Dev/MTPLX` venv and `pkg-combo` untouched. Follows `2026-10-07-segmented-kv-m3.md` (section 5, open item "SSD tier").
Data and scripts: `/Users/joonix/Dev/laya-nl/segment-kv/m3/` (`SSD-DESIGN.md`, `ssd/` direct cold-tier benchmark and the pristine-codec check, `server/` run with `tables_ssd.md`, `raw-S*.jsonl`, `drive_ssd.py`, `run_ssd_all.sh`).

## 1. Design (short; full text in `SSD-DESIGN.md`)

Read first: the cold tier stores each entry as a `payload.json` plus content-addressed blobs (sha256); KV tensors of two blocks or more are `tensor_blocks`, one blob per 256 rows; `put_entry` stages encoded bytes behind `MTPLX_PERSISTENCE_MAX_PENDING_BYTES`, `spill_entry` streams entries above that budget; blobs are written to a temp file and renamed and the entry is installed last; blob lifetime follows the manifest rows. Both writers encode through `TreeCodec`.

Decision: the on-disk format does not change. The codec encodes a `SegmentedKVState` as the stock `(keys, values)` tuple of the same rows. A virtual tensor (`SegmentedRows`) gives the codec `shape`, `dtype`, `nbytes` and slices along axis 2: a slice inside one segment is a view, one across an edge is a concatenation of two small slices, so the history is never gathered (block fingerprints of the incremental encode are computed per chunk of 16 blocks). Consequences:
- Sharing: blocks are content-addressed, so segments and prefixes shared by forks and earlier turns are stored once, without lineage bookkeeping; rows are cut at global 256-row boundaries, so entries agree on the blocks of a shared prefix whatever their segment edges are.
- Eviction, refcounting, crash safety: the existing code, unchanged (temp file plus rename, entry installed last, manifest-derived blob lifetime).
- Restore: unchanged. The decoder returns stock arrays; `restore_cache` makes a layer above `segmented_kv_min_tokens()` one sealed segment. Restore does not rebuild several segments and does not use mmap (open item).
- Compatibility: an entry written from a segmented snapshot is a stock-format entry and restores with the switch off; a stock entry on disk restores with the switch on. Nothing needs refusing.
- Switches: all behind `MTPLX_SEGMENTED_KV`; `MTPLX_SEGMENTED_KV_SSD=0` restores the previous skip of segmented entries (`skip_reason segmented_kv`).

Code: `segmented_kv.py` (`SegmentedRows`, `segmented_state_tensors`, `segmented_ssd_enabled`), `cache_bank/codec.py` (one branch in `TreeCodec.encode`, chunked fingerprints for `SegmentedRows`), `session_bank.py` (the three skip guards now conditional).

## 2. Flag off is today (established)

- `m3/ssd/pristine_check.py` loads the codec of origin/main (9882703f) and the current one and encodes the same stock snapshot with and without incremental reuse: spec, tensor bytes and fingerprints are equal (38 tensors).
- `tests/test_segmented_kv_ssd.py::test_flag_off_leaves_the_stock_path_untouched`: a stock entry written through `put_entry` equals `encode_payload` of its fields (blobs and spec), a second write with the segmented codec branch replaced by a function that raises succeeds (the branch is never entered for stock tensors), and `segmented_ssd_enabled()` is false.
- The branch added to `TreeCodec.encode` is one `isinstance` test before the existing tuple case; no other flag-off code path changed.

## 3. Tests

`tests/test_segmented_kv_ssd.py` (10 tests): segmented snapshot encodes to the stock spec and bytes (with and without reuse, equal fingerprints and reused names); decode of the segmented encode equals the stock rows; `rows_slice` equals the concatenation across edges; slices inside a segment allocate nothing; tier files (payload.json minus timestamp, all blobs) of a segmented entry equal those of the stock entry for both `put_entry` and `spill_entry`; two entries that share their first segment store the shared blocks once; a stock entry on disk restores as segments with the switch on. `test_segmented_kv_bank.py` keeps the skip test under `MTPLX_SEGMENTED_KV_SSD=0`. Full suite on the feature tree: green on stock MLX and on `pkg-lse` (`m3/logs/suite-ssd-{stock,lse}.log`); on `prod-segkv` (after the cherry-pick, conflicts in CHANGELOG and the import block of `session_bank.py`): green on `pkg-lse` (`suite-prod-ssd.log`).

## 4. Direct cold-tier benchmark (established; `m3/ssd/ssd_bench.py`)

Real shapes (16 layers, 4 KV heads, head dim 256, bf16), turn 0 of 80,000 rows then turns of 1,024, 2,048, 3,072 and 4,096 rows (each a sealed segment in the segmented variant), a fork (first two turns plus its own 2,048 rows), a restore of the last entry; `spill_entry`, two runs per variant (ABBA), medians; `put_entry` at 40,000 rows, one run per variant. Random data, the SSD of the test machine, thermal 0.

| write | stock s | segmented s | disk added MB (both) |
|---|---|---|---|
| turn 0, 80K | 3.63 | 3.66 | 5247 |
| turn 1 (+1K) | 2.31 | 2.24 | 79 |
| turn 2 (+2K) | 2.37 | 2.27 | 146 |
| turn 3 (+3K) | 2.50 | 2.41 | 213 |
| turn 4 (+4K) | 2.62 | 2.51 | 280 |
| fork (2 turns shared + 2K) | 2.23 | 2.29 | 146 |
| restore of the last entry | 0.97 | 1.00 | disk total 6.11 GB (both) |

Disk bytes and file contents are the same for both variants (the dedupe of the content-addressed store works for the fork: 146 MB, not 5.4 GB). Write time is dominated by hashing every block of the history again (`spill_entry` has no incremental reuse); that is existing behaviour, equal for stock. `put_entry` at 40K: 2.54 / 2.55 s for turn 0, 0.71 to 0.97 s per later turn for both; MLX peak over the start of the write 0.84 to 1.06 GiB for stock against 0.10 GiB for segmented (the stock path fingerprints whole-tensor transposes, the segmented path 16 blocks at a time; supposition for the cause, the difference is measured).

## 5. Server run, eviction and restart (established for these runs; one pair of repeats per variant)

Same server setup as the m3 run (port 8010, launch command from `config.json`, `pkg-lse`, `prod-segkv`, own SSD directory per start), additionally `MTPLX_SESSION_BANK_MAX_BYTES=7G` so the bank cannot hold two long conversations. Order SA1 SB1 SB2 SA2 (A = switch off, B = on), GPU lock, thermal 0 at the start. Scenario: conversation `s50` (54K prompt, turns t0 to t3), 45 s idle, conversation `s80` (84K, t0 to t2), 60 s idle, follow-up `s50-t4` (after eviction from the bank), `s80-t3`; then the server is stopped and started again on the same SSD directory (warm SSD), and follow-ups `r50-t5` and `r80-t4` are sent. Full table `server/tables_ssd.md`.

| request | A (switch off) | B (switch on, B1 / B2) |
|---|---|---|
| s50-t4, after eviction | cold re-prefill, cached 4019 of 58522, TTFT 143.8 / 142.3 s, peak 41.4 GB | `ssd_clone`, cached 57859 of 58399, TTFT 14.3 / 13.7 s, peak 35.5 GB |
| s80-t3, after eviction | A1: 507 refusal (peak 42.9 GB); A2: cold, cached 4019, TTFT 247.9 s | `ssd_clone`, cached 86120 of 87658, TTFT 49.8 / 48.0 s, peak 37.3 GB |
| r50-t5, after restart | `ssd_clone`, cached 58807, TTFT 24.5 / 24.8 s | `ssd_clone`, cached 58690, TTFT 24.6 / 24.3 s |
| r80-t4, after restart | `ssd_clone`, cached 86090 / 88012, TTFT 70.4 / 21.0 s | `ssd_clone`, cached 88058, TTFT 20.6 / 20.9 s |

In A the evicted conversation was not restored from the SSD in this scenario: the request log shows `cache_source ram`, `ssd_cache_hit false`, cached tokens equal to the 4019-token head (cause not investigated; A's bank entries there came through the reference-lease path). B restored both from the SSD (`ssd_restore_s` 1.26 s and 2.31 s of the TTFT; the rest is suffix prefill and the restore into segments; the TTFT figures of 14 to 50 s were not decomposed further). After the restart, A1's 70 s against A2's 21 s for the same request shows run to run variation in A (not explained).

Disk (whole run, `du -sk` of the SSD directory, which includes orphan blobs not yet reclaimed): B 13.05 GiB before the restart (B1 and B2 alike) and 13.6 GiB after it, against A 17.75 GiB (A1) and 18.8 GiB (A2) before and 18.0 and 19.0 GiB after. Blob files at the end: B about 74,300, A 95,200 and 103,900. The tier's own counters at the end (`physical_bytes`, live manifest bytes): B 8.9 to 9.0 GB, A 7.2 to 8.2 GB; the larger A footprint on disk is orphan and untracked bytes (`orphan_file_bytes` 3.0 GB, `untracked_file_bytes` 7.6 GB at s80-t2 in SA1), the cause of which was not investigated. Writes completed per run (spill path): B persisted every segmented turn (7 writes to s80-t3), A 5 in the same span.

Peak memory over the requests: B 35.5 to 37.5 GB, A 41.4 to 43.1 GB on the eviction requests; on the restart requests equal (31.7 and 37.5 GB).

Write time and bytes per turn in the server could not be taken from the run log (the idle-lane writes carry no duration in the request log or `/health`); they are taken from the direct benchmark in section 4. Not measured: write time while a request is running (idle lane yield), write time for a sampled-temperature conversation.

## 6. Open items

- Restore into several segments or through mmap (restore concatenates blocks into one array, then one conversion copy above the fence).
- Incremental reuse for `spill_entry` (it hashes the whole history each turn, 2.3 to 2.6 s at 80K).
- TTFT after an SSD restore at 80K (49 s for 1.5K new tokens in B) is dominated by something other than the suffix prefill and the 2.3 s restore; not decomposed.
- The at-rest merge of the large history and `/health` reporting (from the m3 report) are still open.
- A's failure to restore from the SSD in this scenario, and the orphan bytes, are observations of the existing code that this work did not touch.

## 7. Follow-up round (2026-10-08): TTFT after an SSD restore, spill hashing, merge, /health

### 7.1 Why the first token took 14 to 50 s after an SSD restore (established; fixed behind the switch)

Measured in-process (`m3/ssdrun.py`: one process writes a 50K conversation through the bank and cold tier, a fresh process restores it; forward calls timed with synchronisation) and with a macOS `sample` of the server during the slow request.
- The restore is not the cost: decode of the blobs 0.5 to 1.0 s, conversion into a segment 0.14 s.
- The suffix prefill is: the first forward of 535 tokens took 10.56 s (1.83 s on the following, RAM-restored turn), the following 64-token forward 1.34 s (0.39 s). `sample` showed the server main thread 90 % of the time in `waitUntilCompleted`, i.e. the GPU is slow, not the host.
- The weights-eviction hypothesis is refuted: page-ins of the slow requests were 2.8 to 3.2 GiB (173k to 197k 16 KiB pages), about the size of the blobs read, not the 15.6 GB of weights; swap-ins were 16 to 507 pages; the RSS drop during the request is the bank evicting.
- Cause: the layer types after the restore (`adapt` log) were `VllmMetalPagedKVCache`. `SessionBank._restore_cold` called `runtime.make_cache()` and ignored the request's `cache_factory`, which every RAM restore uses to build the dense contiguous layout. The SSD-restored conversation therefore prefilled and decoded on the paged layout (about 5x slower prefill, 13 against 23 tok/s decode in the in-process run) and could not become segments. This is pre-existing and also happens with the switch off (A's `ssd_clone` requests show the same 14 to 17 tok/s).
- Fix (behind `MTPLX_SEGMENTED_KV`, `MTPLX_SEGMENTED_KV_SSD`): `_restore_cold` uses the request's cache factories. Test: `test_ssd_restore_builds_the_requests_cache_layout_only_with_the_switch` (factory used with the switch, not without).

Before and after, in-process (`p3` before, `p4` after; same data, 50K, 600-token follow-up after a restart): TTFT 14.22 s to 4.25 s; first forward 10.56 s to 1.92 s; layers after the restore: 16 paged caches to 16 segment caches; peak 32.3 to 27.6 GiB; decode of the following 64 tokens 13.2 to 19.6 tok/s.
Server (`SC1` before, `SC3` after, same scenario, bank cap 7G, switch on, prod-segkv): TTFT s50-t4 (after eviction) 16.3 s to 5.2 s; s80-t3 47.9 s to 8.3 s; after a restart r50-t5 25.4 s to 7.3 s, r80-t4 21.3 s to 7.2 s; decode of those requests 14.5 to 17.7 tok/s to 25.4 to 29.5 tok/s; peak process memory unchanged (35.5 GB). Not yet measured: the same scenario with the switch off after the same fix (that would need the fix outside the switch).

### 7.2 Spill without the full-history rehash (established)

`spill_entry` read every block of the history back and hashed it each turn. A sealed segment never changes, so the sha256 and size of each of its blocks lying inside one segment are kept on the segment (`KVSegment.block_digests`); the next spill references them after the tier has claimed the digest (so the orphan cleanup keeps it) and confirmed the blob file exists, and hashes only new segments and the block at each segment edge. A missing blob is rewritten. With the switch off no segmented rows exist and the codec path is unchanged.
- Flag-off identity: `m3/ssd/store_identity.py` writes stock entries through `put_entry` and `spill_entry` and digests the whole store (blobs, payload.json without timestamps): origin/main (9882703f, a separate worktree) and the current tree give the same digest `23a6efaf9c4cf333…` (94 blob files). `pristine_check.py` still reports equal specs and bytes.
- Tests: `test_spill_does_not_rehash_sealed_segments_it_already_hashed` (a second spill hashes under 45 % of the blocks and writes the same payload and blob digests as the stock entry of the same rows), `test_spill_rehashes_a_block_whose_blob_is_gone`.
- Write time per turn at 80K, direct benchmark (`ssd/ssd_bench.py`, 16 layers, real shapes, persistent per-layer caches that grow by 1,024 to 4,096 rows a turn, `spill_entry`, ABBA, median of 2; before = 598127b5, after = this round): turn 0 3.61 s / 3.68 s; turn 1 2.21 s / 0.25 s; turn 2 2.28 / 0.31; turn 3 2.43 / 0.43; turn 4 2.55 / 0.58. Disk bytes identical (5.97 GB), restore 0.98 / 1.06 s. The remaining time grows with the added rows (hashing the new segment and the edge blocks).

### 7.3 The tiered merge never ran, and now does (established)

Reason: `compact()` merged only segments with reference count 1, but every turn's bank snapshot holds that turn's segments, so nothing was ever alone and the `merge` counter never appeared. A conversation therefore gained two segments per turn (a prompt snapshot and the final snapshot each seal the tail) and would have passed the 12 segments of one fused launch after six turns (estimate from the seal counts: 32 seals per turn over 16 layers). Fix: small adjacent segments (combined at most 16,384 rows, the older at most twice the newer) merge also when snapshots hold them; the snapshot keeps its own pieces, the merged copy is the only new memory, the large history is never copied. Tests: a held pair merges and the snapshot stays bit-identical; 24 turns with two snapshots each (every snapshot kept) stay at 12 segments or fewer.
Model run, 12 follow-up turns of 1,024 tokens at 50K (`seg12`, one process): segments per bank entry 3, 3, 3, 4, 3, 4, 4, 3, 4, 4, 4, 5 after turns 1 to 12 (maximum 5, never near 12). Merge cost per turn: 5 to 23 ms in total over the 16 layers (16 to 48 merges, 19,584 to 231,808 merged rows summed over the layers); transient memory of one layer's merge 4.8 to 52 MiB. Peak over the turn start stays 0.9 to 1.5 GiB; ms per verify 94 to 119 (thermal noise; no trend). Conclusion for the open item: with the merge of held segments the at-rest merge of the large history is not needed for the 12-segment limit; it would only matter for the 50K history itself (not copied) and for the SSD granularity, which does not depend on segments.

### 7.4 /health (established)

With the switch on the session bank section of `/health` has `segmented_kv`: `enabled`, `min_tokens`, `ssd`, `prefill_route`, `entries` (session id, prefix length, segments, sealed bytes, live), `max_segments`, `sealed_bytes_unique`, `merges`, `seals`, `seal_copies`, `route_counts` and `ssd_counts` (`ssd_layer_states_encoded`, `ssd_blocks_reused`, `ssd_blocks_hashed`, `restored_stock_as_segment`). Absent with the switch off. Test: `test_health_block_is_absent_with_the_switch_off_and_reports_segments_with_it_on`.

### 7.5 Suites

Feature tree full suite: green on stock MLX and on `pkg-lse`; `prod-segkv` (cherry-picked, one CHANGELOG conflict resolved): green on `pkg-lse` (`m3/logs/suite-r4-*.log`). The temporary worktrees for the identity and before-measurements were removed.

## 8. Another model: Qwen3.6-35B-A3B (2026-10-08, established)

Model `Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` (local, 20 GB; MoE; 40 layers of which 10 full attention, 16 query / 2 KV heads = GQA 8, head_dim 256), production env of the m3 runs (the 27B launch environment, no FR-Spec), `prod-segkv` tree with `pkg-lse`, in-process with the real session bank and `generate_mtpk` (depth 3, context copy on, greedy), a 50K prefix and four follow-up turns (1,024, 2,048, 3,072, 4,096 new tokens, 200 decoded each, every turn continuing from the run's own output so every bank hit is exact). Order off on on off off on (one process per run), gate before each run, thermal 0 throughout. Medians of 3, `m3/out/q36-50000-*.json`.

| turn | prefill s off / on | decode tok/s off / on | ms per verify off / on | peak over turn start GiB off / on | peak GiB off / on |
|---|---|---|---|---|---|
| 1 | 0.9 / 0.9 | 83.8 / 107.5 | 27 / 27 | 2.68 / 1.65 | 24.1 / 23.1 |
| 2 | 1.7 / 1.7 | 62.9 / 62.6 | 25 / 22 | 1.93 / -0.50 | 24.6 / 21.2 |
| 3 | 2.7 / 2.6 | 70.9 / 68.7 | 28 / 26 | 2.15 / 0.59 | 25.0 / 21.5 |
| 4 | 3.7 / 3.5 | 50.7 / 54.9 | 25 / 22 | 2.43 / 0.72 | 25.4 / 21.7 |

Cold prefill of the 50K prefix (turn 0): 25.6 s off, 25.8 s on; decode 74.7 and 74.8 tok/s.

- Memory: the follow-up peak above the turn start is 1.0 to 2.4 GiB lower with segments from turn 2 on (a negative value means the peak stayed below the memory at the turn start, because the stock snapshot's duplicate was released first); the absolute peak is 3.0 to 3.7 GiB lower. Turn 1 pays the one-off conversion of the stock snapshot (1.65 against 2.68 GiB here, smaller than on the 27B because the model has 10 attention layers and 2 KV heads: 20 KiB of KV per token against 64 KiB).
- Speed: prefill equal within noise; ms per verify equal at turn 1 and 2 to 3 ms (10 %) lower from turn 2; decode tok/s differ by the noise of the greedy trajectories (these vary between 51 and 108 tok/s with the acceptance of the copied text). Thermal 0 in every run.
- Routes (on): `fused_q4` 3,140, `fused_q9` 290, `chunked_q25` 40, `fused_q1` 50, `fused_q2/q3/q12` 50 together, `prefill_lse` 40 (`sdpa_lse_kernel` 140), `merge` 60, `restored_stock_as_segment` 10. No `gather_qN` counter: no verify window and no prefill chunk gathered the history. GQA 8 limits the single-launch window to 4 rows (8 x 4 = 32 rows per KV head); windows of 5 to 8 rows run as sub-windows over the segments, 9 or more through the wide route (`fused_q9`, `fused_q12`).
- Correctness: both variants are deterministic run to run. On and off first differ at generated token 52, 22, 56 and 51 of turns 1 to 4. The top-1/top-2 margin of the base logits at the first difference (turn 1, position 52, cold stock prefill of prompt plus the common 52 tokens, `m3/margin2.py`) is 0.125: the two tokens have logits 17.125 and 17.0, one bf16 step at that magnitude, a tie. Margins for the later turns were not computed (their histories already differ, so they follow from the first difference).
- Not measured: 80K on this model, the SSD tier on it, sampled decoding.

## 9. Head_dim 128 on the AR path: Qwen3-8B-4bit at 22K (2026-10-08, measured)

Setup: `mlx-community--Qwen3-8B-4bit` (36 full-attention layers, 32q/8kv, head_dim 128, no MTP head), `generate_ar` with the session bank, prod-segkv tree, turbo profile env, 22K cold context plus 4 follow-up turns of 256 greedy tokens, ABBA (off on on off off on), medians of 3, thermal 0 throughout (`m3/arrun.py`, `m3/cmp_ar.py`, `out/ar-*-22000-*.json`). Context length is 22K because the model maximum is 40,960 tokens. Peak is `mx.get_peak_memory()` minus active memory at turn start, after `clear_cache` and `reset_peak_memory` (the same measurement as `modelrun.py`; the JSON key is `peak_over_gib`). Turn 0 is a cold prefill and uses stock caches in both variants (segments start at the first restore).

- Support and routes: model support verdict is "supported"; no `gather_*` route in any run. Decode runs on `fused_q1` (plain: 37,008 calls), context-copy windows on `fused_q8..q25`, `merge` and `restored_stock_as_segment` (36) as before. The head_dim 128 kernel therefore works end to end on a real model (kernel tests were already green).
- Plain AR (no context copy), ms per step off -> on, turns 1 to 4: 31.2 -> 31.2, 32.0 -> 31.2, 32.8 -> 32.6, 33.9 -> 33.6. Prefill equal (0.7-2.1 s). Peak over turn start (GiB) off -> on: 3.56 -> 3.94, 3.82 -> 1.08, 7.45 -> 1.60, 7.85 -> 1.41.
- Code edit with context copy (MTPLX_CONTEXT_COPY_AR=1), ms per step off -> on: 101 -> 70, 106 -> 67, 72 -> 50, 76 -> 52. Prose with context copy: turn 1 (no copying, 255 steps) 30.9 -> 30.8; turns 2 to 4 with copy windows 106 -> 66, 115 -> 71, 76 -> 39 (turn 4 on has 74 steps against 26 off because the trajectories differ). Peak over turn start code: 7.19 -> 4.60, 4.01 -> 1.14, 8.22 -> 2.21, 4.48 -> 1.21 GiB; prose turns 2 to 4: 3.83 -> 1.10, 7.49 -> 1.61, 7.87 -> 1.42 GiB. Why the stock path is slower on 9 to 25 row windows is not investigated (assumed: stock multi-row route at head_dim 128); only the measured difference is claimed.
- Fair comparison is ms per step, not tok/s: with context copy, tok/s depends on how much text is copied per step (generated 256 tokens in 12 to 74 steps), and the on and off trajectories diverge at ties and therefore copy different amounts. Do not read the tok/s columns as a speed ratio.
- Context-copy acceptance (measured, runs `out/arcc-*`; acceptance is identical off and on for code): code turns 1 to 4 drafted/accepted 241/241, 250/242, 294/219, 255/226 (1.00, 0.97, 0.74, 0.89). Prose on: turn 1 no context-copy rounds, turns 2 to 4 243/243, 243/243, 184/181. Turn 0 (cold, no copy source): 64 drafted, 12 accepted.
- Greedy identity off vs on: both variants deterministic run to run. Plain: first divergence at generated token 6, 2, 34, 31 (turns 1 to 4); prose: turn 1 token 60 onward; code: no divergence in any turn. Margin at the first divergence (base logits, `margin_ar.py`): plain turn 1 pos 6: tokens 374 vs 320 both 16.875, margin 0.0; prose turn 1 pos 60: 31918 vs 46494 both 32.75, margin 0.0. Both are exact bf16 ties. Margins for later divergences were not computed (histories differ from there).
- Not measured: contexts beyond 22K on this model, sampled decoding, verify windows via an MTP head at head_dim 128 (no such model locally).

## 10. Fence sweep: should the segment fence exist? (2026-10-08, Qwen3.8-27B, measured)

Setup: prod-segkv tree, production env, own mode (`m3/modelrun.py`, exact bank hits: cached tokens = previous prompt + 200 in every turn, all runs). A = flag off (stock cache, compiled verify as in production). B = `MTPLX_SEGMENTED_KV=1` with `MTPLX_SEGMENTED_KV_MIN_TOKENS=0` (segments from the start, eager verify). Contexts 8K, 16K, 24K, 32K; 4 follow-up turns (1024/2048/3072/4096 new tokens); ABBA (off on on off off on), medians of 3. 50K was not re-run: Jeroen decided that `g3` (section 2 of the earlier M3 report) covers it, with the caveat that g3 ran on the feature tree at 365810aa with the default fence 32K, so segments began at the first follow-up turn; same comparison (stock compiled vs segments eager). Data: `out/f8-*`, `python3 cmp.py f8 <ctx>`.

ms per verify, off -> on, turns 1 to 4:
- 8K: 79 -> 72, 80 -> 75, 101 -> 89, 82 -> 74
- 16K: 82 -> 75, 83 -> 76, 89 -> 78, 87 -> 78
- 24K: 85 -> 75, 84 -> 77, 92 -> 85, 88 -> 79
- 32K: 87 -> 80, 88 -> 82, 86 -> 80, 93 -> 86
- 50K (g3, feature tree): 109 -> 105, 108 -> 106, 122 -> 120, 104 -> 103

Segments are 3 to 12 % faster per verify at every context from 8K to 32K and equal to 4 % faster at 50K; no crossover where segments lose. Decode tok/s varies by 10 to 20 % between turns in both variants because greedy trajectories diverge at bf16 ties and acceptance differs (for example 8K turn 4: 26.7 -> 29.9, turn 3: 42.1 -> 36.8); ms per verify is the comparable number. Prefill is equal within 5 % at all contexts (8K turn 4 10.7 vs 10.6 s; 32K 12.3 vs 12.1 s; the g3 50K lse cost of +3..12 % is the one place segments are slower).

Peak over turn start (GiB), off -> on, turns 1 to 4: 8K 2.22 -> 1.23, 3.31 -> 1.28, 3.45 -> 2.28, 4.83 -> 1.01; 16K 3.24 -> 1.26, 3.80 -> 1.26, 3.94 -> 1.45, 5.32 -> 1.68; 24K 3.13 -> 0.87, 3.94 -> 1.42, 4.42 -> 1.61, 5.81 -> 1.63; 32K 3.63 -> 1.09, 4.77 -> 1.23, 4.92 -> 1.50, 6.33 -> 1.76. Segments lower the peak by 1 to 4.5 GiB at every context, and the benefit grows with turns. Active memory after the put is 1 to 2 GiB lower on.

Routes: in turns 1 to 4 no gather route at any context. Turn 0 (cold prefill, segments from the start because the fence is 0) shows `gather_qprefill` at all contexts, and at 8K additionally `gather_q1` 16, `gather_q4` 1,328 and `gather_q9` 32 (short-context cold decode; the gather is then over a small cache). These are only in the cold turn that the fence normally keeps on stock caches; they cost nothing measurable (8K turn 0 ms/verify 79 vs 79, 16K 88 -> 78). Thermal stayed 0 to 1 (2 once after a 50K cold prefill).

Recommendation: (a) no fence. Segments are never slower per verify in this data, use less peak memory, and a fence adds an environment variable, a conversion at the boundary (one-off about 5 GiB transient at 50K to 80K) and a compiled-versus-eager switch for no measured benefit. Caveats (not measured): contexts below 8K, sampled decoding, and that compiled verify is lost with segments (the measured eager cost here is more than offset); if the default is ever set to 0 the cold-turn gather at very short contexts should be checked once, otherwise leave the fence at 0 when segments are on.

### Async chunk layers with segments on (MIN_TOKENS=0)
Question: does `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` (production) still matter with segmented eager verify, now that the strided-copy bug is fixed? a8 = 8, a0 = unset; ABBA a8 a0 a0 a8 a8 a0, medians of 3, 4 follow-up turns (`out/as-*`, `m3/cmp_as.py`).
- 16K ms/verify a8 vs a0 (turns 1 to 4): 74 vs 91, 76 vs 92, 78 vs 89, 78 vs 84; tok/s 23.9 vs 19.7, 22.8 vs 19.0, 23.8 vs 21.0, 23.3 vs 21.8.
- 50K ms/verify: 88 vs 92, 87 vs 92, 99 vs 103, 87 vs 92; tok/s 30.7 vs 29.5, 27.7 vs 26.6, 33.9 vs 32.8, 21.6 vs 20.7. Prefill and peak identical in both.
- Result: async chunking still helps with segments, by 6 to 19 % per verify at 16K and 4 to 5 % at 50K; keep it at 8 (production setting). The old finding that async made segmented verify slower no longer holds. Cause of the gap at 16K not investigated.
- Side observation: with segments from the start on the prod-segkv tree, 50K runs at 87 to 99 ms/verify against 103 to 120 in g3 on the feature tree (older code, different tree, not a controlled comparison).

## 11. Decision: one path, no fence; verification run (2026-10-09, measured)
Jeroen approved option (a). With `MTPLX_SEGMENTED_KV=1` the full-attention layers are segment caches from the first token; the fence, the `MTPLX_SEGMENTED_KV_MIN_TOKENS` setting and the stock-below-the-fence plus conversion code are removed (feature tree 653fe78b and 26660492, prod-segkv e15cab29 and 5a0229e0). Flag off is unchanged: the store digest of `store_identity.py` is the same as before (23a6efaf...) and `pristine_check.py` reports spec, tensors and fingerprints equal.

Gather counters: the cold-turn `gather_*` routes were calls over a single segment, where the gathered view is the segment's own rows (no copy). They now count as `single_q*` (the stock route on a zero-copy view, as the stock cache does); `gather_q*` means a real concatenation. In addition, calls over several segments take the fused kernel at any length (the packed-route length threshold applies to one-segment caches only), which removes the real gathers that short follow-up turns (below 8K) had.

Verification, Qwen3.8-27B, prod-segkv, flag on, no fence setting, one run each (not ABBA), own mode:
- 8K: no `gather_*` in any turn; cold turn: `single_qprefill` 64, `single_q1` 16; ms/verify 84, 86, 89, 104, 91 (turns 0 to 4; f8 on: 79 to 89 for turns 0 to 4 of the same series); bank hits exact in all 4 follow-ups (cached = previous prompt + 200); peak above turn start 1.26, 1.29, 2.23, 1.19 GiB in turns 1 to 4 (f8: 1.23 to 2.28).
- 50K: no `gather_*` in any turn; cold turn `single_qprefill` 240; ms/verify 100, 102, 100, 112, 101 (g3 on: 105, 106, 120, 103; async check on: 87 to 99); cold prefill 149 s with thermal level 2 at the end of turn 0 (earlier runs 127 s); bank hits exact; peak 0.90, 1.11, 1.58, 1.89 GiB in turns 1 to 4.
- Not measured: ABBA repeats of this build (single runs are within the run-to-run spread of earlier tables but cannot show differences below about 5 %).

## 12. Candidate branch and the no-LSE fallback (2026-10-09, measured)

Candidate: branch `prod-segkv-candidate` in worktree `~/Dev/laya-nl/mtplx-prod-candidate`, created from `prod` (85d4a49a, tag prod-2026-10-08, which already carries the AR context-copy commits 23468b07 and 85d4a49a) with the 19 segmented-KV commits of `prod-segkv` cherry-picked in order (CHANGELOG conflicts resolved with `cl_merge.py`; the two AR context-copy cherry-picks of `prod-segkv` are skipped because `prod` has them). Cherry-picking the already adapted `prod-segkv` commits (not the `feat/segmented-kv` ones) avoids redoing the prod-specific conflict resolutions (DSPLIT4/WIDE/TG_M kernel, session_bank imports); the resulting tree equals `prod-segkv` except for the CHANGELOG. `~/Dev/MTPLX-prod` was not touched.

Suites (candidate tree, full `tests`): stock MLX wheel: 0 failures. `pkg-combo-0323` (MLX 0.32.3 combo with return_lse): 8 failures, all outside segmented KV (`test_hc_verify_read` 1, `test_moe_sorted_gather` 3, `test_qwen4_m4_routed_glu` 1, `test_qwen4_route_kernel` 2, `test_release_pins::test_installed_mlx_is_the_pinned_version`). The same 8 fail on plain `prod` (detached worktree at 85d4a49a) with `pkg-combo-0323`, so they come from MLX 0.32.3 (and the pin test from the version pin), not from this branch; they need the MLX build owner or a pin update, not a change here.

### Behaviour on an MLX build without return_lse (stock wheel), flag on
- Feature detection: `sdpa_lse_available()` probes once per process (a tiny `scaled_dot_product_attention(..., return_lse=True)`); False on the stock wheel (measured: `prefill_route: gather`, route counter `gather_qprefill` 160 in the 50K run, no `sdpa_lse_*`).
- Prefill route: `gather` (one contiguous copy of the history per layer and prefill chunk, then the stock SDPA). Decode and verify windows are unaffected: they use the segment kernel, which lives in MTPLX, not in MLX.
- Before: no log line; /health already reported `prefill_route`. Now: one log line at load ("MTPLX_SEGMENTED_KV is on, but this MLX build has no logsumexp output ..."), once per process (feature 398f394a, candidate and prod-segkv cherry-picks; tests in `test_segmented_kv_support.py`).

### Cost of the gather prefill (Qwen3.8-27B, candidate tree, flag on, stock MLX wheel `s` vs pkg-combo-0323 `c`, ABBA s c c s s c, medians of 3, `out/fb-*`, `m3/cmp_fb.py`)
Prefill seconds, turns 0 (cold) to 4 (turn 1 to 4 add 1,024 / 2,048 / 3,072 / 4,096 new tokens):
- 16K: s 35.7, 3.0, 5.7, 8.6, 11.7; c 33.7, 2.9, 5.5, 8.3, 11.0. Follow-up turns: +3 to +6 %.
- 50K: s 135.3, 4.2, 6.8, 10.4, 14.7; c 134.7, 3.5, 6.5, 10.0, 13.9. Follow-up turns: +20 % (turn 1, 0.7 s absolute), +5 %, +4 %, +6 %; sum of follow-up turns 36.1 vs 33.9 s (+6.5 %). The cold turn is equal (a single segment never needs the gather).
- ms/verify: equal within noise (16K 74 to 82 vs 74 to 92; 50K 86 to 102 vs 93 to 104). Peak above turn start: equal (16K 1.23 to 1.63 vs 1.26 to 1.68 GiB; 50K 0.98 to 1.77 vs 0.90 to 1.77 GiB). Thermal 0 to 2.
Conclusion: the fallback costs about 5 % of follow-up prefill (6.5 % over the four turns at 50K), below the 10 % bar, with identical decode speed and memory. Option (a) accept and document, plus the log line, is implemented; option (b) (auto-disable segments without LSE) is not, because it would give up the memory and snapshot benefits (1 to 4.5 GiB lower peak, no 5 GiB first-write copy) for a 5 % prefill cost. Not measured: contexts above 50K (the gather copy grows with the history, so the cost per chunk grows with it), prefill chunk sizes other than 4096.
