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
