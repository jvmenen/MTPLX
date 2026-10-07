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
