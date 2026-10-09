# SSD restore layout fix: measured effect without segmented KV (2026-10-09)

Change: `SessionBank._restore_cold` builds the request's cache layout (`cache_factory`) instead of `runtime.make_cache()`, which for long context is the slower paged KV cache. Commit 26959449 (cherry-picked as cc72d6ac on `test/prod-ssdfix`, based on `prod`). Unit tests (`test_ssd_restore_cache_layout.py`, session bank, ssd_spill, boundary_repersist): all passed.

Setup: Qwen3.8-27B MTPLX Optimized Speed, production launch command, `MTPLX_SEGMENTED_KV` unset, session bank cap 7G, own SSD cache dir per run, one server per run, order P F F P (Xp1, Xf1, Xf2, Xp2). P = current prod, F = prod + fix. Scenario: 50K and 80K conversations, bank cap forces eviction to SSD, then the server is restarted with the SSD cache warm and each conversation continues one turn ("after restart"). KV layout is read from the request log (`paged_kv_num_blocks` > 0 means paged).

## Results: after server restart, warm SSD (restore from SSD)

| Step | Run | Cached / prompt | TTFT s | Prefill s | Decode tok/s | Peak GB | KV layout |
|---|---|---|---|---|---|---|---|
| 50K | P1 | 58828/59874 | 24.4 | 21.2 | 14.9 | 31.6 | paged |
| 50K | P2 | 57989/59563 | 35.6 | 32.7 | 16.1 | 31.7 | paged |
| 50K | F1 | 58828/59874 | 6.9 | 3.7 | 22.9 | 27.9 | dense |
| 50K | F2 | 57989/59563 | 8.7 | 5.3 | 23.1 | 28.0 | dense |
| 80K | P1 | 86110/88176 | 67.4 | 62.5 | 13.9 | 37.4 | paged |
| 80K | P2 | no SSD restore (see note) | 293.2 | 263.0 | 24.1 | 37.9 | dense (cold) |
| 80K | F1 | 86110/88176 | 13.1 | 8.4 | 24.9 | 32.0 | dense |
| 80K | F2 | 87985/88525 | 7.4 | 2.6 | 26.6 | 36.0 | dense |

All SSD restores used mode `ssd_clone`. Reference (RAM restores in the same runs): 24.7-28.8 tok/s.

Summary (SSD restore after restart, P vs F): 50K TTFT 24-36 s -> 7-9 s; 80K TTFT 67 s -> 7-13 s; decode 14-16 tok/s -> 23-27 tok/s; peak memory 3-6 GB lower at 50K. The layout after an SSD restore is paged in P and dense in F in every run.

## After eviction, before restart (same process)

| Run | 50K step | 80K step |
|---|---|---|
| P1 | no SSD restore: cold re-prefill of 58529 tokens (cached 4019), 143 s | request error |
| F1 | no SSD restore: cold re-prefill (cached 4019), 146 s | request error |
| P2 | request error | SSD restore: TTFT 49.3 s, 14.7 tok/s, paged |
| F2 | request error | SSD restore: TTFT 9.3 s, 24.1 tok/s, dense |

This step is erratic in both trees (requests ending in `error`, or no SSD restore at all); it is not caused by the fix and was not investigated. Where an SSD restore did happen (80K, runs 2) the effect matches the after-restart step.

## Measured vs assumed

- Measured: TTFT, prefill time, decode tok/s, peak memory, restore mode and KV layout above, from the request log of each run.
- Assumed: the speedup comes from the layout (dense vs paged); the layout column is measured, the causal link follows from the fix and the RAM-restore reference, not from a separate isolation test.
- Not covered: segmented KV, other models, runs beyond two per variant. The 80K P2 row is a cold re-prefill and should not be used for the SSD comparison.
