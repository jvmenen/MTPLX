# Erratic first turn after a bank spill, same process (2026-10-09)

Follow-up to [ssd-restore-layout-fix](2026-10-09-ssd-restore-layout-fix.md), section "After eviction, before restart". In both prod and prod+fix, the first turn of a conversation the bank had spilled to SSD either re-read almost the whole prompt (cached 4019 of ~58K, ~145 s), ended in `error`, or restored from SSD. After a server restart, SSD restores worked. This note explains why, from the server logs, request logs, SSD manifests and the code. No new GPU run was needed for the cause; a validation run of the fix is queued (see the end).

## Short answer

It is not a disk or timing problem. The spilled entry is on disk, complete, and found. The restore path picks the wrong lane:

1. With no exact prefix in RAM, `restore_or_prefill_prompt_state` runs the near-prefix lane first (`_restore_near_prefix_prompt_state`), and only then the exact lane (`session_bank.restore()`, which does the SSD restore `ssd_clone`).
2. The near lane asks the bank for candidates. The bank asks the SSD tier (`lookup_prefix_boundary`), whose best row is the conversation's own spilled entry: a whole prefix of the prompt (gap 0). The tier hydrates it (5.4 GB for the 50K conversation, 7.4 GB for the 80K one).
3. The near lane refuses that candidate (`matched_out_of_range`: it only serves `matched < prefix_len`; whole prefixes are the exact lane's job).
4. The next candidate is the *other* conversation's RAM entry, which shares the 4,019-token system prompt and holds a recurrent boundary there. The near lane serves it and returns. The exact lane never runs.

Result: restore 4,019 tokens, re-read the rest. Meanwhile the hydrated SSD copy is still referenced by the candidate list while that long prefill runs, on top of the other conversation's RAM entry. When the Mac's free memory is just short, the memory guard aborts the prefill (507 `memory_refusal`) and clears the bank. The next turn then finds RAM empty, the near lane has no RAM candidate to fall back on, and the exact lane does the SSD restore. That is the third outcome. After a restart, RAM is empty for the same reason, so SSD restores work.

## Evidence

Per run, the turn after eviction (`s50-t4`) and the next one (`s80-t3`), from `raw-X*.jsonl` (request log rows and `/health` cold-tier counters) and `server-X*.out`:

| Run | Step | Outcome | Served from | cold `restore_hits` before -> after |
|---|---|---|---|---|
| Xp1 | s50-t4 | 4019 cached, 143 s | s80's RAM entry (`entry_prefix_len` 86110), `requested_matched` 4034, restore point 4019, `candidate_index` 2, `cache_source` ram, `ssd_cache_hit` false | 0 -> 1 |
| Xp1 | s80-t3 | error after 140 s | memory guard: shed `lease-s50` (5.48 GB), then `prefill_system_abort` `under_abort_floor`, engine 41.1 GB | 1 -> 2 |
| Xf1 | s50-t4 | 4019 cached, 146 s | same as Xp1 (`candidate_index` 2, s80's entry) | 0 -> 1 |
| Xf1 | s80-t3 | error after 37 s | `prefill_system_abort` `under_abort_floor`, engine 40.5 GB | 1 -> 2 |
| Xf2 | s50-t4 | error after 10 s | 507 `memory_refusal`: "5.0 GiB was left free ... next prefill chunk needs 5.4 GiB ... 2.7 GiB floor"; shed `lease-s80` (7.38 GB), bank cleared (`allocation_failure_shed`), engine 38.9 GB | 0 -> 1 |
| Xf2 | s80-t3 | SSD restore, 86110 cached, TTFT 9.3 s | `ssd_clone`, `candidate_index` 0 | 1 -> 3 |
| Xp2 | s50-t4 | error after 18 s | same as Xf2 | 0 -> 1 |
| Xp2 | s80-t3 | SSD restore, TTFT 49.3 s | `ssd_clone` | 1 -> 3 |

Reading the table:

- Every step after eviction raises `restore_hits`, also the ones that end up at 4019: the SSD tier found and decoded the spilled entry each time. So the spill had finished, the in-process index knew it, and the token prefix matched. The manifests agree: the s50 entry (57,989 tokens) was written at the s50-t3 commit, about 400 s before s50-t4.
- The 4019-token match is the shared system prompt: every stored entry has its first recurrent boundary at 4019 (head anchor, `MTPLX_SESSION_HEAD_ANCHOR=1`), for both conversations.
- `candidate_index` 2 in the served record: candidate 1 was the refused SSD whole prefix.
- A successful SSD restore raises `restore_hits` by 2: one hydration in the near lane (refused), one in the exact lane (served). The same holds after the restart (Xp1 r50-t5 0 -> 2, r80-t4 2 -> 4). Every SSD restore therefore reads and decodes the entry twice.
- The restart is not what makes it work. In Xp2, the 80K turn *after* the restart again got 4,019 cached (`block_prefix_boundary_clone` from the 50K conversation's entry, `candidate_index` 2, 293 s), because the 50K conversation had just been restored into RAM and became the neighbour.
- Admission planned all four s50-t4 requests as fully cold (`reused_tokens` 0 in `admission-X*.jsonl`); the hydrated copy is not in that plan.

The fix from the layout note is not involved: both trees take the same lanes, and the outcomes split by run, not by tree.

## Why the outcome varies

Three things decide which of the three outcomes a turn gets:

1. Is there a RAM entry from another conversation with a usable boundary at the shared system prompt? Yes: 4019 cached, cold re-read. No: the exact lane restores from SSD.
2. Does the long re-read fit in memory with the hydrated SSD copy still held, plus the RAM bank? The margin was a few hundred MB (5.0 GiB free against a 5.4 GiB chunk). That depends on what else the Mac holds at that moment, so it differs between runs.
3. Did a previous abort clear the bank? Then the next turn finds no RAM neighbour and restores from SSD.

## Code

- `mtplx/generation.py`, `restore_or_prefill_prompt_state`: `exact_prefix_len` comes from RAM only (`session_bank.longest_prefix`). When it is below the prompt length, `_restore_near_prefix_prompt_state(min_restore_tokens=exact_prefix_len)` runs before `session_bank.restore()`, and any state it returns wins.
- `_restore_near_prefix_prompt_state`: rejects `matched >= entry.prefix_len` (`matched_out_of_range`) and moves on to the next candidate.
- `mtplx/session_bank.py`, `near_prefix_candidates` -> `_cold_near_prefix_candidate` -> `SessionBankColdTier.lookup_prefix_boundary`: ranks a gap-0 row as a `near_prefix` candidate and hydrates it (`_restore_row`). The existing `min_useful_matched_tokens` gate only skips rows that cannot beat RAM; with nothing in RAM above 4034 the bar is far below the spilled entry.
- A test comment in `tests/test_cold_prefix_ram_shadow.py` already says a gap-0 match "routes to the EXACT path instead and the near-prefix loop rejects it". The routing only holds when the near lane finds nothing else to serve.

## Fix

Branch `fix/near-lane-defers-ssd-exact` from upstream `main` (9882703f), commit `7a66d488`, worktree `~/Dev/laya-nl/mtplx-ssd-exact-defer`. Not pushed.

- `SessionBankColdTier.exact_prefix_len(tokens, identity...)`: length of the longest stored whole prefix, from the manifest alone (no tensor read, no counters), with the same row gates `_restore_row` applies before reading tensors.
- `SessionBank.cold_exact_prefix_len(...)`: delegates, 0 without a cold tier.
- `restore_or_prefill_prompt_state`: when RAM has no exact prefix, the first near-lane call gets `min_restore_tokens = cold_exact_prefix_len`. A RAM neighbour at 4034 no longer qualifies, and the cold lookup skips the hydration through the existing `min_useful_matched_tokens` gate (bar = floor + 1). The near lane returns nothing, and `session_bank.restore()` reads the entry back (`ssd_clone`). A near candidate that beats the SSD whole prefix still wins. If the exact lane declines, the second near-lane call keeps its floor of 0, as before.
- Side effect: one SSD decode per restore instead of two.
- Tests: `tests/test_near_lane_spilled_exact_prefix.py` (6 tests: the probe reads only the manifest and leaves counters alone; without the floor the spilled entry is hydrated for nothing; at the floor nothing is hydrated; the generation path passes the floor and ends in the SSD restore; a RAM exact prefix keeps its own floor). The generation test fails without the fix. Related files (session bank, SSD, cold tier, prefix and restore tests, `test_prefill_reread`, `test_server_openai`, `test_generation_sustained`, `test_checkpoint_pre_image`) all pass on CPU.

Not covered by the fix:

- A RAM exact prefix that is shorter than an SSD whole prefix: `restore()` serves the RAM one first. The fix leaves that case unchanged, so it is no worse than before.
- Rejected near-lane candidates stay referenced while the served candidate's prefill runs (the `for` loop holds the candidate list). Released by a follow-up change, see the next section.

## Follow-up: refused near candidates are released before the prefill

Branch `fix/near-lane-releases-refused`, stacked on `fix/near-lane-defers-ssd-exact`, commit `336b251e`, worktree `~/Dev/laya-nl/mtplx-near-reject-release`. Not pushed.

What held the refused SSD copy, traced in the code:

- `SessionBank.near_prefix_candidates` returns a plain list of `(entry, matched)`. The cold tier's candidate is a `SessionBankEntry` built in `_cold_near_prefix_candidate`, with the decoded tensors in `cache_snapshot` (materialised MLX arrays: `decode_payload` builds them with `mx.array(np.frombuffer(...))`). Nothing else keeps it: the cold tier has no decode memo (`_restore_row` returns the record and stores nothing), the `ColdPrefixRestoreRecord` is local to the bank call, `last_prefix_diagnostic` holds only numbers, and the candidate is not added to `_entries`.
- `_restore_near_prefix_prompt_state` iterated with `for entry, matched in candidates(...)`, and the served candidate's suffix prefill runs inside that loop body. The loop's iterator keeps the list alive, so every refused candidate (and every untried one after the served one) stayed referenced for the whole prefill. The local `entry` itself is rebound on the next candidate, so it was not the holder; the closures `_near_debug`/`_near_reject` read the same variable and do not pin an old value.
- When the prefill fails (the memory guard's 507), the exception's traceback holds that frame, list included, while the handler sheds the bank.

The change: the lane copies the candidates into a list, takes each one off the list as it tries it, and clears the rest just before the served candidate's suffix prefill. A refused candidate is dropped when the next one is taken. No `mx.clear_cache()` is added: the prefill guard counts the allocator pool as reclaimable (`available + pool` in `_prefill_system_abort_exception`), and the prefill already empties the pool after every chunk by default (`MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY`, default 1).

Could the tier have refused before decoding? Partly. A whole-prefix row (gap 0) is visible in the manifest, but the tier cannot refuse it for every caller: the batched (cohort) lane in `server/openai.py` serves such a row at a recurrent boundary below its end. With `7a66d488` the main case no longer decodes (the floor is at least that row's length). A row whose nearest recurrent boundary is at or below the caller's floor is also known before the tensors (from `payload.json`), but the tier only gets the floor folded into `min_useful_matched_tokens`, which compares the match, not the boundary. Both are left as they are.

Tests: `tests/test_near_lane_releases_refused.py` (3 tests). The first reproduces the 9 October shape on a real bank and cold tier (A spilled, B in RAM sharing the opening, floor 0): the tier decodes A's entry, the lane refuses it and serves B; a weakref checked from inside every forward of the prefill shows A's decoded entry is gone. The second uses a candidate list of four (whole prefix refused, restore failed, served, untried) and checks the three others are gone during the prefill. The third checks the miss reason when nothing is served. The first two fail without the change (the weakrefs are alive). Related files (546 passed) and the full suite (11259 passed, 71 skipped, 1 xfailed) pass on CPU.

Not changed in this commit, seen in the code: a served SSD near candidate held next to its clone during the prefill, and the same loop shape in the Gemma 4 near lane. Both are handled in the next section.

## Follow-up 2: served SSD candidate, Gemma 4 lane, refusal before the decode

Branch `fix/near-lane-followups`, stacked on `fix/near-lane-releases-refused` (`336b251e`), worktree `~/Dev/laya-nl/mtplx-near-followups`. Three commits, not pushed.

**1. A served SSD near candidate is released before its suffix prefill (`56a37bca`).** What held the decoded snapshot after the clone: only the lane's local `entry` (the served candidate). `restore_entry_prefix_cache` returns the new cache and numbers, not the entry; `served_truth` holds numbers; `lease_back` exists only for a `reference_lease` restore, never for an SSD candidate (it has no `cache_ref`, so the restore always clones); `restored` holds the new cache. The inherited GDN boundary records come from the entry but are carried into the new bank entry on purpose, so they stay. The lane now drops `entry` just before the suffix prefill when the candidate came from SSD (`cache_source == "ssd"`, set only by `_cold_near_prefix_candidate`; an SSD entry the exact lane promotes into RAM does not carry it). A RAM entry is kept under another name so its `hits`/`last_access_s` are still updated after the prefill. Adopting the decoded arrays instead of cloning them was not done: the restore would then install views into the snapshot, and when the decode covered more than the restore point those views pin the whole decoded tensors. Releasing is enough, and as a side effect, when the clone is still lazy at that moment (a boundary restore has no seed forward), MLX can reuse the snapshot's buffer for the clone because nothing else holds it.

Measured on CPU (toy runtime, 16 real `KVCache` layers, a real bank and cold tier, the turn diverges at 1,536 of 2,048 stored tokens and is served from SSD, 96 MB restored KV): active MLX memory during the suffix forward 241 MB before, 145 MB after; the difference is exactly the restored KV. Script: scratchpad `measure_served_ssd.py` (not kept). On the 27B the extra copy is the KV up to the restore point, about 100 KB per token by the 9 October numbers (5.4 GB for ~54K tokens), so roughly 5 GB for a 50K restore; that figure is derived, not measured. No GPU run.

Test: `tests/test_near_lane_releases_refused.py::test_a_served_ssd_candidate_is_released_before_its_suffix_prefill` (real bank and cold tier; weakref on the decoded candidate checked in every forward: alive in the seed forward, gone in the suffix forwards). Fails without the change. The RAM case also asserts `hits == 1`.

**2. Gemma 4 near lane (`a732e942`).** `_restore_or_prefill_gemma4_prompt` in `mtplx/backends/gemma4_assistant.py` had the old loop: the tail forward ran inside `for entry, matched in candidates(...)`, so the list held every refused and untried candidate. It also had a second hold: when nothing was served, `entry` stayed bound to the last candidate tried through `cold_prefill`. Same change as the Qwen lane: candidates taken off a list, the rest cleared and a served SSD candidate dropped before the tail forward, `entry = None` before the cold prefill. Tests: `tests/test_gemma4_near_lane_releases.py` (3 tests: refused and untried released, a served RAM entry kept and counted; a served SSD candidate released; the last refused candidate released before the cold prefill). All three fail without the change.

**3. Refusal before the decode when the boundary lies below the floor (`c59212d9`).** Small and clean enough to build. `_restore_row` already reads `payload.json` before any tensor and computes the newest recurrent boundary at or below the match (`_payload_boundary_at_or_below`). `lookup_prefix_boundary` gets `min_restore_point`; a hybrid row whose boundary lies below it is refused undecoded (`ssd_prefix_boundary_below_floor`, counter `prefix_restores_below_caller_floor`) and the next ranked row is tried, the same way as a row without a boundary. `SessionBank` passes the near lane's floor (`min_restore_tokens`) when the tier sets `SUPPORTS_MIN_RESTORE_POINT`. The test is strict (`boundary < floor`), while the serial near lane refuses `boundary <= floor`: the batched (cohort) lane in `server/openai.py` passes the same parameter (512) and does serve a boundary exactly at it, so a boundary *at* the floor still decodes for the serial lane. Whole-prefix rows (gap 0) are unchanged, for the reason in the previous section. Tests: `tests/test_cold_tier_restore_point_floor.py` (4 tests: refused before any decode, at or above the floor still decodes, the next ranked row serves, the bank passes the floor). All four fail without the change.

Checks: ruff on the changed files gives no new findings (the existing ones in `generation.py`, `gemma4_assistant.py`, `cold_tier.py`, `session_bank.py` are unchanged, the new test files are clean); related tests 884 passed, 1 skipped, 1 xfailed; full suite 11267 passed, 71 skipped, 1 xfailed (CPU; the first run failed one test because the new miss reason had no app sentence in `CacheExplanation.swift`, added to the same commit, which is now `c59212d9`). `scripts/check_ai_attribution.py --range origin/main..HEAD` clean.

## Validation run (Xe1, Xe2)

Branch `test/prod-ssdfix-exactdefer` (prod + layout fix + this fix), same scenario, through `gate.sh` (`run_exact_chain.sh`, table `tables_ssdexact.py`).

| Step | Xe1 | Xe2 |
|---|---|---|
| s80-t0 (cold, before eviction) | ok, 253 s | **507 memory abort** (`death_signature`) |
| s80-t1, s80-t2 | RAM clone | 4019 from s50's SSD row, ~265 s each |
| s50-t4 (after eviction) | `ssd_clone` 57989, TTFT 4.0 s | `ssd_clone` 57989, TTFT 3.9 s |
| s80-t3 | `ssd_clone` 86110, TTFT 9.0 s | 4019 from s50's SSD row, 259 s |
| r50-t5 (after restart) | **507 memory abort** (`death_signature`) | `ssd_clone` 58821, 5.8 s |
| r80-t4 (after restart) | `ssd_clone` 87985, 5.2 s | 4019 from s50's SSD row, 265 s |

The step this fix targets is fixed in both runs. s50-t4 went from a 143-146 s re-read or a 507 to an SSD restore in ~4 s. Every SSD restore now raises `restore_hits` by 1 instead of 2: the double decode is gone. The other outliers have different causes.

**Xe2, the 80K conversation at 4019: its state never reached SSD.** The SSD manifest of Xe2 holds no `lease-s80` row at all; Xe1 holds three. The chain of events:
1. s80-t0 (a plain cold prefill, `reusable_prefix_tokens` 0) was aborted by the memory guard. The conversation got no state, and the client continued with an empty assistant reply.
2. s80-t1 had its prompt-prefix commit dropped by the admission (`prompt_publish_skipped`, commit record `{}`). Its postcommit was `aborted` (`foreground_preempted_postcommit`, `stop_token_boundary_mismatch`). Nothing was banked.
3. s80-t2 banked 85,504 tokens in RAM, plus a postcommit at 85,837, but the SSD spill was never admitted: no `reserved_write_bytes`, and `writes_completed` stayed at 3. The deferred spill only runs in quiet windows. Those windows were taken by the 62 s postcommit and then by s50-t4 and s80-t3 (275 s).
4. At the first shutdown, one SSD write was still pending. The flush has a 10 s bound, and the harness sends `kill -9` after 8 s. The "flushed" line is missing, and at restart the reconcile deleted 2,109 orphan files (170 MB).

With no s80 row on SSD, the cold whole-prefix floor is 0. The near lane then correctly serves the best thing there is: the shared system prompt boundary of the 50K conversation's SSD row. The fix behaves as designed here; the gap is persistence, not restore.

**Xe1 r50-t5: memory abort during the SSD restore right after the restart.** The bank had just taken the restored 4.56 GB entry (`bank_bytes_after` 4,558,179,328, `restore_hits` 1). The guard aborted on `death_signature` after 5.5 s: free pages 1.4 GiB under the 3.0 GiB floor, the compressor +2.2 GiB, file-backed memory down from 11.7 to 2.8 GB. The engine held only 27-29 GB. The next turn (r80-t4) restored from SSD fine.

**Memory guard events.** Each run has exactly one abort. The second `prefill_shed_before_abort` in the counts is the copy nested inside the abort record. The aborts are at Xe2 s80-t0 (cold prefill before any eviction) and at Xe1 r50-t5 (exact SSD restore after the restart). Neither is in the near lane, and neither involves a refused hydrated candidate: the near lane hydrated nothing in those steps. Both are `death_signature` (system free pages collapsing), not `under_abort_floor` as in Xp/Xf. At both aborts, wired memory was ~17-19 GB above the engine's own bytes (Xe1 47.3 vs 27.2 GB, Xe2 49.4 vs 30.4 GB), against 3-6 GB at the Xp/Xf aborts.

**Effect on the fixes.** `7a66d488` stays as it is: its step is fixed and no outlier comes from it. The follow-up `336b251e` (release refused near candidates) is not exercised by these failures: none happened while a refused candidate was held. Persistence (point 3-4 above) and the system-memory aborts are separate items.

Proposals (not built):
- Harness: wait for the shutdown flush (kill -9 only after > 10 s) or call the flush before stopping.
- Upstream: a large deferred spill can starve behind long foreground turns until the entry is gone. Measure the spill's wait and encode time first, then decide between an earlier spill and a longer shutdown bound.
- The `death_signature` aborts: find what holds the extra wired memory (another process on the machine, or engine buffers the guard does not count) before tuning anything.

## Measured vs assumed

Measured (logs, request logs, manifests): which entry served each step, at which point, and from which candidate index; the hydration on every post-eviction step; the double hydration on successful SSD restores; the memory-guard aborts with their numbers; the bank being cleared by the abort, followed by an SSD restore on the next turn; the post-restart 4019 case in Xp2; the spill having finished long before.

Assumed (from the code, not measured): that the hydrated copy is still held during the 4019 prefill and is what tips memory under the floor (consistent with the engine bytes, but there was no memory profile); why the RAM neighbour did *not* qualify in the post-restart 80K turns of Xp1/Xf1/Xf2 (probably the boundary state of the freshly restored 50K entry at that moment; not traced); the time saved by dropping the second decode. GPU validation (Xe1, Xe2): s50-t4 measured fixed in both runs, one decode per SSD restore measured. Assumed in the validation analysis: why the s80 spill was never admitted (quiet-window scheduling, read from code and timing, not traced); the exact reason the admission dropped the s80-t1 publish; the source of the extra ~18 GB wired memory at the two `death_signature` aborts.

Follow-up 2: the extra copy of a served SSD candidate during the prefill is measured on CPU with a toy runtime (96 MB, gone after the change); its size on the 27B (about 5 GB for a 50K restore) is derived from the 9 October bytes per token, not measured. That the Gemma 4 lane's holds and the below-floor decodes occur in production is read from the code, not observed in logs.
