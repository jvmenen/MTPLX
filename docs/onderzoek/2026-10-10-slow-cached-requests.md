# Slow requests with an almost fully cached prompt (finding 128)

Date: 10 Oct 2026. Production server on port 8000 (Qwen3.8-27B Optimized-Speed, M5 Pro 64 GB). CPU-only log and code analysis; no GPU jobs, no restarts, server and `~/Dev/MTPLX-prod` only read.

Legend: **[measured]** = in the logs below, **[read]** = from the code, not run, **[assumed]** = reasoning only.

Sources:

- `~/.mtplx/metrics/flight-8000.jsonl` (events `begin`, `prefill`, `s`, `pc`, `end`)
- `~/.mtplx/logs/request-log-8000.jsonl` (one row per request, with `ttft_spans`, `response_tail_wait`, `reread`, `session_restore_served`)
- `~/Dev/nederlandse-loterij/var/mtplx.log` (stdout/stderr of the server that `backend_launcher.py` starts: `<pid_dir>/mtplx.log`). It covers the server of 02:26:48 to ~08:00 (from line 29136 "MTPLX is ready"). The server started at 08:02:14 (pid 15682) writes stdout and stderr to `/dev/null`, so for it only the flight and request logs exist.
- `~/.mtplx/session-bank/manifest.sqlite` (read-only)
- Code: `~/Dev/MTPLX-prod` at `prod-2026-10-10b` (aaece69a); the relevant paths are identical in `prod-2026-10-10` (`git diff` between the tags does not touch them).

## 1. What the flight events measure [read]

- `begin` is written inside `event_stream()` (`openai.py` ~39253), after the prologue. The prologue includes `_await_session_response_tail()`: a wait until the previous turn of the same session has finished its commit. So `begin` comes *after* that wait.
- `prefill` is `note_decode_started()` (~41220): written when the first generated tokens arrive. It marks the end of prefill (start of decode), not the start of work.
- The scheduler queue (`scheduler_queue_s`) and the GPU lock come after `begin`.
- `end.ttft_s` is measured from HTTP arrival (`ttft_spans.origin = http_arrival`).

So `prefill - begin` = scheduler queue + lock + restore + prefill + first round, and `ttft - (prefill - begin)` = everything before `begin`, which is almost entirely `response_tail_wait_s`. The "work" figure in the question was in fact a wait.

`ttft_spans.exclusive_s` explains the full TTFT: for 339 of 342 requests of the old server the unexplained remainder is below 1 s [measured].

## 2. The ten cases [measured]

All ten: `cache_source` ram, `session_restore_mode` clone, restore point equal to the matched history (no near-lane, no SSD restore), `cache_restore_time_s` 0.0, `prompt_mtp_history_time_s` <= 0.1 s, `lock_wait_s` <= 0.2 s, `postcommit_wait` "no_pending". Engine time from scheduler admission to first token matches the new-token count (~250 to 330 tok/s prefill).

| End | Prompt / cached / new | Tail wait | Sched. queue | Engine | What the server was doing meanwhile |
|---|---|---|---|---|---|
| 05:53 | 65447 / 62733 / 2714 | 135 s | 0 | 10.6 s | Other session e71f337d: answer of 3276 tokens, 133 s decode |
| 05:57 | 67290 / 66648 / 642 | 254 s | 0 | 2.8 s | e71f337d re-read 80,255 tokens (247 s prefill): its RAM entry had been dropped, see section 4 |
| 06:08 | 103635 / 103551 / 84 | 105 s | 0 | 0.7 s | 420353ec: 2463 tokens, 99 s decode |
| 06:37 | 112409 / 111742 / 667 | 235 s | 81 s | 4.1 s | 420353ec: 98,905-token prompt with nothing cached (dropped), client disconnected after 618 s without a token; then a new 17,741-token conversation (a7fd11c2) ahead in the queue |
| 07:19 | 94750 / 94335 / 415 | 225 s | 0 | 2.2 s | fb90c3df: new conversation, 14,091 tokens read plus a 6298-token answer (290 s); e71f337d retried a 123,535-token prompt three times and disconnected each time |
| 07:31 | 80721 / 76279 / 4442 | 112 s | 0 | 17.1 s | a7fd11c2: 2606 tokens, 110 s decode |
| 07:35 | 85467 / 85093 / 374 | 162 s | 0 | 1.8 s | a7fd11c2: 3608 tokens, 158 s decode |
| 07:38 | 88085 / 85588 / 2497 | 132 s | 0 | 10.2 s | a7fd11c2: 2902 tokens, 129 s decode |
| 07:40 | 88836 / 88568 / 268 | 158 s | 0 | 1.7 s | a7fd11c2: 3725 tokens, 157 s decode |
| 07:54 | 107064 / 105476 / 1588 | 110 s | 0 | 7.4 s | a99966f8 (1667 tokens, 52 s) and e82a874d (19,182 new tokens, 44 s) |

The `pc` event `generation_final` of the waiting session lands each time within 0.3 s after the other session's request ends; the commit itself took 0.05 to 0.3 s.

## 3. Mechanism: the response tail [read, timeline measured]

With `_stream_terminal_frame_before_commit_enabled()` (on for client-named sessions) the stream sends its last frame and `[DONE]` at the last token. The commit of that turn (`_store_generation_final_history_snapshot`, bank put) runs afterwards as a new `_submit_foreground_model_work` item on the serial model scheduler (`openai.py` ~40400). The client sees the answer, runs its tool and sends the next turn. That turn waits in `_await_session_response_tail` until the commit landed (no deadline, rounds of `STREAM_COMMIT_WAIT_MAX_S`, commit cc17af84).

With two agents on one serial server, the other agent's request is usually already queued when this commit is submitted. The commit runs after that whole request: prefill plus the full decode. So agent A's next turn waits for agent B's complete turn; the wait is only labelled `response_tail_wait` instead of `scheduler_queue`. Before the early terminal frame, the same wait sat at the end of the previous stream [read].

Making the commit a priority item (`_submit_priority_foreground_model_work` exists but is unused) would not shorten this: the commit would run first, but A's next request would then queue behind B's request anyway [assumed, reasoning about FIFO order].

## 4. Second cause: dropped sessions that were never on SSD [measured]

`mtplx.log` of the old server: 153 `prefill_admission_shed`, 46 `pressure_trim` (level 4), 4 `allocation_failure_shed`, 2 `prefill_shed_before_abort`, 1 `prefill_system_abort`. In 8 sheds `idle_release` dropped a whole idle session from RAM; in 7 of them with `on_ssd_entries: 0` and `persistence_cancelled: 1`:

| Dropped session | Longest prefix | RAM | Shed for prompt of |
|---|---|---|---|
| 42b0ea22 | 34,497 | 9.5 GB | 89,992 |
| e71f337d | 87,744 | 7.5 GB | 65,447 (05:52) |
| e71f337d | 103,834 | 8.6 GB | 81,496 |
| e71f337d | 110,304 | 7.8 GB | 96,906 |
| 420353ec | 98,580 | 8.3 GB | 110,846 |
| e71f337d | 122,712 | 9.9 GB | 73,071 |
| e82a874d | 21,317 | 5.7 GB | 116,381 |

The SSD write of a bank entry is deferred to the idle lane (`cold_enqueue.deferred: true` on every put). With two agents working back to back the idle lane rarely gets a turn: the manifest holds one entry for e71f337d (prefix 121,090, written 07:01:58) and none for 420353ec. When the shed releases the session, the queued write is cancelled (`_cancel_queued_persistence`) and the state is gone. The next turn of that session restores 7,571 tokens (another conversation's shared head, near lane, entry `d0f4238d` of session 9ef1b978) and re-reads the rest: 80,255 tokens (247 s), 96,829 (314 s), 103,275 (344 s). These long prefills are what the other agent waits behind (case 05:57, and indirectly 06:37 and 07:19). One prompt of 98,905 tokens ran about 4 minutes of prefill and was then abandoned by the client at 618 s.

This is not the #609 bug. #609 fixes a session whose own entry *is* on SSD and was skipped in favour of a neighbour's system prompt. Here the own entry was not on SSD (`on_ssd_entries: 0` at drop time), so serving the shared 7,571-token head was the best available restore. #609 would not have changed these turns [read + measured]. It is the "spill starvation behind long turns" left open in finding 126.

## 5. Route `bank_eager:promotion_failure:segmented_kv_cache` [read]

`promote_kv_cache_offsets()` (`graphbank.py` ~1515) cannot adopt a `SegmentedKVCache` into the tensor-offset adapter that the compiled verify graph needs (a segmented cache has no single buffer). It counts `promotion_failure:segmented_kv_cache` and the verify runs eager. This is the known trade-off of segmented KV (finding 113), not an error. It touches decode only, never TTFT. Decode median at <32K context: 34.4 tok/s before segments (Qwen3.8, 7 to 10 Oct) against 33.7 with segments [measured, small sample, n=54/27]. For longer contexts no baseline with the same model exists in the logs.

## 6. Queue waiting, old server (02:26:48 to 08:00) [measured]

402 requests started (`begin`), 346 got a first token. `prefill - begin` > 10 s: **112 of 346** (confirmed). Buckets: 0-1 s 66, 1-10 s 168, 10-30 s 42, 30-60 s 27, 60-120 s 25, 120-300 s 14, >300 s 4; p50 4.2 s, p90 76 s. Note this mixes queue and engine time and misses the tail wait.

With `ttft_spans` (341 requests): waiting = `response_tail_wait_s` + `scheduler_queue_s` + `lock_wait_s`.

| Wait | 0-1 s | 1-10 s | 10-30 s | 30-60 s | 60-120 s | 120-300 s | >300 s |
|---|---|---|---|---|---|---|---|
| tail wait | 261 | 16 | 26 | 17 | 15 | 7 | 0 |
| scheduler queue + lock | 258 | 21 | 22 | 11 | 15 | 14 | 1 |
| total wait | 180 | 36 | 48 | 27 | 29 | 20 | 2 |

126 of 341 waited more than 10 s in total (65 via the tail, 63 via the queue, 2 both). Total waiting 152 min (tail 66, queue 87) against 58 min engine time to first token and 228 min decode. TTFT p50 10.2 s, p90 109 s; waiting p50 0.1 s, p75 27 s, p90 92 s, max 382 s. The tail wait and the queue are two labels for the same thing: waiting for the other agent's turn. Finding 116 covers the serial scheduler.

## 7. New server prod-2026-10-10b (since 08:02:14) [measured, up to 08:13]

16 requests (1 warmup of 512 tokens, 1 cancelled), two new conversations (343be61a, 026b39de) of 17K to 55K tokens. Same pattern: 343be61a waited 58, 30 and 32 s on the tail, 026b39de 26 to 207 s in the queue (08:13: behind a 5465-token answer, 202 s decode). No request was slow without a wait: engine time matches the new tokens everywhere (largest 39 s for a cold 17,101-token first turn). One request (08:13:08) spent 18.4 s in `canonicalize_s` (elsewhere at most 0.12 s), not explained. Whether sheds drop sessions on this server cannot be seen: its stdout goes to `/dev/null`; the contexts so far are too small to expect it [assumed].

## 8. Answers

1. Between the `prefill` event and the first token nothing slow happened; `prefill` *is* the first token. The time was spent before `begin`, waiting for the previous turn's commit, which queued behind the other agent's whole request. No SSD restore, no near-lane neighbour pick, no re-prefill counted as cached, no MTP history rebuild, no lock wait, no segment merge or memory trim in these ten requests.
2. Common cause: two agents on one serial server; each waits for the other's whole turn (mostly 2,500 to 3,700-token answers at 23 to 26 tok/s at 85K to 110K context, 110 to 160 s). Made worse by sessions dropped under memory pressure without an SSD copy, whose 80K to 103K re-reads take 4 to 6 minutes.
3. Not fixed in prod-2026-10-10b: #609 addresses another case. The serial wait is by design; the dropped-without-SSD loss is open.
4. New server: 15 real requests, same waiting pattern, no slow-without-wait case.
5. 112 of 346 confirmed for `prefill - begin`; including the tail wait 126 of 341.

## 9. Proposed changes (not built)

1. **Observability (small, clear).** Add `response_tail_wait_s` and `scheduler_queue_s` to the flight `end` event (or emit `begin` at HTTP arrival) and document that `prefill` means first token. Dashboards that count queue waiting should add the tail wait; otherwise half of the waiting is invisible.
2. **Spill before drop.** In the admission shed's `idle_release`, an entry with `on_ssd_entries == 0` should be written to SSD before it is let go (or skipped in favour of other reclaim steps), instead of cancelling its queued write. Cost: one synchronous 7 to 10 GB write while the admitting request waits, a few seconds on the internal SSD [assumed], against 4 to 6 minutes of re-prefill. Risk: the arrays stay alive during the write, which is exactly when memory is short; needs a GPU test. Alternative: give deferred cold writes a slot between foreground items when they have waited longer than N seconds.
3. **Load (Jeroen's choice).** Keep agent contexts below roughly 80K (earlier compaction), or limit answer length, so both sessions fit in RAM and each turn is shorter; or do not run two long agent sessions on one server at the same time.

## 10. Spill before release: built (10 Oct, afternoon)

Branch `fix/spill-before-idle-release` (from upstream main 9882703f, worktree `~/Dev/laya-nl/mtplx-spill-before-release`, commit 75c47505, not pushed). Cherry-picked onto `prod-2026-10-10b` as `test/prod-spill-before-release` (fcbc95cd, worktree `~/Dev/laya-nl/mtplx-prod-spillfix`; one trivial conflict, the capability markers in `cold_tier.py`).

### Where the write was lost [read]

- Every bank put files the entry's SSD write on the idle persistence lane (`_enqueue_cold_entry` -> `_dispatch_persistence`, key `ssd_cold:<session>`, newest wins). The lane runs only after a quiet window of 0.3 s (`ModelWorkScheduler.idle_grace_s`); with the other agent's request always queued it does not run.
- The admission's step 6 (`_run_prefill_admission`, "Whole idle conversations") calls `EngineSessionManager.release_idle_sessions` -> `SessionBank.release_sessions`. That evicts every RAM entry of the chosen session and then calls `_cancel_queued_persistence` for them: the queued job is removed from the scheduler (`cancel_idle_persistence`) and the entry is gone. The docstring said so on purpose: writing out on the request path would "stage its bytes through the page cache and hash them ... while the Mac is short of memory".
- Other paths that also cancel a queued write: steps 4/5 of the admission (`shrink_to_bytes` / `shrink_for_admission`, default `cancel_queued_persistence=True`), step 2b (`cancel_queued_persistence` for entries already out of RAM), step 7 (the request's own siblings), the shed before a prefill abort (`_shed_reusable_memory`), the pressure trim (`_memory_pressure_loop`, on the asyncio thread) and the allocation-failure shed. In the 02:26-08:00 log the whole-session losses came from step 6 (8 releases with entries, 9 entries dropped, 7 of them sessions with nothing on SSD). Over all 153 sheds the LRU step evicted 0 entries and the chain walk 4 (whether those had a queued write is not in the receipt), step 2b never ran, step 7 released 1 entry without cancelling a write; the pressure trim evicted entries in 4 of 46 ticks, the allocation-failure shed 1 entry [measured, counts from `mtplx.log`].
- All seven losses were triggered by the Mac's line, not the engine line: available 5.0-7.0 GB against a shed floor of 5.6-6.6 GB, engine plus growth 38.7-47.1 GB against a 50 GB threshold [measured].

### The change [read]

- `release_sessions(..., write_out_before_release=True)`: for each session it releases, while it holds that session's slot, every entry whose own SSD write is still queued (not published, not a lease) is written to the cold tier first, synchronously on the model-owner thread, through `spill_entry`. Then the release goes on as before; the queued job is cancelled afterwards (it would only rewrite the same entry), which frees its arrays.
- `spill_entry` streams one tensor at a time (units of at most 32 MiB of host copy, then hash, write, drop), whatever the entry's size; it never stages the payload like `put_entry` does. It takes a new `give_up_at_s`: a deadline that replaces the foreground signal, because during a request that signal is always up and the idle-lane yield would stop the write at its first tensor.
- Bound: `RELEASE_WRITE_OUT_MAX_S = 30.0` per release call. A write that runs out of time, that the tier refuses (hourly budget, disk room, size cap) or that fails leaves the old behaviour: the entry is dropped. Nothing partial becomes restorable (the manifest row lands last).
- Only the admission's idle-conversation step opts in. The shed before an abort, the own-session step, the pressure trim and the allocation-failure shed keep cancelling (memory shortest there, or off the owner thread). With the SSD cache off, or a tier without the deadline, nothing changes. No new switch.
- Receipts: `idle_release.written_out_entries`, `idle_release.write_out_s`; eviction log `release_write_out` with outcome and time.

### Memory and time during the write

- Memory [read + assumed]: the entry's arrays are alive anyway until the release frees them; the write adds one unit (<= 32 MiB) of host copy plus the encode's small GPU slices. File pages written land in the page cache; the guard counts file-backed pages as available (`available = free + file-backed`, `system_memory._supply_from_statistics`), so by the guard's own measure the write moves pages from free to file-backed without lowering available. Whether macOS sees it the same under pressure is what the GPU test measures.
- Time [measured earlier, other conditions]: a first `spill_entry` of an 80K entry (5.97 GB on disk) took 3.6 s in the direct benchmark of 8 Oct (`2026-10-08-segmented-kv-ssd.md`, section on write time; thermal 0, no request running). For 7.5-9.9 GB entries that suggests 4-6 s [assumed], against 247-344 s of re-prefill each in the log.
- Segmented KV (production): `spill_entry` on the prod tree encodes segments as stock blocks and does not rehash sealed segments it already hashed (6e3124f7). An entry is written only when its own write is queued; with `MTPLX_SEGMENTED_KV_SSD=0` no write is queued for a segmented entry, so none is written out either.

### Tests [measured, run by the coordinator's chain]

`tests/test_release_writes_out_unwritten.py` (10 tests, real `SessionBank` and `SessionBankColdTier`, engine busy). Related tests green on main+fix and prod+fix; the 8 release/admission tests fail on stock main and stock prod (baseline mode, by behaviour); full suites on main+fix and prod+fix: 0 failed (`m3/server/spill/logs/unit-related.log`, `suites-fix.log`).

### GPU check: correctness only (Jeroen, 10 Oct: no timing)

Harness `~/Dev/laya-nl/segment-kv/m3/server/spill/` (`drive_spill2.py`, `run_spill2.sh`, `tables_spill.py`), prod-2026-10-10b + fix with production's flags (`MTPLX_SEGMENTED_KV=1`, `pkg-combo-0323`), port 8011, through `gate.sh`.

Why the first run (Q1) tested nothing [measured]: B's turns were refused with a 507 sent as an error frame inside the stream, which the old driver did not read as an error. And no release could happen: (1) with two clients alternating strictly, the session that just finished is still "finalizing" when the other one's admission runs (its commit is queued behind that request, the mechanism of section 3), so it is in flight and kept; (2) with a shared system prompt, A's entry is B's near-prefix restore source, and a release never takes a restore source (Q4: `held_because: restore_source_or_not_reached`). The 3-s tool-time fillers of Q3 were refused as background tasks (`is_background_request`: max_tokens <= 48 with a different system prompt).

Fixed harness: turns in sequence (A t0, B t0, A t1, ...) 2 s apart, so the previous turn's commit lands and its conversation is idle as an agent busy in a tool; two filler clients (433 tokens, 64-token answers) send requests back to back the whole time, so the idle lane never gets its 0.3 s quiet window; A and B have different system prompts; stream errors count; engine limit via `MTPLX_MEMORY_LIMIT_BYTES`.

**Q5 (limit 32.5 GB)** [measured]:

| Turn | Release in its admission | Written first | Next turn of the released conversation |
|---|---|---|---|
| B t0 | A (5.33 GiB held) | yes, `release_write_out` outcome written, 6.3 s | A t1: `ssd_clone`, 62,336 of 63,881 cached, TTFT 13.3 s (A t0 cold: 162 s) |
| A t1 | B (3.98 GiB) | yes, 4.0 s | B t1: `ssd_clone`, 59,138 of 60,684 cached, TTFT 13.2 s |
| B t1 | A (4.30 GiB) | yes, 4.7 s | A t2: 507 (below) |
| A t2 | B (4.10 GiB) | yes, 2.5 s | B t2: 507 (below) |

Every release: `on_ssd_entries 1`, `dropped_entries 0`, queued write cancelled after the write. Memory sampled every 20 ms during the four writes: MLX active up to 143-262 MiB above the start in three, 1,312 MiB in one (cause not traced); the Mac's available memory never fell more than 0.2 GiB during a write.

The two t2 turns ended in a 507 during their suffix prefill, after their admission had released (and written) the other conversation: the per-chunk engine check stopped them (engine 28.5 GiB + 1.9 GiB chunk, and 27.4 + 5.4, over the test limit of 30.3 GiB; weights are 20.7 GiB). The admission prices an SSD restore as if nothing were reused, and the restored entry then holds more than that price. This is the artificially low limit of the harness plus existing pricing, not the write-out: the write had finished and its memory was given back before the prefill [measured: order of events; assumed: that the same turns pass at a production limit].

**Q6 (limit 35.0 GB)** [measured]: no 507. Only one idle release (B t2 released A, written first, 4.3 s). Before that, A was dropped by the pressure trim instead: the allocator at 0.997 of the limit during B's cold prefill (warning level, deferred 60 s while busy, then trimmed 2 entries) cancelled A's queued write, and A t1 re-read 63,881 tokens (173 s). The pressure trim runs on the asyncio thread and is left unchanged on purpose; it is a second way to lose an unwritten conversation (in production it evicted entries in 4 of 46 ticks at level 4).

Conclusion: the fix does what it should. An idle conversation whose SSD copy is not written is written first and then released, and its next turn restores from SSD (4 of 4 releases in Q5; 2 returning turns checked, both `ssd_clone` with 97-98 % cached). Open: no run where all six turns pass at once (Q5: 507s at t2 from the test limit; Q6: the trim got there first); the pressure-trim path.

## 11. The pressure trim keeps an unwritten conversation at WARNING: built (10 Oct, evening)

Branch `fix/trim-keeps-unwritten` (stacked on `fix/spill-before-idle-release`, worktree `~/Dev/laya-nl/mtplx-trim-keeps-unwritten`, commit 944fc42d, not pushed). On production as `test/prod-trim-keeps-unwritten` (prod-2026-10-10c, which already has the spill-before-release fix, + 741ac03f; worktree `~/Dev/laya-nl/mtplx-prod-trimfix`).

### How the trim works [read]

- `_memory_pressure_loop` is an asyncio task on the server's event loop, not on the model-owner thread. Every 10 s (2 s while the Mac is under its shed floor) it takes the worst of three levels: macOS memorystatus, the engine's allocator (active + pool >= 0.97 of the Metal limit is WARNING, >= 1.02 CRITICAL) and the Mac's available memory (shed floor WARNING, abort floor or fast compressor/swap growth CRITICAL).
- `_MemoryPressureGuard.decide`: acts on the rising edge, then at most every 120 s while the level stays up. WARNING waits for an idle engine up to 60 s, then trims anyway, while the request runs; CRITICAL never waits.
- Victims: `SessionBank.shrink_to_bytes`, never a session in flight, least recently used first. WARNING target = half of what the bank holds; CRITICAL target = 0 (everything idle). Each evicted entry's own queued SSD write is cancelled (`cancel_queued_persistence=True`): a queued job holds the entry's arrays until the idle lane runs it, so an eviction that leaves the job frees nothing.
- What it protects against: WARNING is the early signal, before the allocator or the Mac runs out; CRITICAL is the last step before swap or a kernel panic (the docstrings cite #144/#305 and two machines that ended in a watchdog panic). Within a request the admission, the per-chunk check (`_PrefillSystemGuard`, 507 on the next chunk that does not fit) and the sustained abort (three CRITICAL ticks) hold the request itself. If the trim did nothing at CRITICAL, the next step is the abort; at WARNING the other guards still stand.
- Difference with the admission release (#618): that runs on the model-owner thread, between work items, holding the released session, when a concrete request needs the memory; there a bounded write is safe. The trim runs on another thread while a prefill may be using the GPU, and is meant to give memory back now.

### How often in production [measured, `var/mtplx.log` last server run 02:26-08:00 + `request-log-8000.jsonl`]

46 trim ticks, 4 took entries (03:29 level 4, 1 entry; 04:34 level 2, 6 entries; 07:19 and 07:41 level 4, 1 entry each). The receipt did not say whether those entries were on SSD. Following each conversation's next turn: the only one that came back after the 04:34 WARNING trim restored 106,815 tokens from SSD (so it was written); in the other three the next turns of the conversations still alive were RAM hits (the evicted entry was not the one they needed) or the conversation did not come back (03:29: the 158K conversation that was being refused with 507s anyway). In this log no re-read caused by the trim was found; the loss was seen only in the GPU run Q6. All 7 lost conversations of section 4 came from the admission's step 6.

### Options weighed [read + assumed]

- (b) write out from the trim, like #618: rejected. `spill_entry` reads MLX arrays and evaluates the encode's slices; on the asyncio thread that would run next to the model-owner thread's prefill (MTPLX keeps all MLX work on one owner thread), and at 0.997 of the limit the write's units come on top.
- (c) let the trim queue a priority write and drop afterwards: the write would run after the current work item, which is when the admission (#618) already writes out a conversation it must release. The scheduler also deliberately has no "overdue" bypass for persistence work. Duplicate machinery, no gain.
- (a) + (d), built: at WARNING the trim passes over an entry whose own SSD write is still queued (`_awaits_queued_write`, the #618 helper) and takes entries on SSD or that nothing will write; CRITICAL unchanged (take everything idle, memory comes first). The conversation leaves RAM later: its write runs in the next quiet window and a later trim takes it, or a request that does not fit writes it out in its admission. Cost: with only unwritten conversations idle, a WARNING trim takes nothing (receipt `bank_unwritten_kept`).
- Receipt: `bank_unwritten_kept` and `bank_unwritten_dropped` (an entry taken with its queued write cancelled, at CRITICAL), so a future log answers question 2 directly.

### Tests [measured]

`tests/test_pressure_trim_keeps_unwritten.py` (5 tests, real loop, `SessionBank` and `SessionBankColdTier`, engine busy): WARNING keeps the unwritten conversation and its queued write; WARNING takes the conversation on SSD instead of an older unwritten one; once its write has run a later trim takes it; without an SSD cache the entry is taken as before; CRITICAL still takes it and counts it. On #618 alone all 5 fail (3 by behaviour, 2 on the missing receipt fields). The first full-suite run found 10 failures: three loop tests' stand-in banks did not accept the new keywords; fixed in the same commit. Re-run (`m3/server/spill/trim/unit.log`): related 261 passed (main+#618+fix) and 275 passed (prod-10c+fix); full suites 11,265 passed / 0 failed and 12,084 passed / 0 failed. Ruff (default rules) adds no findings to the changed files; AI-attribution check clean.

### GPU check Q7: correctness only [measured]

prod-2026-10-10c + fix, production's flags, limit 35.0 GB, same driver as Q6 (`trim/run-Q7.log`, `server-Q7.out`).

| | Q6 (without) | Q7 (with) |
|---|---|---|
| Trim | WARNING, allocator 0.997, deferred 60 s, 2 entries, A's write cancelled | WARNING, allocator 0.998, deferred 60 s, 1 entry, `bank_unwritten_kept 1`, `bank_unwritten_dropped 0` |
| A t1 | cold, 63,881 tokens re-read, TTFT 173 s | RAM clone, 62,336 of 63,881 cached, TTFT 7.6 s |
| Idle release | B t2 released A (written) | A t2 released B: 1 entry written first (4.3 s), 1 sibling dropped |
| B t2 | RAM | SSD clone, 60,907 of 61,950 cached, TTFT 7.2 s |
| 507 / errors | 0 | 0 |

The conversation the trim kept stayed in RAM and its next turn was a RAM hit; the conversation that later had to go was written to SSD first and its next turn restored from SSD. No 507. Which entry the trim did take is not in the receipt (the eviction log is in memory only); `bank_unwritten_dropped 0` says it was not an unwritten one.

### PR choice

Separate PR, stacked on #618: #618's text already says the trim is "left for a separate change", the mechanism differs (a keep rule, no write) and has its own trade-off (a WARNING trim can take nothing) that deserves its own review and its own revert. It uses #618's `_awaits_queued_write`, so it merges after #618.
