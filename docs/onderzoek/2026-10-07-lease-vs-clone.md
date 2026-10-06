# Lease versus clone of the session cache (`MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE=1`): measurement

Date: 2026-10-07. Status: measurement, no MTPLX code changed (production tree, server config and other worktrees untouched; nothing pushed).
Raw data, scripts and logs: `/Users/joonix/Dev/laya-nl/lease-meting/` (`raw-<variant>.jsonl`, `admission-<variant>.jsonl`, `server-<variant>.out`, `drive.py`, `run_variant.sh`, `make_tables.py`, `tables.md`).

## 1. Setup

- Mac17,9, 64 GiB, Qwen3.8-27B Optimized-Speed, launch command taken verbatim from `config/config.json` (`backends.mtplx.launch.command`), only changed as follows: port 8010, a private SSD session-cache directory per start (`MTPLX_SSD_SESSION_CACHE_DIR`, so the production bank is not touched and every start is cold), and `PYTHONPATH` prefixed with a `sitecustomize.py` that logs the return values of `_admission_growth` and `_admission_restore_copies_prefix` (the server runs as child process `python -m mtplx.server.openai`; the hook patches the functions in `__main__` after import, no source change).
- Variant A = production env (clone), variant B = production env plus `MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE=1`. Order A1, B1, B2, A2, each its own server start, GPU lock held per variant, gate (no process above 2 GB, no `BUILDING`, thermal 0, 60 s rest) before each start. A2 had to wait about 50 minutes for the gate (an unrelated 3 GB `Virtualization.framework` process); it ran afterwards unchanged.
- Session identity: the code resolves the session from the headers `x-mtplx-session-id`, `x-session-affinity`, `x-session-id`, `x-openwebui-*` or metadata keys (`mtplx/engine_session.py:2077-2095`). The scenarios use `x-mtplx-session-id`; header-identified sessions get `request_session_keep_live_ref=true`. Anonymous coding-agent tool requests get live refs as well (`_session_keep_live_refs_for_request`), measured separately (scenario `anon`).
- Requests: `/v1/chat/completions`, streaming, thinking off (`enable_thinking=false`), temperature 0, `max_tokens` 400, system head of ~4K tokens. Follow-up turns add a tool-result-like text (300 to 1,500 tokens) plus a short question and echo the model's own previous answer.
  - `c50` / `c80`: first turn 54K / 84K prompt tokens, then 5 follow-ups; fork (same system head and first user turn, other final question, other session id); go-back (session 1 again, from the state after turn 2 with a new user message), then one follow-up on that branch.
  - `s30`: 34K tokens, 3 follow-ups, sampled (temperature 0.7, top_p 0.95, seed 1234).
  - `anon`: 30K tokens, 4 follow-ups, no session header, `tools` present, history as assistant `tool_calls` plus `tool` results (answers not echoed), the shape of Claude CLI traffic.
  - `REF`: the c50 fork prompt on a cold server, as numerical reference.
- Per request: client TTFT, `request-log-8010.jsonl` row (prompt/cached tokens, `session_restore_mode`, `prompt_eval_time_s`, decode tok/s, verify time, active/peak memory), admission pricing from the hook, thermal level, swap, RSS, finish reason.

## 2. Results

Thermal pressure was 0 at every start and rose to 1 to 2 during every variant (heavy cold prefills of 140 to 260 s). All four variants therefore ran partly under thermal pressure; decode at 80K is about 23 to 24 tok/s against 27 to 28 at 50K. A1 and A2 (and B1 and B2) are the repeat pairs: they differ from each other by at most 1 to 3 % on speed.


### c50: follow-up turns t1-t5 (means over the turns)

| Variant | mode | TTFT s | suffix prefill s | prefill tok/s | decode tok/s | ms/verify | active GB t0 -> t{n} | peak GB | max thermal |
|---|---|---|---|---|---|---|---|---|---|
| A1 | clone | 3.10 | 3.02 | 285 | 27.5 | 92.6 | 27.7 -> 32.8 | 33.9 | 1 |
| A2 | clone | 3.14 | 3.06 | 282 | 27.6 | 92.1 | 27.7 -> 32.8 | 33.9 | 1 |
| B1 | reference_lease | 3.16 | 3.08 | 280 | 27.2 | 93.6 | 27.7 -> 32.6 | 33.8 | 2 |
| B2 | reference_lease | 3.16 | 3.09 | 279 | 27.4 | 93.2 | 27.7 -> 32.6 | 33.8 | 2 |

### c80: follow-up turns t1-t5 (means over the turns)

| Variant | mode | TTFT s | suffix prefill s | prefill tok/s | decode tok/s | ms/verify | active GB t0 -> t{n} | peak GB | max thermal |
|---|---|---|---|---|---|---|---|---|---|
| A1 | clone | 3.81 | 3.64 | 237 | 23.9 | 103.3 | 24.9 -> 31.6 | 38.6 | 2 |
| A2 | clone | 3.83 | 3.64 | 237 | 24.0 | 103.7 | 30.2 -> 31.3 | 39.5 | 2 |
| B1 | reference_lease | 3.87 | 3.67 | 235 | 23.8 | 104.6 | 30.2 -> 31.5 | 43.6 | 2 |
| B2 | reference_lease | 3.82 | 3.64 | 237 | 23.1 | 106.6 | 35.8 -> 32.3 | 43.5 | 2 |

### s30: follow-up turns t1-t5 (means over the turns)

| Variant | mode | TTFT s | suffix prefill s | prefill tok/s | decode tok/s | ms/verify | active GB t0 -> t{n} | peak GB | max thermal |
|---|---|---|---|---|---|---|---|---|---|
| A1 | clone | 2.85 | 2.74 | 332 | 26.5 | 92.4 | 31.7 -> 31.1 | 48.4 | 2 |
| A2 | clone | 2.81 | 2.68 | 339 | 27.4 | 89.1 | 37.0 -> 30.9 | 39.5 | 2 |
| B1 | reference_lease | 2.79 | 2.67 | 340 | 27.1 | 90.2 | 37.0 -> 30.8 | 43.6 | 2 |
| B2 | reference_lease | 2.83 | 2.73 | 333 | 25.6 | 96.1 | 31.7 -> 37.4 | 43.5 | 2 |

### Fork and go-back requests

| Scenario | Variant | restore mode | cached / prompt | TTFT s | suffix prefill s | tok/s | active GB after | result |
|---|---|---|---|---|---|---|---|---|
| c50-fork | A1 | block_prefix_boundary_reference_lease | 49152 / 54047 | 16.4 | 14.6 | 23.9 | 32.6 | ok |
| c50-fork | A2 | block_prefix_boundary_reference_lease | 49152 / 54047 | 17.3 | 15.8 | 23.6 | 37.1 | ok |
| c50-fork | B1 | block_prefix_boundary_reference_lease | 49152 / 54047 | 15.9 | 14.5 | 22.9 | 37.1 | ok |
| c50-fork | B2 | block_prefix_boundary_reference_lease | 49152 / 54047 | 15.9 | 14.6 | 22.7 | 37.1 | ok |
| c50-back | A1 | block_prefix_boundary_clone | 49152 / 56738 | 23.0 | 22.8 | 25.4 | 32.5 | ok |
| c50-back | A2 | block_prefix_boundary_clone | 49152 / 56738 | 24.2 | 24.1 | 24.5 | 37.2 | ok |
| c50-back | B1 | block_prefix_boundary_clone | 49152 / 56738 | 23.7 | 23.6 | 23.8 | 41.2 | ok |
| c50-back | B2 | block_prefix_boundary_clone | 49152 / 56738 | 23.7 | 23.6 | 24.1 | 41.2 | ok |
| c50-after-back | A1 | clone | 56927 / 57466 | 2.2 | 2.1 | 27.2 | 31.5 | ok |
| c50-after-back | A2 | clone | 56927 / 57466 | 2.2 | 2.1 | 27.4 | 32.7 | ok |
| c50-after-back | B1 | reference_lease | 56927 / 57466 | 2.1 | 1.9 | 27.1 | 32.7 | ok |
| c50-after-back | B2 | - | - / 57466 | - | - | 0.0 | 23.4 | HTTP 507 refused |
| c80-fork | A1 | cold | 0 / 84049 | 251.4 | 250.5 | 21.1 | 24.9 | ok |
| c80-fork | A2 | cold | 0 / 84049 | 257.0 | 256.1 | 20.6 | 24.9 | ok |
| c80-fork | B1 | cold | 0 / 84049 | 257.0 | 256.2 | 20.6 | 37.2 | ok |
| c80-fork | B2 | block_prefix_boundary_reference_lease | 83982 / 84049 | 2.9 | 1.0 | 21.2 | 25.3 | ok |
| c80-back | A1 | ssd_clone | 86090 / 86734 | 21.7 | 19.2 | 11.7 | 36.9 | ok |
| c80-back | A2 | - | - / 86739 | - | - | 0.0 | 25.0 | HTTP 507 refused |
| c80-back | B1 | - | - / 86739 | - | - | 0.0 | 19.2 | HTTP 507 refused |
| c80-back | B2 | ssd_block_prefix_boundary_clone | 84046 / 86734 | 13.1 | 10.5 | 21.7 | 37.6 | ok |
| c80-after-back | A1 | clone | 86980 / 87519 | 2.7 | 2.5 | 27.3 | 37.5 | ok |
| c80-after-back | A2 | clone | 86095 / 87266 | 5.1 | 4.9 | 21.6 | 36.9 | ok |
| c80-after-back | B1 | ssd_clone | 86095 / 87266 | 37.4 | 34.9 | 12.7 | 37.0 | ok |
| c80-after-back | B2 | - | - / 87519 | - | - | 0.0 | 25.8 | HTTP 507 refused |

### Admission events per variant (all requests)

| Variant | requests | admission sheds | 507 refusals | requests restoring with restore_copies_prefix=true and reuse>0 (admission pricing) |
|---|---|---|---|---|
| A1 | 27 | 8 | 0 | 8 |
| A2 | 27 | 13 | 1 | 12 |
| B1 | 27 | 12 | 1 | 11 |
| B2 | 27 | 8 | 2 | 12 |

### Lease-labelled follow-ups and admission copy pricing (c50/c80, t1-t5)

| Variant | turns with mode=reference_lease | of those with copies=True and reuse>0 in admission | typical restore_copy_bytes GiB (c50 / c80) |
|---|---|---|---|
| A1 | 0 | 0 | - / - |
| A2 | 0 | 0 | - / - |
| B1 | 10 | 5 | - / 5.6 |
| B2 | 10 | 5 | - / 5.6 |

Notes on the tables: `active GB` is MLX active memory after the request (includes the bank); `peak GB` is the monotone process peak from the request log. `restore_copy_bytes` and `growth_bytes` are only priced by the admission in the refined pass; for most 50K follow-ups the first (pessimistic, `reused_tokens=0`) pass already fits and no restore-aware pricing is logged. Where it is logged (80K follow-ups, fork, go-back) `restore_copies_prefix` is true in both variants.

## 3. Findings

Established:

1. The flag changes the label, not the cost. In B all 10 same-session follow-ups (c50 and c80, t1 to t5) report `session_restore_mode=reference_lease`; in A all report `clone`. TTFT, suffix prefill (about 280 tok/s at 50K, 236 tok/s at 80K), decode tok/s and ms per verify of A and B are within the A-A and B-B spread (c50: 3.10 to 3.16 s TTFT, 27.2 to 27.6 tok/s, 92 to 94 ms per verify in all four runs).
2. No memory saving. In the 50K conversation active memory after turn 5 is 32.8 GB (A1, A2) against 32.6 GB (B1, B2); process peak 33.9 against 33.8 GB; growth per turn identical (+4.0 GB at turn 1, then +0.3 to 0.5 GB). With a lease the conversation would be held once; the numbers show it is held twice in both variants.
3. The admission agrees: in B the 80K follow-ups that are priced restore-aware still have `restore_copies_prefix=true` with `restore_copy_bytes` about 5.5 to 5.8 GiB (c80, 84K to 89K reused tokens), the same as A. This is the documented case in `_admission_restore_copies_prefix`: a lazy snapshot beside the live cache (left by every generation-final commit) makes the first write copy the buffers like a clone. The lease only avoids the copy for lease-only entries or settled snapshots; that situation did not occur under the production env in these runs.
4. Fork (scenario 2) behaves the same in A and B at 50K: block-prefix boundary hit at 49,152 of 54,047 tokens, suffix prefill 4.9K tokens in 14.5 to 15.8 s (the mode is labelled `block_prefix_boundary_reference_lease` in A as well). At 80K the fork was a cold 84K prefill (250 to 257 s) in A1, A2 and B1 and a near-prefix hit (1.0 s prefill) in B2: this follows the memory state (sheds and releases of idle sessions), not the flag. The fork does not break the owner's lease: the next turn of session 1 worked in B1 (c50-after-back).
5. Go-back (scenario 3) is a boundary hit at 49,152 of 56,738 tokens in every variant, restore mode `block_prefix_boundary_clone` also in B (a lease is only taken on the live frontier), 22.8 to 24.1 s for a 7.6K-token suffix. Its output is token-identical in A1, A2, B1, B2. At 80K it went through the SSD tier (`ssd_clone`, 11.7 tok/s decode and about 210 ms per verify on the following requests, versus 27 tok/s) or was refused (see 6).
6. Memory refusals occur in both variants, caused by system memory (other apps leave 3.6 to 9.2 GiB free) after reclamation: 507 refusals A1 0, A2 1, B1 1, B2 2 out of 27 requests each; admission sheds 8, 13, 12, 8. The numbers are too small to rank the variants. B1 and B2 hold more after the fork/go-back (active 37.1 and 41.2 GB against 32.5/37.2 in A1/A2), but A2 reaches 37.2 as well, so this follows the reclamation history rather than the flag.
7. The `anon` scenario (no header, tools) restores via `block_prefix_boundary_clone` in all variants, identical modes and TTFT: the flag has no effect on traffic where the prior answer is not echoed (restore at the previous prompt end, 47 % of production restores according to the 2026-10-06 report).

Output identity (greedy, `text_sha` per request over A1, A2, B1, B2):

- c50 t0 to t5 and c50-back: identical in all four runs (6 follow-ups plus go-back, 50K). Lease output equals clone output there.
- c50-fork: A1 = A2 and B1 = B2, but A differs from B (diverges at character 40 against 1,015 for B versus the cold reference `REF`). Both differ from the cold reference eventually; B is closer to it. Suspected: small numerical differences between restore paths (bf16, near-tie logits), not a correctness fault; not proven.
- c80 t0 to t5: A1 = B2 and A2 = B1, i.e. pairs are split across A and B. The split follows the first request (t0 restored the 4K head, prefill chunking depends on the memory state at that moment), not the flag. Within each lease run the follow-ups are consistent.
- s30 (sampled) and anon: differ between runs (restore path and chunk plan differ per run); A1 and A2 also differ from each other there, so no A/B conclusion can be drawn.

Not established:

- Whether a lease saves memory when the bank snapshot is dropped (per-session cap, no SSD incremental encode). Not tested: the production env was the brief.
- Real Claude CLI traffic (metadata session id, many tools); the `anon` scenario imitates its shape only.
- Causes of the greedy differences between restore paths (only observed).

## 4. Risks and advice

- Risk of enabling: a different restore path for the live frontier (outputs equal at 50K and 80K for the same-session follow-ups, but not guaranteed numerically equal), lease-only entries cannot be reclaimed by the owner's own request (B2 507 text: `lease-c50 3.8 GiB (this_request)`), and the admission still prices a full prefix copy, so no relief from 507 refusals or sheds.
- Advice: do not add the flag to the production launch command on this evidence: no gain in memory, TTFT, prefill or decode speed; the label changes and the risk profile is slightly different. The duplicate copy has to be removed at the snapshot (lazy snapshot aliasing the live cache), not by the restore mode. The 80K go-back and fork cases dominate the cost (cold 250 s or SSD restore) and are not improved by the flag.

Thermal and noise: every variant reached thermal level 1 to 2; swap used by the system moved between 3.3 and 6.2 GB during A1 (other apps), with no abort (limit 3 GB growth); highest engine peak 48.4 GB (A1, an 80K SSD restore), 43.6 GB in B, 39.5 GB in A2.
