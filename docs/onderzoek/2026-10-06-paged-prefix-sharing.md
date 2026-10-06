# Block-shared KV cache (paged prefix sharing) in MTPLX: feasibility exploration

Date: 2026-10-06. Status: exploration, no code changed. Measurements are partial (see "Not established").
Setup: Mac17,9, 64 GiB, MTPLX prod tree (v2.12.2 + local changes), MLX build `pkg-combo`, model Qwen3.8-27B Optimized-Speed (16 full-attention layers, 24 Q / 4 KV heads, head_dim 256, about 64 KiB KV per token; 48 GatedDeltaNet layers with fixed-size state), production env flags from the launch command, profile `turbo`, depth 3, temperature 0.

## 1. What the existing "paged" mode is

Established (code reading):

- It is a different memory layout, not sharing. `VllmMetalPagedKVCache` stores `[num_blocks, block_size, kv_heads, head_dim]` per layer, default `block_size=16`; its docstring says logical positions are contiguous from zero so "the block table is trivial" (`mtplx/cache_state.py:992-1001`, `:1005`). Calls build `block_tables = mx.arange(used_blocks)` (`cache_state.py:2475`, `:2863`) and writes use `slot_mapping = arange(offset, offset+steps)` (`:1334`).
- The in-tree verify kernels address pages by identity: `page_idx = n / PAGE_SIZE; row = page_idx*PAGE_SIZE + page_offset` (= `n`) in `mtplx/kernels/sdpa_2pass_paged.py:72`. There is no block table input in the dynamic-offset kernel (`cache_state.py:3505-3560`).
- `BlockOwnedKVCache` (`cache_state.py:191`) holds independent per-cache blocks, but only feeds a Python probe (`mtplx/block_attention.py`, "not a final Metal kernel").
- Selection: `_sustained_prefill_layout()` returns `contiguous_dense_decode` when the prompt is at most `_dense_decode_max_context()`, else `contiguous_then_repage` (`mtplx/generation.py:2480-2503`). The ceiling `auto` is 15 % of RAM divided by 65536 B/token (`generation.py:2567-2640`): 157,286 tokens on this machine. `MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS` is set by `generate_mtpk` itself from the prompt length (`generation.py:209`, `:6741`, `:10983`). Quantized KV (q4/q8) forces repaging.
- The code comment at `generation.py:2570-2574` states that the paged verify path cannot use the packed fast lane (the "147.4k decode cliff"). Upstream issue #506 reports the paged lane performing zero calls in production.
- Sessions: the bank does not share blocks between entries in either layout. A restore is `clone` (views of banked arrays, MLX copy-on-write on first write), `reference_lease` (the entry's live cache is handed over) or a snapshot copy (`mtplx/session_bank.py:2440-2540`). A paged lease grows by geometric block steps with one layer copied at a time (`cache_state.py:754`, `server/openai.py:22154+`).
- Production traffic (request-log-8000, 27 Sep to 6 Oct, 22,555 requests): 113 requests (0.5 %) used paged (q4/q8 experiments, at most 75K), none exceeded the dense ceiling (maximum prompt 146,361). Production is effectively dense.
- Existing "block prefix" restore modes (`block_prefix_boundary_*`) are token-boundary matching of banked checkpoints, not KV block sharing. The SSD tier already stores 256-token blocks as sha256 content blobs (`mtplx/cache_bank/cold_tier.py:95`, `codec.py:100`).

## 2. Cost of the paged route (measured, partial)

Method: one process per variant; production env; prefix built into a SessionBank, then a follow-up request restores it (`clone`) and decodes 300 tokens (identical output hash in both variants). Variant selected with `MTPLX_SUSTAINED_PREFILL_LAYOUT` = `contiguous_dense_decode` / `contiguous_then_repage`, verified through `paged_kv_capacity_tokens` (0 vs 74,912). ABBA planned across processes (dense, paged, paged, dense; the last dense run was cut off, so 1 dense and 2 paged processes), 60 s rest, thermal 0 at start, during and at end, no other process above 2 GB, shared GPU lock. `k=0` is the first decode after the restore, `k=1` the second in the same process.

| Context | Layout | Run | ms per verify | tok/s | tok per verify | Peak GB |
|---|---|---|---|---|---|---|
| 50K | dense | 1 k0 | 111.4 | 25.4 | 2.83 | 29.5 |
| 50K | dense | 1 k1 | 111.5 | 25.4 | 2.83 | 29.5 |
| 50K | paged | 2 k0 | 295.5 | 9.6 | 2.83 | 40.0 |
| 50K | paged | 2 k1 | 184.9 | 15.3 | 2.83 | 35.3 |
| 50K | paged | 3 k0 | 251.0 | 11.3 | 2.83 | 40.0 |
| 50K | paged | 3 k1 | 186.1 | 15.2 | 2.83 | 35.3 |

At 50K the paged route is 1.66x slower per verify in steady state (185 vs 111 ms) and 2.3 to 2.7x on the first decode after a restore. The slowdown is far larger than any thermal effect. Paged also peaks 5.8 to 10.5 GB higher (clone plus repaged copy). Not measured: 80K, 120K and about 140K (near the 157K ceiling); the runs were stopped. An earlier run at thermal 2 (discarded) gave dense 80K 135.7 ms per verify; it is only indicative.

## 3. What block sharing would have saved (request log)

- 4,820 requests restored a cache; 497 (10 %) were leases, the rest clones or snapshot copies. A clone of a 50K+ prefix duplicates on average 4.8 GiB (6.1 GiB at 80K+) at 64 KiB per token (upper bound: includes leases).
- Where restores diverge (4,716 restores with at least 1K cached tokens): 47 % at the previous request's prompt end, 19 % at its prompt plus answer end, 24 % inside the previous prompt (median 126 tokens before its end, p90 1,075; 64 cases 10K or more earlier), 8 % first request of a new session (83 % of those 4,096 tokens or fewer, i.e. the system head), 2 % elsewhere. About 90 % of clone restores are same-session follow-ups, which a live-frontier lease already covers.
- Admission sheds with detail: 111 events, 100 with `restore_copies_prefix=true`, summed `restore_copy_bytes` 450 GiB; in 51 the restore copy alone was at least the reported system shortfall. All 111 requests survived through reclamation (allocator pool, prompt publish, LRU entries), at the price of evicted cache.
- 507 refusals: 42 rows (about 12 episodes after client retries). 17 rows are macOS memory pressure from other apps, unrelated. Five rows (52,492-token prompt with 51,858 cached, needing 0.8 GiB; 90,005-token prompt with 88,576 cached, needing 1.8 GiB) would very likely have passed without the copy (3.2 and 5.4 GiB at 64 KiB per token). Inference: refusals carry no `restore_copy_bytes`, so the copy size is computed from token counts.

## 4. What it takes

| Part | Work | Risk |
|---|---|---|
| Verify/decode kernels | Block-table input and indirection in `sdpa_nax_flash_dsplit` (+DSPLIT4, wide), `sdpa_nax_flash`, `sdpa_nax_tile`, `sdpa_2pass*`, `sdpa_gqa_packed*`; dense layout is head-major `[Hk, cap, D]`, paged is token-major | High: performance (Section 2) and the compiled-verify/graphbank shape stability |
| Prefill attention | Suffix prefill reads the shared prefix: needs a block-table-aware attention in the MLX fork (C++) | High |
| Cache class | Pool with in-place writes (a view write makes MLX copy the whole buffer), tail copy-on-write of the last partial block, verify rollback | Medium-high |
| Session bank | Refcounts, byte accounting by unique blocks, lease/clone/snapshot semantics, eviction, SSD tier (already 256 blocks) | Medium |
| Admission | `_admission_restore_copies_prefix`/`_admission_growth` price at most one block instead of the prefix | Low-medium |
| GatedDeltaNet state | Fixed-size state exists only at anchors (`checkpoint_anchors.py`, head anchor); a fork point needs an anchor or a re-prefill from the last one; states are not shareable between diverging branches | Constrains fork points regardless of layout |

Existing maker work: issue #333 ("VLLM Style Paging") was closed by the maker ("MTPLX already moves long contexts to a paged KV cache"); #506 (paged lane unused), #422 (near-prefix restore watchdog), #592 (one-copy resize) are open. The maker's own answer to the duplicate-copy problem is the one-copy store and live-frontier leases (`server/openai.py:21831+`, docstring of `_admission_growth`). No plan for cross-conversation block sharing was found in the repo, CHANGELOG or issues.

## 5. Variant comparison: fixed 256-token blocks vs per-turn extents

| Aspect | Fixed blocks (256) | Per-turn extents |
|---|---|---|
| Kernel addressing | One table lookup per 256 rows inside a chunk; dsplit chunk (`ceil(n_kv/n_blocks)` rounded to TK, `sdpa_nax_flash_dsplit.py:83`) is a multiple of 256 | Per-threadgroup extent table lookup, or chunks that never cross an extent; today's chunks (about 640 rows at 80K with 128 blocks) are not aligned |
| Buffers | One pool array per layer, kernels see one buffer | A custom Metal kernel takes a fixed set of arrays: separate buffers per extent require one launch per extent (breaks compiled verify) or a pool with its own allocator |
| Extents/blocks per session | 80K = 313 blocks | median 6, p90 about 20, max 144 (turns per session); median new tokens per turn 483 |
| Fragmentation | Internal waste at most 255 rows per conversation | Spare decode capacity per segment, external fragmentation in a pool |
| Eviction / SSD | Uniform unit, aligned with the 256-row cold-tier fingerprint | Variable units; remap to 256 blocks for SSD |
| Fork points | Any position (round down to block, copy at most 255 rows) | Turn ends cheap; mid-segment ranges possible |
| Changes in MTPLX | Fewer new concepts, matches existing paged vocabulary | Fewer table entries but new allocator and kernel contract |

Fork points in real traffic: turn ends (66 %) and the head (about 7 % of restores) cover about 73 %; 24 % diverge inside the previous prompt, usually within the last 126 to 1,075 tokens, which fixed blocks handle by rounding down. The GatedDeltaNet anchor constraint means forks only land where an anchor exists anyway.

## 6. Recommendation

Do not build block sharing now. First measure the smaller step, `MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE=1` (lease instead of clone for the live frontier), which targets the 90 % same-session restores without kernel work. Raise the question with the maker before any kernel work. If sharing is pursued, fixed 256-token blocks in a single pool fit the existing kernels, cold tier and admission better than extents.

## Not established

- Decode cost at 80K, 120K and near the 157K ceiling (runs stopped). Raw data and scripts: `/Users/joonix/Dev/laya-nl/paged-verkenning/` (`results.jsonl`, `bench.py`, `driver.sh`, `STATUS.md`).
- Actual memory saved by blocks (only estimated from token counts).
- Whether `LIVE_FRONTIER_REFERENCE_RESTORE` is active in production (not in the launch command; the server was not running to inspect it).
- Kernel indirection cost (no prototype).
