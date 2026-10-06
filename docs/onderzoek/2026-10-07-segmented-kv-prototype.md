# Segmented KV cache prototype (phase 1, outside the server)

Date: 2026-10-07. Status: measurement and prototype; no MTPLX code changed (production tree and worktrees only imported, read-only; nothing pushed; port 8000 and `config/config.json` untouched).
Raw data, scripts, logs: `/Users/joonix/Dev/laya-nl/segment-kv/` (`segkv.py` library, `k1_correct.py`, `k2_speed.py`, `k3_model.py`, `k4_memory.py`, `k5_merge.py`, `prefill_ref.py`, `run_gated.sh`, `data/`, `logs/`).

## 1. Idea and question

A conversation is a list of immutable KV segments, one per turn. A follow-up turn writes only into a new (growing) tail segment; sealed segments are never written again. A session snapshot, a subagent or a return to an earlier point is then a list of references to the same segments, so MLX copy-on-write never triggers (see `2026-10-06-paged-prefix-sharing.md`, `2026-10-07-lease-vs-clone.md`: +5 to 6 GB at 80K on Qwen3.8-27B whenever the new turn slice-updates a buffer that a snapshot also references).

Questions: (1) is attention over a segment list exact, (2) what does each extra segment cost, (3) does the duplicate disappear, (4) what is needed for prefill.

## 2. Method

Machine and model as in the lease-vs-clone report (Qwen3.8-27B Optimized-Speed, 16 full-attention layers, head_dim 256, 24 q / 4 kv heads, bf16 KV). GPU discipline: shared lock `GPU.lock`, gate (no other process above 2 GB, no `BUILDING`, thermal 0, 60 s rest) before each run, thermal logged per run (`logs/*.thermal`; rows carry `thermal_before/after`). Variants interleaved (order alternated per round, ABBA-style), median of 9 chain executions per round, 5 rounds; speed tables report the median over rounds. Kernel tests use stock MLX 0.32.2 (the kernels are custom Metal); model and prefill runs use the production combo MLX (`pkg-combo`) and the production launch env (DSPLIT4=1, WIDE=1, MULTIROW_QMM=1, `apply_profile_env("turbo")`).

### 2.1 Merge route (no kernel change needed for the math)

`_dsplit_dispatch` already produces per-block partials (`partials`, `sums`, `maxs`) and a reduce kernel merges them exactly (global max, exp-sum, weighted sum). Segments are just more blocks: the partial kernel is launched per segment, the partials are concatenated along the block axis (or read by a reduce that takes a list of buffers) and ONE reduce merges all.

Only one thing is not possible with the existing inputs: "all keys visible". The kernel derives `tail_lo = n_kv - QL` from the same `offset` that bounds the keys, so the early query rows would not see the last QL-1 keys of an old segment (padding with dummy keys would attend them). The prototype therefore patches the kernel source at import time (read-only import, patch in memory): `gp <= row_limit` becomes `(NOMASK != 0 || gp <= row_limit)`, NOMASK a template constant. Old segments run with NOMASK=1, the last segment (holding the verify/draft rows) with the stock tail-causal mask. Everything else in the kernel is byte-identical.

Three routes were built and measured:

- `concat`: per-segment partial launches (blocks padded to a multiple of 32), partials concatenated, production reduce.
- `multi`: per-segment partial launches, local reduce that takes the list of partial buffers directly (block counts need no padding; `starts` prefix table). Same arithmetic as `_paged_reduce_kernel`.
- `fused`: ONE launch for up to 12 segments (Metal allows 31 buffer bindings: q + 2 per segment + meta + scale + 3 outputs). grid.z walks the blocks of all segments; a `meta` array (block starts, lengths, capacities) selects segment and buffers inside the kernel; the same `multi` reduce follows. 20 segments = 2 launches.

Block size per segment: the target chunk (keys per block) of the production dispatcher for one contiguous buffer of the same total length.

## 3. Results

### 3.1 Kernel correctness (established)

`k1_correct.py`, `data/k1_correct.jsonl`, `data/k1b_fused.jsonl`: 704 + 352 cases; totals 50K and 80K keys; segments 1, 2, 3, 6, 12, 20 (even and "agentic" uneven splits); q_len 1..5 and 9, 17, 25; inputs "normal" (scores sigma 1) and "peaked" (queries x5, a few dominating keys, stresses the max merge); fp32 reference with the tail-causal mask.

| route | dist | segmented vs fp32: max abs / rel L2 | contiguous production kernel vs fp32: max abs / rel L2 |
|---|---|---|---|
| multi / concat / fused | normal | 1e-4 to 2e-4 / 2.84e-3 | 1.2e-4 / 2.84e-3 |
| multi / concat / fused | peaked | 1.43e-2 / 2.1e-3 | 1.43e-2 / 2.1e-3 |

Segmented error equals the contiguous kernel's error (both are the bf16 output/partial rounding floor, rel 2.8e-3). Segmented versus contiguous differ by at most one bf16 ulp (max abs 0.0156 at |out| about 2 to 4); 1 segment is bit-identical to the production kernel. No NaN in any case. The merge is exact up to summation order; it is not bit-identical to the contiguous kernel for more than one segment.

### 3.2 Kernel speed (established), ms per layer, dependent chain of 16 layers with own buffers, glue subtracted

`k2_speed.py`, `data/k2_speed.jsonl` (concat, multi), `data/k2b_fused.jsonl` (multi, fused). q_len 4 (shipping verify), even segments; thermal 0 before and after except where the rows say otherwise.

| ctx | segments | contiguous | concat | multi | fused |
|---|---|---|---|---|---|
| 50K | 1 | 0.867 | 0.855 | 0.855 | 0.871 |
| 50K | 2 | 0.858 | 0.942 | 0.918 | 0.858 |
| 50K | 3 | 0.855 | 0.939 | 0.924 | 0.856 |
| 50K | 6 | 0.855 | 0.986 | 0.997 | 0.863 |
| 50K | 12 | 0.856 | 1.162 | 1.031 | 0.870 |
| 50K | 20 | 0.855 | 1.425 | 1.091 | 0.867 |
| 80K | 1 | 1.313 | 1.346 | 1.335 | 1.308 |
| 80K | 2 | 1.309 | 1.409 | 1.369 | 1.331 |
| 80K | 3 | 1.313 | 1.494 | 1.425 | 1.366 |
| 80K | 6 | 1.318 | 1.511 | 1.446 | 1.349 |
| 80K | 12 | 1.310 | 1.638 | 1.517 | 1.344 |
| 80K | 20 | 1.310 | 1.904 | 1.598 | 1.430 |

(Baseline column from `k2`; the fused columns come from the second run `k2b` whose own baselines agree within 2 %. Run-to-run noise of single cells is about 0.03 ms.)

Cost per extra segment at q_len 4: concat about 0.03 ms/layer, multi about 0.012 to 0.016 ms/layer (linear in the segment count), fused about 0.0 to 0.005 ms/layer up to 12 segments (0.09 ms at 20 segments at 80K, where it needs two launches; 0.004 ms at 50K). q_len 17 (wide route): same picture, fused within +0.1 ms of 4.7 ms. Over 16 layers: multi costs about 0.2 to 0.25 ms per forward per extra segment, fused about 0.

Reading (supported by the fused result, but not isolated with a GPU trace): the cost per segment of `multi` is launch/encode overhead of one extra kernel per segment, not bandwidth; the bytes read are identical. It would largely disappear inside the compiled verify graph; `fused` removes it in eager mode as well.

### 3.3 Model level, verify forward (established, eager)

`k3_model.py`, `data/k3_model.json`, `logs/k3.log`. Real prefill of 50K and 80K tokens (corpus: repository docs), then the whole `target_verify` forward with the 16 attention layers replaced by `SegKVCache` (sealed segments + growing tail; the tail capacity is kept at 8192+ because the production route gates on `cache.keys.shape[2] >= 8192`). The segmented route is installed by replacing the two dsplit entry points that `attention_split.py` imports at call time. Eager, not `mx.compile`: the variants run in one process in shuffled order (a compiled verify would freeze the kernel choice per process). Medians of 7 reps. Thermal was 1 (50K) and 2 (80K) during these runs (prefill heats the machine); variants are interleaved so differences are comparable, absolute times are not equal to the cold numbers of other reports.

| ctx | rows | contiguous | 6 seg multi | 6 seg fused | 20 seg multi | 20 seg fused |
|---|---|---|---|---|---|---|
| 50K | 4 | 87.1 | 90.1 | 87.0 | 90.2 | 87.9 |
| 50K | 17 | 176.0 | 178.3 | 175.7 | 180.6 | 176.7 |
| 80K | 4 | 99.2 | 101.8 | 100.3 | 102.2 | 100.1 |
| 80K | 17 | 224.0 | 224.6 | 223.5 | 227.8 | 230.5 |

ms per verify forward. 1 segment (own route, shared buffers) is within 0.5 ms of the contiguous time and bit-identical in logits (max abs diff 0.000). With `fused`, 6 and 20 segments cost 0 to 1.1 % (80K, 4 rows: +1.1 / +0.9 ms), with `multi` 2.6 to 3.6 % at 4 rows. The 80K/17-row 20-segment fused cell (+2.9 %) contradicts the kernel-level result and is within the observed noise (about 2 %) at thermal 2.

Logits: with 6 or 20 segments max abs logit difference 0.16 to 0.34 against logits up to 30.6 (bf16 resolution at 16..32 is 0.125, so 1 to 3 ulp), argmax equal for all 4-row windows. In the 17-row windows at 50K, 1 to 2 of 17 argmaxes differ; a re-run that logged the top-1/top-2 margin of the base logits (`logs/k3_margin50b.log`) shows that every differing row is a bf16 tie in the base logits (margin 0.0, once 0.125 = 1 ulp), so these are not errors of the merge (at 20K, 17 rows, 20 segments: no differences).

### 3.4 Memory (established), `k4_memory.py`, `data/k4_memory_{50000,80000}.json`

16 layers of K and V, (1,4,cap,256) bf16, follow-up turn of 1024 tokens, `mx.get_active_memory()`:

| scenario (80K) | active memory change |
|---|---|
| contiguous, turn written, no second reference (control) | 0 (in-place) |
| contiguous, turn written while a snapshot slice aliases the buffer (lazy or evaluated) | +4.95 GiB (exactly one copy of the buffers) |
| segmented, new turn in a new segment, 2 snapshots (reference lists) alive | +0.125 GiB (= the new segment, 1024 + 1024 slack rows) |
| segmented, 8 verify steps writing 4 rows into the tail and attending over all segments | +0.03 GiB, peak +0.1 |
| branch from the same history | 0 (list of references) |
| fork inside a sealed segment: reference (buffer, n-300), no copy | bit-identical output to a private copy of the prefix |

50K: contiguous +3.125 GiB, segmented +0.125 GiB. The duplicate is gone in the mechanism that matters (the new turn never writes into a buffer another reference holds). The numbers are from a synthetic cache with the real shapes, not from a server run.

### 3.5 Prefill (estimate; not built)

`prefill_ref.py`, `data/prefill_ref.json`. Queries = new tokens, keys = old segments (fully visible) + themselves (causal). Correctness of the lse merge, one layer, 128 new tokens, 80K old keys: fp32 manual attention per segment that returns logsumexp, merged with `exp(l_s - L)` weights: rel error 3.0e-4 / max abs 9e-6 against the fp32 reference for 2, 6 and 20 segments (this is the fp32 matmul noise floor of this GPU path, 15 times below the bf16 level); with bf16 `mx.fast.scaled_dot_product_attention` per segment plus a separate lse pass: rel 4.7e-3 versus 4.4e-3 for the contiguous bf16 SDPA (equal within bf16). So the merge is correct if the per-segment logsumexp is known.

`mx.fast.scaled_dot_product_attention` returns no logsumexp (in MLX `scaled_dot_product_attention.cpp`, `output_logsumexp` exists only for the VJP fallback and the fused forward raises). With today's primitives the lse needs a separate QK^T + logsumexp pass. Timing, one layer, 2048 new tokens over 80K keys (thermal 0 to 2 during the run, so noisy): contiguous causal SDPA 198 to 225 ms; per-segment SDPA without lse 217 to 271 ms (at 20 segments equal to contiguous; the one-segment non-causal call was slower than the causal contiguous call for reasons not investigated); separate lse pass (matmul + logsumexp, naive) 330 ms; total estimate about 560 to 600 ms, i.e. about 3 times contiguous. At 4096 new tokens the ratios are the same (443 versus 1150 to 1240 ms).

What a fast prefill needs (supposition): the steel/NAX attention kernel keeps the running max and sum internally; writing `L = m + log(l)` per row as a second output (fp32, (B,H,Q)) makes the per-segment lse free. Then the cost is per-segment SDPA (about equal to contiguous in total) plus an elementwise merge of the 24 x Q x 256 outputs, estimated at 1.0 to 1.3 times contiguous. Alternative that needs no kernel change: keep prefill on a contiguous staging buffer for the new turn only (queries attend to the old segments through the decode-style kernels in chunks) - not measured.

### 3.6 Merge cost (established), `k5_merge.py`, `data/k5_merge.json`

Copying two small segments into one, 16 layers x K,V: 512+512 tokens 0.81 ms (concatenate; 1.15 ms with slice-assign), 1K+1K 1.19 ms, 2K+2K 2.16 ms, 4K+4K 4.10 ms (about 125 GB/s effective). Against the per-segment attention cost per verify forward (16 layers): multi 0.2 to 0.25 ms, fused about 0 (up to 12 segments). A 1K+1K merge pays back after about 5 forwards with multi; with fused the attention cost does not justify merging, only segment count, bookkeeping and SSD block granularity do.

## 4. Phase 2 sketch (go, behind a switch)

Verdict: GO for phase 2 behind a switch, with `fused` as the attention route. Reasons: exact merge; kernel cost per segment about zero with fused up to 12 segments; verify forward +1 % at 6 to 20 segments; the duplicate disappears in the measured mechanism. Not covered by phase 1: the compiled verify route, the prefill speed, a server run.

- Data structure: per full-attention layer a `SegmentList`: sealed segments `(K, V, n, id)` with exact capacity, plus one growing tail `(K, V, n_used, cap)`. Rope offset = sum of lengths. Attention gate: the capacity check of the packed route (>= 8192) must look at the total length, not the tail capacity. GDN layers stay as today (small state, copied).
- Session bank: an entry stores a list of segment ids (references) + the recurrent anchors; the snapshot IS the list. Sealing the tail at the end of a turn (trim slack by a copy of the tail only) makes it immutable. Fork or return into the middle of a segment: reference `(buffer, n')`, no copy (bit-identical, measured).
- Admission gate: segments shared by several entries are charged once (reference-counted bytes); a new turn costs only the new tail; the 507 pricing for clone/lease disappears for the history part.
- SSD tier: the existing 256-token block hashing maps onto segments (a segment = whole blocks, sealed at a block boundary); incremental encode writes a sealed segment once; restore loads segments directly into sealed buffers.
- Merge policy (tiered, log-structured, as agreed): never merge the large history segment eagerly (a full copy brings the duplicate back while a snapshot or subagent references it). Merge only small recent segments among themselves in geometric tiers (0.5K+0.5K -> 1K, 1K+1K -> 2K, ...), which keeps 2 to 3 live segments and copies only fresh tokens (0.8 to 4 ms for 16 layers, 3.5). Fold history and the recent tier together only in the background when idle and only when no snapshot or lease references the old segments (reference count 1). Because fused makes the segment count nearly free for attention, the policy can be lazy: its job is bounding the segment count (<= 12 per launch) and SSD granularity, not speed.
- Open: compiled verify (meta array as traced input, segment count fixed per trace or padded), prefill lse, interplay with `one_copy.py` (QSA buffers) and the lease modes, MTP-history rows, behaviour at 128K+ and under the 20-segment two-launch case.

## 5. Claims versus supposition

Established (measured): exactness and error figures (3.1); per-segment kernel cost for three routes (3.2); verify-forward cost and logits (3.3); memory mechanism (3.4); merge copy cost (3.6); the lse merge is exact in fp32 (3.5); the argmax differences are bf16 ties (3.3).
Supposition: that the multi route's per-segment cost is launch overhead (supported by fused, not traced); that a compiled verify removes it; the 1.0 to 1.3 x prefill estimate with an lse-emitting kernel.
