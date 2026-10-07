# SDPA logsumexp output for segmented prefill (phase 2a)

Date: 2026-10-07. Local MLX commit only (nothing pushed, no PR, no issue). Branch `sdpa-lse` (902ea1de3) in worktree `/Users/joonix/Dev/mlx-sdpa/wt-sdpa-lse`, based on fork tag `prod` (v0.32.2 + minq + qmm tile patch). Build: `build-lse`, `pkg-lse` (0.32.2.lse). Scripts, data, logs: `/Users/joonix/Dev/laya-nl/segment-kv/prefill-lse/`.

## 1. Question

Segmented prefill needs, per KV segment, the attention output and the per-row logsumexp (LSE) so segments can be merged exactly. `mx.fast.scaled_dot_product_attention` returned no LSE; a separate QK^T+logsumexp pass cost about 3x. Can the fused kernel emit the LSE for free?

## 2. Prior art (established)

- Open/closed MLX PRs #3306 and #3594 (closed, not merged) added an `output_logsumexp` function constant to the fused kernel; maintainers pointed to the flash-attention VJP work.
- Upstream `origin/main` (2b2bb4979, 2026-10-07; our `prod` base is an ancestor) already contains the kernel half: #4563 "Adding VJP for the scaled dot product attention" adds `save_lse` (function constant 303, buffer 8) to `steel_attention.h` and `steel_attention_nax.h` (including the head-dim-split kernel). It is used only internally for training. **There is no public API** (`return_lse` or similar) upstream.
- Consequence: my kernel edits (same constant id and buffer slot, independent implementation) are redundant if `prod` is rebased on a post-#4563 base. What is missing upstream is only the API plus routing. Recommendation: rebase `prod` and carry over only the API/routing commit (supposition on effort: the host-side changes are small; not tried).

## 3. What changed (on the v0.32.2 base)

- `kernels/steel/attn/kernels/steel_attention.h`, `steel_attention_nax.h` (both NAX kernels: plain and head-dim split): function constant 303 `output_lse`, buffer 8 `float* LSE` laid out `[B, H, qL]`; written after the loop as `(max_score + log2(sum_score)) * ln2` (scores are kept in the log2 domain; sinks included).
- `mlx/backend/metal/scaled_dot_product_attention.cpp`: `lse` pointer through `sdpa_full_self_attention_{nax,metal}`, function constant and pipeline hash, output allocation in `eval_gpu`; `has_fused_kernel` no longer rejects LSE, but rejects it for `qL <= 8` (vector kernels have no LSE; the fallback computes it); `use_fallback` returns "fused" whenever LSE is requested and a fused kernel exists (so also for unmasked head_dim 256, which is otherwise routed unfused); a threadgroup-memory check removes the non-NAX float32 head_dim 192/256 case that could never load (pre-existing latent error under `force_fused`).
- `mlx/fast.cpp`, `mlx/fast.h`: `sdpa_impl` returns `{O}` or `{O, LSE}`; new C++ `scaled_dot_product_attention_lse(...) -> pair<array,array>`; the unfused fallback computes the LSE with existing ops (float32, sinks included). The existing primitive already had a second-output slot (`output_logsumexp_`), reused.
- `python/src/fast.cpp`: keyword `return_lse=False` on `mx.fast.scaled_dot_product_attention`; with `True` it returns `(out, lse)`, `lse` float32 `[B, N_q, T_q, 1]`.
- Tests in `python/tests/test_fast_sdpa.py`: `test_sdpa_return_lse` (fp16/bf16/fp32, head dims 64/80/128/256, qL 1..130, qL<kL, none/causal/array mask, sinks, GQA, against an fp32 reference) and `test_sdpa_return_lse_merge` (3 old segments + causal own segment merged by LSE equals one masked attention). With `MLX_ENABLE_TF32=0 MLX_ENABLE_CACHE_THRASHING_CHECK=0`: test_fast_sdpa.py + test_fast.py 53 passed, 3 skipped, 0 failed. Without `MLX_ENABLE_TF32=0` the same files have 74 pre-existing failures on `prod` too (identical count with pkg-combo), and the new float32 tests also fail there because the fp32 reference matmul runs in TF32 (test limitation, not a kernel error).
- Formatting with clang-format 21.1.8 / black / isort (pre-commit versions).

## 4. Correctness (established)

`corr.py`, `data/corr_notf32.json` (256 cases: bf16/fp16/fp32, head dims 64/80/128/256, qL/kL up to 4096/82048, causal with offset and unmasked, Q contiguous and transposed), fp32 reference, `MLX_ENABLE_TF32=0`:

| dtype | LSE max abs error | output rel L2 vs fp32 |
|---|---|---|
| bf16 | <= 6.7e-6 | 1.7e-3 (bf16 rounding floor) |
| fp16 | <= 6.7e-6 | 4.6e-4 |
| fp32 | <= 1.2e-6 | <= 8.6e-7 |

No NaN. With TF32 on (`corr_nax.json`) the fp32 reference itself is inexact (LSE "errors" up to 2.7e-3 are the reference's TF32 matmul, vermoeden: same kernels give 5e-6 with TF32 off).

Bit-identity with `return_lse` off: `bitident.py`, 924 cases (3 dtypes, 7 head dims, 8 shapes, none/causal/array mask, sinks) hashed in `pkg-combo` and `pkg-lse`: **0 differences**. With `return_lse=True` the output differs from the plain call only where the plain call takes the unfused route (head_dim 256 unmasked, or causal below `MLX_SDPA_D256_MINQ`): `return_lse` always uses the fused kernel.

## 5. Cost of LSE on the fused kernel (established, noise about 3 to 5 %)

`bench_lse_cost.py`, `data/lse_cost.jsonl` (5 ABBA rounds, 16-layer chain; `off` = `force_fused=True`). Fused causal, 80K: ratio on/off 0.998 to 1.013 (Q 512..4096) i.e. zero. Fused unmasked: on/off 0.91 to 0.98 (LSE "on" faster by 2 to 9 %); I have no explanation (vermoeden: different code generation of the specialised pipeline); the point is that LSE is not a cost. Thermal 0 before each cell; several 4096-row cells reached 1 during the run (`thermal_after` in the data).

## 6. Segmented prefill vs contiguous (established)

Setup: one layer, ms per layer from a dependent 16-layer chain with own K/V buffers per layer, glue subtracted, 3 ABBA rounds, median; K/V of old segments are views on capacity-padded buffers. `full` = contiguous causal SDPA over old+new keys. `seg_lse` = per-segment `return_lse` SDPA (unmasked), own segment causal, compiled exact merge in fp32. `seg_nomerge` = same without the merge (isolates the merge cost). Environment: **`MLX_SDPA_D256_MINQ=64` (production), so `full` uses the fused causal kernel at every size**. Verified with `route_check.py` (output bitwise equal to `force_fused`): with MINQ=64 contiguous causal is fused at 64..2048 rows; with the default (1024) it is unfused below 1024 rows; unmasked is unfused by default in both cases, hence `return_lse` (which forces fused) is the only way the segments run fused.

Ratio seg_lse / full (absolute `full` in brackets):

| ctx | segments | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| 50K | 1 | 1.12 (35.5 ms) | 1.04 (63.5) | 1.00 (136) | 0.97 (263) |
| 50K | 2 | 1.07 | 0.97 | 0.99 | 0.97 |
| 50K | 6 | 0.97 | 0.95 | 0.99 | 0.99 |
| 50K | 12 | 0.98 | 0.96 | 1.00 | 0.98 |
| 80K | 1 | 1.17 (49.9 ms) | 1.03 (105) | 1.01 (213) | 1.00 (431) |
| 80K | 2 | 1.05 | 0.97 | 0.96 | 0.97 |
| 80K | 6 | 0.97 | 0.92 | 0.95 | 0.94 |
| 80K | 12 | 0.95 | 0.92 | 0.96 | 0.94 |

Reading: 1.0 or below for 2 or more segments (segments are faster than contiguous causal; supposition: the unmasked segment kernel skips the per-block causal bookkeeping and only the small own segment is causal). The 1-segment cells at 512 rows (1.12/1.17) are the cost of two kernel launches plus the merge for a segment split that is then pointless (old + own); the merge alone is about 4 to 9 ms/layer at 512 rows for 80K (seg_lse minus seg_nomerge, 1-segment cell: 8.5 ms, includes the 24x512x256 fp32 merge; supposition: launch/graph overhead dominates at that size).

Old route (separate QK^T+logsumexp pass, no own-segment LSE, 80K): 2.96x (1 seg, 512), 2.85x (1 seg, 2048), 2.68x (6 seg, 512), 2.67x (6 seg, 2048) of contiguous. The new route replaces that by 0.95 to 1.02x.

With the **default** MINQ (1024), `full` is unfused below 1024 rows: seg_lse/full is 0.61 to 0.74 at 512 rows and 0.92 to 1.02 at 1024 (`prefill_defaultminq.jsonl`). The first partial run (default MINQ, 50K, 1 segment, 512: 49.7 vs 34.7 ms) showed exactly this routing artifact; it is kept separately in `data/old_default_minq/` and is not claimed as a gain. The comparison in the table above uses the fused kernel on both sides.

Thermal: every cell started at 0 (the bench waits and retries up to 3 times); `thermal_after` was 0 for all cells except the 4096-row 50K/80K cells (1). Memory peak 6.5 to 7 GB. One other agent's gated run was interleaved by the lock (gaps in the logs), no overlap.

## 7. Verdict

Phase 2b: GO. The LSE is free on the fused kernel, exact to rounding, and segmented prefill attention is 0.92 to 1.0x of contiguous for 2 to 12 segments (at 80K), with the old route at about 2.7 to 3x. Not covered: end-to-end model prefill, compiled graphs, sinks/GQA shapes of other models, the NAX-less path timings (only correctness checked), `qL <= 8` (stays on the existing decode merge).

## 8. Upstream

Kernel part already upstream (#4563). The remaining change (public `return_lse` keyword, routing so that LSE selects the fused kernel for head_dim 256 unmasked, fallback LSE, tests) is separable and small, but MLX requires a human-written PR description and, per policy, an issue first; the upstream maintainers earlier answered that LSE output is mainly a training concern (#3594), so the use case (exact multi-segment merge for inference) would have to be argued. Not proposed yet.
