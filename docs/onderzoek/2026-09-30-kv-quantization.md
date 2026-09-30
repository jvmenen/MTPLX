# KV cache quantization at long context (30 September 2026)

Question: can `--paged-kv-quantization q8` reduce memory at long context on Qwen3.8-27B (and Qwen3.6-35B-A3B) without losing speed? Background: [VONDSTEN-QWEN38](VONDSTEN-QWEN38.md) Q4 (q8 measured -9% at short context) and [feature-gaps](2026-09-30-feature-gaps.md) rank 8.

**Short answer: no, not today.** On both models q8 (and q4) is slower at every context length measured and does not shrink the session bank, which is where the memory pressure comes from. The main cause of the slowdown is a mask check that sends every verify round to a full dequantization. A small fix (branch `perf/kvq8-compiled-verify`) recovers about half of the loss, but q8 stays 11% (20K) to 18-27% (60K) slower than KV off. Recommended setting for both models: KV off.

## Setup

- Hardware: M5 Pro, 64 GB. Code: `perf/definitief` a5df61d0 (read-only) and `origin/main` 1de2b1c0 plus the prototype (6ce1fad4).
- Qwen3.8-27B: Youssofal Optimized-Speed, final-test-3 arguments (turbo, depth 3, draft temperature 0.6), Bink production env. Qwen3.6: Balance-bf16mtp-yb pack with the production arguments (depth 2).
- Fresh server per variant on port 8000 under the GPU lock, swap guard (+2 GB aborts). Prompt: real repository markdown cut to 20K and 60K tokens with the local tokenizer, plus a Dutch summary question, reasoning on. Per length: one warm-up request (cold prefill, fills the bank), three sampled requests of 800 tokens and one greedy request (temperature 0, 400 tokens) for a token comparison. Memory from `/v1/mtplx/snapshot` `mem`, bank entries from `/health`.
- Scripts (local, not in the fork): `~/Dev/laya-nl/kvquant/` (`kvq.zsh`, `kvq.py`, `analyse.py`, `kernelbench.py`), results in `res/`.

## Results

Decode tok/s is the median over the three sampled runs and the greedy run. "Round" is verify forward plus eval per verify call.

### Qwen3.8-27B

| Variant | Context | Decode tok/s | vs KV off | Acceptance D1/D2/D3 | ms per round | Compiled / eager rounds | Active GiB | Peak GiB |
|---|---|---|---|---|---|---|---|---|
| KV off (main) | 20K | 29.2 | | 0.82 / 0.64 / 0.49 | 94 | 651 / 93 | 24.5 | 26.5 |
| KV off (definitief) | 20K | 29.7 | | 0.84 / 0.66 / 0.47 | 92 | 659 / 89 | 24.5 | 26.5 |
| q8 | 20K | 22.0 | -25% | 0.82 / 0.62 / 0.46 | 119 | 564 / 0 | 23.3 | 25.9 |
| q4 | 20K | 19.7 | -33% | 0.80 / 0.61 / 0.48 | 140 | 761 / 0 | 22.6 | 25.4 |
| q8 + mask fix | 20K | 25.9 | -11% | 0.82 / 0.63 / 0.46 | 104 | 766 / 0 | 23.3 | 25.6 |
| KV off (main) | 60K | 23.6 | | 0.82 / 0.63 / 0.45 | 113 | 666 / 116 | 28.3 | 33.0 |
| q8 | 60K | 9.6 | -59% | 0.82 / 0.65 / 0.50 | 306 | 0 / 769 | 26.4 | 34.1 |
| q4 | 60K | 6.7 | -72% | 0.82 / 0.62 / 0.47 | 433 | 0 / 752 | 25.4 | 32.5 |
| q8 + mask fix | 60K | 17.3 | -27% | 0.80 / 0.64 / 0.46 | 161 | 0 / 786 | 26.4 | 35.2 |
| q8 + mask fix, no 32K compiled fence | 60K | 19.3 | -18% | 0.81 / 0.62 / 0.45 | 139 | 774 / 0 | 26.5 | 31.8 |

The 20K and 60K rows of the q8/q4 variants ran in one server (60K after 20K); the KV-off rows on main ran 20K and 60K on separate servers. The first KV-off run on `perf/definitief` (20K then 60K in one server) hit the swap guard during the 60K warm-up (swap 3.6 to 7.4 GB) with three 20K entries (7.4 GiB) in the bank; the q8/q4 runs and the separate 60K run did not swap. With other processes on the Mac this is one observation, not a proven difference.

Session bank, bytes per entry (from `/health` evictions):

| Model | Prefix | KV off | q8 |
|---|---|---|---|
| Qwen3.8-27B | 20K | 2.63 GB | 2.63 GB |
| Qwen3.8-27B | 60K | 5.57 GB | 5.41 GB |
| Qwen3.6-35B-A3B | 60K | 1.93 GB | 1.93 GB |

The marginal growth between 20K and 60K is ~69 KB per token with q8, the bf16 size (64 KB) and not the q8 size.

### Qwen3.6-35B-A3B (Balance-bf16mtp-yb), 60K

| Variant | Decode tok/s | vs KV off | Acceptance D1/D2 | ms per round | Active GiB | Peak GiB |
|---|---|---|---|---|---|---|
| KV off | 63.7 | | 0.80 / 0.60 | 33 | 32.6 | 34.2 |
| q8 | 53.0 | -17% | 0.83 / 0.63 | 43 | 33.7 | 37.1 |

This pack verifies eager (`permanent_eager`), so q8 runs the stock paged q8 two-pass kernel (~3,000 kernel calls per request), not the compiled adapter. Active memory is 1.1 GiB higher and peak 2.9 GiB higher with q8.

### Output quality

Greedy output is not token-identical to KV off in any quantized variant; the first differing token lies between position 34 and 325 of 400 (q8 20K: 34; q8 + fix 20K: 325; q8 + fix 60K: 51; q4 20K: 56; Qwen3.6 q8: 143). The unfixed q8 at 60K happened to match KV off for all 400 tokens. Reading the texts: same structure and content, different wording; no visible quality loss. Acceptance does not change measurably, which fits: the draft and target see the same cache.

### Attention kernel per layer (synthetic tensors)

Qwen3.8-27B geometry (24 query heads, 4 KV heads, head dim 256), verify window 4 rows, median of 30 calls, ms per layer. `kernelbench.py` on `perf/definitief`.

| Context | bf16 NAX flash (dsplit) | bf16 packed | q8 packed-quant | q8 paged two-pass | q8 dequant + SDPA | q4 packed-quant | q4 dequant + SDPA |
|---|---|---|---|---|---|---|---|
| 4K | 0.37 | 0.39 | 0.44 | 0.45 | 1.01 | 0.43 | 1.50 |
| 20K | 0.73 | 0.95 | 1.17 | 1.35 | 4.05 | 1.21 | 6.89 |
| 60K | 1.55 | 2.21 | 2.86 | 3.43 | 11.77 | 2.80 | 21.03 |

The quantized kernels read half (q8) or a quarter (q4) of the bytes and are still 1.6 to 1.8x slower than the bf16 TensorOps flash kernel that KV off uses. The dequant path is 4 to 8x slower again.

## Why: code analysis (`perf/definitief` a5df61d0)

1. **Compiled verify does not detach under q8 (the help text is outdated).** Since commit 482fb423 ("kv-quant 0.4", 26 Aug) the compiled verify bank promotes quantized pages to `TensorOffsetQuantizedPagedKVCache` (`graphbank.py:1170-1193`, class at `cache_state.py:3184`): five compiled leaves (int8 banks, offset, fp32 scale planes), quantize-on-write inside the graph, attention through `sdpa_gqa_packed_tail_quant`. Measured: at 20K all verify rounds ran compiled (0 fallbacks). The sentence "compiled-verify/dense-two-pass fast paths detach while active" in `cli.py:866-879` predates that commit.
2. **The real detach is the layout.** KV quantization forces `contiguous_then_repage` (`generation.py:2101-2104`), so decode runs on paged caches. The turbo profile's compiled-verify fence `MTPLX_COMPILED_VERIFY_MAX_CONTEXT=32768` (`profiles.py:842`) is only checked for paged entries (`graphbank.py:4210-4220`, `3478-3488`); dense KV-off caches keep compiled verify at 60K (666 compiled rounds), q8 falls back to eager above 32K (`context_above_threshold`, 769 eager rounds). The bf16 NAX flash route is dense-cache only (`attention_split.py`, packed route), so q8 also loses that kernel.
3. **The main cost: a mask check.** `TensorOffsetQuantizedPagedKVCache` inherits `make_mask` from the bf16 paged adapter, which always returns the capacity-wide tail-causal bool array (`cache_state.py:2972-2981`). Its `paged_attention` refuses every array mask (`cache_state.py:3369-3370`), so `attention_split.py:408-433` falls back to `cache.state`, which dequantizes the whole capacity to fp32 and back (`cache_state.py:3395-3408`) and runs SDPA over it, for every full-attention layer in every verify round, compiled and eager. That is the dequant column in the kernel table: 16 layers x 11.8 ms at 60K, which matches the measured 306 ms per round. The bf16 adapter has an opt-in escape for the same mask (`MTPLX_PAGED_TAILMASK_ELIDE`, `cache_state.py:2999-3013`); the quantized adapter never got one.
4. **The bank stores bf16.** `VllmMetalPagedKVCache.state` (`cache_state.py:2079`) returns `_active_arrays` (`1997-2016`), which for q8 builds a bf16 mirror of the whole prefix (`_dequant_active_arrays`, `1754-1822`); `snapshot_cache` and the lazy hybrid snapshot (`4346`, `4375`) store `state`. So bank entries, and therefore bank trimming and the SSD cache, see bf16 sizes, and building the mirror at snapshot time adds a transient bf16 copy (the higher peak on Qwen3.6). Restoring re-quantizes (`state` setter, `2083-2095`).
5. **Per verify round under q8**: below the two-pass threshold (1024 tokens, `MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD`) the eager route latches "dequant" (the bf16 mirror); from 1024 on it latches "kernel" (`sdpa_2pass_paged_q8_tail` for q8, the packed-quant bank for q4), once per request (`cache_state.py:1357-1406`). On Qwen3.8-27B that eager route only runs for the few rounds outside the compiled bank; the compiled adapter and the >32K eager rounds go through the adapter and hit point 3.

## Prototype: tail-mask elide (branch `perf/kvq8-compiled-verify`)

Commit 6ce1fad4 on `origin/main`: `MTPLX_KV_QUANT_TAILMASK_ELIDE=1` (default off) lets the quantized adapter drop exactly its own capacity-wide tail mask, whose visibility equals the kernel's built-in tail-causal walk; any other mask still declines. The shape check is shared with the bf16 elide (`_is_own_tail_mask`). Five new tests (`tests/test_kv_quant_tailmask_elide.py`: switch off declines, kernel output equals the dense fallback for q8 and q4 at three shapes, foreign masks decline); 143 tests in the related files pass; ruff adds no findings.

Effect on Qwen3.8-27B: q8 20K 22.0 to 25.9 tok/s (54% of the loss recovered), 60K 9.6 to 17.3 tok/s (55%); with the 32K fence lifted for this run (`MTPLX_COMPILED_VERIFY_MAX_CONTEXT=0`) 19.3 tok/s at 60K (69%). The rest of the gap is the kernel: at 60K the packed-quant kernel costs ~1.3 ms more per layer than bf16 NAX flash, 16 layers x 1.3 = ~21 ms, against a measured difference of 26 ms per round (139 against 113). Not bit-identical to the unfixed q8 path (different attention math: kernel instead of dequant + SDPA), which is expected.

## What q8 would need to be worth it

| Step | Effort | Effect (estimate unless marked measured) |
|---|---|---|
| 1. Tail-mask elide | done (S) | measured: -25% to -11% at 20K, -59% to -27% at 60K |
| 2. Compiled verify above 32K for quantized paged caches | S (fence per cache type) | measured with the fence off: -27% to -18% at 60K; needs a parity check at long context |
| 3. A TensorOps (NAX) q8 kernel with dequant in registers | L | the remaining ~18%; if it reaches bandwidth, q8 could beat bf16 at 60K (half the bytes: ~0.8 against 1.55 ms per layer) |
| 4. Bank snapshots of the quantized payloads instead of the bf16 mirror | L (29 snapshot/restore call sites, SSD format, prefix-boundary slicing) | the actual memory goal: 60K entry on 27B ~5.6 to ~3.6 GB, on 3.6 ~1.9 to ~1.3 GB; no transient bf16 mirror at snapshot time |

Only step 4 addresses bank trimming, and only steps 1 to 3 together make q8 speed-neutral. Until both exist, KV off is better on both models: live KV at 60K on the 27B shrinks by only ~1.9 GiB (active 28.3 to 26.4 GiB) while peak memory is higher and the bank is unchanged.

## Recommendation

- **Qwen3.8-27B: KV off.** q8 costs 25% (20K) to 59% (60K) decode and saves ~2 GiB live at 60K, nothing in the bank. With the prototype still -11% to -27%. q4: worse still.
- **Qwen3.6-35B-A3B: KV off.** q8 costs 17% at 60K and raises active and peak memory.
- **Prototype**: keep on the fork as evidence. A PR is reasonable as a fix of an opt-in feature (it removes a full dequantization per round), after Jeroen's approval; together with a correction of the `--paged-kv-quantization` help text (compiled verify stays engaged up to the context fence).
- **Next step for memory at long context** is the bank itself (step 4) or bank settings, not KV quantization.
