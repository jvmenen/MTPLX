# Segmented KV cache on other model families: analysis

Date: 2026-10-09. Status: step 1 of 3 (read-only analysis, no GPU work). Code read on `feat/segmented-kv` (HEAD a33e7856, `~/Dev/laya-nl/mtplx-segkv`) and mlx-lm 0.31.3 in the `~/Dev/MTPLX` venv. Nothing changed in MTPLX. Scripts and notes: `~/Dev/laya-nl/segment-kv/other-models/` (`support_check.py`, `hook_sig_check.py`, `NOTES.md`).
Earlier reports: `2026-10-07-segmented-kv-m3.md`, `2026-10-08-segmented-kv-ssd.md`.

## 1. What segments need

A model uses segments only when `evaluate_model_support` says so at load time (`configure_split_full_attention`):

- (a) every full-attention layer runs MTPLX's split-attention hook: today only gated Qwen3-Next/Qwen3.5/3.6/3.8 attention and plain q/k-norm Qwen3 attention;
- (b) the segment kernel takes the head_dim (128 and 256) and `gqa * q_len <= 32` for at least q_len 1;
- (c) the layer's cache is a plain `KVCache` (`install_segmented_attention_kv_cache` replaces only those; rotating, recurrent and paged caches stay as they are);
- (d) the profile env has `MTPLX_GQA_PACKED_SDPA=1` and `MTPLX_NAX_FLASH_ROUTE=1`, and the GPU has NAX.

Otherwise the cache stays stock, one log line, `/health` reason.

"Checked" below means: `support_check.py` loaded the model lazily (no weights evaluated, CPU only), ran `configure_split_full_attention` with the turbo profile env and `MTPLX_SEGMENTED_KV=1`, and printed the verdict. "Computed" means arithmetic from the config. "From code" means read, not run.

## 2. Per family

| | Gemma 4 31B (pair) | Mistral 7B Instruct v0.3 4-bit | Ternary Bonsai 2 27B | Apodex-1.1-mini |
|---|---|---|---|---|
| attention class | `gemma4_text.Attention` | `llama.Attention` (`mistral` maps to llama) | `qwen3_next.Qwen3NextAttention` (gated) | `qwen3_next.Qwen3NextAttention` (gated) |
| layers with KV | 10 full + 50 sliding (window 1024) | 32 full | 16 full + 48 GatedDeltaNet | 10 full + 30 GatedDeltaNet |
| head_dim | full 512, sliding 256 | 128 | 256 | 256 |
| q / kv heads (GQA) | full 32/4 (8), sliding 32/16 (2) | 32/8 (4) | 24/4 (6) | 16/2 (8) |
| caches | 10 `KVCache`, 50 rotating (`Gemma4RollbackRotatingKVCache` in the pair runtime) | 32 `KVCache` | 16 `KVCache`, 48 `ArraysCache` | 10 `KVCache`, 30 `ArraysCache` |
| draft | separate assistant model (4 layers, all KV-shared with the target) | none: AR path | MTP head | MTP head |
| KV per token (computed) | 80 KiB (full) + fixed 800 MiB for the windows | 128 KiB | 64 KiB (fp16) | 20 KiB |
| max context (config) | 262,144 | 32,768 | 262,144 | 262,144 |
| verdict today (checked, turbo env) | off: `head_dim_512_unsupported` | off: `attention_not_hooked` | supported | supported |

### Gemma 4 31B (`Youssofal--Gemma4-MTPLX-Optimized-Speed`)

How MTPLX loads it (from code): the bundle root has no `config.json` but an `mtplx_pair.json`; `gemma4_pair.resolve_gemma4_pair_paths` finds `target/` (Gemma 4 31B IT, 4-bit, `model_type gemma4`) and `assistant/` (`gemma4_assistant`, 6-bit). `runtime.load(..., mtp=True)` returns `load_gemma4_assistant_pair(...)` before the generic load, so `configure_split_full_attention` never runs and there is no verdict: segments stay off (fail closed, `/health` `model_not_checked`). The pair runtime has its own forward (`Gemma4TargetAdapter.forward_with_state`), its own cache factory (`make_cache`: `KVCache` for full layers, `Gemma4RollbackRotatingKVCache` for sliding layers), its own bank calls and an eager verify above short contexts.

What blocks segments (from code, unless marked):

1. Kernel: the global layers have head_dim 512 (`global_head_dim`). The segment kernel serves 128 and 256. The head-dim-split kernel at 512 with halves would hold 128 fp32 accumulators per thread, the register spill the dsplit design avoided; a quarter split (`NDH=4`, 128 dims per simdgroup, 64 accumulators like today) fits. The NDH template parameter exists on the prod line (attn4, `MTPLX_NAX_FLASH_DSPLIT4`), not on `feat/segmented-kv`. Threadgroup memory at NDH 4 and 32 query rows is `2 * 8 * 16 * 32 * 4 B` = 32 KiB, the Apple limit; GQA 8 allows 4 verify rows per sub-window, so a verify window of 7 rows (block 6) is 2 sub-windows. If 32 KiB does not compile, the fallback is 16 rows (2 per sub-window) or a single-buffered exchange.
2. Hook: the generic split hook cannot serve this class. Gemma's `Attention.__call__` takes `shared_kv` and `offset` and returns `(out, (keys, values), offset)`. Checked on a tiny CPU Gemma 4 model: after `configure_split_full_attention` the forward fails with `TypeError: split_call() got an unexpected keyword argument 'shared_kv'`. That is also a pre-existing bug outside segments: a Gemma 4 target loaded through `runtime.load(..., mtp=False)` takes the generic path, gets the hook and breaks on its first forward (the same hook code is on upstream main; only the tiny-model case was run). The pair path (`mtp=True`) never installs the hook, so production Gemma is not affected.
3. The generic verdict counts all 60 layers as "full attention" (it walks every `self_attn`); for Gemma only the 10 `full_attention` layers are candidates, the 50 sliding layers keep their ring buffer.
4. Shared KV for the drafter: every target forward returns the last full layer's whole `(keys, values)` (`shared_kv_states["full_attention"]`), and the assistant's full-attention layer attends over all of it on every draft step. With a stock `KVCache` that is a view; `SegmentedKVCache.update_and_fetch` would gather the history (about 512 MiB per round at 64K for that one layer). The handle passed to the drafter has to be the segment list, and the drafter's full layer has to attend with the segment kernel (q_len 1, GQA 8, head_dim 512), including `_slice_shared_kv_states` after rejection and the masks in `make_drafter_masks` (valid length only, which a segment reference truncated at `kv_valid_len` expresses).
5. Pair runtime touch points: `_gemma4_cache_arrays` reads `.keys/.values` (would gather on every prefill chunk), the bank's `extra_state["gemma4_shared_kv_states"]` holds arrays, `promote_gemma4_cache_for_compiled_verify` must decline segmented layers (compiled verify is short-context only), and seal at turn end must happen where the Qwen path does it (bank snapshot).
6. Prefill over segments: the `return_lse` MLX kernel (pkg-lse) is not expected to take head_dim 512, so prefill would use the exact reference route per segment (`sdpa_lse_reference`). The stock Gemma prefill also materializes the score matrix for head_dim 512 (no fused kernel above 256), and its chunk width is sized for that; the per-segment matrix is never larger. To measure.

K and V: `attention_k_eq_v` shares the projection, but values are `v_norm(k)` without rope and keys are rotated, so both are stored (as the stock cache does).

Memory scale (computed): the global KV is 80 KiB per token, 5 GiB at 64K and 10 GiB at 128K, slightly more than Qwen3.8-27B (64 KiB). The copy-on-write problem segments solve should show at the same order. Weights are about 17.7 GB, so in 64 GB the largest practical context is estimated at 128K; to measure in step 2.

### Mistral (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`, downloaded)

Chosen: v0.3, the current 7B Instruct in mlx-community 4-bit (4.08 GB, `~/.mtplx/models/mlx-community--Mistral-7B-Instruct-v0.3-4bit`, placed with `hf download --local-dir` like Qwen3-8B). Config: 32 layers all full attention, head_dim 128, 32/8 heads (GQA 4), `sliding_window: null` (v0.1 had 4096; mlx-lm's llama model makes rotating caches only from `layer_types`, so v0.3 gets 32 plain `KVCache`), rope theta 1e6, 32K context, vocabulary 32,768, chat template in `tokenizer_config.json`. Ministral-8B-2410 has the same attention shape (36 layers, 128K vocabulary), not taken.

What blocks segments: only the hook (checked: `attention_not_hooked`). `llama.Attention` has no `q_norm`/`k_norm`; `_attention_has_plain_q_proj` requires them and the hook body applies them. Needed: a third attention form (q/k/v projections, rope, no norms) in the detection and in the body, on a `SegmentedKVCache` only. The hook installs on the `llama.Attention` class, which mlx-lm shares between Llama, Mistral and every model type remapped to llama; with the switch off the body returns `original_call` for this form, so flag-off identity is by construction and gets a test. Kernel: head_dim 128, GQA 4 exists.

No MTP head: AR path (`runtime.load(..., mtp=False)`, `generate_ar`), decode windows of 1 row. `MTPLX_CONTEXT_COPY_AR` (context copy on the AR path) is not on `feat/segmented-kv`; it is on the prod line (the Qwen3-8B AR runs used `prod-segkv`). Measuring Mistral with context copy needs a tree that has both.

### Ternary Bonsai 2 27B and Apodex-1.1-mini

Both load through the generic path with Qwen3-Next gated attention and head_dim 256; checked "supported" with the turbo env, no code change. Bonsai is the Qwen3.8-27B shape (24/4) in fp16 with Hadamard-rotated ternary projections (`HadamardQuantizedLinear`, the output dimension is still `weight.shape[0]`, so the gate check holds); the kernel takes fp16. Apodex is the Qwen3.6-35B-A3B shape (16/2, GQA 8, 10 full layers), 20 KiB per token, so the gain per token is small.

Caveat (checked): with the `sustained` profile, which `mtplx_runtime.json` recommends for Apodex and Qwen3.6-35B-A3B, the verdict is off (`gqa_packed_sdpa_off`, `nax_flash_route_off`): sustained does not set those two switches. Segments therefore only switch on for these models under turbo or with the two switches set by hand.

## 3. Plan

- Step 2, Gemma 4: segment kernel at head_dim 512 (NDH 4), a Gemma-specific hook for full layers on a segmented cache (sliding layers unchanged), a segment handle as the drafter's full-attention shared KV, the pair runtime's touch points (cache factory, cache arrays, bank extra state, compiled-verify promotion), a verdict for the pair path over the full layers only; tests (kernel vs fp32 reference, no gather, flag-off identity), GPU off/on at 16K and the largest context.
- Step 3, Mistral: plain no-norm attention form in the hook and the support check, tests, AR runs; context-copy runs need the AR context copy on the measured tree (decision open).
- Side finding to fix separately: the generic hook breaks Gemma 4 on `mtp=False`; `configure_split_full_attention` should skip attention classes whose `__call__` has a different signature.

## 4. Segments under the `sustained` profile (VONDSTEN 122)

### Why the two switches are turbo-only

`turbo` is `SUSTAINED_PREFILL_ENV` plus a bundle: NAX verify matmuls, compiled verify, `MTPLX_GQA_PACKED_SDPA=1` (+threshold 8192, since 2.0.0) and `MTPLX_NAX_FLASH_ROUTE=1` (3096f9ea, 2 Sep). The bundle is assigned per model by measurement under a flat-or-better rule (`docs/profiles.md`, `docs/releases/v2.0.1.md`): the 35B-A3B MoE models stay on sustained because "their expert layers bypass the current kernel patch" (the NAX verify matmuls), Gemma 4 because its assistant pair does not use the native verify path, the 4B because turbo measured slightly slower. No reason is given against the two attention switches themselves; they were never evaluated separately on the sustained models. The flash route has a hardware gate since #459: on M1 to M4 (or macOS before 26.2) it is unavailable, so there `nax_unavailable` switches segments off under any profile.

### What segments use the switches for (from code)

- `MTPLX_GQA_PACKED_SDPA`: the hook only calls `decode_segments_attention` when `_mtplx_gqa_packed_sdpa_enabled` is set, and reads the packed threshold (8192) for a cache that is still one segment. The scalar packed kernel itself is never used by segments.
- `MTPLX_NAX_FLASH_ROUTE`: `decode_segments_attention` returns None without it; the segment kernel is the NAX head-dim-split kernel.
- Both: the kernel self-check probes `nax_flash_dsplit_sdpa` only when both are on (`kernel_selfcheck.py`), and the self-check runs only when one of a list of switches is on. Under sustained with only the segment switch, the segment kernel would serve without a boot probe.
- The verdict requires both (`gqa_packed_sdpa_off`, `nax_flash_route_off`).

Two further holes, profile independent: the verdict is made before the self-check (`runtime.py`: `configure_split_full_attention` at load, `maybe_run_model_selfcheck` after it), and a self-check that disables `nax_flash_dsplit_sdpa` makes `decode_segments_attention` return None, so every multi-segment call then gathers the history (the case the verdict exists to prevent). Also, the hook's early exit (`route_off`) sends a segmented cache to the stock forward (a gather per call) under a profile that enables no attention route at all; sustained enables one through `MTPLX_VLLM_METAL_PAGED_ATTN`, so it is not hit there.

### Measured cost (GPU, gate.sh)

Qwen3.6-35B-A3B Optimized-Speed (the sustained default), `sustained` profile, pkg-lse MLX, 50K prefix plus 4 follow-up turns (1-4K new tokens, 200 decoded, greedy, own mode, real `SessionBank`), feature tree c3119c7f (detached snapshot). Variants: `off` = sustained; `on` = sustained + `MTPLX_SEGMENTED_KV=1` + the two switches (what the fix below would give); `offsw` = sustained + the two switches, no segments. Order off on offsw offsw on off off on offsw, medians of 3, thermal 0 throughout. Data `other-models/out/sus-50000-*.json`, `python3 cmp3.py sus 50000`.

| turn | ms per verify off / on / offsw | decode tok/s off / on / offsw | peak over turn start GiB off / on / offsw | prefill s (all three) |
|---|---|---|---|---|
| 0 (cold) | 35.4 / 26.1 / 27.3 | 54.0 / 67.5 / 66.6 | 4.29 / 4.29 / 4.29 | 25.7 |
| 1 | 40.0 / 30.6 / 31.9 | 66.3 / 81.7 / 79.7 | 2.68 / 0.73 / 2.68 | 1.0 |
| 2 | 33.9 / 25.1 / 24.6 | 47.4 / 51.6 / 58.2 | 2.75 / 0.45 / 2.75 | 1.7 |
| 3 | 41.2 / 30.5 / 33.1 | 49.7 / 64.1 / 60.3 | 2.79 / 0.60 / 2.79 | 2.7 |
| 4 | 36.7 / 26.6 / 27.0 | 38.8 / 45.6 / 45.7 | 3.34 / 0.76 / 3.34 | 3.6 |

- Route counters `on`: no `gather_q*`; `fused_q1..4`, `chunked_q9/13/25`, `sdpa_lse_kernel` 1140, `merge` 210 (sums over 3 runs). Active memory after the turn 1.9 to 2.0 GiB lower with segments.
- The ms per verify gain (24 to 28 % with segments, 20 to 27 % for `offsw`) comes from the attention kernel, not from the segments. Segments add the memory gain (peak above turn start 2.7 to 3.3 GiB down to 0.45 to 0.76) at the same speed as `offsw`. Decode tok/s differs more between variants than ms per verify because the outputs diverge and acceptance differs per trajectory.
- Every variant is deterministic run to run. `on` and `offsw` both diverge from `off` (first divergence at 12 to 166 tokens): a different attention kernel, expected at bf16 ties; margins at the first divergence are queued (`out/sus-margin-*.json`).

### Proposed fix (smallest safe, not implemented)

Segments stop depending on the two switches and honour them only as kill switches:

1. Verdict: drop `gqa_packed_sdpa_off`; `nax_flash_route_off` only when `MTPLX_NAX_FLASH_ROUTE=0` is set explicitly; `nax_unavailable` stays.
2. Hook: a segmented cache always takes `decode_segments_attention` (no `_mtplx_gqa_packed_sdpa_enabled` check), and the `route_off` early exit does not apply to a segmented cache. The one-segment threshold keeps 8192 as its default.
3. `decode_segments_attention`: the flash-route check becomes "not switched off".
4. Self-check: runs when `MTPLX_SEGMENTED_KV` is requested and probes `nax_flash_dsplit_sdpa` (at the model's head_dim, 128 or 256); `segmented_kv_enabled()` is False when that lane is disabled, so a failed probe means stock caches, never a gather per call. This also closes the hole under turbo.

With the switch off nothing changes (all four points sit behind `segmented_kv_requested()` or a segmented cache). With the switch on under sustained, the full-attention layers run on the same NAX kernel turbo already ships on M5-class GPUs; the dense attention of other layers and the MTP head keep sustained's routes. Cost per the table: none measured; faster verify and lower peak.

Separate question, outside segments: the two switches alone gave 20 to 27 % fewer ms per verify on Qwen3.6-35B-A3B under sustained (one model, one context, M5 Max). That is a case for evaluating them for the sustained models upstream, under the flat-or-better rule; not part of the segmented KV PR.
