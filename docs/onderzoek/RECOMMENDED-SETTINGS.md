# Recommended MTPLX settings per Qwen model

State: 8 October 2026. Hardware for every number: Apple M5 Pro, 64 GB, unless a row says otherwise. Production code: branch `prod` (v2.12.2 plus our commits, tag `prod-2026-10-08`, see [CARRIED.md](CARRIED.md)). Details and raw numbers: [VONDSTEN.md](VONDSTEN.md), [VONDSTEN-QWEN38.md](VONDSTEN-QWEN38.md) and the reports linked per row.

Audience: Jeroen (to learn what each switch does) and the MTPLX maker (as a basis for adding proven switches to the `turbo` profile).

## 1. How to read this document

**Profile.** A profile is a named bundle of environment settings that MTPLX applies at start. `turbo` is the fastest decode profile and the default for the 27B and 9B flagship packs; `sustained` is the default for every other model. The settings in this document are applied on top of the profile as plain environment variables in the launch command (`config/config.json`, entry `backends.mtplx.launch.command`). They are opt-in and off by default in MTPLX. Switches that `turbo` or the model family already set are not listed as ours (see "What the profile already does").

**Evidence column.**

| Label | Meaning |
|---|---|
| measured on this model (date, finding #) | Run on this exact model with the switch on and off. |
| measured on Qwen3.8 only, expected to carry over because ... | Number is from Qwen3.8-27B. The reason for the expectation is given; the number itself is not a measurement on this model. |
| measured on another Qwen model only | Number from a different model of the family. Read it as a direction, not as a result. |
| assumed | Reasoned from code or model shape, never run. |
| not applicable (reason) | The switch cannot act on this model. |

**One launch command for all models.** The Bink platform starts MTPLX with one command for every model (`{model_path}` is substituted). So a switch that is only meant for one model is still in the command for the others. Switches that refuse themselves on the wrong model (the batch-invariant lane on a dense model) are harmless; switches that merely do nothing are also harmless; switches that were measured to be mixed on another model are flagged below.

**Plain-language glossary.**

| Term | Meaning |
|---|---|
| Draft / MTP head | A small extra network that guesses the next few tokens. The main model then checks all guesses in one pass (the verify pass). Right guesses are free tokens. |
| Depth | How many tokens ahead the draft head guesses. Depth 3 means a verify pass of 4 rows. |
| Verify pass | The one big forward pass that checks the guesses. This is where most decode time goes. |
| Context-copy | Instead of the draft head, copy a block of 8 to 32 tokens from the prompt when the output starts repeating it (editing a file, quoting a document). On by default. |
| Session bank | RAM cache of earlier conversations (their KV cache), so a follow-up turn does not recompute the whole history. |
| SSD tier | The same cache spilled to disk, so it survives eviction and restarts. |
| Prefill | Reading the prompt. Compute-bound; time grows with prompt length. |
| Eager / compiled verify | Whether the verify graph is rebuilt in Python every round (eager) or traced once and reused (compiled). |

## 2. What the profile already does (not ours)

From [profiles.py](../../mtplx/profiles.py) on `prod` and `docs/profiles.md`:

| Already set by `turbo` | Notes |
|---|---|
| `MTPLX_NAX_VERIFY=1`, `MTPLX_NAX_M4_IMPL=vk_k` | Fast verify matmul kernels for 4-bit and 8-bit packs. "6-bit models silently run the stock path" (profile caveat). |
| `MTPLX_COMPILED_VERIFY=1`, `MAX_CONTEXT=32768` | Compiled verify for 4-bit and 8-bit packs. 6-bit packs stay eager (gate `quant_bits_gate`). |
| `MTPLX_GQA_PACKED_SDPA=1` (threshold 8192), `MTPLX_NAX_FLASH_ROUTE=1` | Fast verify attention for 2 to 4 rows. |
| `MTPLX_PREFILL_CHUNK_SIZE_DENSE/REPAGE=2048` | We override this with 4096 (section per model). |
| `MTPLX_WARMUP_LADDER=512,2560`, `MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT=1024`, committed MTP history | Defaults. |
| Context-copy (`MTPLX_CONTEXT_COPY`, min n-gram 6) | On by default on all MTP lanes. Measured on Qwen3.8: agent Edit 2.1x, code edit +23 to +38%, extraction and prose unchanged ([qwen38-speed-options](2026-10-02-qwen38-speed-options.md) section 1). Do not lower the minimum n-gram to 3 (worse). |
| Dense KV for decode up to about 157K tokens, depth and draft temperature 0.6 per family | Family defaults, not ours. |

## 3. The production launch command (current Qwen3.8 recipe)

```
env PYTHONPATH=~/Dev/mlx-sdpa/pkg-combo:~/Dev/MTPLX-prod \
  MLX_SDPA_D256_MINQ=64 \
  MTPLX_MULTIROW_QMM=1 MTPLX_NAX_FLASH_WIDE=1 MTPLX_NAX_FLASH_DSPLIT4=1 \
  MTPLX_BATCH_INVARIANT_PREFILL=1 MTPLX_A3B_MOE_PREFILL_COMBINE=1 \
  MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096 MTPLX_PREFILL_CHUNK_SIZE_REPAGE=4096 \
  MTPLX_PREFILL_MIN_CHUNK_ROWS=1024 \
  MTPLX_SESSION_HEAD_ANCHOR=1 MTPLX_SSD_INCREMENTAL_ENCODE=1 \
  MTPLX_PERSISTENCE_MAX_PENDING_BYTES=4294967296 \
  MTPLX_THINKING_BUDGET=6144 \
  MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_VOCAB=~/.mtplx/frspec/bink-64k.npy MTPLX_FRSPEC_LEGACY=1 \
  MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8 \
  mtplx serve --model {model_path} --port 8000 --no-auth --mtp --yes
```

MLX is the fork build (`jvmenen/mlx` branch `prod`, v0.32.2 plus the MINQ and qmm tile patches, tag `prod-2026-10-04`). Segmented KV is **not** in this command (pending, section 8).

## 4. Qwen3.8-27B (Youssofal Optimized-Speed, production)

Model shape (from `config.json`): dense, 64 layers of which 16 full attention, 24 query / 4 KV heads, head_dim 256, body 4-bit g32, lm_head and embeddings Q8, MTP head 4-bit g64. KV cost 64 KiB per token. Default depth 3, compiled verify on (4-bit).

| Setting | What it does (plain language) | Effect (speed / memory) | Evidence |
|---|---|---|---|
| `MTPLX_FRSPEC_DRAFT=1` + `MTPLX_FRSPEC_VOCAB=bink-64k.npy` + `MTPLX_FRSPEC_LEGACY=1` | The draft head normally scores all 248K words to guess the next one. FR-Spec lets it score only the 64K most common words. The main model still checks every token, so the output stays correct; a word outside the list simply cannot be guessed. The Bink list is ranked on our own (mostly Dutch) traffic. | Built-in code list on the proper binding: reasoning workload +5.4% (31.9 to 33.7 tok/s), code +8.6%, English +7.5%, Dutch -3.5% (thinking on) and -12.4% (thinking off). Legacy lane with built-in list: +7 to +8% (36.4 vs 34.1). Bink list: Q2 reports +12% on the bench, Dutch +10 to +11% (see conflict below). Speed only; memory unchanged. | measured on this model (30 Sep, VONDSTEN-QWEN38 Q2). **Conflict:** the Bink-list rows in Q2 contradict [frspec-qwen36](2026-09-30-frspec-qwen36.md) and finding 95, which say the Bink list was never run on 3.8. Treat the Bink-list gain as unconfirmed until re-run on `prod`. |
| `MTPLX_MULTIROW_QMM=1` (#595) | When the verify pass has 5 to 32 rows (long context-copy blocks, or depth 4+), use matmul kernels that read the weights once instead of once per 5-row tile. Does nothing at the normal 4-row verify. | Kernel -11 to -21% (5 to 16 rows). Verify forward at 2K: -6.6% (5 rows), -12% (8 and 16 rows), -34% (24 rows). End to end: agent edit +25% greedy / +22% at temperature 1.0, code edit +8% / +4%, prose unchanged. | measured on this model (4 Oct, finding 108) |
| `MTPLX_NAX_FLASH_WIDE=1` (#593) | Fast attention for verify windows of 9 to 32 rows (context-copy blocks) instead of the slow general path. | Attention op at 80K, 9 rows: 9.83 to 2.70 ms per layer. End to end on context-copy edits at 50K and 81K: +6 to +13% alone, +13 to +35% together with multi-row qmm. | measured on this model (4 Oct, finding 109) |
| `MTPLX_NAX_FLASH_DSPLIT4=1` (#594) | Splits attention over four parts of the head dimension for the normal 2 to 5-row verify, so the GPU is used better when the context is long. | Attention op +27 to +32%; verify forward -2.3% (20K) to -6.1% (80K). End to end greedy: +4.5% / +4.6% / +1.5% at 50K, +5.5% / +4.7% / +3.6% at 80K (agent edit / code edit / Dutch prose). Short context neutral (within 0.2%). Temperature-1 cells swing both ways (-33% to +20%) because sampled text diverges. | measured on this model (4 Oct, finding 110) |
| `MLX_SDPA_D256_MINQ=64` (MLX fork patch) | MLX uses its fast attention kernel for head_dim 256 only from 1,024 new rows. The patch lowers that to 64, so a follow-up turn that appends 300 to 700 tokens after a long history uses the fast kernel. | 350 and 730-token appends at 50K to 80K: -0.2 to -0.76 s (14 to 19%). 100 and 5,000-token appends: no effect. Logits not bit-identical, greedy 24/24 equal. | measured on this model (4 Oct, finding 101). Custom MLX build required. |
| `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` (#597) | When a long prompt is cut into 4,096-row chunks, merge a small last chunk (under 1,024 rows) into the one before it, because small chunks fall on a slow attention route. | -0.3 to -1.4 s per append at 80K prefix; noise at 20K. Not bit-identical to the old plan (96% top-1, same size as any chunk-shape change). | measured on this model (4 Oct, finding 100) |
| `MTPLX_PREFILL_CHUNK_SIZE_DENSE/REPAGE=4096` | Prompt is read in blocks of 4,096 tokens instead of the profile's 2,048. | 22K cold prefill 55.2 s (4096) vs 57.0 s (2048), a single run, below the 5% noise bar. 6144: no difference at 20K to 80K. 8192: prefill 41.5 / 136.5 s (slower) and the Mac swapped 9.4 GB. | measured on this model (30 Sep, VONDSTEN-QWEN38 Q3): 4096 is not shown better than 2048 here; 8192 is harmful on 64 GB. The 4096 value is justified by the A3B measurement (below), not by this model. |
| `MTPLX_SESSION_HEAD_ANCHOR=1` (#559) | Every new chat starts with the same ~13K tokens (tools and system text). The anchor stores a checkpoint right after that shared head, so a new session reuses all of it instead of 10K to 12K. | On A3B: first-turn TTFT of a new session 3.19 to 1.67 s (-48%), from SSD after restart 2.68 to 2.14 s (-20%). Output identical, peak memory equal. Estimated 5 to 6 s saved per new session in production. | measured on Qwen3.6 only (28 Sep, finding 63), expected to carry over because Qwen3.8 shares the same hybrid GDN checkpoint mechanism and the same 13K shared head. On `prod` the anchor was re-implemented on `checkpoint_anchors`; 43 tests adapted. No separate Qwen3.8 on/off run. |
| `MTPLX_SSD_INCREMENTAL_ENCODE=1` (needs fingerprint fix #596) | When saving a conversation to SSD, write only the new part instead of the whole conversation every time. | On A3B: encode work -58 to -61%, footprint peak at the end 51.4 to 48.2 GB (-3.2 GB), output bit-identical, no measurable slowdown. | measured on Qwen3.6 only (28 Sep, findings 65, 72), expected to carry over because the SSD store is model-independent. Upstream turned it off again over a collision risk (finding 102); #596 fixes the mixer. |
| `MTPLX_PERSISTENCE_MAX_PENDING_BYTES=4294967296` | Caps the queue of conversations waiting to be written to SSD at 4 GB. Safety net against the queue growing and pushing the Mac into swap. | On A3B: queue max 4.1 GB instead of 8.4 GB; 5 sessions dropped from the queue (they reach SSD on their next turn). No speed or output difference. | measured on Qwen3.6 only (28 Sep, finding 67). Note: upstream's default is RAM/32, about 2 GB on this Mac, so our 4 GB is a looser cap than the default. |
| `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` (#579) | Submits the verify graph to the GPU in pieces of 8 layers, so the GPU starts while Python is still building the rest. | **No effect on this model**: with the fix, off 48.9 / 49.1 vs on 49.0 / 49.0 tok/s, 104/104 verify calls compiled, tokens identical. Compiled verify already avoids the rebuild. Before the fix (commit 063d77b5) it made this model slower (49.0 to 45.8 tok/s). | measured on this model (1 Oct, finding 96). Harmless here; it is in the command for the MoE models. With segmented KV on, verify runs eager and the switch matters again: measured 8 Oct with segments from token 0, ms/verify with vs without async 74-78 vs 84-92 at 16K (6 to 19% faster) and 87-99 vs 92-103 at 50K (4 to 5%), prefill and peak equal ([segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 10). Keep it on. |
| `MTPLX_THINKING_BUDGET=6144` | Caps the reasoning block at 6,144 tokens, then forces it closed so the answer starts. Quality guard, not a speed switch. | At 53K and 62K context a long task hit the cap twice and finished normally; without a cap one earlier run never left its thinking block. | measured on this model (30 Sep, Q10; 20 Sep model test). One task, so weak statistically. |
| `MTPLX_BATCH_INVARIANT_PREFILL=1` | Makes prompt scoring independent of batch layout (needed for MoE classification). | Refused on this dense model (`dense_model`), so a no-op. | not applicable (dense model, lane refuses itself; finding 59) |
| `MTPLX_A3B_MOE_PREFILL_COMBINE=1` | One-kernel combine step of MoE layers. | No MoE layers. | not applicable (dense model) |

**Recommendation (Qwen3.8-27B).** Keep the production command as it is, with three caveats: (1) the Bink FR-Spec list on the legacy lane is confirmed on `prod` (8 Oct: +6.5% Dutch and English prose against FR-Spec off, acceptance equal, output byte-identical; the built-in code list loses 8.6% on Dutch, so do not switch to it; [frspec-prod-remeasure](2026-10-08-frspec-prod-remeasure.md)); (2) the two MoE switches can stay, they do nothing here; keep `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8`, it does nothing under compiled verify but helps 4 to 19% once segmented KV is on; (3) do not add KV quantization, BF16 head, depth 4 or a custom depth (section 9). Optional later: segmented KV for memory (section 8, pending).

## 5. Qwen3.6-35B-A3B (MoE)

Model shape: 40 layers of which 10 full attention, 16 query / 2 KV heads, head_dim 256, 256 experts. KV cost 20 KiB per token (against 64 KiB on the 27B). Default depth 2. Two body variants matter: **Optimized-Speed** (4-bit g64 body) and **Optimized-Balance / our `balance-bf16mtp-yb`** (6-bit g64 body). The 6-bit pack is gated off compiled verify and off the NAX verify kernels, which is why some switches act differently on it. Most Bink-specific measurements were done on Balance.

| Setting | What it does (plain language) | Effect (speed / memory) | Evidence |
|---|---|---|---|
| Pack choice: `Qwen3.6-35B-A3B-MTPLX-Balance-bf16mtp-yb` (own pack with BF16 draft head, split experts) | The shipped packs have an int4 draft head. A BF16 head guesses better on this MoE model. Needs the loader fix for fused experts (#574), otherwise the routed experts of the head stay random and acceptance collapses (0.69 / 0.37 instead of 0.947 / 0.848). | 92.0 vs 69.8 tok/s in-process (+17% vs the current Balance pack). On Qwen3.8 the same idea is 2 to 8% slower, so this is model-specific. | measured on this model (30 Sep, finding 90). Gap: #574 is not in the [CARRIED.md](CARRIED.md) table; confirm it is in `prod`. |
| `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` (#579) | Sends the verify graph to the GPU in pieces of 8 layers. This model rebuilds the 40-layer graph in Python every round (6-bit is not compiled), so the GPU waits about 2 ms per round. Pieces let both overlap. | Balance (6-bit): 93.1 to 102.4 tok/s (+10%) after the fix; server +8.8% at nominal temperature; the gain shrinks under sustained heat (never slower). Speed-yb (4-bit), eager vs eager + async: 110.6 to 120.0 tok/s (+8.5%). Memory unchanged. | measured on this model (1 Oct, findings 96, 98) |
| `MTPLX_FRSPEC_DRAFT=1` + Bink list + `MTPLX_FRSPEC_LEGACY=1` | 64K-word draft head (see Qwen3.8 row). | Built-in code list: overall +0.3% (code +5%, English +7%, Dutch -7%). Bink list: overall 87.4 to 92.1 tok/s (+5.4%), code +4.8%, English +7.6%, Dutch +3.7% (thinking on) and +5.0% (off). Agent task green. | measured on this model (30 Sep, finding 95), Balance-bf16mtp-yb pack |
| `MTPLX_BATCH_INVARIANT_PREFILL=1` (#549) + in-forward GDN boundaries (#550) | Prompt reading gives the same numbers whatever the batch layout, which MoE routing otherwise breaks. Needed for reliable classification and for the gain in agent turns. | Scoring route 1.28x faster, final test 3: scoring -36% with equal accuracy, agent conversation -62%, 3.6 GiB less memory after classification. | measured on this model (26 to 28 Sep, findings 29, 70) |
| `MTPLX_A3B_MOE_PREFILL_COMBINE=1` (#558) | Merges the expert outputs in one GPU kernel during prefill. | Cold prefill -6.9%, decode +1.2% alone; the three prefill switches together: cold TTFT -10.4%, agent TTFT -10.9%, peak after cold -3.1%, bit-identical. Individually barely above the 4 to 14% baseline spread. | measured on this model (27 to 28 Sep, finding 64) |
| `MTPLX_PREFILL_CHUNK_SIZE_DENSE/REPAGE=4096` | Read prompts in blocks of 4,096. | Cold -12.2%, agent -10.6%, peak memory +1.6% (alone); in the combined set bit-identical. | measured on this model (28 Sep, finding 64) |
| `MTPLX_SESSION_HEAD_ANCHOR=1` (#559) | Reuse the shared ~13K-token head across new sessions. | New-session TTFT -48% (3.19 to 1.67 s), from SSD -20%; memory equal. | measured on this model (28 Sep, finding 63) |
| `MTPLX_SSD_INCREMENTAL_ENCODE=1` | Write only the new part to SSD. | Encode work -58%, footprint peak -3.2 GB, bit-identical. | measured on this model (28 Sep, findings 65, 72) |
| `MTPLX_PERSISTENCE_MAX_PENDING_BYTES=4 GiB` | Caps the SSD write queue. | Queue max 4.1 GB vs 8.4 GB; no speed or output change. | measured on this model (28 Sep, finding 67) |
| Whole set above together (findings 72, 70) | | Bit-identical; footprint peak 51.7 to 48.0 GB (-7.2%); tail after the last token -91%; costs: first cold 2K prompt TTFT +3.4%, short agent turns +2% (+13 ms). | measured on this model (28 Sep) |
| `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` | Merge a small last prompt chunk into the previous one. | **Mixed on the Balance pack**: 5,000 new tokens at 80K -3.0 s, 9,000 at 20K -1.0 to -3.3 s, but 9,000 at 80K slower in 2 of 3 pairs (+2.3 and +2.6 s); cause not investigated. | measured on this model (4 Oct, finding 100), mixed. It is in the shared launch command, so it also applies here. Consider removing it for this model. |
| `MTPLX_MULTIROW_QMM=1`, `MTPLX_NAX_FLASH_WIDE=1`, `MTPLX_NAX_FLASH_DSPLIT4=1`, `MLX_SDPA_D256_MINQ=64` | Verify and attention kernels measured on the 27B (see Qwen3.8 table). | No Qwen3.6 number exists. `MULTIROW_QMM` is 4-bit only (no effect on the 6-bit Balance pack); on the 4-bit Speed pack it targets the dense matmuls, not the expert matmuls. The two flash kernels have a head-count contract written for GQA 6 (27B); this model has GQA 8 (2 KV heads), where a 4-row window already fills 32 rows per KV head. | assumed (not measured on this model). The v2.12.2 four-variant run (finding 99) shows the carried work as a whole is +9 to +10% decode on this model, but it does not split out these four switches. |
| `MTPLX_THINKING_BUDGET=6144` | Reasoning cap. | Not measured separately here. Issue #583 (empty answer after tool result) shows 1 empty turn in 18 on this model, and the thinking budget does not help; fix #588 is carried. | assumed for the cap; #583 measured (4 Oct) |
| Depth | Stay at depth 2. | D2 vs D3 at 20K / 40K / 60K: 81.3 / 76.1, 69.6 / 67.5, 63.6 / 64.7 tok/s. A third guess at about 0.4 acceptance does not pay on an MoE where drafting is relatively expensive. | measured on this model (30 Sep, finding 93) |
| Segmented KV state | See section 8. | 50K, four follow-up turns: peak above turn start 2.68 / 1.93 / 2.15 / 2.43 GiB (stock) vs 1.65 / -0.50 / 0.59 / 0.72 GiB (segmented); decode within noise; no gather. | measured on this model (8 Oct, [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 8). Pending, not on `prod`. |

**Recommendation (Qwen3.6-35B-A3B).** Use the `bf16mtp-yb` pack at depth 2 with the full production command. The launch command already carries what was measured here (lane, combine, 4096 chunks, head anchor, incremental SSD, 4 GB queue, async chunk 8, Bink FR-Spec). Two review points: drop `MTPLX_PREFILL_MIN_CHUNK_ROWS` for this model (mixed result), and keep `MTPLX_COMPILED_VERIFY_ALLOW_BITS=6` unset (section 9).

## 6. Qwen3.5-9B (Youssofal Optimized-Speed)

Model shape (from `config.json`): 32 layers, hybrid with GDN layers, 16 query / 4 KV heads, head_dim 256, **body 6-bit g64** (so the profile treats it like the 6-bit A3B Balance: stock verify kernels, eager verify). Default depth 2, 10.0 GiB MLX peak in the model test. This model has had little targeted measurement; most rows are carried over by reasoning.

| Setting | What it does | Effect (speed / memory) | Evidence |
|---|---|---|---|
| `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` | Overlaps the Python graph build with GPU work (see section 5). | Expected to help for the same reason as on A3B Balance: 6-bit is not compiled, so the verify graph is rebuilt in Python every round. | measured on Qwen3.6 6-bit only (findings 96, 97: +10%), expected to carry over because the 9B is also a 6-bit pack that runs eager verify. Not run on the 9B. |
| `MTPLX_BATCH_INVARIANT_PREFILL=1` | Batch-independent prompt scoring. | Refused on this dense model (`dense_model`). The 26 Sep model test ran an older build that did install the lane on the 9B: lane cost about 4% on scoring and about 15% on greedy TTFT, 5 of 12 greedy texts differed from main. | not applicable on `prod` (lane refuses dense). The older numbers are from before the MoE-only gate. |
| `MTPLX_A3B_MOE_PREFILL_COMBINE=1` | MoE combine kernel. | No MoE layers. | not applicable |
| `MTPLX_MULTIROW_QMM=1` | Multi-row 4-bit matmuls. | The code and the finding cover 4-bit matmuls only; the 9B body is 6-bit. | not applicable (6-bit body) |
| `MTPLX_NAX_FLASH_WIDE=1`, `MTPLX_NAX_FLASH_DSPLIT4=1`, `MLX_SDPA_D256_MINQ=64` | Attention kernels for head_dim 256 verify windows and short appends. | The 9B has head_dim 256 and GQA 4, inside the kernel contract (head_dim 256, bf16, up to 32 rows). The MLX route logic was checked on this model ("routing confirmed in mlx-lm (Qwen3.5-9B, bf16, causal)", finding 112) but no speed run exists. | measured on Qwen3.8 only, expected to carry over because the head_dim and kernel contract match; the attention share of a 32-layer model is smaller, so the gain is likely smaller. Not run. |
| `MTPLX_PREFILL_CHUNK_SIZE_*=4096`, `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` | Prompt chunking. | No 9B number. Min-chunk was mixed on the 6-bit A3B pack. | assumed |
| `MTPLX_SESSION_HEAD_ANCHOR=1`, `MTPLX_SSD_INCREMENTAL_ENCODE=1`, `MTPLX_PERSISTENCE_MAX_PENDING_BYTES` | Bank and SSD behavior. | Model-independent in design; the 9B model test (26 Sep) showed the full fix set at agent conversation -70% and MLX peak -1.5 GiB (11.56 to 10.02 GiB), but that run predates these three switches and mostly reflects the messages-ttft fix, now upstream. | assumed for these three; the final test 3 (28 Sep, finding 70) reports "9B same agent gain" for the whole set without a table. |
| FR-Spec (`MTPLX_FRSPEC_*`) | Pruned draft head. | The Bink list was ranked on the Qwen3.6 tokenizer; whether the 9B shares the vocabulary and has a compatible draft head layout was not checked. | assumed / unchecked. Do not rely on it until a boot with the switch shows it installed. |
| Segmented KV state | See section 8. | 50K, four follow-up turns, peak above turn start (stock to segmented): 4.03 to 2.48, 3.22 to -0.92, 3.30 to 0.88, 3.78 to 1.18 GiB. Supported (GQA 4, head_dim 256), no gather, first greedy divergence at a bf16 tie (margin 0.125). Only recorded in RESUME.md; no report section, speed numbers not written down here. | measured on this model (8 Oct, `m3/RESUME.md` round 5). Pending, not on `prod`. |

**Recommendation (Qwen3.5-9B).** Run it with the shared production command; the lane refuses itself and the 4-bit-only kernels stay inactive. The only switch with a likely speed gain is `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` (already in the command). Before relying on anything else for this model, run one A/B (async chunk on/off, and FR-Spec installed or not) on a quiet Mac.

## 7. Qwen3-8B (mlx-community 4-bit, no MTP head, head_dim 128)

Model shape: dense, 36 layers all full attention, 32 query / 8 KV heads (GQA 4), head_dim 128, 4-bit g64, 4.6 GB. No draft head, so MTPLX runs it autoregressively (`generate_ar`, one row per step, no speculation). This model is a test vehicle for the head_dim 128 and AR code paths, not a production model. Almost every Bink switch is built for head_dim 256 or for speculative verify.

| Setting | What it does | Effect (speed / memory) | Evidence |
|---|---|---|---|
| `MTPLX_CONTEXT_COPY_AR=1` (#599) | Without a draft head, copy a block from the prompt when the output repeats it and verify the block in one pass. Greedy output stays identical to the one-token loop. | Measured on Llama 3.1 8B 4-bit: editing a 2,458-token file 55.3 to 289.3 tok/s (5.2x), quoting a 4,075-token report 1.3x, free prose unchanged; 20 of 20 greedy streams identical; peak memory unchanged. | measured on another model only (Llama 3.1 8B, 4 Oct, commit f1d7d1e1, CHANGELOG line and finding 103). Not run on Qwen3-8B; expected to carry over because the mechanism does not depend on the model. Hybrid models not measured. The `~/Dev/laya-nl/pld-pr/` folder named in the task does not exist; the numbers come from the CHANGELOG. |
| Segmented KV state (`MTPLX_SEGMENTED_KV=1`) | See section 8. | head_dim 128 kernel written (halves split); kernel and attention tests pass on both trees. First GPU check on this model: fused one-row decode routes, no gather. The speed/memory matrix (22K context, plain, code and prose with context-copy) was still running when this was written. | assumed for effect; kernel tests measured. Pending, not on `prod`. |
| `MTPLX_MULTIROW_QMM=1` | Multi-row 4-bit matmuls. | Acts only when a verify has 5 to 32 rows; with `CONTEXT_COPY_AR` on, copy blocks of 8 to 32 rows are exactly that. The model is 4-bit, so the switch could apply. | assumed (not run; effect on context-copy blocks never measured outside the 27B) |
| `MTPLX_NAX_FLASH_WIDE=1`, `MTPLX_NAX_FLASH_DSPLIT4=1`, `MLX_SDPA_D256_MINQ=64` | head_dim 256 attention kernels and MLX route. | The kernel contract and the MLX route are for head_dim 256. | not applicable (head_dim 128; MLX route for head_dim 128 never measured) |
| FR-Spec (`MTPLX_FRSPEC_*`) | Pruned draft head. | No draft head. | not applicable |
| `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` | Overlaps graph building with GPU work in the verify pass. | AR decode has no speculative verify pass of this kind. | not applicable (no verify pass), not run |
| `MTPLX_SESSION_HEAD_ANCHOR=1` | Checkpoint after the shared head, built on GDN boundaries. | Qwen3 has no GDN layers. | not applicable |
| `MTPLX_BATCH_INVARIANT_PREFILL=1`, `MTPLX_A3B_MOE_PREFILL_COMBINE=1` | MoE lane and combine. | Dense model: lane refuses itself; no MoE. | not applicable |
| `MTPLX_PREFILL_CHUNK_SIZE_*=4096`, `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` | Prompt chunking. | Not run on this model; model context is 40,960 tokens, so the large-context benefit seen on the 27B does not arise. | assumed |
| `MTPLX_SSD_INCREMENTAL_ENCODE=1`, `MTPLX_PERSISTENCE_MAX_PENDING_BYTES` | Bank and SSD behavior. | Model-independent in design. | assumed |
| `MTPLX_THINKING_BUDGET=6144` | Reasoning cap. | Applies only if the chat template uses a reasoning block; not run. | assumed |
| `--mtp` in the launch command | Requests MTP generation. | The segment-kv tests load this model through `runtime.load(path, mtp=False)` and `generate_ar`; whether `mtplx serve --mtp` starts a no-head model cleanly was not checked. | gap, not checked |

**Recommendation (Qwen3-8B).** Treat as AR-only: shared command minus the speculative switches (they are inert), plus `MTPLX_CONTEXT_COPY_AR=1` if the workload is edit-shaped (code or text rewriting). Do not expect any head_dim 256 kernel switch to act. Measure before adopting anything.

## 8. Memory

Settings that trade memory (or time) against each other. Qwen3.8-27B is the only model with a full measurement set; other models are called out.

### 9.1 What uses memory (Qwen3.8-27B, 64 GB Mac)

Weights about 27.6 GiB. Session bank ceiling at rest 17.4 GiB (11 GiB used in the final benchmark, never at the ceiling). Reserve for generation spikes about 7 GiB. KV cost 64 KiB per token (16 full-attention layers): about 5.2 GB at 80K. The 27B production process peaked at 36.5 GB in the head-anchor runs; the 80K stock follow-up turn adds a transient +4.95 GiB because the bank snapshot shares the KV buffer with the live cache and the first write copies it (+3.1 GiB at 50K) (finding 111).

### 9.2 Measured memory settings

| Setting | What it does | Measured effect | Model | Evidence |
|---|---|---|---|---|
| `MTPLX_PERSISTENCE_MAX_PENDING_BYTES` (we use 4 GiB) | Caps the in-RAM queue of conversations waiting for the SSD write. | Queue max 4.1 vs 8.4 GB, no speed change, 5 tasks dropped (reach SSD at their next turn). Upstream default is RAM/32 (about 2 GiB here). | A3B Balance | measured (28 Sep, finding 67) |
| `MTPLX_SSD_INCREMENTAL_ENCODE=1` | Write only the new part of a conversation. | Footprint peak -3.2 GB (-7.2% with the whole set), encode work -58 to -61%. Needs #596. | A3B Balance | measured (28 Sep, findings 65, 72) |
| Prefill chunk 4096 | Larger prompt blocks. | Peak +1.6% on A3B; 8192 swaps (9.4 GB) on the 27B. Do not go above 4096 on 64 GB. | A3B, 27B | measured (findings 64, Q3) |
| KV q8 / q4 | Smaller live KV. | Saves 1.1 to 1.9 GiB live at 60K; bank unchanged; large speed loss. | 27B, A3B | measured, rejected (section 9) |
| Segmented KV state (`MTPLX_SEGMENTED_KV=1`, **pending, not on `prod`**; branches `feat/segmented-kv` and `prod-segkv`) | Stores a conversation as a chain of per-turn segments instead of one buffer, so a follow-up turn never copies the history. Does not change what the model computes; it removes the duplicate. Needs the LSE-capable MLX build (`pkg-lse`) for fast segmented prefill. | See 9.3. | 27B, A3B, 9B | measured (7 to 8 Oct, findings 111, 113 to 115) |
| Segmented KV SSD tier (`MTPLX_SEGMENTED_KV_SSD`, on with the flag) | Segmented conversations are written to and read from SSD without gathering. | See 9.4. | 27B | measured (8 Oct, finding 114, 115) |
| Bank ceiling / `MTPLX_SESSION_BANK_SPIKE_BURSTS` | Lets the bank ceiling recover after a big prefill. | No effect in the measured workload (bank 11 of 17.4 GiB). | A3B | measured, not adopted (finding 8) |
| 507 memory refusals | The server refuses a prefill that would not fit. Not a setting, but the symptom the above reduce. Dashboard now labels it "507 memory" (#600). | Three legitimate refusals on 6 Oct at 88K to 90K context. | 27B | observed (finding in Handled, 6 Oct) |

### 9.3 Segmented KV state: measured effect

All with `generate_mtpk` depth 3, real session bank, production env, one process per run, order off on on off off on, thermal 0, medians of 3.

| Model, context | Peak above turn start per follow-up turn, stock to segmented (GiB) | Decode | Prefill | Source |
|---|---|---|---|---|
| Qwen3.8-27B, 50K | 4.80 / 5.94 / 6.08 / 7.50 to 4.93 / 3.08 / 3.17 / 4.59 (turn 1 pays a one-off conversion) | tok/s -2 to -6%; ms per verify 109/108/122/104 to 105/106/120/103 | +3 to +12% (turn 4 13.5 to 15.2 s) | [segmented-kv-m3](2026-10-07-segmented-kv-m3.md), finding 113 |
| Qwen3.8-27B, 80K | 6.75 / 7.88 / 8.03 / 9.44 to 6.88 / 3.19 / 3.28 / 4.71 | tok/s within noise; ms per verify 115/113/111/121 to 106/105/104/115 | equal | same |
| Qwen3.6-35B-A3B, 50K | 2.68 / 1.93 / 2.15 / 2.43 to 1.65 / -0.50 / 0.59 / 0.72 (absolute peak 3.0 to 3.7 GiB lower) | within greedy noise; ms per verify 27/25/28/25 to 27/22/26/22 | equal | [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 8 |
| Qwen3.5-9B, 50K | 4.03 / 3.22 / 3.30 / 3.78 to 2.48 / -0.92 / 0.88 / 1.18 | not written down | not written down | `m3/RESUME.md` only |
| Qwen3-8B, 22K (AR, head_dim 128) | not yet measured | not yet measured | not yet measured | running |

Server run, Qwen3.8-27B, A B B A (port 8010): stock had one HTTP 507 and 8 sheds, segmented none; peak 41.7 vs 37.8 GB; restore copy 3.2 to 5.75 GB vs 0; a fork of an 84K conversation resumed 82,958 tokens instead of a cold re-prefill (one repeat per variant). Cost: compiled verify cannot trace a segmented cache, so above the fence (`MTPLX_SEGMENTED_KV_MIN_TOKENS`, default = compiled-verify ceiling 32768) verify runs eager, about 3% slower per verify (116.4 vs 112.9 ms at 50K). Greedy text differs from stock only at bf16 ties (margin 0.0 to 0.125).

### 9.4 Segmented KV and the SSD tier (Qwen3.8-27B)

| Case | Stock | Segmented | Source |
|---|---|---|---|
| Fork of an 80K conversation on disk | +5.4 GB | +146 MB | direct cold-tier bench, [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 4 |
| Per-turn spill write at 80K | 2.2 to 2.6 s | 0.25 to 0.58 s (same bytes on disk, 5.97 GB) | [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 7.2 |
| TTFT after eviction (bank cap 7G), 50K / 80K | cold 143 s / 248 s | 5.2 s / 8.3 s | [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 7.1 |
| TTFT after a server restart, 50K / 80K | 25.4 s / 21.3 s (segmented, before the restore-layout fix; stock restores at about the same speed) | 7.3 s / 7.2 s | [segmented-kv-ssd](2026-10-08-segmented-kv-ssd.md) section 7.1 |
| Process peak with eviction | 41.4 GB | 35.5 to 37.3 GB | section 5 |

The restore speed-up came from a separate fix: `_restore_cold` ignored the request's cache factory and built a paged KV cache (about 5x slower prefill, slower decode). That bug also exists in stock without the flag; the fix is only behind the flag so far (finding 115, open as a separate PR).

## 9. Rejected / do not use (all models)

One-line reasons; finding numbers refer to [VONDSTEN.md](VONDSTEN.md) (Q-numbers to [VONDSTEN-QWEN38.md](VONDSTEN-QWEN38.md)).

| Switch or idea | Why not |
|---|---|
| KV quantization q8 / q4 (`--paged-kv-quantization`) | 27B: decode -25% (20K) to -59% (60K), q4 up to -72%; A3B: -4% to -16%, TTFT of short follow-ups +46%. The bank does not shrink (it stores bf16), live KV saves only 1.1 to 1.9 GiB (finding 37, Q4). |
| BF16 MTP head on Qwen3.8 | 2 to 8% slower at equal acceptance (Q13). (It does help on Qwen3.6, section 5.) |
| Depth 4 or more on Qwen3.8 | Maker measured D6 -27%; with multi-row lanes D4 is still -3 to -6% vs D3 on prose; D4 unstable earlier (Q0, finding 108). |
| Depth 3 on Qwen3.6 | Loses or ties at 20K to 60K (finding 93). |
| Tree verification | Offline +4 to +16% tokens per verify, but 6 to 12 rows cost +12% to +104%; nets -3% to -64% (Q16). |
| DFlash 2 drafter | On par on extraction and code, worse on prose, behind context-copy on edits (Q12). |
| Fine-tuning the MTP head on off-policy text | +0.6% tokens per verify (Q15). |
| Context-copy minimum n-gram 3 or 4 | Worse than the default 6 (wrong blocks). |
| RAMP with long copy blocks (`MTPLX_RAMP_ENABLED=1`) | Agent decode +22%, verify +106% per round, bit-identical lost (W1). |
| `MTPLX_COMPILED_VERIFY_ALLOW_BITS=6` / `FORCE=1` (compiled verify on 6-bit) | Eager + async chunk 100.3 vs compiled + async chunk 95.6 tok/s on A3B Balance; 4-bit Speed: eager + async +7.5% over compiled (findings 97, 98). |
| Cost-driven depth `--adaptive-policy cost` | Gains and losses cancel (+4% short, -4% long), 4 of 27 greedy streams diverge (finding 39). |
| `MTPLX_SESSION_BANK_SHED_BOUNDARIES=1` | Does nothing until an entry exceeds 8.7 GiB; dropped 0 times (finding 36). |
| `MTPLX_SESSION_BANK_SPIKE_BURSTS=16` | No effect in the measured workload; never verified under the intended long workload (finding 8). |
| `MTPLX_COMPLETIONS_SESSION_BANK=1`, prefix min-match 128 | No gain without shared prefixes, +0.8 GiB bank, classifier 145 ms slower, changes 8 of 12 greedy turns (findings 3, 4). |
| `MTPLX_SHORT_REQUEST_PRIORITY=1` | Titles wait -61% but agent output text changes in every on-run and +1.2 GB peak (finding 54). Off. |
| `MTPLX_SESSION_SNAPSHOT_SETTLE=1` (T2), K4 async prefill rungs | No gain / slower; the reported losses were disturbed runs (finding 69). |
| `MTPLX_STOCK_SAMPLED_DRAFT_CHAIN=1`, `MLX_MAX_MB_PER_BUFFER=1000` | +0.5 to 1.7%, within noise (Q6). |
| Verify core `linear-gdn-from-conv-stream-skip0` | Broken on Qwen3.8 (Q5). |
| Prefill chunk 6144 / 8192 | No gain at 6144; 8192 swapped 9.4 GB on 64 GB (Q3). |
| MLX main with open PRs #4572 / #4568, or MLX 0.32.3 | Output changes (7 of 240 classifier verdicts flip), 4 tests pin the old order; 0.32.3 is 0 to 2% slower (findings 71, 99). |
| Padding q8 lm_head calls to 13 rows, `MTPLX_GQA_PACKED_WIDE` + `MTPLX_NAX_TILE_ROUTE` | End to end neutral or worse (finding 108 notes). |
| Discounting restore-copy bytes in admission; `MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE=1` (lease) | The copy count is right; lease only changes a label (lease-vs-clone). |
| Existing paged KV layout as prefix sharing | 1.66x slower per verify at 50K, +6 to 10 GB peak (paged-prefix-sharing). |
| Eager merge of all KV segments after every turn | Brings the duplicate back; use tiered merges. |
| Fan mode `max`, 16 experts per token on Qwen3.6, Bonsai-2-27B for Dutch | Noise vs few percent; 25 to 35% slower; invented words in 18 of 24 answers (decisions 2 to 4 Oct). |

## 10. Open / not measured

| Item | Why it matters |
|---|---|
| Bink FR-Spec list on Qwen3.8 on `prod` (legacy lane), Dutch with and without thinking | Sources disagree whether it was measured; Dutch could be -3.5 to -12.4% (built-in list) or +10% (Bink list). |
| Per-switch effect of multi-row qmm, wide and dsplit4 attention, MINQ on Qwen3.6 (4-bit Speed and 6-bit Balance) and on the 9B | Only the carried work as a whole is measured on Qwen3.6 (+9 to +10% decode vs upstream flags, finding 99). |
| Async verify chunk on the 9B | Expected gain by analogy to 6-bit A3B; never run. |
| Whether `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` should stay in the shared command | Mixed on the 6-bit Balance pack, optional on the 27B. |
| Head anchor, incremental SSD and queue cap on Qwen3.8 as separate on/off runs | Measured on A3B Balance; for Qwen3.8 only inside the combined v2.12.2 comparison (D equals C). |
| Pauses between 0.25 and about 1 s between agent turns with an SSD encode running | The slowdown could still occur with fast tool calls (finding 68). |
| Effect of the 5 dropped SSD tasks (queue cap) on reuse after a restart | Not measured (finding 72). |
| Segmented KV: whether a fence should exist at all (queued step 8: 8K to 50K, segments from the start, eager verify), compiled-verify support, 80K on A3B, SSD tier on A3B and the 9B, sampled decoding, wide-window margins | Decides whether segments can be always-on. |
| Segmented KV on Qwen3-8B (head_dim 128 AR path) | Matrix running; the plain-attention hook was only added after the first check. |
| Context-copy AR on Qwen3-8B and on hybrid models | Only Llama 3.1 8B measured. |
| Gemma 4 and Llama-like models with segmented KV | Gemma has sliding layers; Llama attention is not hooked (reports `attention_not_hooked`, keeps the stock cache). |
| `mtplx serve --mtp` with a model that has no MTP head | Not checked. |
| Thinking budget 6144: evaluation due 2 Oct (per notes) | No written result found in the research folder. |

## 11. Contradictions and gaps found between the sources

1. **FR-Spec on Qwen3.8.** [VONDSTEN-QWEN38.md](VONDSTEN-QWEN38.md) Q2 reports a Bink-list run on 3.8 (+12% on the bench, Dutch +10 to +11%), while [frspec-qwen36](2026-09-30-frspec-qwen36.md) and finding 95 say that run was queued and refused. Production uses the legacy lane (`MTPLX_FRSPEC_LEGACY=1`), whereas the Q2 Bink-list rows were measured on the proper-binding branch.
2. **Compiled verify fence.** The profile text says compiled verify engages up to 32,768 tokens. Measurements on the 27B show it compiling at 50K and 80K in production (the ceiling is not applied to dense caches). Segmented KV's default fence follows the profile ceiling, so enabling it would drop compiled verify at those sizes (-3%).
3. **Chunk size 4096.** The profile sets 2,048; our command sets 4,096. On the 27B no difference is shown (22K cold 55.2 vs 57.0 s, single run). The measured gain (-12.2% cold) is from A3B only.
4. **`MTPLX_PREFILL_MIN_CHUNK_ROWS`.** Finding 100 says "enable for Qwen3.8 only"; the shared command applies it to every model, including the Qwen3.6 pack where it was mixed.
5. **Persistence cap.** CARRIED.md and finding 99 call the old queue flag superseded by `MTPLX_PERSISTENCE_MAX_PENDING_BYTES` (upstream default RAM/32, about 2 GiB), but our measured value is 4 GiB, so we run a looser cap than the default.
6. **#574 and FR-Spec PR #578: checked, no gap.** The #574 loader fix (024bba2c, with 13128e2e) is part of the v2.12.2 release `prod` is based on, so it is not carried. #578 (binding the pruned draft head on the configured route) is not on `prod`; production runs FR-Spec on the legacy lane (`MTPLX_FRSPEC_LEGACY=1`), which does not need it. Consequence: the Q2 rows measured on the #578 binding do not describe production, which is one more reason to re-measure FR-Spec on `prod` (queued).
7. **Qwen3.5-9B lane.** The 26 Sep model test installed the batch-invariant lane on the 9B; the MoE-only gate (finding 59) now refuses it on dense models, so the 9B numbers of that test do not describe `prod`.
8. **Segmented KV scope of 9B numbers.** Only in `m3/RESUME.md`, no report section.
9. **Task-named source missing.** `~/Dev/laya-nl/pld-pr/` does not exist; #599 numbers come from the `prod` CHANGELOG and finding 103.
