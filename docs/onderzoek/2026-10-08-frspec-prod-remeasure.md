# FR-Spec on Qwen3.8-27B, re-measured on `prod` exactly as production runs it (2026-10-08)

Question (from RECOMMENDED-SETTINGS.md section 11, item 1): sources disagree whether the Bink FR-Spec list was ever measured on Qwen3.8 in the production configuration (legacy lane). This measures it.

## Setup (measured, not assumed)
- Code under test: `~/Dev/MTPLX-prod` (branch `prod`, tag prod-2026-10-08, read-only) with `~/Dev/mlx-sdpa/pkg-combo` as MLX, started through the real `mtplx serve` command from the production launch entry in `config.json` (read-only). Only changed: port 8010 instead of 8000, and `MTPLX_SSD_SESSION_CACHE_DIR` pointed at a throwaway directory so the production cache is untouched. Everything else is the production env, including `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8`, `MTPLX_THINKING_BUDGET=6144`, context copy (default on).
- Variants (one fresh server process per run, model loaded each time):
  - A: production: `MTPLX_FRSPEC_DRAFT=1`, `MTPLX_FRSPEC_VOCAB=~/.mtplx/frspec/bink-64k.npy`, `MTPLX_FRSPEC_LEGACY=1`. Server log confirms installed: n 65,536, `legacy_swap: True`.
  - B: `MTPLX_FRSPEC_DRAFT=0` (log: "[frspec] disabled"; the other two variables left as they are).
  - C: built-in list: `MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`, DRAFT=1, LEGACY=1.
- Order A B C C B A A B C (3 processes per variant, palindromic so drift cancels), each after the GPU gate (thermal 0).
- Per process: one short warm-up request, then 6 requests through `/v1/chat/completions`, greedy (temperature 0), thinking off, max 400 tokens, fresh prompts of about 1,000 to 1,400 tokens (no cache reuse): Dutch prose (summary of two Dutch documents from this repo), English prose (summary of two research docs), code edit (rewrite a Python file with prefixed function names). Per category: mean of its 2 prompts per process, median over the 3 processes. tok/s and acceptance are the server's own request-log values (`decode_tok_s`, `accepted_drafts`/`drafted_tokens`).
- Scripts and raw data: `~/Dev/laya-nl/segment-kv/m3/fr/` (`frdrive.py`, `frrun.sh`, `queue_fr.sh`, `raw-r*.jsonl`).

## Results (measured)
| Category | A prod (Bink) tok/s | B FR-Spec off tok/s | C built-in tok/s | Acceptance A / B / C | Tokens per verify A / B / C |
|---|---|---|---|---|---|
| Dutch prose | 29.6 | 27.8 | 25.4 | 0.493 / 0.493 / 0.380 | 2.48 / 2.48 / 2.14 |
| English prose | 29.9 | 28.1 | 29.9 | 0.515 / 0.515 / 0.515 | 2.52 / 2.52 / 2.52 |
| Code edit | 164.7 | 164.1 | 164.6 | 1.00 / 1.00 / 1.00 | 21.05 / 21.05 / 21.05 |

Individual process runs differ by under 0.5 tok/s inside a variant (A Dutch 29.6, 29.5, 29.6; B 27.8 x3; C 25.4 x3), so the differences are well outside run noise.

- Bink list (A) vs off (B): Dutch +6.5 %, English +6.4 %, code edit +0.4 % (noise). The gain is a cheaper draft head (64K rows instead of 248K); acceptance is identical to B, so the list loses no drafts on this material.
- Built-in code list (C) vs off: Dutch -8.6 % (acceptance falls from 0.49 to 0.38), English +6.4 %, code +0.3 %. It is worse than the Bink list for Dutch, as expected from a list ranked on code, and equals the Bink list for English.
- Greedy identity: for all 6 prompts the output text is byte-identical across A, B and C and across the 3 repeats (one distinct output hash per prompt and variant; no divergence in the first 400 tokens). FR-Spec changes the draft, not the verified output, and no bf16 tie flipped here.
- The code-edit prompt is copy-heavy: context copy accepts about 21 tokens per verify, so the draft head is hardly used and FR-Spec cannot matter there (164 tok/s in all variants). A code-writing prompt without a copy source was not measured; the earlier report's code gain (+5 to +8 %) therefore cannot be confirmed or refuted by this row.

## What this settles
- The Bink-list gain on Qwen3.8 in the production configuration is confirmed: about +6.5 % for Dutch and English prose with thinking off. It is smaller than the +10 to +12 % that VONDSTEN-QWEN38 Q2 reports for the Bink list, so that figure should not be relied on for the legacy lane. The Dutch loss reported for the built-in list (-3.5 to -12.4 %) is reproduced for the built-in list only (-8.6 % here), not for the Bink list.
- Keep `MTPLX_FRSPEC_DRAFT=1` with the Bink list; `MTPLX_FRSPEC_DRAFT=0` is not needed as a fallback.

## Not measured
Thinking on (reasoning output), sampled decoding, very long contexts (prompts here are about 1.2K tokens), code writing without a copy source, agentic/tool prompts, and concurrent requests.
