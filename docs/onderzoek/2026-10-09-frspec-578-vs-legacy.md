# FR-Spec: the #578 binding against production's legacy lane, and the Bink v2 table (2026-10-09, measured)

Question: on Qwen3.8-27B, does the #578 binding of the pruned draft head (`fix/frspec-configured-route`) do better, equal or worse than the legacy lane production runs (`MTPLX_FRSPEC_LEGACY=1`), both with the Bink list; and is `bink-64k-v2.npy` (built with `mtplx frspec build`, with a sidecar) better than `bink-64k.npy` (production, no sidecar)?

## Setup
- Tree: `test/frspec-578` = candidate (`prod-segkv-candidate`, rebased on prod c3b138ee with #580) plus bc4618c4 (#578) as f329af24; `~/Dev/laya-nl/mtplx-frspec578`. Segmented KV off (short prompts; the question is the draft head). MLX: `pkg-combo-0323` (0.32.3.lse). The production launch command from `config.json` (read-only) with port 8010, a throwaway SSD cache directory and the PYTHONPATH of that tree.
- Variants (a fresh server process per run, 12 runs, order L N V O O V N L L N V O, 3 per variant, each behind the GPU gate at thermal 0):
  - L: production legacy lane, `bink-64k.npy`, `MTPLX_FRSPEC_LEGACY=1` (install report: `legacy_swap: True`, `binding: configured_draft_head`).
  - N: #578 binding, `bink-64k.npy`, no LEGACY (install report `legacy_swap: False`, `binding: configured_draft_head`).
  - V: legacy lane with `bink-64k-v2.npy` (log: installed, 65,536 rows, no tokenizer mismatch; the sidecar check of #580 passed on Qwen3.8).
  - O: `MTPLX_FRSPEC_DRAFT=0`.
- Requests: greedy chat requests through `/v1/chat/completions`, max 400 tokens (500 for thinking), 2 prompts per category: Dutch prose, English prose, code edit (copy-heavy: context copy accepts about 21 tokens per verify), and thinking on (two arithmetic and combinatorics questions, `enable_thinking`). tok/s and acceptance from the server request log; per category the mean of the 2 prompts per process, median over 3 processes. Scripts and raw data: `m3/fr/` (`frdrive2.py`, `frrun2.sh`, `queue_fr2.sh`, `an2.py`, `raw2-*.jsonl`).

## Results
| category | L legacy | N #578 | V legacy + v2 table | O FR-Spec off | acceptance (all FR-Spec variants / O) | tokens per verify |
|---|---|---|---|---|---|---|
| Dutch prose, tok/s | 29.2 | 29.2 | 29.2 | 27.5 | 0.491 / 0.491 | 2.46 |
| English prose | 32.5 | 32.5 | 32.5 | 30.4 | 0.584 / 0.584 | 2.74 |
| code edit | 164.6 | 164.3 | 164.4 | 164.1 | 1.00 / 1.00 | 21.05 |
| thinking on | 41.3 | 41.3 | 41.4 | 40.0 | 0.770 / 0.793 | 3.44 / 3.52 |

- The three FR-Spec variants are identical within 0.3 tok/s in every category (runs of one variant differ by under 0.2 tok/s). The #578 binding is neither faster nor slower than the legacy lane; the v2 table is neither better nor worse than the production table on this material.
- FR-Spec against off: Dutch +6.2 %, English +6.9 %, thinking +3.3 %, code edit +0.3 % (noise). Acceptance is equal in the prose categories, and in thinking it is 0.770 against 0.793 (the pruned head misses some tokens), so the thinking gain comes from the cheaper head despite slightly lower acceptance.
- Greedy identity: for all 8 prompts the output text is byte-identical across L, N, V and O and across the 3 repeats (one distinct hash per prompt and variant).
- Table overlap: 62,993 of 65,536 ids are shared; 2,543 differing ids did not change a single drafted token in these 8 prompts.

## Recommendation
- Production can keep the legacy lane with `bink-64k.npy`; #578 gives the same speed and the same acceptance, so it is a cleanliness fix (the configured route works without LEGACY), not a speed change, and can go upstream on its own merits. If production moves to the #578 binding later, `MTPLX_FRSPEC_LEGACY=1` can be dropped without a measurable change.
- `bink-64k-v2.npy` is a safe swap (passes the tokenizer check, same speed, same output) and brings the sidecar protection of #580; its advantage over the old table on text with other tokens than these prompts is not measured here (the Bink v2 holdout coverage numbers belong to `mtplx frspec build`).
- Not measured: sampled decoding, long contexts, tool prompts, prompts in other languages, and long thinking traces (thinking requests here were cut at 500 tokens).
