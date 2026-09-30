# Feature gaps: built for one model or pack, missing for another (30 September 2026)

MTPLX gates many speed features to the exact model or pack they were measured on (strict contract checks, then fallback to the generic path). That keeps quality safe, but everything just outside those packs misses out: our own Forge packs, Qwen3.8-27B with its configured draft head, a 6-bit body. This list collects the gaps we ran into, ranked by expected gain against effort, for choosing our next PRs.

Scale: effort S (under a day of agent work plus one measurement), M (a few days, touches hot paths), L (new kernel or large refactor). Gain as measured or estimated for our use (Bink agents on Qwen3.6, Qwen3.8 for long quality tasks).

## Ranked

| Rank | Gap | Works for | Missing for | Effort | Expected gain | Evidence | Next step |
|---|---|---|---|---|---|---|---|
| 1 | Forge does not write the measured best depth to `recommended_mtp_depth`, and forge-local packs get no stable `public_model_id` unless the name matches a first-party pattern | first-party branded packs | our Forge packs (Apodex, Qwen3.6 BF16-MTP) | S | usability: packs serve at the right depth with a stable id without hand-editing `mtplx_runtime.json` | verify picked D2 for Apodex, runtime json kept `recommended_mtp_depth: null`; launcher needed a manual id ([building-mtp-packs](2026-09-30-building-mtp-packs.md), step 4) | small PR: stamp verify's best depth and a slug of the branded name |
| 2 | `repetition_penalty` accepted but ignored | presence/frequency penalties only | clients that send it (FrontierAgent; Apodex recommends 1.05) | S (refuse or warn) / M (implement in sampling, draft and verify paths) | correctness; no speed gain; matters for models prone to repetition | finding 91 | PR: first refuse/warn, then implement if a model needs it |
| 3 | Queue wait is not in the request log (`lock_wait_time_s` 0 while requests waited 11 s) | | serial scheduler under parallel load | S | observability of multi-agent load | finding 92 | small PR: add queue-wait seconds per request |
| 4 | FR-Spec draft head is only bound to the live draft route on Flash-Next (`_mtplx_bind_draft_lm_head`); on other models it installs but is unused unless `MTPLX_FRSPEC_LEGACY=1` | Qwen3.8 Flash-Next | Qwen3.8-27B (pack qualifies: lm_head Q8/g64) | S-M | +7-8% decode on 3.8-27B (measured with the legacy lane) | [VONDSTEN-QWEN38](VONDSTEN-QWEN38.md) Q2 | PR: bind the pruned head on the configured-draft-head route, make it the family default for 3.8-27B after an acceptance check on non-code text |
| 5 | Fused MoE experts in MTP heads dropped at load | Forge for Flash-Next (`forge_qwen4_exp`) | Qwen3.5/3.6 loader | done | +32% D2 on a Forge pack of the official release | finding 90, PR #574 | follow up on the PR |
| 6 | FR-Spec vocabulary exists only for Qwen3.8 (`qwen38-code-64k`) | Qwen3.8 | Qwen3.6 (248K vocab, same kind of draft lm_head cost) | M (build a ranked vocab from real traffic, bind as in 4) | unknown; drafts are a small share of a round on the MoE model, estimate a few percent | estimate from the 3.8 round breakdown | only after 4 lands; measure draft share on 3.6 first |
| 7 | Sampled draft chain and K20 prescatter require the Flash-Next FR-Spec route | Flash-Next | other models (stock draft loop: one sync per depth) | M | +0.5-1% on 3.8-27B, inside noise (draft host gap is ~0.5 ms per round) | [VONDSTEN-QWEN38](VONDSTEN-QWEN38.md) Q6, branch `perf/qwen38` befe8933 | park; revisit only together with 4 |
| 8 | Compiled verify detaches when paged KV quantization is on | KV off | long-context use with `--paged-kv-quantization q8` | L | halves KV memory on 3.8-27B (64 KB per token) at no speed cost, instead of -9% | Q4 (`kvq8` variant: 30.8 against 34.1 tok/s) | only if long sessions start cold-prefilling because of memory pressure |
| 9 | `mtp_batch` (real batching with MTP) is tied to one A3B body layout (GDN postconv quantization contract, depth 1, 8 requests, context 131,072, `target_prefix`/`stock` verify) | one 35B-A3B layout | the 6-bit Balance body and our packs | L | only under parallel load; serial gives 86-98 tok/s at 4 parallel requests, no other mode beat it | finding 92 | park until several agents run on the local model at once |
| 10 | Forge has no "keep an existing body, swap only the MTP head" mode | | anyone improving a published pack's head | M | reproducibility of the combo packs (label scoring identical to the original) | [building-mtp-packs](2026-09-30-building-mtp-packs.md), step 5 | low priority; the manual recipe works |
| 11 | `MTPLX_FUSE_GDN_POST_CONV` hard-coded to the A3B contract (40 layers, 16 heads) | 35B-A3B | dense 3.8-27B (48 GDN layers) | M-L | small at best: 3.8-27B verify is bandwidth-bound (matmuls 250-280 GB/s), GDN non-matmul work ~0.14 ms per layer and mostly overlapped | Q6 analysis | do not pursue |

Not gaps (checked and deliberate): the batch-invariant prefill lane and the A3B MoE prefill combine are MoE-only by nature; the long-context depth policy is correctly off for 3.8-27B (D3 beats D2 at 20K-60K, Q7); the tune cap at D3 for Qwen3.8 follows the maker's measurement (D6 -27% against D3).

## Suggested order

1. Quick wins, one small PR each: ranks 1, 2 (refuse or warn first) and 3.
2. The one clear speed win: rank 4 for Qwen3.8-27B, with an acceptance check on Dutch and non-code text because the vocabulary is ranked on code.
3. Everything else only when its trigger appears (parallel agent load, memory pressure at long context, a model that repeats itself).
