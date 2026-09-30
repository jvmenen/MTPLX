# Research: MTPLX as a local classifier and agent backend

This directory belongs to the fork [jvmenen/MTPLX](https://github.com/jvmenen/MTPLX) and lives on the `onderzoek` branch, separate from the feature branches, so it never ends up in a pull request to the original. It holds measurements, findings and agreements from the work on MTPLX: as a fast local classifier (reading probabilities over fixed labels instead of generating text) and as a backend for local agents.

**Are you an agent continuing this work? Read the ground rules below first, then [VONDSTEN.md](VONDSTEN.md).**

## Where to start

1. **[VONDSTEN.md](VONDSTEN.md)**: the running work list. Open items with source and next step; handled items at the bottom. Pick your work here.
2. **The "Branches" table** below: which branch is at which stage.
3. **The report** matching your topic (table "Reports"), for background and earlier measurements.
4. **Background in one go**: [metingen-classifier](2026-09-25-metingen-classifier.md) (where the time goes) and [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) (the lesson on MoE routing and block sizes).

## About the codebase

- MTPLX is largely written by AI agents under the direction of the maker (Youssof): 30 to 90 commits per day, commits authored by "Claude Code" and "Codex", a `mistakes/` directory with lessons for agents. Expect very large files (`server/openai.py` ~39k lines, `generation.py` ~16k), functions hundreds of lines long, many env toggles, and extensive comments with measurements ("receipts").
- Consequence for us: keep PRs small and focused (large restructurings quickly fall behind `main`), rebase branches just before a PR, and read code with agents and file:line references.
- The maker steers by measurements: back every change with numbers on the behavior that changes.

## Ground rules

### Documenting

- **Every investigation or measurement gets a report** in this directory: `YYYY-MM-DD-topic.md`, in English (agreement from 30 September 2026; older reports are in Dutch). New results for an existing topic are added as a dated section to that report.
- **Update three things with every change:** the report, [VONDSTEN.md](VONDSTEN.md) (status and next step; handled items move down with date and outcome) and the tables in this README.
- **Separate measured from estimated.** Note where a number comes from (real model, synthetic tensors, test model, code read only), with hardware, model, profile and date.
- **Public:** the fork is public. No client names, private messages or chat content in reports; only counts and outcomes.
- **Writing style:** short, businesslike, no em-dashes. Run the spelling check (LanguageTool; jargon such as "forward pass" may stay) before committing.
- **Committing and pushing:** in the local worktree `~/Dev/MTPLX-onderzoek`, then `git push fork onderzoek`. Commit messages end with the Co-Authored-By line.

### Code and branches

- **One topic per branch**, based on `origin/main`, each in its own worktree (`~/Dev/MTPLX-<topic>`), so agents don't get in each other's way. Never work in another agent's worktree.
- **Do not change default behavior** unless it is proven to be better; new behavior goes behind a switch first (env or config, in the style of the existing toggles).
- **Clean Code:** small functions with clear names, minimal diffs in the large files (`openai.py` ~39k lines, `generation.py` ~16k), no unrequested reformatting.
- **Tests:** new tests with every change; run the relevant existing tests and compare against the baseline. On a clean `main`, 19 tests fail in this environment (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`) due to our `~/.mtplx/config.toml` and 64 GB of memory (see [falende-tests](2026-09-26-falende-tests.md)): only report deviations from that. Run tests with `MTPLX_CONFIG=/nonexistent` in the env; without that variable the forge tests write directories into the real `~/.mtplx/models`. On a branch with `fix/test-isolation` in it that is no longer necessary. Ruff must not produce new findings (compare the count with `main`).
- **Python:** venv `~/Dev/MTPLX/.venv` (editable, dev and server extras). In another worktree: set `PYTHONPATH=<worktree>` and check with `python -c "import mtplx; print(mtplx.__file__)"`.

### The real model and the server (important)

- **Small test models don't prove enough.** A change that was bit-identical on the test model gave different results on Qwen3.6-35B-A3B (MoE routing depends on the number of rows per forward pass; see [snellere-scoreroute](2026-09-26-snellere-scoreroute.md)). Test anything that touches compute paths on the real model first before a PR.
- **There is one GPU and 64 GB of memory.** The model (~20 GB) may only be loaded once. Never load the model in a second process while a server is running; no GPU-heavy work alongside a measurement (that caused HTTP 507).
- **The production server on port 8000 is managed by the Bink platform** (Bink, Fleur, chat titles and the nightly review use it). The platform starts it itself with the Balance model when needed and stops it after 15 minutes of inactivity.
- **Starting a test server** is only allowed on port 8000, with Jeroen's permission, so the platform doesn't start a second copy:
  - stop: `mtplx stop --port 8000`, wait until `pgrep -f mtplx.server.openai` is empty;
  - start from the worktree: `env PYTHONPATH=<worktree> nohup ~/Dev/MTPLX/.venv/bin/python -P -m mtplx.server.openai ${=ARGS} > log 2>&1 &` with `ARGS=$(cat ~/Dev/laya-nl/mtplx-originele-args.txt)` (the production settings). In zsh use `${=ARGS}`, otherwise all arguments arrive as a single argument. Only the venv-python knows `mtplx`.
  - stop it again afterwards and leave port 8000 free.
- **The platform restarts MTPLX when someone requests a different model.** All users on the Bink platform (agents, chat titles, nightly review) must therefore request the same model (Balance); otherwise the server keeps switching back and forth, and each switch costs minutes. A stopped production server is started by whoever needs it first, with the model that requester asks for.
- **The Bink platform can send its own requests to the test server on port 8000 during a measurement** (seen in [messages-ttft](2026-09-26-messages-ttft.md)). After each workload, check `requests_completed` in `/health` against the expected count and repeat a disturbed run.
- **Measure on a fresh server per variant**: the MLX peak measurement never decreases, and caches would otherwise skew the comparison.
- **Measuring memory:** `/health` only shows `session_bank.effective_max_bytes`; active memory, peak and working memory are in `/v1/mtplx/snapshot` under `mem.*`.
- **Measurement material** (local, not in the fork): `~/Dev/laya-nl/` with `data/scoring-prompts.jsonl` (240 classifier prompts), `verify_scoring_real.py` (old vs. new scoring route), `test_live_classifier.py` and `resultaten/`.

### Pull requests to the original

- **Only open a PR after Jeroen's approval**; pushing to the fork is fine.
- **One clean branch per PR** with only the intended commits (no reverted experiments left in the history).
- **Before the PR**, the checks from `CONTRIBUTING.md`: relevant tests, `python -m build`, `scripts/fresh_venv_smoke.sh`; a CHANGELOG line under `## [Unreleased]` with measurement data (hardware, model, profile, fan mode, date, commit).
- **PR text is about the change**: what, why the current code falls short, and measurements of that change (e.g. bit-identical and faster than the old path). No comparisons between models or results from our own project. In English.

### Agents

- Jeroen uses Opus agents for this work. Give every agent its own worktree, the boundaries above (no server or real model without explicit permission, no pushing) and ask for a report with file:line references.
- At most a few agents at a time: they share the CPU, and measurements on the real model can only run one at a time.

## Reports

| Date | Report | Topic |
|---|---|---|
| 2026-09-25 | [metingen-classifier](2026-09-25-metingen-classifier.md) | Where the time per classification goes, reuse of the prompt prefix, prompt order |
| 2026-09-25 | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Logprobs for the first generated token (branch `feat/first-token-logprobs`, PR #530) |
| 2026-09-25 | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Why a short shared prompt prefix is not reused and how 128 becomes configurable; completions in the session bank |
| 2026-09-26 | [live-test-classifier](2026-09-26-live-test-classifier.md) | Live test of first-token logprobs and prefix reuse together on 240 messages |
| 2026-09-26 | [opschonen-prefix-helpers](2026-09-26-opschonen-prefix-helpers.md) | Faster prefix comparison, one reader per setting |
| 2026-09-26 | [optimalisaties](2026-09-26-optimalisaties.md) | Ranked optimizations with effect, quick wins and what isn't worth it |
| 2026-09-26 | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Top-K without full log-softmax (bit-identical, 1.2x faster); why wider blocks go wrong (MoE routing) |
| 2026-09-26 | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Only tokenizing new conversation pieces; server measurement and finishing touches for the PR |
| 2026-09-26 | [bankplafond](2026-09-26-bankplafond.md) | Why the session bank ceiling sinks and a cautious fix |
| 2026-09-26 | [deepseek-optimalisaties](2026-09-26-deepseek-optimalisaties.md) | Which optimizations from DeepSeek V4.1 apply to MTPLX and Qwen3.6 |
| 2026-09-26 | [completions-overhead](2026-09-26-completions-overhead.md) | Where ~0.8 s outside `elapsed_s` goes (blank retries) and the ~50 ms per chat request (Gemma vocab check); two fixes |
| 2026-09-26 | [falende-tests](2026-09-26-falende-tests.md) | Why 19 tests fail on a clean `main` (user config, memory) and the isolation fix |
| 2026-09-26 | [messages-ttft](2026-09-26-messages-ttft.md) | Why agent turns via `/v1/messages` take 26-45 s to the first token (a false retry after a bare tool call) and the fix |
| 2026-09-26 | [instellingen-36-37-39](2026-09-26-instellingen-36-37-39.md) | KV in q8, cost-driven draft depth, dropping boundaries and the bank switch measured on the integration branch: none of the four in the final config |
| 2026-09-26 | [chatroute-scoped](2026-09-26-chatroute-scoped.md) | Why chat prefill is slower (extra forward for the GDN boundary) and per-turn encoding in scoped mode |
| 2026-09-26 | [batching](2026-09-26-batching.md) | Batching together with MTP: which modes exist and why `mtp_batch` yields little for Bink |
| 2026-09-26 | [compiled-verify-6bit](2026-09-26-compiled-verify-6bit.md) | Why verify on the 6-bit Balance model runs eager (caution, not a bug) and what compiled delivers: +1 to +3% decode, not bit-identical at long context; not in the final config |
| 2026-09-26 | [eindbenchmark](2026-09-26-eindbenchmark.md) | 2.11.3 against the integration branch (and clean 2.12.0) on the real model: agent conversation -64%, scoring route -17% bit-identical, first-token logprobs -37% without completions bank, decode +6% from upstream; attribution per fix. Final test 2 (evening, 49c4a6ee with invariant prefill, without bank and the 128 threshold): scoring route -36%, first-token logprobs -39%, agent -66%; new recommendation for the switchover |
| 2026-09-26 | [batch-invariante-router](2026-09-26-batch-invariante-router.md) | Why prefill rows on A3B depend on block size (split-K, expert and SDPA kernels in MLX, SDPA via `attention_split` under turbo) and a switch that fixes it: scoring bit-identical for any block size from 9 rows, 1.28x faster, no noticeable cost outside scoring; recommended for the final config |
| 2026-09-26 | [gdn-inforward-a3b](2026-09-26-gdn-inforward-a3b.md) | GDN boundaries inside the forward for A3B (finding 21), only with the invariant prefill: bit-identical to the ladder, first-token logprobs with bank from 512 tokens -21%, one fewer forward |
| 2026-09-26 | [voorrang](2026-09-26-voorrang.md) | Priority for short requests in the serial queue (switch, off by default). Repeat measurement 27 Sep with titles of 60 tokens: title wait time p95 -56%, but agents finish 11% later (lost bank hits) and one chat text differs; decision rule not met, no PR |
| 2026-09-26 | [modeltest](2026-09-26-modeltest.md) | Integration branch on Qwen3.5-9B, Ternary-Bonsai-2-27B and Gemma 4 against 2.12.0: Bonsai OK; 9B works (agent -70%) but other greedy texts due to the invariant prefill (extra measurement complete); Gemma without regression, classification routes also unavailable on main; Gemma support for first-token logprobs built |
| 2026-09-27 | [herstelpaden-en-gemma-scoring](2026-09-27-herstelpaden-en-gemma-scoring.md) | Finding 44: recovery after a retry starts from the retry prompt (on the server: fully from the bank, 51 new tokens) and statistics over all attempts. Finding 57: scoring route on Gemma 4 works (60/60, deterministic). Both ready for a PR after approval |
| 2026-09-30 | [qwen38](2026-09-30-qwen38.md) | Qwen3.8-27B dense: baseline, variants (draft temp, FR-Spec, chunk size, KV q8, verify-core), own findings list [VONDSTEN-QWEN38](VONDSTEN-QWEN38.md); plus finding 90 (fused MTP experts in the loader) |
| 2026-09-30 | [building-mtp-packs](2026-09-30-building-mtp-packs.md) | How-to (English): building MTPLX packs with a BF16 MTP head, fused MTP experts, keeping an existing body and swapping only the head, measuring, pitfalls |
| 2026-09-30 | [feature-gaps](2026-09-30-feature-gaps.md) | Features built for one model or pack but missing for another, ranked by effort against expected gain, with a suggested PR order |
| 2026-09-30 | [kv-quantization](2026-09-30-kv-quantization.md) | KV quantization (q8/q4) at 20K and 60K on Qwen3.8-27B and Qwen3.6: slower everywhere (-25% to -72%), bank not smaller (snapshots store bf16); cause is a mask check that forces full dequantization, fix on `perf/kvq8-compiled-verify` recovers about half; recommendation KV off |
| 2026-09-27 | [chat-decode-en-overhead](2026-09-27-chat-decode-en-overhead.md) | Findings 60 and 52, repeated with a 100 W charger: the difference in greedy decoding against 2.11.3 is text luck, not slower computation. Decision: the three upstream changes (correctness fix e36f5ffb, prefill speedups 9a86dd6c and 4314638a) stay; only the cost of the gate helper still to measure (`gatekosten.zsh`). Finding 52: the gap grows with conversation history (2.6 to 52 ms); phase breakdown failed due to a clock mismatch between client and hook, hook version 2 built and tested offline (`meet52b.zsh`) |

## Branches

All branches except `feat/faster-prompt-scoring` live in the fork on GitHub; locally they work in `~/Dev/MTPLX` and the worktrees `~/Dev/MTPLX-<topic>`.

| Branch | Content | Status |
|---|---|---|
| `feat/first-token-logprobs` | First-token logprobs | Pushed; PR [#530](https://github.com/youssofal/MTPLX/pull/530) submitted |
| `feat/first-token-logprobs-gemma4` | First-token logprobs also on the Gemma 4 pair (on `feat/first-token-logprobs`) | Pushed (4f7b3cbb), 8 tests; tested on the server 26 Sep: works and is deterministic. Jeroen's decision: update #530 or a follow-up PR ([modeltest](2026-09-26-modeltest.md)) |
| `feat/prompt-scoring-topk` | Clean PR branch: only the faster top-K of the scoring route | Pushed; PR [#532](https://github.com/youssofal/MTPLX/pull/532) submitted |
| `feat/faster-prompt-scoring` | Working branch scoring route (A, B and reverting B) | Local only; replaced by `feat/prompt-scoring-topk`, may be removed |
| `feat/chat-encode-segment-memo` | Chat encode memo per segment | Pushed; PR [#533](https://github.com/youssofal/MTPLX/pull/533) submitted |
| `fix/bank-ceiling-peak-decay` | Peak reserve of the bank ceiling decays | Pushed (fb1cd817), off by default; measure working memory first |
| `feat/prefix-reuse-block` | Reuse of a prompt prefix shorter than 512 tokens, completions in the session bank | Pushed (fe32206c), live tested; PR decision open (finding 31) |
| `refactor/prefix-helpers` | Faster prefix comparison, one reader per setting (on `feat/prefix-reuse-block`) | Pushed (d32c77b2, 412e1571); PR decision open (finding 31) |
| `fix/prefix-miss-reason` | Real rejection reason from working memory in `/health` (finding 11) | Pushed; PR [#534](https://github.com/youssofal/MTPLX/pull/534) submitted |
| `feat/sessionbank-put-timing` | `sessionbank_put_s` in the stats (finding 23, step 1) | Pushed (dbb4bfea); measured on the real server (put 0.5-4.4 ms, step 2 not needed); no PR, stays as evidence |
| `fix/test-isolation` | Tests no longer read the user config; Laguna route test decoupled from memory (finding 10) | Pushed (0e1b6b91); PR [#535](https://github.com/youssofal/MTPLX/pull/535) submitted |
| `test/classifier-live` | First-token logprobs and prefix reuse combined for the live test | Pushed, for measurements only |
| `fix/completions-overhead` | No blank retries during greedy decoding; `usage` only counts the returned attempt | Pushed (b921b9a6); measured on the real server; PR decision open (finding 42) |
| `fix/gemma4-probe-vocab` | Gemma 4 check without `get_vocab()` (~47 ms per chat request) | Pushed (2a1057c7); measured on the real server; PR decision open (finding 42) |
| `onderzoek-completions-overhead` | Working branch for the overhead investigation, equal to `main` 1de2b1c0 | Local only, may be removed |
| `fix/messages-ttft` | No longer repeating a bare well-formed tool call as orphan markup (finding 35) | Pushed (2f38106b); measured on the real server; PR decision open (finding 43) |
| `onderzoek-messages-ttft` | Working branch for the messages investigation, equal to `main` 1de2b1c0 | Local only, may be removed |
| `perf/chat-scoped-segments` | Encoding scoped chat per turn (finding 34) | Pushed (54c4ca5f); in `perf/integratie`; PR decision open (finding 50) |
| `perf/integratie` | All fixes together for the final benchmark | Local only (`~/Dev/MTPLX-integratie`, now `f26dc9a8` after fast-forward to `perf/invariant-lane-moe-only`); pushing was rejected (finding 48). Final benchmark run (26 Sep, [eindbenchmark](2026-09-26-eindbenchmark.md)), final test 2 with the lane (recommended for the switchover); used by the Bink platform since the switchover (finding 46) |
| `perf/batch-invariant-router` | Batch-invariant prefill (finding 29), commit B and tail boundaries only with the switch | Pushed (1ce69614); measured on the real server (26 Sep); recommended for the final config and `perf/integratie`, PR decision open |
| `perf/a3b-inforward-boundaries` | In-forward GDN boundaries for A3B (finding 21), on `perf/batch-invariant-router` | Pushed (198e7194); measured on the real server (26 Sep, bit-identical, -21% from 512 tokens with bank); in `perf/integratie` (fast-forward); PR decision together with finding 29 |
| `perf/invariant-lane-gate` | Only install the lane if it covers every projection (rejects e.g. Bonsai: `unsupported_linear:HadamardQuantizedLinear`); reason shown in `/health` | Pushed (49c4a6ee); in `perf/integratie` (fast-forward); measured on A3B on the server in final test 2 (lane installed, 30 GDN layers, [eindbenchmark](2026-09-26-eindbenchmark.md)); costs something on dense models (finding 59) |
| `perf/invariant-lane-moe-only` | Lane only on MoE models (finding 59): dense model rejected with `dense_model`, switch `MTPLX_BATCH_INVARIANT_PREFILL_DENSE=1` to install it anyway; on `perf/invariant-lane-gate` | Pushed (f26dc9a8); in `perf/integratie` (fast-forward); this is the branch the Bink platform uses via the switchover (finding 46) |
| `perf/short-request-priority` | Priority for short requests in the serial queue (finding 54) | Pushed (67211d39), off by default; repeat measurement 27 Sep: titles faster, agents finish 11% later; no PR, stays as evidence |
| `fix/stream-recovery-paths` | Reasoning recovery follows the retry prompt, statistics over all attempts (fbde4931); retries off under `MTPLX_AGENT_REWRITES=off` (71270aab) (finding 44) | Pushed; tested on the server 27 Sep; PR after Jeroen's approval |
| `fix/gemma4-prompt-scoring` | Scoring route on Gemma 4 via one forward and a logits cap per block (finding 57) | Pushed (45ed152b); tested on the server 27 Sep; PR after Jeroen's approval |
| `pr/messages-ttft` | PR branch for `fix/messages-ttft` (finding 43): no longer repeating a bare well-formed tool call as orphan markup | Pushed; PR [#539](https://github.com/youssofal/MTPLX/pull/539) submitted |
| `pr/completions-overhead` | PR branch for `fix/completions-overhead` (finding 42): no blank retries at temperature 0 | Pushed; PR [#540](https://github.com/youssofal/MTPLX/pull/540) submitted |
| `pr/gemma4-probe` | PR branch for `fix/gemma4-probe-vocab` (finding 42): Gemma 4 check without `get_vocab()` | Pushed; PR [#541](https://github.com/youssofal/MTPLX/pull/541) submitted |
| `pr/chat-scoped-segments` | PR branch for `perf/chat-scoped-segments` (finding 50): encoding scoped chat per turn | Pushed; PR [#542](https://github.com/youssofal/MTPLX/pull/542) submitted (depends on #533) |
| `pr/first-token-logprobs-gemma4` | PR branch for `feat/first-token-logprobs-gemma4`: first-token logprobs on Gemma 4 | Pushed; PR [#543](https://github.com/youssofal/MTPLX/pull/543) submitted (depends on #530) |
| `perf/kvq8-compiled-verify` | Quantized paged adapter serves its own tail mask (`MTPLX_KV_QUANT_TAILMASK_ELIDE`, off by default) | Pushed (6ce1fad4), measured on the real server (30 Sep, [kv-quantization](2026-09-30-kv-quantization.md)); PR only after Jeroen's approval |
| `onderzoek` | This documentation | Pushed, ongoing |
