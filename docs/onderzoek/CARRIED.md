# Carried on `prod`

Commits on `prod` that are not in the maker's release it is based on. Policy: [FORK-POLICY.md](FORK-POLICY.md).

**Base:** v2.12.2 (9882703f). **Built:** 4 October 2026, from `perf/definitief-2122` plus the work below it. **Deployed:** tag `prod-2026-10-09b` (prod-2026-10-09 with the pre-release regex change reverted; before that prod-2026-10-08 plus MLX 0.32.3 test compatibility and the Claude Code session header #606; config 9 Oct switches MLX to jvmenen/mlx `prod-2026-10-09`). **Previous production:** `perf/definitief` (v2.12.0 base), kept as fallback.

| Commit(s) on `prod` | Change | PR | State | Switch in the launch command |
|---|---|---|---|---|
| 1e4df40e, b89389af, a783459f | Batch-invariant prefill for MoE models, scoring trunk, no cold tail grid | #549 | open | `MTPLX_BATCH_INVARIANT_PREFILL=1` (MoE only; refused on dense Qwen3.8) |
| 0024868e | In-forward GDN boundaries on the invariant lane | #550 | open | on with the lane |
| 16f2b1c5 | One-kernel MoE combine | #558 | open | `MTPLX_A3B_MOE_PREFILL_COMBINE=1` |
| 98df0c71 | Session bank head anchor | #559 | open | `MTPLX_SESSION_HEAD_ANCHOR=1` |
| 85a89e15 | Scoped reasoning history keeps the in-round postcommit | #570 | open | default |
| 8b712048, e05f0b66, 503ee21f | Eager verify async chunk submits | #579 | open | `MTPLX_VERIFY_ASYNC_CHUNK_LAYERS=8` |
| 8e98ba2f, 2149d58a, de73c92a, ce6c5da6, 774967b8 | Short shared prompt prefixes, completions session bank, put-time stat | not filed | fork only | not in the command |
| c6e5485a, 8bcc9246, ca894734 | Async rung states, short-request priority, spike reserve | not filed | fork only | not in the command |
| 71288758 | Prefill chunk plan (merge a small final chunk) | #597 | open | `MTPLX_PREFILL_MIN_CHUNK_ROWS=1024` |
| 460191d0 | Multi-row 4-bit verify matmuls, 5 to 32 rows | #595 | open | `MTPLX_MULTIROW_QMM=1` |
| b9b509ad | Wide verify attention, 9 to 32-row windows | #593 | open | `MTPLX_NAX_FLASH_WIDE=1` |
| 43b61156 | Quarter head-dim split verify attention, 2 to 5 rows | #594 (stacked on #593) | open | `MTPLX_NAX_FLASH_DSPLIT4=1` |
| 939133d4 | Dashboard: verify waterfall colors and labels | #590 | open | dashboard |
| dc02400a | Dashboard: live decode TPS line, 5-minute window | #598 | open | dashboard |
| e61bde6c | Unclosed reasoning block recovered as content when tools are declared (#583) | #588 | open | default |
| dd0cc3c1 | SSD cold tier: uint32 fingerprint mixer | #596 | open | matters with `MTPLX_SSD_INCREMENTAL_ENCODE=1` |
| 10cee526, 7eab5dd4 | Dashboard: refused requests shown as "507 memory" (with reason and retry hint) instead of MISS / 0 | #600 | open | dashboard; server copies `refusal_reason`, `retry_when` into the request row |
| 23468b07, 85d4a49a | Context-copy drafting for AR-only runtimes (models without an MTP head) | #599 | open | `MTPLX_CONTEXT_COPY_AR=1` (not used by Qwen3.8, which has an MTP head) |
| 5fcb2051 | Claude Code's `X-Claude-Code-Session-Id` names the session | #606 | open | default |
| abfea975, c2facca3, df4f4f06, b46cf089 | Tests follow the loaded MLX build (0.32.3): Flash-Next fp32 references, bf16 slot sum spelled out, pin test accepts the validated 0.32.3 build (wheel pin stays 0.32.2). The pre-release regex change 5520cfd3 was reverted (2c8b5277): the misread came from our build name, now `0.32.3.lse` | not filed | fork only | n/a |
| 0eae8d62, 69d17afc, d3a74424, dd28b920, 1da1470b | Changelog, lint, test fixes, dashboard bundle | n/a | n/a | n/a |

**Production flags that are obsolete on v2.12.2** (drop at the switch): `MTPLX_MTP_HISTORY_CACHE_ONLY` (default on upstream), `MTPLX_POSTCOMMIT_AFTER_RESPONSE` and `MTPLX_PERSIST_QUEUE_MAX_GB` (superseded; use `MTPLX_PERSISTENCE_MAX_PENDING_BYTES`).

**MLX:** production runs branch `prod` of [jvmenen/mlx](https://github.com/jvmenen/mlx), tag `prod-2026-10-09`: v0.32.3 plus the minq patch (adapted to upstream #4416), the qmm tile patch, `return_lse` and array-mask tests, built into `~/Dev/mlx-sdpa/pkg-combo-0323` (marker `0.32.3.lse`) with `MLX_SDPA_D256_MINQ=64`; one build for production and segmented KV ([mlx-0323-rebuild](2026-10-08-mlx-0323-rebuild.md)). Fallback: branch `prod-0.32.2`, tag `prod-2026-10-04`, build `pkg-combo`. Upstream forms: branches `sdpa-d256-short-query-v2` (on main with #4416, with array-mask tests) and `qmm-large-m-tile` (not filed; MLX wants an issue first and a description written by Jeroen).

