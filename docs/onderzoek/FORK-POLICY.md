# Fork policy (jvmenen/MTPLX and jvmenen/mlx)

Agreed 4 October 2026. Applies to both forks; MLX-specific notes at the end.

## Branches

| Kind | Name | Rule |
|---|---|---|
| Mirror | `main` | Exact copy of the maker's `main`. Never commit on it; only fast-forward from upstream. |
| Feature | `fix/...`, `perf/...`, `feat/...` | One change per branch, always branched from the maker's `main`, PR-ready (tests, changelog line, the maker's code style). This is where an upstream PR comes from. |
| Production | `prod` | The maker's latest release tag plus our changes that are not upstream (yet). The only branch the Bink config runs (`~/Dev/MTPLX-prod`). |
| Research | `onderzoek` | Reports, `VONDSTEN.md`, this policy and [CARRIED.md](CARRIED.md). No code that runs. |

## Rules

1. **Base `prod` on a release tag**, not on the maker's `main`: a release has passed the maker's gate, `main` can be mid-change.
2. **Every change starts as a feature branch** on the maker's `main` and reaches `prod` by cherry-pick, after its tests pass. A change that only exists on `prod` is lost at the next rebuild.
3. **Rebuild `prod` per upstream release**: a fresh branch from the new tag, then the commits listed in [CARRIED.md](CARRIED.md) that upstream has not taken over. Run the test suite and an end-to-end comparison against the current `prod` (one process per variant, alternating order, Docker and other large processes off). Switch the config only when nothing regresses; keep the previous `prod` as a fallback branch until the new one has run without problems.
4. **Tag every production deployment** (`prod-YYYY-MM-DD`) and name that tag in the config commit, so it is always clear which code served which run.
5. **Keep [CARRIED.md](CARRIED.md) current**: per carried commit the PR number, its state (open, merged, closed, not filed) and the switch in the launch command. A merged PR leaves the list at the next rebuild; a closed PR is either dropped or kept with a reason.
6. **Clean up** feature branches whose PR was merged or closed, only after Jeroen agrees.
7. **No production change without measurement**: a new switch in the launch command needs a measured gain on the production stack, and its result goes into `VONDSTEN.md`.

## MLX

Same structure on `jvmenen/mlx`: `main` mirrors `ml-explore/mlx`, feature branches per upstream PR (branched from MLX `main`, MLX style, `python/tests`), and `prod` is the release tag in use (v0.32.2) plus our patches, built into `~/Dev/mlx-sdpa/pkg-combo`. Upstream rules: open an issue first; Jeroen writes PR descriptions himself (MLX CONTRIBUTING).
