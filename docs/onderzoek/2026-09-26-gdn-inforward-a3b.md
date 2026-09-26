# In-forward GDN-grenzen voor A3B (vondst 21, 26 september 2026)

Tak `perf/a3b-inforward-boundaries` (commit `198e7194`, op `perf/batch-invariant-router` `1ce69614`), worktree `~/Dev/MTPLX-gdnhooks`, gepusht naar de fork en via een fast-forward in `perf/integratie`. Meetscripts en ruwe resultaten lokaal in `~/Dev/laya-nl/gdnhooks/` (niet in de fork).

## Uitkomst

Met de batch-invariante prefill (vondst 29) legt A3B de GDN-grenzen nu vast binnen de brede forward, in plaats van de laatste chunk bij elke grens af te knippen (de staartladder). Op het echte model is dat **volledig bitgelijk** aan de ladder en haalt het de extra forward weg: eerste-token-logprobs met completions-bank voor prompts vanaf 512 tokens van 464/448 naar 365/368 ms p50 (−21%), bijna gelijk aan zonder bank (349 ms, [batch-invariante-router](2026-09-26-batch-invariante-router.md)). De rest blijft binnen ongeveer 1%.

## Wat gebouwd is

- Nieuw bestand `mtplx/gdn_inforward_boundaries.py` met de twee hooks die de prefill-loops al zoeken (`_resolve_inforward_boundary_hooks`, `generation.py:5333`); tot nu toe had alleen `models/qwen4_exp.py` ze.
  - `boundary_capture_scope` (`:37`) en `take_boundary_captures` (`:49`): hetzelfde contract als `qwen4_exp.py`. Een positie telt alleen als elke recurrente cache-entry hem heeft vastgelegd.
  - `_BoundaryCaptureMixin.__call__` (`:91`): met een actieve capture draait elke GDN-laag zijn stock-`__call__` één keer per segment (`[0,p)`, `[p,S)`) op zijn eigen cache en bewaart tussendoor `cache.state`. Per GDN-laag is dat precies wat de ladder rekent, inclusief conv-staart en projecties; er is geen gekopieerde mlx-lm-code. Zonder actieve capture is het de stock-aanroep.
  - `install_gdn_inforward_boundaries` (`:132`): klassewissel voor `qwen3_5.GatedDeltaNet` (de klasse die A3B gebruikt) en `qwen3_next.Qwen3NextGatedDeltaNet`; de parameterboom blijft gelijk.
- `mtplx/runtime.py:872-880`: alleen geïnstalleerd binnen `if batch_invariant_prefill_enabled()`. Zonder de schakelaar van vondst 29 geen hooks en de ladder zoals voorheen, want dan verandert een bredere forward de MoE-routering. De bestaande knop `MTPLX_GDN_BOUNDARY_INFORWARD=0` zet de ladder terug. Het serverlog toont `'gdn_inforward_layers': 30`.
- `generation.py` is niet gewijzigd: de loops en de bankroute bestonden al.

## Tests (testmodel, Metal)

`tests/test_gdn_inforward_boundaries.py`, 11 tests. Een klein 4-bit Qwen3.5-MoE (3 GDN-lagen en 1 attentielaag, routed en shared experts, bf16) met de invariante prefill, door de echte koude streaming-loop en de warme suffix-loop:

- Koud (161 tokens): forwards `[96, 64, 1]` tegen 6 bij de ladder; `MTPLX_PREFILL_CHUNK_TRACE` 2 chunks tegen 5.
- Warm (60 tokens in de cache, suffix van 171): `[96, 74, 1]`.
- Beide bitgelijk aan de ladder: grensposities, elk blad van elke grens (states en hidden), logits, hidden en cache.
- Zonder de invariante prefill of met de knop op 0: de ladder. Zonder actieve capture gelijk aan stock; een gedeeltelijke capture wordt geen grens.

Gerelateerde bestaande tests groen (batch-invariante prefill, beide in-forward-testbestanden, retentie, trunkblokken, stabiel promptbegin). Ruff: nieuwe bestanden schoon, `runtime.py` gelijk aan de basis. Volledige suite niet gedraaid.

## Servermeting (gemeten)

M5 Pro 64 GB, MLX 0.32.2, Qwen3.6-35B-A3B Optimized-Balance, profiel turbo met de productie-argumenten, commit `198e7194`, 26 september 15:43-16:09. Verse server per variant, eigen SSD-cachemap, alle varianten met `MTPLX_BATCH_INVARIANT_PREFILL=1` en `MTPLX_COMPLETIONS_SESSION_BANK=1`. L = ladder (`MTPLX_GDN_BOUNDARY_INFORWARD=0`), H = hooks; volgorde L1, H1, L2, H2. `requests_completed` na elke variant 542 (geen verkeer van het platform), nergens gemiste captures in `/health`. Scripts `runall.zsh` en `analyse.py`, uitvoer in `analyse.txt`, `res/` en `logs/`.

**Bitgelijkheid H tegen L (beide rondes):**

- Eerste-token-logprobs: 240/240 oordelen, kansen en volledige top-20 gelijk.
- Scoreroute: 243/243 prompts bitgelijk.
- NLL op twee documenten: alle 5590 en 4618 posities gelijk.
- Gretige teksten 24/24, chat- en agentteksten (temperatuur 0) 12/12 gelijk.

Dezelfde gelijkheid geldt tussen L1 en L2 en tussen H1 en H2.

**Tijd en forwards:**

| | L1 | H1 | L2 | H2 |
|---|---|---|---|---|
| eerste-token p50 | 338,6 ms | 327,3 ms | 327,4 ms | 329,9 ms |
| eerste-token p90 | 468,0 ms | 365,5 ms | 449,7 ms | 367,7 ms |
| p50 prompts < 512 tokens (189) | 330,4 ms | 321,3 ms | 320,3 ms | 322,9 ms |
| p50 prompts ≥ 512 tokens (51) | 464,1 ms | 364,9 ms | 448,0 ms | 367,5 ms |
| prefill-forwards ≥ 512 (som, mediaan) | 102, 2 | 51, 1 | 102, 2 | 51, 1 |
| nauwkeurigheid | 0,725 | 0,725 | 0,725 | 0,725 |
| decode tok/s | 79,4 | 79,7 | 79,6 | 79,7 |
| chat: TTFT som | 46,47 s | 45,90 s | 46,67 s | 46,47 s |
| chat: TTFT korte beurt (mediaan) | 373 ms | 377 ms | 372 ms | 369 ms |
| agent: TTFT som | 50,28 s | 49,78 s | 50,39 s | 49,88 s |
| agent: TTFT korte beurt (mediaan) | 584 ms | 582 ms | 585 ms | 566 ms |
| prefill-forwards chat / agent (som) | 45 / 44 | 39 / 39 | 45 / 44 | 39 / 39 |

- Prompts vanaf 512 tokens met bank: −21% p50 (gemiddeld 456 naar 366 ms), één prefill-forward in plaats van twee. p90 daalt van ~459 naar ~367 ms.
- Chat en agent: 6 en 5 forwards minder per gesprek; TTFT-som −0,6 tot −1,2%, korte beurten gelijk binnen de spreiding.
- Decode en nauwkeurigheid ongewijzigd.

## Bijvondst: SwitchGLU-padding bij weinig experts (testmodel)

De padding van de invariante prefill voor `SwitchGLU` (`batch_invariant_prefill.py:147-168`, minimaal `4 × experts / top_k` tokens) is bij 8 experts en top-2 niet rij-invariant onder 33 tokens: MLX kiest daar een andere expertkernel. Bij 16 experts (top-2 en top-4), 32 experts (top-4) en 256 experts (A3B) wel. Voor A3B maakt dat niets uit; bij een model met weinig experts moet de ondergrens omhoog. Het testmodel gebruikt daarom 32 experts. Gemeten op synthetische tensors (M5 Pro, MLX 0.32.2).

## Open

- Niet getoetst met de opt-in-routes `MTPLX_FUSE_GDN_POST_CONV` en de gecompileerde target-prefix; die gaan alleen aan in de schedulermodus `mtp_batch` (`server/openai.py:874-886`), de productie draait `serial`.
- PR-besluit hangt aan dat van vondst 29: de hooks werken alleen samen met de invariante prefill.
