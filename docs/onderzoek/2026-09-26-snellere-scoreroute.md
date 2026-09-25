# Snellere scoreroute (26 september 2026)

Tak `feat/faster-prompt-scoring` (gebaseerd op `main` 1de2b1c0), twee commits, niet gepusht. Gebouwd door een Opus-agent met kleine testmodellen en synthetische tensors; het echte model is nog niet gebruikt.

## Commit A (dc9c0e3e): top-K zonder volledige log-softmax

- Per rij één f32-`logsumexp`. De woordenschat wordt verdeeld in blokken van 64 logits (248.320 = 3.880 × 64, dus zonder opvulling); de K blokken met de hoogste maxima blijven over en daarin wordt de exacte top-K gekozen. Daarna f32(logit) − lse alleen voor de K overblijvers en het doeltoken.
- Nieuwe hulpfuncties in `generation.py`: `_row_logsumexp_f32`, `_logprobs_at`, `_exact_top_k_ids`, `_argpartition_top_k`, `_sorted_top_k`, constante `_TOP_K_PREFILTER_BLOCK = 64`.
- Gewone `argpartition` op de ruwe bf16-logits was op de CPU trager dan het oude pad (505 tegen 440 ms), daarom het blokfilter.

| Apparaat | Oud | Nieuw (K = 1 tot 128) | Versnelling |
|---|---|---|---|
| GPU | 47 ms | 3,4-5,2 ms | ~9-14× |
| CPU, bf16 | 410-430 ms | 150-170 ms | ~2,7× |
| CPU, f16 | 410-430 ms | 45-63 ms | ~7-9× |

(één blok van 256 × 248.320, M5 Pro)

**Pariteit:** waarden bitgelijk aan het oude pad (maximaal verschil 0) voor top-K en doel-logprobs. Token-id's verschillen alleen bij exact gelijke ruwe logits; gelijke waarden worden nu op oplopend token-id geordend (voorheen onbepaald). De zorg over bijna-gelijke waarden in bf16 speelt niet: bf16 naar f32 is exact.

## Commit B (736d9480): de trunk in prefill-blokken

- De trunk rekent in blokken van de gewone prefill-grootte (live `prefill_chunk_tokens`, anders profiel 2048); lm_head en top-K blijven per 256 rijen, zodat het geheugen voor logits begrensd blijft.
- `mtp_patch._MTPLXTextModel.logits_from_post_norm` (door `__call__` zelf gebruikt, dus per definitie dezelfde lm_head); in `generation.py`: `_prompt_score_trunk_chunk_size`, `_post_norm_logits_head`, `_prompt_logit_slices`, parameter `trunk_chunk_size`. De server roept scoring aan binnen `prefill_chunk_size_override(state.args.prefill_chunk_tokens)`.
- Runtimes zonder die head (geen MTP-head, laguna, gemma, qwen3_5_mtp) houden de oude 256-rijen-route.
- **Terugzetten zonder release:** `MTPLX_PROMPT_SCORE_TRUNK_CHUNK=256` geeft bitgelijk de oude indeling.

**Waarom 256:** commit d44f9125 en de docstring: het geheugen voor logits begrenzen ("32k memory-balloon"). Geen gecompileerde buckets (prefill is eager), geen lengtegrens in de GDN-kernels, de A3B-batchgeometrie vraagt een prefill-blok van hooguit 2048, niets in `mistakes/` of `notes/`. Volgens commentaar in `generation.py` zijn smalle forwards traag (Flash-Next: 256 rijen ~750-842 tok/s tegen 2048 rijen ~1.720-1.763).

**Pariteit op het kleine hybride MoE-testmodel** (GDN + volledige attentie, 8 experts, f32 en bf16): bij lengtes 1, 255, 256, 257, 2047, 2048, 2049 en 8192 bitgelijk tussen trunk 2048 en trunk 256. Alleen een laatste forward van één rij wijkt af (0,04-0,14 op logits), zoals ook in het oude pad, en die raakt alleen de laatste positie die niet gescoord wordt.

## Eind-tot-eind op een modelplak

(2 A3B-lagen in 4-bit, 256 experts, lm_head op ware grootte, K = 20)

| Tokens | Oud | Alleen A | A+B |
|---|---|---|---|
| 300 | 97 ms | 47 ms | 43 ms |
| 2.048 | 636 ms | 289 ms | 252 ms |
| 8.192 | 2.578 ms | 1.191 ms | 1.029 ms |

B voegt op deze plak 9-14% toe; op het echte model met 40 lagen naar verwachting meer.

## Geheugen

- Trunk per blok: 256 rijen 0,9 GiB, 2048 rijen 4,7 GiB (gemeten op de plak; groeit niet met het aantal lagen); dat is wat een gewone prefill van 2048 rijen al gebruikt.
- Logits en top-K: van 1,9 GiB naar 0,27-0,33 GiB.
- Netto piek bij 8.192 tokens: ongeveer 2 GiB naar ongeveer 4,7 GiB. Op de echte kernels kan dat anders uitvallen.
- Risico: op een krappe geheugenplanning kan de session bank tijdens scoring worden geknepen; dan de knop op 256 zetten.

## Tests

`tests/test_prompt_scoring_topk.py` (25) en `tests/test_prompt_scoring_trunk_chunks.py` (40), plus één servertest; met de bestaande scoring-, echo-, MTP-patch- en vision-tests en `test_no_mlx_imports`: 724 geslaagd, 1 overgeslagen. Nulmetingfouten ongewijzigd. Ruff: geen nieuwe meldingen. CHANGELOG-regels onder Unreleased en één zin in `docs/benchmarking.md`.

## Echte toets

Script `verify_scoring_real.py` (in de lokale werkmap `laya-nl`): per verse server de 240 classifierprompts plus Nederlandse prompts van ~2k, ~4k en ~8k tokens; latency per lengteklasse, fouten, top-1-overeenkomst, maximaal logprob-verschil en geheugen uit `/health`. Drie runs: `main`, tak met `MTPLX_PROMPT_SCORE_TRUNK_CHUNK=256` (alleen A) en tak standaard (A+B). Uitkomst volgt in een apart rapport.

## Echte toets (26 september 2026, 23:35-23:43)

M5 Pro 64 GB, Qwen3.6-35B-A3B MTPLX Optimized-Balance, productie-instellingen, verse server per run, 240 classifierprompts (250-840 tokens) plus Nederlandse prompts van ~2k, ~4k en ~8k tokens, K = 20.

| | main | alleen A (`TRUNK_CHUNK=256`) | A+B (standaard) |
|---|---|---|---|
| p50 <512 tokens (n=188) | 536 ms | 473 ms | 382 ms |
| p50 512-1023 tokens (n=52) | 630 ms | 593 ms | 420 ms |
| ~2k tokens | 2.130 ms | 2.041 ms | 1.416 ms |
| ~4k tokens | 4.284 ms | 4.060 ms | 2.530 ms |
| ~8k tokens | 8.844 ms | 8.367 ms | 4.742 ms |
| Top-1 laatste positie gelijk aan main | | **243/243** | **213/243** |
| Top-1 alle posities | | 121.263/121.263 | 114.292/121.263 |
| Max. logprob-verschil | | **0** | **19,85** (prompt 32, positie 161) |
| Fouten | 0 | 0 | 0 |

**Conclusie:** commit A is bitgelijk en ~10% sneller: klaar voor een PR. Commit B is 1,25× sneller bovenop A (8k: 8,4 naar 4,7 s), maar geeft op het echte model **verkeerde uitkomsten**, ook binnen het eerste blok van 256 rijen; op het kleine testmodel was hij bitgelijk. Oorzaak wordt onderzocht; B wordt niet ingediend zolang de uitkomst niet 243/243 en afrondingsniveau is.

`session_bank.effective_max_bytes` was bij alle drie verse servers 17,39 GiB (voor en na), dus het plafond staat op een verse server ruim; zie vondst 8 over het wegzakken na zware beurten.

## Oorzaak van de afwijking bij commit B, en besluit (27 september 2026)

Commit B bevatte geen codefout. Op Qwen3.6-35B-A3B verandert de uitkomst zodra het model de prompt in blokken van een andere grootte verwerkt, en dat effect is veel groter dan afronding.

- **Patroon:** elke prompt langer dan 256 tokens wijkt af vanaf ~positie 4; prompts van ≤256 tokens (3 stuks) zijn bitgelijk. Rij i hangt dus af van hoeveel rijen er in dezelfde forward zitten, ook binnen de eerste 256.
- **Reproductie op de server** (prompt 32, 507 tokens, blokgroottes 1/128/256/320/512/2048): 256 tegen 320 wijkt af vanaf positie 1 (max 19,8, gemiddeld 0,365 nats); 128 tegen 256 max 19,0; token voor token (1) tegen 256 max 10,5; 320 tegen 2048 en 512 tegen 2048 gelijk tot rij 320 (zelfde eerste forward).
- **Per laag gemeten:** de attentie-uitvoer van laag 0 en de MoE-invoer zijn bitgelijk, maar de router-logits verschillen 1-2 bf16-stappen (tot 0,125): de matmul geeft net andere uitkomsten afhankelijk van het aantal rijen. Het verschil tussen de 8e en 9e gekozen expert is zelf vaak maar 1-2 bf16-stappen (mediaan 0,03-0,06, vaak exact gelijk). Daardoor kiezen al in laag 0 12 van de 128 rijen een andere expertset, oplopend tot 40-77 van de 128 per laag in lagen 14-39. Zo wordt afrondingsruis een echt verschil in uitkomst.
- **Ruis op 25 prompts (9.909 posities):**

| Vergelijking | Top-1 gelijk | Gem. verschil | Max. verschil |
|---|---|---|---|
| 256 tegen 128 | 93,5% | 0,20 nats | 7,45 |
| 256 tegen 2048 | 93,5% | 0,21 nats | 9,52 |
| 128 tegen 2048 | 93,0% | 0,22 nats | 9,52 |

  De gemiddelde logprob per prompt verschuift nauwelijks (−4,72 tegen −4,69): ruis, geen bias.
- **Uitgesloten:** verkeerde norm-variant, de losse lm_head, de blokgrootte-override, cache-offsets op het eerste blok. Het kleine testmodel was bitgelijk omdat zijn router zulke bijna-gelijke keuzes niet heeft.

**Besluit:** commit B teruggedraaid (`8a72563b`); de tak bevat nu alleen commit A. Geen bredere indeling kan 243/243 op afrondingsniveau halen tegen de 256-referentie, en eerdere scores zijn allemaal met 256 berekend.

**Hernieuwde toets (verse server, teruggedraaide tak):** tegen `main` 243/243 top-1 op de laatste positie, 121.263/121.263 over alle posities, maximaal verschil 0. Latency p50: <512 tokens 432 ms (main 536), 512-1023 521 ms (630), ~2k 1,82 s (2,13), ~4k 3,66 s (4,28), ~8k 7,55 s (8,84); mediane versnelling 1,21× (de eerdere run gaf 1,10×, dus deels variatie tussen runs). 539 tests geslaagd, 1 overgeslagen.

**Betekenis breder dan deze tak:** dezelfde ruis zit al in `main`. Scores op dit model hangen ~0,2 nats per token af van de blokindeling; ook token-voor-token verwerken wijkt tot 10,5 nats af van 256-rijen. Dat verklaart de kansverschillen tot 0,2 tussen prompt-scoring en eerste-token-logprobs bij onze classifier (andere rekenindeling). Voor KL-metingen op A3B is dit de ondergrens van de ruis, tenzij de prompt altijd in dezelfde indeling wordt verwerkt. De 1,25× van commit B kan alleen als opt-in met standaard 256; scores met een andere indeling zijn dan niet vergelijkbaar met eerdere.
