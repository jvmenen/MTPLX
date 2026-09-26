# Batch-invariante prefill voor A3B (vondst 29, 26 september 2026)

Tak `perf/batch-invariant-router` (vanaf lokale `perf/integratie` 54c4ca5f), worktree `~/Dev/MTPLX-router`, gepusht naar de fork. Drie commits: `35c161f7` (schakelaar), `6a151a33` (commit B terug, alleen met schakelaar), `2c1e653f` (geen staartgrenzen onder de bankdrempel, alleen met schakelaar). Meetscripts en ruwe resultaten lokaal in `~/Dev/laya-nl/router-invariant/` (niet in de fork).

## Tussenstand (gepauzeerd)

Het werk is op 26 september rond 09:05 gepauzeerd (laptop mee). De meetreeks is halverwege gestopt; de server op poort 8000 is uit.

### Diagnose (gemeten, echt model in een eigen proces, geen server actief)

M5 Pro 64 GB, MLX 0.32.2, Qwen3.6-35B-A3B Optimized-Balance. Script `diag.py`: dezelfde tokens in blokken van 1 tot 2048 rijen, per laag vergeleken (laaginvoer, attentie/GDN-uitvoer, MoE-invoer, router-logits, top-8, routed experts, shared expert). Prompt 32 (507 tokens) en een neutrale Nederlandse tekst van 2600 tokens.

**De router is niet de enige bron.** Waar de rij-afhankelijkheid ontstaat, met identieke invoer (stock, zonder schakelaar):

| Blokgrootte tegen 2048 | Eerste afwijking | Bron |
|---|---|---|
| 256, 320 (prompt 507 tokens) | laag 0 | router-matmul (alle rijen) en shared expert (alle rijen); attentie gelijk |
| 1, 17, 64, 128 | laag 0 | attentie/GDN-projecties (alle rijen) |
| 256, 64 (tekst 2600 tokens) | laag 0 | attentie: het blok van 2048 rijen gebruikt een andere SDPA-kernel |
| vaste routering (router-logits van de referentie opgelegd) | laag 0 | geen routewissels meer, maar de hidden states blijven verschillen (shared expert, attentie) |

Top-8-wissels per laag bij 256 tegen 2048 (prompt 32, 507 rijen): laag 0: 34, laag 3: 88, laag 20: 239, laag 39: 175.

**Oorzaak in MLX 0.32.2 (bron gelezen: `quantized.cpp`, `scaled_dot_product_attention.cpp`):**

1. `quantized_matmul` met 2D-gewichten gebruikt `qmm_splitk` zolang er minder dan ~512 tegels van 32×32 zijn; het aantal delen volgt `512 / (m_tiles × n_tiles)`. Smalle uitgangen (router N=256, shared expert N=512, GDN a/b N=32, k/v N=512, shared-expert-gate N=1) wisselen daardoor per rijtal van afronding, tot ruim 1024 rijen. Gemeten op synthetische tensors met de echte vormen: router pas gelijk aan 2048 rijen vanaf 1025 rijen, shared expert vanaf 513, `o_proj`/`out_proj` vanaf 129, `in_proj_qkv` vanaf 33.
2. Routed experts: `gather_qmm_rhs` (getegeld) pas bij minstens 4 token-expertrijen per expert (128 tokens); daaronder `gather_qmv`, andere afronding.
3. SDPA met head dim 256: de fused kernel pas vanaf 1024 queryrijen, van 9 tot 1023 een niet-fused route (matmul, softmax, matmul), onder 9 de vectorkernel.
4. GDN (gated delta, conv) bleek in alle metingen rij-invariant.

### Wat gebouwd is

`mtplx/batch_invariant_prefill.py` (nieuw), schakelaar `MTPLX_BATCH_INVARIANT_PREFILL=1`, standaard uit, gelezen bij het laden (`mtplx/runtime.py:866-873`). Alleen in de prefill-fase (`attention_phase("prefill")`); decode en verify houden de stock-kernels.

- `split_k_free_quantized_matmul` en `BatchInvariantQuantizedLinear` (`batch_invariant_prefill.py:93-144`): de matmul als twee batches van minstens 33 rijen met uitgezonden gewichten. MLX slaat split-K over voor batches en gebruikt dan één NAX-tegelvorm; aanvullen met nullen tot 66 rijen.
- `BatchInvariantSwitchGLU` (`:147-168`): vult aan tot 128 tokens (4 per expert), zodat altijd de getegelde gesorteerde kernel draait.
- `batch_invariant_sdpa` (`:171-233`): vervangt `mlx_lm.models.qwen3_next.scaled_dot_product_attention` door altijd de fused causale kernel (`force_fused=True`), met nulrijen vóór de query als er minder dan 9 zijn.
- `install_batch_invariant_prefill` (`:240`): alleen klassewissels en één functiehaak; de parameterboom blijft gelijk.
- Commit B terug (`generation.py:8315` `_prompt_score_trunk_chunk_size`, `:8345` `_prompt_logit_slices`, `mtp_patch.py:858`, `server/openai.py:25626`): scoring rekent de trunk in de prefill-blokgrootte, lm_head per 256 rijen, maar alleen standaard als de schakelaar geïnstalleerd is; anders 256 zoals nu.
- Vondst 40 (`generation.py:5045` `_cold_prefill_tail_interval`, aangeroepen bij de twee koude prefill-routes): met de schakelaar geen staartgrenzen voor prompts onder de blokherstel-drempel (512).
- Tests: `tests/test_batch_invariant_prefill.py` (36, op de GPU), aangepast `tests/test_prompt_scoring_trunk_chunks.py` en één servertest. Ruff: geen nieuwe meldingen (`generation.py`, `mtp_patch.py`, `openai.py`, `runtime.py` gelijk aan de basis). Volledige suite nog niet gedraaid.

### Toets van de schakelaar in het eigen proces (gemeten)

Met de schakelaar aan (`diag3.py`, `diag4.log`): hidden states van alle 40 lagen **bitgelijk** tussen blokken van 17, 64, 100, 256, 320, 331 en 1000 rijen en blokken van 2048 rijen, op beide teksten; ook de chat-staartsplitsing `[0,331],[331,395]` en `[0,200],[200,394],[394,395]` tegen één forward. Alleen token voor token (blok 1 of 5) wijkt af: de eerste 8 posities hebben te weinig sleutels om de query aan te vullen, waarna het doorwerkt. Zonder schakelaar wijkt de staartsplitsing op 64 rijen af (716 top-8-wissels over de lagen).

Kosten per kernel (synthetisch, M5 Pro): de split-K-vrije matmul is bij 256 en 2048 rijen even snel als stock of sneller (router 2048 rijen 0,34 tegen 0,36 ms, `in_proj_qkv` 2,69 tegen 2,71 ms); fused SDPA over blokken van 256 rijen 3,1 tegen 4,4 ms. Duur zijn alleen smalle forwards: een forward van 1 rij wordt aangevuld tot 66 rijen (linears), 128 tokens (experts) en 9 queries.

### Servermetingen (gemeten, verse server per variant, productie-argumenten, SSD-cache in een eigen map)

Varianten A (uit), B (aan, trunk standaard = prefill-blok), C (aan, `MTPLX_PROMPT_SCORE_TRUNK_CHUNK=256`). Geen verkeer van het platform gezien (`requests_completed` na de run gelijk aan wat de scripts stuurden: 773, 773, 244).

**Scoring (`verify_scoring_real.py`, 240 prompts plus 2k/4k/8k):**

| | A uit | B aan (trunk groot) | C aan (trunk 256) |
|---|---|---|---|
| p50 <512 tokens | 412 ms | 323 ms | 406 ms |
| p50 512-1023 | 501 ms | 369 ms | 533 ms |
| ~2k | 1,68 s | 1,03 s | 1,65 s |
| ~4k | 3,34 s | 2,03 s | 3,33 s |
| ~8k | 6,87 s | 4,05 s | 6,84 s |

- B tegen A: 1,27× sneller (p50), 8k 1,70×. Uitkomst wijkt af zoals verwacht (de schakelaar rondt anders af): 215/243 top-1 op de laatste positie.
- C tegen A: even snel (1,00×), dus de schakelaar zelf kost bij scoring niets.
- **B tegen C: nog niet bitgelijk**: 241/243, 121.240/121.263 posities. Twee restbronnen:
  1. Bij de classifierprompts wijken alleen de laatste rijen af die in C in een staartblok van minder dan ~17 rijen vallen (bijv. posities 256-266 van een prompt van 267 tokens). Oorzaak (uit de code): commit B roept de lm_head aan buiten `attention_phase("prefill")` (`generation.py:8383`), dus een staart van weinig rijen loopt via de stock-matmul. Oplossing: de lm_head-aanroep binnen de prefill-fase zetten.
  2. De synthetische prompts van 2k/4k/8k wijken vanaf positie 2 bijna overal af. Niet verklaard; in het eigen proces waren 2048 en 256 rijen op 2600 tokens wel bitgelijk. Te onderzoeken: welke prefill-blokgrootte de server echt gebruikt (profiel turbo), en of een andere route (bijv. gecompileerde A3B-prefix, `generation.py:11485`) de klassewissels omzeilt.

**Kwaliteit (A uit tegen B aan):**

- NLL op neutrale tekst (twee Nederlandse documenten van 5590 en 4618 tokens, scoreroute): 1,6382 tegen 1,6384 en 1,8008 tegen 1,8035 nats per token. Verschil per token gemiddeld 0,05; de gemiddelde NLL verschuift nauwelijks. Oordeel: gelijkwaardig.
- Classificatie (240 berichten, teacher-labels): scoreroute nauwkeurigheid 0,717 tegen 0,692, 227/240 dezelfde oordelen; eerste-token-logprobs 0,717 tegen 0,700, 225/240 gelijk. Het verschil (4 tot 6 berichten) past bij de ruis die elke andere rekenindeling geeft (eerder 12/240 tussen routes); een tweede ronde (A2/B2) was gepland maar is niet gedraaid.
- Gretige teksten (24 prompts, temperatuur 0): 6/24 identiek, de rest wijkt af na 11 tot 335 tekens. Beoordeling (Claude Opus 5.5) nog niet gedaan; teksten staan in `res/quality-A-uit.json` en `res/quality-B-aan.json`.

**Kosten van de schakelaar buiten scoring (`bench.py`):**

| | A uit | B aan | verschil |
|---|---|---|---|
| decode tok/s | 81,0 | 82,7 | +2% (ruis; decode-pad onveranderd) |
| TTFT korte prompt | 212 ms | 280 ms | +32% |
| eerste-token-logprobs p50 | 323 ms | 373 ms | +15% |
| scoreroute p50 | 428 ms | 333 ms | −22% |
| chat tot 60K: TTFT som | 46,8 s | 47,8 s | +2% |

De TTFT-kosten komen (verwacht, niet apart gemeten) van de losse forward van het laatste prompttoken (`generation.py:7648` in de basis): die heeft 1 rij en wordt in de prefill-fase aangevuld tot 66/128/9 rijen. Eén meting per variant; een herhaling ontbreekt.

### Nog niet gemeten

- Vondst 40 met de schakelaar (varianten D en E: completions-bank met en zonder staartgrenzen): gestart, afgebroken.
- Herhaling A2/B2 voor spreiding.
- Vondst 21 (in-forward grenzen voor A3B): niet gebouwd. Met de schakelaar is de ladder uitkomstneutraal, dus alleen de extra forward kost nog tijd; een haak in de GDN-laag zoals `qwen4_exp.py:2560-2680` is meer dan een kleine wijziging.

### Hervatten: volgende stappen

1. Lees dit rapport en `~/Dev/laya-nl/router-invariant/` (scripts `diag*.py`, `serve.zsh`, `runall.zsh`, `quality.py`, `qcompare.py`, `synth*.py`, `sdpa_test.py`; resultaten in `res/`, serverlogs in `logs/`). Worktree `~/Dev/MTPLX-router`, tak `perf/batch-invariant-router` (2c1e653f), schoon.
2. Losse forward van 1 rij goedkoop maken: forwards van weinig rijen die altijd los lopen (het laatste prompttoken) niet aanvullen, of kleinere minimums meten. Doel: TTFT en eerste-token-logprobs terug naar het niveau van A zonder de invariantie voor blokken ≥ 9 rijen te verliezen. Opnieuw `diag3.py` draaien (in-process, alleen zonder draaiende server).
3. lm_head in commit B binnen `attention_phase("prefill")` aanroepen (`generation.py:8383`), test toevoegen.
4. Synthetische prompts B tegen C uitzoeken (zie restbron 2): in het eigen proces `score_prompt_logprobs` met `trunk_chunk_size` 256 tegen 2048 op de synthetische tekst van `verify_scoring_real.py`, en in de serverlog de prefill-blokgrootte nagaan.
5. Servermetingen opnieuw: `runall.zsh` (varianten A, B, C, D, E, F, A2, B2; ~8 min per variant, alleen met toestemming van Jeroen; poort 8000 na afloop vrij). Doel: B tegen C 243/243 en maximaal verschil 0; vondst 40 (D tegen E: gelijke oordelen, tijdwinst); spreiding A/A2.
6. Kwaliteitsoordeel over de gretige teksten (Claude Opus 5.5) en de NLL/classificatie van de tweede ronde.
7. Daarna volledige suite (`MTPLX_CONFIG=/nonexistent`, vergelijken met de nulmeting), ruff, commit en `git push fork perf/batch-invariant-router`; docs bijwerken (VONDSTEN 29, 40, 21 en de README-tabellen).
