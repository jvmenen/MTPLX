# Vondsten Qwen3.8-27B

Lopende lijst voor het dichte model Qwen3.8-27B (Youssofal Optimized-Speed: 4-bit g32 body, lm_head Q8/g64, MTP-head bf16). Gestart 30 september 2026. Zelfde spelregels als [VONDSTEN.md](VONDSTEN.md); rapport: [2026-09-30-qwen38](2026-09-30-qwen38.md).

Meetopzet (lokaal, `~/Dev/laya-nl/qwen38/`): `run38.zsh <variant>` start per variant een verse server op poort 8000 met de eindtest-3-argumenten van Qwen3.8 en de Bink-productie-env, op `perf/definitief` (a5df61d0). Daarna `bench38.py` (4 prompts met reasoning aan, serversampling, 1.200 tokens, 2 rondes; plus koude prefill van ~7K en ~22K tokens) en 2x `lus2.py` (echte agenttaak). Hardware: M5 Pro 64 GB, profiel turbo, depth 3.

## Nulmeting (30 sep)

| Meting | Waarde |
|---|---|
| `mtplx tune` (codesuite, thinking uit, 512 tokens): AR / D1 / D2 / D3 | 13,1 / 29,9 / 40,5 / 48,2 tok/s |
| tune D3: acceptatie per positie; verify-forward + eval per ronde | 0,98 / 0,91 / 0,82; 72 + 11 ms |
| bench38 denken (reasoning aan, temp 1.0): mediaan / totaal | 34,1 / 33,9 tok/s; acceptatie 0,88 / 0,71 / 0,57; 3,18 tokens per verify |
| Koude prefill 7.182 / 21.626 tokens | 17,6 / 55,2 s (~400 tok/s) |
| lus2 agenttaak (2x) | groen, 81 en 57 s |

## Open

Ruis: `frspec` (FR-Spec niet actief) kwam op 32,4 tok/s tegen basis 34,1; verschillen onder ~5% gelden als niet aangetoond. Variant `basis2` meet de herhaalbaarheid.


| # | Vondst | Bron | Volgende stap |
|---|---|---|---|
| Q2 | FR-Spec (gesnoeide draft-head met de meegeleverde 64K-Qwen3.8-vocab) staat alleen standaard aan voor Flash-Next. Op het 27B-pack wordt de head zonder `MTPLX_FRSPEC_LEGACY=1` wel geïnstalleerd maar niet gebruikt: de koppeling `_mtplx_bind_draft_lm_head` bestaat alleen op de native MTP-route van Flash-Next (`frspec_draft.py:236-248`). Met de legacy-lane (`MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k MTPLX_FRSPEC_LEGACY=1`, draft-temp 0.6): **36,4 tok/s tegen 34,1 en 33,6 voor twee basisruns (+7 à 8%)**, acceptatie gelijk (0,86 / 0,71 / 0,58), prefill gelijk, agenttaak 2x groen. Prescatter en sampled chain weigeren op deze route (`DraftK20PrescatterIneligible`) | code-analyse; varianten `frspec`, `frchain`, `frlegacy`, `basis2` (30 sep) | Aanbevolen instelling voor 3.8. Nog doen: kwaliteit op niet-code-tekst (de vocab is op code gerangschikt; uitvoer blijft exact, alleen acceptatie kan dalen) en de sampled chain op de legacy-route (agent, zie Q6) |
| Q3 | Prefill ~400 tok/s op 27B dicht: rekengebonden (~20 TFLOPS effectief, vergelijkbaar met Qwen3.6). Winst moet komen uit minder koude prefill (session bank, SSD), niet uit de kernel. Chunk 2048: gelijk aan 4096 (18,4 / 57,0 s tegen 17,6 / 55,2 s; decode 34,2 tok/s). Chunk 8192: prefill 41,5 / 136,5 s en daarna alles traag; de Mac ging swappen (9,4 GB swap, 8,5 mln swapouts) en bleef dat de hele nacht doen. 4096 is dus de juiste waarde; 8192 niet gebruiken op 64 GB | bench38, varianten `ch2k` en `ch8k` (30 sep) | Bankgedrag bij lange agentsessies meten (hoe vaak koude prefill) |
| Q4 | KV 64 KB per token (16 full-attention-lagen) geeft geheugendruk en banktrimming bij ~65K context. `--paged-kv-quantization q8`: decode 30,8 tok/s (−9% tegen basis), prefill gelijk; de compiled-verify-paden vallen af zoals de help aangeeft. Het geheugeneffect is in deze werklast niet gemeten | eerdere modeltest (20 sep), variant `kvq8` (30 sep) | Alleen zinvol als lange sessies door geheugendruk koud prefillen; eerst dat bankgedrag meten |

| Q7 | Decode en acceptatie zakken bij lange context: in de QBR-taak (30 sep) 35 tok/s bij 20K, 22 tok/s bij 60K, 19,7 tok/s bij 75K; acceptatie van 0,85 / 0,67 / 0,58 naar 0,72 / 0,48 / 0,31. MTPLX heeft een dieptebeleid voor lange context (`long_context_mtp_depth_policy`, drempel 98.304), maar dat staat voor dit model uit (`reason: disabled`) | request-log QBR-taak 30 sep | Meten of D2 boven ~50K sneller is dan D3 (vergelijkbare verwachte tokens per ronde, kleiner verify-venster en één draftstap minder); zo ja, drempel en beleid voorstellen. Vraagt een server op poort 8000 zonder Bink-verkeer |
| Q9 | QBR-taak op 3.8 duurde ~60 min: de dispatcher brak af op de uurgrens (`dispatch_timeout_seconds` 3600) nadat het verslag al af was, en de automatische tweede poging begon het te overschrijven | QBR-taak 30 sep | Platformkwestie: concepttaak Daan (uurgrens per backend of taak, geen blinde herstart) |

## Afgehandeld

| # | Vondst | Uitkomst |
|---|---|---|
| Q1 | Onze meetargumenten zetten `--draft-temperature 1.0`; de maker kalibreerde 0.6 als familiestandaard (+9% in zijn meting, commit b8d6e785) | 30 sep: toegepast (resolved 0.6), geen verschil op onze werklast: 34,15 tegen 34,12 tok/s, acceptatie 0,87 / 0,72 / 0,58 tegen 0,88 / 0,71 / 0,57. Voor Bink maakt het niet uit; bij `mtplx serve` zonder vlag geldt 0.6 al |
| Q6 | ~11 ms eval per verify-ronde naast de forward | 30 sep (Opus-agent, rapport `~/Dev/laya-nl/qwen38/verify-analyse.md`): geen extra werk maar de staart van dezelfde GPU-verify (`generation.py:2345`); per ronde 87,7 ms, waarvan 73,5 ms verify en 11,6 ms drafts op de GPU, hooguit 1,5 à 2 ms host met stille GPU. M=4-matmuls halen 250 à 280 GB/s, dicht bij de bandbreedtegrens. Bit-identiek valt hooguit ~2% te winnen. Gebouwd op `perf/qwen38` (befe8933): `MTPLX_STOCK_SAMPLED_DRAFT_CHAIN=1`, zelfde tokens, +0,5 à 1% (binnen ruis), standaard uit. `MLX_MAX_MB_PER_BUFFER=1000`: +1,7%, binnen ruis. Conclusie: winst alleen nog uit minder bytes lezen (FR-Spec, kleinere draft-lm_head) |
| Q5 | Verify-core-varianten | 30 sep: `linear-gdn-from-conv` 33,6 tok/s (gelijk aan basis); `linear-gdn-from-conv-stream-skip0` kapot op dit model (geen acceptatiecijfers in `latest`, 22,8 tok/s). Huidige keuze `-conv-tape` blijft |
| Q8 | "Haperingen" in de stream (`producer_gaps_over_200ms` 129 à 258 per lang verzoek) | 30 sep: geen stall. De maat is de tijd tussen uitgestuurde tokens; met MTP komen per ronde meerdere tokens tegelijk, dus de gaten zijn rondetijden. Bij 75K context is een ronde 130 à 145 ms (p95), 2 à 6% van de rondes haalt 200 ms, pieken tot ~400 ms: de attention over lange context, geen verify- of cachestall |
| Q10 | Denkspiraal bij ~65K context (modeltest 20 sep: denkblok dat niet eindigde) | 30 sep: QBR-taak rondt af. `MTPLX_THINKING_BUDGET=6144` greep twee keer in (09:50 en 09:57, bij 53K en 62K context: 6.144 denktokens, geforceerd gesloten, daarna gewoon verder). Kwaliteit van het verslag benadert de Opus-referentie (sprekers voorzichtig, procespunten compleet), in 60 minuten |
| Q0 | Diepte 4 of meer | Niet opnieuw getest: de maker mat D6 −27% tegen D3 en D4 liet de daemon stilletjes sterven (`backends/descriptors.py:536-555`); cap blijft 3 |
