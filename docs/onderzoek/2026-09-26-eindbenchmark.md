# Eindbenchmark: 2.11.3 tegen de integratietak (26 september 2026)

## Samenvatting

**Update (avond):** [eindtest 2](#eindtest-2-met-invariante-prefill-en-a3b-hooks-26-september-avond) meet de tak mét invariante prefill en A3B-hooks (49c4a6ee, `MTPLX_BATCH_INVARIANT_PREFILL=1`): scoreroute −36%, eerste-token-logprobs −39%, agentgesprek −66%, en een nieuwe aanbeveling voor de overstap. Hieronder eerst eindtest 1 (ochtend).

De integratietak met al onze fixes is op het echte model op elk punt dat telt sneller dan 2.11.3, de versie die het Bink-platform nu draait, zonder kwaliteitsverlies bij de classificatie:

- **Agentgesprekken via `/v1/messages` (zoals de Agent SDK):** een heel gesprek van 12 beurten tot ~60K tokens duurt 58 s in plaats van 164 s (−64%). De wachttijd tot het eerste token halveert: lange beurten 9,0 in plaats van 18,5 s, korte 0,52 in plaats van 1,19 s. Dit komt vrijwel helemaal van de fix voor de onterechte herhaalpoging na een toolaanroep; 2.12.0 zelf lost dat niet op.
- **Classificatie zoals het platform die nu doet (scoreroute):** 443 in plaats van 533 ms per bericht (−17%), met exact dezelfde oordelen en kansen.
- **Classificatie via de nieuwe route (eerste-token-logprobs):** 336 ms per bericht (−37% tegen de huidige route), nauwkeurigheid gelijk (0,717). Die winst is er alleen als de completions-bank (`MTPLX_COMPLETIONS_SESSION_BANK=1`) uit staat; met de bank aan kost dezelfde route 481 ms. De voorgestelde eindconfig moet daarom anders (zie Aanbeveling).
- **Decode:** 80,0 in plaats van 75,3 tok/s (+6%); dat komt van upstream (2.12.0), niet van onze fixes.
- **Lange chat:** korte vervolgbeurten 11 tot 16% sneller; lange beurten en de decode in de chat blijven binnen ±2%.
- **Geheugen:** piek 1,6 GiB lager na het agentgesprek, footprint na de classificatie 3,3 GiB lager.

Alle getallen hieronder zijn gemeten, tenzij er "geschat" bij staat.

## Opzet

- **Hardware:** Apple M5 Pro (Mac17,9), 64 GB, macOS 26.6.2, fanmodus default. Geen thermische of prestatiewaarschuwing in `pmset -g therm` tijdens de runs.
- **Model:** Qwen3.6-35B-A3B MTPLX-Optimized-Balance, productie-argumenten van het platform (`mtplx-originele-args.txt`: profiel turbo, diepte 2, `--preserve-thinking auto`, toolmodus hybrid, SSD-cache aan).
- **Versies en configuratie:**

| Naam | Code | Configuratie |
|---|---|---|
| oud | 2.11.3 (Homebrew-venv, zoals het platform) | productie-argumenten |
| nieuw | `perf/integratie` 54c4ca5f (2.12.0 + alle fixes) | productie-argumenten + `--ram-session-prefix-min-match-tokens 128` + env `MTPLX_COMPLETIONS_SESSION_BANK=1` (voorgestelde eindconfig) |
| main | schone `origin/main` 1de2b1c0 (2.12.0), tijdelijke worktree | productie-argumenten; scheidt upstream van onze fixes |
| kaal | `perf/integratie` 54c4ca5f | productie-argumenten, zonder de twee schakelaars; extra runs na de vondst bij eerste-token-logprobs |

- **Inhoud van nieuw:** eerste-token-logprobs, top-K in de scoreroute, chat-encode-memo, afwijsreden in `/health`, testisolatie, prefix-hergebruik onder 512 tokens met de prefix-helpers, bankplafond-knop (uit), `sessionbank_put_s`, geen blank retries bij temperatuur 0, Gemma-4-controle zonder `get_vocab()`, messages-ttft (geen onterechte herhaalpoging), scoped chat per beurt encoderen. Zie de takkentabel in de [README](README.md).
- **Instrument:** de eindbench (`~/Dev/laya-nl/eindbench`, lokaal): decode (3 × 512 tokens, gretig), classificatie van 240 berichten via de scoreroute van het platform en via eerste-token-logprobs, chatgesprek tot ~60K tokens (gretig, 12 beurten) en agentgesprek via `/v1/messages` tot ~60K tokens (12 beurten, geen seed, zoals de SDK). Verse server en eigen SSD-cachemap per run. Analyse met `eindanalyse.py` (nieuw, zelfde map).
- **Volgorde (om en om, 05:08 tot 06:27):** oud-1, nieuw-1, main-1 (agent op serverstandaard-temperatuur 0,6), oud-2, nieuw-2, main-2 (agent op temperatuur 0), oud-3, nieuw-3 (standaard), oud-4, nieuw-4 (temperatuur 0), daarna kaal-1 en kaal-2 (temperatuur 0). Decode, classificatie en chat zijn in elke run gelijk (gretig), dus daar is de mediaan over 4 runs per versie genomen; agent per temperatuur over 2 runs. Spreiding: halve afstand tussen minimum en maximum, als % van de mediaan.
- **Vreemde verzoeken:** `requests_completed` uit `/health` na elke werklast. Alle 12 runs kloppen: nieuw en kaal 0, 2, 5, 248, 491, 503, 515 (+1 bij nieuw-1: één redeneerherstel); oud en main 248 na de classificatie (eerste-token-logprobs bestaat daar niet) en daarna 24 tot 26 voor 12 agentverzoeken, precies het aantal interne herhaalpogingen. Het platform heeft niet meegemeten; geen run is herhaald.
- **Suite vooraf** op de integratietak (`MTPLX_CONFIG=/nonexistent`): 9.582 geslaagd, 67 overgeslagen, 0 mislukt. Ook de bekende flaky test slaagde.

## Resultaten

Mediaan over de runs, spreiding tussen haakjes. Verschil is nieuw tegen oud.

### Decode (korte prompt, 512 tokens, gretig)

| | oud | nieuw | verschil | main |
|---|---|---|---|---|
| tok/s | 75,3 (±0,6%) | 80,0 (±0,2%) | +6,2% | 79,7 |
| TTFT | 0,216 s | 0,216 s | 0% | 0,217 s |

### Classificatie, 240 berichten

| Route | p50 | p90 | maximum | totaal 240 | nauwkeurigheid |
|---|---|---|---|---|---|
| oud, scoreroute | 533 ms (±1,6%) | 626 ms | 849 ms | 126,8 s | 0,717 |
| main, scoreroute | 535 ms | 625 ms | 851 ms | 127,1 s | 0,717 |
| nieuw, scoreroute | 443 ms (±0,5%) | 535 ms | 740 ms | 107,3 s | 0,717 |
| nieuw, eerste-token-logprobs | 481 ms (±0,2%) | 503 ms | 599 ms | 108,7 s | 0,708 |
| kaal, eerste-token-logprobs | 336 ms (±1,6%) | 356 ms | 434 ms | 77,1 s | 0,717 |

- **Oude route naar oude route** (scoreroute, oud tegen nieuw): p50 −16,9%, p90 −14,5%, totaal −15,4%.
- **Oude route naar beste nieuwe route:** met de voorgestelde eindconfig is de scoreroute zelf de snelste (−17%). Zonder de completions-bank is eerste-token-logprobs de beste: p50 −37%, p90 −43%, totaal −39% (126,8 naar 77,1 s).

### Chat tot ~60K tokens (gretig, `max_tokens` 128)

| | oud | nieuw | verschil | main | kaal |
|---|---|---|---|---|---|
| TTFT lange beurten (~5-11K nieuw) | 8,53 s (±2,4%) | 8,60 s (±0,8%) | +0,8% (ruis) | 8,58 s | 8,41 s |
| TTFT korte beurten | 0,460 s | 0,410 s | −10,9% | 0,467 s | 0,388 s |
| TTFT laatste lange beurt (~60K) | 11,9 s | 11,9 s | 0% | 11,8 s | 11,6 s |
| Decode, mediaan alle beurten | 81,0 tok/s | 79,1 tok/s | −2,3% | 78,4 | 77,9 |
| Decode, lange beurten | 77,1 tok/s | 76,4 tok/s | −0,9% | 78,3 | 77,9 |
| Hele gesprek | 70,9 s | 70,8 s | 0% | 70,9 s | 69,7 s |

Korte beurten per beurt (s, beurt 2 tot 12): oud 0,43 0,41 0,44 0,48 0,74 0,82; nieuw 0,41 0,41 0,43 0,47 0,36 0,40; main 0,40 0,45 0,50 0,52 0,45 0,48. Bij de laatste twee korte beurten hergebruikt 2.11.3 een herstelpunt verder terug (278 tegen 43 nieuwe tokens); dat is in 2.12.0 al beter.

### Agent via `/v1/messages` (12 beurten, ~5K tot ~61K tokens)

| | oud | nieuw | verschil | main |
|---|---|---|---|---|
| **Serverstandaard-temperatuur (realistisch)** | | | | |
| TTFT lange beurten | 18,5 s (±1,2%) | 9,02 s (±0,1%) | −51% | 18,5 s |
| TTFT korte beurten | 1,19 s | 0,52 s | −56% | 1,49 s |
| TTFT p90 | 21,0 s | 10,5 s | −50% | 20,8 s |
| Hele gesprek | 164,4 s (±0,5%) | 58,5 s (±5%) | −64% | 195,9 s |
| Herhaalpogingen per gesprek | 10 | 0 tot 1 | | 9 |
| **Temperatuur 0 (vergelijkbaar)** | | | | |
| TTFT lange beurten | 18,6 s | 9,03 s | −52% | 18,3 s |
| TTFT korte beurten | 1,22 s | 0,51 s | −58% | 1,85 s |
| TTFT p90 | 21,3 s | 10,6 s | −50% | 20,7 s |
| Hele gesprek | 150,0 s | 55,4 s | −63% | 168,7 s |
| Herhaalpogingen per gesprek | 10 | 0 | | 10 |

Kaal (temperatuur 0) geeft dezelfde agentcijfers als nieuw (lange beurten 9,05 s, gesprek 55,5 s). Main heeft maar één run per temperatuur; de korte beurten daar schommelen (0,6 tot 4,7 s).

### Geheugen (GiB, uit `/v1/mtplx/snapshot` en `footprint`)

| Na | Meting | oud | nieuw | verschil | main | kaal |
|---|---|---|---|---|---|---|
| classificatie | MLX-piek | 29,44 | 28,30 | −1,14 | 29,44 | 27,92 |
| classificatie | footprint | 31,68 | 28,35 | −3,33 | 31,69 | 28,06 |
| agent | actief | 36,34 | 36,52 | +0,18 | 36,52 | 36,00 |
| agent | MLX-piek | 40,72 | 39,10 | −1,62 | 40,72 | 38,57 |
| agent | footprint | 41,12 | 41,38 | +0,26 | 40,83 | 40,81 |
| agent | footprint-piek | 45,07 | 43,53 | −1,54 | 44,97 | 43,09 |
| agent | bank (entries) | 10,22 (6) | 11,02 (9) | +0,80 | 10,13 (6) | 10,28 (6) |
| agent | bankplafond (`effective_max_bytes`) | 16,01 | 17,39 | +1,38 | 16,19 | 17,39 |

De geheugencijfers zijn per versie vrijwel gelijk over de runs (spreiding onder 1%, footprint tot 1,7%).

## Kwaliteit

### Classificatie

- **Scoreroute:** oud, main, nieuw en kaal geven op alle 240 berichten hetzelfde oordeel en exact dezelfde kansen (maximaal verschil 0,0000), in elke run. Nauwkeurigheid tegen de teacher-labels overal 0,717. De top-K-fix is dus ook op het echte model bitgelijk.
- **Eerste-token-logprobs:** 228 van 240 dezelfde oordelen als de scoreroute (met en zonder bank). Bij de 12 verschillen heeft elke route er 3 goed; nauwkeurigheid 0,717 zonder bank, 0,708 met bank. Met en zonder bank onderling 236/240, maximaal kansverschil 0,17. De bank knipt de prefill in twee forwards (vondst 40) en verandert daardoor via de MoE-routering (vondst 29) een paar oordelen.
- Binnen elke versie en route zijn alle runs gelijk (240/240, kansverschil 0).

### Gretige uitvoer (27 streams: 3 decode, 12 chat, 12 agent op temperatuur 0)

| Vergelijking | Gelijke streams | Waar het afwijkt |
|---|---|---|
| oud-2 tegen oud-4, nieuw-2 tegen nieuw-4, kaal-1 tegen kaal-2 | 27/27 | nergens: elke versie is deterministisch |
| oud tegen nieuw | 2/27 | decode na 138 van 512 tokens; 10 van 12 chatbeurten na 11 tot 96 tokens; agent alle 12 |
| oud tegen main (upstream) | 4/27 | decode na 138 tokens; 8 chatbeurten na 11 tot 107 tokens; agent alle 12 (beurt 3 tot 8 pas na 86 tot 151 tokens) |
| main tegen kaal (onze fixes) | 17/27 | decode en chat 15/15 gelijk; agent beurt 3 tot 12 vanaf het eerste token |
| kaal tegen nieuw (twee schakelaars) | 19/27 | 8 chatbeurten na 16 tot 96 tokens; decode en agent gelijk |

Verklaring:

- **Upstream verandert de rekenpaden.** Tussen 2.11.3 en 2.12.0 zitten 304 commits. Waarschijnlijk (niet apart bewezen) is het de wijziging "Eager BF16 verification uses the attention-gate rounding already used by compiled verification" uit de CHANGELOG van 2.12.0: op dit 6-bit-model loopt verify altijd eager (vondst 45), dus elke gegenereerde tekst gaat door dat pad. Een ander afrondingspad verschuift via de MoE-routering (vondst 29) na enkele tientallen tokens een gretige keuze.
- **Onze fixes zijn voor decode en chat bitgelijk** (main tegen kaal 15/15). In de agent verschilt de tekst bewust: 2.11.3 en main geven na een kale toolaanroep de uitvoer van een herhaalpoging terug, de fix geeft de eerste, goedgevormde toolaanroep terug (29 tokens) zonder herhaling.
- **De 128-drempel verandert de chatuitvoer.** De completions-bank raakt de chatroute niet, dus komen de 8 afwijkende chatbeurten vrijwel zeker van `--ram-session-prefix-min-match-tokens 128`: een ander herstelpunt geeft een andere indeling van de prefill en daarmee andere MoE-routering. Niet slechter, wel anders.

## Toewijzing

| Winst | Oorzaak | Bewijs |
|---|---|---|
| Agent: TTFT −51 tot −58%, gesprek −63 tot −64%, 10 naar 0 herhaalpogingen | `fix/messages-ttft` ([messages-ttft](2026-09-26-messages-ttft.md)) | main doet nog 9 tot 10 herhaalpogingen en is even traag als oud (korte beurten zelfs trager: 1,49 tot 1,85 s); kaal en nieuw doen er 0 |
| Agent: MLX-piek −1,6 GiB, footprint-piek −1,5 GiB | idem: geen tweede prefill van ~60K tokens per beurt | main heeft dezelfde piek als oud (40,72 GiB) |
| Bankplafond +1,4 GiB | volgt de lagere piek (mechanisme uit [bankplafond](2026-09-26-bankplafond.md)) | kaal en nieuw 17,39, oud en main 16,0 tot 16,2 GiB |
| Scoreroute −17%, footprint na classificatie −3,3 GiB | `feat/prompt-scoring-topk` ([snellere-scoreroute](2026-09-26-snellere-scoreroute.md)) | main = oud (535 ms, zelfde geheugen) |
| Eerste-token-logprobs 336 ms (−37% tegen de huidige route) | `feat/first-token-logprobs` ([eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md)); geen blank retries bij logprobs | route bestaat niet op oud en main; alleen zonder completions-bank |
| Chat, korte beurten −11% (kaal −16%) | Gemma-4-controle zonder `get_vocab()` (~47 ms, [completions-overhead](2026-09-26-completions-overhead.md)), chat-encode-memo en scoped encoderen per beurt ([chat-encode-memo](2026-09-26-chat-encode-memo.md), [chatroute-scoped](2026-09-26-chatroute-scoped.md)) | main 0,467 s, kaal 0,388 s: ~80 ms per beurt; niet verder uitgesplitst per fix |
| Chat, laatste korte beurten 0,74-0,82 naar 0,36-0,48 s | upstream: herstelpunt dichter bij het eind (43 in plaats van 278 nieuwe tokens) | main 0,45-0,48 s |
| Decode +6% (75,3 naar 80,0 tok/s) | upstream 2.11.3 naar 2.12.0 | main 79,7 tok/s; niet toegewezen aan een specifieke commit |
| Chat-decode −2% (mediaan), lange beurten −1% | upstream en schakelaars, binnen ~2% | main 78,4, kaal 77,9, nieuw 79,1 tok/s: te klein om toe te wijzen |

**Geen meetbaar effect in deze werklast:** prefix-hergebruik onder 512 tokens (de 240 berichten delen geen begin van 128 tokens: 0 hits behalve drie herhaalde opwarmprompts), `sessionbank_put_s` en de afwijsreden (alleen telemetrie), testisolatie (alleen tests), bankplafond-knop (staat uit).

## Nieuwe vondst: de completions-bank maakt eerste-token-logprobs 140 ms trager

Met `MTPLX_COMPLETIONS_SESSION_BANK=1` kost eerste-token-logprobs 481 ms per bericht, zonder 336 ms: 145 ms verschil, precies de extra MoE-forward die in [chatroute-scoped](2026-09-26-chatroute-scoped.md) (vondst 40) uit de code werd voorspeld (~140 ms: met een bank wordt een GDN-grens 64 tokens voor het eind vastgelegd en de prefill in twee forwards geknipt). Vondst 40 is daarmee gemeten bevestigd. De scoreroute gebruikt de bank niet en merkt er niets van. De bank levert voor deze berichten niets op (geen gedeeld begin) en kost daarnaast 0,8 GiB (3 extra entries) en een paar afwijkende oordelen.

## Aanbeveling voor de eindconfig

*Vervangen door de aanbeveling onder eindtest 2.* Voor de werklast van het platform: de integratietak **zonder** `MTPLX_COMPLETIONS_SESSION_BANK=1` en zonder `--ram-session-prefix-min-match-tokens 128` (configuratie "kaal"), en de classifier omzetten naar eerste-token-logprobs. Dat geeft de snelste classificatie (336 ms, gelijke nauwkeurigheid), dezelfde agentwinst, bitgelijke chatuitvoer ten opzichte van 2.12.0 en het laagste geheugen. De twee schakelaars zijn pas zinvol voor prompts met een gedeeld begin (vondst 3, 4 en 16); dan eerst de extra forward van vondst 40 oplossen.

## Meetmateriaal

Lokaal, niet in de fork: `~/Dev/laya-nl/eindbench/resultaten/e-{oud,nieuw,main,kaal}-*.json` (met `.ruw.json`), serverlogs in `logs/`, analyse met `eindanalyse.py`. Nieuwe schakelaar `EINDBENCH_KAAL=1` in `instellingen.zsh` (nieuw zonder eigen schakelaars, ook voor een andere worktree via `NIEUW_WORKTREE`). De tijdelijke worktree van `origin/main` is na afloop verwijderd; poort 8000 is vrij.

## Eindtest 2: met invariante prefill en A3B-hooks (26 september, avond)

### Samenvatting

De integratietak met de invariante prefill erbij (`MTPLX_BATCH_INVARIANT_PREFILL=1`, zonder completions-bank en zonder 128-drempel) is op het echte model overal minstens zo snel als in eindtest 1 en maakt de classificatie die het platform nu gebruikt (scoreroute) veel sneller:

- **Scoreroute:** 328 in plaats van 512 ms per bericht (−36%). In eindtest 1 was dat zonder de lane −19%; de lane voegt hier ongeveer 21% toe. Het platform krijgt die winst al bij de overstap, zonder de classifier aan te passen.
- **Eerste-token-logprobs:** 314 ms per bericht (−39% tegen de huidige scoreroute), nauwkeurigheid 0,725. Ten opzichte van eindtest 1 zonder lane (336 ms, −37%) is dat binnen de ruis gelijk.
- **Agentgesprek:** 54 in plaats van 160 s (−66%), eerste token bij lange beurten 8,3 in plaats van 17,2 s. Gelijk aan eindtest 1: dat is de messages-ttft-fix.
- **Chat:** korte vervolgbeurten −18%, lange beurten −2,5%; de decode in de chat is 5% lager dan op 2.11.3 (vooral upstream, zie vondst 60).
- **Geheugen:** MLX-piek na het agentgesprek 2,0 GiB lager, footprint na de classificatie 3,6 GiB lager; de lane kost ten opzichte van eindtest 1 niets meetbaars behalve 0,26 GiB MLX-piek na de classificatie.
- **Kwaliteit:** de lane verandert 12 van de 240 oordelen van de scoreroute (4 goed, 4 goed: nauwkeurigheid gelijk, 0,717), maakt scoreroute en eerste-token-logprobs het vrijwel eens (238/240 tegen 228/240 zonder lane) en geeft andere, niet slechtere gretige teksten.

### Opzet

- **Hardware, model en argumenten:** als in eindtest 1 (M5 Pro, 64 GB, Balance-model, productie-argumenten uit `mtplx-originele-args.txt`).
- **Oud:** 2.11.3 uit de Homebrew-venv, zoals het platform.
- **Nieuw:** `~/Dev/MTPLX-integratie` op 49c4a6ee (`perf/integratie`, fast-forward naar `perf/invariant-lane-gate`): alle fixes uit eindtest 1 plus de invariante prefill ([batch-invariante-router](2026-09-26-batch-invariante-router.md)), de in-forward GDN-grenzen voor A3B ([gdn-inforward-a3b](2026-09-26-gdn-inforward-a3b.md)) en de lane-gate ([modeltest](2026-09-26-modeltest.md)). Env alleen `MTPLX_BATCH_INVARIANT_PREFILL=1`; geen completions-bank, geen `--ram-session-prefix-min-match-tokens 128`.
- **Referentie "kaal":** e-kaal-1 en e-kaal-2 uit eindtest 1 (zelfde tak op 54c4ca5f, zonder lane en zonder schakelaars, agent op temperatuur 0), ochtend van dezelfde dag.
- **Volgorde (17:41 tot 18:16, om en om):** oud-1, nieuw-1, oud-2, nieuw-2 (agent op serverstandaard-temperatuur 0,6), oud-t0, nieuw-t0 (agent op temperatuur 0). Decode, classificatie en chat zijn gretig, dus mediaan over 3 runs per versie; agent over 2 runs (standaard) en 1 run (temperatuur 0). Verse server en eigen SSD-cachemap per run; aangestuurd door een platform-scripttaak, zonder verstoringen.
- **Controles (log `eindtest2.log`):**
  - Lane op nieuw in alle drie de runs geïnstalleerd, gecontroleerd in `/health` en in de serverlog: 391 lineaire lagen, 41 SwitchGLU's, 0 overgeslagen, attentie gehookt, 30 GDN-lagen met in-forward-grenzen. Op oud geen lane (zoals verwacht).
  - `requests_completed` klopt in alle zes runs: opwarmen 2, decode 3, scoreroute 243, eerste-token-logprobs 243 (nieuw) of 0 (oud), chat 12, agent 12 op nieuw (0 herhaalpogingen) en 24 op oud (12 verzoeken plus interne herhaalpogingen). Uitzondering: oud-1 telt 29 agentverzoeken; bij temperatuur 0,6 deed 2.11.3 daar 5 interne pogingen meer, wat past bij het langere gesprek (170 tegen 149 s). Geen vreemd verkeer van het platform.
  - 0 tracebacks in de serverlogs, alle benches exit 0, geen HTTP-fouten in de classificatie (0 van 240 per route).
- **Dagverschil:** oud was 's avonds 2 tot 7% sneller dan 's ochtends in eindtest 1 (decode 77,3 tegen 75,3 tok/s, scoreroute 512 tegen 533 ms, agent lang 17,7 tegen 18,6 s), bij bitgelijke uitvoer (oud-t0 tegen e-oud-4: 27/27 streams gelijk). De kolom "kaal" vergelijk ik daarom via het verschil met de oud-run van dezelfde test, niet via de absolute tijden.

### Resultaten

Mediaan over de runs, spreiding (halve afstand minimum tot maximum) tussen haakjes. "Kaal tegen oud" is kaal uit eindtest 1 tegen oud uit eindtest 1.

| Meting | oud | nieuw | nieuw tegen oud | kaal (eindtest 1) | kaal tegen oud |
|---|---|---|---|---|---|
| Decode, tok/s | 77,3 (±0,2%) | 81,6 (±0,4%) | +5,5% | 80,8 | +7,3% |
| Scoreroute p50 | 512 ms (±0,8%) | 328 ms (±1,2%) | −36% | 430 ms | −19% |
| Scoreroute p90 | 599 ms | 370 ms | −38% | 519 ms | −17% |
| Scoreroute totaal 240 | 121,5 s | 76,7 s | −37% | 104,0 s | −18% |
| Scoreroute ≥512 tokens (52 berichten) p50 / p90 | 594 / 700 ms | 369 / 387 ms | −38% / −45% | 513 / 615 ms | −17% / −17% |
| Eerste-token-logprobs p50 | n.v.t. | 314 ms (±0,9%) | −39% tegen scoreroute | 336 ms | −37% |
| Eerste-token-logprobs p90 | n.v.t. | 351 ms | −41% | 356 ms | −43% |
| Eerste-token-logprobs totaal 240 | n.v.t. | 73,3 s | −40% | 77,1 s | −39% |
| Eerste-token-logprobs ≥512 tokens (51 berichten) p50 / p90 | n.v.t. | 350 / 366 ms | −41% / −48% | 353 / 376 ms | −43% / −49% |
| Agent, standaard: TTFT lang | 17,2 s (±0,2%) | 8,26 s (±0,1%) | −52% | | |
| Agent, standaard: TTFT kort | 1,18 s | 0,55 s (±5%) | −54% | | |
| Agent, standaard: hele gesprek | 159,6 s (±6,7%) | 54,0 s (±5,2%) | −66% | | |
| Agent, temperatuur 0: TTFT lang | 17,7 s | 8,51 s | −52% | 8,90 s | −52% |
| Agent, temperatuur 0: TTFT kort | 1,18 s | 0,57 s | −52% | 0,51 s | −59% |
| Agent, temperatuur 0: hele gesprek | 143,7 s | 52,2 s | −64% | 54,5 s | −64% |
| Chat: TTFT lange beurten | 7,91 s (±0,8%) | 7,71 s (±1,0%) | −2,5% | 8,41 s | −1,4% |
| Chat: TTFT korte beurten | 0,462 s | 0,381 s | −17,6% | 0,388 s | −15,7% |
| Chat: decode, mediaan alle beurten | 81,9 tok/s | 77,7 tok/s | −5,2% | 77,9 | −3,8% |
| Chat: decode, lange beurten | 77,8 tok/s | 76,5 tok/s | −1,7% | 77,9 | +1,0% |
| Chat: hele gesprek | 67,1 s | 65,5 s | −2,4% | 69,7 s | −1,7% |
| MLX-piek na classificatie | 29,44 GiB | 28,18 GiB | −1,26 | 27,92 | −1,52 |
| Footprint na classificatie | 31,69 GiB | 28,09 GiB | −3,60 | 28,06 | −3,62 |
| MLX-piek na agent | 40,72 GiB | 38,67 GiB | −2,05 | 38,58 | −2,14 |
| Footprint-piek na agent | 45,07 GiB | 43,10 GiB | −1,97 | 43,09 | −1,98 |
| Bank na agent (entries) | 10,22 (6) | 10,28 (6) | +0,06 | 10,28 (6) | |
| Bankplafond na agent | 16,02 GiB | 17,39 GiB | +1,37 | 17,39 | |

**Wat de invariante prefill en de hooks er extra aan bijdragen** (afgeleid: verhouding nieuw/oud van vanavond gedeeld door kaal/oud van vanochtend; geen gelijktijdige meting):

- Scoreroute: ongeveer −21% (p50) tot −23% (totaal). Dat past bij de 1,28× uit de servermeting van [batch-invariante-router](2026-09-26-batch-invariante-router.md). Bij berichten vanaf 512 tokens is het effect groter (p90 −45% tegen −17%).
- Eerste-token-logprobs: −3% (p50), +3% (p90), −1% (totaal): binnen de ruis. De in-forward-grenzen gaven in [gdn-inforward-a3b](2026-09-26-gdn-inforward-a3b.md) −21% vanaf 512 tokens alleen mét completions-bank; zonder bank is er één forward en dus niets te winnen.
- Agent, chat, decode en geheugen: binnen 2% gelijk aan kaal, behalve 0,26 GiB meer MLX-piek na de classificatie.

### Kwaliteit

**Classificatie** (240 berichten, nauwkeurigheid tegen de teacher-labels):

| Vergelijking | Zelfde oordeel | Maximaal kansverschil | Nauwkeurigheid |
|---|---|---|---|
| oud scoreroute tegen nieuw scoreroute | 228/240 | 0,34 | 0,717 en 0,717 |
| oud scoreroute tegen nieuw eerste-token-logprobs | 230/240 | 0,36 | 0,717 en 0,725 |
| nieuw scoreroute tegen nieuw eerste-token-logprobs | 238/240 | 0,059 | 0,717 en 0,725 |
| kaal eerste-token-logprobs tegen nieuw eerste-token-logprobs | 232/240 | | 0,717 en 0,725 |
| binnen elke versie en route, alle runs | 240/240 | 0 | |

- In eindtest 1 was de scoreroute bitgelijk tussen oud en nieuw; nu verandert de lane 12 oordelen (5 van berichten vanaf 512 tokens). Van die 12 heeft oud er 4 goed en nieuw er 4: de nauwkeurigheid blijft 0,717. De lane rekent de prefill in vaste blokken met andere afronding (vondst 29); bij twijfelgevallen kantelt dan het oordeel.
- Eerste-token-logprobs tegen de scoreroute: zonder lane 228/240 gelijk, met lane 238/240. De twee routes zien nu vrijwel dezelfde prefill, doordat de uitkomst niet meer van de blokgrootte afhangt. Bij de 2 verschillen heeft eerste-token-logprobs beide goed, vandaar 0,725 tegen 0,717.

**Gretige uitvoer** (27 streams: 3 decode, 12 chat, 12 agent op temperatuur 0):

| Vergelijking | Gelijke streams | Waar het afwijkt |
|---|---|---|
| nieuw-1, nieuw-2, nieuw-t0 onderling (decode en chat) | 15/15 | nergens: deterministisch |
| oud-1, oud-2, oud-t0 onderling (decode en chat) | 15/15 | nergens |
| oud-t0 (vanavond) tegen e-oud-4 (vanochtend) | 27/27 | nergens: 2.11.3 is ook over de dag gelijk |
| oud-t0 tegen nieuw-t0 | 3/27 | decode na 94 van 512 tokens; 9 van 12 chatbeurten na 11 tot 93 tokens; agent alle 12 (messages-ttft-fix, bewust) |
| kaal-1 tegen nieuw-t0 (effect van de lane) | 14/27 | decode na 94 tokens; 8 chatbeurten na 15 tot 96 tokens; agent 10 van 12 gelijk, de eerste twee beurten na 7 en 14 tokens |

De agentruns op standaardtemperatuur verschillen onderling (samplen zonder seed), zoals verwacht. De afwijkingen met lane zijn hetzelfde patroon als in [batch-invariante-router](2026-09-26-batch-invariante-router.md): een andere afronding in de prefill kantelt na enkele tientallen tokens een gretige keuze tussen twee bijna gelijke tokens. Uit die rapportage: NLL +0,01 tot +0,13%, teksten beoordeeld als gelijkwaardig. De teksten van vanavond zijn niet opnieuw inhoudelijk gelezen.

### Aanbeveling voor de overstap van Bink (vervangt de aanbeveling van eindtest 1)

Commit 49c4a6ee (op de fork als `perf/invariant-lane-gate`, gelijk aan de lokale `perf/integratie`) met de productie-argumenten ongewijzigd en één env-variabele:

```
env PYTHONPATH=$HOME/Dev/MTPLX-integratie MTPLX_BATCH_INVARIANT_PREFILL=1 \
  ~/Dev/MTPLX/.venv/bin/python -P -m mtplx.server.openai ${=ARGS}
# ARGS=$(cat ~/Dev/laya-nl/mtplx-originele-args.txt)
# niet zetten: MTPLX_COMPLETIONS_SESSION_BANK, --ram-session-prefix-min-match-tokens
```

- De classifier kan eerst op de scoreroute blijven (328 ms, −36%, nauwkeurigheid gelijk). Omzetten naar eerste-token-logprobs levert daarna nog ~4% en 0,725 op; niet dringend.
- Controle na de overstap: `/health` moet onder `degradation.batch_invariant_prefill` een geïnstalleerde lane met 30 GDN-lagen tonen.
- Let op: de env-variabele geldt voor elk model dat het platform laadt. Op de dichte Qwen3.5-9B kost de lane ~4% op classificatie en ~15% op de gretige TTFT (vondst 59). Zolang alle gebruikers het Balance-model vragen, speelt dat niet.
- Voor een schone installatie buiten de worktree (bijvoorbeeld een eigen venv) is nog een build van 49c4a6ee nodig; dat is niet getoetst.

### Meetmateriaal eindtest 2

Lokaal: `~/Dev/laya-nl/eindbench/eindtest2.log`, `eindtest2-analyse.txt`, `resultaten/e2-{oud,nieuw}-{1,2,t0}.json` (met `.ruw.json`), serverlogs `logs/eindtest2-*`. De splitsing naar 512 tokens en de telling van goed/fout bij afwijkende oordelen zijn met een los script uit de rijen berekend. Poort 8000 is na afloop vrij gelaten.

## Eindtest 3: definitieve set op zes taken (28 september, nacht)

### Samenvatting

De definitieve set (perf/definitief a0c03d3d met invariante prefill, K1, K2, K3 en kopanker) is op het productiemodel duidelijk beter dan 2.11.3 en doet op de dichte modellen geen schade:

- **Qwen3.6 (productie):** een agentgesprek van twaalf beurten duurt 63 in plaats van 167 s (−62%; de schone nieuw-run 50,1 s, −70%). Het eerste token van een lange beurt komt na 7,9 in plaats van 18,0 s. Classificatie via de scoreroute 335 in plaats van 520 ms (−36%) met gelijke nauwkeurigheid (0,717). Na classificatie 3,6 GiB minder geheugen, MLX-piek 1,1 GiB lager. Decode bij een korte prompt +4,9%; chat-decode −6%, net als in eindtest 2 andere tekst en geen tragere berekening.
- **Qwen3.5-9B:** dezelfde agentwinst (111 naar 40 s, herhaalpogingen 8 naar 0), classificatie −16% met precies dezelfde oordelen, geheugenpiek 1,9 GiB lager.
- **Qwen3.8-27B:** een paar procent sneller (agent −3%, korte beurten −15%, classificatie −6%), 3,1 GiB minder na classificatie, MLX-piek 1 GiB hoger.
- **Gemma 4:** chat veel sneller (korte vervolgbeurt 0,5 in plaats van 24,7 s), agent gelijk, scoreroute werkt nu (op 2.11.3 een serverfout). Kost 5 GiB meer werkgeheugen na de agent, omdat nieuw gesprekken vasthoudt.
- **Bonsai-2-27B:** niet te vergelijken; 2.11.3 kan het model niet laden. Nieuw draait stabiel.

### Opzet

- **Oud:** 2.11.3 uit de Homebrew-venv, zoals eindtest 2. **Nieuw:** `~/Dev/MTPLX-definitief` op a0c03d3d (`perf/definitief`), env uit `~/Dev/laya-nl/eindtest3/nieuw.env`: `MTPLX_BATCH_INVARIANT_PREFILL=1`, `MTPLX_MTP_HISTORY_CACHE_ONLY=1` (K1), `MTPLX_PREFILL_CHUNK_SIZE_DENSE=4096` en `_REPAGE=4096` (K2), `MTPLX_A3B_MOE_PREFILL_COMBINE=1` (K3), `MTPLX_SESSION_HEAD_ANCHOR=1` (kopanker). Bewust uit: voorrang, K4, T3, T7, W4.
- **Taken** (platform-scripttaken, `eindtest3.zsh <taak>`, 02:08 tot 04:13): qwen36 (Balance, volledige werklast ~60K, agent op serverstandaard), qwen36-t0 (idem, agent op temperatuur 0), qwen38 (Qwen3.8-27B Speed, klein), gemma (Gemma 4 Speed, eigen werklast), q9b (Qwen3.5-9B Speed, middel), bonsai (Ternary-Bonsai-2-27B Speed, klein). Per taak oud, nieuw, oud, nieuw op een verse server met eigen SSD-cachemap; dezelfde argumenten en model-env voor oud en nieuw.
- **Analyse:** `analyse3.py`, mediaan over twee runs per versie. Qwen3.6 per deel apart (`--deel std` en `--deel t0`).

### Geldigheid

- Alle 22 runs zijn gestart en afgerond (bench exit 0, lader 100 W). 2.11.3 startte bij Qwen3.6, Qwen3.8, Gemma en Qwen3.5-9B normaal; de nieuwe server gaf nergens een traceback.
- **Bonsai:** 2.11.3 stopte beide keren tijdens het laden (`Model type prism_hadamard_qwen35 not supported`). Alleen nieuw is gemeten.
- **Lane:** op Qwen3.6 geïnstalleerd in alle vier nieuw-runs (391 lineaire lagen, 41 SwitchGLU's, 30 GDN-lagen, 41 MoE-combineblokken). Op de dichte modellen geweigerd zoals verwacht: Qwen3.8 en 9B `dense_model`, Gemma `not_reached`, Bonsai `unsupported_linear:HadamardQuantizedLinear`.
- **Gemma, classificatie:** op 2.11.3 geeft de scoreroute HTTP 500 (`Gemma4AssistantRuntime` heeft geen `forward_ar`); op nieuw werkt die (nauwkeurigheid 0,800). Eerste-token-logprobs werkt op geen van beide ("logprobs are not supported on the gemma4_assistant backend"). 2.11.3 geeft voor Gemma geen tokentelling terug, dus decode tok/s is daar alleen uit het serverlog te halen: 21,6 tegen 21,9 tok/s (decode) en 28,4 tegen 27,9 tok/s (chat), dus gelijk.
- **Vreemd verkeer van Bink.** Het platform stuurde tijdens drie Qwen3.6-runs eigen verzoeken naar poort 8000 (titels en de nachtelijke conversatiereview, `max_tokens` 7.000 of 32.000):

  | Run | Verzoeken | Rekentijd | Fase | Gevolg |
  |---|---|---|---|---|
  | qwen36 nieuw-2 | 2 titels (1.057 tokens) | 37,5 s | agent | agentgesprek 75,9 tegen 50,1 s in nieuw-1; mediaan onderschat de winst |
  | qwen36-t0 oud-t0-1 | 3 titels | 66,4 s | agent | oud-agent te traag (192 s) |
  | qwen36-t0 oud-t0-2 | 4, conversatiereview en sessienaam (4.742 tot 23.245 tokens) | 62,4 s | agent | oud-agent te traag (182 s); de controle op `requests_completed` zag dit niet, de marge voor herhaalpogingen dekte het |
  | qwen36-t0 nieuw-t0-2 | 4, conversatiereview (ruim 60K tokens) | 87,0 s | decode | snelheden gelijk aan nieuw-t0-1, maar de sessies bleven in de bank: geheugencijfers van deel 2 onbruikbaar |

  Qwen3.8, Gemma, 9B en Bonsai hadden geen vreemd verkeer. Voor Qwen3.6 gelden daarom de cijfers van deel 1, met de agentwinst als ondergrens; deel 2 levert de bitgelijkheid.
- **Analysebestand:** deel 2 schreef zijn analyse, zoals gepland, over die van deel 1 heen in één bestand over alle acht runs. Daarin waren alleen de agentrijen per temperatuur gesplitst. `analyse3.py` kreeg `--deel std|t0` en een sectie "vreemd verkeer" uit de serverlogs; `eindtest3.zsh` schrijft voortaan `analyse-<taak>.txt`. Nu: `analyse-qwen36.txt` (deel 1), `analyse-qwen36-t0.txt` (deel 2), `analyse-qwen36-samen.txt` (alles), het origineel als `analyse-qwen36-samen-origineel.txt`.

### Resultaten per model

Mediaan over twee runs per versie. Verschil relatief bij tijden en snelheden, absoluut in GiB bij geheugen.

**Qwen3.6-35B-A3B Balance, deel 1** (agent op serverstandaard, productie):

| Meting | oud | nieuw | verschil |
|---|---|---|---|
| Agent: TTFT lange beurten | 18,0 s | 7,94 s | −56% |
| Agent: TTFT korte beurten | 1,16 s | 0,60 s | −48% |
| Agent: hele gesprek | 167,2 s | 63,0 s (schone run 50,1) | −62% (−70%) |
| Agent: herhaalpogingen | 10 | 0 | −10 |
| Scoreroute p50 / totaal | 520 ms / 123,2 s | 335 ms / 78,3 s | −36% / −36% |
| Eerste-token-logprobs p50 | n.v.t. | 314 ms | −40% tegen oud scoreroute |
| Decode korte prompt | 77,2 tok/s | 80,9 tok/s | +4,9% |
| Chat: decode / TTFT lang | 81,8 tok/s / 8,19 s | 76,8 tok/s / 7,12 s | −6,0% / −13% |
| MLX-piek / footprint-piek | 40,72 / 45,05 GiB | 39,64 / 44,20 GiB | −1,08 / −0,85 |
| Footprint na classificatie / na agent | 31,61 / 40,70 GiB | 27,99 / 40,96 GiB | −3,61 / +0,26 |

**Qwen3.6-35B-A3B Balance, deel 2** (agent op temperatuur 0): agent TTFT lang 18,7 naar 7,63 s (−59%), hele gesprek 187,2 naar 48,0 s (−74%; tegen het schone oud van deel 1 −71%), scoreroute 524 naar 340 ms (−35%), eerste-token-logprobs 319 ms, decode 77,0 naar 80,8 tok/s. Geheugen niet bruikbaar (zie geldigheid).

**Qwen3.8-27B Speed** (dicht, lane geweigerd):

| Meting | oud | nieuw | verschil |
|---|---|---|---|
| Agent: TTFT lang / kort | 11,03 / 0,499 s | 10,51 / 0,423 s | −4,7% / −15% |
| Agent: hele gesprek (herhaalpogingen) | 62,4 s (1) | 60,8 s (1) | −2,7% |
| Scoreroute p50 | 1.350 ms | 1.270 ms | −5,9% |
| Eerste-token-logprobs p50 | n.v.t. | 1.240 ms | −8,1% tegen oud scoreroute |
| Decode korte prompt / chat | 31,4 / 38,2 tok/s | 30,9 / 38,5 tok/s | −1,4% / +0,9% |
| Chat: TTFT lang | 10,89 s | 10,90 s | +0,1% |
| MLX-piek / footprint-piek | 34,97 / 40,16 GiB | 36,00 / 40,91 GiB | +1,03 / +0,76 |
| Footprint na classificatie / na agent | 24,11 / 37,24 GiB | 21,00 / 36,73 GiB | −3,10 / −0,51 |

**Gemma 4 Speed** (dicht, lane geweigerd):

| Meting | oud | nieuw | verschil |
|---|---|---|---|
| Agent: TTFT lang / kort | 13,61 / 0,616 s | 13,65 / 0,617 s | +0,3% / +0,2% |
| Agent: hele gesprek | 98,5 s | 98,7 s | +0,2% |
| Scoreroute p50 / nauwkeurigheid | serverfout | 1.375 ms / 0,800 | |
| Eerste-token-logprobs | niet ondersteund | niet ondersteund | |
| Decode korte prompt: TTFT / tok/s | 0,392 s / (serverlog 21,6) | 0,149 s / 22,0 | −62% / gelijk |
| Chat: TTFT lang / kort | 24,37 / 24,7 s | 14,52 / 0,50 s | −40% / −98% |
| Chat: hele blok | 189,0 s | 64,2 s | −66% |
| MLX-piek / footprint-piek | 35,54 / 36,77 GiB | 37,79 / 40,09 GiB | +2,25 / +3,32 |
| Footprint na classificatie / na agent | 20,95 / 26,40 GiB | 21,34 / 31,40 GiB | +0,39 / +5,00 |

**Qwen3.5-9B Speed** (dicht, lane geweigerd):

| Meting | oud | nieuw | verschil |
|---|---|---|---|
| Agent: TTFT lang / kort | 16,99 / 1,447 s | 7,42 / 0,500 s | −56% / −65% |
| Agent: hele gesprek (herhaalpogingen) | 111,4 s (8) | 40,3 s (0) | −64% |
| Scoreroute p50 | 517 ms | 435 ms | −16% |
| Eerste-token-logprobs p50 | n.v.t. | 387 ms | −25% tegen oud scoreroute |
| Decode korte prompt / chat | 63,3 / 68,2 tok/s | 61,8 / 68,8 tok/s | −2,4% / +0,9% |
| Chat: TTFT lang | 7,69 s | 7,64 s | −0,7% |
| MLX-piek / footprint-piek | 21,31 / 25,58 GiB | 19,44 / 23,77 GiB | −1,88 / −1,81 |
| Footprint na classificatie / na agent | 12,80 / 21,76 GiB | 9,36 / 21,13 GiB | −3,44 / −0,62 |

**Ternary-Bonsai-2-27B Speed**, alleen nieuw: agent TTFT lang 9,95 s, kort 0,42 s, gesprek 67,3 s (2 herhaalpogingen); scoreroute 1.207 ms (0,700), eerste-token-logprobs 1.145 ms (0,713); decode 42,5 tok/s, chat 44,0 tok/s; MLX-piek 23,96 GiB.

### Kwaliteit

| Model | Scoreroute oud / nieuw | Zelfde oordeel oud-nieuw | Eerste-token (nieuw) | Gretige streams oud-oud / nieuw-nieuw / oud-nieuw |
|---|---|---|---|---|
| Qwen3.6 deel 1 | 0,717 / 0,717 | 228/240 | 0,725 | 15/15 / 15/15 / 3/15 |
| Qwen3.6 deel 2 | 0,717 / 0,717 | 228/240 | 0,725 | 27/27 / 27/27 / 3/27 |
| Qwen3.8 | 0,763 / 0,763 | 80/80 | 0,750 | 19/19 / 19/19 / 14/19 |
| Gemma 4 | serverfout / 0,800 | | niet ondersteund | 15/15 / 15/15 / 11/15 |
| Qwen3.5-9B | 0,688 / 0,688 | 240/240 | 0,688 | 23/23 / 23/23 / 9/23 |
| Bonsai | niet te laden / 0,700 | | 0,713 | nieuw-nieuw 19/19 |

Beide versies zijn met zichzelf bitgelijk op alle modellen, ook in het agentgesprek bij temperatuur 0. Oud tegen nieuw wijkt op Qwen3.6 af zoals in eindtest 2 (upstream-rekenwijze, invariante prefill en de herhaalpogingsfix). Op de dichte modellen, waar de lane uit staat, is de classificatie bitgelijk met 2.11.3 en wijken de gretige teksten minder af.

### Conclusie

De set is klaar voor Bink: op het productiemodel is een agentgesprek twee tot drie keer zo snel, classificatie een derde sneller met gelijke oordelen, en het geheugen lager. De dichte modellen gaan er gelijk op of vooruit; alleen Gemma houdt meer geheugen vast, en dat is het gevolg van gesprekshergebruik dat de chat juist veel sneller maakt. Bink stuurde tijdens de meting eigen verzoeken naar de testserver; dat raakt alleen Qwen3.6 en maakt de winst eerder kleiner dan groter (vondst 70).

### Meetmateriaal eindtest 3

Lokaal in `~/Dev/laya-nl/eindtest3/`: `eindtest3-<taak>.log`, `analyse-<taak>.txt` (plus `analyse-qwen36-samen.txt`), `resultaten/<model>/e3-*.json` met `.ruw.json` en `.health.json`, `logs/<model>/` (server-, bench- en startlogs). Taken: `taken/done/eva-script-mtplx-eindtest-3-*.md` in het platform. Heranalyse op 28 sep 04:27, zonder nieuwe metingen.
