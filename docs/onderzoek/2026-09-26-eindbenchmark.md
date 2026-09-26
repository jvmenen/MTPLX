# Eindbenchmark: 2.11.3 tegen de integratietak (26 september 2026)

## Samenvatting

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

Voor de werklast van het platform: de integratietak **zonder** `MTPLX_COMPLETIONS_SESSION_BANK=1` en zonder `--ram-session-prefix-min-match-tokens 128` (configuratie "kaal"), en de classifier omzetten naar eerste-token-logprobs. Dat geeft de snelste classificatie (336 ms, gelijke nauwkeurigheid), dezelfde agentwinst, bitgelijke chatuitvoer ten opzichte van 2.12.0 en het laagste geheugen. De twee schakelaars zijn pas zinvol voor prompts met een gedeeld begin (vondst 3, 4 en 16); dan eerst de extra forward van vondst 40 oplossen.

## Meetmateriaal

Lokaal, niet in de fork: `~/Dev/laya-nl/eindbench/resultaten/e-{oud,nieuw,main,kaal}-*.json` (met `.ruw.json`), serverlogs in `logs/`, analyse met `eindanalyse.py`. Nieuwe schakelaar `EINDBENCH_KAAL=1` in `instellingen.zsh` (nieuw zonder eigen schakelaars, ook voor een andere worktree via `NIEUW_WORKTREE`). De tijdelijke worktree van `origin/main` is na afloop verwijderd; poort 8000 is vrij.
