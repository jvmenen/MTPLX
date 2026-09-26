# Modeltest: integratietak op Qwen3.5-9B, Ternary-Bonsai-2-27B en Gemma 4 (vondst 47, 26 september 2026)

De integratietak was alleen op Qwen3.6-35B-A3B gemeten ([eindbenchmark](2026-09-26-eindbenchmark.md)). Deze test draait dezelfde soort werklast op de drie andere modellen in `~/.mtplx/models` en vergelijkt steeds schone upstream met de integratietak. Meetscripts, ruwe resultaten en serverlogs staan lokaal in `~/Dev/laya-nl/modeltest/` (niet in de fork).

## Uitkomst

| Model | Oordeel script | Na analyse |
|---|---|---|
| Qwen3.5-9B Optimized-Speed (`q9b`) | afwijkend | Werkt; agentgesprek −70%, scoreroute −13%, MLX-piek −1,5 GiB. Gretige teksten 5/12 gelijk aan main: vrijwel zeker de invariante prefill (andere afronding in de prefill), uitsluitsel met één extra meting |
| Ternary-Bonsai-2-27B Optimized-Speed (`bonsai`) | OK | Gretige en agentteksten gelijk aan main, scoreroute 59/60 dezelfde oordelen. Met de lane-gate (`49c4a6ee`) krijgt Bonsai de lane niet meer |
| Gemma 4 Optimized-Speed (`gemma`) | kapot | Geen regressie: chat, agent en decode gelijk aan main. Beide classificatieroutes werken op Gemma 4 ook op main niet: de scoreroute geeft HTTP 500 (upstream), eerste-token-logprobs weigert PR #530 bewust met 400. Gemma-steun gebouwd op een aparte tak |

## Opzet

- **Gemeten**, M5 Pro 64 GB, MLX 0.32.2, 26 september 16:27 tot 16:59, onbeheerd via een platform-script-taak.
- **Varianten:** `main` = schone upstream 2.12.0 (`1de2b1c0`, `~/Dev/MTPLX-main`); `int` = `perf/integratie` op `198e7194` (`~/Dev/MTPLX-integratie`) met `MTPLX_BATCH_INVARIANT_PREFILL=1`. Na de run is de integratietak doorgeschoven naar `perf/invariant-lane-gate` (`49c4a6ee`), zie "Wat de lane-gate verandert".
- **Volgorde per model:** main, int, main, int (ABAB), elk op een verse server met eigen SSD-cachemap en eigen serverlog. Env voor beide: `MTPLX_PAGED_KV_QUANT=off`, `MTPLX_VLLM_METAL_PAGED_KV_QUANT=off`.
- **Werklast per variant** (`meet.py`): opwarmen; 12 neutrale prompts gretig (temperatuur 0, 128 tokens, streaming chat); 6 agentbeurten via `/v1/messages` zoals de Claude Agent SDK (tools, geen seed, temperatuur 0, 256 tokens per beurt, tot ~5.500 prompttokens); classificatie van 60 van de 240 classifierprompts via de scoreroute (`echo`, `max_tokens` 0, top-20) en via eerste-token-logprobs (`max_tokens` 1, top-20); decode van 256 tokens, 2 keer. Na elke stap geheugen uit `/health`, `/v1/mtplx/snapshot` en `footprint`.
- **Controle:** alle 12 runs compleet en onverstoord; `requests_completed` klopt met het aantal verzoeken plus de interne herhaalpogingen, geen verkeer van het platform.

### Args per model

Vastgelegd met `mtplx quickstart --dry-run --json` op main en integratie (gelijk), aangevuld met `--no-auth`. Alle drie: profiel turbo, `--generation-mode mtp`, `--verify-strategy capture_commit`, `--scheduler-mode serial`, SSD-cache aan (minimaal 512 tokens), `--tool-prompt-mode hybrid`.

| Model | backend | diepte | chat-template | reasoning | sampler (serverstandaard) |
|---|---|---|---|---|---|
| q9b | `qwen3_next` | 2 | `local_qwen36` | `qwen3`, effort auto | 0,6 / 0,95 / 20 |
| bonsai | `qwen3_next` | 1 | `tokenizer` | `qwen3`, effort medium | 1,0 / 0,95 / 20 |
| gemma | `gemma4_assistant` | 6 | `tokenizer` | `gemma4`, effort auto | 1,0 / 0,95 / 64 |

De lane werd geïnstalleerd op q9b (249 lineaire lagen, attentie, 24 GDN-lagen met in-forward-grenzen) en Bonsai (8 lineaire lagen, attentie, 48 GDN-lagen); op Gemma niet (het Gemma-paar komt niet bij de installatie).

## Resultaten

Mediaan over de twee runs per variant; binnen elke variant waren beide runs bitgelijk (gretig 12/12, agent 6/6, scoreroute 60/60 met kansverschil 0).

### Qwen3.5-9B

| meting | main | integratie | verschil |
|---|---|---|---|
| decode tok/s (256) | 65,4 | 64,1 | −2,1% |
| decode TTFT s | 0,236 | 0,211 | −10,9% |
| gretig TTFT s | 0,257 | 0,243 | −5,5% |
| agent TTFT s (mediaan) | 2,210 | 1,006 | −54,5% |
| agent totaal s | 22,8 | 6,8 | −70,0% |
| agent herhaalpogingen | 5 | 0 | |
| scoreroute p50 ms | 502 | 436 | −13,2% |
| eerste-token p50 ms | n.b. | 399 | |
| nauwkeurigheid (score / eerste-token) | 0,667 / - | 0,667 / 0,667 | |
| MLX-piek GiB | 11,56 | 10,02 | −13,3% |

Gelijkheid main tegen int: gretig 5/12 (eerste afwijking na 56, 191, 202, 231, 363, 379 en 387 tekens), agent 1/6 (verwacht door de messages-ttft-fix), scoreroute 60/60 dezelfde oordelen, maar slechts 9/60 exact dezelfde kansen (maximaal verschil 0,033). Eerste-token-logprobs en scoreroute op int: 60/60 dezelfde oordelen.

### Ternary-Bonsai-2-27B

| meting | main | integratie | verschil |
|---|---|---|---|
| decode tok/s (256) | 45,7 | 44,2 | −3,4% |
| decode TTFT s | 0,385 | 0,364 | −5,4% |
| gretig TTFT s | 0,408 | 0,361 | −11,6% |
| agent TTFT s (mediaan) | 2,462 | 2,281 | −7,3% |
| agent totaal s | 29,2 | 28,1 | −3,9% |
| scoreroute p50 ms | 1276 | 1165 | −8,7% |
| eerste-token p50 ms | n.b. | 1151 | |
| nauwkeurigheid (score / eerste-token) | 0,700 / - | 0,717 / 0,717 | |
| MLX-piek GiB | 14,38 | 13,26 | −7,8% |

Gelijkheid main tegen int: gretig 12/12, agent 6/6, scoreroute 59/60 (maximaal kansverschil 0,0036). Eerste-token tegen scoreroute op int: 60/60.

### Gemma 4

| meting | main | integratie | verschil |
|---|---|---|---|
| decode tok/s (256) | 22,4 | 22,3 | −0,6% |
| decode TTFT s | 0,261 | 0,258 | −1,0% |
| gretig TTFT s | 0,374 | 0,381 | +1,8% |
| agent TTFT s (mediaan) | 2,973 | 2,942 | −1,0% |
| agent totaal s | 74,0 | 74,4 | +0,6% |
| scoreroute, eerste-token | n.b. | n.b. | |
| MLX-piek GiB | 29,32 | 29,32 | 0,0% |

Gelijkheid main tegen int: gretig 12/12, agent 6/6. Beide classificatieroutes niet beschikbaar op beide varianten (zie bevinding 1).

## Bevinding 1: Gemma 4 "kapot"

**Wat er gebeurt** (`logs/meet-gemma-*.log`, `logs/server-gemma-*.log`):

- Eerste-token-logprobs (`/v1/completions`, `max_tokens` 1, `logprobs` 20): **HTTP 400** `logprobs are not supported on the gemma4_assistant backend`. Dat is een bewuste weigering in PR #530: `_reject_unservable_first_token_logprobs`, `mtplx/server/openai.py:25502-25506` op de integratietak. Het rapport [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) noemt die grens ook. Op main is het de oude 400 (`decode-time logprobs ... not supported yet`).
- Scoreroute (`echo`, `max_tokens` 0): **HTTP 500** op main én integratie, met een traceback. Main: `score_prompt_logprobs` → `_forward_ar_optional_hidden` → `rt.forward_ar` (`generation.py:8149` en `:1970`): `'Gemma4AssistantRuntime' object has no attribute 'forward_ar'`. Integratie: `score_prompt_logprobs` (`generation.py:8442`) → `_prompt_logit_slices` (`:8377`) → `_post_norm_logits_head` (`:8357`): `... has no attribute 'model'`. Die laatste functie komt uit `6a151a33` (`perf/batch-invariant-router`), niet uit #530 of #532; de fout is anders, de uitkomst dezelfde. De scoreroute kent het Gemma-paar niet: een upstreamfout (nieuwe vondst 57).

**Waar het zit.** Het Gemma-paar heeft een eigen generatiepad: `generate_ar` (`generation.py:8521-8538`) en `generate_mtpk` (`:9805-9834`) geven een Gemma-verzoek door aan `generate_gemma4_ar` en `generate_gemma4_assistant` in `mtplx/backends/gemma4_assistant.py`. Die kregen van #530 de parameter `first_token_logprobs_top_k` niet, daarom weigert de server het verzoek met 400 in plaats van een antwoord zonder logprobs te geven. Het is dus geen fout in #530 maar een bewuste beperking; ook geen fout in de Gemma-route zelf.

**Meetscript.** Niet de oorzaak: beide verzoeken falen vóór het model de prompt ziet. Wel een fout gevonden: de tokenizer van dit Gemma-4-pakket voegt bij een ruwe completions-prompt geen `<bos>` toe (`post_processor` zonder BOS, gecontroleerd met de tokenizer: eerste token `<|turn>`), terwijl `encode_gemma4_messages` dat bij chat wel doet. `meet.py` zet `<bos>` nu zelf vooraan. De rest van het formaat is gelijk aan `encode_gemma4_messages` zonder denken (lege denkblok `<|channel>thought\n<channel|>`).

**Werkte de scoreroute op Gemma wel?** Nee, op main noch integratie.

**Fix gebouwd** op tak `feat/first-token-logprobs-gemma4` (`4f7b3cbb`, op `fork/feat/first-token-logprobs` `c1962f4e`, worktree `~/Dev/MTPLX-ftl-gemma4`, gepusht naar de fork):

- Beide Gemma-lussen trekken het eerste token uit de targetrij van de prompt (`prompt_state.logits[0]`). Ze krijgen `first_token_logprobs_top_k` en rekenen die rij met dezelfde helper `first_token_logprobs` om (`backends/gemma4_assistant.py`, beide `GenerationOutput`s met `first_token_logprobs`).
- `generate_ar` en `generate_mtpk` geven de K door; de backend-weigering in `_reject_unservable_first_token_logprobs` is weg (de parameter `state` daarmee ook).
- CHANGELOG-regel van #530 aangepast; `docs/api.md` noemde de backendgrens niet.
- Tests: `tests/test_first_token_logprobs_gemma4.py`, 8 tests met een Gemma-target zonder model (zelfde patroon als `test_tail_gemma4_stream_holdback.py`): waarden en top-K gelijk aan een numpy-log-softmax van de promptrij, alleen het eerste token, niets zonder verzoek, en beide ingangen geven de K door. Zonder de fix falen de tests die logprobs vragen. Met `MTPLX_CONFIG=/nonexistent` groen: dit bestand plus `test_first_token_logprobs_api`, `test_tail_gemma4_stream_holdback`, `test_generation_sustained`, `test_api_benchmark_contracts` en beide `test_gemma4_*` (148 geslaagd, 1 overgeslagen), `test_server_openai` 394 geslaagd. Ruff: 510 meldingen in de drie bronbestanden, gelijk aan de basis; testbestand schoon.
- **Nog niet op de server getoetst.** Toets: `zsh ./runall.zsh --alleen gemma-logprobs` in `~/Dev/laya-nl/modeltest` (2 runs, `gemma-ftl-1/2`, zonder lane; `analyse.py` geeft een conclusieregel). De scoreroute blijft daar 500 geven.

**Moet #530 bijgewerkt worden?** Niet nodig: #530 is correct en weigert Gemma netjes. Na een geslaagde servertoets kan `4f7b3cbb` op de PR-tak (fast-forward van `feat/first-token-logprobs`) of als vervolg-PR; besluit Jeroen.

## Bevinding 2: tracebacks en niet-200's op alle modellen

| Wat | Waar | Oorzaak | Oordeel |
|---|---|---|---|
| 1 × HTTP 400 op main (q9b, Bonsai, Gemma) | eerste-token-logprobs | main kent eerste-token-logprobs niet | Verwacht, meetscript |
| 1 × HTTP 500 op main en int (Gemma) | scoreroute | upstreamfout, zie bevinding 1 | Bestaand gedrag |
| 1 × HTTP 400 op int (Gemma) | eerste-token-logprobs | bewuste weigering #530 | Bestaand gedrag van onze PR |
| 2 tracebacks per Gemma-run | serverlog | dezelfde 500, twee keer gelogd (`unhandled server error` en uvicorn) | Bestaand gedrag |
| 1 traceback per run, alle 12 runs | serverlog, bij het stoppen | `CancelledError: Task cancelled, timeout graceful shutdown exceeded` in `listen_for_disconnect` van een `StreamingResponse` | Buiten de meting, zie hieronder |

De laatste traceback komt pas bij `mtplx stop` (tijdens `meet.py` telde `runall.log` 0 tracebacks). Het is een open stream op een niet-generatiepad: de traceback loopt via de tak voor andere paden van `_SmartFanArrivalMiddleware` (`openai.py:25278` op int, `:25026` op main), niet via de generatietak. `meet.py` opent alleen POST-streams op generatiepaden en sluit ze na `[DONE]`. Waarschijnlijk een open SSE-verbinding van buiten de meting, bijvoorbeeld `/v1/mtplx/metrics/stream` vanuit het dashboard of de chat-UI in een browsertab, die na elke herstart opnieuw verbindt. Niet bewezen: de server logt geen toegangsregels. Onschuldig voor de meting (GET-verzoeken tellen niet in `requests_completed`). Nieuwe vondst 58.

## Bevinding 3: Qwen3.5-9B, gretige teksten 5/12 gelijk

- **Eerste token gelijk, afwijking later.** In alle 7 afwijkende teksten zijn het eerste token en de eerste 56 tot 387 tekens gelijk; daarna een andere, even plausibele woordkeus (bijvoorbeeld `*   Topic:` tegen `*   **Topic:**`, "lighthouse stories" tegen "stories about lighthouses"). Binnen elke variant zijn de teksten bitgelijk.
- **De prefill rekent anders.** De scoreroute (alleen prefill) geeft op int op 51 van de 60 prompts andere kansen dan main (maximaal 0,033), bij dezelfde oordelen. Op A3B was de top-K-fix (#532) bitgelijk ([eindbenchmark](2026-09-26-eindbenchmark.md)); de enige andere wijziging in de prefill is de lane: `QuantizedLinear` als batchmatmul van minstens 33 rijen, SDPA altijd gefuseerd, en de in-forward GDN-grenzen. Decode en verify houden de stock-kernels (`batch_invariant_prefill.py`, docstring).
- **Mechanisme.** Een andere afronding in de prefill geeft een iets andere KV-cache; een gretige keuze tussen twee bijna gelijke tokens valt dan na enkele tientallen tokens anders uit. Zelfde patroon als op A3B (8/24 gelijk met de lane, [batch-invariante-router](2026-09-26-batch-invariante-router.md)), hier zonder MoE-routering: de 9B is dicht.
- **Andere fixes.** Op A3B waren alle overige fixes samen ("kaal") voor decode en chat bitgelijk aan 2.12.0. Voor de 9B is dat niet apart gemeten; daarom de extra meting.

**Extra meting (klaar, niet gedraaid):** q9b op de integratietak met de lane expliciet uit, 2 runs:

```zsh
cd ~/Dev/laya-nl/modeltest && zsh ./runall.zsh --alleen q9b-zonder-lane > runall-q9b-zonder-lane.log 2>&1; tail -30 runall-q9b-zonder-lane.log
```

Nieuw in `runall.zsh` (`--alleen`), `serve.zsh` (varianten `intnl` en `ftl`), `modellen.zsh` en `analyse.py` (secties "Extra" achter de samenvatting, met conclusieregel). Uitsluitsel: gretig main/intnl 12/12 en scoreroute exact gelijk betekent dat de lane het verschil geeft; anders verandert een andere fix de rekenpaden en volgt bisecten. Duur ongeveer 5 minuten, poort 8000, akkoord Jeroen nodig.

**Kwaliteit** (voorlopig, alleen de afwijkplaatsen bekeken, niet de hele teksten): andere, maar even plausibele formuleringen; nauwkeurigheid van de classificatie gelijk (0,667).

## Wat de lane-gate verandert voor Bonsai

`49c4a6ee` (`perf/invariant-lane-gate`) controleert het model vóór de installatie. Bonsai gebruikt `HadamardQuantizedLinear` (Prism), die de lane niet kan omzetten; de gate weigert met `unsupported_linear:HadamardQuantizedLinear`, installeert niets (geen lineaire lagen, geen attentiehook, geen in-forward GDN-grenzen) en meldt de reden in `/health` onder `degradation.batch_invariant_prefill`. In deze run (op `198e7194`) had Bonsai een halve lane: 8 lineaire lagen om, de Hadamard-projecties niet, plus 48 GDN-lagen met in-forward-grenzen.

Gevolg (verwacht, niet gemeten): Bonsai rekent de prefill op int weer als op main, dus de scoreroute zou bitgelijk aan main moeten worden (nu 59/60, maximaal 0,0036). De gemeten winst van −8,7% op de scoreroute, −11,6% gretige TTFT en −1,1 GiB MLX-piek kan deels uit de halve lane en de in-forward-grenzen komen; hoeveel daarvan blijft, is niet gemeten. q9b en A3B worden door de gate toegelaten; Gemma meldt `not_reached`.

## Vervolg

- Extra meting q9b zonder lane (commando hierboven), daarna oordeel q9b definitief.
- Servertoets Gemma-eerste-token (`--alleen gemma-logprobs`), daarna besluit over #530.
- Eventueel Bonsai op `49c4a6ee` opnieuw meten (int met `MTPLX_BATCH_INVARIANT_PREFILL=1`, gate weigert) om de winst zonder lane te kennen.
- Voor de PR's van vondst 42 (gemma4-probe), 43 (messages-ttft) en 50 (scoped chat): op de drie modellen geen regressie gezien; Gemma-chat en -agent zijn bitgelijk aan main, de agentwinst van messages-ttft is op q9b −70%.
