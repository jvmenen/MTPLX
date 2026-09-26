# TTFT op `/v1/messages` bij lange toolresultaten (26 september 2026)

Vondst 35: via `/v1/messages` duurde een agentbeurt met een lang toolresultaat (~17K tokens) 26 tot 45 s tot het eerste token, terwijl `prompt_eval_time_s` ~0,2 s was en de nieuwe tokens als `cached_tokens` telden. Via `/v1/chat/completions` kostte een vergelijkbare beurt 5 tot 10 s.

## Uitkomst

Het verschil zit niet in de route. Een goedgevormde kale toolaanroep wordt als "orphan tool-control markup" gezien, waarna de server de beurt opnieuw genereert met een extra stuurbericht. Die herhaalprompt wijkt al op token 663 af van de oorspronkelijke, dus de hele nieuwe inhoud wordt een tweede keer geprefilld. De chatmeting van [chat-encode-memo](2026-09-26-chat-encode-memo.md) zag dit niet, omdat ze een `seed` meestuurde; alle herstelpaden staan dan uit. Een Anthropic-verzoek kan geen seed meegeven.

Fix op `fix/messages-ttft`: de herhaling alleen doen als de toolparser er geen aanroep in vindt. Op de echte server: lange agentbeurten van 25-41 s naar 10-18 s, korte van 3-6 s naar 0,2-0,3 s, met exact dezelfde prompt-, cache- en tokenaantallen als de chatroute met seed.

## Route-analyse (alleen code gelezen, `origin/main` 1de2b1c0)

Regelnummers in `mtplx/server/openai.py` tenzij anders vermeld.

- **Eén handler.** `/v1/messages` (37038) zet het verzoek om met `_anthropic_to_chat_request` (6139; een `tool_result` wordt een bericht met rol `tool`, 5950-6031) en roept `chat_completions` aan (37045). De SSE wordt achteraf vertaald (6367): `message_start` gaat direct weg, keep-alives tijdens de prefill worden lege `thinking_delta`'s. Encode (32472), sessiekoppeling, postcommit-wachten, restore en prefill zijn voor beide routes dezelfde code.
- **Sessie.** Zonder sessieheader koppelt `resolve_session_id` (`engine_session.py:1909`) via de prompt: `longest_prefix`, anders `common_prefix_reuse`, pas daarna een nieuwe anonieme sessie. Gemeten: alle vervolgverzoeken `longest_prefix`.
- **Postcommit-wachten** (vroeg 32571, laat 33404; `engine_session.py:1331`): een lopende commit krijgt standaard hoogstens 0,6 s, marathonbescherming staat uit. Gemeten: hoogstens 0,02 ms per verzoek.
- **Seed.** `AnthropicMessagesRequest` (1944) heeft geen seedveld, dus `request.seed` is op deze route altijd `None`. Vijf herstelpaden in de streamworker doen niets als er een seed is (33615, 33710, 33874, 34070, 34246).
- **Valse degeneratie.** `_tool_fed_degenerate_completion_reason` (7300) haalt alle tooltags weg; blijft er iets zonder letters over, of één woord zonder spaties van hoogstens 160 tekens (7285-7297), dan heet de uitvoer `orphan_tool_control_markup`. Bij `count_lines` met `part=2` blijft `2` over; bij een aanroep met één pad blijft het pad over. Het redeneerherstel en het "stalled promise"-pad kijken eerst of de toolparser een aanroep vindt (33926, 34091); dit pad niet.
- **Herhaalpoging.** `maybe_retry_degenerate_tool_fed_empty_completion` (33710) voegt een user-bericht toe ("Complete the active coding task now. ...") en encodeert opnieuw (33749). Met scoped redeneergeschiedenis verschuift dat de laatste gebruikersvraag, zodat alle assistentbeurten van de toolloop anders worden gerenderd: de herhaalprompt wijkt op token 663 af (CPU-controle met de echte tokenizer). Dit is de "tweede encode in de streamworker" uit het chat-encode-memo-rapport; het is geen postcommit.
- **Redeneerherstel.** Stopt poging 2 midden in het denken (bij een lage `max_tokens`), dan maakt `_repair_reasoning_only_completion_unguarded` (33874) een derde poging van de *oorspronkelijke* prompt plus de tokens van poging 2 plus `\n</think>\n\n` (28484). Die herstelt de store-on-prefill van poging 1.
- **Statistiek.** `prompt_eval_time_s`, `cached_tokens` en `new_prefill_tokens` beschrijven alleen de laatste poging; `ttft_s` loopt van binnenkomst tot het eerste token van de laatste poging (25859, 16582). Vandaar `prompt_tokens` 27005 = 26954 + 48 + 3, `cached_tokens` 26954 en `prompt_eval_time_s` 0,16 s bij een TTFT van 25 s. De client zag zijn eerste inhoud ~0,7 s vóór `ttft_s`: het afgekapte denkwerk van poging 2.

## Meting op de echte server

Gemeten, niet geschat. Apple M5 Pro 64 GB, Qwen3.6-35B-A3B MTPLX-Optimized-Balance, productie-instellingen uit `mtplx-originele-args.txt` (profiel turbo, fanmodus default, `--preserve-thinking auto`, toolmodus hybrid), eigen map voor de SSD-cache per run, verse server per variant. Code: `main` 1de2b1c0 en `fix/messages-ttft` daarop. Zelf gegenereerde neutrale tekst (archiefnotities uit vaste woordlijsten, vaste seed): een agentgesprek met twee tools, afwisselend een toolresultaat van ~17K tokens en een kort, 10 verzoeken van 10K tot 77K tokens, voor beide routes hetzelfde gesprek. Greedy, streaming. Pogingen per verzoek gemeten met een meet-hook in de server (sitecustomize rond `_run_generation_dispatched`, `_encode_messages`, de postcommits en het postcommit-wachten; alleen voor de meting).

| Variant | Route | Seed | `max_tokens` | Code |
|---|---|---|---|---|
| A | messages | geen | 48 | main |
| B | chat | geen | 48 | main |
| C | chat | 7 | 48 | main |
| D | messages | geen | 1024 | main |
| FA | messages | geen | 48 | fix |
| FD | messages | geen | 1024 | fix |

TTFT volgens de server (`ttft_s`, seconden):

| Verzoek | Tokens | A | B | C | D | FA | FD |
|---|---|---|---|---|---|---|---|
| 0 lang | 10.232 | 4,7 | 7,1 | 4,8 | 5,3 | 4,7 | 4,8 |
| 1 kort | 10.285 | 0,2 | 0,3 | 0,2 | 0,2 | 0,2 | (26,5)* |
| 2 lang | 26.954 | 25,3 | 33,1 | 10,3 | 30,0 | 9,8 | 11,3 |
| 3 kort | 27.008 | 3,4 | 3,8 | 0,2 | 3,2 | 0,2 | 0,2 |
| 4 lang | 44.499 | 28,6 | 35,0 | 14,6 | 32,9 | 12,8 | 15,0 |
| 5 kort | 44.553 | 4,8 | 6,1 | 0,3 | 19,8 | 0,2 | 0,3 |
| 6 lang | 60.949 | 34,2 | 42,5 | 17,9 | 67,6 | 15,2 | 17,9 |
| 7 kort | 61.003 | 5,9 | 7,3 | 0,3 | 19,3 | 0,3 | 0,3 |
| 8 lang | 77.323 | 41,2 | 48,5 | 21,1 | 86,6 | 18,2 | 19,8 |
| 9 kort | 77.377 | 4,5 | 5,0 | 0,4 | 25,0 | 0,3 | 0,4 |

\* Tijdens FD stuurden andere clients (het platform dat poort 8000 gebruikt) eigen verzoeken naar de testserver; dit verzoek stond daarachter in de wachtrij. De eerste FD-run was op dezelfde manier verstoord (verzoeken 8 en 9) en is vervangen. De hook toont voor beide FD-runs één poging per verzoek, met dezelfde prefillgroottes als C. De andere runs hadden geen vreemd verkeer.

Wat de hook per verzoek liet zien:

- **A (messages, main):** vanaf het eerste verzoek met een eerdere toolaanroep drie pogingen. Poging 1: gewone prefill van het nieuwe deel (bijvoorbeeld 16.669 tokens, 9,7 s), uitvoer een kale toolaanroep van 29 tokens, `orphan_tool_control_markup`. Poging 2 (`chat.stream.tool_fed_empty_retry`): de eerste keer 27.013 tokens vanaf nul (14,2 s), daarna ~17,9K tokens (13,7-20,0 s), omdat alleen de herhaalprompt van het vorige verzoek een lang begin deelt; stopt bij 48 tokens in het denken. Poging 3 (`chat.stream.reasoning_completion_repair`): 51 nieuwe tokens, 0,16-0,24 s. De TTFT ligt mediaan 10,3 s boven het eerste token van poging 1. Korte beurten: poging 2 herstelt op een 512-grens (bijvoorbeeld 24.576 of 40.960) en prefillt 1,6-3,7K tokens (1,9-4,0 s).
- **B (chat zonder seed):** hetzelfde patroon met dezelfde token- en cache-aantallen als A. Het routeverschil uit vondst 35 is dus een meetartefact. B was over de hele run 20-40% trager dan A (ook verzoek 0, dat geen herstelpad raakt), vermoedelijk warmte.
- **C (chat met seed):** één poging per verzoek, alleen de prefill van het nieuwe deel.
- **D (messages, `max_tokens` 1024):** geen afgekapte denkbeurt bij lange verzoeken, dus twee pogingen in plaats van drie, maar de herhaalprompt vond bij verzoek 2, 6 en 8 niets bruikbaars in de bank (`cached_tokens` 0) en prefillde de hele prompt: 61K tokens in 48,8 s, 77K in 65,6 s. Bij de korte beurten 5, 7 en 9 dacht poging 2 1024 tokens lang (14-19 s) en volgde nog een redeneerherstel. Met een realistische limiet is het effect dus groter, niet kleiner.
- **FA en FD (fix):** één poging per verzoek, geen encode meer in de streamworker. FA heeft per verzoek exact dezelfde `prompt_tokens`, `cached_tokens` en `completion_tokens` als C, en de toolaanroep wordt gewoon als `tool_use` doorgegeven.
- **Uitgesloten:** postcommit-wachten (≤ 0,02 ms), wachten op het model (`lock_wait_time_s` µs), tijd vóór de eerste poging (0,06-0,15 s, groeit met de encode), sessie (altijd `longest_prefix`).

## De fix

Tak `fix/messages-ttft` op `origin/main` 1de2b1c0:

- `_tool_fed_retry_is_parsed_call` naast `_tool_fed_degenerate_completion_reason`: bij de reden `orphan_tool_control_markup` eerst de toolparser (`omlx_extract_tool_calls_with_thinking`, zoals de twee andere herstelpaden al doen); vindt die een aanroep, dan geen herhaling. Lege uitvoer en losse staarten zonder aanroep worden nog steeds herhaald.
- Schakelaar `MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD` (standaard aan, `0` herstelt het oude gedrag).
- Vier tests in `tests/test_tool_fed_retry_parsed_call.py`: de heuristiek zelf is ongewijzigd; kale goedgevormde aanroep wordt niet herhaald en komt als toolaanroep door; met de schakelaar uit wel herhaald; een losse staart zonder aanroep wordt nog herhaald. De tweede test faalt zonder de wijziging.

Controles:

- Volledige testsuite (venv van `~/Dev/MTPLX`, `PYTHONPATH` naar de worktree): 19 mislukt, precies de nulmeting (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`); de vier nieuwe tests slagen.
- Ruff: 2.665 meldingen, gelijk aan `main`; het nieuwe testbestand is schoon.
- `python -m build` en `scripts/fresh_venv_smoke.sh`: geslaagd.
- CHANGELOG-regel onder `## [Unreleased]` met de meting.

## Samenhang met andere vondsten

- **Blank retries** ([completions-overhead](2026-09-26-completions-overhead.md)): gelden alleen zonder streaming (`max_attempts` is 1 bij een stream), dus niet voor deze metingen of voor de Agent SDK, die streamt.
- **Gemma-vocabulairecontrole** (~47 ms per chatverzoek): geldt voor beide routes gelijk en verklaart hier niets.

## Nevenbevindingen (niet in deze fix)

- Het redeneerherstel na een herhaalpoging plakt de tokens van poging 2, gegenereerd onder de herhaalprompt, achter de oorspronkelijke prompt (`openai.py:33947-33950`).
- De herhaalpoging voegt een stuurbericht aan het gesprek toe, terwijl #282 voor de agent-endpoints doorgeven zonder herschrijven belooft (`_agent_rewrites_mode`, 11609); `MTPLX_AGENT_REWRITES=off` schakelt deze herhaling niet uit.
- Een herhaalprompt vindt vaak niets in de bank, omdat hij op token 663 van alle gewone prompts afwijkt; elke echte herhaling kost daardoor een (bijna) volledige prefill.
- `ttft_s` en `prompt_eval_time_s` in `mtplx_stats` beschrijven alleen de laatste poging, en de `tool_fed_empty_retry_*`-velden verdwijnen als daarna nog een redeneerherstel draait. Dat maakte deze vondst lastig te zien.

## Stand

Fix gecommit en gepusht naar de fork (`fix/messages-ttft`, 2f38106b); geen PR geopend, besluit bij Jeroen (vondst 43). Nevenbevindingen staan als vondst 44 open. Poort 8000 is na de metingen vrijgelaten.

Meetmateriaal (lokaal, niet in de fork): `~/Dev/laya-nl/messages-ttft/` met meetplan, meet-hook, start- en meetscript, analyse, CPU-controle (`divergentie.py`) en ruwe resultaten.
