# Vondst 44 en 57: herstelpaden in de streamworker en scoreroute op Gemma 4

Datum: 2026-09-27. Code en tests: M5 Pro, 64 GB, venv `~/Dev/MTPLX/.venv`, `MTPLX_CONFIG=/nonexistent`. Servertoetsen op het echte model op 27 sep (18:00-18:31, als platform-script-taken): vondst 44 op Qwen3.6-35B-A3B Balance met de productie-argumenten, vondst 57 op het Gemma-4-paar (`youssofal-gemma4-mtplx-optimized-speed`, args uit `modeltest/args/gemma.args`). Regelnummers op `origin/main` 1de2b1c0 (2.12.0) tenzij anders vermeld.

## Kern

- **Vondst 44 (tak `fix/stream-recovery-paths`, fbde4931 en 71270aab): werkt op de server.** Het redeneerherstel haalt de herhaalprompt uit de bank (in alle 8 gevallen `cached_tokens` exact gelijk aan de lengte van de herhaalprompt, 51 nieuwe tokens, 0,16-0,25 s) en doet geen volledige prefill. De nieuwe statistiekvelden kloppen met de som van de pogingen. TTFT gelijk aan de nulmeting A. Advies: PR na akkoord Jeroen, (a) en (c) samen; (b) als losse commit of eerst als vraag aan de maker.
- **Vondst 57 (tak `fix/gemma4-prompt-scoring`, 45ed152b): geslaagd.** Scoreroute op Gemma 4 geeft 60 van 60 keer HTTP 200, deterministisch (ook over twee verse servers), top-1 gelijk aan het gretige eerste token. Samen met de Gemma-eerste-token-route wijkt de logprob op 9 van 60 prompts precies één bf16-stap (0,125) af; mediaan 2e-6, labeloordeel 60/60 gelijk. Advies: kleine PR na akkoord Jeroen.

## Vondst 44: herstelpaden in de streamworker

Keten in de streamworker (`mtplx/server/openai.py:34449-34463` zonder sessie, `:34508-34529` met sessie), alleen zonder `seed` en met tools: `inspection_empty_retry` → `tool_fed_empty_retry` (33710) → `reasoning_completion_repair` (33856/33874) → `read_only_force_answer_retry` → `stalled_agent_retry` (34070). Elke stap krijgt het resultaat van de vorige.

### (a) Redeneerherstel plakt de tokens van de herhaalpoging achter de oorspronkelijke prompt

- **Wat:** `_repair_reasoning_only_completion_unguarded` bouwt de herstelprompt met `prompt_ids` uit de afsluiting (33947-33951) en de tokens van `generated`. Na een `tool_fed_empty_retry` zijn die tokens gegenereerd onder de herhaalprompt (gesprek plus stuurbericht, opnieuw gerenderd, wijkt vanaf token ~663 af). `_reasoning_completion_repair_prompt_ids` (28484-28498) plakt ze achter de oorspronkelijke prompt plus `\n</think>\n\n`.
- **Fout of ontwerp:** fout in de samenstelling. Beide paden bestaan sinds 1.0.0 (`488f26dc`); het herstel is geschreven voor de eerste poging ("protocol-completion check", docstring 28466-28474). Geen CHANGELOG-regel, geen `mistakes/`-les en geen commentaar dat het bewust de oorspronkelijke prompt neemt. De herhaalpoging geeft haar prompt nergens door.
- **Kost:** het model maakt zijn denkwerk af in een context die het nooit zag (het stuurbericht ontbreekt, eerdere beurten anders gerenderd). Kwaliteitsrisico, geen tijdsverlies: het herstel vond in de meting de store-on-prefill van poging 1 terug (51 nieuwe tokens, 0,16-0,24 s). Komt voor bij elke herhaalpoging die in het denken stopt: in de messages-ttft-meting bij alle lange beurten met `max_tokens` 48 en bij drie korte beurten met `max_tokens` 1024.
- **Gebouwd** (`fbde4931`): de herhaalpoging zet haar prompt op het resultaat (`_mtplx_attempt_prompt_ids`), het herstel gaat daarvan uit. Schakelaar `MTPLX_REASONING_REPAIR_FOLLOWS_RETRY_PROMPT` (standaard aan, `0` = oud gedrag). Zonder herhaalpoging verandert er niets.
- **Open (server):** of het herstel na de wijziging de herhaalprompt uit de bank haalt. Verwacht wel: poging 2 draait met `commit_prompt_prefix_to_bank=commit_prompt_prefix`, dezelfde vlag als poging 1, en de messages-ttft-meting zag de herhaalprompt van een vorig verzoek in de bank terug. Als dat niet zo is, kost het herstel een volledige prefill.

### (b) Herhaalpoging voegt een stuurbericht toe ondanks #282

- **Wat:** `maybe_retry_degenerate_tool_fed_empty_completion` voegt een user-bericht toe ("Complete the active coding task now...", 33734-33742); `maybe_retry_stalled_agent_tool_promise` doet hetzelfde ("Continue the active coding task now...", 34115-34122). Geen van beide kijkt naar `_agent_rewrites_mode()` (11609).
- **Fout of ontwerp:** een gat in #282 (`34b8d649`, 25 aug). Het contract: "no injected steering text unless a rewrite feature is explicitly enabled" (CHANGELOG 2483-2491, `docs/releases/v2.9.2.md:5-7`) en "off is a hard passthrough guarantee" (docstring 11609-11620). #282 zette de read-only-force-answer-herhaling wel achter een schakelaar (`_read_only_force_answer_enabled`, 8865), deze twee niet; de commit noemt ze niet. Of de maker ze als "protocolherstel" wil houden in de standaardstand is onbekend.
- **Kost:** onder `MTPLX_AGENT_REWRITES=off` toch ingevoegde tekst en een tweede generatie met (bijna) volledige prefill: in de messages-ttft-meting 14-65 s extra per herhaling. Na `fix/messages-ttft` gebeurt de herhaling alleen nog bij echt lege of kapotte uitvoer.
- **Gebouwd** (`71270aab`): beide herhalingen doen niets als `MTPLX_AGENT_REWRITES=off`. Standaardstand en `on` ongewijzigd. Losse commit, zodat hij makkelijk weg kan als de maker het anders ziet.
- **Niet gebouwd:** de herhaling standaard uitzetten (verandert standaardgedrag zonder meting) en de afwijking op token ~663 (gevolg van het extra bericht met scoped redeneergeschiedenis; hoort bij een eventueel besluit over de herhaling zelf).

### (c) Statistiek beschrijft alleen de laatste poging

- **Wat:** elke poging draait `_run_generation` met eigen stats. `ttft_s` loopt van binnenkomst (`request_received_monotonic_s`, 25859) tot het eerste token van díe poging; `prompt_eval_time_s`, `cached_tokens` en `new_prefill_tokens` horen bij die poging. Het herstel start van de oorspronkelijke `request_observability` (33952), zodat de `tool_fed_empty_retry_*`-velden uit de eindstatistiek verdwijnen. `request_elapsed_s` loopt al van binnenkomst tot het laatste token en dekt dus alle pogingen. `server_attempts` (26449) telt alleen blank retries binnen één `_run_generation`.
- **Fout of ontwerp:** geen bewuste keuze te vinden; elke herstelstap is los gebouwd. Het is een meetgat, geen gedragsfout.
- **Kost:** alleen diagnose. Een beurt van 25 s toonde `prompt_eval_time_s` 0,16 s; vondst 35 en 44 waren daardoor lastig te zien.
- **Gebouwd** (`fbde4931`): de keten loopt via `_run_stream_recovery_chain`. Draaiden er meer pogingen, dan krijgt de eindstatistiek `stream_attempts`, `stream_attempts_first_ttft_s`, `stream_attempts_prompt_eval_time_s`, `stream_attempts_new_prefill_tokens` en `stream_attempts_completion_tokens` (ook in `PUBLIC_MTPLX_STATS_KEYS` en `state.last_metrics`), en blijven de herstelvelden van eerdere pogingen staan (`setdefault`). Bestaande velden houden de waarde van de laatste poging; bij één poging verandert de envelop niet.

### Controles `fix/stream-recovery-paths`

- Tak vanaf `origin/main` 1de2b1c0, worktree `~/Dev/MTPLX-recovery`, commits `fbde4931` (a en c) en `71270aab` (b), gepusht naar de fork.
- `tests/test_stream_recovery_paths.py`: 10 tests (herstel na herhaling gaat uit van de herhaalprompt; schakelaar uit geeft het oude plakken; herstel zonder herhaling ongewijzigd; totalen en behouden velden over drie pogingen; geen totalen bij één poging; publieke sleutels; de ketenhelper; onder `off` geen herhaling bij kapotte toolmarkup en bij een beloofde toolaanroep; standaardstand herhaalt nog). De twee `off`-tests falen zonder `71270aab`.
- `test_server_openai`, `test_dashboard_endpoints`, `test_final_tool_preamble_stream_parity`, `test_api_benchmark_contracts` groen.
- Volledige suite: 9391 geslaagd, 67 overgeslagen, 1 mislukt (de bekende Laguna-geheugentest).
- Ruff `openai.py`: 397 meldingen, gelijk aan main; testbestand schoon.

### Servertoets (27 sep, 18:21-18:31)

Script `~/Dev/laya-nl/messages-ttft/recovery-toets.zsh`, log `recovery-toets.log`. Werklast van [messages-ttft](2026-09-26-messages-ttft.md): één agentgesprek van 10 beurten via `/v1/messages` (10K tot 77K tokens, zonder `seed`), op een verse server. R: `max_tokens` 48, RD: `max_tokens` 1024. Nulmetingen A en D (zelfde werklast op `main`, 26 sep). Analyse: `analyse.py A R D RD` (hookregels per poging).

**1. Haalt het herstel de herhaalprompt uit de bank? Ja.**

| Run | Herstelpogingen | Herstelprompt | `cached_tokens` | Nieuw | Prefill |
|---|---|---|---|---|---|
| A (main, 26 sep) | 8 | oorspronkelijke prompt + 51 | lengte oorspronkelijke prompt (26.954 tot 77.377) | 51 | 0,16-0,24 s |
| R (tak) | 8 | herhaalprompt + 51 | lengte herhaalprompt (27.013 tot 77.401) | 51 | 0,16-0,25 s |
| RD (tak, 1024) | 3 | herhaalprompt + 1027 | lengte herhaalprompt (60.988 tot 77.352) | 1027 | 1,3-1,6 s |

- Het herstel gaat nu van de herhaalprompt uit en vindt die volledig in de bank. Geen volledige prefill. Bij RD zijn de 1027 nieuwe tokens de 1024 gegenereerde tokens van de herhaalpoging plus de afsluiting: die moeten hoe dan ook door het model.
- TTFT per beurt, R tegen A: binnen 0,1 tot 2 s gelijk (bijvoorbeeld beurt 8: 41,0 tegen 41,2 s). De tijd zit in de eerste twee pogingen, niet in het herstel.

**2. Werken de nieuwe statistiekvelden? Ja.**

- `stream_attempts` is 3 bij een herstel en 2 bij alleen een herhaalpoging; bij één poging ontbreken de velden (beurt 0 en 1).
- De totalen kloppen met de hookregels. Voorbeeld R beurt 2: `stream_attempts_prompt_eval_time_s` 26,12 = 10,60 + 15,36 + 0,16; `stream_attempts_new_prefill_tokens` 43.733 = 16.669 + 27.013 + 51; `stream_attempts_completion_tokens` 103 = 29 + 48 + 26; `stream_attempts_first_ttft_s` 10,69 (eerste poging). RD beurt 7: 65.723 = 61.003 + 3.693 + 1.027.
- De `tool_fed_empty_retry_*`-velden staan nu naast de `reasoning_completion_repair_*`-velden in de eindstatistiek; op main (A) ontbraken ze na een herstel.

**3. Kanttekening bij RD (niet toe te wijzen aan de fix).** In RD deden de eerste pogingen van beurt 7, 8 en 9 een volledige prefill (61K tot 77K tokens, 45 tot 64 s), waardoor de TTFT daar hoger is dan in D. In D gebeurde hetzelfde bij twee herhaalpogingen (beurt 6 en 8). Zonder `seed` verschillen de teksten per run (RD beurt 6 genereerde 1024 tokens, D 294), dus ook de pogingen en de bankinhoud. Beide runs lopen bij 60K tot 77K tokens tegen bankmissers aan; met één run per stand is niet te zeggen of de fix daar iets aan verandert. Een herhaling met een vaste `seed` zou dat scheiden.

**Meetomstandigheden:** de laptop draaide op de accu met een lader van 8 W. De prefill van de eerste beurt (10K tokens) duurde 5,1 s tegen 4,6 s in A (+10%). Voor de toetsen hierboven (bank-hits, tellingen) maakt dat niet uit.

**Advies:** PR na akkoord Jeroen met (a) en (c): het herstel na een herhaalpoging klopt nu en de statistiek toont alle pogingen, zonder tijdverlies. (b) (herhaalpogingen onder `MTPLX_AGENT_REWRITES=off` uit) is een contractvraag voor de maker: als losse commit in dezelfde PR met een expliciete vraag, of eerst als issue.

## Vondst 57: scoreroute op Gemma 4

- **Wat:** `score_prompt_logprobs` (`mtplx/generation.py:8115`) roept per blok `_forward_ar_optional_hidden` → `rt.forward_ar` aan (1937-1978). `Gemma4AssistantRuntime` (`mtplx/backends/gemma4_assistant.py:1444`) heeft alleen `forward_target`; gevolg `AttributeError` en HTTP 500.
- **Kan de Gemma-backend logits per positie leveren?** Ja. `Gemma4TargetAdapter.forward_with_state` (800-880) geeft de verborgen toestand van alle posities (`hidden`, vóór de norm) en rekent logits alleen als `compute_logits=True`; `logits_from_hidden` (882-893) doet norm, kop en softcap. De generatieprefill (`_gemma4_prefill_prompt`, 2534-2562) draait de hele prompt in één forward zonder logits en rekent daarna alleen de laatste rij.
- **Gebouwd** (`45ed152b`, tak `fix/gemma4-prompt-scoring`, worktree `~/Dev/MTPLX-gemmascore`, gepusht naar de fork):
  - `generation.py`: `_prompt_scoring_logit_chunks` levert `(start, end, logits)` per blok; het bestaande pad is ongewijzigd (zelfde `forward_ar` per blok op één prefillcache), `score_prompt_logprobs` gebruikt de generator en is verder gelijk.
  - `backends/gemma4_assistant.py`: `gemma4_prompt_scoring_logit_chunks` draait dezelfde forward als de generatieprefill (hele prompt, verse cache, zonder logits) en past de logitskop per blok van 256 rijen toe; zo blijft de geheugenbelofte van de scoreroute (hoogstens blok × vocabulaire aan logits) staan.
  - Geen 400 nodig: de route geeft nu echte waarden.
- **Tests** (`tests/test_gemma4_prompt_scoring.py`, 6, zonder model; causale stub-target): waarden en top-K gelijk aan een numpy-log-softmax; één forward zoals de prefill plus een kop per blok (16, 16, 8); blokgrootte verandert niets; de logprob van een gescoord label is gelijk aan die uit de generatieprefill van de prompt zonder label; het endpoint geeft 200 met de juiste waarden. Alle zes falen zonder de wijziging.
  - Terzijde: met een GPU-matmul in de stub wijken rijen ~1e-3 relatief af tussen een forward over n en n+1 posities (MLX op de M5). Op het echte model is daarom geen bitgelijkheid tussen scoreroute en eerste-token-route te verwachten; het meetscript werkt met drempels.
- **Controles:** gerelateerde tests groen (`test_generation_sustained`, `test_api_benchmark_contracts`, `test_draft_telemetry_truth`, `test_tail_gemma4_stream_holdback`, `test_gemma4_session_cache`, `test_gemma4_rotating_cache_trim`, `test_release_qa_fixwave`). Volledige suite: 9387 geslaagd, 67 overgeslagen, 1 mislukt (de bekende Laguna-geheugentest). Ruff: bronbestanden gelijk aan main, testbestand schoon.
- **Testtak voor de servertoets:** `test/gemma4-scoring-met-ftl` (lokaal, niet gepusht, worktree `~/Dev/MTPLX-gemmascore-ftl`, `706207a9`): de fix samengevoegd met `feat/first-token-logprobs-gemma4`, conflict in `score_prompt_logprobs` opgelost (generator plus de helpers `_log_softmax_f32`/`_top_k_logprobs` van #530). Gemma- en eerste-token-tests groen.

### Servertoets (27 sep, 17:58-18:21)

Script `~/Dev/laya-nl/gemma-scoring/runall.zsh`, log `runall.log`, resultaten in `res/`. Vier verse servers met de Gemma-args: `score-1/2` (alleen de fix, 45ed152b) en `score-ftl-1/2` (fix plus eerste-token-logprobs, lokale testtak `test/gemma4-scoring-met-ftl` 706207a9). Per run 60 classifierprompts in Gemma-formaat.

| Maat | score-1 | score-2 | score-ftl-1 | score-ftl-2 |
|---|---|---|---|---|
| Scoreroute HTTP 200 | 60/60 | 60/60 | 60/60 | 60/60 |
| Deterministisch (twee keer dezelfde prompt) | 60/60 | 60/60 | 60/60 | 60/60 |
| Top-1 gelijk aan eerste token | 60/60 (gretig) | 60/60 (gretig) | 60/60 (logprobs) | 60/60 (logprobs) |
| Verschil logprob gescoord token: mediaan / max | n.v.t. | n.v.t. | 1,9e-6 / 0,125 | 1,9e-6 / 0,125 |
| Labeloordeel gelijk | n.v.t. | n.v.t. | 60/60 | 60/60 |
| Gedeelde posities na inkorten: max verschil | 0,025 | 0,025 | 0,025 | 0,025 |
| Scoreroute p50 | 1,46 s | 1,50 s | 1,51 s | 1,52 s |
| Oordeel script | OK | OK | AFWIJKEND | AFWIJKEND |

- Run 1 en 2 zijn per variant op alle 60 prompts exact gelijk: deterministisch over verse servers heen.
- `requests_completed` klopt: 240 per run (bij `score` 300 verstuurd, waarvan 60 eerste-token-verzoeken met de verwachte 400 die niet meetellen).
- **"AFWIJKEND" bij de ftl-variant** komt door de drempel voor het maximum (0,1). Op 9 van 60 prompts verschilt de logprob van het gescoorde token 0,122 tot 0,125, de andere 51 liggen onder 0,0003 (18 exact gelijk). 0,125 is precies één bf16-stap bij logits tussen 16 en 32 (Gemma kapt af op 30): de scoreroute rekent over n posities, de eerste-token-route over n − 1, en die andere forwardvorm rondt een enkele logit één stap anders af (zie de terzijde hierboven). Het labeloordeel en de top-1 zijn op alle 60 gelijk. Dit is geen fout in de fix; de drempel van 0,1 was voor bf16 te streng gekozen.

**Advies:** kleine PR na akkoord Jeroen (`fix/gemma4-prompt-scoring`, 45ed152b, eerst bijwerken op `main`). De testtak met eerste-token-logprobs blijft lokaal; die combinatie hoort bij PR #543.
