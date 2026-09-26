# Chat-encode-memo per segment (26 september 2026)

Tak `feat/chat-encode-segment-memo` (gebaseerd op `main` 1de2b1c0). Eerste versie `6b3bbf54` (25 september, alleen CPU); na de sleutelaanpassing en de servermeting van 26 september is de tak één commit: `9dd094d8`, gepusht naar de fork. De stand van 26 september staat in de laatste sectie.

## Probleem

Bij lange agentgesprekken (via `/v1/messages` naar de chatroute) werd elke beurt het hele gerenderde gesprek opnieuw getokeniseerd. `ChatEncodeCache` gebruikt een hash van de hele payload en mist dus bij elke nieuwe beurt, terwijl de gesprekshistorie al per segment wordt ge-encodeerd (`_encode_rendered_chat_text_segmented`).

## Wat er veranderd is

- `mtplx/chat_encode_cache.py`: nieuwe klasse `ChatSegmentEncodeMemo` (`enabled`, `make_key`, `get`, `put`, `stats`, `_forget`, `_evict_to_bounds`), hulpfunctie `_env_int`, globale `GLOBAL_CHAT_SEGMENT_MEMO`.
- `mtplx/server/openai.py`: hulpfunctie `_chat_segment_encoder(tokenizer, template_observability)`; `_encode_rendered_chat_text_segmented` encodeert elk segment via die hulpfunctie; vijf aanroepen geven `template_observability` door; het postcommit-pad gebruikt de memo zonder tellers.
- Nieuw `tests/test_chat_segment_memo.py` (18 tests), CHANGELOG onder Unreleased / Changed.

## Waarom het exact is

De gesegmenteerde encoder bouwde zijn uitvoer al door elk segment los te encoderen en de id's samen te voegen; de memo bewaart precies die resultaten. `add_special_tokens` staat vast op False. Sleutel: identiteit van de tokenizer (bestaande `_chat_encode_tokenizer_key`: willekeurige id per tokenizerinstantie plus hash van de chattemplate) plus SHA-256 van de exacte segmenttekst. Een herladen model krijgt een nieuwe id; een aangepast eerder bericht krijgt een andere sleutel. Tokenizers zonder weakref slaan de memo over.

## Grenzen en geheugen

LRU van maximaal 1.048.576 tokens en 4.096 entries; id's als int32 (4 bytes per token), sleutels als hex-digest, geen tekst bewaard: ~4 MB id's plus hooguit ~1 MB overhead, genoeg voor ~13 gesprekken van 78K tokens. Een segment groter dan het hele budget wordt niet opgeslagen. Thread-safe: dict-bewerkingen onder een lock, tokeniseren erbuiten.

## Schakelaars en observability

`MTPLX_CHAT_SEGMENT_MEMO=off` zet hem uit (standaard aan); `MTPLX_CHAT_SEGMENT_MEMO_TOKENS` zet het budget. Per verzoek `template_observability["chat_segment_memo"] = {hits, misses, reused_tokens}`; `stats()` geeft de totalen.

## Meting

Apple M5 Pro 64 GB, alleen CPU, tokenizer en chattemplate van Qwen3.6-35B-A3B Balance, synthetisch groeiend agentgesprek met 8 tools en bronbestanden als toolresultaten, mediaan van 7, 25 september.

| Tokens | Tokeniseren uit | koud | warm | Volledige encode incl. template uit → warm |
|---|---|---|---|---|
| 21,8K | 15,4 ms | 16,0 ms | 1,7 ms | 58,6 → 42,3 ms |
| 40,0K | 27,0 ms | 27,3 ms | 1,6 ms | 69,4 → 43,8 ms |
| 78,3K | 51,9 ms | 53,1 ms | 2,3 ms | 95,7 → 46,0 ms |

"Warm": de vorige beurt staat in de memo en alleen het nieuwe segment mist. De volledige-encodetijden schommelen tussen runs (eerder 70/84/112 ms uit tegen 54/54/66 ms warm). De tijd tot het eerste token op een echte server is niet gemeten.

## Tests

39 bestanden (alles rond encode-cache, gesegmenteerde encoder, chat_template, `_encode_messages`, `template_observability`, plus `test_no_mlx_imports` en `test_forge_cli`): 1.812 tests, 19 mislukt (precies de nulmeting), 35 overgeslagen. Pariteit met de echte tokenizer over een groeiend gesprek met toolaanroepen, denkblokken, unicode (inclusief zero-width space), herhaalde segmenten, een heel lang segment en een aanpassing eerder in het gesprek, in hybrid- en compact-modus, met denken aan en uit. Ruff: nieuwe bestanden schoon, `openai.py` gelijk aan voorheen.

## Risico's en niet gedaan

- Als een tokenizer ooit ter plekke van woordenschat verandert zonder templatewijziging, merkt de sleutel dat niet; `len(tokenizer)` aan de sleutel toevoegen dekt dat af.
- Bij een hit op de hele-verzoek-cache worden de oude memo-tellers herhaald; `chat_encode_cache: "hit"` blijft het signaal dat telt.
- Niet-gesegmenteerde transcripten (denken uit, platte route) profiteren niet.
- **De Jinja-template-render (~40 ms bij 78K tokens) domineert nu de encodetijd**; incrementeel renderen is een grotere wijziging.
- Geen tijdteller, alleen hits, misses en hergebruikte tokens. Volledige suite niet gedraaid.

## Afronding voor de PR (26 september 2026)

### Wat er nog veranderd is

- `origin/main` stond nog op 1de2b1c0; een rebase was niet nodig.
- Vondst 27: de memo-sleutel bevat nu ook de woordenschat. Niet `len(tokenizer)`: dat kost op de Qwen3.6-tokenizer (248.077 tokens) **11 ms per aanroep** (de Rust-aanroep met toegevoegde tokens bouwt de hele woordenschat op), en mlx-lm's `TokenizerWrapper` heeft geen `__len__`. In plaats daarvan `vocab_size` plus het aantal toegevoegde tokens (`added_tokens_decoder`), samen ~2,6 µs; de wrapper geeft beide door. Hulpfunctie `_chat_segment_vocab_size` in `mtplx/server/openai.py`; drie nieuwe tests (tokens ter plekke toegevoegd, ook via een wrapper), die zonder de wijziging falen.
- CHANGELOG-regel herschreven met de servermeting hieronder; commit en commitbericht bijgewerkt (`9dd094d8`).

### Controles

- Volledige testsuite (alle 481 bestanden, venv van `~/Dev/MTPLX`, `PYTHONPATH` naar de worktree): 19 mislukt, precies de nulmeting (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`); `test_chat_segment_memo.py` 21 geslaagd.
- Ruff: 3.281 meldingen, gelijk aan `main`; het nieuwe testbestand is schoon.
- `python -m build` en `scripts/fresh_venv_smoke.sh`: geslaagd.

### Meting op de echte server

Gemeten, niet geschat. Apple M5 Pro 64 GB, Qwen3.6-35B-A3B MTPLX-Optimized-Balance, productie-instellingen uit `mtplx-originele-args.txt` (profiel turbo, fanmodus default, `--preserve-thinking auto`, toolmodus hybrid), met alleen een eigen map voor de SSD-cache per run zodat runs elkaars cache niet zien. Verse server per run; `main` vanuit een tijdelijke worktree op 1de2b1c0, de tak vanuit `~/Dev/MTPLX-encode`. Zelf gegenereerde neutrale tekst (archiefnotities uit vaste woordlijsten met een vaste seed), greedy, 48 tokens per antwoord. Encodetijd per verzoek gemeten met een meet-hook in de server (sitecustomize die `_encode_messages` timet, alleen voor de meting), TTFT en `prompt_eval_time_s` uit `mtplx_stats`.

**Eerste bevinding: bij gewone chat doet de memo niets.** De server draait met scoped redeneergeschiedenis (`scoped_reasoning_history=True`). Dan rendert de template afgeronde assistentbeurten zonder denkblok, zijn er geen generatienaden en wordt het hele gesprek in één keer getokeniseerd; de memo wordt niet aangesproken. Gemeten op een groeiend gesprek zonder tools (10K tot 80K tokens, 20 verzoeken via `/v1/chat/completions` en 10 via `/v1/messages`, twee runs per variant): encodetijd gelijk (bij 80K 112 tegen 108 ms, ruis), uitvoer exact gelijk. De eerdere CPU-meting van 25 september gebruikte scoped uit en gaf daardoor een te rooskleurig beeld voor gewone chat.

**Agentgesprekken met toolaanroepen worden in scoped modus wel gesegmenteerd.** Daarom het echte scenario: een agentgesprek met twee tools, afwisselend een lang toolresultaat (~8K tokens) en een kort, van 10K naar 82K tokens (20 verzoeken via `/v1/chat/completions`), plus 10 verzoeken via `/v1/messages` (toolresultaten van ~17K tokens, tot 77K). Drie runs per variant (volgorde main, tak, main, tak, tak, main). Medianen over de drie runs, chatroute:

| Tokens | Soort | `_encode_messages` main → tak | TTFT min prefill main → tak | TTFT main → tak |
|---|---|---|---|---|
| 10.338 | kort | 55,0 → 48,0 ms | 67 → 59 ms | 219 → 207 ms |
| 26.557 | lang | 66,8 → 52,5 ms | 92 → 75 ms | 5.848 → 5.829 ms |
| 42.461 | lang | 83,4 → 62,2 ms | 127 → 97 ms | 6.738 → 6.922 ms |
| 58.051 | kort | 95,7 → 49,5 ms | 143 → 96 ms | 288 → 237 ms |
| 74.080 | kort | 104,7 → 54,1 ms | 163 → 110 ms | 414 → 367 ms |
| 82.251 | lang | 107,1 → 57,4 ms | 189 → 133 ms | 10.135 → 10.376 ms |

- "TTFT min prefill" is `ttft_s − prompt_eval_time_s`: de tijd vóór en rond de prefill, waar de encode in valt. Vanaf 42K tokens 19 tot 56 ms lager (mediaan 42 ms).
- De memo raakte op de server zoals bedoeld: per verzoek alle eerdere segmenten als hit, één miss (het nieuwe toolresultaat).
- Offline (alleen CPU, dezelfde argumenten als de server) splitst het zich zo: tokeniseren bij 82K van 57,8 naar 5,8 ms (lange beurt) of 0,1 ms (korte beurt); de rest van de encode (~43 ms) is de Jinja-render (vondst 26).
- **TTFT van lange beurten is geen goede maat**: de prefill werd over opeenvolgende runs 10 tot 20% trager (warmte; prefill bij 82K 9,5 s in de eerste run, 11,3 s in de zesde). De omgekeerde volgorde in het laatste paar bevestigt dat het volgorde is en geen effect van de tak; de wijziging raakt de GPU-prefill niet.
- `/v1/messages`: de encode vóór de prefill daalt op dezelfde manier (bij 77K ~109 naar 50-85 ms). Daarnaast zag de meet-hook per verzoek een tweede encode in de streamworker (~110 ms bij 77K), die gelijk bleef.

  **Correctie (26 september 2026, [messages-ttft](2026-09-26-messages-ttft.md)):** die tweede encode is geen postcommit, maar de herhaalprompt van de herstelpoging na een lege of "orphan" tooluitvoer (`maybe_retry_degenerate_tool_fed_empty_completion`: de berichten plus een extra user-bericht, `allow_committed_reasoning=True`). De tokenaantallen zijn exact na te rekenen en de encode valt midden in het verzoek. Hij verdwijnt met de fix voor vondst 35.

**Exactheid:** over alle runs (6 agentruns, 4 chatruns, samen 300 verzoeken plus opwarmverzoeken) waren gegenereerde tekst, redenering, toolaanroepen, `completion_tokens`, `prompt_tokens` en `cached_tokens` per verzoek identiek tussen main en de tak.

### Stand

Klaar voor een PR na akkoord van Jeroen; de Engelse PR-tekst staat klaar. Meetmateriaal (lokaal, niet in de fork): `~/Dev/laya-nl/encode-memo/` met de gespreksgenerator, het meetscript, de meet-hook, het startscript en de ruwe resultaten.
