# Chat-encode-memo per segment (26 september 2026)

Tak `feat/chat-encode-segment-memo` (gebaseerd op `main` 1de2b1c0), commit `6b3bbf54`, niet gepusht. Gebouwd door een Opus-agent, alleen CPU, geen server of model geladen.

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
