# Metingen: MTPLX als classifier (25 september 2026)

Opstelling: MacBook M5 Pro 64 GB, MTPLX 2.11.3, Qwen3.6-35B-A3B Optimized-Balance, profiel turbo, diepte 2. De classifier zet een vraag met labels A tot en met E in een prompt die eindigt op `Antwoord:`, plakt er een letter achter en leest via prompt-scoring (`echo: true`, `max_tokens: 0`, `logprobs: 20`) de verdeling op de laatste positie. Testset: 240 Nederlandse chatberichten met vijf labels (bevestiging, correctie, afspraak, verzoek, oordeel).

## 1. Waar de tijd heen gaat (rustige server)

| Prompt | top-20 | top-1 |
|---|---|---|
| 18 tokens | 66 ms (9 KB) | 65 ms |
| 153 tokens | 202 ms (87 KB) | 189 ms |
| 303 tokens | 368 ms (173 KB) | 360 ms |
| 603 tokens | 666 ms (348 KB) | 657 ms |
| 1203 tokens | 1078 ms (697 KB) | 1057 ms |

- Prompt-scoring kost ongeveer 1 ms per prompttoken plus circa 50 ms vast.
- Top-1 of top-20 en de grootte van het JSON-antwoord maken vrijwel niets uit.
- Dezelfde prompt vijf keer scoren kost vijf keer ~330 ms: de scoreroute hergebruikt niets.
- De correctievraag (mediaan 475 tokens) zit op p50 ~430 ms.

## 2. Opbouw van de classifierprompt

Geteld met de Qwen-tokenizer over 1.497 berichten:

| Onderdeel | Mediaan | p90 |
|---|---|---|
| Hele prompt | 475 | 543 |
| Vast deel (systeem, vraag, vijf labels) | 221 | 221 |
| Vorig agent-antwoord (laatste 700 tekens) | 195 | 228 |
| Het bericht zelf | 23 | 88 |

20% van de prompts is 512 tokens of langer, geen enkele 1.024 of langer. Het gedeelde deel (221 tokens) komt nooit boven 512.

## 3. Hergebruik van het promptbegin

| Route | Hergebruik |
|---|---|
| Prompt-scoring (`echo`, `max_tokens: 0`) | Nee, altijd een koude prefill |
| `/v1/completions` genereren | Nee, schrijft niets in de session bank (0 entries na 180 verzoeken) |
| `/v1/chat/completions` | Ja, in stappen van 512 tokens: dezelfde prompt van 615 tokens nogmaals kost 23 ms in plaats van ~300 ms; een prompt van 1.217 tokens met ~1.024 gedeeld meldt `cached_tokens` 1024 |

Met `MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS=128` en `--ssd-session-cache-min-prefix-tokens 128` werd een gedeeld begin van 220 tokens op de chatroute toch niet hergebruikt (`cached_tokens` 0, `last_miss_reason` `ssd_prefix_miss`).

## 4. Proef met eerste-token-logprobs (monkeypatch)

Een tijdelijke patch in het geheugen (wrapper om `generation._sample_from_logits` plus HTTP-middleware) gaf de top-20 van het eerste gegenereerde token terug. Op 60 berichten:

| Variant | p50 | p90 | Nauwkeurigheid |
|---|---|---|---|
| Prompt-scoring (huidig) | 570 ms | 770 ms | 0,68 |
| `/v1/completions`, eerste-token-logprobs | 390 ms | 1.500 ms | 0,68 |
| idem, vast deel vooraan | 450 ms | 1.520 ms | 0,73 |
| `/v1/chat/completions` zonder voorgevuld `Antwoord:` | ~580 ms | ~730 ms | 0,58 |

- Zelfde oordeel als prompt-scoring in 56 van de 60 gevallen; kansen verschillen tot 0,2 door een ander rekenpad.
- De uitschieters van ~1,5 s op de genereerroute worden waarschijnlijk verklaard door blank retries (zie het rapport over eerste-token-logprobs).
- Zonder voorgevuld `Antwoord:` daalt de nauwkeurigheid op de chatroute duidelijk.

## 5. Promptvolgorde (240 berichten, prompt-scoring)

| Model | Volgorde | Nauwkeurigheid | AUROC correctie |
|---|---|---|---|
| Balance | context, bericht, vraag en labels (huidig) | **0,72** | **0,96** |
| Balance | vraag en labels vooraan | 0,70 | 0,92 |
| Speed | huidig | 0,68 | 0,95 |
| Speed | vraag en labels vooraan | 0,62 | 0,92 |

De vraag en labels direct vóór `Antwoord:` werken beter: het model heeft ze dan vers in beeld. Vooraan zetten levert op de scoreroute ook geen snelheid op, want die route hergebruikt niets.

## 6. Gelijktijdigheid en geheugen

- `scheduler.config.max_active_requests = 1`, `decode_batch_max = 1`: één verzoek tegelijk, de rest wacht (gemeten wachttijd p95 6,7 s, maximaal 101 s bij gelijktijdig gebruik door agents en bulkwerk).
- Het plafond voor de session bank daalde naar de ondergrens van 1 GiB (`bank_dynamic_ceiling` in `memory_plan.py`): 48 GiB Metal-limiet min 27,6 GiB gewichten min een piekreserve die meegroeit met de hoogste piek sinds de start (tot de helft van de speelruimte) min het werkgeheugen.
- HTTP 507 ("Insufficient Storage") trad op als een tweede GPU-proces (PyTorch op MPS) tegelijk draaide.
