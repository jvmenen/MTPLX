# Overhead buiten de servermeting bij korte verzoeken (26 september 2026)

Vraag: waar blijft ~0,8 s per verzoek bij korte `/v1/completions`-verzoeken, die buiten `mtplx_stats.elapsed_s` valt (client 1,06 s, `elapsed_s` 0,26 s)? Hangt samen met vondst 2 (uitschieters van ~1,5 s) en vondst 25 (vaste overhead van ~50 ms).

Werktak `onderzoek-completions-overhead` (de naam `onderzoek/...` kan niet naast de tak `onderzoek`), worktree `~/Dev/MTPLX-overhead`. Fixtakken `fix/completions-overhead` (b921b9a6) en `fix/gemma4-probe-vocab` (2a1057c7), beide vanaf `main` 1de2b1c0 en gepusht naar de fork.

## Uitkomst

- **De 0,8 s zijn drie blank retries.** De prompt eindigt op `Antwoord:` en het gretige eerste token is `\n\n`. Die tekst is leeg na `strip()`, dus `_run_generation` genereert opnieuw met een nieuwe seed, tot 3 keer (`--blank-retry-attempts`). Bij `temperature: 0` doet de seed niets: vier keer dezelfde prefill van ~265 ms. `mtplx_stats.elapsed_s` is de enginetijd van alleen de laatste poging; `request_elapsed_s` (1,058 s) en `server_attempts: 4` stonden wel in het antwoord. Dit verklaart ook vondst 2.
- **Bijvangst:** `usage.completion_tokens` telde de weggegooide pogingen mee (4 bij `max_tokens: 1`), omdat de tokentijdstempels per poging bleven oplopen.
- **De ~50 ms van vondst 25 zit in de chatroute, niet in completions:** `is_gemma4_tokenizer` roept bij elk chatverzoek `tokenizer.get_vocab()` aan, en dat bouwt een dict van alle 248k tokens: ~47 ms. Buiten de generatie kost een completionsverzoek maar ~3 ms.

## De route (code gelezen, `main` 1de2b1c0)

- Handler `completions` in `mtplx/server/openai.py:37108`: tokeniseren (`_encode_prompt`), postcommit-sweep via `asyncio.to_thread`, `resolve_request_policy`, dan prompt-scoring (`echo`+`logprobs`, 37189), streaming (37209) of niet-streaming (37575 → `asyncio.to_thread(_run_generation_dispatched)`).
- Completions zet geen `request_received_monotonic_s`, dus `started` in `_run_generation` (25859) is het begin van de generatie; de chatroute zet hem wel (32258, 32766).
- Blank retry: `max_attempts` (25875), lus (25895), stopvoorwaarde `seed_is_explicit or out.text.strip()` (26485). Streaming, logprobs en ook niet-streaming verzoeken met `stop` (die geven een `token_callback` mee en tellen dan als streaming) doen geen retry.
- `elapsed_s` in `mtplx_stats` is de enginetijd (`GenerationStats`) van de laatste poging; `request_elapsed_s`/`server_elapsed_s` lopen vanaf `started` over alle pogingen (26196).
- Tokentelling: `_effective_completion_tokens` (16786) neemt het maximum van gegenereerde tokens en tokentijdstempels; `token_times` werd per poging niet geleegd.
- `is_gemma4_tokenizer` in `mtplx/chat_encoding.py:23`, aangeroepen vanuit `_encode_messages_uncached` (openai.py:14916) bij elk chatverzoek.

## Meting

Gemeten op de echte server: M5 Pro 64 GB, Qwen3.6-35B-A3B Balance, profiel turbo, productie-args, fanmodus default, 26 september. Neutrale gegenereerde tekst, prompt van 384 tokens eindigend op `Antwoord:`, `temperature: 0`, 3 opwarmrondes, daarna 10 herhalingen per variant. Tijdstempels per fase via een tijdelijke `sitecustomize` (buiten de repo) die `generate_mtpk`, de FastAPI-endpointaanroep en de uvicorn-ontvangst/verzending omwikkelt; beide processen gebruiken dezelfde `perf_counter`-klok.

### `main` (mediaan / p90, ms)

| Variant | Client | `elapsed_s` | `request_elapsed_s` | Tot start generatie | Generatie totaal | Pogingen | `completion_tokens` |
|---|---|---|---|---|---|---|---|
| completions, `max_tokens` 1 | 1061 / 1076 | 262 / 263 | 1058 / 1074 | 1 | 1056 | 4 | 4 |
| idem met `seed` | 272 / 284 | 269 / 281 | 270 / 282 | 1 | 269 | 1 | 1 |
| `max_tokens` 2 | 297 / 308 | 293 / 305 | 294 / 305 | 1 | 294 | 1 | 2 |
| `max_tokens` 16 | 480 / 483 | 476 / 480 | 477 / 481 | 1 | 477 | 1 | 16 |
| `max_tokens` 16 met `seed` | 480 / 498 | 476 / 494 | 477 / 495 | 1 | 477 | 1 | 16 |
| `max_tokens` 1 met `stop` | 274 / 286 | 271 / 283 | 271 / 283 | 2 | 271 | 1 | 1 |
| `max_tokens` 16 met `stop` | 480 / 495 | 481 / 481 | 481 / 481 | 1 | 477 | 1 | 15-16 |
| streaming, `max_tokens` 1 | 272 / 276 | 269 / 273 | 270 / 273 | 2 | 270 | 1 | 1 |
| streaming, `max_tokens` 16 | 484 / 490 | 480 / 487 | 481 / 487 | 2 | 481 | 1 | 16 |
| chat, `max_tokens` 1 | 509 / 519 | 430 / 439 | 505 / 516 | 58 | 446 | 1 | 1 |
| prompt-scoring (`echo`, `logprobs` 5) | 443 / 448 | - | - | 1 | 439 | - | 0 |

- Bij `max_tokens` 2 en 16 is de tekst `\n\n<think>...`, dus niet leeg: geen retry. Alleen de 1-tokenvariant zonder seed en zonder `stop` raakt de retry.
- Buiten de generatie kost completions 2-4 ms (ontvangst, handler, antwoord opbouwen en versturen).
- Chat: 51-58 ms tussen binnenkomst en start van de generatie. Fasetijdstempels van één verzoek: `_encode_messages` 48,3 ms, de rest van de handler samen ~1 ms. Los gemeten op de tokenizer (alleen CPU): `is_gemma4_tokenizer` 47-49 ms, `get_vocab()` 45-47 ms.

### Na de fixes

| Variant | `main` | `fix/completions-overhead` | `fix/gemma4-probe-vocab` |
|---|---|---|---|
| completions `max_tokens` 1, client mediaan / p90 | 1061 / 1076 ms | 265 / 274 ms | (ongewijzigd, 4 pogingen) |
| idem pogingen / `completion_tokens` | 4 / 4 | 1 / 1 | 4 / 4 |
| chat `max_tokens` 1, client | 509 ms | 503 ms | 448 ms |
| chat, tot start generatie | 58 ms | 51 ms | 2 ms |

Andere completionsvarianten zijn op de fixtak binnen de ruis gelijk (verse server per tak).

`_encode_messages` los, alleen CPU, Qwen3.6-tokenizer, chat-encode-cache uit, mediaan van 7, token-id's gelijk:

| Chat | `main` | waarvan `is_gemma4_tokenizer` | fix |
|---|---|---|---|
| 46 tokens | 52,7 ms | 52,4 ms | 0,1 ms |
| 10K | 60,0 ms | 52,2 ms | 6,9 ms |
| 26K | 83,9 ms | 63,5 ms | 19,4 ms |
| 41K | 101,3 ms | 67,5 ms | 32,1 ms |
| 80K | 129,0 ms | 64,2 ms | 61,3 ms |

Dit werpt een ander licht op vondst 26: de ~43 ms die aan de Jinja-render werd toegeschreven, is vrijwel zeker deze vocabulairecontrole (constant, onafhankelijk van de lengte). Niet opnieuw gemeten met de memo-tak erbij.

## Fixes

1. `fix/completions-overhead` (b921b9a6): blank retries alleen als `sampler.temperature > 0` (bij gretig decoderen kan een nieuwe seed de uitvoer niet veranderen), en `token_times` per poging legen zodat `usage` en `ttft_s` over de teruggegeven poging gaan. `tests/test_blank_retry_greedy.py` (4 tests, 2 falen op `main`). Diff: 12 regels in `openai.py`.
2. `fix/gemma4-probe-vocab` (2a1057c7): de vijf Gemma-markeringen opzoeken via `backend_tokenizer.token_to_id` (~15 µs), met `get_vocab()` als terugval. Uitkomst gelijk gecontroleerd op de echte tokenizers van Gemma 4 (True) en Qwen3.6 (False). `tests/test_gemma4_tokenizer_probe.py` (5 tests, 2 falen op `main`).

Beide: volledige suite gelijk aan de nulmeting (alleen de 19 bekende mislukkingen), ruff zonder nieuwe meldingen, `python -m build` en `scripts/fresh_venv_smoke.sh` geslaagd, CHANGELOG-regel onder Unreleased. PR-teksten klaar; geen PR geopend.

## Nieuwe waarnemingen (niet uitgezocht)

- **Chat-prefill is trager dan completions voor dezelfde tekst:** 396 tokens via chat `prompt_target_prefill_time_s` 396 ms tegen 257 ms voor 384 tokens via completions (enkele meting, gemeten). Chat heeft daarnaast ~15 ms `commit_time_s` en ~30 ms `prompt_state_unattributed_time_s`.
- **Prompt-scoring is trager dan een generatie van 1 token** op dezelfde prompt: 443 tegen 265 ms (gemeten, tabel boven). Voor de classifier is de genereerroute met eerste-token-logprobs (PR #530) dus waarschijnlijk sneller dan scoring; niet gemeten met logprobs.
- Niet-streaming completions met `stop` doen nooit een blank retry, zonder `stop` wel: het `stop`-pad geeft een `token_callback` mee en telt daardoor als streaming. Met de fix speelt dit alleen nog bij `temperature > 0`.
