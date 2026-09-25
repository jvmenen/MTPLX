# Opschonen: prefixvergelijking en instellingen (26 september 2026)

Tak `refactor/prefix-helpers` (gebouwd op `feat/prefix-reuse-block`), twee commits, niet gepusht. Gedaan door een Opus-agent.

## 1. Sneller prefixen vergelijken (commit d32c77b2)

Dezelfde lus `common_prefix_len` stond drie keer in de code (`session_bank.py`, `engine_session.py`, `cache_bank/cold_tier.py`). Nieuw module `mtplx/token_prefix.py` (alleen standaardbibliotheek, omdat `cache_bank` zonder mlx moet kunnen laden): vergelijkt blokken van 64 tokens in C (`tuple(a[s:e]) == tuple(b[s:e])`) en loopt alleen door het blok met het eerste verschil token voor token. Resultaten zijn identiek, ook bij gemengde lijst- en tuple-invoer. Numpy viel af: de entries zijn Python-tuples, en omzetten per aanroep is trager dan de oude lus.

Microbenchmark, 24 entries per verzoek, µs per verzoek (oud → nieuw):

| Prompt | Gedeeld 0 | Gedeeld 220 | Gedeeld 512 | Volledig gelijk |
|---|---|---|---|---|
| 300 | 2 → 8 | 99 → 51 | 128 → 47 | 131 → 47 |
| 2.000 | 2 → 7 | 98 → 46 | 251 → 84 | 1.024 → 308 |
| 8.000 | 2 → 8 | 100 → 49 | 255 → 89 | 4.008 → 1.182 |

Winst tot ~3 ms per verzoek bij lange gedeelde contexten (agentgesprekken); bij korte prompts enkele tientallen µs. 23 nieuwe tests in `tests/test_token_prefix.py`.

## 2. Eén lezer per instelling (commit 412e1571)

In `runtime_options.py` nieuw: `near_prefix_max_token_gap()`, `near_prefix_min_match_tokens()`, `prefix_block_size()`, `store_on_prefill_min_suffix()`, met gedeelde hulpfunctie `_env_int_at_least` en benoemde constanten (8, 64, 256, 1024). Aanroepen in `generation`, `engine_session`, `session_bank` en `cold_tier` gebruiken ze; een ongebruikte derde lezer is verwijderd. De magische 512 heet nu `_PROMPT_PREFIX_COMMIT_MIN_TOKENS`, met uitleg.

Eén gedragswijziging, als fix in CHANGELOG: `_block_restorable_prefix_tokens` (de schatting bij toelating van de prefill) negeerde de env-instellingen die het herstel zelf wel gebruikt; nu gebruiken beide dezelfde lezers. Zonder gezette env verandert niets. 14 nieuwe tests in `tests/test_session_prefix_env_readers.py`.

## Tests

Relevante set: 1.330 geslaagd, 10 mislukt, precies de nulmeting (9 in `test_public_cli.py`, 1 laguna-test). Zeven andere bestanden: 101 geslaagd, 14 overgeslagen. Ruff: geen nieuwe meldingen, 4 minder in `openai.py`.

## Nog te doen (gezien, niet gedaan)

- `openai._common_prefix_len` is een vierde kopie van de lus.
- `engine_session` hoogt de drempel nog op tot de blokgrootte, terwijl `session_bank` dat op `feat/prefix-reuse-block` niet meer doet; nagaan of dat verschil bedoeld is.
- `cold_tier.DEFAULT_BLOCK_SIZE` (256) en `DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS` (512) dupliceren dezelfde getallen; `MTPLX_SSD_SESSION_CACHE_MIN_PREFIX_TOKENS` wordt met een letterlijke 512 gelezen in `parse_args`.
- `MTPLX_SESSION_BANK_MAX_BYTES`, `_PER_SESSION_BYTES`, `_MAX_ENTRIES` komen 6 tot 9 keer voor.
- `generation._env_int` en losse `raw not in {"0","false",...}`-parsers kunnen naar `env_bool` / `_env_int_at_least`.
- Geen CHANGELOG-regel voor de versnelling.
