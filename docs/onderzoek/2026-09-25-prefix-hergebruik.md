# Hergebruik van een kort promptbegin (25 september 2026)

Tak `feat/prefix-reuse-block`, commit `fe32206c`, gebaseerd op `main` (2.12.0). Gebouwd door een Opus-agent op basis van code lezen en unit-tests, **niet getoetst met een echt model**. Regelnummers verwijzen naar de ongewijzigde code.

## Waarom een gedeeld begin van 220 tokens nooit wordt hergebruikt

Drie oorzaken, die alle drie moeten worden opgelost:

1. **De minimale overeenkomst wordt stil opgehoogd tot de blokgrootte.** `generation.py:4580-4583` neemt `max(MTPLX_SESSION_PREFIX_BLOCK_SIZE (256), MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS (512))`; hetzelfde in `session_bank.py:1349` (standaarden op 248-249) en `cache_bank/cold_tier.py:1090`. De env op 128 zetten levert dus 256 op, en dat is meer dan 220.
2. **Een hybride model (Qwen3-Next) heeft een opgeslagen lineaire-attentietoestand (GDN) nodig op of onder het gedeelde begin.** Die toestand is niet op willekeurige lengte af te knippen. Herstellen accepteert alleen een grens via `recurrent_boundary_at_or_below(matched)` (rond `generation.py:4650`, afwijsreden `boundary_not_better`). De prefill slaat grenzen alleen op een raster op: `MTPLX_GDN_BOUNDARY_TAIL_INTERVAL=256`, `MTPLX_GDN_BOUNDARY_TAIL_BACKOFF=64`, `MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG=1024`, `MTPLX_GDN_BOUNDARY_MAX=8`. Voor een prompt van 300 tot 550 tokens geeft dat een grens op 256 of 512 (of einde min 64), nooit op of onder 220. Dit raster verklaart ook de stappen van 512 die we zagen.
3. **Korte prompts worden zonder grenzen opgeslagen.** De laatste bank-put (`openai.py:26229`) geeft geen grenzen mee; die worden alleen geërfd van een opgeslagen entry die er een prefix van is (`session_bank.py:900-925`). Zo'n entry ontstaat alleen via store-on-prefill (`MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX=1024`) of de prompt-prefix-commit (hard `max(512, ssd_min_prefix)` op `openai.py:20300`). Een prompt van 300 tot 550 tokens krijgt geen van beide. De health toont dan `ssd_prefix_miss`, wat de echte reden in het werkgeheugen verbergt (`session_bank.py:140-190`).

Andere 512-constanten spelen geen rol bij herstel uit het werkgeheugen: `DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS=512` (alleen SSD) en `interval_512` in `engine_session.py:1741` (alleen administratie).

## Wat er veranderd is (standaardgedrag ongewijzigd)

- **Eén lezer voor de drempel:** `runtime_options.block_prefix_min_match_tokens()` vervangt drie losse env-lezingen met een letterlijke 512.
- **Geen ondergrens meer op de blokgrootte** in `session_bank.near_prefix_candidates`. Oudere hybride entries zonder grenzen blijven op blokken uitgelijnd.
- **Nieuwe schakelaar `MTPLX_SESSION_SHARED_PREFIX_EDGE`** (standaard uit): `SessionBank.dominant_shared_prefix_tokens()` bepaalt de lengte van het begin dat de meeste opgeslagen entries delen (bij gelijkspel de kortste); `generation._shared_prefix_edge` maakt daar een vaste rand van, zodat het bestaande mechanisme precies daar een GDN-grens opslaat.
- **Nieuwe vlag en configsleutel:** `--ram-session-prefix-min-match-tokens N` (serve, start, quickstart en doorgifte door de launcher) en `ram_session_prefix_min_match_tokens`. Bij het starten zet dit de drempel op N, zet de randschakelaar aan en zet `STORE_ON_PREFILL_MIN_SUFFIX` op min(1024, N). Zelf gezette env-waarden gaan voor.
- **Nieuwe schakelaar `MTPLX_COMPLETIONS_SESSION_BANK=1`** (standaard uit): `/v1/completions` krijgt de session bank mee, met sessie `anon-completions` en vingerafdruk `endpoint=completions` (gescheiden van chat), zonder live lease; de limiet van 3 entries per sessie begrenst het geheugengebruik. In de stats: `request_session_source="completions_bank"`.
- CHANGELOG-regel onder Unreleased.

## 128 instellen

```bash
mtplx serve ... --ram-session-prefix-min-match-tokens 128
# of
mtplx config set ram_session_prefix_min_match_tokens 128
# en voor de classifier op /v1/completions ook:
MTPLX_COMPLETIONS_SESSION_BANK=1
```

Verwacht verloop: het eerste verzoek is koud, het tweede slaat een grens op bij 220, vanaf het derde wordt op 220 hersteld.

## Kosten (geschat, niet gemeten)

- Per opgeslagen grens: 30 GDN-lagen × 32×128×128 aan toestand, ~62 MiB in fp32 of ~31 MiB in bf16 (datatype niet gecontroleerd), plus ~1,4 MiB conv-toestand.
- KV: ~20 KiB per token.
- Een koude miss met rand kost één extra splitsing van de forward pass.
- Elke opgeslagen prompt kost één extra bank-put (store-on-prefill).

## Uitschieters van ~1,5 s: hypotheses van deze agent (niet getoetst)

1. **Verzoeken wachten op elkaar:** `--scheduler-mode serial` met één `state.lock` (`openai.py:25972`); elke overlap geeft n × ~390 ms. Controle: `lock_wait_time_s` in `mtplx_stats`.
2. **Achtergrondonderhoud op de owner-thread** na eerder chatverkeer: SSD-coldtier-encodes (0,66-3,6 s vóór de yield-fix volgens `tests/test_cold_tier_foreground_yield.py`), snapshot-afronding, idle async postcommit (wachttijd 0,6 s, `engine_session.py:603`), plus de toelatingsronde `abort_cross_session_postcommits`. Controle: `postcommit_cross_session_yield`, coldtier-stats in /health, A/B met `--ssd-session-cache off`.
3. **GPU-geheugen niet meer resident** na ~2,5 s stilte, ~9 ms per GiB (`model_scheduler.py:300-320`). Controle: `gpu_keepalive.warm` per verzoek.
4. **Eenmalige compilatie per nieuwe grootteklasse van 256 tokens** (`a3b_compiled_target_prefix.py`). Zou alleen vroege uitschieters verklaren.
5. **Python-GC-pauzes** (geen `gc.freeze`); lage waarschijnlijkheid.

De agent van de eerste-token-logprobs noemde blank retries als sterkste spoor (4 × ~390 ms). De live test moet uitwijzen welke verklaring klopt.

## Tests

- Nieuw: `tests/test_short_shared_prefix_reuse.py` (25 tests) en `tests/test_completions_session_bank.py` (4 tests), alle groen.
- 40 verwante bestanden: 1.422 geslaagd, 9 mislukt, alle 9 in `test_public_cli.py` en identiek mislukt op ongewijzigde code.
- 17 extra bestanden (server_openai, laguna, vision, postcommit): 722 geslaagd, 1 mislukt (`test_laguna_s_2_1_ar_route_skips_qwen_performance_hooks`), ook identiek op ongewijzigde code.
- Ruff: geen nieuwe meldingen.

## Review: leesbaarheid en snelheid

- `restore_or_prefill_prompt_state` (~5883-6560) en `_restore_near_prefix_prompt_state` (~4550-4960) zijn honderden regels lang; één functie per herstelroute (exact, near, cold) die `PromptState | None` teruggeeft, leest beter.
- `session_bank.near_prefix_candidates` (~1320-1560) mengt zoeken, schaduwen van residente tweelingen, cold lookup en diagnose; opsplitsen.
- Env-lezers met gedupliceerde letterlijke standaarden blijven: `generation.py:4572-4573`, `engine_session.py:831-840`, `_block_restorable_prefix_tokens` (`openai.py:18911`, negeert de env) en de magische `max(512, …)` op `openai.py:20300`.
- `common_prefix_len` (`session_bank.py:317`) is een Python-lus die per entry meerdere keren per verzoek draait; één keer per verzoek omzetten en een numpy-vergelijking scheelt tijd.
- De echte afwijsreden in het werkgeheugen rapporteren in plaats van `ssd_prefix_miss`.
- De twee bijna identieke `_run_generation_dispatched`-aanroepen in de chat-handler (~33120-33200) kunnen een gedeelde kwargs-helper krijgen, zoals completions nu heeft.
