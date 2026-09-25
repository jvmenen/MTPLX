# Optimalisaties met effect (26 september 2026)

Analyse door een Opus-agent op tak `test/classifier-live`, alleen code lezen en CPU-microbenchmarks; niets op de GPU gemeten. Alle GPU-winsten zijn schattingen. Gemeten waarden staan als **[gemeten]**.

Uitgangspunten voor de schattingen: model 20 GB (4-bit), woordenschat 248.320, hidden 2048, 40 lagen, 256 experts met top-8. Een forward pass over 64 of meer rijen raakt vrijwel alle experts, ~18 GB aan gewichten, op de M5 Pro ongeveer 50-75 ms per extra forward. libmlx 0.32.2 doet `argpartition` over de volledige woordenschat als een volledige sortering in meerdere blokken.

## Gerangschikt

| # | Optimalisatie | Gebruik | Geschatte winst | Moeite | Risico | Meten |
|---|---|---|---|---|---|---|
| 1 | Classifier via eerste-token-logprobs + completions in de session bank + gedeelde-prefixrand (staat op de testtak) | classificatie | warm p50 -35 tot -55% als het vaste deel ≥60% van de prompt is | S (config) | middel | `cached_tokens`, `prompt_eval_time_s`; live getoetst: -37% (zie [live-test](2026-09-26-live-test-classifier.md)) |
| 2 | Scoring top-K: blokmaxima op bf16 als voorfilter, exact top-K, één f32-`logsumexp` per rij, alleen aftrekken voor overblijvers en doelen | prompt-scoring | 15-40 ms per blok van 256 rijen (~30-80 ms op 500 tokens) | S-M | laag | `prompt_eval_time_s` bij 300/512/550 tokens, plus pariteitstest oud tegen nieuw |
| 3 | Scoring: trunk in prefill-blokken van 2048, lm_head en top-K per 256 rijen | prompt-scoring | 1-2 forwards minder op 300-550 tokens, ~50-75 ms elk | S-M | laag | als 2 |
| 4 | GDN-grenzen vastleggen binnen de forward pass ook voor de A3B-familie (nu alleen qwen4_exp) | agentgesprekken (en koude classificatie met bank) | ~50-80 ms per warme agentbeurt met een staart boven 256 tokens | M-L | middel | `MTPLX_PREFILL_CHUNK_TRACE=1`, `prompt_eval_time_s` per warme beurt |
| 5 | Memo per segment voor de chat-encode | agentgesprekken | 40-55 ms TTFT per beurt bij 77k tokens, 12-15 ms bij 20k **[gemeten basis]** | S | laag | `_encode_messages` timen, publiceren in `template_observability` |
| 6 | Laatste pending-commit en session-bank-put van het kritieke pad halen | agentgesprekken, classificatie met bank | ~15-30 ms per beurt | M | middel (lockvolgorde) | eerst `put_s` publiceren (snelle winst a) |
| 7 | Laatste pending-commit overslaan voor completions-bankverzoeken | classificatie | ~10-20 ms per verzoek | S | laag | `commit_time_s` |

## Toelichting top 5

1. **Completions-bank.** `_completions_session_bank_kwargs` (openai.py:20309) en `_shared_prefix_edge` (generation.py:5609) herstellen een gedeeld begin; warme verzoeken lezen alleen de wisselende staart in (100-250 tokens, één forward). Prompt-scoring (`echo`) kan dit niet, want die heeft logits voor elke positie nodig. De structurele oplossing is dus de overstap van scoring naar `max_tokens: 1` met logprobs. Koude prompts worden gesplitst op de prefixrand en de staartrand (tot 3 forwards): koud en warm apart meten.
2. **Top-K bij scoring.** Het huidige pad (generation.py:8156-8169, 8245-8264) schrijft via `_log_softmax_f32` een f32-tensor van 256 × 248.320 (254 MB), de aftrekking nog een, en `argpartition` sorteert ~63,6 miljoen f32-waarden per blok. Voorstel, exact op gelijkspel na: rijen hervormen naar [rijen, 1940, 128], maximum per blok, de top-20 blokken houden (2.560 kandidaten), daarin argpartition; de top-K ligt gegarandeerd in de K blokken met de hoogste maxima. Eén f32-`logsumexp` per rij; logprob = f32(logit) − lse, alleen voor overblijvers en doelwaarden. Leesbaar houden met kleine functies (`_block_max_candidates`, `_exact_top_k`, `_row_logsumexp_f32`).
3. **Trunk-blokken bij scoring.** `score_prompt_logprobs` gebruikt `chunk_size=256` (generation.py:8209), de aanroep past dat niet aan (openai.py:25424), terwijl gewone prefill 2048 gebruikt (profiles.py:558-560). De 256 is bedoeld om het geheugen voor logits te begrenzen, niet voor de trunk. `forward_ar(..., return_hidden=True, hidden_variant="post_norm", emit_logits=False)` bestaat (mtp_patch.py:820-860); een kleine `rt.project_logits(hidden)` per 256 rijen houdt het piekgeheugen laag.
4. **Staartladder-forwards op A3B.** `_prefill_spans_with_tail_grid` en `_geometric_tail_edges` (generation.py:5175-5243) maken voor een staart van 3.000 tokens spans [0,2048], [2048,2936], [2936,3000]; de forward van 64 rijen leest nog ~86% van de experts. De eigen docstring (5256-5275) noemt dit "one behind every warm agent turn". Voor Flash-Next leverde grenzen vastleggen binnen de forward 0,25-0,3 s op bij een koude prompt van 4k (M5 Max, 22 september). De hooks bestaan alleen in `models/qwen4_exp.py:2567-2700`. Niet de chunked-WY-aanpak terughalen: omlx meldde dat als verlies, en `gdn_blocked_prefill.py` bevat dat negatieve resultaat.
5. **Chat-encode.** `ChatEncodeCache` (openai.py:14777-14885) hasht de hele payload en mist dus bij elke nieuwe agentbeurt. Tool- en assistentgeschiedenis wordt al per segment ge-encodeerd (`_encode_rendered_chat_text_segmented`, 14557); een begrensde LRU op (tokenizer, segmenttekst) laat elke beurt alleen nieuwe segmenten encoderen. **[gemeten]** Volledige encode: 52-58 ms bij 76k tokens, 14-17 ms bij 22k.

## Snelle winsten (elk een kleine upstream-PR)

- a. `timing_out` meegeven aan de laatste `session_bank.put` (openai.py:26445) en `sessionbank_put_s` publiceren; nu is die kost onzichtbaar.
- b. Scoring minimaal: top-K op de ruwe bf16-logits plus één f32-`logsumexp` per rij; scheelt al twee tensors van 254 MB per blok.
- c. Scoring: trunk in prefill-blokgrootte, lm_head per 256 rijen (punt 3).
- d. Segment-memo voor de encode (punt 5).
- e. Laatste pending-commit-forward overslaan (generation.py:15255-15290) voor sessie `anon-completions`: het gegenereerde token is nooit een prefix van de volgende classifierprompt.

## Niet de moeite, of weerlegd

- Tokenizer-decode bij scoring: **[gemeten]** 2 µs per aanroep, al gememoized per verzoek.
- Numpy-lus bij scoring: **[gemeten]** 0,9 ms per 256 rijen.
- JSON van scoring: **[gemeten]** 2,3 ms voor 185 KiB; orjson is geen afhankelijkheid waard.
- Tokenisatie en template: **[gemeten]** 0,8 ms voor 751 tokens; template 0,4-9 ms.
- Events per ronde: turbo zet `MTPLX_DROP_EVENTS=1` (profiles.py:236).
- Blank retries: staan uit bij streaming en logprobs-verzoeken (openai.py:26068).
- Streaming-decode O(n²): al opgelost (`_IncrementalTokenDecoder`).
- Env-lezingen in de decodelus (6), MTP-geschiedenis bij prefill (~1/40 van de trunk), telemetrie: verwaarloosbaar.
- `tuple(int(...))`-conversies in session_bank.py: **[gemeten]** ~1,2 ms per stuk bij 77k tokens; overlapt met het opschoonwerk aan `common_prefix_len`.

## Open

De vaste overhead van ~50 ms per verzoek is niet toegewezen. Eerste stap: kloktijd bij de client vergelijken met `mtplx_stats.prompt_eval_time_s` en `server_elapsed_s`; daarna één `py-spy`-opname (vraagt sudo) buiten een GPU-meting.

## Tijd van de laatste session-bank-put zichtbaar (26 september 2026, vondst 23 stap 1)

Tak `feat/sessionbank-put-timing` (worktree `~/Dev/MTPLX-putstijd`), gebaseerd op `origin/main` 1de2b1c0. Commit dbb4bfea, gepusht naar de fork.

### Wat er verandert

- `mtplx/server/openai.py:26250-26265` (`_run_generation`): `perf_counter()` rond de laatste `session_bank.put` na een beurt; de duur komt als `stats["sessionbank_put_s"]` (afgerond op 6 cijfers, zelfde idioom als de batchlanes) naast `sessionbank_snapshot_bytes`.
- `mtplx/server/openai.py:20688`: `sessionbank_put_s` in `PUBLIC_MTPLX_STATS_KEYS`, dus zichtbaar in `mtplx_stats` van chat, messages en completions.
- `mtplx/server/openai.py:26339`: dezelfde sleutel in de lijst die naar de metrics-envelope wordt gekopieerd, dus ook in `state.last_metrics` en daarmee in `/v1/mtplx/snapshot` (`latest`, `recent`, openai.py:18507-18508).
- De sleutel staat er alleen als die put gedraaid heeft (het "quiet envelope"-idioom van de maker); antwoorden zonder bankcommit blijven byte-gelijk, de goldens veranderen niet.
- Geen gedragswijziging: de put zelf, de volgorde en de foutafhandeling blijven gelijk. `timing_out` is bewust niet meegegeven: de uitsplitsing (trunk-snapshot, entry-build, cold-enqueue) vraagt een extra publiek veld en de controle `_callable_accepts_keyword` voor testdubbels (zoals openai.py:23215-23216). Dat is een logische vervolgstap als `sessionbank_put_s` groot blijkt.

### Waarom de kost onzichtbaar was (alleen code gelezen)

- `elapsed_s` wordt genomen op openai.py:26197, vóór de put (26251). Dus ook `request_elapsed_s`, `end_to_end_tok_s` en het dashboard missen deze tijd.
- De batchlanes publiceren hun puts al wel: `ar_batch_prompt_boundary_bank_put_s` (openai.py:4627), `ar_batch_row_bank_put_s` (4679) en `mtp_batch_prompt_boundary_bank_put_s` (24742). De seriële lane, die het standaardverkeer bedient, deed dat niet.
- Niet in de meting: de MTP-historie-snapshot in `_generation_final_bank_metadata` (openai.py:26245-26249; `snapshot_cache` op regel 23029) die vlak vóór de put draait. Die zit ook op het kritieke pad.

### Tests (gemeten in deze omgeving, testmodel/stubs, geen echt model)

- Nieuw `tests/test_sessionbank_put_timing.py` (2 tests): met een stubbank waarvan de put 20 ms duurt, staat `sessionbank_put_s` als float van minstens 20 ms in de requeststats, in `_public_mtplx_stats` en in de laatste metrics-rij; zonder final state (geen put) ontbreekt de sleutel overal. De eerste test faalt zonder de wijziging.
- Goldens (`test_request_observability_golden.py`) en `test_api_benchmark_contracts.py`: 44 geslaagd, ongewijzigd.
- Volledige suite: ongeveer 9.365 geslaagd, 67 overgeslagen, 19 mislukt; precies de nulmeting (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`), geen nieuwe. Gerichte set (nieuwe test, goldens, contracten, `test_no_mlx_imports`, `test_runtime_kpis`): 68 geslaagd.
- Ruff over `mtplx` en `tests`: 2665 meldingen, gelijk aan `main`.
- `python -m build` en `scripts/fresh_venv_smoke.sh`: geslaagd.

### Te meten op de echte server (gedaan, zie de controle hieronder)

- `sessionbank_put_s` per beurt op Qwen3.6-35B-A3B (Balance) bij oplopende contextlengte (2k, 8k, 32k) en bij agentbeurten met tools, verse server per variant; naast `ttft_s`, `elapsed_s` en de klantgemeten totale tijd, zodat zichtbaar wordt, welk deel van de staart de put is.
- Of `MTPLX_SESSION_LAZY_SNAPSHOT` (standaard aan, session_bank.py:88) de put al goedkoop maakt; zo ja, dan is stap 2 weinig waard.
- Classifierverkeer (`anon-completions`): hoe vaak de put daar draait en wat hij kost.

### Stap 2: de put van het kritieke pad halen (niet gebouwd, alleen code gelezen)

Waar hij nu staat:

- `_run_generation` draait via `_run_generation_dispatched` (openai.py:25352) als foreground-werk op de model-owner-thread (`_submit_foreground_model_work`, 3921; aanroep 25568).
- De modellock wordt genomen op 25927 en vrijgegeven op 26193; de put op 26251 draait dus zonder `state.lock`, maar nog binnen dezelfde foreground-taak. Het eindframe van de stream (finish_reason, usage, `[DONE]`) en de volgende foreground-taak wachten erop.

Wat nodig zou zijn:

1. **Hergebruik van het bestaande postcommit-pad.** `_store_generation_final_history_snapshot` (openai.py:23095) doet dezelfde commit als idle-postcommit: lock niet-blokkerend (23180), dezelfde `_generation_final_bank_values`/`_metadata`, put met `timing_out` (23215-23217). `_schedule_idle_postcommit_snapshot` (23310) zet het werk op de idle-lane (`_submit_idle_postcommit_model_work`, 4997) en registreert een pending postcommit op de sessie (23644), waar het volgende verzoek van dezelfde sessie begrensd op wacht (32578, 33412).
2. **Lockvolgorde.** De bank heeft geen eigen lock (geen threading in `session_bank.py`; `_entries` wordt direct gemuteerd op 1169 en `_evict_if_needed` op 1172). Veiligheid komt nu uit de serialisatie op de owner-thread. Een uitgestelde put moet dus op de owner-thread blijven en eerst `state.lock` nemen, dan pas de bank aanraken, in dezelfde volgorde als de foreground (lock op 25927, daarna restore en generatie). Nooit bank vóór lock, en niet blokkerend wachten op de lock vanuit de idle-lane (patroon 23180: bij bezet overslaan en loggen).
3. **Levensduur van de cache.** De put leest `final_state.final_trunk_cache` en de MTP-cache. Tot de taak draait moeten die buffers vastgehouden worden; het volgende verzoek mag ze niet via donatie overschrijven. Met lazy snapshots is dit al een bekend risico (settle-job, session_bank.py:2220-2320; kopie bij eerste schrijfactie 66 ms/GB volgens het commentaar op 94-102).

Risico's:

- **Cachemisser op de volgende beurt.** Komt de volgende beurt van dezelfde sessie vóór de postcommit, dan moet hij wachten (pending-postcommit) of koud prefillen. Voor agentsessies met korte pauzes kan dat de winst opeten of omkeren.
- **Geheugen.** Een GB-grote cache blijft langer vastgehouden; onder aanhoudend verkeer kan de idle-lane uitgehongerd raken (zie het commentaar over newest-wins-coalescing bij de cold-enqueue, session_bank.py:2362-2368).
- **Stats verschuiven.** `sessionbank_snapshot_bytes` en `sessionbank_skipped_oversized_snapshot` zijn bij het antwoord nog niet bekend; ze verhuizen naar `session_postcommit_snapshot`.
- **Retries.** Paden die al `commit_final_state_to_bank=False` meegeven (openai.py:33683 e.v.) moeten consistent blijven.

Inschatting: pas doen als de meting op het echte model laat zien dat `sessionbank_put_s` een merkbaar deel van de staart is (orde honderden milliseconden). Met lazy snapshots en uitgestelde cold-tier-serialisatie (session_bank.py:2353-2380) is de put waarschijnlijk al grotendeels goedkoop; wat overblijft, is de entry-build en de dispatch, plus de MTP-historie-snapshot vlak ervoor, die buiten deze meting valt. Als hij wel groot is: achter een schakelaar (bijvoorbeeld `MTPLX_DEFER_FINAL_BANK_COMMIT`, standaard uit) het bestaande postcommit-pad gebruiken, niet een nieuw pad bouwen.

### Controle op de echte server (26 september 2026)

Gemeten op Apple M5 Pro, 64 GB, Qwen3.6-35B-A3B Balance, profiel turbo (productie-instellingen), verse server vanuit de worktree, commit `dbb4bfea`. Zelfgemaakte neutrale tekst, zonder redeneren, `temperature` 0. Drie herhalingen per punt, mediaan. Een groeiend gesprek: elke beurt voegt tekst toe aan het vorige gesprek, dus de 8k- en 32k-beurt hergebruiken het begin.

| Punt | Prompt (tokens) | Hergebruikt | `sessionbank_put_s` | `ttft_s` | `elapsed_s` | Client (s) | Snapshot |
|---|---|---|---|---|---|---|---|
| Chat ~2k | 1.715 | 0 | 0,46 ms | 0,94 | 1,46 | 1,54 | 233 MB |
| Chat ~8k | 8.156 | 1.650 | 1,28 ms | 3,23 | 3,49 | 3,57 | 636 MB |
| Chat ~32k | 34.242 | 8.091 | 4,42 ms | 16,34 | 16,68 | 16,86 | 1,29 GB |
| Agent, beurt met toolaanroep | 2.012 | 0 | 0,49 ms | 1,05 | 1,26 | 1,35 | 240 MB |
| Agent, beurt na toolresultaat | 2.073 | 2.041 | 0,67 ms | 0,15 | 0,39 | 0,46 | 241 MB |
| Classificatie `/v1/completions`, `max_tokens` 1 | 384 | 0 | ontbreekt | 0,27 | 0,26 | 1,06 | 0 |
| Classificatie via prompt-scoring (`echo`, `max_tokens` 0) | 385 | n.v.t. | ontbreekt | n.v.t. | n.v.t. | 0,44 | n.v.t. |

- **Veld zichtbaar (geslaagd).** `sessionbank_put_s` staat in `mtplx_stats` van elk chatantwoord en met dezelfde waarde in `/v1/mtplx/snapshot` → `latest`. Een eerdere reeks die doorschoot naar 10k en 44k tokens gaf 1,4 en 4,8 ms: hetzelfde beeld.
- **Classificatie.** `max_tokens` 1 met `logprobs` geeft op deze tak HTTP 400 (eerste-token-logprobs zitten in PR #530, niet op main); daarom gemeten zonder `logprobs` en via prompt-scoring. `/v1/completions` gebruikt op main de bank niet (`session_prefill_store.skip_reason: no_bank`), dus er is geen put en de sleutel ontbreekt, zoals bedoeld.
- **Conclusie.** De put kost 0,5 tot 4,4 ms, ongeveer 0,03% van de beurt bij 32k en hooguit 0,2% bij een korte agentbeurt. De lazy snapshot en de uitgestelde cold-tier-serialisatie maken hem al goedkoop. **Stap 2 (de put van het kritieke pad halen) is de moeite niet**; het verschil tussen client- en `elapsed_s` (70 tot 180 ms) zit ergens anders.
- **Bijvangst.** Bij `/v1/completions` met `max_tokens` 1 is de clienttijd ~1,06 s tegen `elapsed_s` 0,26 s: ongeveer 0,8 s buiten de gemeten tijd. Past bij vondst 2 en 25; niet verder onderzocht.
- De server is daarna gestopt; poort 8000 is vrij.
