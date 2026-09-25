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
