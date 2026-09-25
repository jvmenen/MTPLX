# Vondsten om nog naar te kijken

Lopende lijst. Per vondst: waar het vandaan komt en wat de volgende stap is. Afgehandelde punten gaan naar de onderste sectie, met datum en uitkomst.

## Open

| # | Vondst | Bron | Volgende stap |
|---|---|---|---|
| 1 | Eerste-token-logprobs werkt in unit-tests, maar is niet getoetst met echte gewichten | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Testserver op de tak starten, vergelijken met prompt-scoring op de 240 testberichten (oordeel, kansverschil, p50/p90) |
| 2 | Uitschieters van ~1,5 s op korte generaties; hypotheses: blank retries, wachten op het lock, achtergrondonderhoud, GPU niet resident | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md), [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Meten met en zonder `seed` / logprobs; tellen hoe vaak het eerste token leeg is |
| 3 | `/v1/completions` gebruikt de session bank nooit (`cache_bypass` altijd waar) | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Opgelost achter `MTPLX_COMPLETIONS_SESSION_BANK=1` (tak `feat/prefix-reuse-block`); live toetsen |
| 4 | Hergebruik alleen in stappen van 512 tokens; 128 instellen via env en CLI werkte niet | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Oorzaak gevonden (drempel opgehoogd tot blokgrootte, GDN-grensraster, korte prompts zonder grenzen); vlag `--ram-session-prefix-min-match-tokens 128` gebouwd; live toetsen en geheugengebruik per grens meten |
| 5 | Chatroute zonder voorgevuld `Antwoord:` is minder nauwkeurig (0,58 tegen 0,68) | idem | Kijken of de chatroute een voorgevuld assistent-antwoord ondersteunt, of een ander promptpatroon dat hetzelfde doet |
| 6 | `score_prompt_logprobs` bouwt per blok een volledige float32-log-softmax (~250 MB) | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Top-K op bf16 met één `logsumexp` per rij; meten wat het scheelt |
| 7 | `_run_generation` (~650 regels) en de completions-handler (~500 regels) zijn lastig leesbaar | idem | Kandidaat voor een aparte opschoon-PR: antwoordopbouw en streamlus uitlichten |
| 8 | Plafond van de session bank zakt naar 1 GiB door een piekreserve die nooit daalt | [metingen-classifier](2026-09-25-metingen-classifier.md) | Nagaan of de reserve na verloop van tijd mag afnemen (piek resetten na een rustige periode) |
| 9 | Eén verzoek tegelijk (`decode_batch_max = 1`) | idem | Uitzoeken of batching met MTP samen kan, en wat het oplevert voor meerdere agents |
| 10 | Negen tests in `tests/test_public_cli.py` falen op een schone `main` in deze omgeving | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Oorzaak bekijken (lokale modelcache?); eventueel melden bij de maker |
| 11 | Health meldt `ssd_prefix_miss` terwijl de echte oorzaak in het werkgeheugen ligt | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Echte afwijsreden rapporteren; kandidaat voor een kleine upstream-PR |
| 12 | `common_prefix_len` is een Python-lus, meerdere keren per verzoek per entry | idem | Numpy-vergelijking, één keer per verzoek omzetten; meten |
| 13 | Gedupliceerde env-lezers met letterlijke standaarden en een magische `max(512, …)` | idem | Opruimen naar één bron per instelling (deels gedaan in `runtime_options`) |
| 14 | `restore_or_prefill_prompt_state`, `_restore_near_prefix_prompt_state` en `near_prefix_candidates` zijn honderden regels | idem | Opsplitsen per herstelroute; aparte opschoon-PR |
| 15 | Eerste-token-logprobs en kort prefix-hergebruik zitten op twee takken | beide rapporten | Samenvoegen op een testtak en live meten: kansen, p50/p90, `cached_tokens`, geheugen |

## Afgehandeld

| Datum | Vondst | Uitkomst |
|---|---|---|
| 2026-09-25 | Vraag en labels vooraan in de prompt voor hergebruik | Slechter op beide modellen (Balance 0,70 tegen 0,72; AUROC 0,92 tegen 0,96); huidige volgorde blijft |
| 2026-09-25 | Alleen de laatste positie teruggeven bij prompt-scoring | Levert niets op: top-1 of top-20 en de JSON-grootte maken geen meetbaar verschil |
