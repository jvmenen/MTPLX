# Vondsten om nog naar te kijken

Lopende lijst. Per vondst: waar het vandaan komt en wat de volgende stap is. Afgehandelde punten gaan naar de onderste sectie, met datum en uitkomst.

## Open

| # | Vondst | Bron | Volgende stap |
|---|---|---|---|
| 1 | Eerste-token-logprobs werkt in unit-tests, maar is niet getoetst met echte gewichten | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Testserver op de tak starten, vergelijken met prompt-scoring op de 240 testberichten (oordeel, kansverschil, p50/p90) |
| 2 | Uitschieters van ~1,5 s op korte generaties, vermoedelijk blank retries | idem | Meten met en zonder `seed` / logprobs; tellen hoe vaak het eerste token leeg is |
| 3 | `/v1/completions` gebruikt de session bank nooit (`cache_bypass` altijd waar) | idem | Uitzoeken wat nodig is om completions te laten hergebruiken; loopt in tak `feat/prefix-reuse-block` |
| 4 | Hergebruik alleen in stappen van 512 tokens; 128 instellen via env en CLI werkte niet | [metingen-classifier](2026-09-25-metingen-classifier.md) | Oorzaak en instelbare granulariteit; loopt in tak `feat/prefix-reuse-block` |
| 5 | Chatroute zonder voorgevuld `Antwoord:` is minder nauwkeurig (0,58 tegen 0,68) | idem | Kijken of de chatroute een voorgevuld assistent-antwoord ondersteunt, of een ander promptpatroon dat hetzelfde doet |
| 6 | `score_prompt_logprobs` bouwt per blok een volledige float32-log-softmax (~250 MB) | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Top-K op bf16 met één `logsumexp` per rij; meten wat het scheelt |
| 7 | `_run_generation` (~650 regels) en de completions-handler (~500 regels) zijn lastig leesbaar | idem | Kandidaat voor een aparte opschoon-PR: antwoordopbouw en streamlus uitlichten |
| 8 | Plafond van de session bank zakt naar 1 GiB door een piekreserve die nooit daalt | [metingen-classifier](2026-09-25-metingen-classifier.md) | Nagaan of de reserve na verloop van tijd mag afnemen (piek resetten na een rustige periode) |
| 9 | Eén verzoek tegelijk (`decode_batch_max = 1`) | idem | Uitzoeken of batching met MTP samen kan, en wat het oplevert voor meerdere agents |
| 10 | Negen tests in `tests/test_public_cli.py` falen op een schone `main` in deze omgeving | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Oorzaak bekijken (lokale modelcache?); eventueel melden bij de maker |

## Afgehandeld

| Datum | Vondst | Uitkomst |
|---|---|---|
| 2026-09-25 | Vraag en labels vooraan in de prompt voor hergebruik | Slechter op beide modellen (Balance 0,70 tegen 0,72; AUROC 0,92 tegen 0,96); huidige volgorde blijft |
| 2026-09-25 | Alleen de laatste positie teruggeven bij prompt-scoring | Levert niets op: top-1 of top-20 en de JSON-grootte maken geen meetbaar verschil |
