# Onderzoek: MTPLX als lokale classifier

Deze map hoort bij de fork `jvmenen/MTPLX` en staat op de tak `onderzoek`, los van de featuretakken, zodat hij niet in een pull request naar het origineel meekomt. Hij bewaart metingen en bevindingen uit het gebruik van MTPLX als snelle lokale classifier (kansen over vaste labels uitlezen in plaats van tekst genereren).

## Rapporten

| Datum | Rapport | Onderwerp |
|---|---|---|
| 2026-09-25 | [metingen-classifier](2026-09-25-metingen-classifier.md) | Waar de tijd per classificatie heen gaat, hergebruik van het promptbegin, promptvolgorde |
| 2026-09-25 | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Logprobs voor het eerste gegenereerde token op `/v1/completions` en `/v1/chat/completions` (tak `feat/first-token-logprobs`) |

## Nog te onderzoeken

Zie [VONDSTEN.md](VONDSTEN.md): een lopende lijst met vondsten waar we nog naar willen kijken.

## Takken in de fork

| Tak | Inhoud | Status |
|---|---|---|
| `feat/first-token-logprobs` | Eerste-token-logprobs, 27 tests | Lokaal gecommit (3c3f3aaf), nog niet live getoetst |
| `feat/prefix-reuse-block` | Hergebruik van een promptbegin korter dan 512 tokens | In onderzoek |
| `onderzoek` | Deze documentatie | Lopend |
