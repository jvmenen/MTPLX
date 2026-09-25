# Onderzoek: MTPLX als lokale classifier

Deze map hoort bij de fork `jvmenen/MTPLX` en staat op de tak `onderzoek`, los van de featuretakken, zodat hij niet in een pull request naar het origineel meekomt. Hij bewaart metingen en bevindingen uit het gebruik van MTPLX als snelle lokale classifier (kansen over vaste labels uitlezen in plaats van tekst genereren).

## Rapporten

| Datum | Rapport | Onderwerp |
|---|---|---|
| 2026-09-25 | [metingen-classifier](2026-09-25-metingen-classifier.md) | Waar de tijd per classificatie heen gaat, hergebruik van het promptbegin, promptvolgorde |
| 2026-09-25 | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Logprobs voor het eerste gegenereerde token op `/v1/completions` en `/v1/chat/completions` (tak `feat/first-token-logprobs`) |
| 2026-09-26 | [optimalisaties](2026-09-26-optimalisaties.md) | Gerangschikte optimalisaties met effect, snelle winsten en wat niet de moeite is |
| 2026-09-26 | [live-test-classifier](2026-09-26-live-test-classifier.md) | Live test van beide takken samen op 240 berichten: snelheid, kwaliteit, hergebruik |
| 2026-09-26 | [opschonen-prefix-helpers](2026-09-26-opschonen-prefix-helpers.md) | Snellere prefixvergelijking, één lezer per instelling (tak `refactor/prefix-helpers`) |
| 2026-09-25 | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Waarom een kort gedeeld promptbegin niet wordt hergebruikt en hoe 128 instelbaar wordt; completions in de session bank (tak `feat/prefix-reuse-block`) |

## Nog te onderzoeken

Zie [VONDSTEN.md](VONDSTEN.md): een lopende lijst met vondsten waar we nog naar willen kijken.

## Takken in de fork

| Tak | Inhoud | Status |
|---|---|---|
| `feat/first-token-logprobs` | Eerste-token-logprobs, 27 tests | Lokaal gecommit (3c3f3aaf), live getoetst |
| `feat/prefix-reuse-block` | Hergebruik van een promptbegin korter dan 512 tokens, completions in de session bank | Lokaal gecommit (fe32206c), live getoetst |
| `refactor/prefix-helpers` | Snellere prefixvergelijking, één lezer per instelling | Lokaal gecommit (d32c77b2, 412e1571) |
| `test/classifier-live` | Beide featuretakken samengevoegd voor de live test | Lokaal |
| `onderzoek` | Deze documentatie | Lopend |
