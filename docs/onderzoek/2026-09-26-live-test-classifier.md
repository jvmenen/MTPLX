# Live test classifier op de testtak (26 september 2026)

Testtak `test/classifier-live` = `feat/first-token-logprobs` (3c3f3aaf) samengevoegd met `feat/prefix-reuse-block` (fe32206c), zonder conflicten; alle nieuwe tests van beide takken groen. Server gestart met de productie-instellingen van 2.11.3 plus `--ram-session-prefix-min-match-tokens 128` en `MTPLX_COMPLETIONS_SESSION_BANK=1`, model Qwen3.6-35B-A3B Optimized-Balance. Testset: 240 Nederlandse chatberichten, correctievraag met vijf labels.

## Varianten

| Code | Route | Prompt |
|---|---|---|
| S | prompt-scoring (`echo`, `max_tokens: 0`) | context, bericht, vraag en labels (huidig) |
| G1 | `/v1/completions`, `max_tokens: 1`, `logprobs: 20` | huidig |
| G2 | idem | vraag en labeluitleg vooraan |
| G3 | idem | vraag en labeluitleg vooraan, plus korte herhaling van vraag en labelnamen vlak voor `Antwoord:` |

## Resultaten

| | Nauwkeurigheid | AUROC correctie | Correctie @0,5 (precisie / recall) | p50 | p90 | max | Hergebruikt (mediaan) | Zelfde oordeel als S |
|---|---|---|---|---|---|---|---|---|
| S | 0,717 | **0,965** | 0,63 / 0,86 | 562 ms | 667 ms | 911 ms | n.v.t. | |
| G1 | 0,708 | 0,962 | 0,65 / 0,92 | 486 ms | 590 ms | 1.142 ms | 0 | 228/240 |
| G2 | 0,696 | 0,923 | 0,52 / 0,78 | 537 ms | 678 ms | 1.844 ms | 0 | 199/240 |
| G3 | **0,742** | 0,930 | 0,53 / 0,78 | **353 ms** | **468 ms** | **626 ms** | 276 tokens | 186/240 |

## Conclusies

- **Eerste-token-logprobs werken met echte gewichten.** G1 geeft vrijwel dezelfde uitkomst als prompt-scoring (228 van 240 gelijk, AUROC 0,962 tegen 0,965) en is 14% sneller in de mediaan.
- **De uitschieters van ~1,5 s zijn weg** op G1 (p90 590 ms). Dat ondersteunt de verklaring via blank retries, die bij logprobs-verzoeken uit staan. Niet apart bewezen.
- **Hergebruik van een kort promptbegin werkt**: G3 hergebruikt in de mediaan 276 tokens en is 37% sneller dan S (353 tegen 562 ms), met de laagste uitschieters.
- **Maar de volgorde kost rangschikkingskwaliteit op de correctievraag**: vraag en labels vooraan geven een AUROC van ~0,93 tegen 0,965, ook met de herhaling aan het eind. De herhaling (G3) herstelt wel de nauwkeurigheid (0,742, de hoogste).
- **Praktisch**: voor het voorfilter van de nachtelijke review (recall op correctie telt) is G1 de beste keuze: sneller, zonder kwaliteitsverlies. Voor snelheidskritische vragen zoals een router is G3 het proberen waard, per vraag te toetsen.

## Open punten uit deze test

- **G2 hergebruikte niets** (mediaan 0), terwijl G3 met hetzelfde begin wel hergebruikte. Oorzaak onbekend: mogelijk bepaalt het "dominante gedeelde begin" in de bank zich op de G3-prompts, of wint de laatst opgeslagen variant. Nog uitzoeken.
- Het geheugengebruik per opgeslagen herstelpunt is niet gemeten.
- De testserver draaide 2.12.0 (hoofdtak), de productie 2.11.3; S op beide versies lag dicht bij elkaar (0,717 tegen 0,72).
