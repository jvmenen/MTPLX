# Vondsten om nog naar te kijken

Lopende lijst. Per vondst: waar het vandaan komt en wat de volgende stap is. Afgehandelde punten gaan naar de onderste sectie, met datum en uitkomst.

## Open

| # | Vondst | Bron | Volgende stap |
|---|---|---|---|
| 2 | Uitschieters van ~1,5 s op korte generaties | [live-test](2026-09-26-live-test-classifier.md) | Weg met logprobs (p90 590 ms), past bij blank retries; nog apart bewijzen door zonder logprobs met en zonder `seed` te meten |
| 3 | `/v1/completions` gebruikt de session bank niet | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md), [live-test](2026-09-26-live-test-classifier.md) | Werkt achter `MTPLX_COMPLETIONS_SESSION_BANK=1` (G3: 276 tokens hergebruikt, 37% sneller); geheugengebruik meten |
| 4 | Hergebruik alleen in stappen van 512 tokens; 128 instellen via env en CLI werkte niet | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Oorzaak gevonden (drempel opgehoogd tot blokgrootte, GDN-grensraster, korte prompts zonder grenzen); vlag `--ram-session-prefix-min-match-tokens 128` gebouwd; live toetsen en geheugengebruik per grens meten |
| 5 | Chatroute zonder voorgevuld `Antwoord:` is minder nauwkeurig (0,58 tegen 0,68) | idem | Kijken of de chatroute een voorgevuld assistent-antwoord ondersteunt, of een ander promptpatroon dat hetzelfde doet |
| 6 | `score_prompt_logprobs` bouwt per blok een volledige float32-log-softmax | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Opgelost in 19 |
| 7 | `_run_generation` (~650 regels) en de completions-handler (~500 regels) zijn lastig leesbaar | idem | Kandidaat voor een aparte opschoon-PR: antwoordopbouw en streamlus uitlichten |
| 8 | Plafond van de session bank zakt naar 1 GiB door een piekreserve die nooit daalt | [metingen-classifier](2026-09-25-metingen-classifier.md) | Nagaan of de reserve na verloop van tijd mag afnemen (piek resetten na een rustige periode) |
| 9 | Eén verzoek tegelijk (`decode_batch_max = 1`) | idem | Uitzoeken of batching met MTP samen kan, en wat het oplevert voor meerdere agents |
| 10 | 19 tests falen op een schone `main` in deze omgeving (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`) | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Oorzaak bekijken (lokale modelcache?); eventueel melden bij de maker |
| 11 | Health meldt `ssd_prefix_miss` terwijl de echte oorzaak in het werkgeheugen ligt | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Echte afwijsreden rapporteren; kandidaat voor een kleine upstream-PR |
| 12 | `common_prefix_len` is een Python-lus | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Gedaan op `refactor/prefix-helpers` (tot 3,4x sneller); vierde kopie in `openai._common_prefix_len` nog omzetten |
| 13 | Gedupliceerde env-lezers en magische 512 | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Grotendeels gedaan; resterend: cold-tier-constanten, `parse_args`-512, bank-limieten, losse bool-parsers |
| 14 | `restore_or_prefill_prompt_state`, `_restore_near_prefix_prompt_state` en `near_prefix_candidates` zijn honderden regels | idem | Opsplitsen per herstelroute; aparte opschoon-PR |
| 15 | Beide takken samengevoegd en live getest | [live-test](2026-09-26-live-test-classifier.md) | Klaar; zie 16 en 17 |
| 16 | Vraag en labels vooraan (nodig voor hergebruik) kost rangschikkingskwaliteit (AUROC ~0,93 tegen 0,965) | [live-test](2026-09-26-live-test-classifier.md) | Andere promptpatronen proberen die hergebruik en een goede rangschikking combineren; per vraag toetsen |
| 17 | G2 hergebruikte niets, G3 met hetzelfde begin wel | idem | Uitzoeken hoe het dominante gedeelde begin wordt gekozen bij gemengde prompts |
| 18 | `engine_session` hoogt de drempel nog op tot de blokgrootte, `session_bank` niet meer | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Nagaan of dat bedoeld is |
| 19 | Scoring: top-K via blokmaxima (commit A) | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Echt getoetst: bitgelijk (243/243, verschil 0), ~10% sneller; klaar voor PR |
| 20 | Scoring: trunk in prefill-blokken (commit B) | idem | Echt getoetst: 1,25× sneller maar FOUT (213/243, logprob-verschil 19,85, ook binnen het eerste blok); oorzaak in onderzoek, niet indienen |
| 21 | GDN-grenzen vastleggen binnen de forward voor A3B (nu alleen qwen4_exp) | idem | Grotere klus; ~50-80 ms per warme agentbeurt |
| 22 | Chat-encode-memo per segment | idem | Kleine PR; 40-55 ms per beurt bij 77k tokens (gemeten basis) |
| 23 | `sessionbank_put_s` publiceren; put en laatste commit van het kritieke pad halen | idem | Eerst zichtbaar maken (kleine PR), dan verplaatsen |
| 24 | Laatste pending-commit overslaan voor `anon-completions` | idem | Kleine PR; ~10-20 ms per classificatie |
| 25 | Vaste overhead van ~50 ms per verzoek onverklaard | idem | Client-kloktijd vs `prompt_eval_time_s`/`server_elapsed_s`; py-spy (sudo) |

## Afgehandeld

| Datum | Vondst | Uitkomst |
|---|---|---|
| 2026-09-25 | Vraag en labels vooraan in de prompt voor hergebruik | Slechter op beide modellen (Balance 0,70 tegen 0,72; AUROC 0,92 tegen 0,96); huidige volgorde blijft |
| 2026-09-25 | Alleen de laatste positie teruggeven bij prompt-scoring | Levert niets op: top-1 of top-20 en de JSON-grootte maken geen meetbaar verschil |
| 2026-09-26 | Eerste-token-logprobs (vondst 1) | PR ingediend: https://github.com/youssofal/MTPLX/pull/530 |
