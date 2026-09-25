# Vondsten om nog naar te kijken

Lopende lijst. Per vondst: waar het vandaan komt en wat de volgende stap is. Afgehandelde punten gaan naar de onderste sectie, met datum en uitkomst.

## Open

| # | Vondst | Bron | Volgende stap |
|---|---|---|---|
| 2 | Uitschieters van ~1,5 s op korte generaties | [live-test](2026-09-26-live-test-classifier.md) | Weg met logprobs (p90 590 ms), past bij blank retries; nog apart bewijzen door zonder logprobs met en zonder `seed` te meten |
| 3 | `/v1/completions` gebruikt de session bank niet | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md), [live-test](2026-09-26-live-test-classifier.md) | Werkt achter `MTPLX_COMPLETIONS_SESSION_BANK=1` (G3: 276 tokens hergebruikt, 37% sneller); geheugengebruik meten |
| 4 | Hergebruik alleen in stappen van 512 tokens; 128 instellen via env en CLI werkte niet | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Oorzaak gevonden (drempel opgehoogd tot blokgrootte, GDN-grensraster, korte prompts zonder grenzen); vlag `--ram-session-prefix-min-match-tokens 128` gebouwd; live toetsen en geheugengebruik per grens meten |
| 5 | Chatroute zonder voorgevuld `Antwoord:` is minder nauwkeurig (0,58 tegen 0,68) | idem | Kijken of de chatroute een voorgevuld assistent-antwoord ondersteunt, of een ander promptpatroon dat hetzelfde doet |
| 7 | `_run_generation` (~650 regels) en de completions-handler (~500 regels) zijn lastig leesbaar | idem | Kandidaat voor een aparte opschoon-PR: antwoordopbouw en streamlus uitlichten |
| 8 | Plafond van de session bank zakt weg | [bankplafond](2026-09-26-bankplafond.md) | Deels weerlegd: reserve kost ~7 GiB, 1 GiB vraagt ook een groot werkgeheugen. Fix gebouwd achter `MTPLX_SESSION_BANK_SPIKE_BURSTS` (fb1cd817, standaard uit); hardwareverificatie volgens rapport. Handmatige omweg: cache leegmaken via de beheerroute reset de piek |
| 9 | Eén verzoek tegelijk (`decode_batch_max = 1`) | idem | Uitzoeken of batching met MTP samen kan, en wat het oplevert voor meerdere agents |
| 10 | 19 tests falen op een schone `main` in deze omgeving (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`) | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Oorzaak bekijken (lokale modelcache?); eventueel melden bij de maker |
| 11 | Health meldt `ssd_prefix_miss` terwijl de echte oorzaak in het werkgeheugen ligt | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Echte afwijsreden rapporteren; kandidaat voor een kleine upstream-PR |
| 12 | `common_prefix_len` is een Python-lus | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Gedaan op `refactor/prefix-helpers` (tot 3,4× sneller); vierde kopie in `openai._common_prefix_len` nog omzetten |
| 13 | Gedupliceerde env-lezers en magische 512 | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Grotendeels gedaan; resterend: cold-tier-constanten, `parse_args`-512, bank-limieten, losse bool-parsers |
| 14 | `restore_or_prefill_prompt_state`, `_restore_near_prefix_prompt_state` en `near_prefix_candidates` zijn honderden regels | idem | Opsplitsen per herstelroute; aparte opschoon-PR |
| 16 | Vraag en labels vooraan (nodig voor hergebruik) kost rangschikkingskwaliteit (AUROC ~0,93 tegen 0,965) | [live-test](2026-09-26-live-test-classifier.md) | Andere promptpatronen proberen die hergebruik en een goede rangschikking combineren; per vraag toetsen |
| 17 | G2 hergebruikte niets, G3 met hetzelfde begin wel | idem | Uitzoeken hoe het dominante gedeelde begin wordt gekozen bij gemengde prompts |
| 18 | `engine_session` hoogt de drempel nog op tot de blokgrootte, `session_bank` niet meer | [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Nagaan of dat bedoeld is |
| 21 | GDN-grenzen vastleggen binnen de forward voor A3B (nu alleen qwen4_exp) | idem | Grotere klus; ~50-80 ms per warme agentbeurt |
| 22 | Chat-encode-memo per segment | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Afgerond op de tak (`9dd094d8`, gepusht): op de echte server bij een agentgesprek met tools `_encode_messages` bij 82K van 107 naar 57 ms, TTFT min prefill vanaf 42K 19-56 ms lager, uitvoer exact gelijk; suite gelijk aan de nulmeting. PR-tekst klaar; wacht op akkoord Jeroen |
| 23 | `sessionbank_put_s` publiceren; put en laatste commit van het kritieke pad halen | idem | Eerst zichtbaar maken (kleine PR), dan verplaatsen |
| 24 | Laatste pending-commit overslaan voor `anon-completions` | idem | Kleine PR; ~10-20 ms per classificatie |
| 25 | Vaste overhead van ~50 ms per verzoek onverklaard | idem | Client-kloktijd tegen `prompt_eval_time_s`/`server_elapsed_s`; py-spy (sudo) |
| 26 | Jinja-template-render kost ~43 ms bij 82K tokens en domineert na de memo de chat-encode | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Incrementeel renderen onderzoeken (grotere wijziging) |
| 28 | Werkgeheugen W in rust onbekend; als W ≥ 9 GiB is dat het echte probleem | [bankplafond](2026-09-26-bankplafond.md) | `/v1/mtplx/snapshot` → `mem.generation_working_bytes` uitlezen tijdens gewoon gebruik door Bink en Fleur |
| 29 | MoE-routering op A3B is niet batch-invariant: de router-matmul geeft per aantal rijen net andere bf16-afronding (oorsprong in de GEMM-kernels van MLX), en omdat de 8e/9e expert vaak bijna gelijk liggen, kiest een token dan een andere expert. Scores hangen daardoor ~0,2 nats/token af van de blokindeling | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Waarschijnlijk op te lossen in MTPLX: de router is klein (2048 × 256 per token), dus router-logits batch-onafhankelijk uitrekenen (vaste optelvolgorde, f32) kost vrijwel niets. **Nog niet starten** (Jeroen, 26 sep: eerst geen proef). Plan als het zover is: schakelaar op een eigen tak; meten of wisselingen tussen blok 256 en 2048 naar ~0 gaan; kwaliteit vergelijken (NLL, classifier-nauwkeurigheid op de 240 berichten, en gegenereerde teksten met en zonder schakelaar, **beoordeeld door Claude Opus 5.5**). Open vraag: is alleen de router genoeg, of voeden de expert-matmuls de ruis in diepere lagen? Daarna pas een issue bij MTPLX, met de meting; MLX alleen als de experts ook moeten. Een bredere trunk bij scoring (vondst 20) kan pas zinvol worden als dit is opgelost |
| 30 | Openstaande PR's [#530](https://github.com/youssofal/MTPLX/pull/530) (eerste-token-logprobs) en [#532](https://github.com/youssofal/MTPLX/pull/532) (top-K scoreroute) | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md), [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Reacties van de maker volgen en verwerken; bij conflicten door nieuwe commits op `main` de tak bijwerken (rebase) en de tests opnieuw draaien |
| 31 | Takken `feat/prefix-reuse-block` en `refactor/prefix-helpers`: gebouwd en (deels) live getoetst, nog geen besluit over een PR | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md), [opschonen](2026-09-26-opschonen-prefix-helpers.md) | Besluit Jeroen: indienen als (kleine) PR's, bijv. eerst de snellere prefixvergelijking los, of eerst vondst 4 en 3 afmaken (geheugengebruik meten) |
| 33 | `/health` toont geen actief- of piekgeheugen, alleen `session_bank.effective_max_bytes` | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md), [bankplafond](2026-09-26-bankplafond.md) | Voor geheugenmetingen `/v1/mtplx/snapshot` gebruiken (`mem.*`); eventueel kleine PR om de kernwaarden ook in `/health` te zetten |
| 34 | Met scoped redeneergeschiedenis (productie-instelling) wordt gewone chat zonder tools niet gesegmenteerd, dus de chat-encode-memo helpt daar niet (bij 80K ~110 ms encode per verzoek) | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Nagaan of er in scoped modus veilige segmentgrenzen zijn (bijvoorbeeld bij elke `<\|im_start\|>`); pas na vondst 22 |
| 35 | `/v1/messages` met een lang toolresultaat (~17K tokens): TTFT 26-45 s terwijl `prompt_eval_time_s` ~0,2 s is en de nieuwe tokens als `cached_tokens` tellen; via `/v1/chat/completions` kost een vergelijkbare beurt 5-10 s. Ook een tweede, niet-gesegmenteerde encode per verzoek in de streamworker (~110 ms bij 77K) | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Gelijk op main en de tak, dus los van vondst 22. Uitzoeken waar de tijd vóór de prefillmeting zit (postcommit-wachten, prefill buiten `prompt_eval_time_s`?) |

## Afgehandeld

| Datum | Vondst | Uitkomst |
|---|---|---|
| 2026-09-25 | Vraag en labels vooraan in de prompt voor hergebruik | Slechter op beide modellen (Balance 0,70 tegen 0,72; AUROC 0,92 tegen 0,96); huidige volgorde blijft |
| 2026-09-25 | Alleen de laatste positie teruggeven bij prompt-scoring | Levert niets op: top-1 of top-20 en de JSON-grootte maken geen meetbaar verschil |
| 2026-09-26 | Eerste-token-logprobs (vondst 1) | PR ingediend: https://github.com/youssofal/MTPLX/pull/530 |
| 2026-09-26 | Scoring: top-K via blokmaxima (vondst 19) | PR ingediend: https://github.com/youssofal/MTPLX/pull/532 |
| 2026-09-26 | Volledige float32-log-softmax per blok bij scoring (vondst 6) | Opgelost door de top-K via blokmaxima, PR #532 |
| 2026-09-26 | Eerste-token-logprobs en prefix-hergebruik samen live getest (vondst 15) | Zie [live-test](2026-09-26-live-test-classifier.md); vervolg in 16 en 17 |
| 2026-09-26 | Scoring: trunk in prefill-blokken (vondst 20) | Teruggedraaid: op A3B hangt de uitkomst af van de blokgrootte door MoE-routering (geen codefout); kan pas terugkomen als vondst 29 is opgelost, en dan als opt-in met standaard 256 |
| 2026-09-26 | Vijf lokale takken veiligstellen (vondst 32) | Gepusht naar de fork: `feat/prefix-reuse-block`, `refactor/prefix-helpers`, `feat/chat-encode-segment-memo`, `fix/bank-ceiling-peak-decay`, `test/classifier-live` |
| 2026-09-26 | Memo-sleutel merkt een ter plekke gewijzigde woordenschat niet op (vondst 27) | Opgelost op `feat/chat-encode-segment-memo`: `vocab_size` plus aantal toegevoegde tokens in de sleutel; `len(tokenizer)` kost 11 ms per aanroep op de Qwen3.6-tokenizer en viel daarom af |
