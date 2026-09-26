# Compiled verify op het 6-bit Balance-model (26 september 2026)

Bij het starten meldt de server op Qwen3.6-35B-A3B Optimized-Balance `compiled-verify permanent-eager: quant_bits_gate:bits=6`. De MTP-verify loopt op dit model daardoor altijd eager in plaats van gecompileerd. Vraag: hoeveel decodesnelheid kost dat, en kan de gecompileerde verify veilig ook voor 6-bit werken? Onderzoek en metingen door een Opus-agent op de echte server.

## Oorzaak van de gate (code en geschiedenis gelezen)

- **De gate:** `mtplx/graphbank.py:1878-1889` (`_compiled_verify_bits_gate_ok`) laat alleen 4-bit en 8-bit toe, ongekwantiseerde modellen en de gecontroleerde Prism-2-bit van Ternary Bonsai. De bits komen uit de eerste gekwantiseerde projectie van de trunk (`_runtime_trunk_quant_bits`, `graphbank.py:1814-1845`). Bij elk ander getal zet `CompiledVerifyBank.__init__` (`graphbank.py:2056-2065`) de bank permanent op eager, met reden `quant_bits_gate:bits=6`.
- **Waarom 6-bit buiten de lijst valt:** voorzichtigheid, geen fout. Het commentaar zegt "Unmeasured quantizations (e.g. the 6-bit 9B) stay eager until measured". De gate kwam in f2b27c03 (2.0.0, 4 juli 2026). In 2.0.1 (b3732cce, CHANGELOG `## [2.0.1]`, `docs/releases/v2.0.1.md:30`) kwamen de 6-bit verify-kernels (split-K hexpack) erbij, met de zin "Compiled verify stays off for 6-bit models", zonder meting of fout erbij. De gecompileerde decode-stack voor de 35B-A3B kwam pas in 2.4.0. De 6-bit Balance is met compiled verify dus nooit gemeten.
- **In `mistakes/` en de CHANGELOG** staat geen fout of regressie van compiled verify op 6-bit, en ook geen kernel die 6-bit niet ondersteunt. De 6-bit-kernels die eager draaien, lopen gecompileerd door dezelfde aanroepen.
- **Er bestaat al een override:** `MTPLX_COMPILED_VERIFY_FORCE=1` (`graphbank.py:1879`, getest in `tests/test_graphbank_compiled_verify.py:2443`). Ook `MTPLX_COMPILED_VERIFY=parity2` omzeilt de gate, voor diagnose. Er is dus niets gebouwd.
- Het turbo-profiel zet `MTPLX_COMPILED_VERIFY_MAX_CONTEXT=32768`. In de metingen liep de verify met de override ook bij ~60K context volledig gecompileerd (49/49 verify-aanroepen, 0 fallbacks).

## Opzet

- **Hardware en model:** Mac17,9, 64 GB; Qwen3.6-35B-A3B Optimized-Balance (6-bit/g64, routergates 8-bit), productie-argumenten, fanmodus default.
- **Code:** integratietak `perf/integratie` via de eindbench (`start.zsh nieuw`): productie-argumenten plus `--ram-session-prefix-min-match-tokens 128` en `MTPLX_COMPLETIONS_SESSION_BANK=1`. Variant: plus `MTPLX_COMPILED_VERIFY_FORCE=1`. De gate is op deze tak gelijk aan `main` (1de2b1c0).
- **Werklasten per variant, verse server en eigen SSD-cachemap:**
  - eindbench `decode,chat,agent` met `--agent-temperature 0`;
  - een gretige reeks (`~/Dev/laya-nl/compiled-verify/gretig.py`): 24 korte prompts (16 tot 30 tokens) en 6 middellange (1,7K tot 5,8K tokens), temperatuur 0, 256 tokens, met acceptatie en de compiled-verify-tellers per verzoek.
- **Vergelijking:** tokens met de Qwen-tokenizer, per stream het aantal gelijke tokens vanaf het begin. Na elke werklast is `requests_completed` gecontroleerd: alle gebruikte runs kloppen (bench 29, daarna gretig 30 verzoeken), het platform heeft niet meegemeten.
- **Meetfouten van onze kant, runs weggelaten:**
  - Een wachtlus met een `pgrep`-patroon dat nooit trof, liet twee meetreeksen een paar minuten door elkaar lopen. Basis-2 en de bench van force-1 zijn daarom weggelaten. Force-1 telt alleen mee voor de correctheid: de server draaide met de override en verwerkte 3633 van 3633 verify-aanroepen gecompileerd.
  - Tijdens de reeks kreeg de integratietak een nieuwe commit van een andere agent (54c4ca5f, alleen chat-encode). Basis-1 draaide op 89a07905, de overige runs op 54c4ca5f. Die commit raakt de verify niet; de gretige uitvoer van basis-1 en basis-3 is exact gelijk.

## Uitkomst

Alle getallen gemeten.

| Werklast | Eager (basis-1, basis-3) | Compiled (force-2, force-3) | Verschil |
|---|---|---|---|
| Decode 512 tokens, korte prompt (tok/s, mediaan van 3) | 81,0 / 80,1 | 82,6 / 81,3 | +1,7% |
| Gretig kort, 24 × 256 tokens (tok/s, mediaan) | 81,9 / 81,8 | 82,7 / 82,6 (force-1: 82,6) | +1% |
| Gretig middel, 6 × 256 tokens, 1,7K tot 5,8K context (tok/s, mediaan) | 85,6 / 88,4 | 89,7 / 90,1 (force-1: 89,0) | +3% |
| Chat, decode lange beurten tot ~60K (tok/s, mediaan) | 79,9 / 80,9 | 82,6 / 82,3 | +2,5% |
| Chat, TTFT lang / kort (s) | 7,92 / 0,42 en 8,34 / 0,40 | 7,99 / 0,40 en 8,32 / 0,39 | binnen ruis |
| Agent tot ~60K, totaal (s) | 53,1 / 54,1 | 53,2 / 54,0 | 0% |
| Geheugen na agent (actief / piek) | 36,13 / 38,71 GiB | 36,13 / 38,72 GiB | gelijk |

- **Acceptatie MTP:** exact gelijk (kort 0,519, middel 0,658, zelfde aantallen per stream).
- **Gretige reeks:** 30 van 30 streams token voor token gelijk aan eager, in alle drie de force-runs.
- **Parity2** (`MTPLX_COMPILED_VERIFY=parity2`, gecompileerde tak leidend, eager kloon vergelijkt elke ronde): 3633 verify-aanroepen, 0 afwijkend, uitvoer 30/30 gelijk.
- **Eindbench-streams:** 26 van 27 gelijk. Eén chatbeurt bij ~16K context (prompt 16.248 tokens) wijkt na 91 van 128 tokens af, in force-2 en force-3 op precies dezelfde plek. Geen toeval dus: bij langere context geeft de gecompileerde verify net andere afronding dan eager, en de MoE-routering versterkt dat (zelfde mechanisme als vondst 29). De tekst blijft zinnig, alleen de volgorde van een opsomming verschilt.

## Oordeel

- **Kosten van de gate:** klein. Compiled verify zou op dit model 1 tot 3% decode opleveren, bij agentbeurten niets: daar domineert de prefill en is de uitvoer kort.
- **Veiligheid:** niet principieel onjuist. Onder ~6K context is compiled bitgelijk aan eager (30/30, parity2 zonder afwijking). Bij lange context is het niet meer bitgelijk (1 van 12 chatbeurten).
- **Drempel (≥5% en bitgelijk):** niet gehaald, op beide punten. Geen gate-wijziging, geen commit; de tak `perf/compiled-verify-6bit` is leeg gebleven.

## Aanbeveling voor de eindconfig

Niets toevoegen: laat `MTPLX_COMPILED_VERIFY_FORCE` uit. Wie 1 tot 3% decode wil ruilen tegen niet-bitgelijke uitvoer bij lange context, zet `MTPLX_COMPILED_VERIFY_FORCE=1` in de env van de server; verder niets nodig.

## Meetmateriaal (lokaal, niet in de fork)

`~/Dev/laya-nl/compiled-verify/`: `gretig.py`, `vergelijk.py`, `run.zsh`/`run2.zsh`, `resultaten/` (gretige runs en serverlogs). Eindbench-resultaten: `~/Dev/laya-nl/eindbench/resultaten/cv-*.json`.
