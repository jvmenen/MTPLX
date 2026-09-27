# Vondst 60 en 52: chat-decode tegen 2.11.3 en overhead buiten `elapsed_s`

Meting 27 sep, 18:30 tot na 19:35, als platform-script-taak: `~/Dev/laya-nl/decode-onderzoek/` (`runall.zsh`, opzet, hypotheses en beslisregels in de README daar). Mac17,9 (M5 Pro, 64 GB), Qwen3.6-35B-A3B Balance, productie-argumenten, fans standaard. Voorbereiding (code, git-geschiedenis, bestaande eindbench-data) staat in die README; dit rapport bevat alleen wat de meting opleverde.

## Kern

- **De meting is grotendeels mislukt, door de stroomvoorziening.** De laptop draaide op de accu (om 16:42 nog 100%) met een USB-lader van 8 W (5 V, 1,5 A). Om 19:35 stond de accu op 4% en liep hij nog steeds leeg. Vanaf ~18:42 zakte de decode van ~75 naar 8 tot 12 tok/s; al vóór dat moment lag de rondetijd ~33% boven die van 26 sep.
- **Bruikbaar:** alleen de eerste run `o1-oud` (2.11.3) en het begin van `m1-main` (2.12.0, `decode_t0` en `chat_t0`, tot 18:39). Die bevestigen het beeld uit de eindbench: gelijke rondetijd, meer rondes in precies de beurten waar de tekst afwijkt.
- **Niet gemeten:** de varianten `main-dense`, `main-gate` en `main-beide` (de vraag welke wijziging de chat-decode terugbrengt naar 2.11.3) en de hele meting van vondst 52. Beide vragen staan nog open.
- **Om 19:35 draaide `runall.zsh` nog** (bij run 3 van 11, `i1-integratie`, op 9 tot 13 tok/s), met een server op poort 8000. Een tweede start van dezelfde taak om 19:30 stopte zichzelf ("runall draait al"). Stoppen van de lopende run is aan Jeroen.
- **Advies:** opnieuw draaien met een volwaardige lader (of aan het net met de meegeleverde adapter), `pmset -g batt` vooraf en achteraf vastleggen, en de runs `o1`, `m1` en `i1` weggooien (`RUNALL_OPNIEUW=1`).

## 1. Wat er gebeurde

| Tijd | Run | Decode `decode_t0` (server) | Opmerking |
|---|---|---|---|
| 18:31-18:37 | o1-oud (2.11.3) | 73,8 tok/s, 266 rondes | Compleet, stabiel |
| 18:37-18:42 | m1-main (2.12.0) | 78,0 tok/s, 250 rondes | `decode_t0`, `decode_std`, `chat_t0` en de eerste pass van `chat_std` bruikbaar |
| 18:42-19:17 | m1-main | 8 tot 32 tok/s | Rest van `chat_std` onbruikbaar; run duurde 40 in plaats van ~6 minuten |
| 19:17-19:35+ | i1-integratie | 9 tot 13 tok/s | Helemaal onbruikbaar |

- Ter vergelijking 26 sep (eindtest 2): 2.11.3 `decode_t0` 266 rondes bij 19,6 ms per ronde (~98 tok/s). Op 27 sep bij o1: 266 rondes (gelijk) bij ~26,1 ms per ronde (+33%). De machine was dus al vóór de ineenstorting trager, vermoedelijk door de stroombeperking.
- Geen thermische waarschuwing (`pmset -g therm`). Stroombron volgens `ioreg`: `"Watts"=8`, `"Description"="usb brick"`, laadstroom negatief, accu 4%. De server draaide met nice 5 (zsh zet achtergrondtaken standaard op nice 5, `BG_NICE`); dat verklaart de daling niet, want de GPU trekt zich daar niets van aan en o1 liep onder dezelfde instelling goed.
- De eerdere metingen van die middag (voorrang 17:36-17:58, Gemma-scoring 17:58-18:21, herstelpaden 18:21-18:31) draaiden ook op de accu. Daar gaat het om vergelijkingen binnen dezelfde sessie (standen om en om, tellingen, bitgelijkheid); absolute tijden liggen 5 tot 10% lager dan op 26 sep. Zie de rapporten [voorrang](2026-09-26-voorrang.md) en [herstelpaden-en-gemma-scoring](2026-09-27-herstelpaden-en-gemma-scoring.md).

## 2. Vondst 60: wat de bruikbare runs laten zien

Uit `logs/meting-o1-oud.log` en `logs/meting-m1-main.log` (gemeten, beide vóór 18:42, onder dezelfde stroombeperking):

**Korte prompt, 512 tokens, gretig (`decode_t0`, 3 keer, alle drie gelijk):**

| | 2.11.3 (o1) | 2.12.0 (m1) | Verschil |
|---|---|---|---|
| Verify-rondes | 266 | 250 | −6% |
| Tokens per ronde | 1,92 | 2,05 | +6,5% |
| Decode (server) | 73,8 tok/s | 78,0 tok/s | +5,7% |
| Rondetijd (afgeleid: 512 / tok/s / rondes) | 26,1 ms | 26,3 ms | +0,7% |

**Chatgesprek, 12 beurten van 128 tokens, gretig (`chat_t0`):**

| Beurt | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Totaal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rondes 2.11.3 | 53 | 49 | 54 | 52 | 54 | 53 | 55 | 49 | 55 | 50 | 55 | 48 | 627 |
| Rondes 2.12.0 | 60 | 50 | 54 | 51 | 54 | 52 | 54 | 57 | 55 | 53 | 57 | 50 | 647 |
| Copy-tokens geaccepteerd 2.11.3 / 2.12.0 | 3 / 0 | 11 / 9 | 2 / 2 | 8 / 8 | 2 / 2 | 8 / 8 | 2 / 2 | 11 / 0 | 2 / 2 | 8 / 0 | 2 / 2 | 11 / 11 | |

- Het aantal rondes is exact dat van de eindbench (627 tegen 647, +3,2%): het verschijnsel is reproduceerbaar.
- De extra rondes zitten in beurt 0, 7 en 9 (+7, +8, +3), precies waar context-copy op 2.12.0 niets meer vindt (0 geaccepteerde copy-tokens). Dat past bij hypothese 1 van de README: de gretige tekst kantelt, en de nieuwe tekst is minder goed te drafen.
- Mediaan decode over de 12 beurten (server): 76,1 tok/s op 2.11.3 tegen 73,6 op 2.12.0 (−3,2%).
- Rondetijd gelijk (+0,7% bij de korte prompt): geen trager rekenen.

**Niet beantwoord:** welke upstream-wijziging de tekst laat kantelen (de grensindeling van de prefill, de verify-numeriek uit e36f5ffb, of beide). Daarvoor zijn de runs `main-dense`, `main-gate` en `main-beide` nodig. Ook de toets op serverstandaard (`chat_std`, temperatuur 0,6 over seeds) is niet bruikbaar: die liep bij m1 grotendeels na 18:42.

## 3. Vondst 52: niet gemeten

De runs `h1-52-hook` en `k1-52-kaal` komen in `runall.zsh` na de negen runs van vondst 60 en zijn niet aan de beurt geweest. De hypotheses (postcommit-controle die het hele gesprek opnieuw codeert, ~19 ms bij 26K tot ~61 ms bij 80K tokens, geschat uit eerdere encode-metingen) blijven ongetoetst.

## 4. Volgende stap

- Lader: een adapter van minstens 70 W, of de meegeleverde; vóór de start `pmset -g batt` en `ioreg -rn AppleSmartBattery | grep AdapterDetails` in het log laten zetten, en afbreken als `Watts` onder 60 ligt. Een kleine controle in `serve.zsh` of `runall.zsh` voorkomt herhaling.
- De mislukte resultaten weggooien en opnieuw draaien (`RUNALL_OPNIEUW=1`), zelfde volgorde. Geschat 40 tot 65 minuten.
- De rest van het plan (beslisregels in `analyse.py`) blijft ongewijzigd.

## Bestanden

- `~/Dev/laya-nl/decode-onderzoek/`: `README.md` (opzet), `runall.zsh`, `bench60.py`, `meet52.py`, `analyse.py`; `logs/meting-*.log` (per verzoek), `resultaten/o1-oud.json`, `m1-main.json` (deels bruikbaar), `i1-integratie.json` (onbruikbaar).
