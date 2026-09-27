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

## 5. Meting 27 september avond (20:38-21:49, 100 W-lader)

Volledige herhaling na de mislukte poging in §1-4, nu met een 100 W-lader in plaats van de 8 W USB-lader. Alle elf runs uit `runall.zsh` doorlopen: `o1-oud`, `m1-main`, `i1-integratie`, `d1-main-dense`, `g1-main-gate`, `b1-main-beide`, `i2-integratie`, `m2-main`, `o2-oud`, `h1-52-hook`, `k1-52-kaal` (`runall.log`, `analyse.txt`).

### 5.1 Welke varianten compleet waren

Tussen ongeveer 21:38 en 21:41 draaide per ongeluk een tweede meetscript mee op poort 8000. Dat heeft twee runs echt beschadigd:

- `i2-integratie` (tweede run van de integratietak): de server gaf halverwege "connection refused"; alleen `decode_t0` en `decode_std` haalden hun volledige aantal verzoeken, de chatmetingen niet.
- `m2-main` (tweede run van main): twee mtplx-servers tegelijk actief op poort 8000, meting direct afgebroken (resultaatbestand vrijwel leeg, 723 bytes).

Beide zijn bij de analyse buiten beschouwing gelaten. De runs die er vlak na kwamen (`o2-oud`, `h1-52-hook`, `k1-52-kaal`) liepen daarna schoon door: exit 0, juiste aantal verzoeken, geen tracebacks. `runall.zsh` merkt in het log toch elke run als "METING ONVOLLEDIG", inclusief deze drie: dat label komt van de geforceerde herhaalvlag (`RUNALL_OPNIEUW`, die de voltooid-check altijd op "nee" zet, ook voor geslaagde runs) en betekent hier niet dat de meting mislukt is. De resultaatbestanden van `o2-oud`, `h1-52-hook` en `k1-52-kaal` hebben een geldig `klaar`-tijdstip en het verwachte aantal verzoeken; alleen `i2-integratie` en `m2-main` hebben een echt VERSTOORD-signaal in het log.

Bruikbare tellingen per variant:

| Variant | Runs die meetellen |
|---|---|
| oud | 2 volledig (o1 + o2) |
| main | 1 volledig (m1); de herhaling (m2) is weggegooid |
| main-dense, main-gate, main-beide | elk 1 volledige run |
| integratie | 1 volledig (i1) plus de decode-tellingen van i2 (chat ontbreekt daar) |
| 52-hook, 52-kaal | elk 1 volledige run, 42 verzoeken |

### 5.2 Vondst 60: systematisch verschil of tekstgeluk

Chatgesprek, gretig (`chat_t0`, mediaan over alle beurten, tegen 2.11.3):

| Variant | Server-decode (tok/s) | Tokens per ronde | Ronde­tijd (ms) | Gelijke gretige antwoorden tegen 2.11.3 |
|---|---|---|---|---|
| main | -4,0% | -3,1% | +0,6% | 4/15 |
| main-dense | +6,8% | -2,3% | -8,4% | 3/15 |
| main-gate | +5,5% | -3,4% | -9,7% | 9/15 |
| main-beide | +10,3% | +0,0% | -10,0% | 15/15 |
| integratie | -4,2% | -3,1% | +5,5% | 3/15 |

Onder serverstandaard (`chat_std`, temperatuur 0,6, 36 tot 72 verzoeken per variant) verdwijnt het verschil in tokens per ronde volledig: alle 95%-betrouwbaarheidsintervallen bevatten 0 (bijvoorbeeld integratie -1,3% [-3,9%, +1,3%], main -1,0% [-3,4%, +1,4%]).

**Conclusie: het verschil bij greedy decoderen (t0) op main en integratie is tekstgeluk, geen systematische regressie.** De rondetijd zelf is gelijk of korter; het verschil in tok/s komt puur doordat de tekst net iets anders uitvalt (minder rondes met een geaccepteerd copy-token), en dat effect verdwijnt zodra je middelt over de serverstandaard-bemonstering.

De uitzondering is **main-beide** (de dense-route-wijziging en de gate-route-wijziging samen toegepast): 100% van de gretige antwoorden is daar bit-voor-bit gelijk aan 2.11.3, tegen 20% voor alleen de dense-fix en 60% voor alleen de gate-fix. Tegelijk is main-beide niet trager (rondetijd -10,0%, dus zelfs iets sneller) en gelijk in tokens per ronde (+0,0%). Dat is een aanwijzing sterk genoeg om niet als toeval af te doen: de combinatie van de verify-numeriek uit e36f5ffb en de prefill-grensindeling samen verklaart en herstelt de tekstverschuiving tussen 2.11.3 en 2.12.0 volledig; los toegepast doet geen van beide dat.

**Kanttekening:** de rondetijd drift tussen twee runs van dezelfde variant tot 25,1% (oud zelf: 28,53 naar 24,96 ms bij `decode_t0`, +14,3%; 39,94 naar 31,94 ms bij `chat_t0`, +25,1%). Verschillen tussen varianten van een paar procent vallen binnen die ruis; alleen de main-beide-uitkomst (100% identieke tekst, consistente rondetijdwinst) staat daar duidelijk boven.

### 5.3 Vondst 52: verdeling van het gat en de postcommit-hercodering

Client-tijd min `elapsed_s` (mediaan, ms), `k1-52-kaal` (zonder meethook, dichtst bij productie) tegen `h1-52-hook` (met meethook):

| Cel | Gat zonder hook (k1) | Gat met hook (h1) | Hook-kosten (totale clienttijd) |
|---|---|---|---|
| chat kort, stream | 7,5 | 8,6 | +11,3 ms |
| chat kort, json | 7,5 | 8,4 | +18,0 ms |
| chat lang, stream | 47,0 | 51,9 | +14,3 ms |
| chat lang, json | 51,9 | 54,1 | +5,7 ms |
| completion kort, stream | 2,6 | 2,9 | +5,2 ms |
| completion kort, json | 2,6 | 2,8 | +4,5 ms |

**De fase-uitsplitsing is niet gelukt.** De hook logde per run 2299 tot 2314 losse gebeurtenissen (`send`, `fn`, `endpoint`, `submit`, `recv`, …), maar `analyse.py` kon voor geen van de 42 verzoeken in beide runs een bruikbaar `voor`/`na`-faseoverzicht reconstrueren ("0 met fasetijden"). Of de postcommit-hercodering van het hele gesprek daadwerkelijk de grootste post is, blijft daardoor onbeantwoord: er is geen betrouwbare verdeling van het gat over fasen, alleen het totale gat per cel.

Wat wel duidelijk is: het gat schaalt sterk met gespreks-/contextlengte. Kort gesprek (weinig geschiedenis): ~7,5 ms. Lang gesprek: 47,0 tot 51,9 ms. Completion zonder geschiedenis: 2,6 ms, nauwelijks meer dan bij een kort gesprek zonder de rest van de bagage. Dat past bij de hypothese dat het vooral aan iets ligt dat met de opgebouwde gespreksgeschiedenis samenhangt (zoals de postcommit-hercodering), maar is met deze meting niet hard bewezen.

**Kanttekening:** de meethook zelf kost tot 18,0 ms per verzoek (chat kort, json). Dat is dezelfde orde van grootte als een deel van het gat dat gemeten wordt; kleine verschillen tussen cellen of runs met en zonder hook niet hard toeschrijven aan echte serveroverhead.

### 5.4 Nieuwe vondst: platform-script-taken hebben een harde grens van 1 uur

Los van vondst 60/52: deze meting liep als platform-script-taak en botste op een grens die niet in `timeout` zit. Zie vondst 62 in [VONDSTEN.md](VONDSTEN.md).

## Bestanden

- `~/Dev/laya-nl/decode-onderzoek/`: `README.md` (opzet), `runall.zsh`, `bench60.py`, `meet52.py`, `analyse.py`; `runall.log` en `analyse.txt` (meting 27 sep avond); `logs/meting-*.log` (per verzoek), `resultaten/o1-oud.json`, `m1-main.json` (deels bruikbaar), `i1-integratie.json` (onbruikbaar) (meting 27 sep 18:30, mislukt door de stroomvoorziening); `resultaten/o1-oud.json`, `o2-oud.json`, `m1-main.json`, `d1-main-dense.json`, `g1-main-gate.json`, `b1-main-beide.json`, `i1-integratie.json`, `h1-52-hook.json`, `k1-52-kaal.json` (meting 27 sep avond, bruikbaar); `resultaten/i2-integratie.json`, `m2-main.json` (verstoord door een tweede meetscript, niet meegeteld).
