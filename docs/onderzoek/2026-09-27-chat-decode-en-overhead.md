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

## 6. Afronding 27 september, laat (zonder server, alleen code en bestaande data)

### 6.1 Vondst 60: besluit en het enige open punt

De drie upstream-wijzigingen die samen de tekstverschuiving tussen 2.11.3 en 2.12.0 verklaren, hebben elk een eigen reden:

- **e36f5ffb (correctheid):** bij de verify-ronde zonder compile rekende de attention-gate in BF16 op een net andere manier dan dezelfde berekening mét compile. Op het 27B-model gaf dat afwijkende uitkomsten tussen die twee paden (67 sigmoid-waarden in de eerste laag). De commit laat het eager pad de gate precies zo uitrekenen als het gecompileerde pad, bit voor bit. Dat de tekst daardoor een fractie anders uitvalt dan op 2.11.3, is het gevolg van een fout die hersteld is.
- **9a86dd6c (snelheid):** een koude prompt hakte het laatste prefill-blok helemaal op in stukjes van 256 rijen; smalle forwards zijn traag. Nu is dat een ladder met bredere stappen: 4K-prompts ongeveer een kwart sneller in de prefill (metingen van de maker op een M5 Max).
- **4314638a (snelheid):** het dichtstbijzijnde herstelpunt ligt nu 64 tokens vóór het eind in plaats van tot 256. Een warme agentbeurt hoeft daardoor hooguit 64 tokens opnieuw te prefillen.

Andere tekst bij temperatuur 0 is daarbij onvermijdelijk (andere forwardvormen geven via de MoE-routering andere afronding, zie vondst 29) en onder serverstandaard verdwijnt het verschil in tokens per ronde volledig (§5.2).

**Besluit Jeroen (27 sep): niets terugdraaien, geen PR.** Vondst 60 is tekstgeluk, geen tragere berekening. De "gecombineerde fix" uit §5.2 (`main-beide`) was een meetvariant om de oorzaak te bewijzen, geen voorstel: hij zou een correctheidsfix en twee prefillversnellingen terugdraaien.

**Correctie op §5.2:** de rondetijdwinst van `main-beide` (−10%) en `main-dense` (−8%) is drift, geen effect van de variant. Verify-tijd per ronde bij `decode_t0` in volgorde van de avond: o1 22,05, m1 34,75 (uitschieter), i1 23,23, d1 20,89, g1 20,58, b1 20,65, i2 20,81, o2 19,50 ms. De machine werd in de loop van de avond sneller (o1 tegen o2: 2.11.3 zelf −12%).

**Open punt: kost de gecompileerde gate-helper van e36f5ffb tijd per verify-ronde?** De helper is een `mx.compile`-functie per attentielaag per verify-ronde (`mtplx/attention_math.py`, alleen in fase `decode_verify`). Indicatie uit dezelfde avond, runs vlak na elkaar: `d1-main-dense` (met helper) 20,89 ms verify per ronde tegen `g1-main-gate` en `b1-main-beide` (zonder) 20,58 en 20,65 ms, dus +1,2 tot +1,5%; `i2` (met helper) 20,81 ms. Dat valt binnen de drift en is niet hard. Meetscript voorbereid: `gatekosten.zsh` (zie §6.3).

### 6.2 Vondst 52: waarom de fase-uitsplitsing mislukte

**Hoofdoorzaak: twee klokken, geen functienamen.** Alle 17 gepatchte functienamen bestonden en vuurden (hooklog `h1-52-hook`: `"ontbreekt": []`, 435 `fn`-, 42 `dispatch`-, 57 `submit`-gebeurtenissen). De koppeling van gebeurtenissen aan verzoeken ging mis op de tijd:

- De client (`meet52.py`, `/usr/bin/python3` 3.9.6) meet met `time.perf_counter()`. In Python 3.9 telt die vanaf de start van het proces (clienttijden ~23 tot 50 s).
- De hook (`hook52/sitecustomize.py` versie 1, regel 54, venv-Python 3.13) meet met dezelfde functie, maar in 3.13 is dat `mach_absolute_time` sinds het opstarten van de Mac (~32.100 s).
- `analyse.py:497` zoekt per verzoek de hookgebeurtenissen in het venster `[begin, gesloten + 50 ms]` van de client. Door het verschil van ~32.000 s viel er niets in; `fasen()` stopt dan op `analyse.py:377` zonder `voor`/`na`, en de telling op `analyse.py:500` gaf "0 met fasetijden". De README beloofde dat `analyse.py` de klok controleert, maar die controle (`klok_ok`) liep pas ná een gevonden gebeurtenis.

**Drie fouten die daarna ook nog in de weg hadden gezeten** (integratietak `f26dc9a8`, `mtplx/server/openai.py`):

- Het venster van 50 ms na `gesloten` bevat het begin van het volgende verzoek (de meting stuurt ze direct achter elkaar); `analyse.py:388` pakt de laatste `dispatch` en koppelt zo in een deel van de gevallen de generatie van het volgende verzoek.
- Bij streaming chat gebeurt de bank-put niet in `_run_generation` (26653-26680) maar in een aparte postcommit-taak op de model-thread: `_store_generation_final_history_snapshot` via `_submit_foreground_model_work` met batch_key `postcommit.stream` (34986-35060). Die functie roept zelf `_generation_final_postcommit_compatibility` (23005) aan, dus versie 1 telde de controle dubbel (los en binnen het opslaan) en zag de wachttijd van die taak niet.
- Bij completions begint `elapsed_s` pas in `_run_generation` op de model-thread (26260: `request_received_monotonic_s` ontbreekt, dan `perf_counter()`), na de wachtrij. Versie 1 hookte `_run_generation` niet en schatte het begin vanaf de `dispatch`.

**De "hookkosten tot 18 ms" in §5.3 kloppen niet.** Die vergeleken de totale clienttijd met en zonder hook, inclusief het toeval in de generatie zelf (`elapsed_s` verschilde tussen de runs 4 tot 18 ms). Op het gat (client min `elapsed_s`) was het verschil 0,2 tot 4,9 ms. Offline gemeten kost een gebeurtenis in versie 1 3,3 µs (JSON, lock, write en flush), bij ~55 gebeurtenissen per verzoek ~0,2 ms.

### 6.3 Reparatie en voorbereide metingen

**Hook versie 2** (`hook52/sitecustomize.py`):

- Eén klok: de hook meet met `perf_counter_ns` (`mach_absolute_time`), de client (`meet52.py`) met `CLOCK_UPTIME_RAW`, dezelfde klok. De hook schrijft beide klokken bij het patchen weg, zodat de analyse een verschil zou zien en corrigeren.
- Goedkoper: per gebeurtenis alleen een tuple in een lijst (0,2 µs in plaats van 3,3 µs). Wegschrijven gebeurt bij `/health` (vóór en na de werklast), bij afsluiten en als veiligheidsnet bij een zeer lange lijst.
- Nieuw: `_run_generation` (begin van `elapsed_s` uit `request_received_monotonic_s`, eerste en laatste token, einde), nestdiepte per getimede functie (geen dubbeltelling), wachttijd van elke taak op de model-thread met batch_key, `_skipped_idle_postcommit_snapshot` en `_metrics_envelope`.

**Analyse** (`analyse52.py`): per verzoek alleen gebeurtenissen tussen begin en `gesloten` (socketgebeurtenissen via de poort), en het gat per constructie opgesplitst in VOOR (verbinden, versturen, body, ASGI, json, pydantic, FastAPI, handler tot `elapsed_s`) en NA (staart van `_run_generation` met bankwaarden en put, overdracht naar de aanroeper, postcommit met controle en history-encode, wachtrij van de postcommit, stats, berichtdelen, JSON-render, overig, versturen, ontvangst, sluiten). Hookkosten nu op het gat, met bootstrap-interval.

**Offline getoetst** (geen server, geen model):

- Nep-FastAPI-app met dezelfde functienamen en threadopbouw als de integratietak (model-thread, stream-worker, postcommit-taak) en bekende vertragingen, met client en server op verschillende Python-versies zoals echt: 12 van 12 verzoeken met fasetijden, klok consistent in 12/12, en alle ingebouwde vertragingen teruggevonden (bankwaarden 1,50, put 2,00, history-encode 3,2 tot 3,5, postcommit-wachtrij 4,0, stats 0,50 ms). Hookkosten op het gat: +0,1 tot +0,4 ms.
- De echte `mtplx.server.openai` van de integratietak geïmporteerd (zonder model): alle namen aanwezig, `patch_module` zonder ontbrekende namen. De echte `_run_generation_dispatched` bereikt via de module-globals de gehookte `_submit_foreground_model_work` en `_run_generation` (met stubs voor het model): `request_received_monotonic_s` komt exact door, eerste en laatste token worden gezet. Echte `_generation_final_postcommit_compatibility` roept de gehookte `_history_ids_for_postcommit` aan (nestdiepte 1); `_public_mtplx_stats`, `_build_timings` en `_usage_payload` lopen met de hook.

**Voorbereide metingen** (lokaal in `~/Dev/laya-nl/decode-onderzoek/`, elk als platform-script-taak onder 45 minuten; lader van minstens 60 W, anders stopt het script; controle op `requests_completed`, pid en tracebacks; stopt vóór 45 minuten):

- `gatekosten.zsh`: `main` (1de2b1c0) tegen `main-gate` (e36f5ffb teruggedraaid, worktree `~/Dev/MTPLX-decode-gate`, gemaakt als hij ontbreekt), zes paren ABAB op verse servers, per run 6 × `decode_t0` en 6 × `decode_std` (korte prompt, 512 tokens). `gatekosten.py`: verify-tijd, verify-forward en decode-tijd per ronde, verschil met bootstrap-interval over paren en over verzoeken; `CONCLUSIE GATE` per werklast. Geschat 20 tot 25 minuten.
- `meet52b.zsh`: integratietak, om en om met en zonder hook, drie paren (`h2`/`k2` tot `h4`/`k4`), `meet52.py` (chat ~2K en ~32K tokens, streaming en niet-streaming, 6 herhalingen plus opwarmen; completions-controle), daarna `analyse52.py`. Geschat 8 tot 12 minuten.

## Bestanden

- `~/Dev/laya-nl/decode-onderzoek/`: `README.md` (opzet), `runall.zsh`, `bench60.py`, `meet52.py`, `analyse.py`; `runall.log` en `analyse.txt` (meting 27 sep avond); `logs/meting-*.log` (per verzoek), `resultaten/o1-oud.json`, `m1-main.json` (deels bruikbaar), `i1-integratie.json` (onbruikbaar) (meting 27 sep 18:30, mislukt door de stroomvoorziening); `resultaten/o1-oud.json`, `o2-oud.json`, `m1-main.json`, `d1-main-dense.json`, `g1-main-gate.json`, `b1-main-beide.json`, `i1-integratie.json`, `h1-52-hook.json`, `k1-52-kaal.json` (meting 27 sep avond, bruikbaar); `resultaten/i2-integratie.json`, `m2-main.json` (verstoord door een tweede meetscript, niet meegeteld). Voorbereid 27 sep laat (§6): `gemeen.zsh` (gedeelde controles), `gatekosten.zsh` en `gatekosten.py` (gate-kosten, `bench60.py --decode-herhalingen`), `hook52/sitecustomize.py` versie 2, `meet52b.zsh` en `analyse52.py` (vondst 52); `meet52.py` meet nu met `CLOCK_UPTIME_RAW`.
