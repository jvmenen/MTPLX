# Plafond van de session bank (26 september 2026)

Tak `fix/bank-ceiling-peak-decay` (gebaseerd op `main` 1de2b1c0), commit `fb1cd817`, niet gepusht. Analyse en bouw door een Opus-agent; niets op hardware gemeten.

## Hypothese: deels weerlegd

De vermoeden was dat de piekreserve, die meegroeit met de hoogste geheugenpiek sinds de start en nooit daalt, het plafond op de ondergrens van 1 GiB vastzet. Dat klopt maar half.

Getallen voor deze machine (`plan_memory`; `max_bytes` komt exact uit op 18.683.107.738, gelijk aan /health): bruikbaar 48 GiB, gewichten 27,6 GiB, speelruimte 20,4 GiB, reservemaximum = speelruimte/2 = 10,2 GiB, maximale bank in rust 17,4 GiB.

Met de reserve op het maximum geldt: **plafond = max(1 GiB, min(17,4, max(bruikbaar − levenspiek + bank_nu, 10,2 − W)))**, met W het werkgeheugen.

- **In rust na een zware beurt:** 10,2 − W. Met W = 1 GiB is dat 9,2 GiB, tegen 16,4 GiB met de vaste reserve van 3 GiB. De plakkende reserve kost dus ~7,2 GiB bank, maar brengt hem niet naar de ondergrens.
- **Precies 1 GiB** vraagt allebei: een levenspiek binnen ~1 GiB van de Metal-limiet én W ≥ 9,2 GiB op dat moment. De eerdere /health-meting is dus vrijwel zeker tijdens een zwaar verzoek gedaan, of er stond iets anders dan de bank in het geheugen. Reserve en werkgeheugen wegen even zwaar.
- **W** = `mx.get_active_memory()` − gewichten − bank. De MLX-bufferpool (`get_cache_memory`) telt niet mee. W bevat live KV en prefill-kladruimte, gecompileerde buffers, arrays van de draft-head, openstaande postcommit-caches, en eventuele retrievalmodellen of verschil tussen gewichten in het geheugen en op schijf. Uitlezen via `/v1/mtplx/snapshot` → `mem.generation_working_bytes`.
- **De levenspiek-meting is in beide richtingen onzuiver:** uitzettingen uit de bank verhogen hem (piek − actief groeit tot het maximum), groei van de bank drukt hem terug naar de vaste 3 GiB.

## Geschiedenis die het ontwerp bepaalt

- **15474acd (29-08-2026):** een vaste reserve van 3 GiB liet de allocator 0,992-1,001 van de limiet halen en gaf bij elke zware beurt de waarschuwingsbanner; gemeten piek 12,4 GiB.
- **28c218d5:** maximum van speelruimte/2 toegevoegd, zodat één beurt de bank niet voor altijd uithongert.
- **444a2b9a:** het 93k-incident: uitzettingen door het dynamische plafond raakten de live keten (TTFT 54-57 s), opgelost met `protect_active`.
- **Het leegmaken van de cache via de beheerroute roept al `mx.reset_peak_memory()` aan**; de makers noemen de levenspiek zelf een "ratchet". Dat reset ook de reserve van vandaag en werkt dus nu al als handmatige omweg.

## De wijziging

- `memory_plan.RecentSpikes`: neemt de grootste piek van de open burst en de laatste N afgesloten bursts.
- De geheugenbewakingslus sluit een burst af bij de overgang van druk naar rust; `EngineSessionManager.close_spike_burst` legt de piek vast en herstart de MLX-piekmeting.
- Ongewijzigd: de vaste 3 GiB, het maximum van speelruimte/2, de ondergrens van 1 GiB en `protect_active`.
- Knop `MTPLX_SESSION_BANK_SPIKE_BURSTS=N` (alleen env, zoals de andere `SESSION_BANK_*`-knoppen), **standaard uit**, omdat met N gezet `peak_memory_bytes` "sinds de laatste rust" gaat betekenen in plaats van "sinds de start", en omdat na N rustige bursts de eerste nieuwe zware prefill een vollere bank tegenkomt (de bannersituatie van 29-08, één keer per cyclus, tot de reserve weer oploopt). Voorgestelde waarde bij proberen: N = 16.
- Randgevallen conservatief: een burst die de tick van 10 s nooit als druk ziet, of continu verkeer zonder rusttick, blijft in het open venster.
- ~95 regels in 3 bronbestanden, plus 3 testbestanden en een CHANGELOG-regel (gemarkeerd als geanalyseerd, niet gemeten). Ruff: geen nieuwe meldingen.

## Tests

Nieuw: rekenwerk van de productieplanning (maximum tegen ondergrens), een zware burst die N bursts wordt vastgehouden en dan losgelaten, een tweede zware beurt die opnieuw oplaadt, geen krimp binnen een burst, terugval naar 3 GiB na reset, het maximum van de speelruimte, env-parsing, integratie met een nep-allocator (aan tegen uit: uit reset de piek nooit), en de bewakingsovergang over 5 ticks. Volledige suite: dezelfde 19 mislukkingen als de nulmeting, geen nieuwe.

## Verificatie op hardware (nog niet gedaan)

1. Twee starts: A zonder knop, B met `MTPLX_SESSION_BANK_SPIKE_BURSTS=16`, verder gelijk.
2. Na elke beurt vastleggen (/health en `/v1/mtplx/snapshot`): `session_bank.{effective_max_bytes, dynamic_ceiling_bytes, total_nbytes, entries}`, `mem.{active,peak,cache}_memory_bytes`, `mem.generation_working_bytes`, `mem.phys_footprint_bytes`, `allocator_fraction`, `memory_pressure_level/source`, `system_available_bytes`, `memory_guard_events`, SSD-herstel, TTFT en `cached_tokens`.
3. Werklast: 3 agentsessies van ~20k tokens, dan één koude prefill van ~60k in een 4e sessie; dan 50 korte chatbeurten in sessie 1 met pauzes van ≥15 s (elke beurt een eigen burst); dan sessies 2 en 3 hervatten; dan een tweede prefill van 60k.
4. Bevestiging: in B stijgt het rustplafond van ~10,2 − W naar ~17,4 − W na 16 korte bursts, A blijft vlak; sessies 2 en 3 herstellen warm in B; bij de tweede zware prefill vuren `dynamic_ceiling`-events vóór de piek, `allocator_fraction` blijft onder 0,97, hooguit één banner.
5. Lees ook `generation_working_bytes` in rust: is die 9 GiB of meer, dan is het werkgeheugen het echte probleem en helpt deze knop niet.
6. Veiligheid: `memory_pressure` en `sysctl vm.swapusage` volgen tijdens de zware prefills; B afbreken bij `allocator_fraction` ≥ 0,99, drukniveau ≥ 2 met bron `allocator`, groeiende swap of Metal-allocatiefouten in het log.

## Risico's

Betekenis van de piektelemetrie per verzoek verandert met de knop aan; één banner-risico per cyclus; als W de grootste term is, is de winst beperkt tot het rustplafond (hooguit ~7,2 GiB); een kleine race tussen rustcontrole en reset is conservatief voor de reserve maar kan de piekstatistiek van een net startend verzoek iets afkappen.
