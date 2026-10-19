# SSD-cache: minder schrijven, sneller inlezen

Datum: 19 oktober 2026. Werkdocument om thuis op de laptop verder te gaan.

Het gaat om de SSD-laag van de SessionBank (`mtplx/cache_bank/`): opgeslagen
gesprekscaches die bij een volgende beurt of na een herstart terug in het
geheugen worden gezet. Dit document bevat:

- wat er al gebouwd is en hoe je dat test;
- hoe de SSD-cache nu werkt;
- de overige opties;
- de uitzoeking van punt 8, gequantiseerde KV;
- het idee om bestanden af te stemmen op de restore-punten van Qwen 3.8;
- een voorstel voor de volgorde.

Alles hieronder komt uit het lezen van de code en uit simulaties in een
Linux-sandbox zonder MLX. Op een Mac is nog niets gemeten.

---

## 1. Wat er al gebouwd is: branch `ccr-b1c48c0b-dv7vfp`

De branch staat in `jvmenen/mtplx`. Er is nog geen pull request.

| Commit | Inhoud |
|---|---|
| `6f0394a` | Read SSD cache restores in parallel, past the page cache |
| (deze commit) | dit document |

### Wat er veranderd is

- **Parallel inlezen.** Het nieuwe bestand `mtplx/cache_bank/blob_io.py`
  bevat `read_into_many`. Die leest alle blobs van één tensor tegelijk in,
  met een threadpool, rechtstreeks in één buffer. Een losse blob groter dan
  8 MiB wordt in stukken van 4 MiB verdeeld en ook parallel gelezen. Vroeger
  las `cold_tier._restore_row` elk blok van 256 tokens met een eigen
  `Path.read_bytes()`, achter elkaar, op de modelthread.
- **Eén host-kopie per tensor.** Als `read_tensor` de hooks `read_into` en
  `allocate` heeft, maakt `codec.py` één `mx.array` per tensor in plaats van
  één per blok. De blokken zijn views daarop en de concatenate draait op de
  GPU. Zonder hook werkt alles zoals vroeger.
- **Eén staging-buffer per restore** (`blob_io.StagingBuffer`). De buffer
  groeit tot de grootste tensor, wordt hergebruikt en daarna vrijgegeven.
  Zonder hergebruik moest elke nieuwe buffer pagina voor pagina worden
  ingefaulted. In de sandbox maakte dat het nieuwe pad trager dan het oude.
- **`F_NOCACHE` op macOS** voor het lezen en schrijven van blobs. Zo vult de
  SSD-cache de file cache niet, die op een Mac het geheugen deelt met het
  model.
- Het formaat op schijf is ongewijzigd. Oude stores blijven werken.

Een kleine correctie op eerdere uitleg: het scheelt per byte geen twee
volledige kopieën. Het blijven er twee: inlezen en `mx.array`. Wat wegvalt
is de overhead per blok, de seriële I/O en, met `F_NOCACHE`, de extra kopie
in de page cache.

### Nieuwe instellingen (ook in `docs/server.md`)

| Variabele | Standaard | Betekenis |
|---|---|---|
| `MTPLX_SSD_READ_THREADS` | `8` | Aantal threads voor een restore (1–64). `1` leest alles achter elkaar. |
| `MTPLX_SSD_NOCACHE` | `1` | Alleen op macOS: `F_NOCACHE` op blob-I/O. Met `0` gaat dit uit, voor lezen en schrijven samen. |

### Bekende afweging

Met `F_NOCACHE` op de schrijfkant wordt een entry die meteen na het opslaan
weer wordt ingelezen niet meer uit het RAM geladen, maar van de SSD.
Bovendien gaat schrijven synchroner. Houd daarom de schrijfachterstand in de
gaten (`MTPLX_SSD_WRITER_BACKLOG_BYTES`, de SSD-stats in `/health`). Loopt
die op, test dan met `MTPLX_SSD_NOCACHE=0`.

### Testen op de laptop

```bash
git fetch origin ccr-b1c48c0b-dv7vfp
git checkout ccr-b1c48c0b-dv7vfp

# 1. Unit tests. De tweede test heeft MLX nodig en is in de sandbox nooit gedraaid.
pytest tests/test_cold_tier_blob_io.py \
       tests/test_cold_tier_parallel_restore.py \
       tests/test_cache_bank.py \
       tests/test_cold_tier_*.py

# 2. Leessnelheid: het oude pad tegen het nieuwe, met koude cache.
#    --purge voert 'sudo purge' uit voor elke ronde.
python scripts/bench_ssd_blob_read.py --purge --limit-gib 4
MTPLX_SSD_READ_THREADS=4  python scripts/bench_ssd_blob_read.py --purge --limit-gib 4
MTPLX_SSD_READ_THREADS=16 python scripts/bench_ssd_blob_read.py --purge --limit-gib 4
MTPLX_SSD_NOCACHE=0       python scripts/bench_ssd_blob_read.py --purge --limit-gib 4
#    Zonder bestaande store (~/.mtplx/session-bank):
python scripts/bench_ssd_blob_read.py --synthetic /tmp/ssd-bench --purge

# 3. End-to-end: een lang gesprek, server herstarten, dezelfde
#    conversatie doorzetten. Vergelijk in de serverlog
#    "SessionBank SSD restored ... restore_s=" tussen main en deze branch.
```

Wat de sandbox liet zien (Linux, 4 cores, warme page cache, dus geen
SSD-meting): het oude pad haalde ~7 GiB/s, het nieuwe ~4 GiB/s. Bij warme
cache is deze VM memcpy-gebonden en zijn page faults duur. Dit zegt dus
niets over koude reads op Apple-NVMe. Ter vergelijking: de repo heeft eerder
zelf gemeten dat `pread` met meerdere threads 12,9 GiB/s haalt, tegen
1,40 GiB/s voor mmap-faulting (`mtplx/ple_row_gather.py`).

Voor merge nodig:

- de MLX-tests zijn groen;
- de benchmark laat bij koude cache winst zien;
- `restore_s` is lager bij een echte restore;
- de schrijfachterstand loopt niet op.

---

## 2. Hoe de SSD-cache nu werkt

- **Schrijven.** `codec.TreeCodec` knipt elke KV-tensor in blokken van 256
  tokens langs as 2. Per blok gebeurt dit:
  1. `mx.eval`;
  2. `bytes(memoryview)`, de kopie van GPU naar host;
  3. sha256;
  4. schrijven naar `blobs/<xx>/<sha>.bin` via een tijdelijk bestand en
     hernoemen.
  Een blob die al bestaat wordt niet opnieuw geschreven, maar stap 1 tot en
  met 3 gebeuren wel elke beurt opnieuw.
- **Inlezen.** `cold_tier._restore_row` → `codec.decode_payload` of
  `decode_payload_prefix` → per blok een blobbestand.
- **Geen compressie en geen fsync.** Er is een schrijfbudget per uur van
  128 GiB.
- **Veel bestanden.** De code noemt een gemeten store met 816.000 bestanden
  (`cold_tier.py`, rond `DISK_USAGE_SCAN_DUTY_DIVISOR`).
- **Recurrente checkpoints (GDN-boundaries).** Volgens een commentaar in
  `codec.decode_gdn_boundaries` is één boundary ~90 MB op de 27B, en kan een
  entry er tot acht hebben.
- **`MTPLX_SSD_INCREMENTAL_ENCODE` (#556) staat uit.** De fingerprint is geen
  echte inhoudsidentiteit: twee float32-sign-flips heffen elkaar modulo 2^32
  op.

---

## 3. Overige opties, op volgorde van verwachte winst

### Minder schrijven

1. **Incremental encode (#556) repareren.** Dit scheelt het meeste schrijfwerk
   per beurt. Er zijn twee manieren:
   - **Niet-lineair mengen.** Meng elk element niet-lineair voordat het
     wordt opgeteld, bijvoorbeeld met een murmur3-finalizer:
     `x ^= x>>16; x *= 0x85ebca6b; x ^= x>>13; x *= 0xc2b2ae35; x ^= x>>16`.
     Daarna de gewogen sommen zoals nu. Een flip in een hoog bit werkt dan
     door in de lage bits. Voeg een test toe met de bestaande
     sign-flip-gevallen.
   - **Token-keten als sleutel.** Gebruik voor volle blokken de token-keten
     (`chain_block_hashes` bestaat al) plus model-, epoch- en dtype-identiteit
     als sleutel. Dan is er geen GPU-reductie nodig. De afweging: een blok
     dat opnieuw is berekend is semantisch gelijk, maar niet gegarandeerd
     bit-gelijk. Er kan dan een iets ander, maar even geldig, blok worden
     hergebruikt.
2. **Minder of slimmere GDN-boundaries.** Tot ~720 MB per entry op de 27B.
   Bewaar alleen boundaries die een prefix-restore realistisch gebruikt,
   bijvoorbeeld het einde van de systeemprompt, de tooldefinities en de
   laatste beurten. Kopieer of hash een boundary die gelijk is aan die van de
   vorige entry niet opnieuw.
3. **Goedkopere hash.** sha256 op één thread haalt ~2 GB/s. BLAKE3 of xxh3,
   of sha256 verdeeld over threads, haalt dat knelpunt weg. Dit verandert de
   sleutels in de store, dus er is een nieuwe formatversie of migratie nodig.

### Sneller inlezen

4. **Grotere bestanden.** Bundel per entry of per laag de blokken in één
   segmentbestand en zet de offsets in `payload.json`. Dat zijn veel minder
   open/stat/close-aanroepen, en het lezen wordt sequentieel. De afweging:
   hergebruik van identieke blokken tussen entries wordt lastiger. Dat kan
   per bereik van blokken, of via een referentie naar het segment van een
   vorige entry.
5. **Inlezen en kopiëren laten overlappen.** Lees tensor i+1 al in terwijl
   tensor i naar `mx.array` wordt gekopieerd. Dat vraagt twee staging-buffers
   in `StagingBuffer`.
6. **Al lezen terwijl het verzoek nog wordt voorbereid.** Start de SSD-read
   zodra de entry bekend is, terwijl template-encode en de wachtrij voor de
   modelthread nog bezig zijn. Dat kost tijdelijk wat extra RAM.

### Minder bytes per waarde

7. **Lossless compressie.** Een byte-shuffle plus LZ4, via Apple's
   libcompression, levert bij bf16-KV waarschijnlijk 15–30% minder data op.
   Bij een SSD van 5–7 GB/s is dat maar net de moeite van de decompressie
   waard. Eerst meten.
8. **Gequantiseerde KV gequantiseerd opslaan.** Zie §4.

### Het model zelf laden

9. **Modelgewichten bij het opstarten.** `mlx_lm.utils.load` leest de
   safetensors pas bij de eerste evaluatie. Een parallelle `pread`-warmup
   vooraf kan een koude start versnellen. Het risico is dat de page cache
   tijdelijk naast de gewichten in het geheugen staat. Meten met
   `sudo purge` en een koude start.

### Eerst meten

Voeg aan de logregel van een restore een tijdsverdeling toe: lezen,
host-kopie (`mx.array`) en GPU-eval/concatenate. Dat is een kleine wijziging
die laat zien of punt 4 en 5 iets opleveren, of dat het knelpunt ergens
anders zit.

---

## 4. Uitzoeking punt 8: gequantiseerde KV

### Bevinding

Bij `--kv-quant q8` of `q4` slaat de SSD-cache de KV gedequantiseerd op,
op volle bf16/fp16-breedte. Dat is twee tot bijna vier keer meer bytes dan
nodig.

- `snapshot_cache` en `snapshot_cache_lazy_hybrid` (`mtplx/cache_state.py`,
  rond regel 4903) nemen `entry.state`.
- `VllmMetalPagedKVCache.state` → `_active_arrays` → bij `kv_quant`:
  `_dequant_active_arrays` / `_paged_range_flat` geeft arrays terug in
  `self._dtypes`, dus bf16 of fp16.
- Restore: de `state`-setter → `_load_contiguous_state` → `_write_tail`
  quantiseert opnieuw (`kv_quant.quantize_symmetric`).
- Het project noteert dit zelf in `server/openai.py`, rond regel 22116: een
  prompt van 250K tokens op de 27B is als snapshot 15,3 GiB, tegen 4,9 GiB
  aan q4-pagina's. Dat geldt ook voor de RAM-snapshots van de SessionBank.

### Relevantie

KV-quantisatie staat standaard **uit**. `--paged-kv-quantization` volgt
`MTPLX_PAGED_KV_QUANT`, en die is standaard `off`. Het profiel van de 27B
zet het bewust uit ("MTP dead under kv4"). Dit punt helpt dus alleen wie de
server met q8 of q4 draait.

### Is opnieuw quantiseren exact?

Ja, volgens een numpy-simulatie. Met MLX is dit nog niet gecontroleerd.

- Opzet: 20.000 rijen van 128 waarden, schalen 0,1–8, q8 en q4,
  bf16- en fp16-bron.
- Stappen: `quantize` → `dequant` naar bf16/fp16 → opnieuw `quantize`.
- Resultaat: 0% verschil in q en 0% verschil in de schalen.
- Waarom het klopt:
  - De grootste waarde (|q| = qmax) komt na de bf16-afronding terug op
    exact dezelfde max_abs, dus de schaal blijft gelijk.
  - Voor de overige waarden is de relatieve fout van bf16 hooguit 2⁻⁹. Dat
    is ≤ 127·2⁻⁹ ≈ 0,25 kwantisatiestap, dus altijd minder dan een halve
    stap, en ze ronden terug naar dezelfde q.

Gevolg: q plus de fp32-schalen opslaan geeft na dequantisatie exact dezelfde
bf16-bytes als wat er nu wordt opgeslagen. Dit is dus verliesvrij.

### Winst per KV-blok (head_dim 128, fp32-schaal per rij)

| Cache | Nu op SSD | Gequantiseerd opgeslagen |
|---|---|---|
| q8 | 2 B per waarde | ~1,03 B (≈ 52%) |
| q4 | 2 B per waarde | ~0,53 B (≈ 27%) |

Schrijven, lezen en schijfruimte gaan in dezelfde verhouding omlaag.

### Voorstel: in de codec, zonder de cacheklassen aan te raken

1. Doe dit alleen als KV-quantisatie aanstaat (`paged_kv_quant_mode_from_env()`
   is niet `off`), zodat er bij standaardgebruik geen GPU-werk bijkomt.
2. Per KV-blok in `TreeCodec._encode_tensor_blocks`:
   - `q, s = quantize_symmetric(chunk, bits)`;
   - `back = dequantize_symmetric(q, s, bits, head_dim).astype(chunk.dtype)`;
   - als `mx.array_equal(view_uint16(back), view_uint16(chunk))`: sla q
     (int8, of gepakte uint8 bij q4) en s (fp32) op als nieuw speckind,
     bijvoorbeeld `{"kind": "tensor_q", "bits": 8, ...}` per blok;
   - anders: sla het blok op zoals nu.
3. De decoder voert exact dezelfde MLX-bewerkingen uit als de controle bij
   het encoden, dus exactheid is per blok bewezen en niet aangenomen.
4. Het blijft compatibel met bestaande stores: oude specs blijven geldig, en
   alleen nieuwe entries krijgen het nieuwe kind. Kijk of dit een
   `COLD_TIER_FORMAT_VERSION`-bump nodig heeft. Waarschijnlijk niet, als de
   decoder van een oudere versie het onbekende kind netjes weigert.
5. Tests:
   - een round trip met q8- en q4-gequantiseerde input is bit-gelijk aan de
     bf16-opslag;
   - willekeurige bf16 (niet-gequantiseerd) valt terug op ruwe opslag;
   - prefix-decode werkt over blokken heen.

### Latere, grotere stap

Laat de paged cache zelf een gequantiseerde `state` geven, zodat ook de
RAM-snapshots van de SessionBank niet meer op volle breedte staan (de
15,3 tegen 4,9 GiB). Dat raakt het geheugenbudget in `server/openai.py`, de
leases en alle restorepaden.

---

## 5. Idee: bestanden afstemmen op de restore-punten van Qwen 3.8

Dit idee is nog niet gebouwd en niet gemeten.

### Waarom het past

Qwen 3.8 is een hybride model: naast attention heeft het een recurrente
GDN-state. Zo'n model kan alleen hervatten op een opgeslagen checkpoint, een
"restore-punt". De KV wordt dan teruggeknipt tot dat punt en de rest van de
prompt wordt opnieuw berekend. Zie `mtplx/checkpoint_anchors.py` en
`_boundary_true_restore_enabled` in `mtplx/session_bank.py`, die standaard
aanstaat.

De SSD-cache knipt de KV nu in blokken van 256 tokens. Bij een hybride model
kan een restore nooit tussen twee checkpoints beginnen, dus die fijne
indeling levert voor het herstellen niets op. Het geeft alleen veel kleine
bestanden, met als enige winst dat identieke blokken gedeeld kunnen worden.

### De checkpoints zijn stabiel

Sinds 30 september 2026 kiest `retain_checkpoints` ankers die niet
verschuiven als het gesprek groeit:

- een vast raster van één checkpoint per 8.192 tokens (`GRID_TOKENS`). Bij
  krapte dunt dat uit naar 16K, 32K enzovoort, en een bewaard ankerpunt
  blijft op zijn plek;
- het einde van de stabiele prompt-prefix;
- het restorepunt van de vorige beurt;
- het einde van de prompt, 64 tokens ervoor (`tail_backoff`);
- het nieuwste checkpoint.

Standaard houdt een gesprek er acht (`MTPLX_GDN_BOUNDARY_MAX`), of zoveel als
het budget in bytes toelaat. Eén checkpoint is ~90 MB op de 27B en
~115,6 MB op Flash-Next.

### Het ontwerp: segmenten van checkpoint tot checkpoint

- **Indeling.** Eén bestand per stuk tussen twee opeenvolgende checkpoints.
  Daarin staan de KV-rijen van alle attention-lagen voor dat tokenbereik,
  aaneengesloten per laag, met de MTP-history-rijen erbij. De recurrente
  state van het checkpoint aan het eind is een eigen bestand, of hangt achter
  aan het segment.
- **Inlezen.** Een restore op checkpoint k leest precies segment 1 tot en
  met k en één recurrente state. Dat zijn een paar grote, sequentiële
  bestanden, met niets te veel. Bij het raster van 8K zijn de segmenten
  honderden MB groot in plaats van duizenden bestanden van een halve MB, en
  dat is precies waar een NVMe goed in is. Dit maakt optie 4 uit §3
  concreet.
- **Schrijven.** Segmenten vóór het vorige restorepunt veranderen tussen
  beurten niet. Maak de sleutel van een segment de token-keten tot dat
  checkpoint (`chain_block_hashes` bestaat al) plus de identiteit van model,
  epoch en dtype. Dan schrijft een nieuwe beurt alleen de nieuwe segmenten
  aan het eind, zonder de oude segmenten naar de host te kopiëren of te
  hashen. Voor hybride modellen bereikt dat het doel van incremental encode
  (§3, punt 1) zonder GPU-fingerprint, en dus zonder het
  sign-flip-probleem.
- **Exacte restores** lezen nog steeds alle segmenten. Dat gaat wel sneller,
  omdat het grote sequentiële bestanden zijn, en de parallelle reader uit §1
  kan een groot segment in stukken tegelijk lezen.

### Waar je op moet letten

- **Bit-gelijkheid.** Een sleutel op basis van tokens gaat ervan uit dat de
  KV vóór een checkpoint bit-gelijk is aan die van de vorige entry. Binnen
  één sessie is dat zo, omdat die rijen uit dezelfde live cache komen. Na
  een cache-miss met opnieuw berekende KV, bijvoorbeeld met een andere
  chunk-indeling, is het semantisch gelijk, maar niet gegarandeerd
  bit-gelijk. Er zijn twee keuzes: accepteren, of één goedkope controle per
  segment, zoals de echte sha van het eerste en laatste blok vergelijken.
- **Een prompteinde dat verschuift.** Het segment rond het oude prompteinde
  kan per beurt anders liggen. Dat is alleen een klein segment aan de staart
  dat opnieuw geschreven wordt.
- **Checkpoints die uitdunnen.** Als het raster van 8K naar 16K gaat, kun je
  twee bestaande segmenten laten staan en een restore op het 16K-punt beide
  laten lezen. De segmentgrenzen hoeven dus geen exacte kopie te zijn van de
  checkpoints die nu bewaard worden. Ze moeten alleen op checkpoints vallen.
- **Modellen zonder recurrente state** hebben geen checkpoints. Daar is de
  prefix-restore per blok wel nuttig. Die krijgen een vast raster,
  bijvoorbeeld ook 8.192 tokens, of houden de huidige blokken.
- **Flash Next** heeft daarnaast QSA-state en een n-gram-tabel. Bekijk apart
  welke state per segment of per checkpoint mee moet.
- **Formaat.** Dit is een nieuwe `COLD_TIER_FORMAT_VERSION`. Oude stores gaan
  via de bestaande archiveer- of migratieroute (`_archive_legacy_store_if_needed`).
  Opruimen en verwijderen (`reconcile.py`, `_evict_until_room`) moeten
  segmenten met refcounts kennen, net als blobs nu.

### Eerste stappen

1. Meten hoe vaak SSD-restores exact zijn en hoe vaak ze een prefix
   gebruiken, en op welke checkpoints prefix-restores landen. De tellers
   `restore_hits` en `prefix_restores_*` staan al in de stats van de cold
   tier.
2. Een prototype van alleen de leeskant: bij het schrijven de blokken
   per checkpoint-bereik bundelen in één bestand, met de bestaande spec en
   offsets. Daarna vergelijken met `restore_s` op dezelfde gesprekken.
3. Dan de schrijfkant met sleutels op basis van de token-keten.

---

## 6. Voorstel voor de volgorde

1. Branch `ccr-b1c48c0b-dv7vfp` testen op de Mac (§1). Is het goed: PR
   openen. Is het niet goed: met de benchmark uitzoeken of de threads of
   `F_NOCACHE` het probleem is.
2. Tijdsverdeling per restore in de log (§3, "Eerst meten").
3. Segmenten op restore-punten (§5). Voor Qwen 3.8 regelt dit grotere
   bestanden (optie 4) en minder schrijven (optie 1) in één keer.
4. Incremental encode repareren (§3, punt 1), voor modellen zonder
   recurrente state of als §5 niet doorgaat.
5. Lezen en kopiëren laten overlappen (§3, punt 5), afhankelijk van de
   meting.
6. Gequantiseerde opslag (§4), alleen als je zelf met `--kv-quant` draait.
