# DeepSeek V4.1: welke optimalisaties helpen MTPLX? (26 september 2026)

Bronnenonderzoek door een Opus-agent: technische rapporten van DeepSeek gelezen (V4.1-Flash, V4, DSpark), MTPLX-code gelezen op `main` c1962f4e. Niets gemeten, geen server gestart. Getallen met **[bron]** komen uit een rapport of release notes; **[geschat]** is eigen rekenwerk uit de modelconfig; de rest is een inschatting.

## Wat DeepSeek V4.1 is

DeepSeek-V4.1-Flash bestaat: uitgebracht rond 10 september 2026, technisch rapport arXiv 2609.19969 (17 september 2026), MIT-licentie. Het is een multimodaal MoE-model met 552B backbone-parameters plus 196B Engram-parameters, 40 lagen, 384 routed experts (top-6) plus 1 shared expert, context tot 1M tokens. Het bouwt voort op DeepSeek-V4 (april 2026, arXiv 2606.19348; V4-Pro 1,6T en V4-Flash 284B), dat weer voortbouwt op V3.2 (DeepSeek Sparse Attention). Een "V4.1-Pro" is in de bronnen niet gevonden.

Kernclaims V4.1 **[bron]**: globale KV-cache 890 bytes per token (1/4 van V4-Flash), persistente KV-cache op SSD 1/8 van V4-Flash, prefill activeert 8B parameters per token tegen 16B bij decode.

Bijna alles wat die winst geeft, is in het model getraind. Voor Qwen3.6-35B-A3B blijven vooral de inferentie- en opslagideeën over.

## Ons model in getallen **[geschat, uit config.json]**

- 40 lagen, 30 Gated DeltaNet en 10 volledige attentie (elke 4e laag), 2 KV-heads van 256 dims, partial RoPE (64 van 256 dims).
- KV in bf16: 10 × 2 × 2 × 256 × 2 B = **20 KiB per token**; 90k tokens is 1,72 GiB.
- GDN-toestand per grens: 30 lagen × 32 heads × 128 × 128 × 4 B (f32, `mlx_lm/models/gated_delta.py:240`) ≈ **60 MiB**. MTPLX bewaart tot 8 grenzen per entry (`generation.py:4998-5003`), dus tot ~480 MiB per entry: bij een entry van 20k tokens meer dan de KV zelf (~390 MiB).
- Gewichten: 6-bit affine, router-gate 8-bit (`quantization` in config.json).

## Gerangschikt

| # | Optimalisatie (DeepSeek-bron) | Op Qwen3.6 | Verwachte winst | Moeite | Wat MTPLX al heeft |
|---|---|---|---|---|---|
| 1 | GDN-grenzen behandelen als "periodic checkpointing" met korte levensduur (V4 §3.5.2, V4.1 §3.2.1) | ja | tot ~50% meer bankruimte bij contexten tot ~25k **[geschat]** | S (knoppen), M (tiering) | grenzen, afwerpknop standaard uit |
| 2 | KV-cache in 8 bit (V4: FP8 KV met RoPE-dims in BF16; V4.1: FP4 met QAT) | deels (post-hoc, geen QAT) | KV -50%, decode -4% op 16k **[bron MTPLX, M5 Max]** | S | q8/q4 opt-in, productie staat uit |
| 3 | Rij-invariante (batch-invariante) matmul, te beginnen bij de router (V4 §3.3) | ja, als inferentietechniek | ontsluit mogelijk commit B: scoring 1,25× **[gemeten eerder]**; stabiele scores | M (diagnose + router), XL (volledig) | bewust van het probleem, kleine vensters exact |
| 4 | Alleen de opslag kwantiseren: bank en SSD in q8, live KV in bf16 (V4.1 §2.4.4: "FP4 reduces storage rather than accelerates matmul") | deels | ~2× KV-tokens in bank en SSD zonder decodekost **[geschat]** | M | nee |
| 5 | Gemengde precisie KV: RoPE-deel hoger dan de rest (V4 §2) | deels | betere q4-kwaliteit; q4 = 1/4 geheugen | S-M | nee, één schaal per vector |
| 6 | Kostgestuurde verificatielengte (DSpark-scheduler) | ja | onbekend, meten | S (meten) | `CostModelDepthPolicy` bestaat, niet actief |
| 7 | DSpark-drafter trainen op bevroren backbone | deels | onzeker op MoE met één gebruiker | XL | nee |

## Uitwerking kansrijke punten

### 1. GDN-grenzen als periodieke checkpoints

**DeepSeek:** V4 kent drie opties voor de toestand van sliding-window-attentie in de prefixcache: alles opslaan, elke `p` tokens een checkpoint, of niets opslaan en herberekenen. V4.1 haalt die toestand helemaal uit de persistente cache en houdt hem alleen minuten in een klein RAM-deel, omdat hij binnen een sessie kort hergebruikt wordt en daarna dood is. Globale KV blijft wel lang bewaard.

**MTPLX:** GDN-grenzen zijn precies zulke checkpoints: tot 8 per entry (`generation.py:4998`), een fijner raster van 256 op het laatste blok (`generation.py:5006`), en ze worden ook naar SSD geschreven (`cache_bank/codec.py:320-336`). De afwerpknop `MTPLX_SESSION_BANK_SHED_BOUNDARIES` bestaat, maar staat standaard uit (`session_bank.py:143-193`; de docstring beschrijft een 70× TTFT-klif toen grenzen een entry uit de bank drukten).

**Voor ons:** bij onze contexten zijn grenzen per entry vergelijkbaar met of groter dan de KV **[geschat]**. Dat raakt direct het bankplafond. Stappen:
1. Meten hoe een bankentry is opgebouwd (KV tegen grenzen) na een agentbeurt van 20k en 60k tokens.
2. Afwerpknop aan, en `MTPLX_GDN_BOUNDARY_MAX` omlaag (minimaal 2), gemeten op `cached_tokens` en TTFT bij hervatten.
3. Pas daarna het V4.1-idee: grenzen alleen in RAM met korte levensduur, niet naar SSD; op SSD alleen KV en de eindtoestand.

Risico: minder grenzen betekent vaker een koude prefill bij een prefix die halverwege afwijkt. Het lange-termijnhergebruik (hervatten van een sessie) heeft alleen de eindtoestand nodig.

### 2. KV-cache in 8 bit

**DeepSeek:** V4 slaat KV op in FP8 met de RoPE-dims in BF16 (ongeveer de helft van BF16). V4.1 gaat naar FP4 (E2M1, één E4M3-schaal per 16 kanalen), maar alleen met quantization-aware training; de SWA-cache blijft FP8 "vanwege gevoeligheid voor kwantisatie".

**MTPLX:** q8 en q4 bestaan (`kv_quant.py`, `cache_state.py:3239-3247`) en werken volgens de release notes v2.10.0 op de Qwen 3.5/3.6-familie: q8 kost ~4% decode, q4 ~19%, bij de helft en een kwart van het KV-geheugen (`docs/releases/v2.10.0.md:87-94`, gemeten op M5 Max bij 16k) **[bron]**. Onze productie draait met `--paged-kv-quantization off`.

**Voor ons:** 90k tokens van 1,72 naar ~0,87 GiB **[geschat]**, en de bank past naar verwachting twee keer zoveel KV-tokens, omdat de gekwantiseerde tensors in de cache zelf zitten (niet nagegaan of de bank ze ongewijzigd opslaat). Qwen is niet met kwantisatie getraind, dus q8 is de realistische stap; q4 alleen na punt 5. Meten: classifier (240 prompts, top-1 en logprob-verschil tegen de 256-referentie), decode-snelheid bij 20k en 80k, en `session_bank.effective_max_bytes` met aantal entries.

### 3. Rij-invariante matmul, te beginnen bij de router

**DeepSeek:** V4 maakt alle kernels bitgelijk ongeacht de batchpositie: geen split-KV in attentie (twee kernels met dezelfde optelvolgorde), cuBLAS volledig vervangen door DeepGEMM zonder split-K, en ze melden verwaarloosbare kosten na eigen optimalisaties **[bron]**. Het gaat om CUDA-kernels; code voor Metal is er niet.

**Ons knelpunt:** in [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) verschillen de router-logits 1-2 bf16-stappen afhankelijk van het aantal rijen, waardoor experts wisselen en commit B (1,25× sneller) niet bitgelijk is. MTPLX kent het mechanisme: MLX kiest de kernel op (M, K, N); onder de matvec-grens is `qmv` per rij, daarboven hangt de split-K-verdeling van de vorm af (`packed_concats.py:8-18`, `proj_fusion.py:185`). De bestaande `qwen_row_owned_router.py` doet alleen top-K en normalisatie na de stock-gate, alleen voor 1 tot 16 rijen bij decode (`qwen_row_owned_router.py:24, 83-85, 398-405`); prefill gebruikt de stock-route.

**Voor ons:** de router-gate is klein (2048 × 256 per laag, 8-bit). Een eigen Metal-kernel met één threadgroup per rij en een vaste optelvolgorde is per constructie rij-invariant en kost vrijwel niets. Of dat genoeg is, is onzeker: ook de expert-matmuls (`gather_qmm`, rijen per expert wisselen) en attentie kunnen per vorm verschillen, en dan flippen routes in latere lagen alsnog. Daarom eerst een diagnose: per laag meten waar de eerste afwijking ontstaat met een rij-invariante gate. Blijft alleen afrondingsruis over zonder routewissels, dan kan commit B terug. Anders is volledige invariantie (qmm, gather_qmm, SDPA, GDN) nodig, en dat is een MLX-fork-klus met onbekende snelheidskosten.

### 4. Alleen de opslag kwantiseren

**DeepSeek:** V4.1 kiest FP4 voor opslag, niet voor rekenen: waarden worden vóór de attentie gedekwantiseerd, zodat het op alle hardware werkt.

**Voor ons:** live KV blijft bf16 (geen decodekost, geen kwaliteitsverlies in de lopende beurt), alleen bank- en SSD-entries worden q8 (`cache_bank/codec.py` slaat nu elke tensor op in zijn eigen dtype). Bij herstel één keer dekwantiseren, ~1,7 GiB bij 90k, naar verwachting enkele tientallen ms op de GPU **[geschat]**. Nadeel: een warm herstel is niet meer bitgelijk aan een koude prefill; de pariteitscontrole van de bank moet dat toestaan. Alleen zinvol als punt 2 (q8 live) om kwaliteits- of snelheidsredenen afvalt.

### 5. Gemengde precisie voor de KV

`kv_quant.quantize_symmetric` gebruikt één f32-schaal per headvector over alle 256 dims (`kv_quant.py:81-100`). V4 houdt de RoPE-dims in BF16. Bij Qwen3.6 zijn dat 64 van de 256 K-dims, die een ander bereik hebben dan de rest. Een aparte schaal of BF16 voor dat deel maakt q4 mogelijk bruikbaarder: K = 192 × 0,5 + 64 × 2 = 224 B tegen 512 B, V volledig q4 128 B **[geschat]**. Alleen de moeite als q4 nodig blijkt.

### 6. Kostgestuurde verificatielengte

DSpark kiest per verzoek hoeveel draft-tokens worden geverifieerd, op basis van voorspelde acceptatie en gemeten doorvoercurves. MTPLX heeft dat voor één gebruiker al: `CostModelDepthPolicy` (geporteerd uit omlx, `adaptive.py:319`) en `ExpectedValueDepthPolicy` (`adaptive.py:69`), te kiezen met `--adaptive-policy` (`server/openai.py:22049-22095`). Onze productie gebruikt vaste diepte 2. Een meting met `cost` tegen vast is een goedkope proef; de maker meldt dat het beleid "depth-2 lock" doorbreekt, maar niet voor A3B gemeten.

## Wat niet de moeite is

| Optimalisatie | Waarom niet |
|---|---|
| Causal Encoder-Decoder (YOCO-achtig, prefill ~halveert) | Vraagt een model dat zo getraind is; Qwen-lagen maken hun eigen KV. Aantrekkelijk voor onze lange prefill, maar niet overdraagbaar. |
| CSA, HCA, CSA2 (gedeelde KV over lagen), DSA/lightning indexer (V3.2), Hierarchical Sparse Indexer | Allemaal getrainde architectuur. Het blokmaximum-idee van de indexer gebruiken we al in de top-K van commit A. |
| SWA Bounded Replay | Qwen heeft geen sliding window; GDN-toestand is niet venster-begrensd, dus een toestand die alleen uit de laatste tokens is nagerekend is onbetrouwbaar zonder training (V4.1 oefent het zelfs in post-training). |
| FP4-expertgewichten, MXFP4, FP4-indexer | QAT-afhankelijk; ons model staat al op 6 bit. MXFP4 in MLX geeft geen geheugenwinst tegen 4-bit affine en de kwaliteit is ongetoetst. |
| mHC, Single-Pass mHC, Mega-mHC-kernel | Qwen heeft geen hyper-connections. |
| Engram, hash-routering in eerste lagen, Sqrt(Softplus)-router, loss-vrije load balancing per modaliteit | Trainingskeuzes. Load balancing is voor training en multi-GPU; op één GPU telt alleen welke experts een forward raakt. |
| Expert-parallel (DeepEP, Mega-MoE-overlap), EPD-disaggregatie | Multi-GPU en clusters. |
| Deterministische backward (atomics vervangen) | Alleen training. |
| DSpark-drafter trainen (punt 7) | Kan op een bevroren backbone (DeepSpec-repo), maar DSpark is getest op dichte Qwen3-4B/8B/14B met GPU-servers; bij MoE op één GPU raakt elke extra verificatierij extra experts, dus langere drafts lonen minder. Pas overwegen als punt 6 laat zien dat diepte 3+ loont. |
| DeepSeek-modellen zelf draaien | MTPLX ondersteunt V4-Flash al (`deepseek_v4_attention_island.py`), maar 284B (V4-Flash) en 552B + 196B (V4.1-Flash) passen niet in 64 GB. |

## Bronnen

- DeepSeek-V4.1-Flash, technisch rapport: https://arxiv.org/abs/2609.19969 (pdf gelezen: §2.2 CED, §2.3 CSA2, §2.4.4 FP4 KV, §3.2 persistente KV en Bounded Replay)
- Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Aankondiging: https://api-docs.deepseek.com/news/news260910/
- DeepSeek-V4, technisch rapport: https://arxiv.org/abs/2606.19348 (§3.3 batch-invariante kernels, §3.5 KV-cachebeheer en on-disk-opslag)
- DSpark: https://arxiv.org/abs/2607.05147
- DeepSeek-V3.2-Exp (DSA): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- Kernels genoemd in V4.1 §3.2: https://github.com/deepseek-ai/FlashMLA, https://github.com/deepseek-ai/DeepGEMM
- MTPLX: `docs/releases/v2.10.0.md` (KV-kwantisatie), code op `main` c1962f4e
