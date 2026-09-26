# Chatroute: prefill en encode in scoped modus (26 september 2026)

Tak `perf/chat-scoped-segments` (vanaf lokale `perf/integratie` 89a07905; `fork/perf/integratie` bestaat niet), worktree `~/Dev/MTPLX-chatprefill`, commit `54c4ca5f`, gepusht naar de fork. Geen server gestart, geen model geladen; alleen code gelezen en de tokenizer op de CPU.

## Vondst 40: chat-prefill ~140 ms trager dan completions

**Verklaring (code gelezen, niet gemeten): chat prefillt dezelfde prompt in twee forwards, completions in één.**

- Chat geeft altijd de session bank mee (`mtplx/server/openai.py:33334-33344`); completions niet: op `main` helemaal niet, op de integratietak alleen met `MTPLX_COMPLETIONS_SESSION_BANK=1` (`openai.py:20423-20447`, standaard uit).
- Met een bank zet `restore_or_prefill_prompt_state` een `gdn_boundary_sink` (`mtplx/generation.py:6472-6478`, `MTPLX_GDN_BOUNDARY_CAPTURE` standaard aan). Daarmee plant `_prefill_boundary_plan` (`generation.py:5314`) een grens 64 tokens voor het eind van de prompt (`_gdn_boundary_tail_interval`, 256, `:5031`).
- Die grens wordt normaal binnen de ene forward vastgelegd (in-forward-hooks, `:5275-5311`), maar alleen als het model `boundary_capture_scope`/`take_boundary_captures` heeft. Die hooks bestaan alleen in `mtplx/models/qwen4_exp.py:6011-6018`. Qwen3.6-35B-A3B is `model_type: qwen3_5_moe` en draait op het mlx-lm-model, dus valt terug op de ladder: de forward wordt bij de grens geknipt.
- Uitgerekend met de echte planfunctie: 395 tokens zonder bank `[(0, 395)]`, met bank `[(0, 331), (331, 395)]`; plus in beide gevallen de losse forward van het laatste token (`generation.py:7648`). Chat doet dus één MoE-forward van 64 rijen extra. Het commentaar bij `_gdn_boundary_inforward_enabled` (`:5279-5282`) meldt dat elke forward van 64 rijen of meer vrijwel alle experts leest en ~0,1 s kost, ongeacht de breedte. Dat past bij ~140 ms, en die tijd valt binnen `prompt_target_prefill_time_s` (de forwardtijd uit `_prefill`, `:7613-7627`, via `:10275`).
- De ~30 ms `prompt_state_unattributed_time_s` past bij het werk buiten de timer: bankopzoekingen (`:6202-6289`) en `_capture_gdn_boundary` na elke span (`:7641`, momentopname van de recurrente toestand). `commit_time_s` hoort bij het decoderen (`:10443`), niet bij de prefill.
- **Niet verklaard door `fix/gemma4-probe-vocab`:** die probe zit in `_encode_messages`, vóór de generatie. In de meting daalde "tot start generatie" van 58 naar 2 ms, maar `prompt_target_prefill_time_s` valt daarbuiten.

**Bevestigen met één meting:** hetzelfde verzoek via chat en completions en `mtplx_stats.prefill_chunks` vergelijken (bestaand veld, `openai.py:20894`, `generation.py:1663`): verwacht 2 tegen 1. Hardere toets op een verse server met `MTPLX_GDN_BOUNDARY_CAPTURE=0`: chat-prefill hoort dan naar ~257 ms te zakken. Het omgekeerde kan ook: completions met `MTPLX_COMPLETIONS_SESSION_BANK=1` hoort naar ~396 ms te stijgen.

**Mogelijke fix (niet gebouwd):** geen staartgrenzen plannen voor een prompt die korter is dan het minimum waarvoor de bank of de SSD-cache hem toch bewaart (512), of de in-forward-hooks ook voor `qwen3_5_moe` leveren. Beide raken het rekenpad en moeten op het echte model getoetst worden.

## Vondst 34: memo werkt niet voor gewone chat in scoped modus

**Uitkomst: veilig en gebouwd.** In scoped modus rendert gewone chat zonder naden en eindigt `_encode_messages_uncached` in één `_encode_rendered_chat_text`-aanroep (`openai.py`, pad "seam-less thinking render", en het pad met denken uit).

**Waarom knippen voor elke `<|im_start|>` exact is:** `<|im_start|>` is in de Qwen3.6-tokenizer een added token met `normalized=False`, `lstrip=False`, `rstrip=False` (gecontroleerd; geen enkel added token heeft strip of normalized). De tokenizer haalt zulke tokens eruit vóór de NFC-normalisatie en de pre-tokenizer-regex, dus geen merge of normalisatie loopt eroverheen. Ook letterlijke `<|im_start|>` in berichtinhoud wordt in beide gevallen als speciaal token gelezen.

**Bewijs (gemeten, alleen CPU, echte tokenizer en chattemplate van Qwen3.6-35B-A3B Balance):**
- Ruwe renders: 3.156 renders (400 gegenereerde gesprekken, denken aan/uit, scoped aan/uit, plus de directe HF-render), 105.804 knippunten: 0 verschillen.
- Via `_encode_messages` met de nieuwe code, memo koud en warm over groeiende gesprekken: 8.088 controles, 131.446 memo-hits, 0 verschillen tegen de schakelaar uit. Gesprekken met unicode (CJK, emoji met ZWJ, combinerende tekens, zero-width, BOM), `\r\n`, lange beurten, redenering, lege berichten, systeemprompt, letterlijke en halve markeringen, en voor een derde met tools (hybrid).

**Wat gebouwd is (`openai.py`, ~60 regels):** `_encode_rendered_chat_turns` gebruikt de bestaande `_encode_rendered_chat_text_segmented` met grenzen uit `_chat_turn_boundaries`; `_chat_turn_segments_enabled` en `_chat_turn_open_is_atomic` bewaken het. Twee aanroepen vervangen (render zonder naden met denken aan, en denken uit). Alleen actief als de memo aan staat en `<|im_start|>` atomair is; `MTPLX_CHAT_TURN_SEGMENTS=off` zet het uit. Standaard aan, omdat het bitgelijk bewezen is. Eerste beurten (geen assistentgeschiedenis) lopen nog via `apply_chat_template(tokenize=True)` en zijn ongewijzigd.

**Meting (gemeten, alleen CPU, M5 Pro, scoped, denken aan, groeiend gesprek zonder tools uit `gesprek.py` seed 11, chat-encode-cache uit, mediaan van 7; een andere agent gebruikte tegelijk de machine):**

| Tokens | Soort | Uit | Koud (lege memo) | Warm (vorige beurt in memo) |
|---|---|---|---|---|
| 10.049 | kort | 7,2 ms | 7,6 ms | 7,6 ms |
| 25.619 | kort | 18,9 ms | 19,9 ms | 0,5 ms |
| 41.308 | lang | 32,7 ms | 30,5 ms | 6,1 ms |
| 56.537 | kort | 43,7 ms | 42,8 ms | 0,9 ms |
| 72.558 | lang | 60,5 ms | 57,1 ms | 7,3 ms |
| 80.483 | kort | 65,7 ms | 62,8 ms | 1,3 ms |

Koud kost het niets extra. Warm blijft alleen het nieuwe stuk over (~6-7 ms voor een beurt van ~7K tokens). De Jinja-render is bij dit gesprek geen factor van betekenis (de hele encode warm 1,3 ms bij 80K); dat bevestigt vondst 26 (de eerdere ~43 ms was de Gemma-controle). Op een echte server niet gemeten; verwacht (geschat) ~60 ms minder TTFT per chatbeurt bij 80K.

**Tests:** nieuw `tests/test_chat_turn_segments.py` (13 tests: grenzen, memo, bewakers, schakelaars, pariteit op de echte tokenizer). Gerichte set van 29 testbestanden rond encode, chattemplate en scoped geschiedenis plus `test_no_mlx_imports`: 989 geslaagd, 34 overgeslagen, 0 mislukt. Ruff: `openai.py` 397 meldingen, gelijk aan de basis; nieuw testbestand schoon. Volledige suite niet gedraaid (CPU sparen). CHANGELOG-regel onder Unreleased / Changed.

**Open:** op de echte server meten (encode en TTFT, uitvoer gelijk); eerste beurt ook per beurt encoderen zodat beurt 2 al hits heeft (klein effect).
