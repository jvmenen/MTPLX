# Eerste-token-logprobs (25 september 2026)

Tak `feat/first-token-logprobs`, commit `3c3f3aaf`, gebaseerd op `main` (2.12.0). Gebouwd door een Opus-agent, niet live getoetst.

## Doel

Een classifier moet de top-K verdeling van het volgende token na een prompt kunnen uitlezen via de gewone generatieroute, in plaats van via de losse prompt-scoringroute die altijd koud inleest. Tot nu toe weigerden `/v1/completions` (behalve `echo` met `max_tokens: 0`) en `/v1/chat/completions` elke vorm van logprobs.

## Wat er veranderd is

**`mtplx/generation.py`**
- Nieuwe dataclass `FirstTokenLogprobs(token_id, logprob, top)` en veld `GenerationOutput.first_token_logprobs`.
- Hulpfuncties `_log_softmax_f32` en `_top_k_logprobs`; `score_prompt_logprobs` gebruikt ze nu ook (zelfde rekenwerk).
- `first_token_logprobs(row, token_id, top_k)`: `mx.eval`, kopie naar het hostgeheugen, gesorteerd, zodat latere cache-updates het resultaat niet veranderen.
- `generate_ar` en `generate_mtpk` krijgen `first_token_logprobs_top_k`. Op de plek waar het eerste token gekozen wordt, rekenen ze vanaf de ruwe rij `logits[0]`, vóór grammaticamasker, temperatuur, straffen en steering. Bij een volledige cachehit wordt de herstelde `prompt_state.logits` gebruikt.

**`mtplx/server/openai.py`**
- `_run_generation` geeft de waarde door en zet `first_token_logprobs` in het resultaat. Blank retries staan uit bij logprobs-verzoeken.
- `_run_generation_dispatched`: logprobs-verzoeken slaan de live AR-batchroute over (`scheduler_lane=solo_logprobs`). Op de MTP-batchroute weigert `_validate_mtp_batch_request_contract` ze met een 400, zoals bij `response_format`.
- Hulpfuncties: `_reject_unservable_first_token_logprobs`, `_require_first_token_logprobs` (geeft 500 als de engine niets opleverde, nooit stil een 200), `_completion_first_token_logprobs`, `_chat_first_token_logprobs`, `_chat_first_token_logprobs_top_k`.

## API-gedrag

- Alleen het eerste token, en `max_tokens` moet 1 zijn (anders 400). Zo krijgt elk teruggegeven token zijn logprob en ziet niemand een halve lijst voor een volledige aan.
- `/v1/completions`: `logprobs: K` (0 tot `MTPLX_PROMPT_LOGPROBS_MAX`, standaard 128) geeft `{tokens, token_logprobs, top_logprobs, text_offset: [0], token_ids}`. De top-K bevat altijd het gekozen token met zijn echte waarde. Prompt-scoring (`echo`, `max_tokens: 0`) is ongewijzigd.
- `/v1/chat/completions`: `logprobs: true` met `top_logprobs: K` geeft `{content: [{token, logprob, bytes, top_logprobs: [...]}], refusal: null}`. `bytes` is de UTF-8 van de gedecodeerde tokentekst.
- 400 bij: `stream: true`, een niet-lege `stop`, K boven het maximum of negatief, `top_logprobs` zonder `logprobs: true`, de backend gemma4_assistant.
- Is het eerste token een stoptoken, dan wordt dat gewoon gerapporteerd (de zichtbare tekst is dan leeg).
- De Anthropic-route `/v1/messages` vertaalt via de chatroute en laat logprobs weg; de Anthropic-API kent geen logprobs.

## Tests

- Nieuw in `tests/test_generation_sustained.py`: vergelijking met een referentie-log-softmax, K=0, AR en MTPK bij temperatuur 0 en 0,6 (ruwe waarden), grammaticamasker heeft geen invloed, volledige cachehit gebruikt de herstelde logits, None zonder verzoek.
- Nieuw `tests/test_first_token_logprobs_api.py` (19 tests): vorm van het antwoord op beide routes, de 400-gevallen, 500 als de engine niets teruggeeft, geen blank retry, AR-batch overgeslagen, MTP-batch geweigerd.
- Gedraaid: alle testbestanden die generatie, logprobs of dispatch raken (73 bestanden) plus `test_no_mlx_imports` en `test_runtime_kpis`: 1.982 geslaagd, 24 overgeslagen, 9 mislukt. De 9 zitten in `tests/test_public_cli.py` en falen ook zonder de wijziging (afhankelijk van de lokale omgeving).
- Ruff: geen nieuwe meldingen.
- Niet gedaan: live toets met echte gewichten, volledige suite (476 bestanden), `python -m build`, `fresh_venv_smoke.sh`, CHANGELOG-regel.

## Bevindingen uit de review

- **`/v1/completions` gebruikt de session bank nooit.** `session_bank` wordt niet doorgegeven, dus `cache_bypass` is altijd waar. Alleen de chatroute profiteert van hergebruik.
- **`score_prompt_logprobs` is zwaarder dan nodig.** Per blok van 256 posities wordt een volledige float32-log-softmax over de hele woordenschat gebouwd (~250 MB bij 248k tokens) vóór de top-K. Goedkoper: argpartition op de bf16-logits, per rij één keer `logsumexp`, en alleen de K overblijvers en de doelwaarden aftrekken. `_mx_lazy_shape` volgt dat patroon al.
- **`first_token_logprobs` zelf** is één reductie over één rij, ruim onder een milliseconde.
- **Leesbaarheid:** `_run_generation` (~650 regels) en de completions-handler (~500 regels, vooral streaming) zouden winnen bij het uitlichten van het samenstellen van het antwoord en de streamlus in functies met een duidelijke naam.

## Mogelijke oorzaken van uitschieters van ~1,5 s (uit de code, niet gemeten)

1. **Blank retries (sterkste spoor).** Een verzoek zonder seed waarvan de tekst na `strip()` leeg is (een spatie, regeleinde of stoptoken als eerste token), wordt tot `--blank-retry-attempts` keer (standaard 3) opnieuw gegenereerd: vier volledige generaties, 4 × ~390 ms ≈ 1,5 s. Staat uit bij logprobs-verzoeken; bij andere verzoeken voorkomt een expliciete `seed` het.
2. **Bankwerk vóór het antwoord.** De laatste `session_bank.put` loopt synchroon in de worker; niet-streamende chat wacht op `store_postcommit_snapshot`; `generate_mtpk` kan binnen het lock een snapshot van het promptbegin maken.
3. **SSD-coldtier-encode** op de owner/idle-lane, die pas op een tensorgrens wijkt voor een nieuw verzoek.
4. **Koude GPU-werkset** na een stille periode (`gpu_keepalive` `warm=false`), volgens de code ~9 ms per GiB.
5. **Leeggemaakte MLX-cache** als een postcommit-snapshot wordt overgeslagen door een stoptoken-grens (alleen sessiechats).
6. **Toelatingsronde:** elk verzoek breekt eerst openstaande postcommits van andere sessies af.
