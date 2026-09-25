# Onderzoek: MTPLX als lokale classifier en agent-backend

Deze map hoort bij de fork [jvmenen/MTPLX](https://github.com/jvmenen/MTPLX) en staat op de tak `onderzoek`, los van de featuretakken, zodat hij nooit in een pull request naar het origineel meekomt. Hij bewaart metingen, bevindingen en afspraken uit het werk aan MTPLX: als snelle lokale classifier (kansen over vaste labels uitlezen in plaats van tekst genereren) en als backend voor lokale agents.

**Ben je een agent die hier verder werkt? Lees eerst de spelregels hieronder, dan [VONDSTEN.md](VONDSTEN.md).**

## Waar te beginnen

1. **[VONDSTEN.md](VONDSTEN.md)**: de lopende werklijst. Open punten met bron en volgende stap; afgehandelde punten onderaan. Kies hier je werk.
2. **De tabel "Takken in de fork"** hieronder: welke tak welke stand heeft.
3. **Het rapport** dat bij je onderwerp hoort (tabel "Rapporten"), voor de achtergrond en eerdere metingen.
4. **Achtergrond in één keer**: [metingen-classifier](2026-09-25-metingen-classifier.md) (waar de tijd heen gaat) en [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) (de les over MoE-routering en blokgroottes).

## Spelregels

### Documenteren

- **Elk onderzoek of elke meting krijgt een rapport** in deze map: `JJJJ-MM-DD-onderwerp.md`, in het Nederlands. Nieuwe uitkomsten bij een bestaand onderwerp voeg je als sectie met datum toe aan dat rapport.
- **Werk bij elke wijziging drie dingen bij:** het rapport, [VONDSTEN.md](VONDSTEN.md) (status en volgende stap; afgehandeld naar onderen met datum en uitkomst) en de tabellen in deze README.
- **Scheid gemeten van geschat.** Schrijf erbij waar een getal vandaan komt (echt model, synthetische tensors, testmodel, alleen code gelezen), met hardware, model, profiel en datum.
- **Openbaar:** de fork is publiek. Geen klantnamen, privéberichten of inhoud van chats in rapporten; alleen aantallen en uitkomsten.
- **Schrijfstijl:** kort, zakelijk, geen em-dashes. Vóór het committen de spellingscontrole draaien (LanguageTool; vakjargon zoals "forward pass" mag blijven).
- **Committen en pushen:** in de lokale worktree `~/Dev/MTPLX-onderzoek`, daarna `git push fork onderzoek`. Commitberichten eindigen met de Co-Authored-By-regel.

### Code en takken

- **Eén onderwerp per tak**, gebaseerd op `origin/main`, elk in een eigen worktree (`~/Dev/MTPLX-<onderwerp>`), zodat agents elkaar niet in de weg zitten. Nooit werken in een worktree van een andere agent.
- **Standaardgedrag niet veranderen** zonder dat het bewezen beter is; nieuw gedrag eerst achter een schakelaar (env of config, in de stijl van de bestaande knoppen).
- **Clean Code:** kleine functies met duidelijke namen, minimale diffs in de grote bestanden (`openai.py` ~39k regels, `generation.py` ~16k), geen ongevraagd herformatteren.
- **Tests:** nieuwe tests bij elke wijziging; de relevante bestaande tests draaien en vergelijken met de nulmeting. Op een schone `main` falen in deze omgeving al 19 tests (9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader`, 1 `test_laguna_model`): meld alleen afwijkingen daarvan. Ruff mag geen nieuwe meldingen geven (vergelijk het aantal met `main`).
- **Python:** venv `~/Dev/MTPLX/.venv` (editable, dev- en server-extras). In een andere worktree: `PYTHONPATH=<worktree>` zetten en controleren met `python -c "import mtplx; print(mtplx.__file__)"`.

### Het echte model en de server (belangrijk)

- **Kleine testmodellen bewijzen niet genoeg.** Een wijziging die op het testmodel bitgelijk was, gaf op Qwen3.6-35B-A3B andere uitkomsten (MoE-routering hangt af van het aantal rijen per forward; zie [snellere-scoreroute](2026-09-26-snellere-scoreroute.md)). Alles wat rekenpaden raakt, eerst op het echte model toetsen vóór een PR.
- **Er is één GPU en 64 GB geheugen.** Het model (~20 GB) mag maar één keer geladen zijn. Nooit het model in een tweede proces laden terwijl er een server draait; geen GPU-zwaar werk naast een meting (dat gaf HTTP 507).
- **De productieserver op poort 8000 wordt beheerd door het Bink-platform** (Bink, Fleur, chat-titels en de nachtelijke review gebruiken hem). Het platform start hem zelf met het Balance-model als hij nodig is en stopt hem na 15 minuten stilte.
- **Een testserver starten** gaat alleen op poort 8000, na toestemming van Jeroen, zodat het platform geen tweede kopie start:
  - stoppen: `mtplx stop --port 8000`, wachten tot `pgrep -f mtplx.server.openai` leeg is;
  - starten vanuit de worktree: `env PYTHONPATH=<worktree> nohup ~/Dev/MTPLX/.venv/bin/python -P -m mtplx.server.openai ${=ARGS} > log 2>&1 &` met `ARGS=$(cat ~/Dev/laya-nl/mtplx-originele-args.txt)` (de productie-instellingen). In zsh `${=ARGS}` gebruiken, anders komen alle argumenten als één argument binnen. Alleen de venv-python kent `mtplx`.
  - na afloop weer stoppen en poort 8000 vrij laten.
- **Meet op een verse server per variant**: de MLX-piekmeting daalt nooit, en caches vertekenen anders de vergelijking.
- **Meetmateriaal** (lokaal, niet in de fork): `~/Dev/laya-nl/` met `data/scoring-prompts.jsonl` (240 classifierprompts), `verify_scoring_real.py` (scoreroute oud tegen nieuw), `test_live_classifier.py` en `resultaten/`.

### Pull requests naar het origineel

- **Alleen na akkoord van Jeroen** een PR openen; pushen naar de fork mag.
- **Een schone tak per PR** met alleen de bedoelde commits (geen teruggedraaide experimenten in de geschiedenis).
- **Vóór de PR** de controles uit `CONTRIBUTING.md`: relevante tests, `python -m build`, `scripts/fresh_venv_smoke.sh`; een CHANGELOG-regel onder `## [Unreleased]` met meetgegevens (hardware, model, profiel, fanmodus, datum, commit).
- **PR-tekst gaat over de wijziging**: wat, waarom de huidige code tekortschiet, en metingen van die wijziging (bijv. bitgelijk en sneller dan het oude pad). Geen vergelijkingen tussen modellen of resultaten van ons eigen project. In het Engels.

### Agents

- Voor dit werk gebruikt Jeroen Opus-agents. Geef elke agent een eigen worktree, de grenzen hierboven (geen server of echt model zonder expliciete toestemming, niet pushen) en vraag om een rapport met file:line-verwijzingen.
- Maximaal een paar agents tegelijk: ze delen de CPU, en metingen op het echte model kunnen maar één voor één.

## Rapporten

| Datum | Rapport | Onderwerp |
|---|---|---|
| 2026-09-25 | [metingen-classifier](2026-09-25-metingen-classifier.md) | Waar de tijd per classificatie heen gaat, hergebruik van het promptbegin, promptvolgorde |
| 2026-09-25 | [eerste-token-logprobs](2026-09-25-eerste-token-logprobs.md) | Logprobs voor het eerste gegenereerde token (tak `feat/first-token-logprobs`, PR #530) |
| 2026-09-25 | [prefix-hergebruik](2026-09-25-prefix-hergebruik.md) | Waarom een kort gedeeld promptbegin niet wordt hergebruikt en hoe 128 instelbaar wordt; completions in de session bank |
| 2026-09-26 | [live-test-classifier](2026-09-26-live-test-classifier.md) | Live test eerste-token-logprobs en prefix-hergebruik samen op 240 berichten |
| 2026-09-26 | [opschonen-prefix-helpers](2026-09-26-opschonen-prefix-helpers.md) | Snellere prefixvergelijking, één lezer per instelling |
| 2026-09-26 | [optimalisaties](2026-09-26-optimalisaties.md) | Gerangschikte optimalisaties met effect, snelle winsten en wat niet de moeite is |
| 2026-09-26 | [snellere-scoreroute](2026-09-26-snellere-scoreroute.md) | Top-K zonder volledige log-softmax (bitgelijk, 1,2× sneller); waarom bredere blokken fout gaan (MoE-routering) |
| 2026-09-26 | [chat-encode-memo](2026-09-26-chat-encode-memo.md) | Alleen nieuwe gespreksstukken tokeniseren |
| 2026-09-26 | [bankplafond](2026-09-26-bankplafond.md) | Waarom het plafond van de session bank wegzakt en een voorzichtige fix |

## Takken in de fork

| Tak | Inhoud | Status |
|---|---|---|
| `feat/first-token-logprobs` | Eerste-token-logprobs | PR [#530](https://github.com/youssofal/MTPLX/pull/530) ingediend |
| `feat/prompt-scoring-topk` | Schone PR-tak: alleen de snellere top-K van de scoreroute | PR [#532](https://github.com/youssofal/MTPLX/pull/532) ingediend |
| `feat/faster-prompt-scoring` | Werktak scoreroute (A, B en het terugdraaien van B) | Lokaal; vervangen door `feat/prompt-scoring-topk` |
| `feat/chat-encode-segment-memo` | Chat-encode-memo per segment | Lokaal gecommit (6b3bbf54); meting op echte server open |
| `fix/bank-ceiling-peak-decay` | Piekreserve van het bankplafond laat afnemen | Lokaal gecommit (fb1cd817), standaard uit; eerst werkgeheugen meten |
| `feat/prefix-reuse-block` | Hergebruik van een promptbegin korter dan 512 tokens, completions in de session bank | Lokaal gecommit (fe32206c), live getoetst |
| `refactor/prefix-helpers` | Snellere prefixvergelijking, één lezer per instelling (op `feat/prefix-reuse-block`) | Lokaal gecommit (d32c77b2, 412e1571) |
| `test/classifier-live` | Eerste-token-logprobs en prefix-hergebruik samengevoegd voor de live test | Lokaal, alleen voor metingen |
| `onderzoek` | Deze documentatie | Lopend |
