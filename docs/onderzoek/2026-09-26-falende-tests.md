# Falende tests op een schone main (vondst 10)

Datum: 2026-09-26. Omgeving: M5 Pro, 64 GB, macOS 26, venv `~/Dev/MTPLX/.venv`, `origin/main` op 1de2b1c0 (2.12.0). Alleen tests gedraaid, geen server en geen echt model.

## Uitkomst

Alle 19 mislukkingen komen door onvoldoende isolatie van de tests, niet door een fout in MTPLX. Twee oorzaken:

| Groep | Tests | Oorzaak | Soort |
|---|---|---|---|
| A | 9 `test_public_cli`, 8 `test_forge_cli`, 1 `test_hf_loader` | De tests lezen de echte `~/.mtplx/config.toml` | Omgeving (onze config), gat in de testisolatie |
| B | 1 `test_laguna_model` | Geheugencontrole voor Laguna-S-2.1 vraagt 85,3 GiB, deze Mac heeft 64 GB | Omgeving (hardware), gat in de testisolatie |

Bewijs: met een lege tijdelijke `HOME` slagen de 18 tests van groep A. Met alleen `model = ...` in een tijdelijke config falen de 9 `test_public_cli`-tests; met alleen `model_dir = ...` de 8 `test_forge_cli`-tests en de `test_hf_loader`-test.

## Groep A: gebruikersconfig

- `tests/conftest.py` isoleert de modelcache met `MTPLX_MODEL_DIR`, maar niet de config. `mtplx/config.py:18` zet `DEFAULT_CONFIG_PATH` op `~/.mtplx/config.toml`; `user_config_path` (`mtplx/config.py:113`) gebruikt die tenzij `MTPLX_CONFIG` is gezet. Veel tests zetten `MTPLX_CONFIG` zelf; deze 18 niet.
- `apply_user_config` (`mtplx/config.py:229-251`) laat de config voorgaan op de env:
  - `model` vervangt het standaardmodel bij `quickstart`, `tune` en `bench run/tune` (`_apply_model_default`, regel 254). Ons model staat niet in de geïsoleerde cache, dus de CLI geeft een pull-hint in plaats van JSON of de verwachte modelnaam (9 `test_public_cli`).
  - `model_dir` wordt `--model-root` bij `forge` (regel 248-249). De tests zoeken het resultaat in `tmp_path/models` (`FileNotFoundError`), terwijl forge in de echte `~/.mtplx/models` schrijft (8 `test_forge_cli`).
  - `model_dir` wordt `--cache-dir` bij `remove` (`_apply_cache_default`, regel 276-278), waardoor de foutmelding over een dubbelzinnige verwijzing anders luidt (1 `test_hf_loader`).
- **Bijwerking:** de buildtests in `tests/test_forge_cli.py` (model `Fixture-MTPLX-Speed` en `Qwen/Qwen3.5-9B`) schreven via `model_dir` in de echte cache; forge kiest bij een bestaande naam het volgende vrije achtervoegsel `-N`. De mappen zijn een gevolg, geen oorzaak: zonder `model_dir` in de config wijst alles naar de geïsoleerde cache en slagen de tests, ook met de mappen aanwezig. Elke suiterun liet mappen `Fixture-MTPLX-Speed-N` en `Qwen-Qwen3.5-9B-MTPLX-Speed-N` achter in `~/.mtplx/models` (klein, ~8 KB elk). Op 26 september om 02:05 stonden er 161; het aantal groeit zolang andere worktrees zonder de fix de suite draaien (met de fix komen er geen bij). Alle `Fixture-MTPLX-Speed*`- en `Qwen-Qwen3.5-9B-MTPLX-Speed*`-mappen kunnen weg; niet opgeruimd, dat beslist Jeroen (`rm -r ~/.mtplx/models/Fixture-MTPLX-Speed* ~/.mtplx/models/Qwen-Qwen3.5-9B-MTPLX-Speed*`; de echte modellen heten `Youssofal--...`).

## Groep B: geheugencontrole Laguna

`test_laguna_s_2_1_ar_route_skips_qwen_performance_hooks` roept `runtime.load` aan zonder de controle `_preflight_laguna_system_memory` (`mtplx/runtime.py:72-86`, aangeroepen op regel 680) te vervangen. Op een Mac onder 85,3 GiB faalt de test daardoor, al gaat hij niet over geheugen. De controle heeft een eigen test met een vaste meting van 64 GB, en de vergelijkbare routetest verderop in het bestand vervangt de controle wel. Bij de maker (meer geheugen) slaagt de test.

## Fix

Tak `fix/test-isolation` (worktree `~/Dev/MTPLX-tests`, commit 0e1b6b91, gepusht naar de fork), alleen tests, 10 regels:

- `tests/conftest.py`: de autouse-fixture `_hermetic_mtplx_state` zet `MTPLX_CONFIG` op een niet-bestaand bestand in zijn tijdelijke map, naast de bestaande `MTPLX_MODEL_DIR`-regel.
- `tests/test_laguna_model.py`: in de AR-routetest de geheugencontrole vervangen, zoals in de andere routetest.

Controle:

- De 19 tests en de verwante bestanden (`test_config`, `test_onboarding`, `test_draft_launch_provenance`, `test_no_mlx_imports`) slagen; het aantal mappen in `~/.mtplx/models` blijft gelijk.
- Volledige suite: 9382 geslaagd, 69 overgeslagen, 0 mislukt (was 19 mislukt); geen nieuwe mappen in `~/.mtplx/models`.
- Ruff: 5 meldingen in de twee bestanden, voor en na gelijk (dus geen nieuwe). `python -m build` slaagt.
- Geen CHANGELOG-regel: de maker schrijft die niet voor wijzigingen in alleen tests (bijv. 135f2fa3 in `conftest.py`).

PR-tekst klaar, PR alleen na akkoord van Jeroen.
