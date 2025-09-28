# TODO — quantum_trader

Denne filen oppsummerer hva som er gjort, hva som gjenstår, og konkrete prioriterte oppgaver (kort og handlingsorientert).

## Kort status
- Stress-harness og frontend Docker-testflow er implementert.
- Per-iterasjon og aggregerte JSON-artefakter skrives til `artifacts/stress/`.
- Frontend Docker-image bygges én gang (caching), repeterbare frontend-kjøringer finnes i `scripts/stress/run_frontend_repeats.py`.
- Sentral orkestrator: `scripts/stress/harness.py` (kjører pytest, backtest, frontend).
- Uploadskript: `scripts/stress/upload_artifacts.py` (krever `aws`/`gsutil`/`az` CLI på PATH).

## Immediate next 3 (høy prioritet)
1. Gjør `harness.py` til eneste entrypoint/full orchestrator — FULLFØRT
   - Resultat: `scripts/stress/run_frontend_repeats.py` er en tynn wrapper; `harness.py` har auto-zip (`--zip-after`) og bugfix for stdout/skyggevariabel. CI-smoke kjører harness.
   - Filer: `scripts/stress/harness.py`, `scripts/stress/run_frontend_repeats.py`, `.github/workflows/stress-tests.yml`, `.vscode/tasks.json`.

2. Sett opp artifact upload i CI — FULLFØRT
   - Resultat: `stress-tests.yml` laster opp `artifacts/stress/**` som GitHub-artifact og støtter valgfri sky-opplasting (S3/GCS/Azure) via secrets, med retries/backoff. `.env.example` dokumenterer creds.
   - Filer: `.github/workflows/stress-tests.yml`, `.env.example`, `scripts/stress/upload_artifacts.py` (zip-only, retries).

3. CI-watcher for `frontend-audit-prod` + npm-audit-triage — FULLFØRT
   - Resultat: Triage-script (`scripts/tools/npm_audit_triage.py`) genererer Markdown og JSON-sammendrag. Integrert i `ci.yml` og `daily-frontend-audit.yml`, artefakter lastes opp, og valgfri automatisk issue-opprettelse når secret `AUDIT_CREATE_ISSUES` er `true`.
   - Filer: `scripts/tools/npm_audit_triage.py`, `.github/workflows/ci.yml`, `.github/workflows/daily-frontend-audit.yml`.

## Prioritert backlog (lengre liste)
- Høy
  - Fjern gamle helper-filer og slå sammen gjenværende duplikater. — FULLFØRT
    - Fjernet: `scripts/stress/run_frontend_repeats.py.tmp`, `scripts/stress/run_frontend_repeats.py.new`, `scripts/stress/run_frontend_docker.py`.
  - Legg til automatisk issue-opprettelse for audit-triage (valgfritt, guarded av secret) — FULLFØRT
  - Robust opplasting: retries, multipart for store filer. — FULLFØRT
  - Automatisk rotasjon/retention for artifacts (retain last N, zip+delete eldre). — FULLFØRT

- Medium
  - Forbedre aggregated.json-format (current shape: nested `frontend` with `frontend_summary`). Tilpass etter behov. — FULLFØRT (lagt til `stats`-seksjon, kompatibel med eksisterende forbrukere; rapport bruker stats når tilgjengelig; HTML-rapport viser prosent og gnistdiagram per task)
  - CI hygiene: kjøre mypy/ruff/black/Bandit og fikse resterende advarsler i CI. — FULLFØRT (alle verktøy kjørt lokalt via `.venv`: ruff, black --check, mypy, bandit passer uten funn)
  - Docker image pinning / reproducible builds (multi-arch hvis ønsket). — FULLFØRT (frontend testimage pinnet til digest; overstyr via `NODE_IMAGE_REF`).

- Lav
  - Dashboard / enkel frontend for å vise trender (pass rate, avg duration).
  - Evolutionary experiments (parameterized runs med forskjellige node-versjoner / deps).

## Viktige filer og hva de gjør
- `scripts/stress/harness.py` — sentral orchestrator. Kjører pytest, backtest og frontend (Docker).
- `scripts/stress/run_frontend_repeats.py` — frontend-only repeat-runner (build once + run N ganger).
- `scripts/stress/merge_frontend_into_aggregated.py` — merge frontend_aggregated.json inn i artifacts/stress/aggregated.json (shape X).
- `scripts/stress/upload_artifacts.py` — zip + upload via CLI (s3/gs/azure). Krever CLI på PATH.
- `frontend/Dockerfile.test` — Dockerfile brukt for reproducible frontend test runs.
- `artifacts/stress/` — hvor per-iterasjon og aggregerte JSONer lagres.

### Note (2025-09-28)
- `scripts/stress/harness.py` is now the canonical orchestrator for stress runs (pytest, backtest, frontend). The older `scripts/stress/run_frontend_repeats.py` has been converted to a tiny compatibility wrapper that delegates to the harness; prefer calling `harness.py` directly in new automation.
 - `stress-tests.yml` publiserer artefakter og kan laste opp til skyleverandør (se secrets). Audit-triage kjører i CI og daily-audit og produserer Markdown/JSON.
 - Automatisk GitHub-issue opprettes/oppdateres for audit-triage når `AUDIT_CREATE_ISSUES` secret er `true`.

## Kommandoer (PowerShell) — praktisk
- Kjør full harness (1 iterasjon som standard):
```powershell
python scripts/stress/harness.py --count 1
```

- Frontend-only repeter (build once, kjør 100 ganger):
```powershell
#$env:DOCKER_FORCE_BUILD='1' # optional, tving rebuild
python scripts/stress/run_frontend_repeats.py --count 100
```

- Merge frontend-aggregert inn i hovedaggregat:
```powershell
python scripts/stress/merge_frontend_into_aggregated.py
```

- Zip og last opp artifacts til S3 (krever aws CLI på PATH og konfigurert creds):
```powershell
python scripts/stress/upload_artifacts.py --provider s3 --dest s3://my-bucket/path/stress.zip
```

## Operasjonelle anbefalinger
- Kjør store runs (1000+) med periodic upload to cloud to avoid disk exhaustion.
- Sett `DOCKER_FORCE_BUILD=1` ved endringer i frontend for å sikre image rebuild.
- I CI: bruk en steg som bygger samme `frontend/Dockerfile.test` og kjører `npm ci`/`vitest` i samme image for parity.

## Security / credentials
- Do not commit tokens or credentials. Use environment variables or CI secrets.
- If using SDK (boto3/gcs/azure) prefer reading credentials from environment/CI secret store.

## Hvis du vil at jeg skal gjøre det nå
- Skriv hvilket punkt jeg skal prioritere: `1` (harness centralization), `2` (uploader / provider), `3` (CI watcher/npm-audit).
- Eller be meg om å opprette denne TODO som en issue/PR i repo (jeg kan lage en PR hvis du vil).

---
Oppdatert: 2025-09-28
