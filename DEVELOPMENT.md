# Development Guide — Quantum Trader

This guide covers local stress runs, retention, and artifact handling.

## Prerequisites
- Python 3.11+ installed and on PATH (`python`/`python3`)
- Docker installed (for frontend tests inside a container)
- Optional: Node.js if you want to run frontend tests without Docker

## Environment
- Copy `.env.example` to `.env` (optional) and adjust values as needed.
- Useful env vars for stress runs:
  - `STRESS_PREFER_DOCKER=1` — prefer running frontend tests in Docker
  - `DOCKER_FORCE_BUILD=1` — force rebuild of frontend test image
  - `STRESS_KEEP_ZIPS=5` — keep latest N zip archives after a run with `--zip-after`
  - `STRESS_KEEP_ITERS=200` — keep latest N per-iteration JSON files after `--zip-after`
  - `STRESS_PRUNE_ALERT_THRESHOLD=100` — emit warning if retention removes more than this many iteration files in one pass
  - `STRESS_PRUNE_ALERT_WEBHOOK=https://hooks.slack.com/...` — optional webhook invoked when threshold is exceeded
  - `STRESS_REPORT_OUTDIR=relative/or/absolute/path` — override where `generate_report.py` reads/writes aggregated/report (useful for testing)

## Running stress harness
Run a single iteration and zip artifacts after:

```bash
python scripts/stress/harness.py --count 1 --zip-after
```

Run 10 iterations, prefer Docker, and keep last 200 iteration JSON files:

```bash
export STRESS_PREFER_DOCKER=1
export STRESS_KEEP_ITERS=200
python scripts/stress/harness.py --count 10 --zip-after
```

Artifacts:
- Iteration results: `artifacts/stress/iter_*.json`
- Aggregated: `artifacts/stress/aggregated.json` (includes `stats` section)
- HTML report: `artifacts/stress/report.html`
- Zips: `artifacts/stress_artifacts_<timestamp>.zip`

## Retention utilities
- Zip rotation is automatic with `STRESS_KEEP_ZIPS` or via uploader `--retain N`.
- Prune old iteration JSONs manually:

```bash
python scripts/stress/retention.py --keep 200 --warn-over 100
```

## Cloud upload (optional)
Use `scripts/stress/upload_artifacts.py` to zip and upload stress artifacts.
Pick one provider and set credentials in your env (`.env` can help):

```bash
python scripts/stress/upload_artifacts.py --provider s3 --dest s3://my-bucket/path/stress.zip --retries 3 --retry-delay 3 --retain 5
```

Pinned provider SDKs for CI are defined in `requirements-ci-upload.txt`.

## CI notes
- Workflow `Stress tests` builds a Docker image for frontend tests and runs one iteration with `--zip-after`.
- It uploads artifacts and (optionally) zips to cloud when secrets are set.
- The HTML report is generated and uploaded as `stress-report` artifact.
- Frontend audit triage runs in main CI and daily audit; optional issue is created when `AUDIT_CREATE_ISSUES` secret is `true`.

## Frontend test image (digest pin)
- `frontend/Dockerfile.test` supports a build arg `BASE_IMAGE` (defaults to `node:20-bullseye-slim`).
- For full reproducibility in CI, set a digest ref:

```yaml
env:
  NODE_IMAGE_REF: node@sha256:<digest>
```

The stress workflow defaults to `node:20-bullseye-slim@sha256:1c2b56658c1ea4737e92c76057061a2a5f904bdb2db6ccd45bb97fda41496b80`; override via the `NODE_IMAGE_REF` secret if you need a different digest.
