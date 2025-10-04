# Quantum Trader – Backlog# Quantum Trader – Backlog



This file lists actionable follow-ups now that the repo reflects the demo snapshot. Tasks are grouped by priority; tick them off or move them to GitHub issues as they are completed.This file lists actionable follow-ups now that the repo reflects the demo snapshot. Tasks are grouped

by priority; tick them off or move them to GitHub issues as they are completed.

## High

## High

- [x] **Secrets & configuration**: introduce a central config layer (e.g. `pydantic-settings`) that reads environment variables, removes API keys from SQLite, and masks values in logs/UI.- [x] **Secrets & configuration**: introduce a central config layer (e.g. `pydantic-settings`) that

- [x] **Dependency hygiene**: pin backend requirements, split runtime vs dev tooling, and document optional extras (ccxt, Postgres drivers) without pulling them into default installs.      reads environment variables, removes API keys from SQLite, and masks values in logs/UI.

- [x] **CI policy**: split fast sanity checks from heavy integration jobs (e.g. ccxt smoke) so maintainers can opt in per PR.- [x] **Dependency hygiene**: pin backend requirements, split runtime vs dev tooling, and document

- [x] **Real adapters**: replace mock `/prices` and `/signals` endpoints with real providers (ccxt, sentiment feeds) and add unit tests for the exchange factory.      optional extras (ccxt, Postgres drivers) without pulling them into default installs.

- [x] **Frontend cleanup**: prune duplicate `.new`, `.bak`, `.tsx` files; ensure components read from the updated APIs and handle empty/error states gracefully.- [x] **CI policy**: split fast sanity checks from heavy integration jobs (e.g. ccxt smoke) so

      maintainers can opt in per PR.

## Medium- [x] **Real adapters**: replace mock `/prices` and `/signals` endpoints with real providers (ccxt,

      sentiment feeds) and add unit tests for the exchange factory.

- [ ] **Database story**: decide whether to keep SQLite or restore PostgreSQL + Alembic migrations; update docs accordingly and add seed data scripts for demos.- [x] **Frontend cleanup**: prune duplicate `.new`, `.bak`, `.tsx` files; ensure components read from

- [x] **Training pipeline**: wire up `ai_engine/feature_engineer.py` + `train_and_save.py`, document how to regenerate models, and publish evaluation/backtesting results.      the updated APIs and handle empty/error states gracefully.

- [x] **Observability**: add structured logging, metrics (Prometheus/client), and `/health` endpoints for k8s/uptime checks.

- [ ] **Documentation refresh**: extend `DEVELOPMENT.md` and `CONTRIBUTING.md` with full backend & frontend setup, pre-commit usage, and stress harness tips (partially done, keep iterating).## Medium

- [ ] **Database story**: decide whether to keep SQLite or restore PostgreSQL + Alembic migrations;

## Low      update docs accordingly and add seed data scripts for demos.

- [x] **Training pipeline**: wire up `ai_engine/feature_engineer.py` + `train_and_save.py`, document

- [ ] **UX polish**: add notifications/toasts for key events (saving settings, executing orders), and improve the settings page with masked key previews.      how to regenerate models, and publish evaluation/backtesting results.

- [x] **Deployment**: create production Dockerfiles, compose/k8s manifests, and CI jobs that build and push container images.- [x] **Observability**: add structured logging, metrics (Prometheus/client), and `/health` endpoints

- [ ] **Architecture diagram**: update or redraw the system diagram to reflect the demo-first architecture and the planned roadmap.      for k8s/uptime checks.

- [ ] **Documentation refresh**: extend `DEVELOPMENT.md` and `CONTRIBUTING.md` with full backend &

---      frontend setup, pre-commit usage, and stress harness tips (partially done, keep iterating).



Feel free to turn completed items into GitHub issues or PRs and strike them out here.## Low
- [ ] **UX polish**: add notifications/toasts for key events (saving settings, executing orders),
      and improve the settings page with masked key previews.
- [x] **Deployment**: create production Dockerfiles, compose/k8s manifests, and CI jobs that build
      and push container images.
- [ ] **Architecture diagram**: update or redraw the system diagram to reflect the demo-first
      architecture and the planned roadmap.

---

Feel free to turn completed items into GitHub issues or PRs and strike them out here.
