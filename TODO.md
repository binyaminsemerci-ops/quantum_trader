# Quantum Trader — Delivery Backlog

Denne backloggen sporer gjenstående arbeid for å levere en autonom, multi-børs kryptohandelsplattform som kan kjøre 24/7 på en hardenert VPS.

## Progress Snapshot

- ✅ Scheduler med provider-failover, Prometheus-metrikker og telemetri-dokumentasjon.
- ✅ Risikovern med vedvarende state, admin audit logging og JSON-strukturert logging.
- ✅ Baseline XGBoost-modell og treningspipeline sjekket inn.
- 🔄 Frontend-dashboard er funksjonelt, men trenger produksjonsherding og testdekning.
- 🔄 Bybit/KuCoin-adaptere og full handelsutførelse gjenstår.
- 🔄 DevOps-automatisering (CI/CD, hemmeligheter, dashboards/alarmer) er under arbeid.

## Critical Path To Production

- [ ] Ship autentisert adminoverflate (tokens/rolle-sjekker) for risikostyring og mutable endepunkter.
- [ ] Implementer Bybit- og KuCoin-adaptere med sandbox-ordre og kvitteringssporing.
- [ ] Bygg samlet handelsmotor med posisjonsstørrelse, leverage-kontroller og PnL-avstemming.
- [ ] Automatiser retraining-livssyklus med evalueringsporter, driftsovervåkning og promotions.
- [ ] Etabler produksjonsoperasjoner: dashboards, alarmer, backup/restore og VPS-utrulling.

## Platform Foundations & Risk

- [x] Persist risk guard state (SQLite/Redis) med admin-overstyring og reset-endepunkter.
- [x] Audit logging for risikobedrifter støtter compliance-gjennomgang.
- [x] JSON-strukturert logging med `X-Request-ID` og konfigurerbar logg-nivå.
- [ ] Auth/autorisasjon for mutable endepunkter og adminverktøy.
- [ ] Harden scheduler-konfigurasjon (runtime overrides, persistens, manuell trigger, graceful shutdown).
- [ ] Dokumentér dataingest/restore playbook i `docs/`.

## Exchange Connectivity & Execution

- [ ] Implementer Bybit REST/WebSocket-adaptere samsvarende med Binance-rutene.
- [ ] Implementer KuCoin (eller alternativ) adapter for multi-børs kravet.
- [ ] Bygg samlet ordreutførelse med posisjonsstørrelse, leverage og kvitteringssporing.
- [ ] Sandbox/paper trading integrasjonstester som treffer ekte API-er bak feature flagg.
- [ ] Eksponer exchange health og rate-limit telemetri via Prometheus og `/health`.
- [ ] Etabler realtime feeds (WebSocket/SSE) for fills, posisjoner og orderbok.

## AI Lifecycle & Data

- [x] Baseline XGBoost-artefakter under versjonskontroll.
- [ ] Spor retraining-jobber (status, metrics, alerts) og begrens parallellitet.
- [ ] Validér modeller før promotion (hold-out metrics, drift, rollback tooling).
- [ ] Eksponer modellprestasjon og metadata via API/UI.
- [ ] Automatiser evalueringsporter slik at kun grønne modeller går live.
- [ ] Etabler kontinuerlig paper-trading loop for modellobservasjon.
- [ ] Versjoner feature pipelines og dokumenter data lineage.

## Observability & Operations

- [x] Prometheus-metrikker for HTTP, scheduler, providere og risikovern.
- [x] Emit Prometheus-teller for adminhendelser per event/severity og suksessutfall.
- [ ] Instrumentér cache hit/miss og modell-inferens tid/kvalitet.
- [ ] Publiser Grafana/Azure dashboards og alert-regler iht `docs/telemetry_plan.md`.
- [ ] Emit strukturerte hendelser for handelsbeslutninger, retrain-resultat og failovers.
- [ ] Sentraliser logger (Loki/ELK) og definér retention + søk.
- [ ] Implementer deployment-/versjonssporing og del endringslogger med operatører.
- [ ] Sett opp syntetiske helsesjekker for eksterne API-er/børser.

## Frontend Experience

- [ ] Migrer legacy-paneler til kontrollert state med last/feilhåndtering.
- [ ] Visualiser backend health, risiko og nøkkel-metrikker i dashboardet.
- [ ] Externaliser API-endepunkter og pollingintervaller via Vite env-config og prod-profiler.
- [ ] Legg til sanntidsoppdateringer (WebSocket/SSE) for priser, signaler og handler.
- [ ] Utvid unit coverage og legg til Playwright-røyktester for kritiske brukerreiser.
- [ ] Harden buildpipeline (lint/typecheck/test før deploy) og dokumenter releaseflyt.

## Infrastructure & Deployment

- [ ] Harden Docker Compose for prod (volumer, restart policies, healthchecks, secrets).
- [ ] Sett opp CI pipelines for image-builds, sikkerhetsskanning og artefaktpromotering.
- [ ] Lever systemd/PM2 (eller tilsvarende) templates for VPS-drift.
- [ ] Automatiser database migrasjoner og risk-state backups under deploy.
- [ ] Implementer hemmelighetshåndtering (Vault/Azure Key Vault eller sikrede env-vars).
- [ ] Etabler katastrofeberedskap (backups, runbooks, failover-øvelser).

## Quality, Compliance & Testing

- [ ] Bygg end-to-end testsuite for handels-happy path og feilhåndtering.
- [ ] Lag last-/stress-tester for scheduler og signalendepunkter.
- [ ] Legg til sikkerhetstesting (OWASP, dependency/secret scanning) i CI.
- [ ] Dokumenter regulatorisk scope (KYC/AML) og audit trail strategi.
- [ ] Formaliser releaseprosessen med godkjenninger og rollback-guidelines.

## Documentation & Developer Experience

- [x] README oppdatert med arkitektur, setup og roadmap.
- [x] Staging guide, telemetri-plan, risk guard spec og failover-plan publisert.
- [ ] Utvid backend/README med runbooks, feilsøking og incident response.
- [ ] Lever onboarding scripts/guider for Windows, macOS og Linux.
- [ ] Vedlikehold changelog og release notes for deploys og modellpromotions.
- [ ] Utvid OpenAPI beskrivelser og publiser klienteksempler.
- [ ] Rydd opp dupliserte READMEs/backups og dokumenter filstruktur for nye utviklere.
