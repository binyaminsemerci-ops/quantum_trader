# Backend Execution Plan

Progress tracker for the remaining backend work identified on 2025-11-07.

## Task List

1. **Instrumentation: cache + model telemetry**
   - [x] Add Prometheus counters for cache hit/miss in `backend/utils/cache.py` and in-memory caches.
   - [x] Expose helper wrappers to emit metrics from cache callers.
   - [x] Add histogram/timer around model inference paths (AI signal generation).
   - [x] Cover scheduler + cache instrumentation with unit tests.

2. **Risk guard hardening**
   - [x] Implement price sanity checks prior to execution submissions.
   - [x] Extend risk guard service/tests for the new validation path.
   - [x] Document failover drill + emergency shutdown process.

3. **Model governance automation**
   - [x] Scaffold data ingestion & feature pipeline scripts with logging/metrics.
   - [x] Capture training run metadata (config + metrics) in persistent store.
   - [x] Surface current model version/status via backend endpoint.

4. **Operational documentation & compliance**
   - [x] Expand backend README with deployment/incident runbooks.
   - [x] Add compliance checklist covering secrets, access controls, regulatory notes.

5. **Database promotion**
   - [x] Schedule and document Alembic upgrade procedure for staging/production.
   - [x] Validate latest migration against production snapshot (dry run script).

---

Work each top-level group sequentially; after touching a file run the backend test suite to keep the tree green.
