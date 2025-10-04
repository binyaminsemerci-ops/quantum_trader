# Quantum Trader – Backlog

This file lists actionable follow-ups now that the repo reflects the demo snapshot. Tasks are grouped by priority; tick them off or move them to GitHub issues as they are completed.

## High Priority

- [x] **Secrets & configuration**: introduce a central config layer (e.g. `pydantic-settings`) that reads environment variables, removes API keys from SQLite, and masks values in logs/UI.
- [x] **Dependency hygiene**: pin backend requirements, split runtime vs dev tooling, and document optional extras (ccxt, Postgres drivers) without pulling them into default installs.
- [x] **CI policy**: split fast sanity checks from heavy integration jobs (e.g. ccxt smoke) so maintainers can opt in per PR.
- [x] **Real adapters**: replace mock `/prices` and `/signals` endpoints with real providers (ccxt, sentiment feeds) and add unit tests for the exchange factory.
- [x] **Frontend cleanup**: prune duplicate `.new`, `.bak`, `.tsx` files; ensure components read from the updated APIs and handle empty/error states gracefully.
- [x] **Code modernization**: update deprecated datetime.utcnow() to datetime.now(timezone.utc) for future compatibility.

## Medium Priority

- [x] **Documentation polish**: review all README files, add API documentation with OpenAPI, and create deployment guides.

## Low Priority

- ✅ **Documentation polish**: enhanced API documentation with comprehensive OpenAPI metadata, created detailed API.md and DEPLOYMENT.md guides, enhanced README with architecture diagrams, added comprehensive inline documentation for all major API endpoints while maintaining 58 passing tests
- ✅ **Documentation recovery**: restored all README files from git history and committed to main branch
- ✅ **Error boundaries**: implemented comprehensive exception handling with logging and consistent error responses
- ✅ **Enhanced API client**: created type-safe frontend API client with proper error handling
- ✅ **Integration tests**: added 13+ comprehensive workflow and error handling tests (50 total tests now pass)
- ✅ **Logging system**: structured logging with rotation, multiple levels, and error tracking
- ✅ **Database architecture**: implemented PostgreSQL + Alembic migrations with comprehensive setup guide and demo data seeding scripts
- ✅ **Performance monitoring**: comprehensive metrics collection for HTTP requests, database queries, system resources with /api/metrics endpoints
