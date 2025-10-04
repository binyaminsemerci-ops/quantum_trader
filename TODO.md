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

- [ ] **Database story**: decide whether to keep SQLite or restore PostgreSQL + Alembic migrations; update docs accordingly and add seed data scripts for demos.
- [x] **Feature parity**: make sure all UI components read from the updated APIs (no more hardcoded/mocked data) and that settings persist between sessions.
- [x] **Error boundaries**: add React error boundaries to the frontend and proper exception handling in backend routes.
- [x] **Testing coverage**: add integration tests for the main trading workflows and extend unit test coverage.

## Low Priority

- [ ] **Performance monitoring**: add metrics/logging for response times, database queries, and frontend bundle size.
- [ ] **Documentation polish**: review all README files, add API documentation with OpenAPI, and create deployment guides.
- [ ] **UX polish**: add notifications/toasts for key events (saving settings, executing orders), loading states, and better error messages.
- [ ] **Security audit**: review authentication flow, add rate limiting, and ensure all user inputs are properly validated.

## Recently Completed

- ✅ **Repository cleanup**: removed duplicate quantum_trader directory and backup files
- ✅ **Dependency fixes**: installed pytest-asyncio and resolved async test support  
- ✅ **Code quality**: fixed all datetime deprecation warnings across codebase
- ✅ **Test infrastructure**: all 37 backend tests now pass with proper async support
- ✅ **Documentation recovery**: restored all README files from git history and committed to main branch
- ✅ **Error boundaries**: implemented comprehensive exception handling with logging and consistent error responses
- ✅ **Enhanced API client**: created type-safe frontend API client with proper error handling
- ✅ **Integration tests**: added 13+ comprehensive workflow and error handling tests (50 total tests now pass)
- ✅ **Logging system**: structured logging with rotation, multiple levels, and error tracking