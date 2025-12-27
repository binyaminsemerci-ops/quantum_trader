# Admin Token Usage

This note captures the current authentication surface for administrative API
endpoints and how the shared token is wired through the stack.

## Environment configuration

- `QT_ADMIN_TOKEN` is the single shared secret used to gate admin-only
  operations.
- Configure it in `backend/.env` (or the platform's secret store) for every
  environment where unbounded access is not acceptable. Use a long, random
  string.
- Leaving `QT_ADMIN_TOKEN` unset disables the guard. This is only acceptable for
  temporary local experiments.

`backend/.env.example` includes the variable with guidance and the backend
README describes the operational expectations for staging/production.

## HTTP usage

When `QT_ADMIN_TOKEN` is set, clients must provide the matching value in the
`X-Admin-Token` header. The following endpoints currently enforce the header:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/risk` | Returns risk guard snapshot and configuration. |
| `POST` | `/risk/kill-switch` | Overrides the kill switch (true/false/null). |
| `POST` | `/risk/reset` | Clears recorded trades and overrides. |
| `GET` | `/settings` | Retrieves application settings (secure or full view). |
| `POST` | `/settings` | Updates trading and notification preferences. |
| `GET` | `/trades` | Returns historical trade data with filters. |
| `POST` | `/trades` | Executes manual trade requests (used by admin tooling). |
| `GET` | `/trades/recent` | Provides deterministic sample trades for QA tooling. |
| `GET` | `/trade_logs` | Retrieves structured trade log entries for auditing. |
| `WS` | `/ws/dashboard` | Streams live dashboard metrics for admin consoles. |
| `POST` | `/ai/scan` | Runs model-assisted scanning across provided symbols. |
| `POST` | `/ai/reload` | Reloads AI model and scaler artifacts. |
| `POST` | `/ai/train` | Schedules background model retraining. |
| `GET` | `/ai/tasks` | Lists recorded AI training task state. |
| `GET` | `/ai/tasks/{task_id}` | Retrieves details for a specific training task. |
| `GET` | `/ai/status` | Returns current model/scaler load status. |

Additional mutable surfaces (scheduler controls, settings, etc.) will adopt the
same header as they graduate from the TODO backlog. Keep the header name and
semantics consistent.

## Testing defaults

`backend/tests/conftest.py` ensures `QT_ADMIN_TOKEN` defaults to
`test-admin-token` during pytest runs so fixtures can interact with the admin
surface without additional configuration. Tests that hit the endpoints reuse the
constant via helper modules (see `backend/tests/test_risk_admin_audit.py` and
`backend/tests/test_settings.py`).

If you run the backend manually while keeping the token set to the default test
value, send `X-Admin-Token: test-admin-token` with your requests. For any shared
environment, rotate the token to a unique secret.

## Verification quick check

```pwsh
# Expect a 401 without the header
curl http://localhost:8000/risk

# Repeat with the header to receive the payload
curl http://localhost:8000/risk -H "X-Admin-Token: $env:QT_ADMIN_TOKEN"
```

The application logs failures via `risk.auth.*` and `admin.auth.*` events, and
audit entries are written to the path controlled by `QT_ADMIN_AUDIT_PATH`
(configured by tests). Each audit entry now includes structured metadata such
as `category`, `action`, and `severity`, enabling downstream alerts or SIEM
rules to pivot on event importance. Use those logs to monitor access attempts
and validate that the guard is operating correctly.
