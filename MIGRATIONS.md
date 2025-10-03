# Database Migrations (Alembic)

Alembic is configured for this project to manage schema evolution.

## Layout

- `alembic.ini`: Alembic configuration file.
- `alembic/env.py`: Environment script tying Alembic to `backend.database.Base.metadata`.
- `alembic/versions/`: Individual revision scripts.

## Initial Baseline

Revision `0001_baseline` approximates the current schema (tables created previously by `Base.metadata.create_all`).

## Usage

Set (or rely on default) database URL:

```bash
export QUANTUM_TRADER_DATABASE_URL=sqlite:///backend/data/trades.db
```

Windows PowerShell:

```powershell
$env:QUANTUM_TRADER_DATABASE_URL='sqlite:///backend/data/trades.db'
```

### Upgrade to latest

```bash
alembic upgrade head
```

### Create a new revision (autogenerate diffs)

```bash
alembic revision --autogenerate -m "add new table X"
```

Review the generated file under `alembic/versions/` and adjust if needed.

### Downgrade one revision

```bash
alembic downgrade -1
```

## Development Tips

- Prefer adding new columns/tables via Alembic revisions rather than `create_all` once baseline is established.
- For experimental local changes you can still use `create_all`, but commit a migration before merging.
- Keep revisions linear unless you intentionally need branching (rare for this project scope).

## Troubleshooting

- If autogenerate misses changes, ensure all models are imported before Alembic runs (so metadata is populated).
- If you see duplicate module issues in mypy, confirm `pyproject.toml` has `explicit_package_bases = true`.
- If a revision fails partway, you may need to manually inspect the DB and either fix or run `alembic stamp` to realign.

