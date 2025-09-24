Alembic migration helper notes

This repository currently uses `Base.metadata.create_all()` (see
`backend/database.py`) for local development and tests. The repository now
includes a minimal Alembic scaffold so teams can adopt schema migrations if
desired. The scaffold is intentionally small and requires you to install
Alembic before use.

Quick start (local)

1. Install Alembic in your environment:

```pwsh
pip install alembic
```

2. Configure a real DB URL if you don't want to use the default SQLite file.
   Edit `migrations/alembic.ini` or set `sqlalchemy.url` via environment.

3. Create a migration (autogenerate):

```pwsh
alembic -c migrations/alembic.ini revision --autogenerate -m "init"
```

4. Apply migrations locally:

```pwsh
alembic -c migrations/alembic.ini upgrade head
```

Notes & safety
- Running autogenerate against the SQLite file used by tests in CI is
  usually a no-op in CI (ephemeral FS). For production, configure a proper DB
  and review generated migrations before committing them.
- The included `migrations/versions/0001_initial.py` is a safe hand-authored
  initial migration that mirrors the current `TradeLog` and `Settings` models.
