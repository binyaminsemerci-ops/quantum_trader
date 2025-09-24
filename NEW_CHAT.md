NEW CHAT: feature/new-chat-20250924-0300

Snapshot (2025-09-24)

- Branch: feature/new-chat-20250924-0300 (local)
- Current working file: ai_engine/agents/xgb_agent.py
- Key recent changes:
  - Added robust Twitter and CryptoPanic clients with retries and caching.
  - Hardened Binance client to use env vars and lazy imports.
  - Refactored XGBAgent to be resilient to missing deps, async-friendly, and integrate sentiment/news.
  - Training harness (`ai_engine/train_and_save.py`) can fall back to synthetic dataset; training artifacts were created in `ai_engine/models/`.
  - Requirements pinned and adjusted to resolve binary incompatibilities.
  - Added `/api/ai/tasks` endpoints and basic frontend/test scaffolding.
- Tests: Ran targeted pytest files; fixed a corrupted pytest in venv; two client & agent tests pass.

Goals for this new chat branch

1) Add unit tests for `ai_engine/train_and_save.py` (mocking external_data endpoints).
2) Add CI workflow (GitHub Actions) that runs: install backend reqs, pytest, ruff/black checks.
3) Build a small frontend panel for `/api/ai/tasks` and `/api/ai/scan`.

Immediate next task (pick one):
- [ ] Implement tests for `train_and_save` (recommended next).
- [ ] Create GitHub Actions `ci.yml` to run tests & linters.
- [ ] Build a minimal React panel that lists training tasks.

How to run local checks

# Activate venv and run tests
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
pytest backend/tests -q

# Run training (quick):
python -c "from ai_engine.train_and_save import train_and_save; train_and_save(symbols=['BTCUSDT','ETHUSDT'], limit=120)"

Notes
- If you want me to proceed, reply with which immediate task (1/2/3) to pick and I'll start implementing changes on this branch.
- If you prefer a different branch name, say so and I'll rename.
