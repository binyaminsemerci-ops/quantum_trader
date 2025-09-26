This repository keeps `ccxt` as an optional integration dependency to avoid pulling in heavy exchange libraries during normal CI runs.

- Runtime requirements (used by main CI build) are installed from `backend/requirements.txt` with `ccxt` filtered out.
- Integration tests that require `ccxt` (adapter smoke tests) run in the `integration-ccxt` workflow job and install `backend/requirements-ccxt.txt` which contains `ccxt>=2.0.0`.

To run the integration job locally or in CI, ensure `ccxt` is installed in your environment:

```bash
python -m pip install -r backend/requirements-ccxt.txt
```

If you prefer to always install `ccxt` in CI, move the `ccxt` line back into `backend/requirements.txt` (or remove the filter step in `.github/workflows/ci.yml`).
