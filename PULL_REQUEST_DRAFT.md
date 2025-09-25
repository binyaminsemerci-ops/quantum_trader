PR: Mypy fixes, test helpers, and CI updates

Summary
- Apply small, safe mypy-friendly changes across backend and ai_engine (casts, Optional annotations, guarded imports).
- Move test shims into `tests/_helpers` and use them in tests so CI/test runs do not require heavy runtime deps.
- Add a lightweight GitHub Actions workflow `.github/workflows/ci-mypy.yml` that installs common type stubs before running mypy and pytest.

Notes for reviewers
- I intentionally kept the production package `__init__` files and removed only the temporary production shims. The test helpers live in `tests/_helpers` to avoid shipping placeholder logic in prod modules.
- The CI workflow is conservative: it installs `pandas-stubs` and `scikit_learn_stubs` to make mypy pass in CI. If you'd rather keep CI minimal, we can instead add narrow `# type: ignore[...]` comments where needed.

Next steps
- If CI reports issues, I will iterate on the exact mypy failures and either add targeted ignores or install any additional stub packages required.
- Optionally, we can later replace test stubs with real implementations and remove the test helpers.
