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

Checklist & CI guidance
- [ ] Wait for GitHub Actions on this branch (chore/mypy-fixes) to complete.
- [ ] If mypy fails in CI, collect the mypy job logs (error messages list files and lines).
- [ ] For each mypy failure, prefer one of:
	- Add a minimal, local fix (Optional[Any], cast, TYPE_CHECKING import), or
	- Add a narrow `# type: ignore[...]` on the offending line (only when necessary), or
	- Add the missing stub package to `.github/workflows/ci-mypy.yml` if multiple errors stem from the same missing stub.
- [ ] If tests fail, inspect pytest output and reproduce locally using the workflow's Python version.

How to collect CI logs locally (if you want me to fetch them):
1) Create a GitHub token with repo/read:packages and actions:read (or use the existing `MY_PAT`).
2) Run the helper script included at `scripts/monitor_pr_ci.py` (see README-like instructions below).

