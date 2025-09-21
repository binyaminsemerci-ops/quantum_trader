Playwright smoke test scaffold

This folder contains a minimal Playwright smoke test that checks the SPA shell loads.

How to run locally

1. Install Playwright (recommended):

```powershell
cd frontend
npm i -D @playwright/test playwright
npx playwright install
```

2. Run the smoke test (starts in headed mode by default):

```powershell
# point PLAYWRIGHT_BASE_URL at your dev server if different
$env:PLAYWRIGHT_BASE_URL='http://127.0.0.1:3001/' ; npx playwright test tests/playwright/smoke.spec.ts
```

CI guidance

- Add Playwright as a devDependency in your CI job, run `npx playwright install --with-deps` on Linux, and run the same test command. Or use the official Playwright GitHub Action.

Notes

- This PR only adds the test scaffold and documentation. Installing Playwright changes package.json; install/playwright should be done by the maintainer when they accept the PR.
