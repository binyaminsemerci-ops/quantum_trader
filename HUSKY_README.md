Husky install notes
===================

If you run `npm install` from within the `frontend` folder the `prepare` lifecycle script may fail because Husky cannot find the repository `.git` when invoked from certain environments (CI sandboxes or nested shells).

Recommended local steps (PowerShell):

```powershell
cd <repo-root>/frontend
npm install
# make sure prepare runs where .git is visible, or run it explicitly from the frontend dir
npm run prepare
```

If you must run `npm run prepare` from a nested environment and Husky complains that `.git` can't be found, set the `GIT_DIR` environment variable to the repo root before running prepare:

```powershell
# from the frontend folder
$env:GIT_DIR = 'C:\path\to\your\repo\.git'
npm run prepare
```

If you prefer to skip prepare during install (for CI or when creating a lockfile here), install with:

```powershell
npm install --ignore-scripts
# then run prepare manually on a machine with the repo available
npm run prepare
```

This repo's migration branch included a lockfile created with `--ignore-scripts` to avoid sandbox lifecycle failures. Developers should run `npm run prepare` locally once node_modules are present.
