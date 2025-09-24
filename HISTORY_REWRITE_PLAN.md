History rewrite plan (safe): remove accidentally committed `.venv` blobs

Overview
- During work on `chore/agent-instructions-alembic`, a local virtualenv `.venv/` was accidentally
  committed and pushed. Although `.venv/` is now untracked and added to `.gitignore`, the large
  binary blobs remain in remote history. Removing them requires rewriting repository history and
  force-pushing cleaned refs. This document provides a safe plan using `git-filter-repo` (preferred)
  and a fallback using the BFG Repo-Cleaner.

Prerequisites & warnings
- You must be a repository admin or have push rights.
- This is disruptive: all contributors must re-clone or carefully update local clones after the
  rewrite. Coordinate a freeze window with the team (30–60 minutes).
- Take a backup mirror before making changes.

Preferred method: git-filter-repo

1) Install git-filter-repo
```powershell
pip install git-filter-repo
```

2) Create a mirror clone and backup
```powershell
cd C:\temp
git clone --mirror https://github.com/binyaminsemerci-ops/quantum_trader.git quantum_trader-mirror.git
Compress-Archive -Path quantum_trader-mirror.git -DestinationPath quantum_trader-mirror-backup.zip
```

3) Run filter to remove `.venv` and specific files
```powershell
cd quantum_trader-mirror.git
# Remove the entire .venv folder from history
git filter-repo --path .venv --invert-paths

# Alternatively, remove a specific file by path (example)
# git filter-repo --path .venv/Lib/site-packages/xgboost/lib/xgboost.dll --invert-paths
```

4) Verify object removal
```powershell
git count-objects -vH
git for-each-ref --format='%(refname) %(objectname)' refs/heads/
```

5) Force-push cleaned refs to origin (coordinate with team!)
```powershell
git push --force --mirror https://github.com/binyaminsemerci-ops/quantum_trader.git
```

6) Post-actions for contributors
- Ask contributors to re-clone the repo. If they must preserve local work, provide explicit
  instructions to fetch the cleaned refs and rebase their branches.

Fallback: BFG Repo-Cleaner

1) Download BFG jar: https://rtyley.github.io/bfg-repo-cleaner/
2) Mirror clone and backup (same as above)
3) Run BFG to delete `.venv` folders or specific filenames
```powershell
java -jar bfg.jar --delete-folders .venv --no-blob-protection quantum_trader-mirror.git
cd quantum_trader-mirror.git
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force --mirror https://github.com/binyaminsemerci-ops/quantum_trader.git
```

Verification
- After rewrite, confirm repository size reduction with `git count-objects -vH` and GitHub repo insights.

Rollback
- Keep the backup tarball until all consumers confirm the rewrite is good.
- If needed, restore by force-pushing the backup mirror.

Notes
- This plan intentionally removes `.venv` entries. If other large files were accidentally pushed, add
  their paths to the filtering step.
