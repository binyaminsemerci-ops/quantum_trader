Subject: Urgent — repository history rewritten and action required

Summary
- The repository history was rewritten to remove large/unsafe files and sensitive blobs. The cleaned history has been pushed to `origin/main`.

What you must do now
1. Rotate secrets immediately if you used any credentials previously stored in the repo (API keys, cloud credentials, CI secrets). Prioritize: GitHub Actions secrets, AWS/GCP keys, Binance keys, database passwords.
2. Re-sync your local clone. If you have no local changes, re-clone or hard-reset your branch.

Quick recovery commands
- Re-clone fresh (recommended):
  git clone https://github.com/binyaminsemerci-ops/quantum_trader.git

- If you have local changes to keep:
  git checkout -b save-local
  git add -A && git commit -m "WIP: save-local"
  git fetch origin
  git switch --force main
  git reset --hard origin/main

Secret rotation checklist
- GitHub Actions: go to repository Settings → Secrets and rotate any secrets used in workflows.
- Cloud providers (AWS/GCP/Azure): rotate IAM/Service Account keys, revoke old keys and update secrets.
- Exchanges (Binance): rotate API keys and remove old API IP whitelists if suspect.

If you need help
- Ping @devops or reply here; I can draft the messages and run the reset in a coordinated window.
