How to provide GH token to the CI watcher

The watcher (`scripts/ci-watch.ps1`) needs a GitHub token to call the REST API when running as a background service.

Preferred (secure): store token in Windows Credential Manager and let the watcher load it.

Steps:
1. Install the CredentialManager PowerShell module (one-time, requires admin or per-user install):
   pwsh -NoProfile -Command "Install-Module -Name CredentialManager -Scope CurrentUser -Force"

2. Store the token under the target name `QuantumTraderCIWatcher_GH`:
   pwsh -NoProfile -Command "Import-Module CredentialManager; New-StoredCredential -Target 'QuantumTraderCIWatcher_GH' -Username 'gh' -Password 'ghp_your_real_token_here' -Persist LocalMachine"

3. The watcher will attempt to load this token automatically if `GH_TOKEN` is not set in the environment.

Alternative (less secure): set GH_TOKEN in NSSM service env:
  .\scripts\tools\nssm.exe set QuantumTraderCIWatcher AppEnvironmentExtra "GH_TOKEN=ghp_your_real_token_here"
  .\scripts\tools\nssm.exe restart QuantumTraderCIWatcher

Notes:
- Storing tokens in Credential Manager is recommended for security.
- Ensure the service runs under an account that can access the stored credential (LocalMachine persistence works for services running as LocalSystem).
- If using a fine-grained PAT, make sure it has Actions: Read and repository contents read for this repository.
