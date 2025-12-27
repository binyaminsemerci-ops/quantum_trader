# Switch GHCR publish workflow to use GITHUB_TOKEN

Summary

- This branch updates `.github/workflows/docker-publish-3.yml` to grant `packages: write` permission and to log into GHCR using the built-in `GITHUB_TOKEN` instead of an org PAT.

Why

- Using `GITHUB_TOKEN` avoids storing long-lived PATs and is the recommended approach for CI publishing.

Prerequisite for merge (action required by org admin)

1. Go to the Organization settings -> Actions -> Policies (or "Actions settings" depending on UI).

2. Allow GitHub Actions to create packages for the organization or permit workflows to create organization packages using the `GITHUB_TOKEN`.

   - The specific setting is typically named: "Allow GitHub Actions to create and approve packages" or similar.

3. Optionally, review and approve the workflow run for this repo if your org enforces workflow approval for first-time contributors.

Notes

- If the org policy remains set to block package creation via GITHUB_TOKEN, the workflow will fail with an authorization error similar to:

  "denied: installation not allowed to Create organization package"

- If you prefer, you can temporarily keep using the org PAT in the `GHCR_PAT_ORG` secret; this PR intends to switch back to GITHUB_TOKEN once org settings allow it.

How to test after org change

1. Merge this PR after the org admin flips the policy.

2. Trigger the workflow (push or manual) and verify the image is published to `ghcr.io/<org>/quantum_trader`.

Contact

- If you need me to provide the exact nav steps for your org console, tell me the org name and I can supply the step-by-step UI path.
