# Why you might need GHCR_PAT


If your organization blocks the default GITHUB_TOKEN from writing packages, the
workflow `workflow_at_sha.yml` will skip pushing to ghcr.io unless an explicit
Personal Access Token is provided via the repository secret `GHCR_PAT`.

Create a PAT with these scopes:

- For classic PAT: check `write:packages` and `repo` (if the repo is private).
- For fine-grained PAT: grant repository access to this repo and the
  "Packages: Write" permission.

How to add the secret
----------------------

1. Generate the token:
   - Open your GitHub tokens page (https://github.com/settings/tokens) and
     create a new token. Choose a classic PAT or a fine-grained token as
     preferred.
   - Click "Generate new token" (classic) or "Generate new fine-grained token"
   - Select the scopes described above and create the token.

2. Add the repository secret:
   - Go to the repository page -> Settings -> Secrets -> Actions -> New
   - Name: GHCR_PAT
   - Value: paste the token
   - Save

After adding `GHCR_PAT`, rerun the workflow. The publish job will then log in
to ghcr.io using the token and attempt to push the image.

If you prefer to allow the workflow to push using the built-in `GITHUB_TOKEN`,
ask your organization admins to permit package write access for GitHub Actions
in the organization policies.
