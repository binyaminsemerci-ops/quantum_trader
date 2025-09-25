Local configuration

This folder contains helper code to load runtime configuration and secrets.

- `load_config()` reads environment variables and (optionally) a `.env` file at the
  repository root when `python-dotenv` is installed.

Create a `.env` file from the repository-root `.env.example` for local development.
Do NOT commit real credentials to the repository. Use GitHub Secrets for CI and
production deployments.
