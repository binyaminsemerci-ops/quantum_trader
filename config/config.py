"""Configuration loader for environment secrets and runtime settings.

This module centralizes reading secrets from environment variables and an
optional .env file (python-dotenv). It intentionally does not store any
secrets in the repository; add values to your local `.env` or set CI/host
environment secrets instead.

Supported environment variables:
- BINANCE_API_KEY
- BINANCE_API_SECRET
- CRYPTOPANIC_KEY
- X_API_KEY (Twitter / X developer key)
- X_API_SECRET
- X_BEARER_TOKEN

Usage:
	from config.config import load_config
	cfg = load_config()
	if cfg.binance_api_key:
		# use cfg.binance_api_key, cfg.binance_api_secret

"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
	# python-dotenv is a runtime dependency (listed in requirements). If
	# present, load .env files from the repository root so local development
	# is convenient. If not present, environment variables will still be read.
	from dotenv import load_dotenv

	_DOTENV_AVAILABLE = True
except Exception:
	_DOTENV_AVAILABLE = False


@dataclass
class Config:
	# Binance exchange credentials
	binance_api_key: Optional[str] = None
	binance_api_secret: Optional[str] = None

	# CryptoPanic news API key
	cryptopanic_key: Optional[str] = None

	# Twitter / X developer credentials (optional)
	x_api_key: Optional[str] = None
	x_api_secret: Optional[str] = None
	x_bearer_token: Optional[str] = None


def _load_dotenv_if_present() -> None:
	"""Load a .env file from the repository root if python-dotenv is installed.

	Loading is best-effort and non-fatal: absence of python-dotenv or a
	.env file will not raise an exception.
	"""
	if not _DOTENV_AVAILABLE:
		return
	# Search for a .env file in repository root (two levels up from config)
	repo_root = Path(__file__).resolve().parents[1]
	env_path = repo_root / '.env'
	if env_path.exists():
		load_dotenv(dotenv_path=str(env_path))


def load_config() -> Config:
	"""Return a Config populated from environment variables (and .env).

	The result is intentionally light-weight and returns None for keys not
	present so callers can check presence before attempting to call external
	APIs.
	"""
	_load_dotenv_if_present()

	return Config(
		binance_api_key=os.getenv('BINANCE_API_KEY'),
		binance_api_secret=os.getenv('BINANCE_API_SECRET'),
		cryptopanic_key=os.getenv('CRYPTOPANIC_KEY'),
		x_api_key=os.getenv('X_API_KEY') or os.getenv('TWITTER_API_KEY') or None,
		x_api_secret=os.getenv('X_API_SECRET') or os.getenv('TWITTER_API_SECRET') or None,
		x_bearer_token=os.getenv('X_BEARER_TOKEN') or os.getenv('TWITTER_BEARER_TOKEN') or None,
	)


__all__ = ["Config", "load_config"]
