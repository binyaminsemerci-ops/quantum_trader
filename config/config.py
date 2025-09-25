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

	# Coinbase / KuCoin credentials (optional)
	coinbase_api_key: Optional[str] = None
	coinbase_api_secret: Optional[str] = None

	kucoin_api_key: Optional[str] = None
	kucoin_api_secret: Optional[str] = None

	# CryptoPanic news API key
	cryptopanic_key: Optional[str] = None

	# Twitter / X developer credentials (optional)
	x_api_key: Optional[str] = None
	x_api_secret: Optional[str] = None
	x_bearer_token: Optional[str] = None


# Default market quote conventions used across the codebase
# - DEFAULT_QUOTE: spot / main base (we'll prefer USDC as the repo-wide default)
# - FUTURES_QUOTE: markets where margin/futures use USDT (e.g., USDT-margined futures)
DEFAULT_QUOTE = 'USDC'
FUTURES_QUOTE = 'USDT'

# Default base coins (high-volume layer-1 and layer-2 tokens) used for
# training, dashboards and defaults. Keep this list small and focused on
# highest-volume markets to avoid long data pulls during dev runs.
DEFAULT_BASE_COINS = ['BTC', 'ETH', 'BNB', 'SOL', 'ARB']

# Default exchange used when callers do not specify one. This can be
# overridden via environment (set in CI or local .env) or changed here.
DEFAULT_EXCHANGE = os.getenv('DEFAULT_EXCHANGE', 'binance')


def make_pair(base: str, quote: str | None = None) -> str:
	"""Return a concatenated exchange-style pair (e.g. BTCUSDC).

	If quote is None the DEFAULT_QUOTE is used.
	"""
	if quote is None:
		quote = DEFAULT_QUOTE
	return f"{base.upper()}{quote.upper()}"


def futures_pair(base: str) -> str:
	"""Return a futures/cross-margin pair using the FUTURES_QUOTE (USDT).

	Use this when preferring futures-style execution for a base asset.
	"""
	return make_pair(base, quote=FUTURES_QUOTE)


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
		coinbase_api_key=os.getenv('COINBASE_API_KEY'),
		coinbase_api_secret=os.getenv('COINBASE_API_SECRET'),
		kucoin_api_key=os.getenv('KUCOIN_API_KEY'),
		kucoin_api_secret=os.getenv('KUCOIN_API_SECRET'),
		cryptopanic_key=os.getenv('CRYPTOPANIC_KEY'),
		x_api_key=os.getenv('X_API_KEY') or os.getenv('TWITTER_API_KEY') or None,
		x_api_secret=os.getenv('X_API_SECRET') or os.getenv('TWITTER_API_SECRET') or None,
		x_bearer_token=os.getenv('X_BEARER_TOKEN') or os.getenv('TWITTER_BEARER_TOKEN') or None,
	)


__all__ = ["Config", "load_config"]


def masked_config_summary(cfg: Config) -> dict:
	"""Return a safe, masked summary of which secrets are present.

	The function intentionally does not reveal secret values. It returns a
	dictionary that indicates presence and a short masked preview for ease of
	runtime logging or a health UI.
	"""
	def _mask(s: Optional[str]) -> Optional[str]:
		if not s:
			return None
		if len(s) <= 8:
			return "*" * (len(s) - 2) + s[-2:]
		return s[:2] + "*" * (len(s) - 6) + s[-4:]

	return {
		'binance': {'present': bool(cfg.binance_api_key and cfg.binance_api_secret), 'key_preview': _mask(cfg.binance_api_key)},
		'coinbase': {'present': bool(cfg.coinbase_api_key and cfg.coinbase_api_secret), 'key_preview': _mask(cfg.coinbase_api_key)},
		'kucoin': {'present': bool(cfg.kucoin_api_key and cfg.kucoin_api_secret), 'key_preview': _mask(cfg.kucoin_api_key)},
		'cryptopanic': {'present': bool(cfg.cryptopanic_key), 'key_preview': _mask(cfg.cryptopanic_key)},
		'twitter': {'present': bool(cfg.x_api_key or cfg.x_bearer_token), 'key_preview': _mask(cfg.x_api_key or cfg.x_bearer_token)},
	}

__all__.append('masked_config_summary')
