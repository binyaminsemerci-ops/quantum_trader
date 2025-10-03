#!/usr/bin/env python3
"""Production API Configuration Setup Script.

This script helps configure live API keys and enables production features
for the Quantum Trader system.
"""

import shutil
import sys
from pathlib import Path
from typing import Optional


def create_production_env() -> bool:
    """Create a production-ready .env file with live API configuration."""
    # Check if .env already exists
    env_path = Path(".env")
    if env_path.exists():
        backup_path = Path(".env.backup")
        shutil.copy(env_path, backup_path)

    # Read .env.example as template
    example_path = Path(".env.example")
    if not example_path.exists():
        return False

    with open(example_path) as f:
        template = f.read()

    # Production configuration options
    production_config = {
        "ENABLE_LIVE_MARKET_DATA": "1",
        "ENABLE_REAL_TRADING": "0",  # Start with paper trading
        "BINANCE_TESTNET": "1",  # Start with testnet
        "DEFAULT_BALANCE": "10000",
        "DEFAULT_RISK_PERCENT": "1.0",
        "CCXT_TIMEOUT_MS": "30000",  # Increased timeout for live data
    }

    for key, value in production_config.items():
        pass

    # Get API keys from user

    api_keys = {}

    # Binance API
    binance_key = input("Binance API Key: ").strip()
    if binance_key:
        binance_secret = input("Binance API Secret: ").strip()
        if binance_secret:
            api_keys["BINANCE_API_KEY"] = binance_key
            api_keys["BINANCE_API_SECRET"] = binance_secret

    # Twitter/X API
    x_bearer = input("X Bearer Token: ").strip()
    if x_bearer:
        api_keys["X_BEARER_TOKEN"] = x_bearer

    # Build production .env content
    env_content = template

    # Apply production settings
    for key, value in production_config.items():
        # Replace existing lines or add new ones
        if f"{key}=" in env_content:
            # Find and replace existing line
            lines = env_content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    break
            env_content = "\n".join(lines)
        else:
            # Add new line
            env_content += f"\n{key}={value}"

    # Apply API keys
    for key, value in api_keys.items():
        if f"{key}=" in env_content:
            lines = env_content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    break
            env_content = "\n".join(lines)
        else:
            env_content += f"\n{key}={value}"

    # Write production .env
    with open(env_path, "w") as f:
        f.write(env_content)


    return True


def verify_configuration() -> Optional[bool]:
    """Verify the production configuration is working."""
    try:
        # Import and test configuration
        sys.path.insert(0, str(Path.cwd()))
        from config.config import settings


        # Test API connectivity (if configured)
        if hasattr(settings, "binance_api_key") and settings.binance_api_key:
            try:
                import ccxt

                exchange = ccxt.binance(
                    {
                        "apiKey": settings.binance_api_key,
                        "secret": getattr(settings, "binance_api_secret", ""),
                        "testnet": getattr(settings, "binance_testnet", True),
                    },
                )
                exchange.fetch_balance()
            except Exception as e:
                print(f"⚠️ Exchange validation warning: {e}")

        return True

    except Exception:
        return False


def main() -> None:
    """Main setup function."""
    if not create_production_env():
        sys.exit(1)

    if not verify_configuration():
        sys.exit(1)



if __name__ == "__main__":
    main()
