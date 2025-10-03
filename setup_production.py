#!/usr/bin/env python3
"""
Production API Configuration Setup Script

This script helps configure live API keys and enables production features
for the Quantum Trader system.
"""

import sys
from pathlib import Path
import shutil


def create_production_env():
    """Create a production-ready .env file with live API configuration."""

    print("🚀 Quantum Trader Production Configuration Setup")
    print("=" * 50)

    # Check if .env already exists
    env_path = Path(".env")
    if env_path.exists():
        backup_path = Path(".env.backup")
        shutil.copy(env_path, backup_path)
        print(f"✅ Backed up existing .env to {backup_path}")

    # Read .env.example as template
    example_path = Path(".env.example")
    if not example_path.exists():
        print("❌ .env.example not found!")
        return False

    with open(example_path, "r") as f:
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

    print("\n📋 Production Configuration:")
    for key, value in production_config.items():
        print(f"  {key}={value}")

    # Get API keys from user
    print("\n🔑 API Key Configuration:")
    print("Enter your API credentials (press Enter to skip):")

    api_keys = {}

    # Binance API
    print("\n--- Binance API ---")
    binance_key = input("Binance API Key: ").strip()
    if binance_key:
        binance_secret = input("Binance API Secret: ").strip()
        if binance_secret:
            api_keys["BINANCE_API_KEY"] = binance_key
            api_keys["BINANCE_API_SECRET"] = binance_secret
            print("✅ Binance API configured")

    # Twitter/X API
    print("\n--- Twitter/X API (optional for sentiment) ---")
    x_bearer = input("X Bearer Token: ").strip()
    if x_bearer:
        api_keys["X_BEARER_TOKEN"] = x_bearer
        print("✅ Twitter/X API configured")

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

    print("\n✅ Production .env created!")
    print(f"📁 Configuration saved to: {env_path.absolute()}")

    return True


def verify_configuration():
    """Verify the production configuration is working."""
    print("\n🔍 Verifying Configuration...")

    try:
        # Import and test configuration
        sys.path.insert(0, str(Path.cwd()))
        from config.config import settings

        print(f"✅ Live market data: {settings.enable_live_market_data}")
        print(f"✅ Default symbols: {len(settings.default_symbols)} symbols")
        print(f"✅ Starting equity: ${settings.starting_equity:,.2f}")

        # Test API connectivity (if configured)
        if hasattr(settings, "binance_api_key") and settings.binance_api_key:
            print("🔄 Testing Binance API connection...")
            try:
                import ccxt

                exchange = ccxt.binance(
                    {
                        "apiKey": settings.binance_api_key,
                        "secret": getattr(settings, "binance_api_secret", ""),
                        "testnet": getattr(settings, "binance_testnet", True),
                    }
                )
                exchange.fetch_balance()
                print("✅ Binance API connection successful")
                print(
                    f"   Account type: {'Testnet' if settings.binance_testnet else 'Live'}"
                )
            except Exception as e:
                print(f"⚠️  Binance API test failed: {e}")

        print("\n✅ Configuration verification complete!")
        return True

    except Exception as e:
        print(f"❌ Configuration verification failed: {e}")
        return False


def main():
    """Main setup function."""
    if not create_production_env():
        sys.exit(1)

    if not verify_configuration():
        print("\n⚠️  Some configuration issues detected. Check the output above.")
        sys.exit(1)

    print("\n🎉 Production setup complete!")
    print("\nNext steps:")
    print("1. Review your .env file settings")
    print("2. Run: python production_risk_manager.py --test")
    print("3. Run: python production_monitor.py --start")
    print("4. Start trading with: python main_train_and_backtest.py train --production")


if __name__ == "__main__":
    main()
