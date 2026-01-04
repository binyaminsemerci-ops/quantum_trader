#!/usr/bin/env python3
"""
Patch config.py to use hardcoded API keys instead of environment variables.
This is a temporary workaround until we can rebuild the Docker image.
"""
import os

config_file = "/app/config/config.py"

# Read the file
with open(config_file, 'r') as f:
    content = f.read()

# Replace the API key/secret lines with hardcoded values
content = content.replace(
    'binance_api_key=dashboard.get("api_key") or os.environ.get("BINANCE_API_KEY"),',
    'binance_api_key=dashboard.get("api_key") or os.environ.get("BINANCE_API_KEY") or "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD",'
)
content = content.replace(
    'binance_api_secret=dashboard.get("api_secret") or os.environ.get("BINANCE_API_SECRET"),',
    'binance_api_secret=dashboard.get("api_secret") or os.environ.get("BINANCE_API_SECRET") or "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja",'
)

# Write it back
with open(config_file, 'w') as f:
    f.write(content)

print("✅ Patched config.py with hardcoded API keys")
print("⚠️ WARNING: This is a temporary fix - rebuild the image to make it permanent!")
