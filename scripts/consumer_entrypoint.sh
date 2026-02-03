#!/bin/bash
# Entrypoint script to copy updated code before running
echo "🔄 Copying updated code from mounted volume..."

# Copy trade_intent_runner.py to proper location
if [ -f "/mnt/code/trade_intent_runner.py" ]; then
    cp /mnt/code/trade_intent_runner.py /app/trade_intent_runner.py
    echo "✅ Copied trade_intent_runner.py"
fi

# Copy backend directory if needed (already mounted as ro)
# No need to copy since it's mounted directly

echo "🚀 Starting consumer with API keys..."
# Use exec to replace bash process with python, ensuring environment is inherited
exec env \
  BINANCE_API_KEY="your_binance_testnet_api_key_here" \
  BINANCE_API_SECRET="your_binance_testnet_api_secret_here" \
  python /app/trade_intent_runner.py

