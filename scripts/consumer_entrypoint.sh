#!/bin/bash
# Entrypoint script to copy updated code before running
echo "🔄 Copying updated code from mounted volume..."

# Copy trade_intent_runner.py to proper location
if [ -f "/mnt/code/trade_intent_runner.py" ]; then
    cp /mnt/code/trade_intent_runner.py /app/trade_intent_runner.py
    echo "✅ Copied trade_intent_runner.py"
fi

# Set API keys explicitly (Docker environment variables don't propagate)
export BINANCE_API_KEY="e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
export BINANCE_API_SECRET="ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"
echo "🔑 API keys loaded: KEY=${#BINANCE_API_KEY} chars, SECRET=${#BINANCE_API_SECRET} chars"

# Copy backend directory if needed (already mounted as ro)
# No need to copy since it's mounted directly

echo "🚀 Starting consumer..."
python /app/trade_intent_runner.py

