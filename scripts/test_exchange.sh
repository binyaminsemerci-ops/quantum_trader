#!/bin/bash
# Test Exchange Connection - Loads keys from /etc/quantum/governor.env
# Usage: ./test_exchange.sh

set -e

echo "🔐 Testing Binance Testnet Exchange Connection"
echo "==============================================="
echo ""

# Load keys from config
if [ -f /etc/quantum/governor.env ]; then
    export $(grep "^BINANCE_TESTNET_API_KEY=" /etc/quantum/governor.env | xargs)
    export $(grep "^BINANCE_TESTNET_API_SECRET=" /etc/quantum/governor.env | xargs)
fi

# Verify keys loaded (lengths only, no actual keys)
if [ -n "$BINANCE_TESTNET_API_KEY" ]; then
    echo "✅ API Key loaded: ${#BINANCE_TESTNET_API_KEY} chars"
else
    echo "❌ API Key not found"
    exit 1
fi

if [ -n "$BINANCE_TESTNET_API_SECRET" ]; then
    echo "✅ API Secret loaded: ${#BINANCE_TESTNET_API_SECRET} chars"
else
    echo "❌ API Secret not found"
    exit 1
fi

echo ""

# Run exchange dump
cd /home/qt/quantum_trader
python3 scripts/dump_exchange_positions.py
