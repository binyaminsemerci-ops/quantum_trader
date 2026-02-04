#!/bin/bash
# Secret Scrubbing Script - Remove hardcoded API keys

echo "🧼 Scrubbing hardcoded API keys from repository..."

# Pattern to replace actual keys with placeholders
OLD_KEY="e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
OLD_SECRET="ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"

# Safe placeholders
NEW_KEY="your_binance_testnet_api_key_here"
NEW_SECRET="your_binance_testnet_api_secret_here"

# Files to scrub
FILES=(
    "DASHBOARD_PROBLEMS_ANALYSIS.md"
    "SYSTEM_LIVE_TRADING_ACTIVATED.md"
    "backend/main.py"
    "backend/services/execution/execution.py"
    "backend/services/monitoring/position_monitor.py"
    "check_binance_testnet.py"
    "check_testnet_pos.py"
    "close_testnet.py"
    "config/config.py"
    "docker-compose.vps.yml.archive"
    "ops/proof_bridge_patch.sh"
    "ops/test_testnet_balance.py"
    "patch_config.py"
    "scripts/consumer_entrypoint.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Scrubbing: $file"
        sed -i "s/$OLD_KEY/$NEW_KEY/g" "$file"
        sed -i "s/$OLD_SECRET/$NEW_SECRET/g" "$file"
    else
        echo "  ⚠️  Not found: $file"
    fi
done

echo ""
echo "✅ Scrubbing complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit: git add -A && git commit -m 'security: scrub hardcoded API keys from repo'"
echo "3. Push: git push origin main"
echo "4. ROTATE KEYS on Binance Testnet (required!)"
echo "5. Update VPS config: /etc/quantum/*.env with new keys"
