#!/bin/bash
# Run all setup tasks

cd /home/qt/quantum_trader

echo "🚀 Running All Setup Tasks..."
echo ""

# 1. Test backup restore first (no user input needed)
echo "📦 Task 1/3: Testing Backup Restore"
bash scripts/test_backup_restore.sh
echo ""

# 2. Alertmanager (needs user input)
echo "🔔 Task 2/3: Configure Alertmanager"
bash scripts/setup_alertmanager.sh
echo ""

# 3. Let's Encrypt (needs user input)
echo "🔒 Task 3/3: Setup Let's Encrypt SSL"
bash scripts/setup_letsencrypt.sh
echo ""

echo "✅ All tasks completed!"
echo ""
echo "Status check:"
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep quantum
