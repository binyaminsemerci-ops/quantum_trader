#!/bin/bash
# Test Redis Backup & Restore
set -e

CONTAINER="quantum_redis"

echo "=== Redis Backup & Restore Test ==="

# Step 1: Write test data
echo "[1/5] Writing test data to Redis..."
docker exec $CONTAINER redis-cli SET test_backup_key "backup_test_$(date +%s)"
TEST_VALUE=$(docker exec $CONTAINER redis-cli GET test_backup_key)
echo "      Wrote: $TEST_VALUE"

# Step 2: Run backup
echo "[2/5] Running backup..."
/home/qt/quantum_trader/scripts/backup-redis.sh

# Step 3: Modify data
echo "[3/5] Modifying data..."
docker exec $CONTAINER redis-cli SET test_backup_key "MODIFIED"
MODIFIED_VALUE=$(docker exec $CONTAINER redis-cli GET test_backup_key)
echo "      Modified to: $MODIFIED_VALUE"

# Step 4: Find latest backup
LATEST_BACKUP=$(ls -t /home/qt/backups/redis/redis_*.rdb.gz | head -1)
echo "[4/5] Restoring from: $LATEST_BACKUP"

# Step 5: Restore
/home/qt/quantum_trader/scripts/restore-redis.sh "$LATEST_BACKUP"

# Verify
RESTORED_VALUE=$(docker exec $CONTAINER redis-cli GET test_backup_key)
echo "[5/5] Restored value: $RESTORED_VALUE"

if [ "$TEST_VALUE" = "$RESTORED_VALUE" ]; then
    echo "✅ BACKUP & RESTORE TEST PASSED"
    exit 0
else
    echo "❌ TEST FAILED: Values don't match"
    exit 1
fi
