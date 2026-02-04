#!/bin/bash
# Quantum Trader - Backup Restore Test Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups/postgres"
TEST_DB="quantum_trader_test"

echo "================================================"
echo "   Quantum Trader Backup Restore Test"
echo "================================================"
echo ""

# Find most recent backup
echo "=== Finding Most Recent Backup ==="
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.sql.gz 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No backups found in $BACKUP_DIR"
    exit 1
fi

BACKUP_SIZE=$(du -h "$LATEST_BACKUP" | cut -f1)
echo "✅ Found backup: $(basename "$LATEST_BACKUP") ($BACKUP_SIZE)"
echo ""

# Confirm test
echo "⚠️  This test will:"
echo "   1. Create a test database: $TEST_DB"
echo "   2. Restore backup into test database"
echo "   3. Verify data integrity"
echo "   4. Clean up test database"
echo ""
echo "   (Production database 'quantum_trader' will NOT be affected)"
echo ""
read -p "Continue? (y/N): " confirm

if [ "$confirm" != "y" ]; then
    echo "Test cancelled"
    exit 0
fi

# Create test database
echo ""
echo "=== Creating Test Database ==="
docker exec quantum_postgres psql -U quantum -c "DROP DATABASE IF EXISTS $TEST_DB;" 2>/dev/null || true
docker exec quantum_postgres psql -U quantum -c "CREATE DATABASE $TEST_DB;"
echo "✅ Test database created: $TEST_DB"

# Restore backup
echo ""
echo "=== Restoring Backup ==="
echo "This may take a moment..."

if gunzip < "$LATEST_BACKUP" | docker exec -i quantum_postgres psql -U quantum -d "$TEST_DB" > /tmp/restore.log 2>&1; then
    echo "✅ Backup restored successfully"
else
    echo "❌ Restore failed! Check /tmp/restore.log"
    cat /tmp/restore.log
    exit 1
fi

# Verify data
echo ""
echo "=== Verifying Data Integrity ==="

# Check tables exist
echo "Checking tables..."
TABLE_COUNT=$(docker exec quantum_postgres psql -U quantum -d "$TEST_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
echo "  Tables found: $(echo $TABLE_COUNT | xargs)"

# Check if any tables have data
TOTAL_ROWS=0
TABLES=$(docker exec quantum_postgres psql -U quantum -d "$TEST_DB" -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

if [ -n "$TABLES" ]; then
    echo ""
    echo "Table statistics:"
    while IFS= read -r table; do
        table=$(echo "$table" | xargs)
        if [ -n "$table" ]; then
            row_count=$(docker exec quantum_postgres psql -U quantum -d "$TEST_DB" -t -c "SELECT COUNT(*) FROM \"$table\";" 2>/dev/null || echo "0")
            row_count=$(echo "$row_count" | xargs)
            echo "  $table: $row_count rows"
            TOTAL_ROWS=$((TOTAL_ROWS + row_count))
        fi
    done <<< "$TABLES"
else
    echo "  ⚠️  No tables found (this is normal for a new database)"
fi

# Database size
echo ""
DB_SIZE=$(docker exec quantum_postgres psql -U quantum -d "$TEST_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$TEST_DB'));")
echo "Database size: $(echo $DB_SIZE | xargs)"

# Clean up
echo ""
echo "=== Cleaning Up ==="
docker exec quantum_postgres psql -U quantum -c "DROP DATABASE $TEST_DB;"
echo "✅ Test database removed"

echo ""
echo "================================================"
echo "   ✅ Backup Restore Test Complete!"
echo "================================================"
echo ""
echo "Results:"
echo "  ✅ Backup file: $(basename "$LATEST_BACKUP")"
echo "  ✅ Backup size: $BACKUP_SIZE"
echo "  ✅ Restore: SUCCESS"
echo "  ✅ Tables: $(echo $TABLE_COUNT | xargs)"
echo "  ✅ Total rows: $TOTAL_ROWS"
echo "  ✅ Database size: $(echo $DB_SIZE | xargs)"
echo ""

if [ "$TOTAL_ROWS" -eq 0 ]; then
    echo "⚠️  Note: Database is currently empty (no data in tables)"
    echo "   This is normal if the system is newly deployed"
    echo "   Backups will contain data once the system is in use"
else
    echo "✅ Backup contains data and is fully functional"
fi

echo ""
echo "Restore command for disaster recovery:"
echo "  cat $LATEST_BACKUP | gunzip | docker exec -i quantum_postgres psql -U quantum quantum_trader"
echo ""
