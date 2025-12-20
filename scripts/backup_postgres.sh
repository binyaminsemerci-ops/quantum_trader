#!/bin/bash
# Quantum Trader - PostgreSQL Backup Script
# Runs daily backups with 7-day retention

set -e

BACKUP_DIR="/home/qt/quantum_trader/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="quantum_trader_${DATE}.sql.gz"
CONTAINER_NAME="quantum_postgres"
DB_NAME="quantum_trader"
DB_USER="quantum"
RETENTION_DAYS=7

echo "=== PostgreSQL Backup Started: $(date) ==="

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Perform backup
echo "Backing up database: $DB_NAME..."
docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_DIR/$BACKUP_FILE"

# Check if backup was successful
if [ -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE" | cut -f1)
    echo "✅ Backup successful: $BACKUP_FILE ($BACKUP_SIZE)"
else
    echo "❌ Backup failed!"
    exit 1
fi

# Remove old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "quantum_trader_*.sql.gz" -mtime +$RETENTION_DAYS -delete
REMAINING=$(ls -1 "$BACKUP_DIR"/quantum_trader_*.sql.gz 2>/dev/null | wc -l)
echo "Backups remaining: $REMAINING"

# List recent backups
echo ""
echo "Recent backups:"
ls -lh "$BACKUP_DIR"/quantum_trader_*.sql.gz | tail -5

echo "=== PostgreSQL Backup Completed: $(date) ==="
