#!/bin/bash
# ============================================================================
# REDIS BACKUP SCRIPT - QUANTUM TRADER
# ============================================================================
# Creates compressed backup of Redis data with retention management
# Usage: ./backup-redis.sh
# Cron: 0 */6 * * * /home/qt/quantum_trader/scripts/backup-redis.sh
# ============================================================================

set -euo pipefail

# Configuration
REDIS_CONTAINER="quantum_redis"
BACKUP_DIR="/home/qt/backups/redis"
RETENTION_DAYS=14
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="redis_backup_${TIMESTAMP}.rdb.gz"
LOG_FILE="/home/qt/backups/redis/backup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# STEP 1: Pre-flight checks
# ============================================================================
log_info "Starting Redis backup process..."

# Check if Redis container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
    log_error "Redis container '${REDIS_CONTAINER}' is not running"
    exit 1
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# ============================================================================
# STEP 2: Trigger Redis BGSAVE
# ============================================================================
log_info "Triggering Redis BGSAVE..."

# Execute BGSAVE command
if ! docker exec "$REDIS_CONTAINER" redis-cli BGSAVE | grep -q "Background saving started"; then
    log_error "Failed to trigger BGSAVE"
    exit 1
fi

# Wait for BGSAVE to complete (check every 2 seconds, max 60 seconds)
WAIT_COUNT=0
MAX_WAIT=30

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    LASTSAVE_STATUS=$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE)
    sleep 2
    NEW_LASTSAVE=$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE)
    
    if [ "$NEW_LASTSAVE" != "$LASTSAVE_STATUS" ]; then
        log_success "BGSAVE completed successfully"
        break
    fi
    
    WAIT_COUNT=$((WAIT_COUNT + 1))
    
    if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
        log_error "BGSAVE timeout after 60 seconds"
        exit 1
    fi
done

# ============================================================================
# STEP 3: Copy and compress dump.rdb
# ============================================================================
log_info "Copying and compressing Redis dump..."

# Copy dump.rdb from container
TEMP_RDB="/tmp/redis_dump_${TIMESTAMP}.rdb"
if ! docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "$TEMP_RDB"; then
    log_error "Failed to copy dump.rdb from container"
    exit 1
fi

# Compress the dump
if ! gzip -c "$TEMP_RDB" > "${BACKUP_DIR}/${BACKUP_FILE}"; then
    log_error "Failed to compress backup"
    rm -f "$TEMP_RDB"
    exit 1
fi

# Cleanup temp file
rm -f "$TEMP_RDB"

# Get backup size
BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
log_success "Backup created: ${BACKUP_FILE} (${BACKUP_SIZE})"

# ============================================================================
# STEP 4: Verify backup integrity
# ============================================================================
log_info "Verifying backup integrity..."

# Check if file is valid gzip
if ! gzip -t "${BACKUP_DIR}/${BACKUP_FILE}"; then
    log_error "Backup file is corrupted"
    exit 1
fi

# Check file size (should be > 1KB)
FILE_SIZE=$(stat -f%z "${BACKUP_DIR}/${BACKUP_FILE}" 2>/dev/null || stat -c%s "${BACKUP_DIR}/${BACKUP_FILE}")
if [ "$FILE_SIZE" -lt 1024 ]; then
    log_error "Backup file is too small (${FILE_SIZE} bytes)"
    exit 1
fi

log_success "Backup integrity verified"

# ============================================================================
# STEP 5: Cleanup old backups (retention policy)
# ============================================================================
log_info "Applying retention policy (${RETENTION_DAYS} days)..."

# Find and delete backups older than RETENTION_DAYS
DELETED_COUNT=$(find "$BACKUP_DIR" -name "redis_backup_*.rdb.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)

if [ "$DELETED_COUNT" -gt 0 ]; then
    log_info "Deleted ${DELETED_COUNT} old backup(s)"
fi

# ============================================================================
# STEP 6: Show backup summary
# ============================================================================
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -name "redis_backup_*.rdb.gz" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)

log_success "Backup completed successfully!"
log_info "Total backups: ${TOTAL_BACKUPS}"
log_info "Total size: ${TOTAL_SIZE}"
log_info "Latest backup: ${BACKUP_FILE} (${BACKUP_SIZE})"

# ============================================================================
# STEP 7: Optional - Copy to off-site location
# ============================================================================
# Uncomment and configure for off-site backups:
# 
# REMOTE_BACKUP_DIR="user@remote-server:/backups/quantum_trader/redis"
# log_info "Copying backup to off-site location..."
# if scp "${BACKUP_DIR}/${BACKUP_FILE}" "$REMOTE_BACKUP_DIR/"; then
#     log_success "Off-site backup completed"
# else
#     log_error "Off-site backup failed (non-critical)"
# fi

exit 0
