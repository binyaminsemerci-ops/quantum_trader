#!/bin/bash
# ============================================================================
# REDIS RESTORE SCRIPT - QUANTUM TRADER
# ============================================================================
# Restores Redis data from compressed backup
# Usage: ./restore-redis.sh <backup_file>
# Example: ./restore-redis.sh /home/qt/backups/redis/redis_backup_20251216_120000.rdb.gz
# ============================================================================

set -euo pipefail

# Configuration
REDIS_CONTAINER="quantum_redis"
BACKUP_DIR="/home/qt/backups/redis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============================================================================
# STEP 1: Validate input
# ============================================================================
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <backup_file>"
    log_info "Available backups:"
    ls -lh "$BACKUP_DIR"/redis_backup_*.rdb.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Check if file is valid gzip
if ! gzip -t "$BACKUP_FILE" 2>/dev/null; then
    log_error "Backup file is corrupted or not a valid gzip file"
    exit 1
fi

log_success "Backup file validated: $BACKUP_FILE"

# ============================================================================
# STEP 2: Confirm restore operation
# ============================================================================
log_warning "⚠️  WARNING: This will REPLACE all current Redis data!"
log_info "Current Redis data will be lost."
read -p "Are you sure you want to continue? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    log_info "Restore cancelled by user"
    exit 0
fi

# ============================================================================
# STEP 3: Check Redis container status
# ============================================================================
log_info "Checking Redis container..."

if ! docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
    log_error "Redis container '${REDIS_CONTAINER}' is not running"
    log_info "Start it with: docker start ${REDIS_CONTAINER}"
    exit 1
fi

# ============================================================================
# STEP 4: Create safety backup of current data
# ============================================================================
log_info "Creating safety backup of current Redis data..."

SAFETY_BACKUP="/tmp/redis_safety_backup_$(date +%Y%m%d_%H%M%S).rdb"
if docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "$SAFETY_BACKUP" 2>/dev/null; then
    log_success "Safety backup created: $SAFETY_BACKUP"
else
    log_warning "Could not create safety backup (Redis may be empty)"
fi

# ============================================================================
# STEP 5: Stop Redis container
# ============================================================================
log_info "Stopping Redis container..."

if ! docker stop "$REDIS_CONTAINER"; then
    log_error "Failed to stop Redis container"
    exit 1
fi

log_success "Redis container stopped"

# ============================================================================
# STEP 6: Decompress and restore backup
# ============================================================================
log_info "Decompressing and restoring backup..."

# Decompress backup to temp file
TEMP_RDB="/tmp/restore_dump_$(date +%Y%m%d_%H%M%S).rdb"
if ! gunzip -c "$BACKUP_FILE" > "$TEMP_RDB"; then
    log_error "Failed to decompress backup"
    docker start "$REDIS_CONTAINER"
    exit 1
fi

# Copy restored dump.rdb to container
if ! docker cp "$TEMP_RDB" "${REDIS_CONTAINER}:/data/dump.rdb"; then
    log_error "Failed to copy backup to container"
    rm -f "$TEMP_RDB"
    docker start "$REDIS_CONTAINER"
    exit 1
fi

# Cleanup temp file
rm -f "$TEMP_RDB"

log_success "Backup restored to container"

# ============================================================================
# STEP 7: Start Redis container
# ============================================================================
log_info "Starting Redis container..."

if ! docker start "$REDIS_CONTAINER"; then
    log_error "Failed to start Redis container"
    exit 1
fi

# Wait for Redis to be ready
sleep 3

# Check Redis health
if docker exec "$REDIS_CONTAINER" redis-cli PING | grep -q "PONG"; then
    log_success "Redis is responding"
else
    log_error "Redis is not responding after restore"
    exit 1
fi

# ============================================================================
# STEP 8: Verify restored data
# ============================================================================
log_info "Verifying restored data..."

# Get number of keys
KEY_COUNT=$(docker exec "$REDIS_CONTAINER" redis-cli DBSIZE | grep -o '[0-9]*')
log_info "Restored keys: ${KEY_COUNT}"

# Get Redis info
REDIS_VERSION=$(docker exec "$REDIS_CONTAINER" redis-cli INFO SERVER | grep redis_version | cut -d: -f2 | tr -d '\r')
USED_MEMORY=$(docker exec "$REDIS_CONTAINER" redis-cli INFO MEMORY | grep used_memory_human | cut -d: -f2 | tr -d '\r')

log_success "Restore completed successfully!"
log_info "Redis version: ${REDIS_VERSION}"
log_info "Memory used: ${USED_MEMORY}"
log_info "Total keys: ${KEY_COUNT}"

if [ -f "$SAFETY_BACKUP" ]; then
    log_info "Safety backup available at: $SAFETY_BACKUP"
    log_info "Delete it when you're sure the restore is correct"
fi

exit 0
