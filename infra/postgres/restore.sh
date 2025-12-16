#!/bin/bash
# Postgres Restore Script
# SPRINT 3 - Module B: Postgres HA
#
# Restores Postgres database from backup file

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection
PGHOST="${PGHOST:-postgres}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-quantum_trader}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD}"

# Backup file
BACKUP_FILE="$1"

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

check_dependencies() {
    command -v psql >/dev/null 2>&1 || {
        log "ERROR: psql not found. Install postgresql-client."
        exit 1
    }
    
    command -v gunzip >/dev/null 2>&1 || {
        log "ERROR: gunzip not found."
        exit 1
    }
}

validate_backup_file() {
    if [ -z "${BACKUP_FILE}" ]; then
        log "ERROR: No backup file specified"
        log "Usage: $0 <backup_file.sql.gz>"
        log "Example: $0 /backups/quantum_trader_20231204_020000.sql.gz"
        exit 1
    fi
    
    if [ ! -f "${BACKUP_FILE}" ]; then
        log "ERROR: Backup file not found: ${BACKUP_FILE}"
        exit 1
    fi
    
    log "Backup file: ${BACKUP_FILE}"
    SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    log "File size: ${SIZE}"
}

confirm_restore() {
    log "=========================================="
    log "WARNING: This will DROP and RECREATE the database!"
    log "Database: ${PGDATABASE}"
    log "Host: ${PGHOST}:${PGPORT}"
    log "=========================================="
    
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        log "Restore cancelled"
        exit 0
    fi
}

restore_database() {
    log "Starting database restore..."
    
    # Drop existing connections
    log "Terminating existing connections..."
    PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d postgres \
        -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${PGDATABASE}' AND pid <> pg_backend_pid();" \
        2>/dev/null || true
    
    # Drop database
    log "Dropping database: ${PGDATABASE}"
    PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${PGDATABASE};"
    
    # Create database
    log "Creating database: ${PGDATABASE}"
    PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d postgres \
        -c "CREATE DATABASE ${PGDATABASE};"
    
    # Restore from backup
    log "Restoring from backup..."
    gunzip -c "${BACKUP_FILE}" | PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d "${PGDATABASE}"
    
    if [ $? -eq 0 ]; then
        log "Restore completed successfully!"
    else
        log "ERROR: Restore failed"
        exit 1
    fi
}

verify_restore() {
    log "Verifying restore..."
    
    # Count tables
    TABLE_COUNT=$(PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d "${PGDATABASE}" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    
    log "Tables restored: ${TABLE_COUNT}"
    
    # List tables
    log "Tables in database:"
    PGPASSWORD="${PGPASSWORD}" psql \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d "${PGDATABASE}" \
        -c "\dt"
}

# ============================================================================
# MAIN
# ============================================================================

log "========================================"
log "Quantum Trader Postgres Restore"
log "========================================"

check_dependencies
validate_backup_file
confirm_restore
restore_database
verify_restore

log "Restore process complete!"

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
# Restore from local file:
# ./restore.sh /backups/quantum_trader_20231204_020000.sql.gz

# Restore from S3:
# aws s3 cp s3://quantum-trader-backups/quantum_trader_20231204_020000.sql.gz /tmp/
# ./restore.sh /tmp/quantum_trader_20231204_020000.sql.gz

# Restore from Azure:
# az storage blob download \
#   --container-name quantum-backups \
#   --name quantum_trader_20231204_020000.sql.gz \
#   --file /tmp/quantum_trader_20231204_020000.sql.gz
# ./restore.sh /tmp/quantum_trader_20231204_020000.sql.gz
