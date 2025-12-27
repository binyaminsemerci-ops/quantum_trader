#!/bin/bash
# Postgres Automated Backup Script
# SPRINT 3 - Module B: Postgres HA
#
# Backs up Postgres database to S3/Azure Blob with retention policy

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

# Backup settings
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="quantum_trader_${TIMESTAMP}.sql.gz"

# Cloud storage (choose one)
S3_BUCKET="${S3_BUCKET:-}"  # e.g., s3://quantum-trader-backups
AZURE_CONTAINER="${AZURE_CONTAINER:-}"  # e.g., quantum-backups

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

check_dependencies() {
    command -v pg_dump >/dev/null 2>&1 || {
        log "ERROR: pg_dump not found. Install postgresql-client."
        exit 1
    }
}

create_backup() {
    log "Starting backup: ${BACKUP_FILE}"
    
    mkdir -p "${BACKUP_DIR}"
    
    # Perform pg_dump
    PGPASSWORD="${PGPASSWORD}" pg_dump \
        -h "${PGHOST}" \
        -p "${PGPORT}" \
        -U "${PGUSER}" \
        -d "${PGDATABASE}" \
        --format=plain \
        --no-owner \
        --no-acl \
        | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"
    
    if [ $? -eq 0 ]; then
        log "Backup created successfully: ${BACKUP_DIR}/${BACKUP_FILE}"
        
        # Get file size
        SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
        log "Backup size: ${SIZE}"
    else
        log "ERROR: Backup failed"
        exit 1
    fi
}

upload_to_cloud() {
    if [ -n "${S3_BUCKET}" ]; then
        log "Uploading to S3: ${S3_BUCKET}"
        aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}" "${S3_BUCKET}/${BACKUP_FILE}"
        
        if [ $? -eq 0 ]; then
            log "Upload to S3 successful"
        else
            log "ERROR: S3 upload failed"
            exit 1
        fi
    
    elif [ -n "${AZURE_CONTAINER}" ]; then
        log "Uploading to Azure Blob: ${AZURE_CONTAINER}"
        az storage blob upload \
            --container-name "${AZURE_CONTAINER}" \
            --name "${BACKUP_FILE}" \
            --file "${BACKUP_DIR}/${BACKUP_FILE}"
        
        if [ $? -eq 0 ]; then
            log "Upload to Azure successful"
        else
            log "ERROR: Azure upload failed"
            exit 1
        fi
    
    else
        log "No cloud storage configured, backup kept locally only"
    fi
}

cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days"
    
    # Local cleanup
    find "${BACKUP_DIR}" -name "quantum_trader_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
    
    # S3 cleanup
    if [ -n "${S3_BUCKET}" ]; then
        aws s3 ls "${S3_BUCKET}/" | while read -r line; do
            FILE=$(echo "$line" | awk '{print $4}')
            if [[ "$FILE" =~ quantum_trader_.*\.sql\.gz ]]; then
                FILE_DATE=$(echo "$FILE" | grep -oP '\d{8}' | head -1)
                CUTOFF_DATE=$(date -d "${RETENTION_DAYS} days ago" +%Y%m%d)
                
                if [ "$FILE_DATE" -lt "$CUTOFF_DATE" ]; then
                    log "Deleting old S3 backup: ${FILE}"
                    aws s3 rm "${S3_BUCKET}/${FILE}"
                fi
            fi
        done
    fi
    
    # Azure cleanup
    if [ -n "${AZURE_CONTAINER}" ]; then
        az storage blob list \
            --container-name "${AZURE_CONTAINER}" \
            --query "[?contains(name, 'quantum_trader_')].name" \
            --output tsv | while read -r FILE; do
            
            FILE_DATE=$(echo "$FILE" | grep -oP '\d{8}' | head -1)
            CUTOFF_DATE=$(date -d "${RETENTION_DAYS} days ago" +%Y%m%d)
            
            if [ "$FILE_DATE" -lt "$CUTOFF_DATE" ]; then
                log "Deleting old Azure backup: ${FILE}"
                az storage blob delete \
                    --container-name "${AZURE_CONTAINER}" \
                    --name "${FILE}"
            fi
        done
    fi
}

# ============================================================================
# MAIN
# ============================================================================

log "========================================"
log "Quantum Trader Postgres Backup"
log "========================================"

check_dependencies
create_backup
upload_to_cloud
cleanup_old_backups

log "Backup complete!"

# ============================================================================
# CRON SCHEDULE (add to crontab)
# ============================================================================
# Daily backup at 02:00 UTC:
# 0 2 * * * /app/infra/postgres/backup.sh >> /var/log/postgres_backup.log 2>&1

# ============================================================================
# KUBERNETES CRONJOB (alternative)
# ============================================================================
# apiVersion: batch/v1
# kind: CronJob
# metadata:
#   name: postgres-backup
# spec:
#   schedule: "0 2 * * *"
#   jobTemplate:
#     spec:
#       template:
#         spec:
#           containers:
#           - name: backup
#             image: postgres:15-alpine
#             command: ["/bin/sh", "/scripts/backup.sh"]
#             envFrom:
#             - secretRef:
#                 name: postgres-credentials
#             volumeMounts:
#             - name: scripts
#               mountPath: /scripts
#           restartPolicy: OnFailure
#           volumes:
#           - name: scripts
#             configMap:
#               name: backup-scripts
