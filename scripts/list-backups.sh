#!/bin/bash
# ============================================================================
# LIST REDIS BACKUPS - QUANTUM TRADER
# ============================================================================
# Shows all available Redis backups with details
# Usage: ./list-backups.sh
# ============================================================================

BACKUP_DIR="/home/qt/backups/redis"

echo "============================================"
echo "REDIS BACKUPS - QUANTUM TRADER"
echo "============================================"
echo ""

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

BACKUPS=$(find "$BACKUP_DIR" -name "redis_backup_*.rdb.gz" -type f | sort -r)

if [ -z "$BACKUPS" ]; then
    echo "❌ No backups found in $BACKUP_DIR"
    exit 0
fi

COUNT=0
echo "Available backups:"
echo ""
echo "Date/Time          | Size    | Age      | File"
echo "-------------------|---------|----------|------------------------------"

while IFS= read -r backup; do
    COUNT=$((COUNT + 1))
    
    # Extract date from filename (format: redis_backup_YYYYMMDD_HHMMSS.rdb.gz)
    FILENAME=$(basename "$backup")
    DATE_STR=$(echo "$FILENAME" | grep -oP '\d{8}_\d{6}')
    
    if [ -n "$DATE_STR" ]; then
        DATE_PART=$(echo "$DATE_STR" | cut -d_ -f1)
        TIME_PART=$(echo "$DATE_STR" | cut -d_ -f2)
        FORMATTED_DATE="${DATE_PART:0:4}-${DATE_PART:4:2}-${DATE_PART:6:2} ${TIME_PART:0:2}:${TIME_PART:2:2}:${TIME_PART:4:2}"
    else
        FORMATTED_DATE="Unknown"
    fi
    
    # Get file size (human readable)
    SIZE=$(du -h "$backup" | cut -f1)
    
    # Get file age
    if [ "$(uname)" = "Darwin" ]; then
        AGE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$backup")
    else
        AGE=$(stat -c "%y" "$backup" | cut -d. -f1)
    fi
    
    # Calculate days old
    if [ "$(uname)" = "Darwin" ]; then
        FILE_TIME=$(stat -f "%m" "$backup")
    else
        FILE_TIME=$(stat -c "%Y" "$backup")
    fi
    CURRENT_TIME=$(date +%s)
    DAYS_OLD=$(( (CURRENT_TIME - FILE_TIME) / 86400 ))
    
    if [ $DAYS_OLD -eq 0 ]; then
        AGE_STR="Today"
    elif [ $DAYS_OLD -eq 1 ]; then
        AGE_STR="1 day"
    else
        AGE_STR="${DAYS_OLD} days"
    fi
    
    printf "%-18s | %-7s | %-8s | %s\n" "$FORMATTED_DATE" "$SIZE" "$AGE_STR" "$FILENAME"
done <<< "$BACKUPS"

echo ""
echo "Total backups: $COUNT"
echo "Backup directory: $BACKUP_DIR"

# Show total size
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "Total size: $TOTAL_SIZE"

echo ""
echo "To restore a backup, use:"
echo "./restore-redis.sh $BACKUP_DIR/<backup_file>"
