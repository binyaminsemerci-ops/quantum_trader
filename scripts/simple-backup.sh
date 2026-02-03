#!/bin/bash
# Simple Redis Backup - No complex wait logic
BACKUP_DIR="/home/qt/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"
docker exec quantum_redis redis-cli BGSAVE
sleep 5
docker cp quantum_redis:/data/dump.rdb "$BACKUP_DIR/redis_${DATE}.rdb"
gzip "$BACKUP_DIR/redis_${DATE}.rdb"
find "$BACKUP_DIR" -name "redis_*.rdb.gz" -mtime +14 -delete
echo "[$(date)] Backup: redis_${DATE}.rdb.gz ($(du -h $BACKUP_DIR/redis_${DATE}.rdb.gz | cut -f1))"
