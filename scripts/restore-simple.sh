#!/bin/bash
# Redis Restore - Recreate container method
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup.rdb.gz>"
    ls -lh /home/qt/backups/redis/*.rdb.gz 2>/dev/null
    exit 1
fi

BACKUP="$1"
echo "Restoring from: $BACKUP"

# Stop and remove container
docker stop quantum_redis
docker rm quantum_redis

# Extract backup to volume
VOLUME_PATH=$(docker volume inspect quantum_trader_redis_data --format '{{.Mountpoint}}')
sudo gunzip -c "$BACKUP" > "$VOLUME_PATH/dump.rdb"

# Recreate container
cd ~/quantum_trader
docker compose -f docker-compose.vps.yml up -d redis

sleep 3
docker exec quantum_redis redis-cli PING
echo "✅ Restore complete"
