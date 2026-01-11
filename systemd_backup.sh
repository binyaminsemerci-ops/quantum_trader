#!/bin/bash
set -euo pipefail

TS=$(date +%F_%H-%M-%S)
echo "TS=$TS"

B=/root/backup_quantum_${TS}
mkdir -p "$B"
echo "Backup dir: $B"
echo ""

echo "=== 3.1 CONFIG + SCRIPTS ==="
cp -a /home/qt/quantum_trader/.env "$B"/ 2>/dev/null || echo "No .env found"
cp -a /root/*.sh "$B"/ 2>/dev/null || echo "No shell scripts"
echo "Config backup done"
echo ""

echo "=== 3.2 LEARNING DIRS ==="
for d in models data runtime backups; do
  P="/home/qt/quantum_trader/$d"
  if [ -d "$P" ]; then
    echo "Backing up $P"
    tar -czf "$B/${d}_${TS}.tgz" -C /home/qt/quantum_trader "$d"
    ls -lh "$B/${d}_${TS}.tgz"
  else
    echo "Skip missing dir: $P"
  fi
done
echo ""

echo "=== 3.3 REDIS RDB ==="
if [ -f /var/lib/redis/dump.rdb ]; then
  cp -a /var/lib/redis/dump.rdb "$B/redis_dump.rdb"
  ls -lh "$B/redis_dump.rdb"
else
  echo "No Redis dump found"
fi
echo ""

echo "=== 3.4 SYSTEMD UNITS ==="
cp -a /etc/systemd/system/quantum*.service "$B/" 2>/dev/null || echo "No unit files"
cp -a /etc/systemd/system/quantum*.target "$B/" 2>/dev/null || echo "No target files"
echo ""

echo "=== 3.5 FINAL TAR ==="
tar -czf /root/backup_quantum_${TS}.tgz -C /root "backup_quantum_${TS}"
ls -lh /root/backup_quantum_${TS}.tgz
echo ""
echo "BACKUP READY: /root/backup_quantum_${TS}.tgz"
echo "Timestamp: $TS"
