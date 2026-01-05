#!/bin/bash
# Quantum Trader - Disk Space Guard
# Monitors disk usage and reports when threshold exceeded

THRESHOLD=80
LOG_FILE="/var/log/disk-monitor.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Check root disk
ROOT_USE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

echo "[$DATE] Disk check: Root at ${ROOT_USE}%" >> $LOG_FILE

if [ "$ROOT_USE" -gt "$THRESHOLD" ]; then
    echo "[$DATE] ⚠️ WARNING: Root disk at ${ROOT_USE}%" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Top directories:" | tee -a $LOG_FILE
    du -xhd1 / 2>/dev/null | sort -h | tail -10 | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Docker usage:" | tee -a $LOG_FILE
    docker system df | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Containerd snapshots:" | tee -a $LOG_FILE
    du -sh /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs 2>/dev/null | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
fi
