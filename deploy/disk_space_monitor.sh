#!/bin/bash
# Disk Space Monitoring Script
# Alerts when disk usage exceeds thresholds

# Configuration
ROOT_DISK="/"
VOLUME_DISK="/mnt/HC_Volume_104287969"
ALERT_THRESHOLD=85
CRITICAL_THRESHOLD=95
LOG_FILE="/var/log/quantum/disk-monitor.log"

# Get current timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Function to get disk usage percentage
get_disk_usage() {
    df "$1" | awk 'NR==2 {print $5}' | sed 's/%//'
}

# Function to get disk details
get_disk_details() {
    df -h "$1" | awk 'NR==2 {printf "Used: %s / %s (Free: %s)", $3, $2, $4}'
}

# Check root disk
ROOT_USAGE=$(get_disk_usage "$ROOT_DISK")
ROOT_DETAILS=$(get_disk_details "$ROOT_DISK")

# Check volume disk
VOLUME_USAGE=$(get_disk_usage "$VOLUME_DISK")
VOLUME_DETAILS=$(get_disk_details "$VOLUME_DISK")

# Log current status
echo "[$TIMESTAMP] Root: ${ROOT_USAGE}% | Volume: ${VOLUME_USAGE}%" >> "$LOG_FILE"

# Alert if root disk exceeds threshold
if [ "$ROOT_USAGE" -ge "$CRITICAL_THRESHOLD" ]; then
    echo "[$TIMESTAMP] 🚨 CRITICAL: Root disk at ${ROOT_USAGE}% - $ROOT_DETAILS" | tee -a "$LOG_FILE"
    # Send to system log
    logger -t disk-monitor -p user.crit "CRITICAL: Root disk usage ${ROOT_USAGE}%"
elif [ "$ROOT_USAGE" -ge "$ALERT_THRESHOLD" ]; then
    echo "[$TIMESTAMP] ⚠️  WARNING: Root disk at ${ROOT_USAGE}% - $ROOT_DETAILS" | tee -a "$LOG_FILE"
    logger -t disk-monitor -p user.warning "WARNING: Root disk usage ${ROOT_USAGE}%"
fi

# Alert if volume disk exceeds threshold  
if [ "$VOLUME_USAGE" -ge "$CRITICAL_THRESHOLD" ]; then
    echo "[$TIMESTAMP] 🚨 CRITICAL: Volume disk at ${VOLUME_USAGE}% - $VOLUME_DETAILS" | tee -a "$LOG_FILE"
    logger -t disk-monitor -p user.crit "CRITICAL: Volume disk usage ${VOLUME_USAGE}%"
elif [ "$VOLUME_USAGE" -ge "$ALERT_THRESHOLD" ]; then
    echo "[$TIMESTAMP] ⚠️  WARNING: Volume disk at ${VOLUME_USAGE}% - $VOLUME_DETAILS" | tee -a "$LOG_FILE"
    logger -t disk-monitor -p user.warning "WARNING: Volume disk usage ${VOLUME_USAGE}%"
fi

# Check venv size on volume
VENV_SIZE=$(du -sh /opt/quantum/venvs 2>/dev/null | awk '{print $1}')
if [ -n "$VENV_SIZE" ]; then
    echo "[$TIMESTAMP] Venv size: $VENV_SIZE" >> "$LOG_FILE"
fi

# Check log size on volume
LOG_SIZE=$(du -sh /var/log/quantum 2>/dev/null | awk '{print $1}')
if [ -n "$LOG_SIZE" ]; then
    echo "[$TIMESTAMP] Log size: $LOG_SIZE" >> "$LOG_FILE"
fi

# Rotate monitor log if it exceeds 10MB
if [ -f "$LOG_FILE" ]; then
    LOG_FILE_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
    if [ "$LOG_FILE_SIZE" -gt 10485760 ]; then
        mv "$LOG_FILE" "${LOG_FILE}.1"
        echo "[$TIMESTAMP] Disk monitor log rotated" > "$LOG_FILE"
    fi
fi
