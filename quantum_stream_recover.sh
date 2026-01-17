#!/bin/bash
set -euo pipefail

# Quantum Stream Recovery - Auto-cleanup zombie consumers and recover stale pending
# Run before execution service starts (ExecStartPre)

LOGFILE="/var/log/quantum/stream_recover.log"
STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"

log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOGFILE"
}

# Check TESTNET mode
READONLY=0
if [ -f /etc/quantum/testnet.env ]; then
    log "✅ TESTNET mode detected - full recovery enabled"
else
    log "⚠️ LIVE mode detected - running read-only checks only"
    READONLY=1
fi

# Ensure group exists
redis-cli XGROUP CREATE "$STREAM" "$GROUP" 0 MKSTREAM 2>/dev/null || log "Group already exists"

# Claim stale pending (idle > 60 seconds)
log "Claiming stale pending messages (idle > 60s)..."
CLAIMED=$(redis-cli XAUTOCLAIM "$STREAM" "$GROUP" "recover-$$" 60000 0 COUNT 200 2>/dev/null | grep -c "^1)" || echo 0)
log "Claimed $CLAIMED stale messages for re-processing"

# Safe zombie cleanup (idle > 1 hour AND pending==0)
if [ "${READONLY:-0}" == "0" ]; then
    log "Checking for zombie consumers (idle > 1h, pending=0)..."
    ZOMBIES=0
    redis-cli XINFO CONSUMERS "$STREAM" "$GROUP" | awk '
        /^name/ { name=$2 }
        /^pending/ { pending=$2 }
        /^idle/ { 
            idle=$2
            if (idle > 3600000 && pending == 0) {
                print name
            }
        }
    ' | while read zombie; do
        if [ -n "$zombie" ]; then
            log "Deleting zombie consumer: $zombie (idle > 1h, pending=0)"
            redis-cli XGROUP DELCONSUMER "$STREAM" "$GROUP" "$zombie"
            ZOMBIES=$((ZOMBIES + 1))
        fi
    done
    log "Deleted $ZOMBIES zombie consumers"
else
    log "LIVE mode - skipping zombie cleanup"
fi

log "✅ Stream recovery complete"
exit 0
