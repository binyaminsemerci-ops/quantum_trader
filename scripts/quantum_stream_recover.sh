#!/bin/bash
set -euo pipefail

LOGFILE=/var/log/quantum/stream_recover.log
STREAM_AR=quantum:stream:apply.result
STREAM_TI=quantum:stream:trade.intent
GROUP=quantum:group:execution:trade.intent

source /etc/quantum/core_gates.env 2>/dev/null || {
    ZOMBIE_IDLE_MS=3600000
    STALE_IDLE_MS=60000
}

log() { echo "$(date -Iseconds) | $*" | tee -a $LOGFILE; }

# Abort if LIVE mode
if grep -q 'TRADING_MODE=LIVE' /etc/quantum/testnet.env 2>/dev/null; then
    log "ABORT: LIVE mode - recovery disabled"
    exit 0
fi

log "=== ZOMBIE + STALE RECOVERY START ==="

# Ensure groups exist
redis-cli XGROUP CREATE $STREAM_AR $GROUP 0 MKSTREAM 2>/dev/null || true
redis-cli XGROUP CREATE $STREAM_TI $GROUP 0 MKSTREAM 2>/dev/null || true

# XAUTOCLAIM stale messages
for STREAM in $STREAM_AR $STREAM_TI; do
    CLAIMED=$(redis-cli XAUTOCLAIM $STREAM $GROUP quantum-execution ${STALE_IDLE_MS:-60000} 0 COUNT 100 2>/dev/null | wc -l)
    log "XAUTOCLAIM $STREAM: $CLAIMED items"
done

# Delete zombie consumers using Python (more reliable)
python3 /tmp/clean_zombies.py 2>/dev/null || log "Python cleanup skipped"

log "=== RECOVERY COMPLETE ==="
