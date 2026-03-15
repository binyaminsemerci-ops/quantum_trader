#!/bin/bash
set -euo pipefail

LOGFILE=/var/log/quantum/stream_recover.log
STREAM=quantum:stream:trade.intent
GROUP=quantum:group:execution:trade.intent

source /etc/quantum/core_gates.env

log() { echo "$(date -Iseconds) | $*" | tee -a $LOGFILE; }

# Abort if LIVE
if grep -q 'TRADING_MODE=LIVE' /etc/quantum/testnet.env 2>/dev/null; then
    log "ABORT: LIVE mode detected - recovery disabled"
    exit 0
fi

log "=== ZOMBIE + STALE RECOVERY START ==="

# Ensure group exists
redis-cli XGROUP CREATE $STREAM $GROUP 0 MKSTREAM 2>/dev/null || true

# XAUTOCLAIM stale messages (idle > 60s)
log "XAUTOCLAIM: idle > ${STALE_IDLE_MS}ms"
CLAIMED=$(redis-cli XAUTOCLAIM $STREAM $GROUP quantum-execution $STALE_IDLE_MS 0 COUNT 100 | wc -l)
log "XAUTOCLAIM: $CLAIMED items processed"

# XGROUP DELCONSUMER zombies (idle > 1h, pending = 0)
log "Scanning for zombie consumers..."
redis-cli XINFO CONSUMERS $STREAM $GROUP | awk '
BEGIN { name=""; idle=0; pending=0; zombie_idle='$ZOMBIE_IDLE_MS'; }
/name/ { name=$2; }
/idle/ { idle=$2; }
/pending/ { 
    pending=$2; 
    if (idle > zombie_idle && pending == 0 && name != "") {
        print "ZOMBIE: " name " (idle=" idle "ms, pending=0)";
        system("redis-cli XGROUP DELCONSUMER '$STREAM' '$GROUP' " name " >> '$LOGFILE' 2>&1");
    }
}
' | tee -a $LOGFILE

log "=== RECOVERY COMPLETE ==="
