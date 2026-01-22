#!/bin/bash
# P0.EXIT_GUARD - Deploy strict exit-gating to exit_monitor_service.py

echo "=== PHASE 3: DEPLOY EXIT GUARD ==="

SOURCE="/home/qt/quantum_trader/services/exit_monitor_service.py"
TARGET="/tmp/exit_monitor_guarded.py"

# Step 1: Copy source
cp "$SOURCE" "$TARGET"
echo "✅ File copied to $TARGET"

# Step 2: Insert import for redis.asyncio after line 24
sed -i '24a import redis.asyncio as redis' "$TARGET"
sed -i '24a from collections import defaultdict' "$TARGET"

# Step 3: Add redis_client after binance_client line
sed -i '/^binance_client: Optional\[Client\] = None$/a redis_client: Optional[redis.Redis] = None' "$TARGET"

# Step 4: Add exit guard stats - find line with "exits_triggered" and add after it
sed -i '/    "exits_triggered": 0,/a \    "exits_deduped": 0,\n    "exits_cooldown": 0,\n    "exits_already_closed": 0,' "$TARGET"

# Step 5: Add dedup check function before send_close_order
# Find line number of "async def send_close_order"
LINE=$(grep -n "^async def send_close_order" "$TARGET" | cut -d: -f1)

# Insert guard functions before send_close_order
sed -i "${LINE}i\\
async def check_exit_dedup(redis_client, position_id: str) -> bool:\\
    \"\"\"Check if exit already sent for this position. Returns True if should skip.\"\"\"\\
    try:\\
        key = f\"quantum:dedup:exit:{position_id}\"\\
        result = await redis_client.set(key, \"1\", nx=True, ex=300)\\
        if not result:\\
            logger.info(f\"🔴 EXIT_DEDUP skip pos={position_id}\")\\
            stats[\"exits_deduped\"] += 1\\
            return True\\
        return False\\
    except Exception as e:\\
        logger.error(f\"❌ EXIT_DEDUP failed: {e}\")\\
        return False  # Fail open\\
\\
async def check_exit_cooldown(redis_client, symbol: str, side: str) -> bool:\\
    \"\"\"Check cooldown for symbol/side. Returns True if should skip.\"\"\"\\
    try:\\
        key = f\"quantum:cooldown:exit:{symbol}:{side}\"\\
        exists = await redis_client.exists(key)\\
        if exists:\\
            logger.info(f\"⏸️ EXIT_COOLDOWN skip symbol={symbol} side={side}\")\\
            stats[\"exits_cooldown\"] += 1\\
            return True\\
        await redis_client.set(key, \"1\", ex=30)\\
        return False\\
    except Exception as e:\\
        logger.error(f\"❌ EXIT_COOLDOWN failed: {e}\")\\
        return False\\
\\
async def check_already_closed(symbol: str) -> bool:\\
    \"\"\"Check if position already closed. Returns True if should skip.\"\"\"\\
    if symbol not in tracked_positions:\\
        logger.info(f\"🔴 EXIT_ALREADY_CLOSED symbol={symbol}\")\\
        stats[\"exits_already_closed\"] += 1\\
        return True\\
    return False\\
\\
" "$TARGET"

# Step 6: Add guards at start of send_close_order function
# Find line after "async def send_close_order" and add guards
SEND_LINE=$(grep -n "^async def send_close_order" "$TARGET" | cut -d: -f1)
INSERT_AT=$((SEND_LINE + 2))

sed -i "${INSERT_AT}i\\
    # === EXIT GUARDS ===\\
    if await check_already_closed(position.symbol):\\
        return\\
    \\
    position_id = f\"{position.symbol}_{position.order_id}\"\\
    if await check_exit_dedup(redis_client, position_id):\\
        return\\
    \\
    if await check_exit_cooldown(redis_client, position.symbol, position.side):\\
        return\\
" "$TARGET"

# Step 7: Update log message in send_close_order to include position_id
sed -i 's/f"🎯 EXIT TRIGGERED:/f"📤 EXIT_PUBLISH:/' "$TARGET"

# Step 8: Initialize redis_client in startup function
sed -i '/async def startup():/,/eventbus = EventBusClient/ {
    /eventbus = EventBusClient/i \    # Connect to Redis for exit guards\
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")\
    redis_client = redis.from_url(redis_url, decode_responses=True)\
    logger.info(f"✅ Redis connected for exit guards: {redis_url}")\

}' "$TARGET"

# Step 9: Update health endpoint to include new stats
sed -i '/class HealthResponse(BaseModel):/,/last_check_time:/ {
    /tracked_positions: int/a \    exits_deduped: int\
    exits_cooldown: int\
    exits_already_closed: int
}' "$TARGET"

sed -i '/@app.get("\/health"/,/last_check_time=/ {
    /exits_triggered=/a \        exits_deduped=stats["exits_deduped"],\
        exits_cooldown=stats["exits_cooldown"],\
        exits_already_closed=stats["exits_already_closed"],
}' "$TARGET"

# Validate syntax
echo "Validating syntax..."
python3 -m py_compile "$TARGET"

if [ $? -eq 0 ]; then
    echo "✅ Syntax validation PASSED"
    echo "✅ Patched file ready: $TARGET"
    
    # Deploy
    echo ""
    echo "Deploying patched file..."
    cp "$TARGET" "$SOURCE"
    echo "✅ File deployed to $SOURCE"
    
    # Restart service
    echo ""
    echo "Restarting quantum-exit-monitor service..."
    systemctl restart quantum-exit-monitor.service
    sleep 2
    
    systemctl --no-pager -l status quantum-exit-monitor.service | head -35
else
    echo "❌ Syntax validation FAILED"
    python3 -m py_compile "$TARGET" 2>&1
    exit 1
fi
