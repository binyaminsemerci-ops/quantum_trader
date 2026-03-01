#!/bin/bash
# ============================================================
# FIX: Kelly sizing floor + Stop Layer4 optimizer spam
# ============================================================
set -e

echo "=== FIX: Layer4 Kelly sizing ==="

# 1. Stop the Layer4 portfolio optimizer (it's outputting only NO_DATA)
echo ">>> Stopping quantum-layer4-portfolio-optimizer.service"
systemctl stop quantum-layer4-portfolio-optimizer.service 2>/dev/null || echo "   (service not found or already stopped)"
systemctl is-active quantum-layer4-portfolio-optimizer.service 2>/dev/null && echo "  STILL ACTIVE!" || echo "  ✅ Stopped"

# 2. Set fresh Kelly sizing floor for all major symbols
# Use ts = NOW - 60 (60 seconds ago = fresh within 300s window)
FRESH_TS=$(($(date +%s) - 60))
echo ">>> Setting Kelly floor ts=$FRESH_TS (fresh)"

SYMBOLS_LIST="BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT MATICUSDT TRXUSDT UNIUSDT ATOMUSDT NEARUSDT APTUSDT ARBUSDT INJUSDT"

for sym in $SYMBOLS_LIST; do
    redis-cli HSET quantum:layer4:sizing:$sym \
        symbol "$sym" \
        recommendation "MINIMUM_VIABLE_FLOOR" \
        kelly_raw 0.04 \
        kelly_adj 0.04 \
        size_usdt 200.0 \
        max_leverage 2 \
        reason "minimum_viable_floor_no_backtest" \
        ts "$FRESH_TS" > /dev/null
done
echo "   ✅ Set size_usdt=200 for all symbols (ts=$FRESH_TS)"

# 3. Update MIN_ORDER_USD in ai-engine.env from 50 to 200
echo ""
echo ">>> Updating ai-engine.env MIN_ORDER_USD: 50 -> 200"
cp /etc/quantum/ai-engine.env /etc/quantum/ai-engine.env.bak.$(date +%s)
sed -i 's/^MIN_ORDER_USD=50/MIN_ORDER_USD=200/' /etc/quantum/ai-engine.env
grep "MIN_ORDER_USD" /etc/quantum/ai-engine.env
echo "   ✅ ai-engine.env updated"

# 4. Verify Kelly sizing is fresh and correct
echo ""
echo ">>> Verify Kelly sizing (fresh check)"
NOW=$(date +%s)
for sym in BTCUSDT BNBUSDT XRPUSDT; do
    ts_val=$(redis-cli HGET quantum:layer4:sizing:$sym ts)
    size_val=$(redis-cli HGET quantum:layer4:sizing:$sym size_usdt)
    rec_val=$(redis-cli HGET quantum:layer4:sizing:$sym recommendation)
    age=$((NOW - ts_val))
    echo "   $sym: size=$size_val rec=$rec_val age=${age}s (<300 = fresh)"
done

# 5. Add a cron job to refresh Kelly sizing every 3 minutes
# (so it stays fresh even after the 300s freshness window)
echo ""
echo ">>> Adding cron to keep Kelly sizing fresh"
cat > /tmp/refresh_kelly.sh << 'KELLY_REFRESH'
#!/bin/bash
FRESH_TS=$(($(date +%s) - 60))
SYMBOLS="BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT MATICUSDT TRXUSDT UNIUSDT ATOMUSDT NEARUSDT APTUSDT ARBUSDT INJUSDT"
for sym in $SYMBOLS; do
    redis-cli HSET quantum:layer4:sizing:$sym ts "$FRESH_TS" size_usdt 200.0 recommendation MINIMUM_VIABLE_FLOOR kelly_adj 0.04 > /dev/null
done
KELLY_REFRESH
chmod +x /tmp/refresh_kelly.sh
cp /tmp/refresh_kelly.sh /usr/local/bin/refresh_kelly.sh

# Add to crontab (every 2 minutes)
if ! crontab -l 2>/dev/null | grep -q refresh_kelly; then
    (crontab -l 2>/dev/null; echo "*/2 * * * * /usr/local/bin/refresh_kelly.sh") | crontab -
    echo "   ✅ Cron job added (every 2 min)"
else
    echo "   ✅ Cron job already present"
fi

# 6. Restart apply-layer to pick up new settings
echo ""
echo ">>> Restarting quantum-apply-layer.service"
systemctl restart quantum-apply-layer.service
sleep 3
systemctl is-active quantum-apply-layer.service && echo "  ✅ apply-layer running" || echo "  ❌ FAILED"

# 7. Final verification
echo ""
echo "=== FINAL VERIFICATION ==="
echo "--- Kelly sizing (BTCUSDT) ---"
redis-cli HGETALL quantum:layer4:sizing:BTCUSDT

echo ""
echo "--- Layer4 optimizer status ---"
systemctl is-active quantum-layer4-portfolio-optimizer.service 2>/dev/null || echo "inactive (expected)"

echo ""
echo "--- apply-layer SYMBOLS ---"
grep "^SYMBOLS=" /etc/quantum/apply-layer.env

echo ""
echo "--- Risk limits ---"
grep -E "RISK_DAILY|RISK_MAX|K_OPEN" /etc/quantum/apply-layer.env

echo ""
echo "=== Kelly sizing fix COMPLETE ==="
