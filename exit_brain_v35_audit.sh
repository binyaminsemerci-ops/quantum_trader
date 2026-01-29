#!/bin/bash
set -euo pipefail

echo "=== EXIT BRAIN V3.5 OPERATIONAL AUDIT ==="
echo "Timestamp: $(date -u)"
echo "Git commit: $(cd /home/qt/quantum_trader && git rev-parse --short HEAD)"
echo ""

echo "=== STEP 0: DISCOVER SERVICE ==="
echo "1) Position monitor services:"
systemctl list-units --type=service --all | grep -i position | head -50 || echo "No position services found"
echo ""

echo "2) Position monitor unit files:"
systemctl list-unit-files | grep -i position | head -50 || echo "No position unit files"
echo ""

echo "3) Environment config:"
grep -R "EXIT_BRAIN" /etc/quantum 2>/dev/null | head -80 || echo "No EXIT_BRAIN config in /etc/quantum"
echo ""

echo "4) Systemd position monitor config:"
grep -R "position_monitor" /etc/systemd/system 2>/dev/null | head -80 || echo "No systemd config found"
echo ""

# Try common service names
UNIT_PM=""
for name in quantum-position-monitor quantum-position_monitor position-monitor position_monitor; do
    if systemctl list-unit-files | grep -q "^${name}.service"; then
        UNIT_PM="${name}.service"
        echo "✅ Found service: $UNIT_PM"
        break
    fi
done

if [ -z "$UNIT_PM" ]; then
    echo "❌ FAIL: No position monitor service found"
    exit 1
fi

echo ""
echo "=== STEP 1: SERVICE HEALTH ==="
echo "Service: $UNIT_PM"
systemctl is-active "$UNIT_PM" || echo "⚠️  Service not active"
echo ""

systemctl status "$UNIT_PM" --no-pager -l | head -80
echo ""

echo "Recent logs (last 200 lines, showing last 120):"
journalctl -u "$UNIT_PM" -n 200 --no-pager | tail -120
echo ""

echo "=== STEP 2: CODE PATH VERIFICATION ==="
echo "1) ExitBrainV35 usage:"
grep -R "ExitBrainV35" /home/qt/quantum_trader --include="*.py" | head -50 || echo "Not found"
echo ""

echo "2) build_exit_plan calls:"
grep -R "build_exit_plan" /home/qt/quantum_trader --include="*.py" | head -50 || echo "Not found"
echo ""

echo "3) Order creation in position_monitor:"
grep -R "create_order\|futures_create_order" /home/qt/quantum_trader/backend/services/monitoring/position_monitor.py | head -50 || echo "Not found"
echo ""

echo "4) TP/SL order types:"
grep -R "STOP\|TAKE_PROFIT\|reduceOnly\|closePosition" /home/qt/quantum_trader/backend/services/monitoring --include="*.py" | head -80 || echo "Not found"
echo ""

echo "=== STEP 3: EXCHANGE REALITY CHECK ==="
echo "Looking for existing position/order check scripts..."
ls -la /home/qt/quantum_trader/scripts 2>/dev/null | head -50 || echo "No scripts dir"
echo ""

echo "Finding position/order tools:"
find /home/qt/quantum_trader -maxdepth 3 -type f -name "*position*" -o -name "*order*" 2>/dev/null | head -80 || echo "None found"
echo ""

echo "Searching for Binance client usage:"
grep -R "fetch.*position\|open_positions\|get_position" /home/qt/quantum_trader --include="*.py" | head -50 || echo "Not found"
echo ""

echo "=== STEP 4: DIRECT EXCHANGE CHECK (READ-ONLY) ==="
echo "Attempting to query open positions and orders..."

# Try to use existing check scripts
if [ -f "/home/qt/quantum_trader/check_exit_brain_positions.py" ]; then
    echo "Running check_exit_brain_positions.py:"
    cd /home/qt/quantum_trader && python3 check_exit_brain_positions.py || echo "Script failed"
elif [ -f "/home/qt/quantum_trader/check_orders_simple.py" ]; then
    echo "Running check_orders_simple.py:"
    cd /home/qt/quantum_trader && python3 check_orders_simple.py || echo "Script failed"
else
    echo "Creating minimal position/order checker..."
    python3 << 'PYEOF'
import os
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

try:
    from binance.client import Client
    
    # Check testnet vs live
    use_testnet = os.getenv("STAGING_MODE", "false").lower() == "true" or os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("❌ No Binance credentials found")
        sys.exit(1)
    
    if use_testnet:
        print("Using Binance TESTNET")
        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = 'https://testnet.binancefuture.com'
    else:
        print("Using Binance LIVE")
        client = Client(api_key, api_secret)
    
    # Fetch positions
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    print(f"\n📊 Open positions: {len(open_positions)}")
    for pos in open_positions[:5]:  # Show first 5
        symbol = pos['symbol']
        qty = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        pnl = float(pos['unRealizedProfit'])
        print(f"  {symbol}: qty={qty:.4f}, entry=${entry:.2f}, mark=${mark:.2f}, PnL=${pnl:.2f}")
        
        # Get orders for this symbol
        orders = client.futures_get_open_orders(symbol=symbol)
        tp_orders = [o for o in orders if 'TAKE_PROFIT' in o.get('type', '')]
        sl_orders = [o for o in orders if 'STOP' in o.get('type', '')]
        
        print(f"    TP orders: {len(tp_orders)}, SL orders: {len(sl_orders)}")
        for order in orders[:3]:  # Show first 3 orders
            otype = order.get('type', 'UNKNOWN')
            side = order.get('side', '')
            price = order.get('stopPrice') or order.get('price', '0')
            print(f"      {otype} {side} @ ${price}")
    
    if not open_positions:
        print("⚠️  No open positions found - cannot verify protective orders")
    
except Exception as e:
    print(f"❌ Exchange check failed: {e}")
    import traceback
    traceback.print_exc()
PYEOF
fi

echo ""
echo "=== AUDIT COMPLETE ==="
echo ""
echo "VERDICT CRITERIA:"
echo "PASS requires:"
echo "  1) Service active and running loop"
echo "  2) ExitBrainV35 used in code"
echo "  3) At least 1 open position has protective orders (TP/SL)"
echo "  4) Recent activity in logs"
echo ""
echo "Check output above to determine PASS/FAIL"
