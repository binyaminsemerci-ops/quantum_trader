#!/bin/bash
echo "=== apply_layer full startup log ==="
journalctl -u quantum-apply-layer --no-pager -n 40 -q 2>&1 | tail -40

echo ""
echo "=== exit_ownership import result ==="
journalctl -u quantum-apply-layer --no-pager -n 1000 -o short-precise -q 2>&1 | grep -iE "exit_own|EXIT_OWNER|WARN.*exit|EXIT_OWNERSHIP" | tail -10

echo ""
echo "=== governor in apply_layer (from log) ==="
journalctl -u quantum-apply-layer --no-pager -n 1000 -o short-precise -q 2>&1 | grep -iE "Governor active|GOVERNOR|three permits|TESTNET execution" | tail -10
