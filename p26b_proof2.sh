#!/bin/bash
echo "=== PROOF 2: Service logs OK ==="
journalctl -u quantum-harvest-proposal.service --since '2 min ago' --no-pager | tail -80
