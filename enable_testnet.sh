#!/bin/bash
# Enable TESTNET mode after IP is whitelisted

echo "🔄 Switching to TESTNET mode..."
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "sed -i 's/EXECUTION_MODE=PAPER/EXECUTION_MODE=TESTNET/' /home/qt/quantum_trader/.env && docker restart quantum_execution && sleep 5 && docker logs --tail=30 quantum_execution 2>&1 | grep -E 'TESTNET|Balance|MODE|STARTED'"

echo -e "\n✅ Done! Check if TESTNET mode is active above."
