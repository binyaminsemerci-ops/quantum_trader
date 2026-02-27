#!/bin/bash
set -e
# Backup
cp /etc/quantum/harvest-proposal.env /etc/quantum/harvest-proposal.env.bak.$(date +%s)

# Remove phantom symbols from SYMBOLS line
sed -i 's/SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,OPUSDT,ARBUSDT,INJUSDT,DOTUSDT,STXUSDT/SYMBOLS=ETHUSDT,OPUSDT,ARBUSDT,DOTUSDT,STXUSDT/' /etc/quantum/harvest-proposal.env

echo "Updated SYMBOLS:"
grep SYMBOLS /etc/quantum/harvest-proposal.env

# Delete stale phantom proposals
redis-cli DEL \
  quantum:harvest:proposal:SOLUSDT \
  quantum:harvest:proposal:BTCUSDT \
  quantum:harvest:proposal:BNBUSDT \
  quantum:harvest:proposal:XRPUSDT \
  quantum:harvest:proposal:INJUSDT \
  quantum:harvest:heat:SOLUSDT \
  quantum:harvest:heat:BTCUSDT \
  quantum:harvest:heat:BNBUSDT \
  quantum:harvest:heat:XRPUSDT \
  quantum:harvest:heat:INJUSDT \
  quantum:harvest_v2:state:SOLUSDT \
  quantum:harvest_v2:state:BTCUSDT \
  quantum:harvest_v2:state:BNBUSDT \
  quantum:harvest_v2:state:XRPUSDT \
  quantum:harvest_v2:state:INJUSDT
echo "Phantom keys deleted"

# Restart service
systemctl daemon-reload
systemctl restart quantum-harvest-proposal.service
sleep 3
systemctl status quantum-harvest-proposal.service --no-pager | head -20
echo "DONE"
