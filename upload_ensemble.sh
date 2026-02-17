#!/bin/bash
# Upload ensemble_manager.py to VPS

# Base64 encode and upload
base64 -w 0 /mnt/c/quantum_trader/ai_engine/ensemble_manager.py | ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "base64 -d > /opt/quantum/ai_engine/ensemble_manager.py && echo 'Upload complete' && ls -lh /opt/quantum/ai_engine/ensemble_manager.py"
