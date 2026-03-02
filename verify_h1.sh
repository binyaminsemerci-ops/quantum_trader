#!/bin/bash
echo "--- intent_bridge (828-837) ---"
sed -n '828,837p' /opt/quantum/microservices/intent_bridge/main.py

echo ""
echo "--- harvest_brain (936-943) ---"
sed -n '936,943p' /opt/quantum/microservices/harvest_brain/harvest_brain.py

echo ""
echo "--- risk_brake (131-140) ---"
sed -n '131,140p' /opt/quantum/microservices/risk_brake_v1_patch.py
