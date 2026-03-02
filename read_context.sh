#!/bin/bash
echo "--- intent_bridge (828-848) ---"
sed -n '828,848p' /opt/quantum/microservices/intent_bridge/main.py

echo ""
echo "--- harvest_brain (933,950) ---"
sed -n '933,950p' /opt/quantum/microservices/harvest_brain/harvest_brain.py
