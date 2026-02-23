#!/bin/bash
echo "=== Container names ==="
docker ps --format "{{.Names}}" | grep -iE "intent|apply|bridge"
echo "---"
docker ps --format "{{.Names}}" | head -30
