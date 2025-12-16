#!/bin/bash
cd /tmp
python3 /mnt/c/quantum_trader/test_health_endpoint.py > health_test_output.txt 2>&1
echo "Done - check /tmp/health_test_output.txt and /tmp/health_response.json"
