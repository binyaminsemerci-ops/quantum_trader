#!/bin/bash
echo "=== PRE-CALIBRATION BASELINE ===" | tee /root/logs/pre_calibration_baseline.txt
echo "Timestamp: $(date -Iseconds)" | tee -a /root/logs/pre_calibration_baseline.txt
echo "" | tee -a /root/logs/pre_calibration_baseline.txt
redis-cli GET quantum:equity:current | jq '.' | tee -a /root/logs/pre_calibration_baseline.txt
echo "" | tee -a /root/logs/pre_calibration_baseline.txt
echo "Decision count (last 100):" | tee -a /root/logs/pre_calibration_baseline.txt
redis-cli LRANGE quantum:decisions:history 0 99 | wc -l | tee -a /root/logs/pre_calibration_baseline.txt
echo "" | tee -a /root/logs/pre_calibration_baseline.txt
systemctl status quantum-ai-engine | grep Active | tee -a /root/logs/pre_calibration_baseline.txt
systemctl status quantum-harvest-consumer | grep Active | tee -a /root/logs/pre_calibration_baseline.txt
echo "" | tee -a /root/logs/pre_calibration_baseline.txt
echo "✅ Baseline saved to /root/logs/pre_calibration_baseline.txt"
