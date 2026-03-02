#!/bin/bash
journalctl -u quantum-intent-executor.service --since "2 minutes ago" --no-pager 2>/dev/null | tail -30 > /tmp/ie_log_check.txt
echo "WRONGTYPE_COUNT=$(journalctl -u quantum-intent-executor.service --since '2 minutes ago' --no-pager 2>/dev/null | grep -c WRONGTYPE)" >> /tmp/ie_log_check.txt
echo "PERMIT_FOUND_COUNT=$(journalctl -u quantum-intent-executor.service --since '2 minutes ago' --no-pager 2>/dev/null | grep -c 'Permit found')" >> /tmp/ie_log_check.txt
echo "EXECUTED_TRUE=$(journalctl -u quantum-intent-executor.service --since '2 minutes ago' --no-pager 2>/dev/null | grep -c 'executed=True')" >> /tmp/ie_log_check.txt
