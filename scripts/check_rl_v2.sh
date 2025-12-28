#!/bin/bash
echo "=== RL Feedback Bridge v2 Status ==="
docker ps --filter name=quantum_rl_feedback_v2 --format 'table {{.Names}}\t{{.Status}}'
echo ""
echo "=== Recent Logs ==="
docker logs quantum_rl_feedback_v2 --tail 15 2>&1 || echo "Container not found"
echo ""
echo "=== Image Info ==="
docker images | grep rl-feedback-v2 || echo "Image not found"
