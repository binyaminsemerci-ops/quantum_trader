#!/usr/bin/env bash
set -e
echo "Ports:"; ss -tlnp | egrep '(:3000|:9090|:9091)' || true
echo "Grafana:"; curl -sS http://localhost:3000/api/health || true
echo "Prom:"; curl -sS http://localhost:9091/-/healthy || curl -sS http://localhost:9090/-/healthy || true
echo "Prom ExecStart:"; systemctl cat prometheus | grep -E 'listen-address|ExecStart' | head -3
