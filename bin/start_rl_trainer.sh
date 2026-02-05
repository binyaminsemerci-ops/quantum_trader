#!/bin/bash
cd /opt/quantum
export PYTHONPATH=/opt/quantum
source /opt/quantum/venvs/runtime/bin/activate
exec python microservices/rl_training/main.py
