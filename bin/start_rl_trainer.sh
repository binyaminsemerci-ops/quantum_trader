#!/bin/bash
QT_BASE="${QT_BASE_DIR:-/home/qt/quantum_trader}"
cd "$QT_BASE"
export PYTHONPATH="$QT_BASE"
source "${VIRTUAL_ENV:-/home/qt/quantum_trader_venv}/bin/activate"
exec python microservices/rl_training/main.py
