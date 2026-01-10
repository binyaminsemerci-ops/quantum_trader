# Makefile for Quantum Trading System
# ALL operations use ops/run.sh wrapper (Golden Contract)

.PHONY: help quality-gate scoreboard diagnose build-patchtst-dataset audit train-patchtst

# Service name for AI Engine operations
SERVICE := ai-engine

# Wrapper script (MANDATORY - DO NOT BYPASS)
RUN := ops/run.sh $(SERVICE)

help:
	@echo "Quantum Trader - Golden Contract Makefile"
	@echo ""
	@echo "Model Safety:"
	@echo "  quality-gate           - Check model quality (telemetry-only, BLOCKER)"
	@echo "  diagnose               - Diagnose variance collapse (NO training/activation)"
	@echo "  scoreboard             - View all models status"
	@echo ""
	@echo "Training:"
	@echo "  build-patchtst-dataset - Build temporal sequences for PatchTST"
	@echo "  train-patchtst         - Train PatchTST model"
	@echo ""
	@echo "Compliance:"
	@echo "  audit                  - Run Golden Contract audit"
	@echo ""
	@echo "All targets use ops/run.sh wrapper (NEVER bypass)"

quality-gate:
	@echo "Running quality gate (BLOCKER)..."
	$(RUN) ops/model_safety/quality_gate.py

diagnose:
	@echo "Running diagnosis mode (NO training/activation)..."
	$(RUN) ops/model_safety/diagnose_collapse.py

scoreboard:
	@echo "Generating model scoreboard..."
	$(RUN) ops/model_safety/scoreboard.py

build-patchtst-dataset:
	@echo "Building PatchTST sequence dataset..."
	$(RUN) scripts/build_patchtst_sequence_dataset.py

train-patchtst:
	@echo "Training PatchTST model..."
	$(RUN) ops/training/train_patchtst.py

audit:
	@echo "Running contract compliance audit..."
	$(RUN) ops/audit_contract.py
