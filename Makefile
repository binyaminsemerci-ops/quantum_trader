.PHONY: help quality-gate scoreboard build-patchtst-dataset

PYTHON := /opt/quantum/venvs/ai-engine/bin/python

help:
	@echo "Quantum Trader - Model Safety Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  quality-gate           - Check model quality (BLOCKER)"
	@echo "  scoreboard             - View all models status"
	@echo "  build-patchtst-dataset - Build temporal sequences for PatchTST"
	@echo ""
	@echo "Usage: make <target>"

quality-gate:
	@echo "Running quality gate (BLOCKER)..."
	$(PYTHON) ops/model_safety/quality_gate.py

scoreboard:
	@echo "Generating model scoreboard..."
	$(PYTHON) ops/model_safety/scoreboard.py

build-patchtst-dataset:
	@echo "Building PatchTST sequence dataset..."
	$(PYTHON) scripts/build_patchtst_sequence_dataset.py
