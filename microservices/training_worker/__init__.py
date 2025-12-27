"""
Training Worker Service
========================
Listens for model retraining jobs from Strategic Evolution (Phase 4T)
and executes actual model training.

This service bridges the gap between:
- Strategic Evolution (job scheduler)
- Model training execution
- Redis streams (job queue)
"""

__version__ = "1.0.0"
