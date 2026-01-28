#!/usr/bin/env python3
"""Test Pydantic settings loading."""

import sys
import os
sys.path.insert(0, '/home/qt/quantum_trader')

# Set working directory to match service
os.chdir('/home/qt/quantum_trader')

print(f"Working directory: {os.getcwd()}")
print(f".env exists: {os.path.exists('.env')}")

from microservices.ai_engine.config import settings

print("\n" + "="*80)
print("PYDANTIC SETTINGS INSPECTION")
print("="*80)

print(f"\nENSEMBLE_MODELS attribute exists: {hasattr(settings, 'ENSEMBLE_MODELS')}")
print(f"ENSEMBLE_MODELS value: {getattr(settings, 'ENSEMBLE_MODELS', 'MISSING')}")
print(f"ENSEMBLE_MODELS type: {type(getattr(settings, 'ENSEMBLE_MODELS', None))}")
print(f"bool(ENSEMBLE_MODELS): {bool(getattr(settings, 'ENSEMBLE_MODELS', False))}")

if hasattr(settings, 'ENSEMBLE_MODELS'):
    val = settings.ENSEMBLE_MODELS
    print(f"\nDetailed inspection:")
    print(f"  - Value: {val}")
    print(f"  - Length: {len(val) if val else 0}")
    print(f"  - Is list: {isinstance(val, list)}")
    print(f"  - repr: {repr(val)}")

# Check for env override
env_val = os.getenv('AI_ENGINE_ENSEMBLE_MODELS')
print(f"\nEnvironment variable AI_ENGINE_ENSEMBLE_MODELS: {env_val}")

# Dump all settings
print("\n" + "="*80)
print("ALL SETTINGS")
print("="*80)
for key, value in settings.model_dump().items():
    if 'ENSEMBLE' in key or 'MODEL' in key:
        print(f"{key}: {value}")
