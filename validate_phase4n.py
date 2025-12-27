#!/usr/bin/env python3
"""
Phase 4N Validation Script
Tests Adaptive Leverage Engine functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microservices'))

from exitbrain_v3_5.adaptive_leverage_engine import test_adaptive_engine

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4N - ADAPTIVE LEVERAGE ENGINE VALIDATION")
    print("=" * 70)
    print()
    
    success = test_adaptive_engine()
    
    print()
    print("=" * 70)
    if success:
        print("✅ VALIDATION PASSED")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        sys.exit(1)
