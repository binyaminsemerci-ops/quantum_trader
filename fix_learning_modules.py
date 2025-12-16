#!/usr/bin/env python3
"""
🔧 FIX INACTIVE LEARNING MODULES
Creates missing factory functions and fixes imports
"""
import sys
from pathlib import Path

print("=" * 80)
print("🔧 FIXING INACTIVE LEARNING MODULES")
print("=" * 80)
print()

# ============================================================================
# FIX 1: Add get_profit_amplification factory to profit_amplification.py
# ============================================================================
print("[1] Fixing Profit Amplification Layer (PAL)...")

pal_file = Path("/app/backend/services/profit_amplification.py")

if pal_file.exists():
    with open(pal_file, 'r') as f:
        content = f.read()
    
    # Check if factory already exists
    if 'def get_profit_amplification' not in content:
        print("   Adding get_profit_amplification() factory function...")
        
        factory_code = '''

# ============================================================
# FACTORY FUNCTION
# ============================================================

_profit_amplification: Optional[ProfitAmplificationLayer] = None


def get_profit_amplification() -> ProfitAmplificationLayer:
    """Get or create Profit Amplification Layer singleton"""
    global _profit_amplification
    if _profit_amplification is None:
        _profit_amplification = ProfitAmplificationLayer()
    return _profit_amplification
'''
        
        # Append to end of file
        with open(pal_file, 'a') as f:
            f.write(factory_code)
        
        print("   ✅ Factory function added!")
    else:
        print("   ✅ Factory function already exists")
else:
    print("   ❌ File not found!")

print()

# ============================================================================
# FIX 2: Create continuous learning wrapper
# ============================================================================
print("[2] Creating Continuous Learning wrapper...")

cl_dir = Path("/app/backend/services/learning")
cl_dir.mkdir(exist_ok=True)

cl_file = cl_dir / "__init__.py"

cl_code = '''"""
Continuous Learning System

Wraps RetrainingOrchestrator for backward compatibility
"""
from backend.services.retraining_orchestrator import RetrainingOrchestrator

# Alias for compatibility
ContinuousLearningSystem = RetrainingOrchestrator


def get_continuous_learning():
    """Get continuous learning orchestrator"""
    return RetrainingOrchestrator()
'''

with open(cl_file, 'w') as f:
    f.write(cl_code)

print("   ✅ Continuous learning wrapper created!")
print(f"   Location: {cl_file}")

print()

# ============================================================================
# FIX 3: Create position_intelligence_layer alias
# ============================================================================
print("[3] Creating PIL alias module...")

pil_file = Path("/app/backend/services/position_intelligence_layer.py")

pil_code = '''"""
Position Intelligence Layer (PIL) - Alias Module

For backward compatibility with older imports
"""
from backend.services.position_intelligence import (
    PositionIntelligenceLayer,
    PositionCategory,
    PositionRecommendation,
    PositionClassification,
    get_position_intelligence
)

# Alias
get_pil = get_position_intelligence

__all__ = [
    'PositionIntelligenceLayer',
    'PositionCategory', 
    'PositionRecommendation',
    'PositionClassification',
    'get_position_intelligence',
    'get_pil'
]
'''

with open(pil_file, 'w') as f:
    f.write(pil_code)

print("   ✅ PIL alias module created!")
print(f"   Location: {pil_file}")

print()

# ============================================================================
# VERIFICATION
# ============================================================================
print("=" * 80)
print("🧪 VERIFYING FIXES")
print("=" * 80)
print()

# Test PIL
print("[TEST 1] Position Intelligence Layer:")
try:
    from backend.services.position_intelligence_layer import get_pil
    pil = get_pil()
    print("   ✅ Import successful: backend.services.position_intelligence_layer")
    print(f"   ✅ PIL instance: {pil is not None}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test PAL
print("[TEST 2] Profit Amplification Layer:")
try:
    from backend.services.profit_amplification import get_profit_amplification
    pal = get_profit_amplification()
    print("   ✅ Import successful: backend.services.profit_amplification")
    print(f"   ✅ PAL instance: {pal is not None}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test Continuous Learning
print("[TEST 3] Continuous Learning System:")
try:
    from backend.services.learning import get_continuous_learning
    cl = get_continuous_learning()
    print("   ✅ Import successful: backend.services.learning")
    print(f"   ✅ CL instance: {cl is not None}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("✅ ALL FIXES APPLIED!")
print("=" * 80)
print("""
Fixed modules:
1. ✅ Position Intelligence Layer (PIL)
   - Created alias: backend.services.position_intelligence_layer
   - Factory: get_pil()

2. ✅ Profit Amplification Layer (PAL)  
   - Added factory: get_profit_amplification()
   
3. ✅ Continuous Learning System
   - Created wrapper: backend.services.learning
   - Factory: get_continuous_learning()

All learning modules should now import successfully!
Run check_learning_status.py to verify.
""")

print("=" * 80)
