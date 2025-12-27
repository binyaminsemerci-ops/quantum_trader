"""
Quick test to verify RL leverage integration
"""
import sys
sys.path.insert(0, '/app')

# Test that leverage parameter is properly typed
from typing import Optional
from backend.services.execution.execution import BinanceExecution

async def test_leverage_signature():
    """Test that submit_order accepts leverage parameter"""
    import inspect
    sig = inspect.signature(BinanceExecution.submit_order)
    print(f"✓ submit_order signature: {sig}")
    
    # Check leverage parameter exists
    params = sig.parameters
    if 'leverage' in params:
        print(f"✓ leverage parameter found: {params['leverage']}")
        print(f"  - Type: {params['leverage'].annotation}")
        print(f"  - Default: {params['leverage'].default}")
    else:
        print("✗ leverage parameter NOT found!")
        return False
    
    return True

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_leverage_signature())
    exit(0 if result else 1)
