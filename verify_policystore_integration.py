"""
Quick Integration Verification Script

Verifies PolicyStore AI component integration is working correctly.
Run after backend startup to check all connections.
"""

import requests
import json
import sys
from typing import Dict, Any


def print_header(text: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def check_mark(condition: bool) -> str:
    """Return check mark or X based on condition."""
    return "✅" if condition else "❌"


def verify_backend_running() -> bool:
    """Verify backend is accessible."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def verify_policystore_available() -> tuple[bool, Dict[str, Any]]:
    """Verify PolicyStore API is available."""
    try:
        response = requests.get("http://localhost:8000/api/policy/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('available', False), data
        return False, {}
    except:
        return False, {}


def verify_policystore_initialized() -> tuple[bool, Dict[str, Any]]:
    """Verify PolicyStore has policy data."""
    try:
        response = requests.get("http://localhost:8000/api/policy", timeout=5)
        if response.status_code == 200:
            data = response.json()
            policy = data.get('policy', {})
            return bool(policy.get('risk_mode')), policy
        return False, {}
    except:
        return False, {}


def verify_msc_integration(policy: Dict[str, Any]) -> tuple[bool, str]:
    """Verify MSC AI can write to PolicyStore."""
    # Check if policy has MSC-writable fields
    has_risk_mode = 'risk_mode' in policy
    has_max_risk = 'max_risk_per_trade' in policy
    has_max_pos = 'max_positions' in policy
    has_min_conf = 'global_min_confidence' in policy
    
    success = has_risk_mode and has_max_risk and has_max_pos and has_min_conf
    
    details = f"risk_mode={policy.get('risk_mode', 'MISSING')}, "
    details += f"max_risk={policy.get('max_risk_per_trade', 'MISSING')}, "
    details += f"max_pos={policy.get('max_positions', 'MISSING')}, "
    details += f"min_conf={policy.get('global_min_confidence', 'MISSING')}"
    
    return success, details


def verify_opprank_integration(policy: Dict[str, Any]) -> tuple[bool, str]:
    """Verify OpportunityRanker can write to PolicyStore."""
    rankings = policy.get('opp_rankings', {})
    
    if not rankings:
        return False, "No rankings found (OpportunityRanker hasn't run yet)"
    
    num_symbols = len(rankings)
    top_symbol = max(rankings.items(), key=lambda x: x[1]) if rankings else None
    
    details = f"{num_symbols} symbols ranked"
    if top_symbol:
        details += f", top: {top_symbol[0]} ({top_symbol[1]:.3f})"
    
    return True, details


def trigger_msc_evaluation() -> tuple[bool, str]:
    """Try to trigger MSC evaluation."""
    try:
        response = requests.post("http://localhost:8000/msc/evaluate", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return True, f"Status: {data.get('status', 'unknown')}"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def trigger_opprank_update() -> tuple[bool, str]:
    """Try to trigger OpportunityRanker update."""
    try:
        response = requests.post("http://localhost:8000/opportunities/update", timeout=30)
        if response.status_code == 200:
            return True, "Rankings updated"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def main():
    """Run verification checks."""
    
    print_header("PolicyStore AI Integration Verification")
    
    # Check 1: Backend Running
    print("1. Checking backend...")
    backend_ok = verify_backend_running()
    print(f"   {check_mark(backend_ok)} Backend running at http://localhost:8000")
    
    if not backend_ok:
        print("\n❌ FAILED: Backend not running!")
        print("   Start backend with: python backend/main.py")
        sys.exit(1)
    
    # Check 2: PolicyStore Available
    print("\n2. Checking PolicyStore API...")
    ps_available, ps_status = verify_policystore_available()
    print(f"   {check_mark(ps_available)} PolicyStore API available")
    
    if ps_available:
        print(f"      └─ Current risk mode: {ps_status.get('current_risk_mode', 'unknown')}")
        print(f"      └─ Last updated: {ps_status.get('last_updated', 'never')}")
    else:
        print("\n❌ FAILED: PolicyStore not available!")
        sys.exit(1)
    
    # Check 3: PolicyStore Initialized
    print("\n3. Checking PolicyStore initialization...")
    ps_init, policy = verify_policystore_initialized()
    print(f"   {check_mark(ps_init)} PolicyStore initialized with data")
    
    if not ps_init:
        print("\n❌ FAILED: PolicyStore not initialized!")
        sys.exit(1)
    
    # Check 4: MSC AI Integration
    print("\n4. Checking MSC AI integration...")
    msc_ok, msc_details = verify_msc_integration(policy)
    print(f"   {check_mark(msc_ok)} MSC AI fields present")
    print(f"      └─ {msc_details}")
    
    # Check 5: OpportunityRanker Integration
    print("\n5. Checking OpportunityRanker integration...")
    opp_ok, opp_details = verify_opprank_integration(policy)
    print(f"   {check_mark(opp_ok)} OpportunityRanker rankings present")
    print(f"      └─ {opp_details}")
    
    # Summary
    print_header("Integration Status Summary")
    
    all_checks = [backend_ok, ps_available, ps_init, msc_ok]
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"Core Checks:      {passed}/{total} passed")
    print(f"MSC AI:           {check_mark(msc_ok)} Connected")
    print(f"OpportunityRank:  {check_mark(opp_ok)} Connected")
    
    if not opp_ok:
        print("\n💡 Tip: Trigger OpportunityRanker update:")
        print("   curl -X POST http://localhost:8000/opportunities/update")
    
    # Optional: Trigger tests
    print("\n" + "="*70)
    run_triggers = input("Run live integration tests? (y/n): ").lower().strip()
    
    if run_triggers == 'y':
        print("\n6. Triggering MSC AI evaluation...")
        msc_trigger_ok, msc_trigger_msg = trigger_msc_evaluation()
        print(f"   {check_mark(msc_trigger_ok)} MSC evaluation triggered")
        print(f"      └─ {msc_trigger_msg}")
        
        print("\n7. Triggering OpportunityRanker update...")
        opp_trigger_ok, opp_trigger_msg = trigger_opprank_update()
        print(f"   {check_mark(opp_trigger_ok)} OpportunityRanker update triggered")
        print(f"      └─ {opp_trigger_msg}")
        
        if msc_trigger_ok or opp_trigger_ok:
            print("\n   Fetching updated policy...")
            import time
            time.sleep(2)
            _, updated_policy = verify_policystore_initialized()
            
            print("\n   Updated PolicyStore state:")
            print(f"   Risk Mode: {updated_policy.get('risk_mode')}")
            print(f"   Rankings:  {len(updated_policy.get('opp_rankings', {}))} symbols")
    
    # Final verdict
    print_header("Final Verdict")
    
    if all(all_checks):
        print("✅ ALL CHECKS PASSED!")
        print("\n🎉 PolicyStore AI integration is working correctly!")
        print("\nNext steps:")
        print("  • Monitor with: python demo_policystore_integration.py")
        print("  • View policy: curl http://localhost:8000/api/policy")
        print("  • Check logs for: '✅ Policy written to PolicyStore'")
        sys.exit(0)
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nReview the errors above and check:")
        print("  • Backend logs for PolicyStore initialization")
        print("  • MSC AI scheduler logs")
        print("  • OpportunityRanker logs")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Verification cancelled")
        sys.exit(1)
