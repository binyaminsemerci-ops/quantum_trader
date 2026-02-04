#!/usr/bin/env python3
"""
P3 Harvest Restore - Deployment Verification
Checks that all components are correctly integrated
"""

import re
import sys
from pathlib import Path

def check_harvest_brain():
    """Verify harvest_brain.py has all P3 components"""
    print("🔍 Checking harvest_brain.py...")
    
    file_path = Path('microservices/harvest_brain/harvest_brain.py')
    content = file_path.read_text(encoding='utf-8')
    
    checks = {
        'P2 imports': 'from ai_engine.risk_kernel_harvest import',
        'Position.age_sec': 'age_sec: float = 0.0',
        'Position.peak_price': 'peak_price: float = 0.0',
        'apply.result stream': "quantum:stream:apply.result",
        'compute_harvest_proposal': 'p2_result = compute_harvest_proposal(',
        'KILL_SCORE log': 'KILL_SCORE=',
        'PARTIAL_25 handler': "harvest_action == 'PARTIAL_25'",
        'PARTIAL_50 handler': "harvest_action == 'PARTIAL_50'",
        'PARTIAL_75 handler': "harvest_action == 'PARTIAL_75'",
        'FULL_CLOSE handler': "harvest_action == 'FULL_CLOSE_PROPOSED'",
        '_get_market_state': 'def _get_market_state(self, symbol: str)',
        '_get_harvest_theta': 'def _get_harvest_theta(self)',
    }
    
    results = {}
    for name, pattern in checks.items():
        found = pattern in content
        results[name] = found
        status = "✅" if found else "❌"
        print(f"  {status} {name}")
    
    missing = [k for k, v in results.items() if not v]
    
    if missing:
        print(f"\n⚠️  Missing components: {', '.join(missing)}")
        print("\n🔧 Action required:")
        print("   1. Run: python p3_harvest_patch_guide.py")
        print("   2. Follow manual patch instructions")
        return False
    else:
        print("\n✅ harvest_brain.py: ALL CHECKS PASSED")
        return True

def check_exitbrain_lsf():
    """Verify ExitBrain v3.5 has LSF integration"""
    print("\n🔍 Checking exitbrain_v3_5...")
    
    file_path = Path('microservices/exitbrain_v3_5/exit_brain.py')
    if not file_path.exists():
        print("  ❌ exit_brain.py not found")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    checks = {
        'AdaptiveLeverageEngine import': 'from .adaptive_leverage_engine import',
        'compute_levels call': 'self.adaptive_engine.compute_levels(',
        'LSF logging': 'LSF=',
        'tp1_pct usage': 'adaptive_levels.tp1_pct',
        'tp2_pct usage': 'adaptive_levels.tp2_pct',
        'tp3_pct usage': 'adaptive_levels.tp3_pct',
        'harvest_scheme': 'adaptive_levels.harvest_scheme',
    }
    
    results = {}
    for name, pattern in checks.items():
        found = pattern in content
        results[name] = found
        status = "✅" if found else "❌"
        print(f"  {status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ExitBrain v3.5: LSF INTEGRATION VERIFIED")
    else:
        print("\n❌ ExitBrain v3.5: LSF integration incomplete")
    
    return all_passed

def check_adaptive_leverage_engine():
    """Verify adaptive_leverage_engine.py has LSF formulas"""
    print("\n🔍 Checking adaptive_leverage_engine.py...")
    
    file_path = Path('microservices/exitbrain_v3_5/adaptive_leverage_engine.py')
    if not file_path.exists():
        print("  ❌ adaptive_leverage_engine.py not found")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    checks = {
        'compute_lsf method': 'def compute_lsf(self, leverage: float)',
        'LSF formula': '1.0 / (1.0 + math.log(lev + 1.0))',
        'TP1 formula': 'tp1 = base_tp * (0.6 + lsf)',
        'TP2 formula': 'tp2 = base_tp * (1.2 + lsf / 2.0)',
        'TP3 formula': 'tp3 = base_tp * (1.8 + lsf / 4.0)',
        'SL formula': 'sl = base_sl * (1.0 + (1.0 - lsf) * 0.8)',
        'harvest_scheme_for': 'def harvest_scheme_for(self, leverage: float)',
        'AdaptiveLevels dataclass': '@dataclass(frozen=True)\nclass AdaptiveLevels:',
    }
    
    results = {}
    for name, pattern in checks.items():
        found = pattern in content
        results[name] = found
        status = "✅" if found else "❌"
        print(f"  {status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ AdaptiveLeverageEngine: LSF FORMULAS VERIFIED")
    else:
        print("\n❌ AdaptiveLeverageEngine: LSF formulas incomplete")
    
    return all_passed

def check_risk_kernel_harvest():
    """Verify risk_kernel_harvest.py exists and has P2 functions"""
    print("\n🔍 Checking ai_engine/risk_kernel_harvest.py...")
    
    file_path = Path('ai_engine/risk_kernel_harvest.py')
    if not file_path.exists():
        print("  ❌ risk_kernel_harvest.py not found")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    checks = {
        'compute_harvest_proposal': 'def compute_harvest_proposal(',
        'compute_kill_score': 'def compute_kill_score(',
        'compute_R_net': 'def compute_R_net(',
        'HarvestTheta': 'class HarvestTheta:',
        'PositionSnapshot': 'class PositionSnapshot:',
        'MarketState': 'class MarketState:',
        'T1_R threshold': 'T1_R: float = 2.0',
        'T2_R threshold': 'T2_R: float = 4.0',
        'T3_R threshold': 'T3_R: float = 6.0',
        'kill_threshold': 'kill_threshold: float = 0.6',
    }
    
    results = {}
    for name, pattern in checks.items():
        found = pattern in content
        results[name] = found
        status = "✅" if found else "❌"
        print(f"  {status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ risk_kernel_harvest.py: P2 KERNEL VERIFIED")
    else:
        print("\n❌ risk_kernel_harvest.py: P2 kernel incomplete")
    
    return all_passed

def main():
    print("=" * 60)
    print("P3 HARVEST RESTORE - DEPLOYMENT VERIFICATION")
    print("=" * 60)
    print()
    
    results = {
        'harvest_brain': check_harvest_brain(),
        'exitbrain_lsf': check_exitbrain_lsf(),
        'adaptive_engine': check_adaptive_leverage_engine(),
        'risk_kernel': check_risk_kernel_harvest(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {component}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - READY FOR DEPLOYMENT")
        print("\n Next steps:")
        print("   1. Deploy to VPS: git push && ssh root@46.224.116.254 'cd /root/quantum_trader && git pull'")
        print("   2. Restart service: systemctl restart quantum-harvest-brain")
        print("   3. Monitor logs: journalctl -u quantum-harvest-brain -f --no-pager | grep -E '\\[HARVEST\\]|R=|KILL_SCORE='")
        print("   4. Check Redis: redis-cli --scan --pattern 'quantum:position:*'")
        sys.exit(0)
    else:
        print("\n⚠️  SOME CHECKS FAILED - REVIEW REQUIRED")
        print("\n Action:")
        print("   1. Review failed checks above")
        print("   2. Run p3_harvest_patch_guide.py for instructions")
        print("   3. Re-run this script after fixes")
        sys.exit(1)

if __name__ == '__main__':
    main()
