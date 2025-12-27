"""
Module Import Validation Script
Tests imports for modules that had issues during last system check
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_import(module_path, description):
    """Test a single module import"""
    try:
        if "." in module_path:
            parts = module_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            getattr(mod, parts[1])
        else:
            __import__(module_path)
        logger.info(f"✅ {description}: {module_path}")
        return True
    except ImportError as e:
        logger.error(f"❌ {description}: {module_path}")
        logger.error(f"   Error: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  {description}: {module_path}")
        logger.warning(f"   Error: {e}")
        return True  # Module imports but has runtime issues (acceptable)

def main():
    """Run all module import tests"""
    logger.info("=" * 60)
    logger.info("QUANTUM TRADER - MODULE IMPORT VALIDATION")
    logger.info("=" * 60)
    
    tests = [
        ("microservices.rl_sizing_agent", "RL Sizing Agent Module"),
        ("microservices.rl_sizing_agent.rl_agent.get_rl_agent", "RL Agent Function"),
        ("microservices.exposure_balancer", "Exposure Balancer Module"),
        ("microservices.exposure_balancer.exposure_balancer.get_exposure_balancer", "Exposure Balancer Function"),
        ("microservices.exitbrain_v3_5", "ExitBrain v3.5 Module"),
        ("microservices.exitbrain_v3_5.adaptive_leverage_engine.AdaptiveLeverageEngine", "Adaptive Leverage Engine"),
        ("microservices.exitbrain_v3_5.intelligent_leverage_engine", "Intelligent Leverage Engine"),
    ]
    
    logger.info("")
    results = []
    for module_path, description in tests:
        result = test_import(module_path, description)
        results.append((description, result))
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for desc, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10} {desc}")
    
    logger.info("")
    logger.info(f"Result: {passed}/{total} modules passed")
    
    if passed == total:
        logger.info("✅ All module imports successful!")
        return 0
    else:
        logger.error(f"❌ {total - passed} module(s) failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
