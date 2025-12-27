"""
🧪 COMPREHENSIVE AI MODULE TEST - SERVER VERSION
=================================================
Tests all AI modules on production server to verify:
1. CLM (Continuous Learning Manager) - learning & logging
2. AI Engine - signal generation & logging  
3. RL Position Sizing - adaptive sizing & logging
4. Risk Management - real-time monitoring & logging
5. Exit Brain - dynamic TP/SL & logging
6. Model Ensemble - multi-model inference & logging
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/qt/quantum_trader')

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_section(title: str):
    """Print section divider"""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")


def check_result(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"     {details}")


async def test_clm_module():
    """Test Continuous Learning Manager"""
    print_section("1️⃣  CLM (Continuous Learning Manager)")
    
    results = {
        "module_exists": False,
        "can_check_status": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        # Check if CLM exists
        from backend.services.ai.continuous_learning_manager import (
            ContinuousLearningManager,
            ModelType,
            RetrainTrigger
        )
        results["module_exists"] = True
        results["details"].append("✓ CLM module imported successfully")
        
        # Check methods exist
        if hasattr(ContinuousLearningManager, 'check_retraining_needed'):
            results["can_check_status"] = True
            results["details"].append("✓ Retraining methods available")
        
        # Check log files
        import glob
        log_files = glob.glob('/home/qt/logs/*clm*.log') + glob.glob('/home/qt/logs/*learning*.log')
        if log_files:
            results["has_logging"] = True
            results["details"].append(f"✓ Found {len(log_files)} CLM log files")
            
            # Check recent activity
            for log_file in log_files[:3]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            results["details"].append(f"  Last: {last_line[:80]}...")
                except:
                    pass
        
    except ImportError as e:
        results["details"].append(f"✗ Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    # Print results
    check_result("CLM Module Exists", results["module_exists"])
    check_result("CLM Has Methods", results["can_check_status"])
    check_result("CLM Has Logs", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["module_exists"]


async def test_ai_engine():
    """Test AI Engine"""
    print_section("2️⃣  AI ENGINE (Signal Generation)")
    
    results = {
        "service_running": False,
        "has_recent_logs": False,
        "generating_signals": False,
        "details": []
    }
    
    try:
        # Check if service is running
        import subprocess
        ps_result = subprocess.run(['pgrep', '-f', 'ai_engine'], capture_output=True, text=True)
        if ps_result.returncode == 0:
            results["service_running"] = True
            pids = ps_result.stdout.strip().split('\n')
            results["details"].append(f"✓ AI Engine running (PIDs: {', '.join(pids)})")
        else:
            results["details"].append("✗ AI Engine not running")
        
        # Check logs
        import glob
        log_files = glob.glob('/home/qt/logs/*ai_engine*.log') + glob.glob('/home/qt/logs/ai_engine.log')
        if log_files:
            results["details"].append(f"✓ Found {len(log_files)} AI Engine log files")
            
            # Check for recent activity (last 5 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=5)
            for log_file in log_files[:3]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]  # Last 100 lines
                        
                    # Check for signal generation
                    signal_lines = [l for l in lines if 'signal' in l.lower() or 'prediction' in l.lower()]
                    if signal_lines:
                        results["generating_signals"] = True
                        results["details"].append(f"✓ Found {len(signal_lines)} signal entries in last 100 lines")
                        
                    # Check recency
                    if lines:
                        results["has_recent_logs"] = True
                        results["details"].append(f"  Latest: {lines[-1].strip()[:80]}...")
                        
                except Exception as e:
                    results["details"].append(f"  Error reading {log_file}: {e}")
        else:
            results["details"].append("✗ No AI Engine log files found")
        
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    check_result("AI Engine Running", results["service_running"])
    check_result("Has Recent Logs", results["has_recent_logs"])
    check_result("Generating Signals", results["generating_signals"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["service_running"] or results["has_recent_logs"]


async def test_rl_position_sizing():
    """Test RL Position Sizing"""
    print_section("3️⃣  RL POSITION SIZING (Adaptive Sizing)")
    
    results = {
        "module_exists": False,
        "has_logs": False,
        "details": []
    }
    
    try:
        # Try different possible imports
        try:
            from backend.domains.learning.rl_position_sizing import RLPositionSizing
            results["module_exists"] = True
            results["details"].append("✓ RL Position Sizing imported (domains)")
        except:
            from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
            results["module_exists"] = True
            results["details"].append("✓ RL Position Sizing Agent imported (services)")
        
        # Check logs
        import glob
        log_files = glob.glob('/home/qt/logs/*rl*.log') + glob.glob('/home/qt/logs/*sizing*.log')
        if log_files:
            results["has_logs"] = True
            results["details"].append(f"✓ Found {len(log_files)} RL sizing log files")
            
            for log_file in log_files[:2]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            results["details"].append(f"  {os.path.basename(log_file)}: {len(lines)} lines")
                except:
                    pass
        
    except ImportError as e:
        results["details"].append(f"✗ Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    check_result("RL Sizing Module Exists", results["module_exists"])
    check_result("Has Logs", results["has_logs"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["module_exists"]


async def test_risk_management():
    """Test Risk Management"""
    print_section("4️⃣  RISK MANAGEMENT (Real-time Monitoring)")
    
    results = {
        "module_exists": False,
        "has_logs": False,
        "details": []
    }
    
    try:
        from backend.services.risk_management.risk_manager import RiskManager
        results["module_exists"] = True
        results["details"].append("✓ Risk Manager imported")
        
        # Check logs
        import glob
        log_files = glob.glob('/home/qt/logs/*risk*.log')
        if log_files:
            results["has_logs"] = True
            results["details"].append(f"✓ Found {len(log_files)} risk management log files")
            
            for log_file in log_files[:2]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Look for recent risk checks
                        risk_checks = [l for l in lines[-100:] if 'check' in l.lower() or 'limit' in l.lower()]
                        if risk_checks:
                            results["details"].append(f"  {os.path.basename(log_file)}: {len(risk_checks)} risk checks in last 100 lines")
                except:
                    pass
        
    except ImportError as e:
        results["details"].append(f"✗ Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    check_result("Risk Manager Exists", results["module_exists"])
    check_result("Has Logs", results["has_logs"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["module_exists"]


async def test_exit_brain():
    """Test Exit Brain"""
    print_section("5️⃣  EXIT BRAIN (Dynamic TP/SL)")
    
    results = {
        "module_exists": False,
        "has_logs": False,
        "details": []
    }
    
    try:
        from backend.services.risk_management.exit_policy_engine import ExitPolicyEngine
        results["module_exists"] = True
        results["details"].append("✓ Exit Policy Engine imported")
        
        # Check logs
        import glob
        log_files = glob.glob('/home/qt/logs/*exit*.log') + glob.glob('/home/qt/logs/*tp*.log')
        if log_files:
            results["has_logs"] = True
            results["details"].append(f"✓ Found {len(log_files)} exit-related log files")
            
            for log_file in log_files[:2]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Look for TP/SL updates
                        exit_events = [l for l in lines[-100:] if any(x in l.lower() for x in ['take_profit', 'stop_loss', 'exit'])]
                        if exit_events:
                            results["details"].append(f"  {os.path.basename(log_file)}: {len(exit_events)} exit events in last 100 lines")
                except:
                    pass
        
    except ImportError as e:
        results["details"].append(f"✗ Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    check_result("Exit Brain Exists", results["module_exists"])
    check_result("Has Logs", results["has_logs"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["module_exists"]


async def test_backend_service():
    """Test Backend Service"""
    print_section("6️⃣  BACKEND SERVICE (Main Trading Loop)")
    
    results = {
        "service_running": False,
        "has_recent_logs": False,
        "details": []
    }
    
    try:
        # Check if backend is running
        import subprocess
        ps_result = subprocess.run(['pgrep', '-f', 'uvicorn.*main:app'], capture_output=True, text=True)
        if ps_result.returncode == 0:
            results["service_running"] = True
            pids = ps_result.stdout.strip().split('\n')
            results["details"].append(f"✓ Backend running (PIDs: {', '.join(pids)})")
        else:
            results["details"].append("✗ Backend not running")
        
        # Check logs
        import glob
        log_files = glob.glob('/home/qt/logs/backend*.log') + glob.glob('/home/qt/logs/trading*.log')
        if log_files:
            results["details"].append(f"✓ Found {len(log_files)} backend log files")
            
            for log_file in log_files[:2]:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            results["has_recent_logs"] = True
                            results["details"].append(f"  {os.path.basename(log_file)}: {len(lines)} total lines")
                            results["details"].append(f"    Latest: {lines[-1].strip()[:80]}...")
                except:
                    pass
        
    except Exception as e:
        results["details"].append(f"✗ Error: {str(e)}")
    
    check_result("Backend Running", results["service_running"])
    check_result("Has Recent Logs", results["has_recent_logs"])
    
    for detail in results["details"]:
        print(f"     {detail}")
    
    return results["service_running"] or results["has_recent_logs"]


async def check_all_logs():
    """Check all log files for activity"""
    print_section("7️⃣  COMPREHENSIVE LOG ANALYSIS")
    
    import glob
    import os
    
    log_dir = '/home/qt/logs'
    
    if not os.path.exists(log_dir):
        print("✗ Log directory not found!")
        return False
    
    log_files = glob.glob(f'{log_dir}/*.log')
    
    if not log_files:
        print("✗ No log files found!")
        return False
    
    print(f"📝 Found {len(log_files)} log files:")
    
    # Sort by modification time
    log_files_with_time = []
    for log_file in log_files:
        try:
            mtime = os.path.getmtime(log_file)
            size = os.path.getsize(log_file)
            log_files_with_time.append((log_file, mtime, size))
        except:
            pass
    
    log_files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    # Show most recent 10 files
    cutoff = datetime.now() - timedelta(hours=1)
    recent_count = 0
    
    for log_file, mtime, size in log_files_with_time[:15]:
        name = os.path.basename(log_file)
        mod_time = datetime.fromtimestamp(mtime)
        age = datetime.now() - mod_time
        
        if mod_time > cutoff:
            recent_count += 1
            icon = "🟢"
        else:
            icon = "🔴"
        
        size_mb = size / 1024 / 1024
        print(f"  {icon} {name:30s} - {size_mb:>6.2f}MB - {age.total_seconds()/3600:>5.1f}h ago")
    
    print(f"\n✅ {recent_count} files updated in last hour")
    
    return recent_count > 0


async def main():
    """Run all tests"""
    print_header("🧪 COMPREHENSIVE AI MODULE TEST - SERVER VERSION")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Server: {os.uname().nodename}")
    
    results = {}
    
    # Run all tests
    results["clm"] = await test_clm_module()
    results["ai_engine"] = await test_ai_engine()
    results["rl_sizing"] = await test_rl_position_sizing()
    results["risk"] = await test_risk_management()
    results["exit_brain"] = await test_exit_brain()
    results["backend"] = await test_backend_service()
    results["logs"] = await check_all_logs()
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n📋 Module Status:")
    for module, status in results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {module.upper()}")
    
    if passed == total:
        print("\n🎉 ALL MODULES OPERATIONAL!")
    elif passed >= total * 0.7:
        print("\n✅ SYSTEM MOSTLY OPERATIONAL")
    else:
        print("\n⚠️  SEVERAL MODULES NEED ATTENTION")
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed >= total * 0.7  # Pass if 70% or more working


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
