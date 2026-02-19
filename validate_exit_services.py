#!/usr/bin/env python3
"""
VPS Exit Monitor Validator
=========================

Sjekker om VPS exit monitor servicen faktisk bruker den nye exit-formelen
eller den gamle hardkodede prosentene.

Author: Exit Validation Team  
Date: 2026-02-18
"""

import os
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

def check_local_exit_services():
    """Sjekk hvilke exit-relaterte servicer som finnes lokalt"""
    
    logger.info("🔍 CHECKING LOCAL EXIT SERVICES")
    logger.info("="*50)
    
    # Find exit service files
    exit_files = [
        "exit_monitor_service_patched.py",
        "quantum_exit_monitor.py", 
        "exit_monitor.py",
        "check_exit_brain_executor_status.py"
    ]
    
    found_files = []
    for filename in exit_files:
        for path in Path(".").rglob(filename):
            found_files.append(str(path))
            logger.info(f"📄 Found: {path}")
    
    return found_files

def analyze_exit_code(filepath):
    """Analyser exit service kode for å se om den bruker ny eller gammel logikk"""
    
    logger.info(f"\n🔬 ANALYZING: {filepath}")
    logger.info("="*50)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new formula usage
        new_formula_indicators = [
            "from common.exit_math import", 
            "compute_dynamic_stop",
            "evaluate_exit", 
            "RiskSettings",
            "Position(",
            "Account(",
            "Market("
        ]
        
        # Check for old hardcoded percentages  
        old_formula_indicators = [
            "* 1.025",  # +2.5%
            "* 0.985",  # -1.5%
            "* 1.015",  # +1.5%
            "* 0.975",  # -2.5%
            "TRAILING_STOP_PCT = 0.015"  # 1.5%
        ]
        
        new_formula_count = sum(1 for indicator in new_formula_indicators if indicator in content)
        old_formula_count = sum(1 for indicator in old_formula_indicators if indicator in content)
        
        logger.info(f"📊 NEW FORMULA indicators: {new_formula_count}")
        for indicator in new_formula_indicators:
            if indicator in content:
                logger.info(f"  ✅ {indicator}")
        
        logger.info(f"📊 OLD FORMULA indicators: {old_formula_count}")  
        for indicator in old_formula_indicators:
            if indicator in content:
                logger.info(f"  ⚠️  {indicator}")
        
        # Verdict
        if new_formula_count >= 3:
            logger.info(f"🟢 VERDICT: Uses NEW dynamic formula")
            return "NEW"
        elif old_formula_count >= 2:
            logger.info(f"🔴 VERDICT: Uses OLD hardcoded percentages")
            return "OLD"
        else:
            logger.info(f"🟡 VERDICT: Mixed or unclear")
            return "MIXED"
            
    except Exception as e:
        logger.error(f"❌ Error analyzing {filepath}: {e}")
        return "ERROR"

def check_service_status():
    """Sjekk status på exit-relaterte Windows services"""
    
    logger.info(f"\n🔧 CHECKING WINDOWS SERVICES")
    logger.info("="*50)
    
    service_names = [
        "quantum-exit-monitor",
        "ExitBrain",
        "QuantumTrader"
    ]
    
    for service in service_names:
        try:
            result = subprocess.run(
                ["sc", "query", service], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                if "RUNNING" in result.stdout:
                    logger.info(f"🟢 {service}: RUNNING")
                else:
                    logger.info(f"🟡 {service}: NOT RUNNING")
            else:
                logger.info(f"⚫ {service}: NOT FOUND")
                
        except Exception as e:
            logger.error(f"❌ Error checking {service}: {e}")

def main():
    """Main validation function"""
    
    logger.info("🚀 VPS EXIT MONITOR VALIDATION")
    logger.info("="*60)
    
    # Check local exit services
    exit_files = check_local_exit_services()
    
    if not exit_files:
        logger.warning("⚠️  No exit service files found locally")
        return
    
    # Analyze each found file
    analysis_results = {}
    for filepath in exit_files[:3]:  # Limit to first 3 files
        result = analyze_exit_code(filepath)
        analysis_results[filepath] = result
    
    # Check service status
    check_service_status()
    
    # Summary
    logger.info(f"\n📈 VALIDATION SUMMARY")
    logger.info("="*60)
    
    new_count = len([r for r in analysis_results.values() if r == "NEW"])
    old_count = len([r for r in analysis_results.values() if r == "OLD"])
    
    logger.info(f"Files using NEW formula: {new_count}")
    logger.info(f"Files using OLD formula: {old_count}")
    
    if new_count > old_count:
        logger.info(f"🟢 CONCLUSION: System primarily uses NEW dynamic exit formula")
    elif old_count > new_count:
        logger.info(f"🔴 CONCLUSION: System primarily uses OLD hardcoded percentages")
    else:
        logger.info(f"🟡 CONCLUSION: Mixed implementation - needs investigation")
    
    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    if old_count > 0:
        logger.info(f"1. Update services using old formula to use common.exit_math")
        logger.info(f"2. Verify VPS deployment uses latest exit_monitor_service_patched.py")
        logger.info(f"3. Check systemd service files point to correct script")

if __name__ == "__main__":
    main()