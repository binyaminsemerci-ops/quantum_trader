#!/usr/bin/env python3
"""
Core Loop Validation Script
============================
Validates that the Tier 1 Core Execution Loop is working correctly.

Checks:
1. All services running (systemd status)
2. Redis topics have messages
3. Approval rate is reasonable (20-50%)
4. Average confidence above threshold
5. End-to-end flow working

Usage:
    python3 ops/validate_core_loop.py

Author: Quantum Trader Team
Date: 2026-01-12
"""
import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.services.eventbus_bridge import EventBusClient, get_recent_signals


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_service_status(service_name: str) -> bool:
    """Check if systemd service is active"""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", f"quantum-{service_name}.service"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️  Error checking {service_name}: {e}")
        return False


def check_port_listening(port: int) -> bool:
    """Check if port is listening"""
    try:
        result = subprocess.run(
            ["netstat", "-tuln"],
            capture_output=True,
            text=True
        )
        return f":{port} " in result.stdout
    except Exception:
        # Fallback: try curl
        try:
            subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True,
                timeout=2
            )
            return True
        except Exception:
            return False


async def check_redis_topics() -> Dict[str, int]:
    """Check Redis topic message counts"""
    topics = {}
    
    try:
        async with EventBusClient() as bus:
            for topic in [
                "trade.signal.v5",
                "trade.signal.safe",
                "trade.execution.res",
                "trade.position.update"
            ]:
                length = await bus.get_stream_length(topic)
                topics[topic] = length
    except Exception as e:
        print(f"⚠️  Error checking Redis: {e}")
    
    return topics


async def analyze_signal_flow() -> Dict[str, Any]:
    """Analyze signal flow and approval rate"""
    analysis = {
        "signals_generated": 0,
        "signals_approved": 0,
        "approval_rate": 0.0,
        "avg_confidence": 0.0,
        "signal_variety": set(),
        "time_to_execution_avg": 0.0
    }
    
    try:
        # Get recent signals (last 100)
        signals = await get_recent_signals("trade.signal.v5", 100)
        approved = await get_recent_signals("trade.signal.safe", 100)
        executions = await get_recent_signals("trade.execution.res", 100)
        
        if signals:
            analysis["signals_generated"] = len(signals)
            analysis["signals_approved"] = len(approved)
            
            # Calculate approval rate
            if len(signals) > 0:
                analysis["approval_rate"] = len(approved) / len(signals)
            
            # Calculate average confidence
            confidences = [s.get("confidence", 0) for s in signals]
            if confidences:
                analysis["avg_confidence"] = sum(confidences) / len(confidences)
            
            # Get signal variety
            actions = [s.get("action", "UNKNOWN") for s in signals]
            analysis["signal_variety"] = set(actions)
            
            # Calculate time to execution (if we have both)
            if approved and executions:
                times = []
                for appr in approved[-10:]:  # Last 10
                    appr_time = datetime.fromisoformat(appr["timestamp"].replace("Z", ""))
                    # Find matching execution
                    for exe in executions:
                        exe_time = datetime.fromisoformat(exe["timestamp"].replace("Z", ""))
                        if exe["symbol"] == appr["symbol"]:
                            delta = (exe_time - appr_time).total_seconds()
                            if 0 <= delta <= 60:  # Within 60 seconds
                                times.append(delta)
                                break
                
                if times:
                    analysis["time_to_execution_avg"] = sum(times) / len(times)
    
    except Exception as e:
        print(f"⚠️  Error analyzing signals: {e}")
    
    return analysis


# ============================================================================
# MAIN VALIDATION
# ============================================================================

async def main():
    """Run all validations"""
    print("=" * 70)
    print("TIER 1 CORE LOOP VALIDATION")
    print("=" * 70)
    
    # Track results
    checks_passed = 0
    checks_total = 0
    
    # ========================================================================
    # CHECK 1: SERVICES
    # ========================================================================
    print("\n[1/5] Checking services...")
    print("-" * 70)
    
    services = {
        "risk-safety": 8003,
        "execution": 8002,
        "position-monitor": 8004
    }
    
    services_ok = 0
    for service_name, port in services.items():
        checks_total += 1
        
        is_active = check_service_status(service_name)
        is_listening = check_port_listening(port)
        
        if is_active and is_listening:
            print(f"✅ {service_name}: ACTIVE (port {port})")
            checks_passed += 1
            services_ok += 1
        elif is_active:
            print(f"⚠️  {service_name}: ACTIVE but port {port} not listening")
        else:
            print(f"❌ {service_name}: INACTIVE")
    
    print(f"\nServices OK: {services_ok}/{len(services)}")
    
    # ========================================================================
    # CHECK 2: REDIS TOPICS
    # ========================================================================
    print("\n[2/5] Checking Redis topics...")
    print("-" * 70)
    
    topics = await check_redis_topics()
    
    if topics:
        checks_total += 1
        
        for topic, length in topics.items():
            print(f"📊 {topic}: {length} messages")
        
        # Check if we have messages flowing
        if topics.get("trade.signal.v5", 0) > 0:
            checks_passed += 1
            print("\n✅ Redis topics have messages")
        else:
            print("\n⚠️  No signals in Redis (system may be starting)")
    else:
        print("❌ Could not connect to Redis")
    
    # ========================================================================
    # CHECK 3: SIGNAL FLOW
    # ========================================================================
    print("\n[3/5] Analyzing signal flow...")
    print("-" * 70)
    
    analysis = await analyze_signal_flow()
    
    if analysis["signals_generated"] > 0:
        checks_total += 3
        
        print(f"Signals generated: {analysis['signals_generated']}")
        print(f"Signals approved:  {analysis['signals_approved']}")
        print(f"Approval rate:     {analysis['approval_rate']*100:.1f}%")
        print(f"Avg confidence:    {analysis['avg_confidence']:.3f}")
        print(f"Signal variety:    {analysis['signal_variety']}")
        
        if analysis["time_to_execution_avg"] > 0:
            print(f"Avg time to exec:  {analysis['time_to_execution_avg']:.2f}s")
        
        # Check approval rate (should be 20-50%)
        if 0.20 <= analysis["approval_rate"] <= 0.50:
            print("\n✅ Approval rate OK (20-50%)")
            checks_passed += 1
        elif analysis["approval_rate"] < 0.20:
            print("\n⚠️  Approval rate LOW (<20%) - risk controls may be too strict")
        else:
            print("\n⚠️  Approval rate HIGH (>50%) - risk controls may be too loose")
        
        # Check average confidence
        if analysis["avg_confidence"] >= 0.65:
            print("✅ Average confidence OK (≥0.65)")
            checks_passed += 1
        else:
            print(f"⚠️  Average confidence LOW (<0.65)")
        
        # Check signal variety (should have BUY and HOLD at minimum)
        if len(analysis["signal_variety"]) >= 2:
            print("✅ Signal variety OK (multiple actions)")
            checks_passed += 1
        else:
            print(f"⚠️  Signal variety LOW (only {analysis['signal_variety']})")
    else:
        print("⚠️  No signals to analyze yet")
    
    # ========================================================================
    # CHECK 4: END-TO-END FLOW
    # ========================================================================
    print("\n[4/5] Checking end-to-end flow...")
    print("-" * 70)
    
    checks_total += 1
    
    if topics:
        has_signals = topics.get("trade.signal.v5", 0) > 0
        has_approved = topics.get("trade.signal.safe", 0) > 0
        has_executions = topics.get("trade.execution.res", 0) > 0
        has_positions = topics.get("trade.position.update", 0) > 0
        
        flow_complete = has_signals and has_approved and has_executions
        
        print(f"{'✅' if has_signals else '❌'} Signals generated")
        print(f"{'✅' if has_approved else '❌'} Signals approved")
        print(f"{'✅' if has_executions else '❌'} Orders executed")
        print(f"{'✅' if has_positions else '⚠️ '} Positions tracked")
        
        if flow_complete:
            print("\n✅ End-to-end flow working")
            checks_passed += 1
        else:
            print("\n⚠️  End-to-end flow incomplete")
    else:
        print("❌ Cannot check flow (Redis unavailable)")
    
    # ========================================================================
    # CHECK 5: HEALTH ENDPOINTS
    # ========================================================================
    print("\n[5/5] Testing health endpoints...")
    print("-" * 70)
    
    for service_name, port in services.items():
        checks_total += 1
        
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "status" in result.stdout:
                print(f"✅ {service_name}: /health OK")
                checks_passed += 1
            else:
                print(f"⚠️  {service_name}: /health returned error")
        except Exception as e:
            print(f"❌ {service_name}: /health unreachable")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    success_rate = (checks_passed / checks_total * 100) if checks_total > 0 else 0
    
    print(f"\nChecks passed: {checks_passed}/{checks_total} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("\n✅ CORE LOOP OK ✅")
        print("\nTier 1 is fully operational. Ready for production testing.")
        return 0
    elif success_rate >= 70:
        print("\n⚠️  CORE LOOP DEGRADED ⚠️")
        print("\nSome components need attention. Check logs for details.")
        return 1
    else:
        print("\n❌ CORE LOOP FAILED ❌")
        print("\nMultiple components are not working. Check deployment.")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
