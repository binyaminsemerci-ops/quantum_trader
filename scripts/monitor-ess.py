#!/usr/bin/env python3
"""
ESS Monitoring Dashboard

Real-time monitoring of Emergency Stop System status, near-misses, and activation history.

Usage:
    python scripts/monitor-ess.py [--watch] [--interval SECONDS]

Options:
    --watch         Continuous monitoring mode (refreshes every interval)
    --interval      Refresh interval in seconds (default: 5)
    --history       Number of historical activations to show (default: 10)

Author: Quantum Trader Team
Date: 2024-11-30
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp to readable string."""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_str


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


async def get_ess_status(base_url: str) -> dict[str, Any]:
    """Fetch ESS status from API."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/api/emergency/status", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}", "available": False}
        except aiohttp.ClientError as e:
            return {"error": str(e), "available": False}
        except asyncio.TimeoutError:
            return {"error": "Request timeout", "available": False}


async def get_health(base_url: str) -> dict[str, Any]:
    """Fetch backend health status."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}"}
        except aiohttp.ClientError as e:
            return {"error": str(e)}
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}


async def get_policy_store(base_url: str) -> dict[str, Any]:
    """Fetch PolicyStore configuration."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/api/policy", timeout=5) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}"}
        except aiohttp.ClientError as e:
            return {"error": str(e)}
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}


def load_activation_log(limit: int = 10) -> list[dict]:
    """Load ESS activation history from log file."""
    log_path = Path("data/ess_activations.log")
    
    if not log_path.exists():
        return []
    
    activations = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    activations.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"{Colors.WARNING}Warning: Failed to read activation log: {e}{Colors.ENDC}")
    
    return activations[-limit:]  # Return last N entries


def print_header():
    """Print dashboard header."""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print("  EMERGENCY STOP SYSTEM (ESS) - MONITORING DASHBOARD")
    print("=" * 80)
    print(f"{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()


def print_ess_status(status: dict[str, Any]):
    """Print ESS status section."""
    print(f"{Colors.BOLD}ESS STATUS{Colors.ENDC}")
    print("-" * 80)
    
    if "error" in status:
        print(f"{Colors.FAIL}❌ Error: {status['error']}{Colors.ENDC}")
        return
    
    if not status.get("available", False):
        print(f"{Colors.FAIL}❌ ESS not available{Colors.ENDC}")
        if "message" in status:
            print(f"   {status['message']}")
        return
    
    is_active = status.get("active", False)
    ess_status = status.get("status", "UNKNOWN")
    
    if is_active:
        print(f"{Colors.FAIL}🚨 STATUS: {ess_status} (ACTIVE){Colors.ENDC}")
        print(f"{Colors.FAIL}   Trading is HALTED{Colors.ENDC}")
        print(f"   Reason: {status.get('reason', 'Unknown')}")
        print(f"   Triggered by: {status.get('triggered_by', 'Unknown')}")
        print(f"   Timestamp: {format_timestamp(status.get('timestamp', ''))}")
        
        # Calculate duration
        try:
            ts = datetime.fromisoformat(status.get('timestamp', '').replace('Z', '+00:00'))
            duration = (datetime.now(ts.tzinfo) - ts).total_seconds()
            print(f"   Duration: {format_duration(duration)}")
        except:
            pass
        
        # Show details
        details = status.get("details", {})
        if details:
            print(f"   Details:")
            for key, value in details.items():
                print(f"      - {key}: {value}")
    else:
        print(f"{Colors.OKGREEN}✅ STATUS: {ess_status} (INACTIVE){Colors.ENDC}")
        print(f"{Colors.OKGREEN}   Trading is ENABLED{Colors.ENDC}")
    
    print()


def print_health_status(health: dict[str, Any]):
    """Print backend health section."""
    print(f"{Colors.BOLD}BACKEND HEALTH{Colors.ENDC}")
    print("-" * 80)
    
    if "error" in health:
        print(f"{Colors.FAIL}❌ Error: {health['error']}{Colors.ENDC}")
        return
    
    status = health.get("status", "unknown")
    print(f"Status: {Colors.OKGREEN if status == 'healthy' else Colors.WARNING}{status.upper()}{Colors.ENDC}")
    
    if "event_driven_active" in health:
        print(f"Event-driven mode: {'✅ Active' if health['event_driven_active'] else '❌ Inactive'}")
    
    # Risk snapshot
    risk = health.get("risk", {})
    if risk and isinstance(risk, dict) and "error" not in risk:
        print(f"Risk Guard: ✅ Operational")
        if "circuit_breaker" in risk:
            cb = risk["circuit_breaker"]
            print(f"   Circuit Breaker: {'🔴 TRIPPED' if cb.get('is_tripped') else '✅ OK'}")
    
    print()


def print_thresholds(policy: dict[str, Any]):
    """Print ESS threshold configuration."""
    print(f"{Colors.BOLD}ESS THRESHOLDS{Colors.ENDC}")
    print("-" * 80)
    
    if "error" in policy:
        print(f"{Colors.WARNING}⚠️  PolicyStore unavailable{Colors.ENDC}")
        return
    
    # Try to extract ESS thresholds from policy or environment
    import os
    
    thresholds = {
        "Max Daily Loss": f"{os.getenv('QT_ESS_MAX_DAILY_LOSS', '10.0')}%",
        "Max Equity DD": f"{os.getenv('QT_ESS_MAX_EQUITY_DD', '15.0')}%",
        "Max SL Hits": f"{os.getenv('QT_ESS_MAX_SL_HITS', '5')} in {os.getenv('QT_ESS_ERROR_WINDOW_MINUTES', '60')}m",
        "Max Stale Data": f"{os.getenv('QT_ESS_MAX_STALE_SECONDS', '300')}s",
        "Check Interval": f"{os.getenv('QT_ESS_CHECK_INTERVAL', '5')}s",
    }
    
    for name, value in thresholds.items():
        print(f"   {name}: {value}")
    
    print()


def print_activation_history(history: list[dict], limit: int = 10):
    """Print ESS activation history."""
    print(f"{Colors.BOLD}ACTIVATION HISTORY (Last {limit}){Colors.ENDC}")
    print("-" * 80)
    
    if not history:
        print(f"{Colors.OKGREEN}✅ No activations recorded{Colors.ENDC}")
        print()
        return
    
    print(f"{Colors.WARNING}⚠️  {len(history)} activation(s) found{Colors.ENDC}")
    print()
    
    for i, entry in enumerate(reversed(history[-limit:]), 1):
        event_type = entry.get("event", "unknown")
        timestamp = format_timestamp(entry.get("timestamp", ""))
        
        if event_type == "activation":
            print(f"{Colors.FAIL}{i}. 🚨 ACTIVATION{Colors.ENDC}")
            print(f"   Time: {timestamp}")
            print(f"   Reason: {entry.get('reason', 'Unknown')}")
            print(f"   Triggered by: {entry.get('triggered_by', 'Unknown')}")
            
            details = entry.get("details", {})
            if details:
                for key, value in details.items():
                    print(f"      - {key}: {value}")
        
        elif event_type == "reset":
            print(f"{Colors.OKGREEN}{i}. ✅ RESET{Colors.ENDC}")
            print(f"   Time: {timestamp}")
            print(f"   Reset by: {entry.get('reset_by', 'Unknown')}")
            duration = entry.get("duration_seconds", 0)
            print(f"   Duration: {format_duration(duration)}")
        
        print()


async def monitor_once(base_url: str, history_limit: int):
    """Run monitoring once and print dashboard."""
    clear_screen()
    print_header()
    
    # Fetch data concurrently
    ess_status_task = get_ess_status(base_url)
    health_task = get_health(base_url)
    policy_task = get_policy_store(base_url)
    
    ess_status, health, policy = await asyncio.gather(
        ess_status_task, health_task, policy_task
    )
    
    # Print sections
    print_ess_status(ess_status)
    print_health_status(health)
    print_thresholds(policy)
    
    # Load and print history
    history = load_activation_log(limit=history_limit)
    print_activation_history(history, limit=history_limit)
    
    print(f"{Colors.OKCYAN}Press Ctrl+C to exit{Colors.ENDC}")


async def monitor_continuous(base_url: str, interval: int, history_limit: int):
    """Run monitoring in continuous mode."""
    try:
        while True:
            await monitor_once(base_url, history_limit)
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{Colors.OKCYAN}Monitoring stopped{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="ESS Monitoring Dashboard - Real-time status and activation history"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous monitoring mode (refreshes every interval)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of historical activations to show (default: 10)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.watch:
            asyncio.run(monitor_continuous(args.url, args.interval, args.history))
        else:
            asyncio.run(monitor_once(args.url, args.history))
    except KeyboardInterrupt:
        print(f"\n{Colors.OKCYAN}Exiting...{Colors.ENDC}")
        sys.exit(0)


if __name__ == "__main__":
    main()
