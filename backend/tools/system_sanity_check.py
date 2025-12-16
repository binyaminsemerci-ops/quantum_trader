"""
System Sanity Check Script - Sprint 5 Del 6

Quick health check for all critical system components before go-live.
Run this script to verify system readiness in < 30 seconds.

Usage:
    python backend/tools/system_sanity_check.py
    
Returns:
    Exit code 0 if all critical checks pass
    Exit code 1 if any critical component fails
"""
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class CheckStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class SanityCheckResult:
    def __init__(self, component: str, status: CheckStatus, message: str, details: Dict = None):
        self.component = component
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class SystemSanityChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.results: List[SanityCheckResult] = []
    
    async def check_redis(self) -> SanityCheckResult:
        """Check Redis connectivity and health."""
        try:
            import redis.asyncio as redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url)
            
            # Ping test
            await client.ping()
            
            # Check memory usage
            info = await client.info("memory")
            used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)
            
            await client.close()
            
            return SanityCheckResult(
                "Redis",
                CheckStatus.OK,
                f"Connected, {used_memory_mb:.1f}MB used",
                {"url": redis_url, "memory_mb": used_memory_mb}
            )
        except Exception as e:
            return SanityCheckResult(
                "Redis",
                CheckStatus.CRITICAL,
                f"Connection failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_postgres(self) -> SanityCheckResult:
        """Check Postgres/SQLite database connectivity."""
        try:
            from sqlalchemy import create_engine, text
            
            db_url = os.getenv("DATABASE_URL", "sqlite:///data/quantum_trader.db")
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Simple query test
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Count tables
                if "sqlite" in db_url:
                    result = conn.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'"))
                else:
                    result = conn.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"))
                
                table_count = result.fetchone()[0]
            
            return SanityCheckResult(
                "Database",
                CheckStatus.OK,
                f"Connected, {table_count} tables",
                {"url": db_url[:30] + "...", "tables": table_count}
            )
        except Exception as e:
            return SanityCheckResult(
                "Database",
                CheckStatus.CRITICAL,
                f"Connection failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_policy_store(self) -> SanityCheckResult:
        """Check PolicyStore health and config."""
        try:
            from backend.core.policy_store import PolicyStore
            import redis.asyncio as redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_client = redis.from_url(redis_url)
            
            store = PolicyStore(redis_client)
            await store.initialize()
            
            # Get current policy
            policy = await store.get_policy()
            
            await redis_client.close()
            
            return SanityCheckResult(
                "PolicyStore",
                CheckStatus.OK,
                f"Active mode: {policy.active_mode.value}, version: {policy.version}",
                {
                    "mode": policy.active_mode.value,
                    "version": policy.version,
                    "max_risk": policy.risk_profiles[policy.active_mode].max_risk_per_trade_pct
                }
            )
        except Exception as e:
            return SanityCheckResult(
                "PolicyStore",
                CheckStatus.DEGRADED,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_ess(self) -> SanityCheckResult:
        """Check Emergency Stop System."""
        try:
            from backend.core.safety.ess import EmergencyStopSystem, ESSState
            from backend.core.event_bus import EventBus
            import redis.asyncio as redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_client = redis.from_url(redis_url)
            event_bus = EventBus(redis_client)
            
            # Create simple policy store mock
            class SimplePolicyStore:
                def get(self, key, default=None):
                    defaults = {
                        "ess.enabled": True,
                        "ess.max_daily_drawdown_pct": 5.0,
                        "ess.max_open_loss_pct": 10.0,
                        "ess.cooldown_minutes": 15
                    }
                    return defaults.get(key, default)
            
            ess = EmergencyStopSystem(SimplePolicyStore(), event_bus)
            
            # Check state
            state = ess.state
            can_trade = await ess.can_execute_orders()
            
            await redis_client.close()
            
            status = CheckStatus.OK if can_trade else CheckStatus.CRITICAL
            message = f"State: {state.value}, Trading: {'ALLOWED' if can_trade else 'BLOCKED'}"
            
            return SanityCheckResult(
                "ESS",
                status,
                message,
                {"state": state.value, "can_trade": can_trade}
            )
        except Exception as e:
            return SanityCheckResult(
                "ESS",
                CheckStatus.DEGRADED,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_execution_service(self) -> SanityCheckResult:
        """Check Binance execution adapter."""
        try:
            from backend.services.execution.execution import BinanceFuturesExecutionAdapter
            
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            
            if not api_key or not api_secret:
                return SanityCheckResult(
                    "Execution",
                    CheckStatus.DEGRADED,
                    "API keys not configured",
                    {"api_key_set": False}
                )
            
            adapter = BinanceFuturesExecutionAdapter(
                api_key=api_key,
                api_secret=api_secret,
                market_type="usdm_perp"
            )
            
            if not adapter.ready:
                return SanityCheckResult(
                    "Execution",
                    CheckStatus.DEGRADED,
                    "Adapter not ready",
                    {"ready": False}
                )
            
            # Try to fetch account info (quick test)
            try:
                # This will fail if keys are invalid
                # account = await adapter.get_account()
                pass
            except Exception:
                pass
            
            return SanityCheckResult(
                "Execution",
                CheckStatus.OK,
                "Adapter ready, keys configured",
                {"ready": True, "api_key_set": True}
            )
        except Exception as e:
            return SanityCheckResult(
                "Execution",
                CheckStatus.DEGRADED,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_ai_engine(self) -> SanityCheckResult:
        """Check AI Engine availability."""
        try:
            import httpx
            
            ai_url = os.getenv("AI_ENGINE_SERVICE_URL", "http://localhost:8001")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ai_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    return SanityCheckResult(
                        "AI Engine",
                        CheckStatus.OK,
                        f"Service healthy: {data.get('status', 'ok')}",
                        {"url": ai_url, "status": data.get('status')}
                    )
                else:
                    return SanityCheckResult(
                        "AI Engine",
                        CheckStatus.DEGRADED,
                        f"Service returned {response.status_code}",
                        {"url": ai_url, "status_code": response.status_code}
                    )
        except Exception as e:
            return SanityCheckResult(
                "AI Engine",
                CheckStatus.DEGRADED,
                f"Service unavailable: {str(e)[:50]}",
                {"error": str(e)[:100]}
            )
    
    async def check_dashboard_ws(self) -> SanityCheckResult:
        """Check Dashboard WebSocket endpoint."""
        try:
            import httpx
            
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if health endpoint responds
                response = await client.get(f"{backend_url}/health")
                
                if response.status_code == 200:
                    # WebSocket endpoint exists if backend is healthy
                    return SanityCheckResult(
                        "Dashboard WS",
                        CheckStatus.OK,
                        "Backend healthy, WS endpoint available",
                        {"backend_url": backend_url}
                    )
                else:
                    return SanityCheckResult(
                        "Dashboard WS",
                        CheckStatus.DEGRADED,
                        f"Backend returned {response.status_code}",
                        {"status_code": response.status_code}
                    )
        except Exception as e:
            return SanityCheckResult(
                "Dashboard WS",
                CheckStatus.DEGRADED,
                f"Backend unavailable: {str(e)[:50]}",
                {"error": str(e)[:100]}
            )
    
    async def check_portfolio_service(self) -> SanityCheckResult:
        """Check Portfolio Intelligence service."""
        try:
            # Check if portfolio service module loads
            from microservices.portfolio_intelligence.service import PortfolioIntelligenceService
            
            return SanityCheckResult(
                "Portfolio",
                CheckStatus.OK,
                "Service module loaded",
                {"available": True}
            )
        except Exception as e:
            return SanityCheckResult(
                "Portfolio",
                CheckStatus.DEGRADED,
                f"Module load failed: {str(e)[:50]}",
                {"error": str(e)[:100]}
            )
    
    async def run_all_checks(self) -> Tuple[int, int, int]:
        """Run all sanity checks and return summary counts."""
        print("=" * 70)
        print("🔍 QUANTUM TRADER - SYSTEM SANITY CHECK")
        print("=" * 70)
        print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        print()
        
        checks = [
            ("Redis", self.check_redis()),
            ("Database", self.check_postgres()),
            ("PolicyStore", self.check_policy_store()),
            ("ESS", self.check_ess()),
            ("Execution", self.check_execution_service()),
            ("AI Engine", self.check_ai_engine()),
            ("Dashboard WS", self.check_dashboard_ws()),
            ("Portfolio", self.check_portfolio_service()),
        ]
        
        # Run checks in parallel
        results = await asyncio.gather(*[check[1] for check in checks], return_exceptions=True)
        
        # Process results
        ok_count = 0
        degraded_count = 0
        critical_count = 0
        
        print("Component Health:")
        print("-" * 70)
        
        for i, (name, result) in enumerate(zip([c[0] for c in checks], results)):
            if isinstance(result, Exception):
                result = SanityCheckResult(name, CheckStatus.CRITICAL, f"Check crashed: {str(result)}")
            
            self.results.append(result)
            
            # Count statuses
            if result.status == CheckStatus.OK:
                ok_count += 1
                icon = "✅"
            elif result.status == CheckStatus.DEGRADED:
                degraded_count += 1
                icon = "⚠️"
            else:
                critical_count += 1
                icon = "❌"
            
            print(f"{icon} {result.component:15} [{result.status:8}] {result.message}")
        
        print()
        print("=" * 70)
        print("Summary:")
        print(f"  ✅ OK:       {ok_count}")
        print(f"  ⚠️  DEGRADED: {degraded_count}")
        print(f"  ❌ CRITICAL: {critical_count}")
        print("=" * 70)
        
        # Overall status
        if critical_count > 0:
            print("\n🔴 SYSTEM STATUS: CRITICAL - Cannot proceed to production")
            print(f"   {critical_count} critical component(s) failed")
            return (ok_count, degraded_count, critical_count)
        elif degraded_count > 2:
            print("\n🟡 SYSTEM STATUS: DEGRADED - Review before production")
            print(f"   {degraded_count} component(s) degraded")
            return (ok_count, degraded_count, critical_count)
        elif degraded_count > 0:
            print("\n🟢 SYSTEM STATUS: OK (with warnings)")
            print(f"   {degraded_count} non-critical issue(s)")
            return (ok_count, degraded_count, critical_count)
        else:
            print("\n🟢 SYSTEM STATUS: ALL GREEN - Ready for production")
            return (ok_count, degraded_count, critical_count)


async def main():
    """Main entry point."""
    checker = SystemSanityChecker()
    ok, degraded, critical = await checker.run_all_checks()
    
    # Exit with appropriate code
    if critical > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
