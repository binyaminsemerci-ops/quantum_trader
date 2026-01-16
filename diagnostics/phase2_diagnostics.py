"""
Fase 2 Diagnostic Script - Circuit Breaker + Redis
==================================================
Comprehensive diagnostics for Phase 2 issues
"""
import asyncio
import redis.asyncio as aioredis
import requests
import subprocess
import json
from datetime import datetime

class Phase2Diagnostics:
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "position_monitor": {},
            "circuit_breaker": {},
            "redis_connectivity": {},
            "recommendations": []
        }
    
    async def check_position_monitor(self):
        """Check Position Monitor is still running"""
        print("\n" + "="*60)
        print("1️⃣ POSITION MONITOR STATUS")
        print("="*60)
        
        try:
            # Check recent logs
            result = subprocess.run(
                ["journalctl", "-u", "quantum-backend.service", "-n", "20", "--no-pager"],
                capture_output=True,
                text=True
            )
            
            logs = result.stdout + result.stderr
            
            # Count orders with positionSide=BOTH
            both_count = logs.count("positionSide=BOTH")
            error_count = logs.count("APIError")
            
            status = "✅ HEALTHY" if both_count > 0 and error_count == 0 else "⚠️ CHECK NEEDED"
            
            self.results["position_monitor"] = {
                "status": status,
                "orders_with_both": both_count,
                "api_errors": error_count
            }
            
            print(f"   Status: {status}")
            print(f"   Orders with positionSide=BOTH: {both_count}")
            print(f"   API Errors: {error_count}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["position_monitor"]["error"] = str(e)
    
    async def check_circuit_breaker(self):
        """Diagnose circuit breaker status"""
        print("\n" + "="*60)
        print("2️⃣ CIRCUIT BREAKER DIAGNOSTICS")
        print("="*60)
        
        try:
            # Try to get status from backend API (if endpoint exists)
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                health = response.json()
                print(f"   Backend Health: {health.get('status', 'unknown')}")
                
                # Check for circuit breaker in health response
                if 'circuit_breaker' in health:
                    cb = health['circuit_breaker']
                    self.results["circuit_breaker"] = cb
                    print(f"   Circuit Breaker State: {cb.get('state', 'unknown')}")
                    print(f"   Active: {cb.get('active', 'unknown')}")
                else:
                    print("   ⚠️  No circuit breaker info in health endpoint")
                    self.results["circuit_breaker"]["status"] = "NO_ENDPOINT"
                    
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️  Could not reach backend API: {e}")
                self.results["circuit_breaker"]["api_error"] = str(e)
            
            # Check logs for circuit breaker activity
            result = subprocess.run(
                ["journalctl", "-u", "quantum-backend.service", "--no-pager"],
                capture_output=True,
                text=True
            )
            
            logs = result.stdout + result.stderr
            circuit_mentions = [line for line in logs.split('\n') if 'circuit' in line.lower()]
            
            if circuit_mentions:
                print(f"\n   Found {len(circuit_mentions)} circuit breaker log entries")
                print("   Recent entries:")
                for line in circuit_mentions[-5:]:
                    print(f"     {line[:100]}")
                
                self.results["circuit_breaker"]["log_mentions"] = len(circuit_mentions)
            else:
                print("   ℹ️  No circuit breaker activity in logs")
                self.results["circuit_breaker"]["log_mentions"] = 0
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["circuit_breaker"]["error"] = str(e)
    
    async def check_redis_connectivity(self):
        """Diagnose Redis connectivity issues"""
        print("\n" + "="*60)
        print("3️⃣ REDIS CONNECTIVITY DIAGNOSTICS")
        print("="*60)
        
        # Check Redis is running
        try:
            result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            redis_ping = result.stdout.strip()
            print(f"   Redis Ping: {redis_ping}")
            self.results["redis_connectivity"]["redis_alive"] = redis_ping == "PONG"
            
        except Exception as e:
            print(f"   ❌ Redis ping failed: {e}")
            self.results["redis_connectivity"]["redis_alive"] = False
        
        # Check connectivity from services (systemd services use localhost)
        for service in ["quantum-cross-exchange", "quantum-eventbus-bridge"]:
            try:
                # Services connect to redis via localhost, just verify service is active
                result = subprocess.run(
                    ["systemctl", "is-active", f"{service}.service"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                is_active = result.stdout.strip() == "active"
                status = "✅" if is_active else "❌"
                print(f"   {service} service: {status} {result.stdout.strip()}")
                
                self.results["redis_connectivity"][f"{service}_active"] = is_active
                
            except Exception as e:
                print(f"   ❌ {service} status check failed: {e}")
                self.results["redis_connectivity"][f"{service}_check_error"] = str(e)
        
        # Check recent errors from service logs
        for service in ["quantum-cross-exchange", "quantum-eventbus-bridge"]:
            try:
                result = subprocess.run(
                    ["journalctl", "-u", f"{service}.service", "-n", "100", "--no-pager"],
                    capture_output=True,
                    text=True
                )
                
                logs = result.stdout + result.stderr
                redis_errors = [line for line in logs.split('\n') if 'redis' in line.lower() and 'error' in line.lower()]
                
                if redis_errors:
                    print(f"\n   {service} Redis Errors:")
                    print(f"     Total: {len(redis_errors)}")
                    print(f"     Last error: {redis_errors[-1][:100]}")
                    
                    self.results["redis_connectivity"][f"{service}_errors"] = len(redis_errors)
                else:
                    print(f"   {service}: ✅ No Redis errors")
                    self.results["redis_connectivity"][f"{service}_errors"] = 0
                
            except Exception as e:
                print(f"   ❌ Could not check {service} logs: {e}")
    
    async def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "="*60)
        print("4️⃣ RECOMMENDATIONS")
        print("="*60)
        
        # Position Monitor recommendations
        pm = self.results["position_monitor"]
        if pm.get("api_errors", 0) > 0:
            rec = "⚠️  Position Monitor has API errors - investigate error types"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        elif pm.get("orders_with_both", 0) > 0:
            rec = "✅ Position Monitor working correctly with positionSide=BOTH"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        
        # Circuit Breaker recommendations
        cb = self.results["circuit_breaker"]
        if cb.get("status") == "NO_ENDPOINT":
            rec = "🔧 IMPLEMENT: Add /api/circuit-breaker/status endpoint to backend"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        
        if cb.get("log_mentions", 0) > 0:
            rec = "🔍 INVESTIGATE: Circuit breaker is mentioned in logs - check activation reason"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        
        # Redis recommendations
        redis = self.results["redis_connectivity"]
        if not redis.get("redis_alive", False):
            rec = "🚨 CRITICAL: Redis is not responding - check service health"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        
        error_services = []
        for service in ["quantum_cross_exchange", "quantum_eventbus_bridge"]:
            if redis.get(f"{service}_errors", 0) > 10:
                error_services.append(service)
        
        if error_services:
            rec = f"🔧 FIX NEEDED: Implement Redis Connection Manager with retry logic for: {', '.join(error_services)}"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
        
        if not self.results["recommendations"]:
            rec = "✅ No critical issues found - system appears healthy"
            print(f"   {rec}")
            self.results["recommendations"].append(rec)
    
    async def run_all_diagnostics(self):
        """Run all diagnostic checks"""
        print("\n🔍 PHASE 2 DIAGNOSTICS - STARTING")
        print(f"Timestamp: {self.results['timestamp']}")
        
        await self.check_position_monitor()
        await self.check_circuit_breaker()
        await self.check_redis_connectivity()
        await self.generate_recommendations()
        
        print("\n" + "="*60)
        print("✅ DIAGNOSTICS COMPLETE")
        print("="*60)
        
        # Save results to JSON
        with open("/tmp/phase2_diagnostics.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\nResults saved to: /tmp/phase2_diagnostics.json")
        return self.results

async def main():
    diagnostics = Phase2Diagnostics()
    results = await diagnostics.run_all_diagnostics()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("\n1. Review recommendations above")
    print("2. Implement Circuit Breaker Management API")
    print("3. Implement Redis Connection Manager")
    print("4. Test fixes on testnet")
    print("5. Monitor for 24 hours before Phase 3")

if __name__ == "__main__":
    asyncio.run(main())
