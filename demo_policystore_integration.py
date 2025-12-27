"""
PolicyStore AI Integration Demo

Demonstrates how AI components interact with the PolicyStore:
1. MSC AI updates risk parameters
2. OpportunityRanker writes rankings
3. All components read from shared state

Run this after starting the backend to see live integration.
"""

import asyncio
import time
import requests
from datetime import datetime


class PolicyStoreMonitor:
    """Monitor PolicyStore changes in real-time."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.policy_url = f"{base_url}/api/policy"
    
    def get_policy(self):
        """Get current policy."""
        try:
            response = requests.get(self.policy_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting policy: {e}")
            return None
    
    def print_policy_summary(self, policy_data):
        """Print formatted policy summary."""
        if not policy_data:
            return
        
        policy = policy_data.get('policy', {})
        
        print("\n" + "="*70)
        print(f"  PolicyStore State @ {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        # Core Configuration
        print("\n📊 Core Configuration:")
        print(f"   Risk Mode:           {policy.get('risk_mode', 'UNKNOWN')}")
        print(f"   Max Risk/Trade:      {policy.get('max_risk_per_trade', 0):.3f}")
        print(f"   Max Positions:       {policy.get('max_positions', 0)}")
        print(f"   Min Confidence:      {policy.get('global_min_confidence', 0):.2f}")
        
        # Strategies
        strategies = policy.get('allowed_strategies', [])
        if strategies:
            print(f"\n🎯 Active Strategies: ({len(strategies)})")
            for strat in strategies[:5]:
                print(f"   • {strat}")
            if len(strategies) > 5:
                print(f"   ... and {len(strategies) - 5} more")
        else:
            print(f"\n🎯 Active Strategies: All allowed")
        
        # Opportunity Rankings
        rankings = policy.get('opp_rankings', {})
        if rankings:
            print(f"\n🔍 Opportunity Rankings: ({len(rankings)} symbols)")
            sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            for symbol, score in sorted_rankings[:5]:
                print(f"   {symbol:12s} {score:.3f} {'█' * int(score * 20)}")
            if len(rankings) > 5:
                print(f"   ... and {len(rankings) - 5} more")
        else:
            print(f"\n🔍 Opportunity Rankings: Not yet computed")
        
        # Model Versions
        versions = policy.get('model_versions', {})
        if versions:
            print(f"\n🤖 Model Versions:")
            for model, version in versions.items():
                print(f"   {model}: {version}")
        
        # Metadata
        last_updated = policy.get('last_updated', 'Never')
        print(f"\n⏰ Last Updated: {last_updated}")
        print("="*70 + "\n")
    
    async def watch_changes(self, interval=5):
        """Watch for policy changes."""
        print("👁️  Watching PolicyStore for changes...")
        print("   Press Ctrl+C to stop\n")
        
        last_policy = None
        
        try:
            while True:
                current_policy = self.get_policy()
                
                if current_policy:
                    # Check if policy changed
                    if current_policy != last_policy:
                        self.print_policy_summary(current_policy)
                        last_policy = current_policy
                
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")


def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


async def demo_integration():
    """Demonstrate PolicyStore integration."""
    
    print("\n" + "="*70)
    print("  PolicyStore AI Integration Demo")
    print("="*70)
    
    # Check backend
    print("\n🔍 Checking backend status...")
    if not check_backend_health():
        print("\n❌ Backend not running!")
        print("   Please start the backend first:")
        print("   python backend/main.py")
        return
    
    print("✅ Backend is running\n")
    
    # Create monitor
    monitor = PolicyStoreMonitor()
    
    # Show current state
    print("📸 Current PolicyStore State:")
    current = monitor.get_policy()
    monitor.print_policy_summary(current)
    
    # Explain integration points
    print("\n" + "="*70)
    print("  AI Component Integration Points")
    print("="*70)
    
    print("\n🧠 MSC AI (Meta Strategy Controller):")
    print("   • Evaluates system performance every 30 minutes")
    print("   • Writes risk_mode, max_risk, max_positions, min_confidence")
    print("   • Updates allowed_strategies based on performance")
    print("   • Next evaluation: Check MSC scheduler logs")
    
    print("\n🔍 OpportunityRanker:")
    print("   • Ranks market opportunities every 5 minutes")
    print("   • Writes opp_rankings to PolicyStore")
    print("   • Updates when: /opportunities/update endpoint called")
    
    print("\n🛡️ RiskGuard (Future):")
    print("   • Reads max_risk_per_trade and max_positions")
    print("   • Enforces limits dynamically")
    
    print("\n🎯 Orchestrator (Future):")
    print("   • Reads global_min_confidence for signal filtering")
    print("   • Reads opp_rankings for symbol selection")
    
    print("\n📚 Continuous Learning (Future):")
    print("   • Writes model_versions after retraining")
    
    print("\n" + "="*70)
    print("  Live Monitoring")
    print("="*70)
    
    print("\n📡 Starting real-time monitoring...")
    print("   This will show any changes to the PolicyStore")
    print("   Try these actions in another terminal:\n")
    print("   • Trigger MSC evaluation:")
    print("     curl -X POST http://localhost:8000/msc/evaluate")
    print("\n   • Update opportunity rankings:")
    print("     curl -X POST http://localhost:8000/opportunities/update")
    print("\n   • Change risk mode manually:")
    print("     curl -X POST http://localhost:8000/api/policy/risk_mode/AGGRESSIVE")
    print("\n   • Update specific fields:")
    print("     curl -X PATCH http://localhost:8000/api/policy \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"max_risk_per_trade\": 0.02}'")
    
    # Start monitoring
    await monitor.watch_changes(interval=3)


if __name__ == "__main__":
    try:
        asyncio.run(demo_integration())
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped")
