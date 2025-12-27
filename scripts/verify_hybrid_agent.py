#!/usr/bin/env python3
"""
Quick verification that Hybrid Agent is running in production
"""
import requests
import sys

def check_hybrid_agent():
    """Check if Hybrid Agent is active and healthy"""
    try:
        # Check health
        response = requests.get("http://localhost:8000/api/test/hybrid/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n" + "="*70)
            print("🤖 HYBRID AGENT STATUS")
            print("="*70)
            
            print(f"\n[OK] Status: {data['status'].upper()}")
            print(f"   Mode: {data['mode']}")
            print(f"   TFT loaded: {'[OK]' if data['tft_loaded'] else '❌'}")
            print(f"   XGBoost loaded: {'[OK]' if data['xgb_loaded'] else '❌'}")
            
            weights = data['weights']
            print(f"\n[CHART] Ensemble Weights:")
            print(f"   TFT: {weights['tft']*100:.0f}%")
            print(f"   XGBoost: {weights['xgb']*100:.0f}%")
            print(f"   Agreement Bonus: +{weights['agreement_bonus']*100:.0f}%")
            
            print(f"\n💬 {data['message']}")
            print("="*70 + "\n")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_config():
    """Check current AI model configuration"""
    try:
        response = requests.get("http://localhost:8000/api/test/hybrid/config", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print("⚙️  CONFIGURATION:")
            print(f"   Current mode: {data['current_mode'].upper()}")
            print(f"   Recommendation: {data['recommendation']}")
            print()
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"[WARNING]  Config check failed: {e}")
        return False


if __name__ == "__main__":
    success = check_hybrid_agent()
    check_config()
    
    if success:
        print("[OK] Hybrid Agent is running and healthy!")
        print("   Monitor signals: python scripts/monitor_tft_signals.py")
        print("   Weekly review: python scripts/performance_review.py")
        sys.exit(0)
    else:
        print("❌ Hybrid Agent is not running correctly")
        sys.exit(1)
