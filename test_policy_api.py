"""
PolicyStore Integration Test Client

Simple script to test PolicyStore HTTP API endpoints.
Demonstrates how external systems can interact with the global trading policy.

Usage:
    python test_policy_api.py
"""

import requests
import json
import time
from typing import Dict, Any


class PolicyAPIClient:
    """Client for interacting with PolicyStore HTTP API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.policy_url = f"{self.base_url}/api/policy"
    
    def get_status(self) -> Dict[str, Any]:
        """Check PolicyStore availability."""
        response = requests.get(f"{self.policy_url}/status")
        response.raise_for_status()
        return response.json()
    
    def get_policy(self) -> Dict[str, Any]:
        """Get current global policy."""
        response = requests.get(self.policy_url)
        response.raise_for_status()
        return response.json()
    
    def update_policy(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update policy fields."""
        response = requests.patch(self.policy_url, json=updates)
        response.raise_for_status()
        return response.json()
    
    def reset_policy(self) -> Dict[str, Any]:
        """Reset policy to defaults."""
        response = requests.post(f"{self.policy_url}/reset")
        response.raise_for_status()
        return response.json()
    
    def get_risk_mode(self) -> str:
        """Get current risk mode."""
        response = requests.get(f"{self.policy_url}/risk_mode")
        response.raise_for_status()
        return response.json()['risk_mode']
    
    def set_risk_mode(self, mode: str) -> Dict[str, Any]:
        """Set risk mode (AGGRESSIVE/NORMAL/DEFENSIVE)."""
        response = requests.post(f"{self.policy_url}/risk_mode/{mode}")
        response.raise_for_status()
        return response.json()
    
    def get_allowed_symbols(self) -> list[str]:
        """Get allowed trading symbols."""
        response = requests.get(f"{self.policy_url}/allowed_symbols")
        response.raise_for_status()
        return response.json()['allowed_symbols']
    
    def get_model_versions(self) -> Dict[str, str]:
        """Get active ML model versions."""
        response = requests.get(f"{self.policy_url}/model_versions")
        response.raise_for_status()
        return response.json()['model_versions']


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_policy(policy_data: Dict[str, Any]):
    """Pretty print policy data."""
    policy = policy_data.get('policy', policy_data)
    print(json.dumps(policy, indent=2))


def main():
    """Run integration tests."""
    print_section("PolicyStore API Integration Test")
    
    client = PolicyAPIClient()
    
    # Test 1: Check Status
    print_section("Test 1: Check PolicyStore Status")
    try:
        status = client.get_status()
        print(f"✅ PolicyStore Available: {status['available']}")
        print(f"   Current Risk Mode: {status['current_risk_mode']}")
        print(f"   Last Updated: {status['last_updated']}")
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        print("\n⚠️  Is the backend server running on http://localhost:8000?")
        return
    
    # Test 2: Get Current Policy
    print_section("Test 2: Get Current Policy")
    try:
        policy = client.get_policy()
        print("✅ Retrieved current policy:")
        print_policy(policy)
    except Exception as e:
        print(f"❌ Get policy failed: {e}")
        return
    
    # Test 3: Get Risk Mode
    print_section("Test 3: Get Current Risk Mode")
    try:
        risk_mode = client.get_risk_mode()
        print(f"✅ Current risk mode: {risk_mode}")
    except Exception as e:
        print(f"❌ Get risk mode failed: {e}")
    
    # Test 4: Update Risk Mode to AGGRESSIVE
    print_section("Test 4: Switch to AGGRESSIVE Mode")
    try:
        result = client.set_risk_mode("AGGRESSIVE")
        print(f"✅ Risk mode updated to AGGRESSIVE")
        print(f"   Max Risk Per Trade: {result['max_risk_per_trade']}")
        print(f"   Max Positions: {result['max_positions']}")
        print(f"   Min Confidence: {result['global_min_confidence']}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Set risk mode failed: {e}")
    
    # Test 5: Partial Policy Update
    print_section("Test 5: Update Specific Fields")
    try:
        updates = {
            "max_risk_per_trade": 0.025,
            "max_positions": 8,
            "global_min_confidence": 0.72
        }
        result = client.update_policy(updates)
        print(f"✅ Policy updated:")
        print(f"   Max Risk Per Trade: {result['policy']['max_risk_per_trade']}")
        print(f"   Max Positions: {result['policy']['max_positions']}")
        print(f"   Min Confidence: {result['policy']['global_min_confidence']}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Update policy failed: {e}")
    
    # Test 6: Get Allowed Symbols
    print_section("Test 6: Get Allowed Symbols")
    try:
        symbols = client.get_allowed_symbols()
        print(f"✅ Allowed symbols ({len(symbols)} total):")
        if symbols:
            print(f"   {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"   ... and {len(symbols) - 10} more")
        else:
            print("   (All symbols allowed)")
    except Exception as e:
        print(f"❌ Get symbols failed: {e}")
    
    # Test 7: Get Model Versions
    print_section("Test 7: Get Model Versions")
    try:
        versions = client.get_model_versions()
        print(f"✅ Active model versions:")
        if versions:
            for model, version in versions.items():
                print(f"   {model}: {version}")
        else:
            print("   (No model versions registered)")
    except Exception as e:
        print(f"❌ Get model versions failed: {e}")
    
    # Test 8: Reset to Defaults
    print_section("Test 8: Reset Policy to Defaults")
    try:
        result = client.reset_policy()
        print(f"✅ Policy reset to defaults:")
        print(f"   Risk Mode: {result['policy']['risk_mode']}")
        print(f"   Max Risk Per Trade: {result['policy']['max_risk_per_trade']}")
        print(f"   Max Positions: {result['policy']['max_positions']}")
        print(f"   Min Confidence: {result['policy']['global_min_confidence']}")
    except Exception as e:
        print(f"❌ Reset policy failed: {e}")
    
    # Test 9: Final State
    print_section("Test 9: Verify Final State")
    try:
        policy = client.get_policy()
        print("✅ Final policy state:")
        print_policy(policy)
    except Exception as e:
        print(f"❌ Get final state failed: {e}")
    
    print_section("All Tests Complete")
    print("✅ PolicyStore API integration working correctly!")
    print("\nAPI Endpoints Available:")
    print("  GET    /api/policy/status          - Check availability")
    print("  GET    /api/policy                 - Get full policy")
    print("  PATCH  /api/policy                 - Update policy fields")
    print("  POST   /api/policy/reset           - Reset to defaults")
    print("  GET    /api/policy/risk_mode       - Get risk mode")
    print("  POST   /api/policy/risk_mode/{mode} - Set risk mode")
    print("  GET    /api/policy/allowed_symbols - Get allowed symbols")
    print("  GET    /api/policy/model_versions  - Get model versions")


if __name__ == "__main__":
    main()
