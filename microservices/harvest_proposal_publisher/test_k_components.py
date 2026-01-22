"""
P2.6B: Unit tests for k_components extraction
Tests the _extract_k_components method for robustness.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from microservices.harvest_proposal_publisher.main import HarvestProposalPublisher


def test_extract_k_components_with_valid_data():
    """Test extraction when all components present"""
    publisher = HarvestProposalPublisher(
        redis_url="redis://localhost:6379",
        symbols=["BTCUSDT"],
        publish_interval=10,
    )
    
    harvest_output = {
        "kill_score": 0.753,
        "audit": {
            "k_components": {
                "regime_flip": 1.0,
                "sigma_spike": 0.0,
                "ts_drop": 0.208,
                "age_penalty": 0.042,
            }
        }
    }
    
    result = publisher._extract_k_components(harvest_output)
    
    assert result["regime_flip"] == 1.0, f"Expected 1.0, got {result['regime_flip']}"
    assert result["sigma_spike"] == 0.0, f"Expected 0.0, got {result['sigma_spike']}"
    assert result["ts_drop"] == 0.208, f"Expected 0.208, got {result['ts_drop']}"
    assert result["age_penalty"] == 0.042, f"Expected 0.042, got {result['age_penalty']}"
    print("✅ test_extract_k_components_with_valid_data PASSED")


def test_extract_k_components_missing_audit():
    """Test extraction when audit field missing"""
    publisher = HarvestProposalPublisher(
        redis_url="redis://localhost:6379",
        symbols=["BTCUSDT"],
        publish_interval=10,
    )
    
    harvest_output = {
        "kill_score": 0.5,
    }
    
    result = publisher._extract_k_components(harvest_output)
    
    assert result["regime_flip"] == 0.0, f"Expected 0.0, got {result['regime_flip']}"
    assert result["sigma_spike"] == 0.0, f"Expected 0.0, got {result['sigma_spike']}"
    assert result["ts_drop"] == 0.0, f"Expected 0.0, got {result['ts_drop']}"
    assert result["age_penalty"] == 0.0, f"Expected 0.0, got {result['age_penalty']}"
    print("✅ test_extract_k_components_missing_audit PASSED")


def test_extract_k_components_missing_k_components():
    """Test extraction when k_components field missing"""
    publisher = HarvestProposalPublisher(
        redis_url="redis://localhost:6379",
        symbols=["BTCUSDT"],
        publish_interval=10,
    )
    
    harvest_output = {
        "kill_score": 0.5,
        "audit": {
            "tranche_weights": {},
        }
    }
    
    result = publisher._extract_k_components(harvest_output)
    
    assert result["regime_flip"] == 0.0
    assert result["sigma_spike"] == 0.0
    assert result["ts_drop"] == 0.0
    assert result["age_penalty"] == 0.0
    print("✅ test_extract_k_components_missing_k_components PASSED")


def test_extract_k_components_with_invalid_values():
    """Test extraction when values are invalid (not numeric)"""
    publisher = HarvestProposalPublisher(
        redis_url="redis://localhost:6379",
        symbols=["BTCUSDT"],
        publish_interval=10,
    )
    
    harvest_output = {
        "kill_score": 0.5,
        "audit": {
            "k_components": {
                "regime_flip": "invalid",
                "sigma_spike": None,
                "ts_drop": 0.2,
                "age_penalty": 0.1,
            }
        }
    }
    
    result = publisher._extract_k_components(harvest_output)
    
    assert result["regime_flip"] == 0.0, "Invalid string should default to 0.0"
    assert result["sigma_spike"] == 0.0, "None should default to 0.0"
    assert result["ts_drop"] == 0.2, f"Expected 0.2, got {result['ts_drop']}"
    assert result["age_penalty"] == 0.1, f"Expected 0.1, got {result['age_penalty']}"
    print("✅ test_extract_k_components_with_invalid_values PASSED")


def test_extract_k_components_partial_missing():
    """Test extraction when some components missing"""
    publisher = HarvestProposalPublisher(
        redis_url="redis://localhost:6379",
        symbols=["BTCUSDT"],
        publish_interval=10,
    )
    
    harvest_output = {
        "kill_score": 0.5,
        "audit": {
            "k_components": {
                "regime_flip": 1.0,
                "ts_drop": 0.3,
                # sigma_spike and age_penalty missing
            }
        }
    }
    
    result = publisher._extract_k_components(harvest_output)
    
    assert result["regime_flip"] == 1.0
    assert result["sigma_spike"] == 0.0, "Missing component should default to 0.0"
    assert result["ts_drop"] == 0.3
    assert result["age_penalty"] == 0.0, "Missing component should default to 0.0"
    print("✅ test_extract_k_components_partial_missing PASSED")


if __name__ == "__main__":
    print("=== P2.6B K-Components Extraction Tests ===\n")
    
    try:
        test_extract_k_components_with_valid_data()
        test_extract_k_components_missing_audit()
        test_extract_k_components_missing_k_components()
        test_extract_k_components_with_invalid_values()
        test_extract_k_components_partial_missing()
        
        print("\n✅ All tests PASSED")
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
