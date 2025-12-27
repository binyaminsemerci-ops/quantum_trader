"""
Unit tests for PolicyStore module.

Tests cover:
- Policy validation
- Serialization/deserialization
- Atomic operations
- Thread safety
- Merging logic
- All CRUD operations
"""

import pytest
import threading
import time
from datetime import datetime
from policy_store import (
    GlobalPolicy,
    PolicyValidator,
    PolicySerializer,
    PolicyMerger,
    PolicyDefaults,
    InMemoryPolicyStore,
    RiskMode,
    PolicyValidationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_policy():
    """Default policy for testing."""
    return PolicyDefaults.create_default()


@pytest.fixture
def store():
    """Fresh in-memory store for each test."""
    return InMemoryPolicyStore()


@pytest.fixture
def populated_store():
    """Store with some initial data."""
    s = InMemoryPolicyStore()
    s.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["STRAT_1", "STRAT_2"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
        "opp_rankings": {"BTCUSDT": 0.9, "ETHUSDT": 0.85},
        "model_versions": {"xgboost": "v1", "lightgbm": "v2"},
    })
    return s


# ============================================================================
# TEST GLOBALOLICY DATACLASS
# ============================================================================

def test_global_policy_creation():
    """Test creating GlobalPolicy with defaults."""
    policy = GlobalPolicy()
    assert policy.risk_mode == "NORMAL"
    assert policy.allowed_strategies == []
    assert policy.max_risk_per_trade == 0.01
    assert policy.max_positions == 10
    assert policy.global_min_confidence == 0.65


def test_global_policy_to_dict():
    """Test converting GlobalPolicy to dict."""
    policy = GlobalPolicy(
        risk_mode="AGGRESSIVE",
        allowed_strategies=["S1", "S2"],
        max_risk_per_trade=0.02,
    )
    d = policy.to_dict()
    assert d["risk_mode"] == "AGGRESSIVE"
    assert d["allowed_strategies"] == ["S1", "S2"]
    assert d["max_risk_per_trade"] == 0.02
    assert "last_updated" in d


def test_global_policy_from_dict():
    """Test creating GlobalPolicy from dict."""
    data = {
        "risk_mode": "DEFENSIVE",
        "allowed_strategies": ["S1"],
        "max_positions": 5,
        "unknown_field": "ignored",  # Should be filtered
    }
    policy = GlobalPolicy.from_dict(data)
    assert policy.risk_mode == "DEFENSIVE"
    assert policy.allowed_strategies == ["S1"]
    assert policy.max_positions == 5


# ============================================================================
# TEST POLICY VALIDATOR
# ============================================================================

def test_validator_valid_policy():
    """Test that valid policy passes validation."""
    policy = {
        "risk_mode": "NORMAL",
        "max_risk_per_trade": 0.015,
        "max_positions": 8,
        "global_min_confidence": 0.7,
        "allowed_strategies": ["S1", "S2"],
        "allowed_symbols": ["BTC", "ETH"],
        "opp_rankings": {"BTC": 0.9, "ETH": 0.8},
        "model_versions": {"xgb": "v1"},
    }
    # Should not raise
    PolicyValidator.validate(policy)


def test_validator_invalid_risk_mode():
    """Test that invalid risk mode fails validation."""
    with pytest.raises(PolicyValidationError, match="Invalid risk_mode"):
        PolicyValidator.validate({"risk_mode": "INVALID_MODE"})


def test_validator_invalid_max_risk():
    """Test that invalid max_risk_per_trade fails."""
    with pytest.raises(PolicyValidationError, match="max_risk_per_trade"):
        PolicyValidator.validate({"max_risk_per_trade": 1.5})
    
    with pytest.raises(PolicyValidationError, match="max_risk_per_trade"):
        PolicyValidator.validate({"max_risk_per_trade": -0.01})


def test_validator_invalid_max_positions():
    """Test that invalid max_positions fails."""
    with pytest.raises(PolicyValidationError, match="max_positions"):
        PolicyValidator.validate({"max_positions": 0})
    
    with pytest.raises(PolicyValidationError, match="max_positions"):
        PolicyValidator.validate({"max_positions": 150})


def test_validator_invalid_confidence():
    """Test that invalid confidence fails."""
    with pytest.raises(PolicyValidationError, match="global_min_confidence"):
        PolicyValidator.validate({"global_min_confidence": 1.5})


def test_validator_invalid_types():
    """Test that wrong types fail validation."""
    with pytest.raises(PolicyValidationError, match="must be a list"):
        PolicyValidator.validate({"allowed_strategies": "not_a_list"})
    
    with pytest.raises(PolicyValidationError, match="must be a dict"):
        PolicyValidator.validate({"opp_rankings": ["not", "a", "dict"]})


def test_validator_invalid_ranking_scores():
    """Test that ranking scores must be 0-1."""
    with pytest.raises(PolicyValidationError, match="must be between 0 and 1"):
        PolicyValidator.validate({"opp_rankings": {"BTC": 1.5}})


# ============================================================================
# TEST POLICY SERIALIZER
# ============================================================================

def test_serializer_to_dict():
    """Test serializing policy to dict."""
    policy = GlobalPolicy(risk_mode="AGGRESSIVE", max_positions=15)
    serializer = PolicySerializer()
    d = serializer.to_dict(policy)
    assert d["risk_mode"] == "AGGRESSIVE"
    assert d["max_positions"] == 15


def test_serializer_from_dict():
    """Test deserializing dict to policy."""
    data = {"risk_mode": "DEFENSIVE", "max_positions": 5}
    serializer = PolicySerializer()
    policy = serializer.from_dict(data)
    assert policy.risk_mode == "DEFENSIVE"
    assert policy.max_positions == 5


def test_serializer_json_roundtrip():
    """Test JSON serialization roundtrip."""
    original = GlobalPolicy(
        risk_mode="NORMAL",
        allowed_strategies=["S1", "S2"],
        opp_rankings={"BTC": 0.9},
    )
    serializer = PolicySerializer()
    
    # To JSON and back
    json_str = serializer.to_json(original)
    restored = serializer.from_json(json_str)
    
    assert restored.risk_mode == original.risk_mode
    assert restored.allowed_strategies == original.allowed_strategies
    assert restored.opp_rankings == original.opp_rankings


# ============================================================================
# TEST POLICY MERGER
# ============================================================================

def test_merger_simple_update():
    """Test merging simple fields."""
    base = {
        "risk_mode": "NORMAL",
        "max_positions": 10,
        "allowed_strategies": ["S1"],
    }
    partial = {
        "risk_mode": "AGGRESSIVE",
    }
    
    merger = PolicyMerger()
    merged = merger.merge(base, partial)
    
    assert merged["risk_mode"] == "AGGRESSIVE"
    assert merged["max_positions"] == 10  # Unchanged
    assert merged["allowed_strategies"] == ["S1"]  # Unchanged


def test_merger_nested_dict():
    """Test merging nested dictionaries."""
    base = {
        "opp_rankings": {"BTC": 0.9, "ETH": 0.8},
        "model_versions": {"xgb": "v1"},
    }
    partial = {
        "opp_rankings": {"SOL": 0.85},  # Add new symbol
        "model_versions": {"lgb": "v2"},  # Add new model
    }
    
    merger = PolicyMerger()
    merged = merger.merge(base, partial)
    
    # Should have merged both dicts
    assert merged["opp_rankings"] == {"BTC": 0.9, "ETH": 0.8, "SOL": 0.85}
    assert merged["model_versions"] == {"xgb": "v1", "lgb": "v2"}


def test_merger_timestamp_update():
    """Test that timestamp is always updated."""
    base = {
        "risk_mode": "NORMAL",
        "last_updated": "2020-01-01T00:00:00Z",
    }
    partial = {
        "risk_mode": "AGGRESSIVE",
        "last_updated": "2020-01-01T00:00:00Z",  # Should be ignored
    }
    
    merger = PolicyMerger()
    merged = merger.merge(base, partial)
    
    # Timestamp should be new
    assert merged["last_updated"] != "2020-01-01T00:00:00Z"


def test_merger_immutability():
    """Test that merge doesn't mutate inputs."""
    base = {"risk_mode": "NORMAL", "allowed_strategies": ["S1"]}
    partial = {"risk_mode": "AGGRESSIVE"}
    
    merger = PolicyMerger()
    merged = merger.merge(base, partial)
    
    # Original should be unchanged
    assert base["risk_mode"] == "NORMAL"
    assert merged["risk_mode"] == "AGGRESSIVE"


# ============================================================================
# TEST POLICY DEFAULTS
# ============================================================================

def test_defaults_create_default():
    """Test creating default policy."""
    policy = PolicyDefaults.create_default()
    assert policy.risk_mode == "NORMAL"
    assert policy.max_risk_per_trade == 0.01
    assert policy.max_positions == 10
    assert policy.global_min_confidence == 0.65


def test_defaults_create_conservative():
    """Test creating conservative policy."""
    policy = PolicyDefaults.create_conservative()
    assert policy.risk_mode == "DEFENSIVE"
    assert policy.max_risk_per_trade == 0.005
    assert policy.max_positions == 5
    assert policy.global_min_confidence == 0.75


def test_defaults_create_aggressive():
    """Test creating aggressive policy."""
    policy = PolicyDefaults.create_aggressive()
    assert policy.risk_mode == "AGGRESSIVE"
    assert policy.max_risk_per_trade == 0.02
    assert policy.max_positions == 15
    assert policy.global_min_confidence == 0.55


# ============================================================================
# TEST IN-MEMORY POLICY STORE
# ============================================================================

def test_store_initial_state(store):
    """Test that store initializes with default policy."""
    policy = store.get()
    assert "risk_mode" in policy
    assert "max_positions" in policy
    assert "last_updated" in policy


def test_store_get_returns_copy(store):
    """Test that get() returns a copy, not reference."""
    policy1 = store.get()
    policy1["risk_mode"] = "HACKED"
    
    policy2 = store.get()
    assert policy2["risk_mode"] != "HACKED"


def test_store_update_full_policy(store):
    """Test updating entire policy."""
    new_policy = {
        "risk_mode": "AGGRESSIVE",
        "allowed_strategies": ["S1", "S2", "S3"],
        "max_risk_per_trade": 0.02,
        "max_positions": 15,
        "global_min_confidence": 0.55,
        "allowed_symbols": ["BTC", "ETH", "SOL"],
        "opp_rankings": {"BTC": 0.95, "ETH": 0.9, "SOL": 0.88},
        "model_versions": {"xgb": "v5", "lgb": "v3"},
    }
    
    store.update(new_policy)
    retrieved = store.get()
    
    assert retrieved["risk_mode"] == "AGGRESSIVE"
    assert retrieved["allowed_strategies"] == ["S1", "S2", "S3"]
    assert retrieved["max_risk_per_trade"] == 0.02
    assert retrieved["opp_rankings"]["BTC"] == 0.95


def test_store_update_with_validation_error(store):
    """Test that invalid update is rejected."""
    invalid_policy = {
        "risk_mode": "INVALID_MODE",
    }
    
    with pytest.raises(PolicyValidationError):
        store.update(invalid_policy)
    
    # Store should be unchanged
    policy = store.get()
    assert policy["risk_mode"] == "NORMAL"  # Still default


def test_store_patch_single_field(populated_store):
    """Test patching a single field."""
    original_strategies = populated_store.get()["allowed_strategies"]
    
    populated_store.patch({"risk_mode": "AGGRESSIVE"})
    
    policy = populated_store.get()
    assert policy["risk_mode"] == "AGGRESSIVE"
    # Other fields unchanged
    assert policy["allowed_strategies"] == original_strategies


def test_store_patch_nested_dict(populated_store):
    """Test patching nested dictionaries."""
    populated_store.patch({
        "opp_rankings": {"SOLUSDT": 0.92},  # Add new symbol
    })
    
    policy = populated_store.get()
    # Should have merged with existing
    assert "BTCUSDT" in policy["opp_rankings"]
    assert "ETHUSDT" in policy["opp_rankings"]
    assert policy["opp_rankings"]["SOLUSDT"] == 0.92


def test_store_patch_validation_error(populated_store):
    """Test that invalid patch is rejected."""
    original_risk = populated_store.get()["max_risk_per_trade"]
    
    with pytest.raises(PolicyValidationError):
        populated_store.patch({"max_risk_per_trade": 5.0})
    
    # Should be unchanged
    policy = populated_store.get()
    assert policy["max_risk_per_trade"] == original_risk


def test_store_reset(populated_store):
    """Test resetting store to default."""
    # Verify it's populated
    assert len(populated_store.get()["allowed_strategies"]) > 0
    
    # Reset
    populated_store.reset()
    
    # Should be back to default
    policy = populated_store.get()
    assert policy["allowed_strategies"] == []
    assert policy["risk_mode"] == "NORMAL"


def test_store_get_policy_object(populated_store):
    """Test getting policy as typed dataclass."""
    policy_obj = populated_store.get_policy_object()
    
    assert isinstance(policy_obj, GlobalPolicy)
    assert policy_obj.risk_mode == "NORMAL"
    assert "STRAT_1" in policy_obj.allowed_strategies


def test_store_timestamp_auto_update(store):
    """Test that timestamp is automatically updated."""
    store.update({"risk_mode": "NORMAL"})
    time1 = store.get()["last_updated"]
    
    time.sleep(0.01)  # Small delay
    
    store.patch({"risk_mode": "AGGRESSIVE"})
    time2 = store.get()["last_updated"]
    
    assert time1 != time2


# ============================================================================
# TEST THREAD SAFETY
# ============================================================================

def test_store_concurrent_reads(populated_store):
    """Test that concurrent reads don't corrupt data."""
    results = []
    errors = []
    
    def read_policy():
        try:
            for _ in range(100):
                policy = populated_store.get()
                results.append(policy["risk_mode"])
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=read_policy) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0
    assert all(r == "NORMAL" for r in results)


def test_store_concurrent_writes(store):
    """Test that concurrent writes are atomic."""
    errors = []
    
    def update_risk_mode(mode):
        try:
            for _ in range(50):
                store.patch({"risk_mode": mode})
        except Exception as e:
            errors.append(e)
    
    threads = [
        threading.Thread(target=update_risk_mode, args=("AGGRESSIVE",)),
        threading.Thread(target=update_risk_mode, args=("DEFENSIVE",)),
        threading.Thread(target=update_risk_mode, args=("NORMAL",)),
    ]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should complete without errors
    assert len(errors) == 0
    
    # Final state should be valid
    final_policy = store.get()
    assert final_policy["risk_mode"] in ["AGGRESSIVE", "DEFENSIVE", "NORMAL"]


def test_store_read_write_consistency(store):
    """Test read/write consistency under concurrent access."""
    stop_flag = threading.Event()
    write_count = [0]
    read_errors = []
    
    def writer():
        count = 0
        while not stop_flag.is_set() and count < 100:
            store.patch({"max_positions": count % 20 + 1})
            write_count[0] = count
            count += 1
            time.sleep(0.001)
    
    def reader():
        while not stop_flag.is_set():
            try:
                policy = store.get()
                # Validate that data is consistent
                assert 1 <= policy["max_positions"] <= 20
            except AssertionError as e:
                read_errors.append(e)
    
    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(5)]
    
    writer_thread.start()
    for t in reader_threads:
        t.start()
    
    # Let them run for a bit
    time.sleep(0.2)
    stop_flag.set()
    
    writer_thread.join()
    for t in reader_threads:
        t.join()
    
    # No consistency errors should occur
    assert len(read_errors) == 0


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_store_empty_update(store):
    """Test updating with empty dict."""
    original = store.get()
    store.patch({})
    updated = store.get()
    
    # Only timestamp should change
    assert updated["risk_mode"] == original["risk_mode"]


def test_store_overwrite_nested_dict_completely(store):
    """Test that non-merge fields overwrite completely."""
    store.update({
        "allowed_strategies": ["S1", "S2", "S3"],
    })
    
    # Patch with new list (should replace, not merge)
    store.patch({
        "allowed_strategies": ["S4"],
    })
    
    policy = store.get()
    assert policy["allowed_strategies"] == ["S4"]


def test_store_custom_initial_policy():
    """Test initializing store with custom policy."""
    custom = PolicyDefaults.create_aggressive()
    store = InMemoryPolicyStore(initial_policy=custom)
    
    policy = store.get()
    assert policy["risk_mode"] == "AGGRESSIVE"
    assert policy["max_risk_per_trade"] == 0.02


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_workflow():
    """Test complete workflow simulating real usage."""
    # 1. Initialize store
    store = InMemoryPolicyStore()
    
    # 2. MSC AI sets initial policy
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["MOMENTUM_1", "MEANREV_2"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
    })
    
    # 3. OppRank updates symbol rankings
    store.patch({
        "opp_rankings": {
            "BTCUSDT": 0.92,
            "ETHUSDT": 0.88,
        }
    })
    
    # 4. CLM updates model versions
    store.patch({
        "model_versions": {
            "xgboost": "v14",
            "lightgbm": "v11",
            "nhits": "v9",
            "patchtst": "v7",
        }
    })
    
    # 5. MSC AI changes risk mode
    store.patch({"risk_mode": "AGGRESSIVE"})
    
    # 6. Verify final state
    policy = store.get_policy_object()
    assert policy.risk_mode == "AGGRESSIVE"
    assert len(policy.allowed_strategies) == 2
    assert policy.opp_rankings["BTCUSDT"] == 0.92
    assert policy.model_versions["xgboost"] == "v14"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
