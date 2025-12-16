"""
PolicyStore - Central configuration and state management for Quantum Trader AI system.

This module provides the single source of truth for global trading parameters,
risk settings, allowed strategies, symbol rankings, and model versions.

All AI components read from PolicyStore to ensure coherent decision-making.
MSC AI and OppRank write to it to update system-wide policies.

Design principles:
- Thread-safe atomic operations
- Multiple storage backend support
- Clean separation of concerns
- Type-safe with validation
- Efficient serialization
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Protocol, Any, Optional
from enum import Enum
from copy import deepcopy


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class RiskMode(str, Enum):
    """Valid risk modes for the trading system."""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


# ============================================================================
# POLICY STRUCTURE
# ============================================================================

@dataclass
class GlobalPolicy:
    """
    The complete global policy state for Quantum Trader.
    
    This is the canonical structure for all system-wide configuration.
    All components read these values to coordinate their behavior.
    
    Attributes:
        risk_mode: Current risk posture (AGGRESSIVE/NORMAL/DEFENSIVE)
        allowed_strategies: List of strategy IDs permitted to trade
        allowed_symbols: List of symbols permitted for trading (from OppRank)
        max_risk_per_trade: Maximum fraction of capital to risk per trade
        max_positions: Maximum number of concurrent open positions
        global_min_confidence: Minimum confidence threshold for all signals
        opp_rankings: Symbol scores from OpportunityRanker
        model_versions: Active version strings for each ML model
        system_health: Optional health status indicators
        custom_params: Extensibility for additional parameters
        last_updated: Timestamp of last policy update
    """
    risk_mode: str = "NORMAL"
    allowed_strategies: list[str] = field(default_factory=list)
    allowed_symbols: list[str] = field(default_factory=list)
    max_risk_per_trade: float = 0.01
    max_positions: int = 10
    global_min_confidence: float = 0.65
    opp_rankings: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    system_health: dict[str, Any] = field(default_factory=dict)
    custom_params: dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalPolicy:
        """Create from dictionary, handling missing fields gracefully."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ============================================================================
# VALIDATION
# ============================================================================

class PolicyValidationError(Exception):
    """Raised when policy validation fails."""
    pass


class PolicyValidator:
    """
    Validates policy data before storage.
    
    Ensures all constraints are met:
    - Valid risk modes
    - Numeric ranges
    - Required fields
    - Data types
    """

    @staticmethod
    def validate(policy_dict: dict[str, Any]) -> None:
        """
        Validate a policy dictionary.
        
        Args:
            policy_dict: The policy data to validate
            
        Raises:
            PolicyValidationError: If validation fails
        """
        # Validate risk_mode
        if "risk_mode" in policy_dict:
            risk_mode = policy_dict["risk_mode"]
            valid_modes = {mode.value for mode in RiskMode}
            if risk_mode not in valid_modes:
                raise PolicyValidationError(
                    f"Invalid risk_mode '{risk_mode}'. Must be one of {valid_modes}"
                )

        # Validate max_risk_per_trade
        if "max_risk_per_trade" in policy_dict:
            risk = policy_dict["max_risk_per_trade"]
            if not isinstance(risk, (int, float)) or risk <= 0 or risk > 1:
                raise PolicyValidationError(
                    f"max_risk_per_trade must be between 0 and 1, got {risk}"
                )

        # Validate max_positions
        if "max_positions" in policy_dict:
            max_pos = policy_dict["max_positions"]
            if not isinstance(max_pos, int) or max_pos < 1 or max_pos > 100:
                raise PolicyValidationError(
                    f"max_positions must be between 1 and 100, got {max_pos}"
                )

        # Validate global_min_confidence
        if "global_min_confidence" in policy_dict:
            conf = policy_dict["global_min_confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                raise PolicyValidationError(
                    f"global_min_confidence must be between 0 and 1, got {conf}"
                )

        # Validate allowed_strategies is a list
        if "allowed_strategies" in policy_dict:
            strats = policy_dict["allowed_strategies"]
            if not isinstance(strats, list):
                raise PolicyValidationError(
                    f"allowed_strategies must be a list, got {type(strats)}"
                )

        # Validate allowed_symbols is a list
        if "allowed_symbols" in policy_dict:
            symbols = policy_dict["allowed_symbols"]
            if not isinstance(symbols, list):
                raise PolicyValidationError(
                    f"allowed_symbols must be a list, got {type(symbols)}"
                )

        # Validate opp_rankings is a dict with numeric values
        if "opp_rankings" in policy_dict:
            rankings = policy_dict["opp_rankings"]
            if not isinstance(rankings, dict):
                raise PolicyValidationError(
                    f"opp_rankings must be a dict, got {type(rankings)}"
                )
            for symbol, score in rankings.items():
                if not isinstance(score, (int, float)) or score < 0 or score > 1:
                    raise PolicyValidationError(
                        f"opp_rankings['{symbol}'] must be between 0 and 1, got {score}"
                    )

        # Validate model_versions is a dict
        if "model_versions" in policy_dict:
            versions = policy_dict["model_versions"]
            if not isinstance(versions, dict):
                raise PolicyValidationError(
                    f"model_versions must be a dict, got {type(versions)}"
                )


# ============================================================================
# SERIALIZATION
# ============================================================================

class PolicySerializer:
    """Handles conversion between GlobalPolicy dataclass and dict/JSON."""

    @staticmethod
    def to_dict(policy: GlobalPolicy) -> dict[str, Any]:
        """Convert GlobalPolicy to dictionary."""
        return policy.to_dict()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GlobalPolicy:
        """Convert dictionary to GlobalPolicy."""
        return GlobalPolicy.from_dict(data)

    @staticmethod
    def to_json(policy: GlobalPolicy) -> str:
        """Serialize GlobalPolicy to JSON string."""
        return json.dumps(policy.to_dict(), indent=2)

    @staticmethod
    def from_json(json_str: str) -> GlobalPolicy:
        """Deserialize JSON string to GlobalPolicy."""
        data = json.loads(json_str)
        return GlobalPolicy.from_dict(data)


# ============================================================================
# POLICY MERGING
# ============================================================================

class PolicyMerger:
    """
    Safely merges partial policy updates into existing policy.
    
    Handles nested dictionaries and ensures timestamp updates.
    """

    @staticmethod
    def merge(base_policy: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
        """
        Merge partial update into base policy.
        
        Args:
            base_policy: The existing complete policy
            partial: Partial updates to apply
            
        Returns:
            New merged policy dict
        """
        merged = deepcopy(base_policy)
        
        for key, value in partial.items():
            if key == "last_updated":
                # Don't allow manual timestamp override
                continue
                
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Deep merge for nested dicts (e.g., opp_rankings, model_versions)
                merged[key] = {**merged[key], **value}
            else:
                # Direct replacement for other types
                merged[key] = deepcopy(value)
        
        # Update timestamp
        merged["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        return merged


# ============================================================================
# DEFAULTS
# ============================================================================

class PolicyDefaults:
    """Factory for default policy configurations."""

    @staticmethod
    def create_default() -> GlobalPolicy:
        """Create a default empty policy with safe initial values."""
        return GlobalPolicy(
            risk_mode=RiskMode.NORMAL.value,
            allowed_strategies=[],
            allowed_symbols=[],
            max_risk_per_trade=0.01,
            max_positions=10,
            global_min_confidence=0.65,
            opp_rankings={},
            model_versions={},
            system_health={},
            custom_params={},
        )

    @staticmethod
    def create_conservative() -> GlobalPolicy:
        """Create a conservative policy for defensive trading."""
        return GlobalPolicy(
            risk_mode=RiskMode.DEFENSIVE.value,
            allowed_strategies=[],
            allowed_symbols=[],
            max_risk_per_trade=0.005,
            max_positions=5,
            global_min_confidence=0.75,
            opp_rankings={},
            model_versions={},
            system_health={},
            custom_params={},
        )

    @staticmethod
    def create_aggressive() -> GlobalPolicy:
        """Create an aggressive policy for high-risk trading."""
        return GlobalPolicy(
            risk_mode=RiskMode.AGGRESSIVE.value,
            allowed_strategies=[],
            allowed_symbols=[],
            max_risk_per_trade=0.02,
            max_positions=15,
            global_min_confidence=0.55,
            opp_rankings={},
            model_versions={},
            system_health={},
            custom_params={},
        )


# ============================================================================
# PROTOCOL (INTERFACE)
# ============================================================================

class PolicyStore(Protocol):
    """
    Interface for policy storage backends.
    
    All implementations must provide these methods with atomic guarantees.
    """

    def get(self) -> dict[str, Any]:
        """
        Retrieve the current global policy.
        
        Returns:
            Complete policy dictionary
        """
        ...

    def update(self, new_policy: dict[str, Any]) -> None:
        """
        Replace the entire policy atomically.
        
        Args:
            new_policy: Complete new policy dictionary
            
        Raises:
            PolicyValidationError: If validation fails
        """
        ...

    def patch(self, partial: dict[str, Any]) -> None:
        """
        Update only specified fields, preserving others.
        
        Args:
            partial: Dictionary with fields to update
            
        Raises:
            PolicyValidationError: If validation fails
        """
        ...

    def reset(self) -> None:
        """
        Reset policy to default empty state.
        """
        ...

    def get_policy_object(self) -> GlobalPolicy:
        """
        Retrieve policy as typed dataclass.
        
        Returns:
            GlobalPolicy instance
        """
        ...


# ============================================================================
# IN-MEMORY IMPLEMENTATION
# ============================================================================

class InMemoryPolicyStore:
    """
    Thread-safe in-memory implementation of PolicyStore.
    
    Suitable for:
    - Unit testing
    - Development
    - Embedded systems
    - Single-process deployments
    
    Uses a threading.Lock for atomic operations.
    """

    def __init__(self, initial_policy: Optional[GlobalPolicy] = None):
        """
        Initialize the in-memory store.
        
        Args:
            initial_policy: Optional starting policy. If None, uses default.
        """
        self._lock = threading.RLock()
        if initial_policy is None:
            initial_policy = PolicyDefaults.create_default()
        self._policy: dict[str, Any] = initial_policy.to_dict()
        self._validator = PolicyValidator()
        self._merger = PolicyMerger()
        self._serializer = PolicySerializer()

    def get(self) -> dict[str, Any]:
        """Retrieve the current policy (thread-safe copy)."""
        with self._lock:
            return deepcopy(self._policy)

    def update(self, new_policy: dict[str, Any]) -> None:
        """
        Replace entire policy atomically.
        
        Args:
            new_policy: New complete policy
            
        Raises:
            PolicyValidationError: If validation fails
        """
        with self._lock:
            # Validate first
            self._validator.validate(new_policy)
            
            # Update timestamp
            updated = deepcopy(new_policy)
            updated["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Atomic replacement
            self._policy = updated

    def patch(self, partial: dict[str, Any]) -> None:
        """
        Update only specified fields.
        
        Args:
            partial: Fields to update
            
        Raises:
            PolicyValidationError: If validation fails
        """
        with self._lock:
            # Merge with existing
            merged = self._merger.merge(self._policy, partial)
            
            # Validate merged result
            self._validator.validate(merged)
            
            # Atomic update
            self._policy = merged

    def reset(self) -> None:
        """Reset to default policy."""
        with self._lock:
            default = PolicyDefaults.create_default()
            self._policy = default.to_dict()

    def get_policy_object(self) -> GlobalPolicy:
        """Retrieve as typed GlobalPolicy instance."""
        with self._lock:
            return self._serializer.from_dict(self._policy)


# ============================================================================
# POSTGRESQL STUB
# ============================================================================

class PostgresPolicyStore:
    """
    PostgreSQL-backed implementation of PolicyStore.
    
    Storage schema:
        CREATE TABLE policy_store (
            id INTEGER PRIMARY KEY DEFAULT 1,
            policy_json JSONB NOT NULL,
            last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
            CONSTRAINT single_row CHECK (id = 1)
        );
    
    Features:
    - Single-row table (enforced by constraint)
    - JSONB for efficient querying
    - Row-level locking for atomicity
    - Timestamp index
    
    Usage:
        store = PostgresPolicyStore(connection_pool)
        policy = store.get()
    """

    def __init__(self, connection_pool: Any):
        """
        Initialize with a connection pool.
        
        Args:
            connection_pool: Database connection pool (e.g., psycopg2.pool)
        """
        self._pool = connection_pool
        self._validator = PolicyValidator()
        self._merger = PolicyMerger()
        self._serializer = PolicySerializer()

    def get(self) -> dict[str, Any]:
        """
        Retrieve current policy from database.
        
        Returns:
            Policy dictionary
            
        Implementation:
            SELECT policy_json FROM policy_store WHERE id = 1;
        """
        raise NotImplementedError("PostgreSQL implementation requires psycopg2")

    def update(self, new_policy: dict[str, Any]) -> None:
        """
        Replace entire policy atomically.
        
        Args:
            new_policy: New policy dictionary
            
        Implementation:
            BEGIN;
            UPDATE policy_store 
            SET policy_json = %s, last_updated = NOW() 
            WHERE id = 1;
            COMMIT;
        """
        raise NotImplementedError("PostgreSQL implementation requires psycopg2")

    def patch(self, partial: dict[str, Any]) -> None:
        """
        Update specific fields using JSONB operations.
        
        Args:
            partial: Fields to update
            
        Implementation:
            BEGIN;
            SELECT policy_json FROM policy_store WHERE id = 1 FOR UPDATE;
            -- merge in application
            UPDATE policy_store SET policy_json = %s WHERE id = 1;
            COMMIT;
        """
        raise NotImplementedError("PostgreSQL implementation requires psycopg2")

    def reset(self) -> None:
        """
        Reset to default policy.
        
        Implementation:
            UPDATE policy_store 
            SET policy_json = %s, last_updated = NOW() 
            WHERE id = 1;
        """
        raise NotImplementedError("PostgreSQL implementation requires psycopg2")

    def get_policy_object(self) -> GlobalPolicy:
        """Retrieve as GlobalPolicy dataclass."""
        raise NotImplementedError("PostgreSQL implementation requires psycopg2")


# ============================================================================
# REDIS STUB
# ============================================================================

class RedisPolicyStore:
    """
    Redis-backed implementation for high-performance scenarios.
    
    Storage:
        Key: "quantum_trader:policy"
        Value: JSON string
        
    Features:
    - Sub-millisecond reads
    - Atomic operations with WATCH/MULTI/EXEC
    - Optional TTL for auto-expiry
    - Pub/Sub for change notifications
    
    Usage:
        store = RedisPolicyStore(redis_client)
        policy = store.get()
    """

    def __init__(self, redis_client: Any):
        """
        Initialize with Redis client.
        
        Args:
            redis_client: Redis client instance (e.g., redis.Redis)
        """
        self._redis = redis_client
        self._key = "quantum_trader:policy"
        self._validator = PolicyValidator()
        self._merger = PolicyMerger()
        self._serializer = PolicySerializer()

    def get(self) -> dict[str, Any]:
        """Retrieve from Redis."""
        raise NotImplementedError("Redis implementation requires redis-py")

    def update(self, new_policy: dict[str, Any]) -> None:
        """Atomic update using Redis transaction."""
        raise NotImplementedError("Redis implementation requires redis-py")

    def patch(self, partial: dict[str, Any]) -> None:
        """Partial update with WATCH/MULTI/EXEC."""
        raise NotImplementedError("Redis implementation requires redis-py")

    def reset(self) -> None:
        """Reset to default."""
        raise NotImplementedError("Redis implementation requires redis-py")

    def get_policy_object(self) -> GlobalPolicy:
        """Retrieve as dataclass."""
        raise NotImplementedError("Redis implementation requires redis-py")


# ============================================================================
# SQLITE IMPLEMENTATION
# ============================================================================

class SQLitePolicyStore:
    """
    SQLite-backed implementation for embedded deployments.
    
    Storage:
        Single-row table with JSON column
        
    Features:
    - File-based persistence
    - ACID guarantees
    - No external dependencies
    - Suitable for single-process deployments
    
    Usage:
        store = SQLitePolicyStore("policy.db")
        policy = store.get()
    """

    def __init__(self, db_path: str):
        """
        Initialize with SQLite database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._validator = PolicyValidator()
        self._merger = PolicyMerger()
        self._serializer = PolicySerializer()
        # Would initialize connection and create table here

    def get(self) -> dict[str, Any]:
        """Retrieve from SQLite."""
        raise NotImplementedError("SQLite implementation requires sqlite3")

    def update(self, new_policy: dict[str, Any]) -> None:
        """Atomic update with transaction."""
        raise NotImplementedError("SQLite implementation requires sqlite3")

    def patch(self, partial: dict[str, Any]) -> None:
        """Partial update in transaction."""
        raise NotImplementedError("SQLite implementation requires sqlite3")

    def reset(self) -> None:
        """Reset to default."""
        raise NotImplementedError("SQLite implementation requires sqlite3")

    def get_policy_object(self) -> GlobalPolicy:
        """Retrieve as dataclass."""
        raise NotImplementedError("SQLite implementation requires sqlite3")


# ============================================================================
# FACTORY
# ============================================================================

class PolicyStoreFactory:
    """Factory for creating appropriate PolicyStore implementations."""

    @staticmethod
    def create(backend: str, **kwargs) -> PolicyStore:
        """
        Create a PolicyStore instance.
        
        Args:
            backend: Backend type ("memory", "postgres", "redis", "sqlite")
            **kwargs: Backend-specific arguments
            
        Returns:
            PolicyStore implementation
            
        Example:
            store = PolicyStoreFactory.create("memory")
            store = PolicyStoreFactory.create("postgres", connection_pool=pool)
        """
        if backend == "memory":
            return InMemoryPolicyStore(**kwargs)
        elif backend == "postgres":
            return PostgresPolicyStore(**kwargs)
        elif backend == "redis":
            return RedisPolicyStore(**kwargs)
        elif backend == "sqlite":
            return SQLitePolicyStore(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
