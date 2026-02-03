"""
PolicyStore - Single Source of Truth for AI Trading Parameters

NO hardcoded trading decisions. NO fallbacks. FAIL-CLOSED.

If policy missing/stale => SKIP trade with reason code.

Policy fields (ALL required):
- universe_symbols: List of symbols AI selected
- leverage_by_symbol: Map of symbol -> leverage (AI decided)
- harvest_params: AI-generated harvest formula parameters
- kill_params: AI-generated kill score parameters
- valid_until_epoch: Policy expiration timestamp
- policy_version: Policy version for auditing

Usage:
    policy = PolicyStore.load()
    if policy.is_valid():
        leverage = policy.get_leverage(symbol)
    else:
        # SKIP trade - no fallback!
        logger.error("POLICY_MISSING_FIELD or POLICY_STALE")
"""

import redis
import json
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import hashlib


@dataclass
class PolicyData:
    """AI-generated trading policy (immutable)"""
    universe_symbols: List[str]
    leverage_by_symbol: Dict[str, float]
    harvest_params: Dict[str, float]
    kill_params: Dict[str, float]
    valid_until_epoch: float
    policy_version: str
    policy_hash: str
    
    def is_stale(self) -> bool:
        """Check if policy expired"""
        return time.time() > self.valid_until_epoch
    
    def contains_symbol(self, symbol: str) -> bool:
        """Check if symbol in universe"""
        return symbol in self.universe_symbols
    
    def get_leverage(self, symbol: str) -> Optional[float]:
        """Get AI-decided leverage for symbol (no fallback!)"""
        return self.leverage_by_symbol.get(symbol)
    
    def get_harvest_param(self, param_name: str) -> Optional[float]:
        """Get AI-decided harvest parameter (no fallback!)"""
        return self.harvest_params.get(param_name)
    
    def get_kill_param(self, param_name: str) -> Optional[float]:
        """Get AI-decided kill score parameter (no fallback!)"""
        return self.kill_params.get(param_name)


class PolicyStore:
    """
    Load AI policy from Redis with fail-closed semantics.
    
    Binary invariant: If policy invalid => return None (caller MUST skip)
    """
    
    REDIS_KEY = "quantum:policy:current"
    REDIS_STREAM = "quantum:stream:policy.update"
    
    # Required fields (missing any = INVALID)
    REQUIRED_FIELDS = [
        "universe_symbols",
        "leverage_by_symbol",
        "harvest_params",
        "kill_params",
        "valid_until_epoch",
        "policy_version"
    ]
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize policy store.
        
        Args:
            redis_client: Redis connection (or create default)
        """
        self.redis = redis_client or redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
    
    def load(self) -> Optional[PolicyData]:
        """
        Load current policy from Redis.
        
        Returns:
            PolicyData if valid, None if missing/invalid/stale
            
        Binary contract:
            - None => caller MUST skip trade
            - PolicyData => caller MAY proceed
        """
        try:
            # Load policy hash from Redis
            policy_raw = self.redis.hgetall(self.REDIS_KEY)
            
            # Check all required fields present
            missing_fields = [f for f in self.REQUIRED_FIELDS if f not in policy_raw]
            if missing_fields:
                print(f"[PolicyStore] POLICY_MISSING_FIELD: {missing_fields}")
                return None
            
            # Parse JSON fields
            try:
                universe_symbols = json.loads(policy_raw["universe_symbols"])
                leverage_by_symbol = json.loads(policy_raw["leverage_by_symbol"])
                harvest_params = json.loads(policy_raw["harvest_params"])
                kill_params = json.loads(policy_raw["kill_params"])
                valid_until_epoch = float(policy_raw["valid_until_epoch"])
                policy_version = policy_raw["policy_version"]
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[PolicyStore] POLICY_PARSE_ERROR: {e}")
                return None
            
            # Compute policy hash for auditing
            policy_hash = self._compute_hash(policy_raw)
            
            # Create policy object
            policy = PolicyData(
                universe_symbols=universe_symbols,
                leverage_by_symbol=leverage_by_symbol,
                harvest_params=harvest_params,
                kill_params=kill_params,
                valid_until_epoch=valid_until_epoch,
                policy_version=policy_version,
                policy_hash=policy_hash
            )
            
            # Check if stale
            if policy.is_stale():
                print(f"[PolicyStore] POLICY_STALE: expired at {valid_until_epoch}")
                return None
            
            # Success - log for audit
            print(f"[PolicyStore] POLICY_LOADED: version={policy_version} hash={policy_hash[:8]}")
            
            return policy
            
        except redis.RedisError as e:
            print(f"[PolicyStore] REDIS_ERROR: {e}")
            return None
        except Exception as e:
            print(f"[PolicyStore] UNKNOWN_ERROR: {e}")
            return None
    
    def save(
        self,
        universe_symbols: List[str],
        leverage_by_symbol: Dict[str, float],
        harvest_params: Dict[str, float],
        kill_params: Dict[str, float],
        valid_for_seconds: int = 3600,
        policy_version: str = "1.0"
    ) -> bool:
        """
        Save AI policy to Redis.
        
        Args:
            universe_symbols: AI-selected symbol list
            leverage_by_symbol: AI-decided leverage per symbol
            harvest_params: AI-generated harvest formula parameters
            kill_params: AI-generated kill score parameters
            valid_for_seconds: Policy TTL (default 1 hour)
            policy_version: Version identifier
            
        Returns:
            True if saved successfully
        """
        try:
            valid_until_epoch = time.time() + valid_for_seconds
            
            policy_data = {
                "universe_symbols": json.dumps(universe_symbols),
                "leverage_by_symbol": json.dumps(leverage_by_symbol),
                "harvest_params": json.dumps(harvest_params),
                "kill_params": json.dumps(kill_params),
                "valid_until_epoch": str(valid_until_epoch),
                "policy_version": policy_version
            }
            
            # Save to Redis
            self.redis.hset(self.REDIS_KEY, mapping=policy_data)
            
            # Publish update event
            self.redis.xadd(
                self.REDIS_STREAM,
                {
                    "event": "policy.updated",
                    "version": policy_version,
                    "timestamp": str(time.time())
                },
                maxlen=100
            )
            
            policy_hash = self._compute_hash(policy_data)
            print(f"[PolicyStore] POLICY_SAVED: version={policy_version} hash={policy_hash[:8]}")
            
            return True
            
        except Exception as e:
            print(f"[PolicyStore] SAVE_ERROR: {e}")
            return False
    
    def _compute_hash(self, policy_raw: Dict[str, str]) -> str:
        """Compute deterministic hash for policy auditing"""
        # Sort keys for deterministic hash
        sorted_items = sorted(policy_raw.items())
        hash_input = json.dumps(sorted_items, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()


# Convenience functions
def load_policy() -> Optional[PolicyData]:
    """Load current policy (convenience wrapper)"""
    store = PolicyStore()
    return store.load()


def save_policy(
    universe_symbols: List[str],
    leverage_by_symbol: Dict[str, float],
    harvest_params: Dict[str, float],
    kill_params: Dict[str, float],
    valid_for_seconds: int = 3600,
    policy_version: str = "1.0"
) -> bool:
    """Save policy (convenience wrapper)"""
    store = PolicyStore()
    return store.save(
        universe_symbols=universe_symbols,
        leverage_by_symbol=leverage_by_symbol,
        harvest_params=harvest_params,
        kill_params=kill_params,
        valid_for_seconds=valid_for_seconds,
        policy_version=policy_version
    )
