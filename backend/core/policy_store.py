"""PolicyStore v2 - Redis-backed global configuration with JSON backup.

This module provides atomic, concurrent-safe access to trading policies
with automatic Redis persistence and periodic JSON snapshots.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.models.policy import (
    PolicyConfig,
    PolicyUpdateEvent,
    RiskMode,
    RiskProfile,
    DEFAULT_RISK_PROFILES,
    create_default_policy,
)

logger = logging.getLogger(__name__)


class PolicyStore:
    """
    Global policy store with Redis primary storage and JSON backup.
    
    Features:
    - Atomic reads/writes via Redis
    - Periodic JSON snapshots (every 5 minutes)
    - Automatic recovery from Redis failure
    - EventBus integration for policy updates
    - Thread-safe and async-safe
    
    Usage:
        store = PolicyStore(redis_client)
        await store.initialize()
        
        # Read current policy
        policy = await store.get_policy()
        
        # Update policy
        await store.set_policy(new_policy, updated_by="admin")
        
        # Switch risk mode
        await store.switch_mode(RiskMode.DEFENSIVE, updated_by="risk_manager")
    """
    
    REDIS_KEY = "quantum:policy:current"
    SNAPSHOT_PATH = Path("data/policy_snapshot.json")
    SNAPSHOT_INTERVAL = 300  # 5 minutes
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: Optional[object] = None,
        snapshot_path: Optional[Path] = None,
    ):
        """
        Initialize PolicyStore.
        
        Args:
            redis_client: Async Redis client
            event_bus: Optional EventBus instance for broadcasting updates
            snapshot_path: Optional custom path for JSON snapshot
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.snapshot_path = snapshot_path or self.SNAPSHOT_PATH
        
        # In-memory cache (optional - reduces Redis calls)
        self._cache: Optional[PolicyConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 5.0  # Cache for 5 seconds
        self._redis_healthy = True
        self._last_health_check = datetime.utcnow()
        
        # Snapshot task
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Subscribe to Redis recovery events for cache invalidation (CRITICAL FIX #5)
        if event_bus:
            try:
                event_bus.subscribe("system.redis_recovered", self._handle_redis_recovered)
            except Exception:
                pass  # EventBus might not be initialized yet
        
        logger.info(
            f"PolicyStore initialized: redis_key={self.REDIS_KEY}, "
            f"snapshot_path={self.snapshot_path}"
        )
    
    async def initialize(self) -> None:
        """
        Initialize store and start background tasks.
        
        - Loads policy from Redis or creates default
        - Starts snapshot task
        """
        # Ensure snapshot directory exists
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create policy
        policy = await self._load_from_redis()
        
        if policy is None:
            # Try loading from snapshot
            policy = await self._load_from_snapshot()
        
        if policy is None:
            # Create default policy
            logger.warning("No policy found in Redis or snapshot - creating default")
            policy = create_default_policy()
            await self._save_to_redis(policy)
        
        self._cache = policy
        self._cache_timestamp = datetime.utcnow()
        
        # Start background snapshot task
        self._running = True
        self._snapshot_task = asyncio.create_task(self._snapshot_worker())
        
        logger.info(
            f"PolicyStore initialized successfully: active_mode={policy.active_mode.value}, "
            f"version={policy.version}"
        )
    
    async def shutdown(self) -> None:
        """Gracefully shutdown store and save final snapshot."""
        self._running = False
        
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        
        # Save final snapshot
        if self._cache:
            await self._save_snapshot(self._cache)
        
        logger.info("PolicyStore shutdown complete")
    
    async def redis_health_check(self) -> bool:
        """Check if Redis is healthy (CRITICAL FIX #1 - Trading Gate)."""
        try:
            # Throttle health checks to once per 5 seconds
            now = datetime.utcnow()
            if (now - self._last_health_check).total_seconds() < 5.0:
                return self._redis_healthy
            
            self._last_health_check = now
            
            # Quick PING check
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            self._redis_healthy = True
            return True
        
        except Exception:
            self._redis_healthy = False
            return False
    
    async def _handle_redis_recovered(self, event_data: dict) -> None:
        """Handle Redis recovery by invalidating cache (CRITICAL FIX #5)."""
        logger.warning(
            f"Redis recovered - invalidating PolicyStore cache to prevent stale data"
        )
        self._cache = None
        self._cache_timestamp = None
        self._redis_healthy = True
        
        # Force reload from Redis
        try:
            await self.get_policy(use_cache=False)
            logger.info("PolicyStore reloaded from Redis successfully")
        except Exception as e:
            logger.error(f"Failed to reload policy after Redis recovery: {e}")
    
    async def get_policy(self, use_cache: bool = True) -> PolicyConfig:
        """
        Get current policy configuration.
        
        Args:
            use_cache: If True, use in-memory cache if fresh
        
        Returns:
            Current PolicyConfig
        
        Raises:
            RuntimeError: If policy cannot be loaded
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._cache
        
        # Load from Redis
        policy = await self._load_from_redis()
        
        if policy is None:
            # Fallback to snapshot
            policy = await self._load_from_snapshot()
        
        if policy is None:
            # Critical error - no policy available
            logger.critical("CRITICAL: No policy available from Redis or snapshot!")
            raise RuntimeError("Policy unavailable")
        
        # Update cache
        self._cache = policy
        self._cache_timestamp = datetime.utcnow()
        
        return policy
    
    async def set_policy(
        self,
        policy: PolicyConfig,
        updated_by: str = "system",
    ) -> None:
        """
        Set new policy configuration.
        
        Args:
            policy: New PolicyConfig
            updated_by: User/system that made the update
        """
        # Update metadata
        policy.last_updated = datetime.utcnow()
        policy.updated_by = updated_by
        policy.version += 1
        
        # Save to Redis (atomic)
        await self._save_to_redis(policy)
        
        # Update cache
        self._cache = policy
        self._cache_timestamp = datetime.utcnow()
        
        # Broadcast update via EventBus
        if self.event_bus:
            event = PolicyUpdateEvent(
                previous_mode=self._cache.active_mode if self._cache else policy.active_mode,
                new_mode=policy.active_mode,
                updated_by=updated_by,
            )
            try:
                await self.event_bus.publish("policy.updated", event.dict())
            except Exception as e:
                logger.error(f"Failed to publish policy update event: {e}")
        
        logger.info(
            "Policy updated",
            active_mode=policy.active_mode.value,
            version=policy.version,
            updated_by=updated_by,
        )
    
    async def switch_mode(
        self,
        new_mode: RiskMode,
        updated_by: str = "system",
    ) -> None:
        """
        Switch to a different risk mode.
        
        Args:
            new_mode: Target RiskMode
            updated_by: User/system that made the change
        """
        policy = await self.get_policy(use_cache=False)  # Force fresh read
        previous_mode = policy.active_mode
        
        # Switch mode (validates mode exists)
        policy.switch_mode(new_mode, updated_by=updated_by)
        
        # Save updated policy
        await self._save_to_redis(policy)
        
        # Update cache
        self._cache = policy
        self._cache_timestamp = datetime.utcnow()
        
        # Broadcast mode change
        if self.event_bus:
            event = PolicyUpdateEvent(
                previous_mode=previous_mode,
                new_mode=new_mode,
                updated_by=updated_by,
            )
            try:
                await self.event_bus.publish("policy.mode.changed", event.dict())
            except Exception as e:
                logger.error(f"Failed to publish mode change event: {e}")
        
        logger.warning(
            f"Risk mode changed: {previous_mode.value} -> {new_mode.value} by {updated_by}"
        )
    
    async def reload_policy(self) -> PolicyConfig:
        """
        Force reload policy from Redis (bypass cache).
        
        Returns:
            Fresh PolicyConfig from Redis
        """
        return await self.get_policy(use_cache=False)
    
    async def set_emergency_mode(
        self,
        enabled: bool,
        reason: str = "",
        updated_by: str = "system",
    ) -> None:
        """
        PATCH-P0-01: Set emergency mode (single source of truth).
        
        Args:
            enabled: True to activate emergency mode
            reason: Reason for emergency activation
            updated_by: User/system that triggered emergency
        """
        policy = await self.get_policy(use_cache=False)
        
        policy.emergency_mode = enabled
        policy.allow_new_trades = not enabled  # Block trades when emergency
        policy.emergency_reason = reason if enabled else None
        policy.emergency_activated_at = datetime.utcnow() if enabled else None
        
        await self.set_policy(policy, updated_by=updated_by)
        
        logger.critical(
            f"Emergency mode {'ACTIVATED' if enabled else 'DEACTIVATED'}: {reason}",
            extra={
                "emergency_mode": enabled,
                "reason": reason,
                "updated_by": updated_by,
            }
        )
    
    async def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode (PATCH-P0-01)."""
        policy = await self.get_policy()
        return policy.emergency_mode
    
    async def can_open_new_trades(self) -> bool:
        """Check if new trades are allowed (PATCH-P0-01)."""
        policy = await self.get_policy()
        return policy.allow_new_trades and not policy.emergency_mode
    
    async def get_active_risk_config(self) -> RiskModeConfig:
        """
        SPRINT 1 - D1: Get active risk mode configuration.
        
        Returns:
            Current RiskModeConfig based on active_mode
        
        Example:
            config = await policy_store.get_active_risk_config()
            max_leverage = config.max_leverage
            max_drawdown = config.max_daily_drawdown
        """
        policy = await self.get_policy()
        return policy.get_active_config()
    
    async def get_value(self, path: str, default=None):
        """
        SPRINT 1 - D1: Get policy value by dot-notation path.
        
        Args:
            path: Dot-separated path (e.g., "risk.max_leverage")
            default: Default value if path not found
        
        Returns:
            Value at path or default
        
        Examples:
            max_leverage = await store.get_value("risk.max_leverage", 5.0)
            min_confidence = await store.get_value("risk.global_min_confidence", 0.5)
        """
        try:
            config = await self.get_active_risk_config()
            
            # Map common paths to RiskModeConfig attributes
            path_mappings = {
                "risk.max_leverage": config.max_leverage,
                "risk.max_risk_pct_per_trade": config.max_risk_pct_per_trade,
                "risk.max_daily_drawdown": config.max_daily_drawdown,
                "risk.max_positions": config.max_positions,
                "risk.global_min_confidence": config.global_min_confidence,
                "risk.scaling_factor": config.scaling_factor,
                "risk.position_size_cap": config.position_size_cap,
                "ai.enable_rl": config.enable_rl,
                "ai.enable_meta_strategy": config.enable_meta_strategy,
                "ai.enable_pal": config.enable_pal,
                "ai.enable_pba": config.enable_pba,
                "ai.enable_clm": config.enable_clm,
                "ai.enable_retraining": config.enable_retraining,
                "ai.enable_dynamic_tpsl": config.enable_dynamic_tpsl,
            }
            
            if path in path_mappings:
                return path_mappings[path]
            
            # Try direct attribute access on config
            if "." in path:
                parts = path.split(".", 1)
                if hasattr(config, parts[1]):
                    return getattr(config, parts[1])
            
            return default
            
        except Exception as e:
            logger.warning(f"Failed to get value for path '{path}': {e}")
            return default
    
    def get(self, key: str, default=None):
        """
        Synchronous get method for backwards compatibility.
        
        This is a synchronous wrapper that returns cached values or defaults.
        For real-time Redis values, use get_value() instead.
        
        Args:
            key: Configuration key (dot-notation path)
            default: Default value if key not found
        
        Returns:
            Value from cache or default
        
        Note:
            This method uses cached policy data. If you need fresh data
            from Redis, use await get_value() instead.
        """
        try:
            # Try to use cached policy if available
            if self._cache and self._cache_timestamp:
                cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
                if cache_age < self._cache_ttl:
                    config = self._cache.get_active_config()
                    
                    # Map common paths
                    path_mappings = {
                        "rl_v3.training.enabled": True,
                        "rl_v3.training.interval_minutes": 30,
                        "rl_v3.training.episodes_per_run": 2,
                        "rl_v3.training.save_after_each_run": True,
                        "risk.max_leverage": config.max_leverage,
                        "risk.max_positions": config.max_positions,
                        "risk.global_min_confidence": config.global_min_confidence,
                    }
                    
                    if key in path_mappings:
                        return path_mappings[key]
            
            # Return default if cache miss
            return default
            
        except Exception as e:
            logger.warning(f"PolicyStore.get() failed for key '{key}': {e}")
            return default
    
    async def get_active_risk_profile(self) -> RiskProfile:
        """
        Get active risk profile based on current risk mode.
        
        Returns:
            RiskProfile for active risk mode
        
        Raises:
            RuntimeError: If profile cannot be loaded
        """
        policy = await self.get_policy()
        active_mode = policy.active_mode
        
        # Get from DEFAULT_RISK_PROFILES
        if active_mode not in DEFAULT_RISK_PROFILES:
            logger.error(
                f"Risk profile not found for mode {active_mode.value}, "
                f"falling back to NORMAL"
            )
            active_mode = RiskMode.NORMAL
        
        profile = DEFAULT_RISK_PROFILES[active_mode]
        
        logger.debug(
            f"Active risk profile: {profile.name}, "
            f"max_leverage={profile.max_leverage}, "
            f"max_risk_pct={profile.max_risk_pct_per_trade*100:.1f}%, "
            f"max_positions={profile.max_open_positions}"
        )
        
        return profile
    
    def get_active_risk_profile_name(self) -> str:
        """
        Get name of active risk profile (sync version for compatibility).
        
        Returns:
            Name of active risk profile
        """
        if self._cache is None:
            return RiskMode.NORMAL.value
        
        return self._cache.active_mode.value
    
    def _is_cache_valid(self) -> bool:
        """Check if in-memory cache is still valid."""
        if self._cache is None or self._cache_timestamp is None:
            return False
        
        age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        
        # [SPRINT 5 - PATCH #8] Auto-refresh if policy older than 10 minutes
        if age > 600:  # 10 minutes
            logger.warning(f"[PATCH #8] Policy cache aged {age:.0f}s (>10min), forcing refresh")
            return False
        
        return age < self._cache_ttl
    
    async def _load_from_redis(self) -> Optional[PolicyConfig]:
        """Load policy from Redis with failover detection."""
        try:
            # Check Redis connection with ping
            await self.redis.ping()
            
            data = await self.redis.get(self.REDIS_KEY)
            if data is None:
                return None
            
            policy_dict = json.loads(data)
            policy = PolicyConfig(**policy_dict)
            
            # Update cache timestamp on successful Redis read
            self._redis_last_connected = datetime.utcnow()
            return policy
        
        except redis.RedisError as e:
            logger.error(f"Failed to load policy from Redis (failover triggered): {e}")
            # Trigger snapshot refresh if Redis was down >30s
            await self._check_failover_refresh()
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse policy JSON from Redis: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading policy from Redis: {e}")
            return None
    
    async def _check_failover_refresh(self) -> None:
        """Check if Redis failover requires snapshot refresh."""
        if not hasattr(self, '_redis_last_connected'):
            self._redis_last_connected = datetime.utcnow()
            return
        
        downtime = (datetime.utcnow() - self._redis_last_connected).total_seconds()
        
        if downtime > 30:
            logger.warning(
                f"Redis failover detected (downtime: {downtime:.1f}s). "
                "Refreshing policy from snapshot to prevent stale data."
            )
            # Force reload from snapshot
            fresh_policy = await self._load_from_snapshot()
            if fresh_policy:
                self._cache = fresh_policy
                self._cache_timestamp = datetime.utcnow()
                # Attempt to sync back to Redis
                try:
                    await self._save_to_redis(fresh_policy)
                    logger.info("Policy synced back to Redis after failover")
                except Exception as e:
                    logger.error(f"Failed to sync policy to Redis after failover: {e}")
    
    async def _save_to_redis(self, policy: PolicyConfig) -> None:
        """Save policy to Redis (atomic operation)."""
        try:
            data = policy.json()
            await self.redis.set(self.REDIS_KEY, data)
            logger.debug(f"Policy saved to Redis: version={policy.version}")
        
        except redis.RedisError as e:
            logger.error(f"Failed to save policy to Redis: {e}")
            raise
    
    async def _load_from_snapshot(self) -> Optional[PolicyConfig]:
        """Load policy from JSON snapshot file."""
        if not self.snapshot_path.exists():
            return None
        
        try:
            with open(self.snapshot_path, "r") as f:
                policy_dict = json.load(f)
            
            policy = PolicyConfig(**policy_dict)
            logger.info(f"Policy loaded from snapshot: version={policy.version}")
            return policy
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse snapshot JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None
    
    async def _save_snapshot(self, policy: PolicyConfig) -> None:
        """Save policy to JSON snapshot file."""
        try:
            # Write to temp file first (atomic write)
            temp_path = self.snapshot_path.with_suffix(".tmp")
            
            with open(temp_path, "w") as f:
                json.dump(
                    policy.dict(),
                    f,
                    indent=2,
                    default=str,  # Handle datetime serialization
                )
            
            # Atomic rename
            temp_path.replace(self.snapshot_path)
            
            logger.debug(f"Policy snapshot saved: version={policy.version}")
        
        except Exception as e:
            logger.error(f"Failed to save policy snapshot: {e}")
    
    async def _snapshot_worker(self) -> None:
        """Background task that periodically saves policy snapshots."""
        logger.info(f"Snapshot worker started: interval={self.SNAPSHOT_INTERVAL}")
        
        while self._running:
            try:
                await asyncio.sleep(self.SNAPSHOT_INTERVAL)
                
                if self._cache:
                    await self._save_snapshot(self._cache)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot worker error: {e}")
        
        logger.info("Snapshot worker stopped")


# Singleton instance (initialized by application)
_policy_store: Optional[PolicyStore] = None


def get_policy_store() -> PolicyStore:
    """
    Get singleton PolicyStore instance.
    
    Returns:
        PolicyStore instance
    
    Raises:
        RuntimeError: If store not initialized
    """
    if _policy_store is None:
        raise RuntimeError(
            "PolicyStore not initialized. Call initialize_policy_store() first."
        )
    return _policy_store


async def initialize_policy_store(
    redis_client: Redis,
    event_bus: Optional[object] = None,
) -> PolicyStore:
    """
    Initialize global PolicyStore singleton.
    
    Args:
        redis_client: Async Redis client
        event_bus: Optional EventBus for broadcasting updates
    
    Returns:
        Initialized PolicyStore
    """
    global _policy_store
    
    _policy_store = PolicyStore(redis_client, event_bus)
    await _policy_store.initialize()
    
    return _policy_store


async def shutdown_policy_store() -> None:
    """Shutdown global PolicyStore singleton."""
    global _policy_store
    
    if _policy_store:
        await _policy_store.shutdown()
        _policy_store = None
