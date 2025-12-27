"""
RL v3 Training Configuration - Schedule and settings for periodic training.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.domains.policy_store_v2 import PolicyStoreV2


@dataclass
class RLv3TrainingSchedule:
    """Training schedule configuration."""
    
    enabled: bool = True
    interval_minutes: int = 60       # how often to train
    episodes_per_run: int = 3        # how many episodes per training cycle
    max_concurrent_runs: int = 1     # avoid overlapping runs
    save_after_each_run: bool = True


@dataclass
class RLv3TrainingConfig:
    """Complete training configuration."""
    
    schedule: RLv3TrainingSchedule = None
    
    def __post_init__(self):
        if self.schedule is None:
            self.schedule = RLv3TrainingSchedule()


# Default configuration instance
DEFAULT_TRAINING_CONFIG = RLv3TrainingConfig()


def load_training_config_from_policy_store(
    policy_store: Optional["PolicyStoreV2"] = None
) -> RLv3TrainingConfig:
    """
    Load training configuration from PolicyStore v2.
    
    Reads policy keys:
    - rl_v3.training.enabled
    - rl_v3.training.interval_minutes
    - rl_v3.training.episodes_per_run
    
    Args:
        policy_store: PolicyStore v2 instance (optional)
        
    Returns:
        RLv3TrainingConfig with values from PolicyStore or defaults
    """
    if policy_store is None:
        return DEFAULT_TRAINING_CONFIG
    
    try:
        # Read from PolicyStore
        enabled = policy_store.get("rl_v3.training.enabled", True)
        interval_minutes = policy_store.get("rl_v3.training.interval_minutes", 60)
        episodes_per_run = policy_store.get("rl_v3.training.episodes_per_run", 3)
        save_after_each_run = policy_store.get("rl_v3.training.save_after_each_run", True)
        
        # Validate types and ranges
        if not isinstance(enabled, bool):
            enabled = bool(enabled)
        if not isinstance(interval_minutes, int) or interval_minutes < 0:
            interval_minutes = 60
        if not isinstance(episodes_per_run, int) or episodes_per_run < 1:
            episodes_per_run = 3
        if not isinstance(save_after_each_run, bool):
            save_after_each_run = bool(save_after_each_run)
        
        return RLv3TrainingConfig(
            schedule=RLv3TrainingSchedule(
                enabled=enabled,
                interval_minutes=interval_minutes,
                episodes_per_run=episodes_per_run,
                max_concurrent_runs=1,
                save_after_each_run=save_after_each_run
            )
        )
    except Exception:
        # Fall back to defaults on any error
        return DEFAULT_TRAINING_CONFIG
