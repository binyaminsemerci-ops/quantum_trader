"""
Dependencies

Dependency injection for service components.
Provides fake/mock implementations for testing.
"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path


logger = logging.getLogger(__name__)


class FakePolicyStore:
    """Fake PolicyStore for testing"""
    
    def __init__(self):
        self._policies = {}
    
    def get_policy(self, key: str) -> Optional[Any]:
        return self._policies.get(key)
    
    def set_policy(self, key: str, value: Any):
        self._policies[key] = value


class FakeDataSource:
    """Fake data source for testing"""
    
    def __init__(self):
        self._data = []
    
    async def fetch_training_data(
        self,
        lookback_days: int = 90,
        min_samples: int = 100
    ) -> Dict[str, Any]:
        """Fetch training data"""
        return {
            "features": [],
            "labels": [],
            "sample_count": 150,
            "feature_names": ["rsi", "macd", "volume"],
            "lookback_days": lookback_days
        }


class FakeModelRegistry:
    """Fake model registry for testing"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("data/models")
        self._models = {}
    
    def register_version(
        self,
        model_type: str,
        version: str,
        model_object: Any,
        metrics: Dict[str, float]
    ) -> str:
        """Register new model version"""
        model_id = f"{model_type}_{version}"
        self._models[model_id] = {
            "model_type": model_type,
            "version": version,
            "model_object": model_object,
            "metrics": metrics,
            "registered_at": "2025-12-04T12:00:00Z"
        }
        logger.info(f"[FakeModelRegistry] Registered {model_id}")
        return model_id
    
    def get_version(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model version"""
        return self._models.get(model_id)
    
    def list_versions(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all model versions"""
        if model_type:
            return [
                v for v in self._models.values()
                if v["model_type"] == model_type
            ]
        return list(self._models.values())


class FakeEventBus:
    """Fake EventBus for testing"""
    
    def __init__(self):
        self._subscribers = {}
        self._published_events = []
    
    def subscribe(self, event_type: str, handler):
        """Subscribe to event"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.info(f"[FakeEventBus] Subscribed to {event_type}")
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event"""
        self._published_events.append({
            "event_type": event_type,
            "event_data": event_data
        })
        logger.info(f"[FakeEventBus] Published {event_type}")
        
        # Call subscribers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"[FakeEventBus] Handler error: {e}")
    
    def get_published_events(self) -> List[Dict[str, Any]]:
        """Get all published events"""
        return self._published_events.copy()
    
    def get_published_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get published events of specific type"""
        return [
            evt for evt in self._published_events
            if evt["event_type"] == event_type
        ]
    
    def clear_published_events(self):
        """Clear published events"""
        self._published_events.clear()


def create_fake_dependencies(config):
    """
    Create fake dependencies for testing.
    
    Returns:
        Tuple of (policy_store, data_source, model_registry, event_bus)
    """
    policy_store = FakePolicyStore()
    data_source = FakeDataSource()
    model_registry = FakeModelRegistry()
    event_bus = FakeEventBus()
    
    return policy_store, data_source, model_registry, event_bus
