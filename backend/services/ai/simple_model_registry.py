"""Simple Model Registry Implementation for CLM"""
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
import json


class ModelType(Enum):
    """Model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "nhits"
    PATCHTST = "patchtst"


@dataclass
class ModelMetadata:
    """Model metadata"""
    model_type: ModelType
    version: str
    path: str
    trained_at: datetime
    metrics: dict
    is_active: bool = False


class ModelRegistry:
    """Model registry protocol"""
    def register(self, metadata: ModelMetadata): ...
    def get_active(self, model_type: ModelType) -> Optional[ModelMetadata]: ...
    def get_candidates(self, model_type: ModelType, limit: int = 5) -> List[ModelMetadata]: ...


class SimpleModelRegistry(ModelRegistry):
    """Simple file-based model registry"""
    
    def __init__(self, storage_path: str = "data/model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_path / "registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> dict:
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                # Convert to ModelMetadata objects
                registry = {}
                for model_type_str, models_list in data.items():
                    registry[model_type_str] = [
                        ModelMetadata(
                            model_type=ModelType(m['model_type']),
                            version=m['version'],
                            path=m['path'],
                            trained_at=datetime.fromisoformat(m['trained_at']),
                            metrics=m.get('metrics', {}),
                            is_active=m.get('is_active', False)
                        )
                        for m in models_list
                    ]
                return registry
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {}
        for model_type_str, models_list in self.models.items():
            data[model_type_str] = [
                {
                    'model_type': m.model_type.value,
                    'version': m.version,
                    'path': m.path,
                    'trained_at': m.trained_at.isoformat(),
                    'metrics': m.metrics,
                    'is_active': m.is_active
                }
                for m in models_list
            ]
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register(self, metadata: ModelMetadata):
        """Register a new model"""
        model_type_str = metadata.model_type.value
        if model_type_str not in self.models:
            self.models[model_type_str] = []
        
        # Deactivate other models of same type
        for m in self.models[model_type_str]:
            m.is_active = False
        
        # Add new model as active
        metadata.is_active = True
        self.models[model_type_str].append(metadata)
        self._save_registry()
    
    def get_active(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """Get active model for type"""
        model_type_str = model_type.value
        if model_type_str not in self.models:
            return None
        
        for m in self.models[model_type_str]:
            if m.is_active:
                return m
        return None
    
    def get_candidates(self, model_type: ModelType, limit: int = 5) -> List[ModelMetadata]:
        """Get recent models for type"""
        model_type_str = model_type.value
        if model_type_str not in self.models:
            return []
        
        # Sort by trained_at descending
        sorted_models = sorted(
            self.models[model_type_str],
            key=lambda m: m.trained_at,
            reverse=True
        )
        return sorted_models[:limit]
