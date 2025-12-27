"""
Model Registry - Centralized storage for ML model versions with metadata.

Provides:
- Model versioning and status tracking (TRAINING, SHADOW, ACTIVE, RETIRED)
- Metrics storage and comparison
- File-based model artifact storage
- PostgreSQL metadata storage
- Promotion workflows
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "nhits"
    PATCHTST = "patchtst"
    RL_META = "rl_meta"
    RL_SIZING = "rl_sizing"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModelArtifact:
    """Complete model artifact with metadata."""
    
    model_id: str
    model_type: ModelType
    version: str
    status: ModelStatus
    metrics: Dict[str, float]
    model_object: Any  # The actual model (XGB, LGBM, etc.)
    
    # Optional metadata
    training_config: Optional[Dict] = None
    training_data_range: Optional[Dict] = None  # {start, end}
    feature_count: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "version": self.version,
            "status": self.status.value,
            "metrics": self.metrics,
            "training_config": self.training_config,
            "training_data_range": self.training_data_range,
            "feature_count": self.feature_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "notes": self.notes,
        }


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Centralized registry for all ML model versions.
    
    Storage:
    - PostgreSQL: Model metadata
    - Filesystem: Model artifacts (pickled models)
    
    Features:
    - Version tracking
    - Status management (TRAINING → SHADOW → ACTIVE → RETIRED)
    - Metrics comparison
    - Safe promotion workflows
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        models_dir: str = "models",
    ):
        self.db = db_session
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelRegistry initialized: models_dir={self.models_dir}")
    
    async def register_model(
        self,
        model_artifact: ModelArtifact,
        save_file: bool = True,
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_artifact: Complete model artifact with metadata
            save_file: Whether to save model to disk
            
        Returns:
            model_id (string)
        """
        # Generate model_id if not provided
        if not model_artifact.model_id:
            model_artifact.model_id = self._generate_model_id(
                model_artifact.model_type,
                model_artifact.version
            )
        
        # Save model artifact to file
        if save_file and model_artifact.model_object is not None:
            file_path = self._save_model_file(
                model_artifact.model_id,
                model_artifact.model_object
            )
            model_artifact.file_path = str(file_path)
            model_artifact.file_size_bytes = file_path.stat().st_size
        
        # Insert into database
        query = text("""
            INSERT INTO model_registry (
                model_id, model_type, version, status, metrics,
                training_config, training_data_range, feature_count,
                created_at, file_path, file_size_bytes, notes
            ) VALUES (
                :model_id, :model_type, :version, :status, :metrics,
                :training_config, :training_data_range, :feature_count,
                :created_at, :file_path, :file_size_bytes, :notes
            )
            ON CONFLICT (model_id) DO UPDATE SET
                status = EXCLUDED.status,
                metrics = EXCLUDED.metrics,
                notes = EXCLUDED.notes
        """)
        
        self.db.execute(query, {
            "model_id": model_artifact.model_id,
            "model_type": model_artifact.model_type.value,
            "version": model_artifact.version,
            "status": model_artifact.status.value,
            "metrics": json.dumps(model_artifact.metrics),
            "training_config": json.dumps(model_artifact.training_config) if model_artifact.training_config else None,
            "training_data_range": json.dumps(model_artifact.training_data_range) if model_artifact.training_data_range else None,
            "feature_count": model_artifact.feature_count,
            "created_at": model_artifact.created_at,
            "file_path": model_artifact.file_path,
            "file_size_bytes": model_artifact.file_size_bytes,
            "notes": model_artifact.notes,
        })
        
        self.db.commit()
        
        logger.info(
            f"Registered model: {model_artifact.model_id} "
            f"({model_artifact.model_type.value} v{model_artifact.version}) "
            f"status={model_artifact.status.value}"
        )
        
        return model_artifact.model_id
    
    async def get_model(
        self,
        model_id: str,
        load_object: bool = False,
    ) -> Optional[ModelArtifact]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model identifier
            load_object: Whether to load the model object from disk
            
        Returns:
            ModelArtifact or None
        """
        query = text("""
            SELECT 
                model_id, model_type, version, status, metrics,
                training_config, training_data_range, feature_count,
                created_at, promoted_at, retired_at,
                file_path, file_size_bytes, notes
            FROM model_registry
            WHERE model_id = :model_id
        """)
        
        result = self.db.execute(query, {"model_id": model_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        artifact = self._row_to_artifact(row, load_object=load_object)
        return artifact
    
    async def get_active_model(
        self,
        model_type: ModelType,
        load_object: bool = False,
    ) -> Optional[ModelArtifact]:
        """Get the currently active model for a type."""
        query = text("""
            SELECT 
                model_id, model_type, version, status, metrics,
                training_config, training_data_range, feature_count,
                created_at, promoted_at, retired_at,
                file_path, file_size_bytes, notes
            FROM model_registry
            WHERE model_type = :model_type AND status = :status
            ORDER BY promoted_at DESC NULLS LAST
            LIMIT 1
        """)
        
        result = self.db.execute(query, {
            "model_type": model_type.value,
            "status": ModelStatus.ACTIVE.value,
        })
        row = result.fetchone()
        
        if not row:
            return None
        
        return self._row_to_artifact(row, load_object=load_object)
    
    async def get_shadow_model(
        self,
        model_type: ModelType,
        load_object: bool = False,
    ) -> Optional[ModelArtifact]:
        """Get the shadow model for a type (most recent)."""
        query = text("""
            SELECT 
                model_id, model_type, version, status, metrics,
                training_config, training_data_range, feature_count,
                created_at, promoted_at, retired_at,
                file_path, file_size_bytes, notes
            FROM model_registry
            WHERE model_type = :model_type AND status = :status
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        result = self.db.execute(query, {
            "model_type": model_type.value,
            "status": ModelStatus.SHADOW.value,
        })
        row = result.fetchone()
        
        if not row:
            return None
        
        return self._row_to_artifact(row, load_object=load_object)
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 50,
    ) -> List[ModelArtifact]:
        """
        List models with optional filters.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Max results
            
        Returns:
            List of ModelArtifacts (without model objects loaded)
        """
        conditions = []
        params = {"limit": limit}
        
        if model_type:
            conditions.append("model_type = :model_type")
            params["model_type"] = model_type.value
        
        if status:
            conditions.append("status = :status")
            params["status"] = status.value
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = text(f"""
            SELECT 
                model_id, model_type, version, status, metrics,
                training_config, training_data_range, feature_count,
                created_at, promoted_at, retired_at,
                file_path, file_size_bytes, notes
            FROM model_registry
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        result = self.db.execute(query, params)
        rows = result.fetchall()
        
        return [self._row_to_artifact(row, load_object=False) for row in rows]
    
    async def promote_shadow_to_active(
        self,
        model_type: ModelType,
        shadow_model_id: Optional[str] = None,
    ) -> bool:
        """
        Promote a shadow model to active.
        
        Steps:
        1. Retire current active model
        2. Promote shadow model to active
        3. Update timestamps
        
        Args:
            model_type: Type of model to promote
            shadow_model_id: Specific shadow model (or use latest)
            
        Returns:
            True if promotion succeeded
        """
        async with self.db.begin():
            # Get shadow model
            if shadow_model_id:
                shadow = await self.get_model(shadow_model_id)
                if not shadow or shadow.status != ModelStatus.SHADOW:
                    logger.error(f"Model {shadow_model_id} is not a SHADOW model")
                    return False
            else:
                shadow = await self.get_shadow_model(model_type)
                if not shadow:
                    logger.error(f"No SHADOW model found for {model_type.value}")
                    return False
            
            # Retire current active model
            current_active = await self.get_active_model(model_type)
            if current_active:
                await self._update_model_status(
                    current_active.model_id,
                    ModelStatus.RETIRED,
                    retired_at=datetime.utcnow()
                )
                logger.info(f"Retired active model: {current_active.model_id}")
            
            # Promote shadow to active
            await self._update_model_status(
                shadow.model_id,
                ModelStatus.ACTIVE,
                promoted_at=datetime.utcnow()
            )
            logger.info(f"Promoted shadow to active: {shadow.model_id}")
        
        return True
    
    async def retire_model(
        self,
        model_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Retire a model.
        
        Args:
            model_id: Model to retire
            reason: Optional reason for retirement
            
        Returns:
            True if succeeded
        """
        model = await self.get_model(model_id)
        if not model:
            logger.error(f"Model not found: {model_id}")
            return False
        
        notes = model.notes or ""
        if reason:
            notes += f"\nRetired: {reason}"
        
        await self._update_model_status(
            model_id,
            ModelStatus.RETIRED,
            retired_at=datetime.utcnow(),
            notes=notes
        )
        
        logger.info(f"Retired model: {model_id} ({reason or 'no reason'})")
        return True
    
    async def _update_model_status(
        self,
        model_id: str,
        status: ModelStatus,
        promoted_at: Optional[datetime] = None,
        retired_at: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Update model status and timestamps."""
        updates = ["status = :status"]
        params = {"model_id": model_id, "status": status.value}
        
        if promoted_at:
            updates.append("promoted_at = :promoted_at")
            params["promoted_at"] = promoted_at
        
        if retired_at:
            updates.append("retired_at = :retired_at")
            params["retired_at"] = retired_at
        
        if notes:
            updates.append("notes = :notes")
            params["notes"] = notes
        
        query = text(f"""
            UPDATE model_registry
            SET {', '.join(updates)}
            WHERE model_id = :model_id
        """)
        
        self.db.execute(query, params)
        self.db.commit()
    
    def _save_model_file(
        self,
        model_id: str,
        model_object: Any,
    ) -> Path:
        """Save model object to file."""
        # 🔥 FIX: N-HiTS and PatchTST are PyTorch models - use torch.save
        is_pytorch = hasattr(model_object, 'state_dict') and callable(getattr(model_object, 'state_dict'))
        
        if is_pytorch:
            file_path = self.models_dir / f"{model_id}.pth"
            try:
                import torch
                # Save complete checkpoint including model architecture info
                checkpoint = {
                    'model_state_dict': model_object.state_dict(),
                    'input_size': getattr(model_object, 'input_dim', 120),
                    'hidden_size': getattr(model_object, 'hidden_dim', 256),
                    'num_features': getattr(model_object, 'input_dim', 49),
                    'model_class': model_object.__class__.__name__,
                }
                torch.save(checkpoint, file_path)
                logger.info(f"Saved PyTorch model checkpoint: {file_path} ({file_path.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to save PyTorch model with torch.save: {e}")
                # Fallback to pickle (will work but might be less optimal)
                file_path = self.models_dir / f"{model_id}.pkl"
                with open(file_path, "wb") as f:
                    pickle.dump(model_object, f)
                logger.warning(f"Saved PyTorch model with pickle fallback: {file_path}")
        else:
            # XGBoost, LightGBM - use pickle
            file_path = self.models_dir / f"{model_id}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(model_object, f)
            logger.info(f"Saved model file: {file_path} ({file_path.stat().st_size} bytes)")
        
        return file_path
    
    def _load_model_file(self, file_path: str) -> Any:
        """Load model object from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # 🔥 FIX: Check if PyTorch model (.pth extension)
        if path.suffix == '.pth':
            try:
                import torch
                # Load state dict only - model structure must be recreated by caller
                model_state = torch.load(path, map_location='cpu')
                logger.info(f"Loaded PyTorch model state dict from: {path}")
                return model_state
            except Exception as e:
                logger.error(f"Failed to load PyTorch model from {path}: {e}")
                raise
        else:
            # XGBoost, LightGBM - use pickle
            with open(path, "rb") as f:
                model_object = pickle.load(f)
            return model_object
    
    def _row_to_artifact(
        self,
        row,
        load_object: bool = False,
    ) -> ModelArtifact:
        """Convert database row to ModelArtifact."""
        model_object = None
        if load_object and row.file_path:
            try:
                model_object = self._load_model_file(row.file_path)
            except Exception as e:
                logger.error(f"Failed to load model object: {e}")
        
        return ModelArtifact(
            model_id=row.model_id,
            model_type=ModelType(row.model_type),
            version=row.version,
            status=ModelStatus(row.status),
            metrics=json.loads(row.metrics) if isinstance(row.metrics, str) else row.metrics,
            model_object=model_object,
            training_config=json.loads(row.training_config) if row.training_config and isinstance(row.training_config, str) else row.training_config,
            training_data_range=json.loads(row.training_data_range) if row.training_data_range and isinstance(row.training_data_range, str) else row.training_data_range,
            feature_count=row.feature_count,
            created_at=row.created_at,
            promoted_at=row.promoted_at,
            retired_at=row.retired_at,
            file_path=row.file_path,
            file_size_bytes=row.file_size_bytes,
            notes=row.notes,
        )
    
    def _generate_model_id(
        self,
        model_type: ModelType,
        version: str,
    ) -> str:
        """Generate unique model ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{model_type.value}_v{version}_{timestamp}"


# ============================================================================
# Database Schema Creation
# ============================================================================

def create_model_registry_table(db_session: Session) -> None:
    """Create model_registry table if not exists (SQLite-compatible, sync)."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id VARCHAR(255) PRIMARY KEY,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            metrics TEXT NOT NULL,
            training_config TEXT,
            training_data_range TEXT,
            feature_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            promoted_at TIMESTAMP,
            retired_at TIMESTAMP,
            file_path TEXT,
            file_size_bytes INTEGER,
            notes TEXT,
            UNIQUE(model_type, version)
        );
    """)
    
    # Create indices separately for SQLite compatibility
    idx_type_status = text("CREATE INDEX IF NOT EXISTS idx_model_type_status ON model_registry(model_type, status);")
    idx_created = text("CREATE INDEX IF NOT EXISTS idx_created_at ON model_registry(created_at);")
    idx_promoted = text("CREATE INDEX IF NOT EXISTS idx_promoted_at ON model_registry(promoted_at);")
    
    db_session.execute(create_table_sql)
    db_session.execute(idx_type_status)
    db_session.execute(idx_created)
    db_session.execute(idx_promoted)
    db_session.commit()
    logger.info("Created model_registry table")
