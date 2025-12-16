"""
RealModelRegistry - Production Model Version Management for CLM

PostgreSQL-based model registry for storing, versioning, and managing
trained model artifacts with complete metadata tracking.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, LargeBinary, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import json

from backend.services.ai.continuous_learning_manager import (
    ModelType,
    ModelStatus,
    ModelArtifact,
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class ModelRecord(Base):
    """SQLAlchemy model for model_artifacts table."""
    
    __tablename__ = "model_artifacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False, index=True)
    version = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    
    trained_at = Column(DateTime, nullable=False)
    promoted_at = Column(DateTime, nullable=True)
    retired_at = Column(DateTime, nullable=True)
    
    # Metrics (stored as JSON)
    metrics = Column(Text, nullable=False)
    
    # Training metadata
    training_range_start = Column(DateTime, nullable=True)
    training_range_end = Column(DateTime, nullable=True)
    data_points = Column(Integer, nullable=False, default=0)
    
    # Configuration
    training_params = Column(Text, nullable=True)
    feature_config = Column(Text, nullable=True)
    
    # Model artifact (pickled binary)
    model_artifact = Column(LargeBinary, nullable=False)
    
    # File paths (optional, for large models)
    model_path = Column(String(500), nullable=True)
    scaler_path = Column(String(500), nullable=True)
    
    def __repr__(self):
        return f"<ModelRecord {self.model_type} {self.version} ({self.status})>"


class RealModelRegistry:
    """
    Production model registry with PostgreSQL persistence.
    
    Manages model versioning, promotion, and rollback capabilities.
    """
    
    def __init__(
        self,
        db_session: Session,
        model_save_dir: str = "/app/data/models",
    ):
        """
        Initialize RealModelRegistry.
        
        Args:
            db_session: SQLAlchemy database session
            model_save_dir: Directory for storing large model files
        """
        self.db = db_session
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create table if not exists
        self._ensure_table_exists()
        
        logger.info("[ModelRegistry] Initialized with PostgreSQL persistence")
    
    def _ensure_table_exists(self):
        """Create model_artifacts table if it doesn't exist."""
        try:
            from backend.database import engine
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.info("[ModelRegistry] Table 'model_artifacts' verified")
        except Exception as e:
            logger.warning(f"[ModelRegistry] Could not create table: {e}")
    
    def get_active(self, model_type: ModelType) -> Optional[ModelArtifact]:
        """
        Get currently active model for given type.
        
        Args:
            model_type: Model type to query
        
        Returns:
            ModelArtifact or None if no active model
        """
        try:
            record = (
                self.db.query(ModelRecord)
                .filter(
                    ModelRecord.model_type == model_type.value,
                    ModelRecord.status == ModelStatus.ACTIVE.value
                )
                .order_by(ModelRecord.promoted_at.desc())
                .first()
            )
            
            if record:
                artifact = self._record_to_artifact(record)
                logger.info(
                    f"[ModelRegistry] Active model for {model_type.value}: {artifact.version}"
                )
                return artifact
            else:
                logger.warning(f"[ModelRegistry] No active model for {model_type.value}")
                return None
                
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to get active model: {e}")
            raise
    
    def save_model(self, artifact: ModelArtifact) -> None:
        """
        Save model artifact to database.
        
        Args:
            artifact: ModelArtifact to save
        """
        try:
            # Serialize model object
            model_binary = pickle.dumps(artifact.model_object)
            
            # Convert training_range to datetime if needed
            training_range_start = None
            training_range_end = None
            if artifact.training_range:
                from datetime import datetime as dt
                from pandas import Timestamp
                
                # Convert to datetime if not already
                start_val = artifact.training_range[0]
                end_val = artifact.training_range[1]
                
                if isinstance(start_val, (dt, Timestamp)):
                    training_range_start = start_val if isinstance(start_val, dt) else start_val.to_pydatetime()
                    training_range_end = end_val if isinstance(end_val, dt) else end_val.to_pydatetime()
                # If integers (row indices), skip - not valid datetime
            
            # Create record
            record = ModelRecord(
                model_type=artifact.model_type.value,
                version=artifact.version,
                status=artifact.status.value,
                trained_at=artifact.trained_at,
                metrics=json.dumps(artifact.metrics),
                training_range_start=training_range_start,
                training_range_end=training_range_end,
                data_points=artifact.data_points,
                training_params=json.dumps(artifact.training_params) if artifact.training_params else None,
                feature_config=json.dumps(artifact.feature_config) if artifact.feature_config else None,
                model_artifact=model_binary,
            )
            
            self.db.add(record)
            self.db.commit()
            
            logger.info(
                f"[ModelRegistry] Saved model: {artifact.model_type.value} {artifact.version}"
            )
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"[ModelRegistry] Failed to save model: {e}")
            raise
    
    def promote(self, model_type: ModelType, new_version: str) -> None:
        """
        Promote candidate model to active status.
        
        Args:
            model_type: Model type
            new_version: Version to promote
        """
        try:
            # Retire current active model
            current_active = (
                self.db.query(ModelRecord)
                .filter(
                    ModelRecord.model_type == model_type.value,
                    ModelRecord.status == ModelStatus.ACTIVE.value
                )
                .first()
            )
            
            if current_active:
                current_active.status = ModelStatus.RETIRED.value
                current_active.retired_at = datetime.utcnow()
                logger.info(
                    f"[ModelRegistry] Retired old active: {current_active.version}"
                )
            
            # Promote new model
            new_model = (
                self.db.query(ModelRecord)
                .filter(
                    ModelRecord.model_type == model_type.value,
                    ModelRecord.version == new_version
                )
                .first()
            )
            
            if new_model:
                new_model.status = ModelStatus.ACTIVE.value
                new_model.promoted_at = datetime.utcnow()
                self.db.commit()
                
                logger.info(
                    f"[ModelRegistry] ✅ Promoted {model_type.value} {new_version} to ACTIVE"
                )
            else:
                raise ValueError(f"Model not found: {model_type.value} {new_version}")
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"[ModelRegistry] Promotion failed: {e}")
            raise
    
    def retire(self, model_type: ModelType, version: str) -> None:
        """
        Mark model as retired.
        
        Args:
            model_type: Model type
            version: Version to retire
        """
        try:
            record = (
                self.db.query(ModelRecord)
                .filter(
                    ModelRecord.model_type == model_type.value,
                    ModelRecord.version == version
                )
                .first()
            )
            
            if record:
                record.status = ModelStatus.RETIRED.value
                record.retired_at = datetime.utcnow()
                self.db.commit()
                
                logger.info(
                    f"[ModelRegistry] Retired {model_type.value} {version}"
                )
            else:
                logger.warning(
                    f"[ModelRegistry] Model not found for retirement: {version}"
                )
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"[ModelRegistry] Retirement failed: {e}")
            raise
    
    def get_model_history(
        self,
        model_type: ModelType,
        limit: int = 10
    ) -> list[ModelArtifact]:
        """
        Get model version history.
        
        Args:
            model_type: Model type
            limit: Maximum number of versions to return
        
        Returns:
            List of ModelArtifacts, newest first
        """
        try:
            records = (
                self.db.query(ModelRecord)
                .filter(ModelRecord.model_type == model_type.value)
                .order_by(ModelRecord.trained_at.desc())
                .limit(limit)
                .all()
            )
            
            artifacts = [self._record_to_artifact(r) for r in records]
            
            logger.info(
                f"[ModelRegistry] Retrieved {len(artifacts)} versions for {model_type.value}"
            )
            
            return artifacts
            
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to get history: {e}")
            raise
    
    def rollback_to_version(self, model_type: ModelType, version: str) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            model_type: Model type
            version: Version to rollback to
        """
        logger.info(
            f"[ModelRegistry] Rolling back {model_type.value} to {version}"
        )
        
        # Promote the specified version
        self.promote(model_type, version)
    
    def _record_to_artifact(self, record: ModelRecord) -> ModelArtifact:
        """
        Convert database record to ModelArtifact.
        
        Args:
            record: ModelRecord from database
        
        Returns:
            ModelArtifact
        """
        # Deserialize model object
        model_object = pickle.loads(record.model_artifact)
        
        # Parse JSON fields
        metrics = json.loads(record.metrics) if record.metrics else {}
        training_params = json.loads(record.training_params) if record.training_params else None
        feature_config = json.loads(record.feature_config) if record.feature_config else None
        
        # Build training range
        training_range = None
        if record.training_range_start and record.training_range_end:
            training_range = (record.training_range_start, record.training_range_end)
        
        artifact = ModelArtifact(
            model_type=ModelType(record.model_type),
            version=record.version,
            trained_at=record.trained_at,
            metrics=metrics,
            model_object=model_object,
            status=ModelStatus(record.status),
            training_range=training_range,
            feature_config=feature_config,
            training_params=training_params,
            data_points=record.data_points,
        )
        
        return artifact
