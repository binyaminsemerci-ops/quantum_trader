"""
Calibration System - Shared Data Types

This module defines all data structures used across the calibration system.
These types are used for confidence calibration, ensemble weight adjustment,
and configuration management.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class CalibrationStatus(str, Enum):
    """Status of a calibration job"""
    PENDING_ANALYSIS = "pending_analysis"
    ANALYSIS_COMPLETE = "analysis_complete"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class ValidationSeverity(str, Enum):
    """Severity levels for validation checks"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WeightChange:
    """Represents a change in ensemble model weight"""
    model: str
    before: float
    after: float
    delta: float
    delta_pct: float
    reason: str


@dataclass
class ValidationCheck:
    """Result of a safety validation check"""
    name: str
    passed: bool
    severity: ValidationSeverity
    description: str
    result: Any
    details: Optional[str] = None


@dataclass
class ConfidenceCalibration:
    """
    Confidence calibration using isotonic regression.
    
    Maps predicted confidence → actual win rate to improve calibration.
    """
    enabled: bool
    method: str  # "isotonic", "platt", "none"
    mapping: Dict[float, float]  # Binned lookup table
    mse_before: float  # Mean squared error before calibration
    mse_after: float   # Mean squared error after calibration
    improvement_pct: float  # % improvement
    sample_size: int
    
    def apply(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence value"""
        if not self.enabled or not self.mapping:
            return raw_confidence
        
        # Interpolate from lookup table
        import numpy as np
        bins = sorted(self.mapping.keys())
        values = [self.mapping[b] for b in bins]
        
        # Clip to valid range
        calibrated = np.interp(raw_confidence, bins, values)
        return float(np.clip(calibrated, 0.0, 1.0))


@dataclass
class EnsembleWeightCalibration:
    """
    Ensemble model weight adjustments based on performance analysis.
    
    Micro-adjustments (±5-10%) to model weights based on:
    - Precision
    - False positive rate
    - Contribution to winning trades
    """
    enabled: bool
    weights: Dict[str, float]  # model -> weight (sum = 1.0)
    changes: List[WeightChange]
    total_delta: float  # Sum of absolute changes
    sample_size: int
    
    def get_weight(self, model: str, default: float = 0.25) -> float:
        """Get calibrated weight for a model"""
        if not self.enabled:
            return default
        return self.weights.get(model, default)


@dataclass
class HoldBiasCalibration:
    """
    HOLD signal bias adjustment (optional component).
    
    Adjusts internal HOLD confidence mapping when HOLD signals
    are systematically over- or under-confident.
    """
    enabled: bool
    adjustment_factor: float  # 0.8-1.2 range
    hold_accuracy: float
    missed_opportunities: int  # Trades where HOLD was wrong
    sample_size: int


@dataclass
class CalibrationMetadata:
    """Metadata about calibration performance and characteristics"""
    win_rate_improvement: Optional[float] = None
    confidence_mse_before: Optional[float] = None
    confidence_mse_after: Optional[float] = None
    weight_stability: Optional[float] = None
    validation_status: str = "unknown"
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Calibration:
    """
    Complete calibration configuration.
    
    This is the main artifact produced by calibration analysis
    and deployed to the AI Engine.
    """
    version: str
    created_at: datetime
    based_on_trades: int
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    
    # Three calibration components
    confidence: ConfidenceCalibration
    weights: EnsembleWeightCalibration
    hold_bias: HoldBiasCalibration
    
    # Metadata
    metadata: CalibrationMetadata
    
    # Validation
    validation_checks: List[ValidationCheck] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "based_on_trades": self.based_on_trades,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            
            "confidence_calibration": {
                "enabled": self.confidence.enabled,
                "method": self.confidence.method,
                "mapping": {str(k): v for k, v in self.confidence.mapping.items()},
                "mse_before": self.confidence.mse_before,
                "mse_after": self.confidence.mse_after,
                "improvement_pct": self.confidence.improvement_pct,
                "sample_size": self.confidence.sample_size
            },
            
            "ensemble_weights": {
                "enabled": self.weights.enabled,
                "weights": self.weights.weights,
                "changes": [asdict(c) for c in self.weights.changes],
                "total_delta": self.weights.total_delta,
                "sample_size": self.weights.sample_size
            },
            
            "hold_bias": {
                "enabled": self.hold_bias.enabled,
                "adjustment_factor": self.hold_bias.adjustment_factor,
                "hold_accuracy": self.hold_bias.hold_accuracy,
                "missed_opportunities": self.hold_bias.missed_opportunities,
                "sample_size": self.hold_bias.sample_size
            },
            
            "metadata": self.metadata.to_dict(),
            
            "validation_checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "description": c.description,
                    "result": str(c.result),
                    "details": c.details
                }
                for c in self.validation_checks
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Calibration":
        """Create Calibration from dictionary"""
        # Parse confidence calibration
        conf_data = data["confidence_calibration"]
        confidence = ConfidenceCalibration(
            enabled=conf_data["enabled"],
            method=conf_data["method"],
            mapping={float(k): v for k, v in conf_data["mapping"].items()},
            mse_before=conf_data["mse_before"],
            mse_after=conf_data["mse_after"],
            improvement_pct=conf_data["improvement_pct"],
            sample_size=conf_data["sample_size"]
        )
        
        # Parse ensemble weights
        weight_data = data["ensemble_weights"]
        changes = [WeightChange(**c) for c in weight_data["changes"]]
        weights = EnsembleWeightCalibration(
            enabled=weight_data["enabled"],
            weights=weight_data["weights"],
            changes=changes,
            total_delta=weight_data["total_delta"],
            sample_size=weight_data["sample_size"]
        )
        
        # Parse hold bias
        hold_data = data["hold_bias"]
        hold_bias = HoldBiasCalibration(
            enabled=hold_data["enabled"],
            adjustment_factor=hold_data["adjustment_factor"],
            hold_accuracy=hold_data["hold_accuracy"],
            missed_opportunities=hold_data["missed_opportunities"],
            sample_size=hold_data["sample_size"]
        )
        
        # Parse metadata
        metadata = CalibrationMetadata(**data["metadata"])
        
        # Parse validation checks
        validation_checks = [
            ValidationCheck(
                name=c["name"],
                passed=c["passed"],
                severity=ValidationSeverity(c["severity"]),
                description=c["description"],
                result=c["result"],
                details=c.get("details")
            )
            for c in data.get("validation_checks", [])
        ]
        
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            based_on_trades=data["based_on_trades"],
            approved_by=data.get("approved_by"),
            approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
            confidence=confidence,
            weights=weights,
            hold_bias=hold_bias,
            metadata=metadata,
            validation_checks=validation_checks
        )


@dataclass
class CalibrationJob:
    """
    Represents a calibration analysis job.
    
    Created when calibration is initiated, tracks status through
    analysis → approval → deployment lifecycle.
    """
    id: str
    status: CalibrationStatus
    calibration: Calibration
    report_path: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "calibration": self.calibration.to_dict(),
            "report_path": self.report_path,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }


@dataclass
class ValidationReport:
    """
    Result of calibration validation.
    
    Contains all safety checks that must pass before deployment.
    """
    passed: bool
    checks: List[ValidationCheck]
    errors: List[str]
    warnings: List[str]
    risk_score: float  # 0.0-1.0, lower is safer
    
    def has_critical_failures(self) -> bool:
        """Check if any critical validation failed"""
        return any(
            not c.passed and c.severity == ValidationSeverity.CRITICAL
            for c in self.checks
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed
        
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        
        return (
            f"{status} - {passed}/{total} checks passed\n"
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}\n"
            f"Risk score: {self.risk_score:.2%}"
        )


# Exception types for calibration system
class CalibrationError(Exception):
    """Base exception for calibration system"""
    pass


class CalibrationNotReadyError(CalibrationError):
    """Raised when Learning Cadence has not authorized calibration"""
    pass


class CalibrationNotAuthorizedError(CalibrationError):
    """Raised when calibration is not in allowed actions"""
    pass


class CalibrationValidationError(CalibrationError):
    """Raised when calibration fails validation checks"""
    pass


class CalibrationDeploymentError(CalibrationError):
    """Raised when calibration deployment fails"""
    pass
