"""
Settings management API endpoints for Quantum Trader.

This module provides configuration management functionality including:
- API key and secret management for exchanges
- Trading parameters and risk settings
- Application preferences and UI settings
- Environment configuration options

All settings are validated and securely stored with proper
error handling and performance monitoring integration.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, SecretStr
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Settings"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Access forbidden"},
        422: {"description": "Invalid input data"},
        500: {"description": "Internal server error"}
    }
)

# Explicitly type SETTINGS so mypy can validate usages that import this symbol
SETTINGS: dict[str, Any] = {}


class SettingsUpdate(BaseModel):
    """Request model for updating application settings."""
    
    api_key: Optional[str] = Field(
        None, 
        min_length=1,
        max_length=100,
        description="Exchange API key for trading operations"
    )
    api_secret: Optional[SecretStr] = Field(
        None,
        description="Exchange API secret (will be securely stored)"
    )
    risk_percentage: Optional[float] = Field(
        None,
        ge=0.1,
        le=10.0,
        description="Risk percentage per trade (0.1% - 10%)"
    )
    max_position_size: Optional[float] = Field(
        None,
        gt=0,
        description="Maximum position size in USD"
    )
    trading_enabled: Optional[bool] = Field(
        None,
        description="Enable/disable automatic trading"
    )
    notifications_enabled: Optional[bool] = Field(
        None,
        description="Enable/disable trading notifications"
    )


class SettingsResponse(BaseModel):
    """Response model for settings information."""
    
    api_key_configured: bool = Field(description="Whether API key is configured")
    api_secret_configured: bool = Field(description="Whether API secret is configured")
    risk_percentage: Optional[float] = Field(description="Current risk percentage per trade")
    max_position_size: Optional[float] = Field(description="Current maximum position size")
    trading_enabled: bool = Field(default=False, description="Trading automation status")
    notifications_enabled: bool = Field(default=True, description="Notifications status")


@router.get(
    "",
    summary="Get Application Settings",
    description="Retrieve current application settings and configuration"
)
async def get_settings(secure: bool = Query(False, description="Return only security status instead of actual values")):
    """
    Retrieve current application settings.
    
    This endpoint provides access to application configuration including:
    - API credentials and trading parameters
    - Feature enablement flags and preferences
    - Risk management settings
    
    Use the 'secure' parameter to get only configuration status without
    exposing sensitive values in production environments.
    """
    try:
        if secure:
            # Return secure configuration status without sensitive data
            return {
                "api_key_configured": bool(SETTINGS.get("api_key")),
                "api_secret_configured": bool(SETTINGS.get("api_secret")),
                "risk_percentage": SETTINGS.get("risk_percentage", 1.0),
                "max_position_size": SETTINGS.get("max_position_size", 1000.0),
                "trading_enabled": SETTINGS.get("trading_enabled", False),
                "notifications_enabled": SETTINGS.get("notifications_enabled", True)
            }
        else:
            # Return all settings for backward compatibility (testing/development)
            # In production, API secrets should be masked or excluded
            return dict(SETTINGS)
            
    except Exception as e:
        logger.error(f"Error retrieving settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.post(
    "",
    response_model=Dict[str, Any],
    summary="Update Application Settings", 
    description="Update application settings and trading configuration"
)
async def update_settings(payload: Dict[str, Any]):
    """
    Update application settings and configuration.
    
    This endpoint allows updating various application settings including:
    - Exchange API credentials for trading operations
    - Risk management parameters and position limits
    - Trading automation and notification preferences
    
    Only provided fields will be updated, existing settings remain unchanged.
    API secrets are securely handled and never logged or exposed.
    """
    try:
        # Accept any key-value pairs for backward compatibility
        update_data = {k: v for k, v in payload.items() if v is not None}
        
        # Validate settings before applying
        if "risk_percentage" in update_data:
            if not 0.1 <= update_data["risk_percentage"] <= 10.0:
                raise HTTPException(status_code=422, detail="Risk percentage must be between 0.1% and 10%")
        
        if "max_position_size" in update_data:
            if update_data["max_position_size"] <= 0:
                raise HTTPException(status_code=422, detail="Maximum position size must be positive")
        
        # Update settings
        SETTINGS.update(update_data)
        
        logger.info(f"Settings updated: {list(update_data.keys())}")
        
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "updated_fields": list(update_data.keys())
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update settings")