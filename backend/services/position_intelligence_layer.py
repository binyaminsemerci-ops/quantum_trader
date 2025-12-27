"""
Position Intelligence Layer (PIL) - Alias Module

For backward compatibility with older imports
"""
from backend.services.position_intelligence import (
    PositionIntelligenceLayer,
    PositionCategory,
    PositionRecommendation,
    PositionClassification,
    get_position_intelligence
)

# Alias
get_pil = get_position_intelligence

__all__ = [
    'PositionIntelligenceLayer',
    'PositionCategory', 
    'PositionRecommendation',
    'PositionClassification',
    'get_position_intelligence',
    'get_pil'
]
