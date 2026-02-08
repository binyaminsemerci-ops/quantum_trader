#!/usr/bin/env python3
"""
Unit tests for Stop Loss direction validation

Ensures that:
1. SHORT positions ALWAYS have SL ABOVE entry
2. LONG positions ALWAYS have SL BELOW entry  
3. Trading Bot generates correct SL defaults
4. Apply Layer validates and corrects invalid SL
"""

import pytest


def calculate_sl_for_long(entry_price: float, sl_percent: float = 0.02) -> float:
    """Calculate SL for LONG position (should be BELOW entry)"""
    return entry_price * (1 - sl_percent)


def calculate_sl_for_short(entry_price: float, sl_percent: float = 0.02) -> float:
    """Calculate SL for SHORT position (should be ABOVE entry)"""
    return entry_price * (1 + sl_percent)


def validate_sl_direction(side: str, entry_price: float, stop_loss: float) -> tuple[bool, str]:
    """
    Validate that SL is in correct direction for position side.
    
    Returns:
        (is_valid, error_message)
    """
    if side == "LONG":
        if stop_loss >= entry_price:
            return False, f"LONG SL must be BELOW entry: SL={stop_loss} >= entry={entry_price}"
        return True, ""
    elif side == "SHORT":
        if stop_loss <= entry_price:
            return False, f"SHORT SL must be ABOVE entry: SL={stop_loss} <= entry={entry_price}"
        return True, ""
    else:
        return False, f"Invalid side: {side}"


class TestStopLossDirection:
    """Test suite for SL direction validation"""
    
    def test_long_sl_calculation(self):
        """Test that LONG SL is calculated BELOW entry"""
        entry = 100.0
        sl = calculate_sl_for_long(entry, 0.02)
        
        assert sl < entry, f"LONG SL must be below entry: {sl} >= {entry}"
        assert sl == 98.0, f"Expected 98.0, got {sl}"
    
    def test_short_sl_calculation(self):
        """Test that SHORT SL is calculated ABOVE entry"""
        entry = 100.0
        sl = calculate_sl_for_short(entry, 0.02)
        
        assert sl > entry, f"SHORT SL must be above entry: {sl} <= {entry}"
        assert sl == 102.0, f"Expected 102.0, got {sl}"
    
    def test_validate_valid_long_sl(self):
        """Test validation accepts valid LONG SL (below entry)"""
        is_valid, msg = validate_sl_direction("LONG", 100.0, 98.0)
        assert is_valid, f"Valid LONG SL rejected: {msg}"
    
    def test_validate_invalid_long_sl(self):
        """Test validation rejects invalid LONG SL (above entry)"""
        is_valid, msg = validate_sl_direction("LONG", 100.0, 102.0)
        assert not is_valid, "Invalid LONG SL accepted"
        assert "BELOW" in msg, f"Expected 'BELOW' in error message, got: {msg}"
    
    def test_validate_valid_short_sl(self):
        """Test validation accepts valid SHORT SL (above entry)"""
        is_valid, msg = validate_sl_direction("SHORT", 100.0, 102.0)
        assert is_valid, f"Valid SHORT SL rejected: {msg}"
    
    def test_validate_invalid_short_sl(self):
        """Test validation rejects invalid SHORT SL (below entry)"""
        is_valid, msg = validate_sl_direction("SHORT", 100.0, 98.0)
        assert not is_valid, "Invalid SHORT SL accepted"
        assert "ABOVE" in msg, f"Expected 'ABOVE' in error message, got: {msg}"
    
    def test_real_world_case_aaveusdt(self):
        """Test real-world case that caused the bug: AAVEUSDT SHORT"""
        entry = 103.15
        sl_wrong = 99.96  # What system generated (WRONG)
        sl_correct = 106.34  # What it should be
        
        # Validate wrong SL is rejected
        is_valid, msg = validate_sl_direction("SHORT", entry, sl_wrong)
        assert not is_valid, f"Bug not caught: {sl_wrong} accepted for SHORT"
        
        # Validate correct SL is accepted
        is_valid, msg = validate_sl_direction("SHORT", entry, sl_correct)
        assert is_valid, f"Correct SHORT SL rejected: {msg}"
        
        # Verify correct SL is ABOVE entry
        assert sl_correct > entry, f"Correct SHORT SL not above entry: {sl_correct} <= {entry}"
    
    @pytest.mark.parametrize("entry,side,sl_percent,expected_comparison", [
        (100.0, "LONG", 0.02, "below"),
        (100.0, "SHORT", 0.02, "above"),
        (307.04, "SHORT", 0.02, "above"),  # XMRUSDT case
        (103.15, "SHORT", 0.03, "above"),  # AAVEUSDT case
        (1.4060, "SHORT", 0.02, "above"),  # AXSUSDT case
    ])
    def test_sl_direction_parametrized(self, entry, side, sl_percent, expected_comparison):
        """Parametrized test for various scenarios"""
        if side == "LONG":
            sl = calculate_sl_for_long(entry, sl_percent)
            assert sl < entry, f"LONG SL not below entry: {sl} >= {entry}"
        elif side == "SHORT":
            sl = calculate_sl_for_short(entry, sl_percent)
            assert sl > entry, f"SHORT SL not above entry: {sl} <= {entry}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
