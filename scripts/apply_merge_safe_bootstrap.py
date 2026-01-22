#!/usr/bin/env python3
"""
Apply merge-safe bootstrap logic to exit_monitor_service.py
P0.4 Phase 2: Preserve existing tracked position metadata during bootstrap
"""

import sys
import re

def apply_merge_safe_bootstrap(filepath):
    """Replace bootstrap logic with merge-safe version"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the bootstrap function's position tracking logic
    # Looking for line containing "# Track position" or "position = TrackedPosition("
    start_idx = None
    for i, line in enumerate(lines):
        if '# Track position' in line and i > 520:  # After line 520 (in bootstrap function)
            start_idx = i
            break
    
    if start_idx is None:
        print("❌ Could not find '# Track position' marker")
        return False
    
    # Find the end of this block (looking for the logger.info line after tracked_positions assignment)
    end_idx = None
    for i in range(start_idx, min(start_idx + 40, len(lines))):
        if 'logger.info(' in lines[i] and '🔄 BOOTSTRAP:' in lines[i]:
            # Include the closing paren line
            for j in range(i, min(i + 5, len(lines))):
                if ')' in lines[j] and 'leverage' in lines[j]:
                    end_idx = j + 1
                    break
            if end_idx:
                break
    
    if end_idx is None:
        print("❌ Could not find end of bootstrap block")
        return False
    
    # The new merge-safe code
    new_block = """            # MERGE-SAFE: Preserve existing metadata if position already tracked
            if symbol in tracked_positions:
                existing = tracked_positions[symbol]
                logger.info(
                    f"🔄 BOOTSTRAP MERGE: {symbol} updating from Binance | "
                    f"qty: {existing.quantity:.4f} -> {abs(position_amt):.4f} | "
                    f"entry: {existing.entry_price:.4f} -> {entry_price:.4f}"
                )
                # Update from Binance (source of truth)
                existing.quantity = abs(position_amt)
                existing.entry_price = entry_price
                existing.leverage = leverage
                # Preserve existing metadata (order_id, opened_at, highest_price, lowest_price, TP/SL)
                # Only update TP/SL if still default (not manually set)
                if existing.take_profit is None or existing.stop_loss is None:
                    existing.take_profit = tp
                    existing.stop_loss = sl
                bootstrapped += 1
            else:
                # New position from Binance
                position = TrackedPosition(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=abs(position_amt),
                    leverage=leverage,
                    take_profit=tp,
                    stop_loss=sl,
                    order_id=f"BOOTSTRAP_{symbol}",
                    opened_at="",
                    highest_price=entry_price if side == "BUY" else 999999.0,
                    lowest_price=entry_price if side == "SELL" else 0.0
                )
                tracked_positions[symbol] = position
                bootstrapped += 1
                logger.info(
                    f"🔄 BOOTSTRAP NEW: {symbol} {side} | "
                    f"Entry={entry_price:.4f} | Size={abs(position_amt):.4f} | Lev={leverage}x"
                )

"""
    
    # Replace the block
    new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Merge-safe bootstrap applied: replaced lines {start_idx+1}-{end_idx}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 apply_merge_safe_bootstrap.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = apply_merge_safe_bootstrap(filepath)
    sys.exit(0 if success else 1)
