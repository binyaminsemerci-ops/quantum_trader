#!/usr/bin/env python3
"""
Protect Existing Unprotected Positions with Exit Brain Logic

This script:
1. Scans for unprotected positions (no TP/SL orders)
2. Uses Exit Brain v3 to calculate optimal TP/SL
3. Places orders via Binance API
4. Respects Exit Brain-controlled positions (won't override)
5. Supports dry-run mode for safety

Usage:
    # Dry-run (no actual orders):
    python protect_existing_positions.py

    # Live execution:
    python protect_existing_positions.py --live

    # Specific symbol:
    python protect_existing_positions.py --symbol SOLUSDT --live
"""

import asyncio
import os
import sys
from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from binance.client import Client
from binance.exceptions import BinanceAPIException

# Exit Brain imports
from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitPlan, ExitLeg, ExitKind
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.integration import to_dynamic_tpsl


class PositionProtector:
    """Protects existing unprotected positions using Exit Brain logic"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        
        # Use test or live API keys based on environment
        api_key = os.getenv('BINANCE_TEST_API_KEY') or os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_TEST_API_SECRET') or os.getenv('BINANCE_API_SECRET')
        use_testnet = os.getenv('BINANCE_USE_TESTNET', 'false').lower() == 'true'
        
        self.client = Client(api_key, api_secret, testnet=use_testnet)
        self.exit_brain = ExitBrainV3()
        
        # Check if Exit Brain is enabled
        self.exit_brain_enabled = os.getenv('EXIT_BRAIN_V3_ENABLED', 'false').lower() == 'true'
        
        print(f"🧠 Exit Brain v3: {'✅ ENABLED' if self.exit_brain_enabled else '❌ DISABLED'}")
        print(f"🔧 Mode: {'🔍 DRY-RUN (no actual orders)' if self.dry_run else '⚡ LIVE (will place orders)'}")
        print()
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active futures positions"""
        try:
            positions = self.client.futures_position_information()
            active = [
                p for p in positions 
                if float(p['positionAmt']) != 0
            ]
            return active
        except BinanceAPIException as e:
            print(f"❌ Error fetching positions: {e}")
            return []
    
    def get_open_orders(self, symbol: str) -> Dict[str, List[Dict]]:
        """Get open TP/SL orders for symbol"""
        try:
            orders = self.client.futures_get_open_orders(symbol=symbol)
            tp_orders = [o for o in orders if o['type'] == 'TAKE_PROFIT_MARKET']
            sl_orders = [o for o in orders if o['type'] == 'STOP_MARKET']
            return {'tp': tp_orders, 'sl': sl_orders}
        except BinanceAPIException as e:
            print(f"❌ Error fetching orders for {symbol}: {e}")
            return {'tp': [], 'sl': []}
    
    def is_position_protected(self, symbol: str) -> bool:
        """Check if position has TP or SL orders"""
        orders = self.get_open_orders(symbol)
        has_protection = len(orders['tp']) > 0 or len(orders['sl']) > 0
        return has_protection
    
    async def calculate_exit_brain_tpsl(
        self, 
        symbol: str, 
        side: str, 
        entry_price: float,
        position_size: float
    ) -> Optional[Dict[str, Any]]:
        """Use Exit Brain to calculate optimal TP/SL"""
        try:
            # Build context for Exit Brain
            ctx = ExitContext(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=position_size,
                leverage=10.0,  # Default, can be fetched from position
                unrealized_pnl_pct=0.0,  # Will be calculated
                market_regime="NORMAL",  # Default
                risk_mode="NORMAL",  # Default
                rl_tp_hint=None,  # Let Exit Brain decide
                rl_sl_hint=None,
            )
            
            # Build exit plan
            plan: ExitPlan = await self.exit_brain.build_exit_plan(ctx)
            
            # Convert to dynamic TPSL format
            result = to_dynamic_tpsl(plan, ctx)
            
            return result
            
        except Exception as e:
            print(f"❌ Error calculating Exit Brain TP/SL for {symbol}: {e}")
            return None
    
    def get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_size = f['tickSize']
                            # Count decimals in tick_size
                            return len(str(tick_size).rstrip('0').split('.')[-1])
            return 2  # Default
        except:
            return 2
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision for symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = f['stepSize']
                            return len(str(step_size).rstrip('0').split('.')[-1])
            return 0  # Default
        except:
            return 0
    
    def round_price(self, price: float, symbol: str) -> str:
        """Round price to symbol's precision"""
        precision = self.get_price_precision(symbol)
        return f"{price:.{precision}f}"
    
    def round_quantity(self, quantity: float, symbol: str) -> str:
        """Round quantity to symbol's precision"""
        precision = self.get_quantity_precision(symbol)
        return f"{abs(quantity):.{precision}f}"
    
    async def place_tp_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        tp_price: float
    ) -> bool:
        """Place take profit order"""
        try:
            # Determine order side (opposite of position)
            order_side = 'SELL' if side == 'LONG' else 'BUY'
            position_side = 'LONG' if side == 'LONG' else 'SHORT'
            
            # Round values
            qty_str = self.round_quantity(quantity, symbol)
            price_str = self.round_price(tp_price, symbol)
            
            if self.dry_run:
                print(f"   [DRY-RUN] Would place TP: {order_side} {qty_str} @ {price_str} (positionSide: {position_side})")
                return True
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                positionSide=position_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=price_str,
                closePosition=False,
                quantity=qty_str,
                workingType='CONTRACT_PRICE'
            )
            print(f"   ✅ TP order placed: {order_side} {qty_str} @ {price_str} (Order ID: {order['orderId']})")
            return True
            
        except BinanceAPIException as e:
            print(f"   ❌ Failed to place TP order: {e}")
            return False
    
    async def place_sl_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        sl_price: float
    ) -> bool:
        """Place stop loss order"""
        try:
            # Determine order side (opposite of position)
            order_side = 'SELL' if side == 'LONG' else 'BUY'
            position_side = 'LONG' if side == 'LONG' else 'SHORT'
            
            # Round values
            qty_str = self.round_quantity(quantity, symbol)
            price_str = self.round_price(sl_price, symbol)
            
            if self.dry_run:
                print(f"   [DRY-RUN] Would place SL: {order_side} {qty_str} @ {price_str} (positionSide: {position_side})")
                return True
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                positionSide=position_side,
                type='STOP_MARKET',
                stopPrice=price_str,
                closePosition=False,
                quantity=qty_str,
                workingType='CONTRACT_PRICE'
            )
            print(f"   ✅ SL order placed: {order_side} {qty_str} @ {price_str} (Order ID: {order['orderId']})")
            return True
            
        except BinanceAPIException as e:
            print(f"   ❌ Failed to place SL order: {e}")
            return False
    
    async def protect_position(self, position: Dict[str, Any]) -> bool:
        """Protect a single position with Exit Brain logic"""
        symbol = position['symbol']
        position_amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        unrealized_pnl = float(position['unRealizedProfit'])
        
        # Determine side
        side = 'LONG' if position_amt > 0 else 'SHORT'
        
        print(f"\n🎯 {symbol}: {side} {abs(position_amt)} @ ${entry_price:.4f}")
        print(f"   PnL: ${unrealized_pnl:.2f}")
        
        # Check if already protected
        if self.is_position_protected(symbol):
            print(f"   ✅ Already protected (has TP/SL orders) - SKIPPING")
            return False
        
        print(f"   ⚠️  UNPROTECTED - calculating Exit Brain TP/SL...")
        
        # Calculate Exit Brain TP/SL
        tpsl = await self.calculate_exit_brain_tpsl(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            position_size=abs(position_amt)
        )
        
        if not tpsl:
            print(f"   ❌ Failed to calculate TP/SL")
            return False
        
        # Extract TP/SL percentages (note: these are decimals, e.g., 0.03 = 3%)
        tp_percent = tpsl.get('tp_percent', 0.03)  # Default 3%
        sl_percent = tpsl.get('sl_percent', 0.025)  # Default 2.5%
        
        # Convert to percentage for display
        tp_pct = tp_percent * 100
        sl_pct = sl_percent * 100
        
        # Calculate TP/SL prices
        if side == 'LONG':
            tp_price = entry_price * (1 + tp_percent)
            sl_price = entry_price * (1 - sl_percent)
        else:
            tp_price = entry_price * (1 - tp_percent)
            sl_price = entry_price * (1 + sl_percent)
        
        print(f"   📊 Exit Brain calculated:")
        print(f"      TP: {tp_pct:.2f}% → ${tp_price:.4f}")
        print(f"      SL: {sl_pct:.2f}% → ${sl_price:.4f}")
        
        # Place orders (partial position for TP, full position for SL)
        tp_quantity = abs(position_amt) * 0.5  # 50% for first TP
        sl_quantity = abs(position_amt)  # 100% for SL
        
        tp_success = await self.place_tp_order(symbol, side, tp_quantity, tp_price)
        sl_success = await self.place_sl_order(symbol, side, sl_quantity, sl_price)
        
        if tp_success and sl_success:
            print(f"   ✅ Position protected successfully!")
            return True
        else:
            print(f"   ⚠️  Partial protection (TP: {tp_success}, SL: {sl_success})")
            return False
    
    async def protect_all_positions(self, target_symbol: Optional[str] = None):
        """Protect all unprotected positions"""
        print("=" * 60)
        print("🛡️  PROTECT EXISTING POSITIONS WITH EXIT BRAIN")
        print("=" * 60)
        
        if not self.exit_brain_enabled:
            print("\n⚠️  WARNING: EXIT_BRAIN_V3_ENABLED=false")
            print("   This script will still work, but it won't use Exit Brain logic.")
            print("   Consider activating Exit Brain for consistency.\n")
        
        # Get active positions
        positions = self.get_active_positions()
        
        if not positions:
            print("\n✅ No active positions found.")
            return
        
        print(f"\n📊 Found {len(positions)} active position(s)")
        
        # Filter by symbol if specified
        if target_symbol:
            positions = [p for p in positions if p['symbol'] == target_symbol]
            print(f"   Filtering for: {target_symbol}")
        
        if not positions:
            print(f"\n✅ No positions found for {target_symbol}")
            return
        
        # Process each position
        protected_count = 0
        skipped_count = 0
        failed_count = 0
        
        for position in positions:
            try:
                success = await self.protect_position(position)
                if success:
                    protected_count += 1
                else:
                    # Check if it was skipped (already protected) or failed
                    if self.is_position_protected(position['symbol']):
                        skipped_count += 1
                    else:
                        failed_count += 1
            except Exception as e:
                print(f"\n❌ Error processing {position['symbol']}: {e}")
                failed_count += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"✅ Protected: {protected_count}")
        print(f"⏭️  Skipped (already protected): {skipped_count}")
        print(f"❌ Failed: {failed_count}")
        
        if self.dry_run:
            print("\n🔍 DRY-RUN MODE - No actual orders were placed")
            print("   Run with --live flag to place real orders")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Protect existing unprotected positions with Exit Brain logic'
    )
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Place real orders (default: dry-run)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Protect specific symbol only (e.g., SOLUSDT)'
    )
    
    args = parser.parse_args()
    
    # Initialize protector
    protector = PositionProtector(dry_run=not args.live)
    
    # Protect positions
    await protector.protect_all_positions(target_symbol=args.symbol)


if __name__ == "__main__":
    asyncio.run(main())
